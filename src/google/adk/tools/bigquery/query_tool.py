# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import functools
import json
import types
from typing import Callable

from google.auth.credentials import Credentials
from google.cloud import bigquery

from . import client
from ..tool_context import ToolContext
from .config import BigQueryToolConfig
from .config import WriteMode

BIGQUERY_SESSION_INFO_KEY = "bigquery_session_info"


def execute_sql(
    project_id: str,
    query: str,
    credentials: Credentials,
    config: BigQueryToolConfig,
    tool_context: ToolContext,
) -> dict:
  try:
    # Get BigQuery client
    bq_client = client.get_bigquery_client(
        project=project_id, credentials=credentials
    )

    # BigQuery connection properties where applicable
    bq_connection_properties = None

    if not config or config.write_mode == WriteMode.BLOCKED:
      dry_run_query_job = bq_client.query(
          query,
          project=project_id,
          job_config=bigquery.QueryJobConfig(dry_run=True),
      )
      if dry_run_query_job.statement_type != "SELECT":
        return {
            "status": "ERROR",
            "error_details": "Read-only mode only supports SELECT statements.",
        }
    elif config.write_mode == WriteMode.PROTECTED:
      # In protected write mode, write operation only to a temporary artifact is
      # allowed. This artifact must have been created in a BigQuery session. In
      # such a scenario the session info (session id and the anonymous dataset
      # containing the artifact) is persisted in the tool context.
      bq_session_info = tool_context.state.get(BIGQUERY_SESSION_INFO_KEY, None)
      if bq_session_info:
        bq_session_id, bq_session_dataset_id = bq_session_info
      else:
        session_creator_job = bq_client.query(
            "SELECT 1",
            project=project_id,
            job_config=bigquery.QueryJobConfig(
                dry_run=True, create_session=True
            ),
        )
        bq_session_id = session_creator_job.session_info.session_id
        bq_session_dataset_id = session_creator_job.destination.dataset_id

        # Remember the BigQuery session info for subsequent queries
        tool_context.state[BIGQUERY_SESSION_INFO_KEY] = (
            bq_session_id,
            bq_session_dataset_id,
        )

      # Session connection property will be set in the query execution
      bq_connection_properties = [
          bigquery.ConnectionProperty("session_id", bq_session_id)
      ]

      # Check the query type w.r.t. the BigQuery session
      dry_run_query_job = bq_client.query(
          query,
          project=project_id,
          job_config=bigquery.QueryJobConfig(
              dry_run=True,
              connection_properties=bq_connection_properties,
          ),
      )
      if (
          dry_run_query_job.statement_type != "SELECT"
          and dry_run_query_job.destination.dataset_id != bq_session_dataset_id
      ):
        return {
            "status": "ERROR",
            "error_details": (
                "Protected write mode only supports SELECT statements, or write"
                " operations in the anonymous dataset of a BigQuery session."
            ),
        }

    # Finally execute the query and fetch the result
    job_config = (
        bigquery.QueryJobConfig(connection_properties=bq_connection_properties)
        if bq_connection_properties
        else None
    )
    row_iterator = bq_client.query_and_wait(
        query,
        job_config=job_config,
        project=project_id,
        max_results=config.max_query_result_rows,
    )
    rows = []
    for row in row_iterator:
      row_values = {}
      for key, val in row.items():
        try:
          # if the json serialization of the value succeeds, use it as is
          json.dumps(val)
        except:
          val = str(val)
        row_values[key] = val
      rows.append(row_values)

    result = {"status": "SUCCESS", "rows": rows}
    if (
        config.max_query_result_rows is not None
        and len(rows) == config.max_query_result_rows
    ):
      result["result_is_likely_truncated"] = True
    return result
  except Exception as ex:  # pylint: disable=broad-except
    return {
        "status": "ERROR",
        "error_details": str(ex),
    }


def get_execute_sql(config: BigQueryToolConfig) -> Callable[..., dict]:
  """Get the execute_sql tool customized as per the given tool config.

  Args:
      config: BigQuery tool configuration indicating the behavior of the
        execute_sql tool.

  Returns:
      callable[..., dict]: A version of the execute_sql tool respecting the tool
      config.
  """

  if not config or config.write_mode == WriteMode.BLOCKED:
    from ._docstrings import execute_sql_read_only_mode

    execute_sql.__doc__ = execute_sql_read_only_mode.docstring
    return execute_sql

  # Create a new function object using the original function's code and globals.
  # We pass the original code, globals, name, defaults, and closure.
  # This creates a raw function object without copying other metadata yet.
  execute_sql_wrapper = types.FunctionType(
      execute_sql.__code__,
      execute_sql.__globals__,
      execute_sql.__name__,
      execute_sql.__defaults__,
      execute_sql.__closure__,
  )

  # Use functools.update_wrapper to copy over other essential attributes
  # from the original function to the new one.
  # This includes __name__, __qualname__, __module__, __annotations__, etc.
  # It specifically allows us to then set __doc__ separately.
  functools.update_wrapper(execute_sql_wrapper, execute_sql)

  # Now, set the new docstring
  if config.write_mode == WriteMode.PROTECTED:
    from ._docstrings import execute_sql_protected_write_mode

    execute_sql_wrapper.__doc__ = execute_sql_protected_write_mode.docstring
  else:
    from ._docstrings import execute_sql_write_mode

    execute_sql_wrapper.__doc__ = execute_sql_write_mode.docstring

  return execute_sql_wrapper
