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

"""Tool to execute SQL queries against Bigtable."""
from __future__ import annotations

import functools
import json
import types
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional

from google.api_core.exceptions import GoogleAPICallError
from google.auth.credentials import Credentials
from google.cloud import bigtable
from google.cloud.bigtable.data.execute_query import ExecuteQueryIterator
from google.cloud.bigtable.data.execute_query import ExecuteQueryValueType
from google.cloud.bigtable.data.execute_query.metadata import SqlType
from google.cloud.bigtable.row_filters import RowFilter

from . import client
from ..tool_context import ToolContext
from .config import BigtableToolConfig

MAX_ROWS = 50


def get_execute_sql(config: BigtableToolConfig) -> Callable[..., dict]:
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

  execute_sql_wrapper.__doc__ = execute_sql.__doc__
  return execute_sql_wrapper


def execute_sql(
    project_id: str,
    instance_id: str,
    query: str,
    credentials: Credentials,
    config: BigtableToolConfig,
    # parameters: Dict[str, ExecuteQueryValueType] | None = None,
    # parameter_types: Dict[str, SqlType.Type] | None = None
) -> dict:
  """Execute a GoogleSQL query from a Bigtable table.

  Args:
      project_id (str): The GCP project id in which the query should be
        executed.
      instance_id (str): The instance id of the Bigtable database.
      query (str): The Bigtable SQL query to be executed.
      credentials (Credentials): The credentials to use for the request.
      config (BigtableToolConfig): The configuration for the tool.
  Returns:
      Dictionary containing the status and the rows read.

  Examples:
      Fetch data or insights from a table:

          >>> execute_sql("my_project",
          ... "SELECT * from mytable")
          {
            "status": "SUCCESS",
            "rows": [
                {
                    "user_id": 1,
                    "user_name": "Alice"
                }
            ]
          }
  """

  try:
    bt_client = client.get_bigtable_data_client(
        project=project_id, credentials=credentials
    )
    eqi = bt_client.execute_query(
        query=query,
        instance=instance_id,
        # parameters=parameters,
        # parameter_types=parameter_types,
    )
    return {"status": "SUCCESS", "rows": read_result(eqi)}

  except Exception as ex:
    return {
        "status": "ERROR",
        "error_details": str(ex),
    }


def read_result(eqi: ExecuteQueryIterator) -> List[Dict[str, Any]]:
  results = []

  try:
    for row in eqi:
      results.append(dict(row.fields))
  except GoogleAPICallError as e:
    print(f"An error occurred: {e}")
  finally:
    # Close the results iterator when done
    eqi.close()

  return results
