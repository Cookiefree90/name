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

docstring = """\
Run a BigQuery or BigQuery ML SQL query in the project and return the result.

Args:
    project_id (str): The GCP project id in which the query should be
      executed.
    query (str): The BigQuery SQL query to be executed.
    credentials (Credentials): The credentials to use for the request.
    config (BigQueryToolConfig): The configuration for the tool.
    tool_context (ToolContext): The context for the tool.

Returns:
    dict: Dictionary representing the result of the query.
          If the result contains the key "result_is_likely_truncated" with
          value True, it means that there may be additional rows matching the
          query not returned in the result.

Examples:
    Fetch data or insights from a table:

        >>> execute_sql("my_project",
        ... "SELECT island, COUNT(*) AS population "
        ... "FROM bigquery-public-data.ml_datasets.penguins GROUP BY island")
        {
          "status": "SUCCESS",
          "rows": [
              {
                  "island": "Dream",
                  "population": 124
              },
              {
                  "island": "Biscoe",
                  "population": 168
              },
              {
                  "island": "Torgersen",
                  "population": 52
              }
          ]
        }
"""
