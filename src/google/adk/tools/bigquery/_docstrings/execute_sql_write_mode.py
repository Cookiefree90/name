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

    Create a table with schema prescribed:

        >>> execute_sql("my_project",
        ... "CREATE TABLE my_project.my_dataset.my_table "
        ... "(island STRING, population INT64)")
        {
          "status": "SUCCESS",
          "rows": []
        }

    Insert data into an existing table:

        >>> execute_sql("my_project",
        ... "INSERT INTO my_project.my_dataset.my_table (island, population) "
        ... "VALUES ('Dream', 124), ('Biscoe', 168)")
        {
          "status": "SUCCESS",
          "rows": []
        }

    Create a table from the result of a query:

        >>> execute_sql("my_project",
        ... "CREATE TABLE my_project.my_dataset.my_table AS "
        ... "SELECT island, COUNT(*) AS population "
        ... "FROM bigquery-public-data.ml_datasets.penguins GROUP BY island")
        {
          "status": "SUCCESS",
          "rows": []
        }

    Delete a table:

        >>> execute_sql("my_project",
        ... "DROP TABLE my_project.my_dataset.my_table")
        {
          "status": "SUCCESS",
          "rows": []
        }

    Copy a table to another table:

        >>> execute_sql("my_project",
        ... "CREATE TABLE my_project.my_dataset.my_table_clone "
        ... "CLONE my_project.my_dataset.my_table")
        {
          "status": "SUCCESS",
          "rows": []
        }

    Create a snapshot (a lightweight, read-optimized copy) of en existing
    table:

        >>> execute_sql("my_project",
        ... "CREATE SNAPSHOT TABLE my_project.my_dataset.my_table_snapshot "
        ... "CLONE my_project.my_dataset.my_table")
        {
          "status": "SUCCESS",
          "rows": []
        }

    Create a BigQuery ML linear regression model:

        >>> execute_sql("my_project",
        ... "CREATE MODEL `my_dataset.my_model` "
        ... "OPTIONS (model_type='linear_reg', input_label_cols=['body_mass_g']) AS "
        ... "SELECT * FROM `bigquery-public-data.ml_datasets.penguins` "
        ... "WHERE body_mass_g IS NOT NULL")
        {
          "status": "SUCCESS",
          "rows": []
        }

    Evaluate BigQuery ML model:

        >>> execute_sql("my_project",
        ... "SELECT * FROM ML.EVALUATE(MODEL `my_dataset.my_model`)")
        {
          "status": "SUCCESS",
          "rows": [{'mean_absolute_error': 227.01223667447218,
                    'mean_squared_error': 81838.15989216768,
                    'mean_squared_log_error': 0.0050704473735013,
                    'median_absolute_error': 173.08081641661738,
                    'r2_score': 0.8723772534253441,
                    'explained_variance': 0.8723772534253442}]
        }

    Evaluate BigQuery ML model on custom data:

        >>> execute_sql("my_project",
        ... "SELECT * FROM ML.EVALUATE(MODEL `my_dataset.my_model`, "
        ... "(SELECT * FROM `my_dataset.my_table`))")
        {
          "status": "SUCCESS",
          "rows": [{'mean_absolute_error': 227.01223667447218,
                    'mean_squared_error': 81838.15989216768,
                    'mean_squared_log_error': 0.0050704473735013,
                    'median_absolute_error': 173.08081641661738,
                    'r2_score': 0.8723772534253441,
                    'explained_variance': 0.8723772534253442}]
        }

    Predict using BigQuery ML model:

        >>> execute_sql("my_project",
        ... "SELECT * FROM ML.PREDICT(MODEL `my_dataset.my_model`, "
        ... "(SELECT * FROM `my_dataset.my_table`))")
        {
          "status": "SUCCESS",
          "rows": [
              {
                "predicted_body_mass_g": "3380.9271650847013",
                ...
              }, {
                "predicted_body_mass_g": "3873.6072435386004",
                ...
              },
              ...
          ]
        }

    Delete a BigQuery ML model:

        >>> execute_sql("my_project", "DROP MODEL `my_dataset.my_model`")
        {
          "status": "SUCCESS",
          "rows": []
        }

Notes:
    - If a destination table already exists, there are a few ways to overwrite
    it:
        - Use "CREATE OR REPLACE TABLE" instead of "CREATE TABLE".
        - First run "DROP TABLE", followed by "CREATE TABLE".
    - If a model already exists, there are a few ways to overwrite it:
        - Use "CREATE OR REPLACE MODEL" instead of "CREATE MODEL".
        - First run "DROP MODEL", followed by "CREATE MODEL".
"""
