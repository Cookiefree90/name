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

from typing import Optional
from unittest import mock

from google.adk.tools.base_tool import BaseTool
from google.adk.tools.bigtable import BigtableCredentialsConfig
from google.adk.tools.bigtable import BigtableToolset
from google.adk.tools.bigtable.config import BigtableToolConfig
from google.adk.tools.bigtable.query_tool import get_execute_sql
from google.adk.tools.tool_context import ToolContext
from google.auth.credentials import Credentials
from google.cloud import bigtable
from google.cloud.bigtable.data.execute_query import ExecuteQueryIterator
import pytest


async def get_tool(
    name: str, tool_config: Optional[BigtableToolConfig] = None
) -> BaseTool:
  """Get a tool from Bigtable toolset."""
  credentials_config = BigtableCredentialsConfig(
      client_id="abc", client_secret="def"
  )

  toolset = BigtableToolset(
      credentials_config=credentials_config,
      tool_filter=[name],
      bigtable_tool_config=tool_config,
  )

  tools = await toolset.get_tools()
  assert tools is not None
  assert len(tools) == 1
  return tools[0]


@pytest.mark.asyncio
async def test_execute_sql_declaration():
  """Test Bigtable get_execute_sql tool declaration."""
  tool_name = "execute_sql"
  tool = await get_tool(tool_name, BigtableToolConfig())
  assert tool.name == tool_name
  # Basic check on description, can be more detailed
  assert "Execute a GoogleSQL query from a Bigtable table" in tool.description


def test_get_execute_sql_basic():
  """Test get_execute_sql tool basic functionality."""
  project = "my_project"
  instance_id = "my_instance"
  query = "SELECT * FROM my_table"
  credentials = mock.create_autospec(Credentials, instance=True)
  tool_config = BigtableToolConfig()

  with mock.patch(
      "google.adk.tools.bigtable.client.get_bigtable_data_client"
  ) as mock_get_client:
    mock_client = mock.MagicMock()
    mock_get_client.return_value = mock_client
    mock_iterator = mock.create_autospec(ExecuteQueryIterator, instance=True)
    mock_client.execute_query.return_value = mock_iterator

    # Mock row data
    mock_row = mock.MagicMock()
    mock_row.fields = {"col1": "val1", "col2": 123}
    mock_iterator.__iter__.return_value = [mock_row]

    execute_sql_func = get_execute_sql(config=tool_config)
    result = execute_sql_func(
        project_id=project,
        instance_id=instance_id,
        credentials=credentials,
        query=query,
        config=tool_config,
    )

    expected_rows = [{"col1": "val1", "col2": 123}]
    assert result == {"status": "SUCCESS", "rows": expected_rows}
    mock_client.execute_query.assert_called_once_with(
        query=query, instance=instance_id
    )
    mock_iterator.close.assert_called_once()


def test_get_execute_sql_error():
  """Test get_execute_sql tool error handling."""
  project = "my_project"
  instance_id = "my_instance"
  query = "SELECT * FROM my_table"
  credentials = mock.create_autospec(Credentials, instance=True)
  tool_config = BigtableToolConfig()

  with mock.patch(
      "google.adk.tools.bigtable.client.get_bigtable_data_client"
  ) as mock_get_client:
    mock_client = mock.MagicMock()
    mock_get_client.return_value = mock_client
    mock_client.execute_query.side_effect = Exception("Test error")

    execute_sql_func = get_execute_sql(config=tool_config)
    result = execute_sql_func(
        project_id=project,
        instance_id=instance_id,
        credentials=credentials,
        query=query,
        config=tool_config,
    )
    assert result == {"status": "ERROR", "error_details": "Test error"}
