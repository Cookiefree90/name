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

from unittest import mock

from google.adk.tools.bigtable import BigtableCredentialsConfig
from google.adk.tools.bigtable import BigtableTool
from google.adk.tools.bigtable.config import BigtableToolConfig
from google.adk.tools.tool_context import ToolContext
from google.auth.credentials import Credentials
import pytest


async def sample_tool_func(
    credentials: Credentials, config: BigtableToolConfig
):
  return {"credentials": credentials, "config": config}


@pytest.mark.asyncio
async def test_bigtable_tool_run_async():
  """Test BigtableTool run_async method."""
  credentials_config = BigtableCredentialsConfig(
      client_id="abc", client_secret="def"
  )
  tool_config = BigtableToolConfig()
  tool = BigtableTool(
      func=sample_tool_func,
      credentials_config=credentials_config,
      bigtable_tool_config=tool_config,
  )

  tool_context = mock.create_autospec(ToolContext, instance=True)
  mock_creds = mock.create_autospec(Credentials, instance=True)

  with mock.patch.object(
      tool._credentials_manager,
      "get_valid_credentials",
      return_value=mock_creds,
  ) as mock_get_creds:
    result = await tool.run_async(args={}, tool_context=tool_context)

    mock_get_creds.assert_called_once_with(tool_context)
    assert result == {"credentials": mock_creds, "config": tool_config}


@pytest.mark.asyncio
async def test_bigtable_tool_run_async_auth_in_progress():
  """Test BigtableTool run_async when auth is in progress."""
  credentials_config = BigtableCredentialsConfig(
      client_id="abc", client_secret="def"
  )
  tool = BigtableTool(
      func=sample_tool_func, credentials_config=credentials_config
  )

  tool_context = mock.create_autospec(ToolContext, instance=True)

  with mock.patch.object(
      tool._credentials_manager, "get_valid_credentials", return_value=None
  ) as mock_get_creds:
    result = await tool.run_async(args={}, tool_context=tool_context)

    mock_get_creds.assert_called_once_with(tool_context)
    assert "User authorization is required" in result
