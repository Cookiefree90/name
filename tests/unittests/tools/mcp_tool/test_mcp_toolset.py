import pytest
from unittest.mock import MagicMock, AsyncMock

from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset
from google.adk.tools.mcp_tool import MCPTool
from mcp import StdioServerParameters
from src.google.adk.tools.mcp_tool.mcp_session_manager import SseServerParams
from google.adk.tools.base_toolset import ToolPredicate

@pytest.mark.asyncio
async def test_init_mcp_toolset():
    """Test that the MCPToolset is initialized correctly."""
    mock_connection_params = MagicMock()
    mcp_toolset = MCPToolset(connection_params=mock_connection_params)
    assert mcp_toolset._connection_params == mock_connection_params

def test_create_mcp_toolset_invalid_connection_params():
    """Test creating the MCPToolset with invalid connection parameters."""
    with pytest.raises(ValueError):
        MCPToolset(connection_params=None)

@pytest.mark.asyncio
async def test_get_tools():
    """Test getting tools from the MCPToolset."""
    mock_connection_params = MagicMock(spec=StdioServerParameters)
    mock_connection_params.command = "test_command"
    mock_connection_params.args = []  # Add the missing 'args' attribute
    mcp_toolset = MCPToolset(connection_params=mock_connection_params)
    mock_session = AsyncMock()
    mock_mcp_tool = MagicMock()
    mock_mcp_tool.name = "test_tool"
    mock_mcp_tool.description = "test_description"
    mock_session.list_tools = AsyncMock(return_value=MagicMock(tools=[mock_mcp_tool]))
    mcp_toolset._session = mock_session  # Assign the mock session
    tools = await mcp_toolset.get_tools()
    assert isinstance(tools, list)
    assert len(tools) == 1
    assert isinstance(tools[0], MCPTool)
    assert tools[0].name == "test_tool"
    assert tools[0].description == "test_description"

@pytest.mark.asyncio
async def test_close_mcp_toolset():
    """Test closing connection to the MCP server."""
    mock_connection_params = MagicMock()
    mcp_toolset = MCPToolset(connection_params=mock_connection_params)
    mcp_toolset._exit_stack = AsyncMock()
    await mcp_toolset.close()

@pytest.mark.asyncio
async def test_get_tools_error():
    """Test handling errors during tool listing."""
    mock_connection_params = MagicMock(spec=StdioServerParameters)
    mock_connection_params.command = "test_command"
    mock_connection_params.args = []  
    mcp_toolset = MCPToolset(connection_params=mock_connection_params)
    mock_session = AsyncMock()
    mock_session.list_tools = AsyncMock(side_effect=Exception("Failed to list tools"))
    mcp_toolset._session = mock_session
    mcp_toolset._exit_stack = AsyncMock() 
    mcp_toolset._exit_stack.aclose = AsyncMock()
    with pytest.raises(Exception, match="Failed to list tools"):  
        await mcp_toolset.get_tools()
    await mcp_toolset._exit_stack.aclose()

@pytest.mark.asyncio
async def test_initialize():
    """Test the _initialize method."""
    mock_connection_params = MagicMock()
    mcp_toolset = MCPToolset(connection_params=mock_connection_params)
    mock_session_manager = AsyncMock()
    mock_session = MagicMock()
    mock_session_manager.create_session = AsyncMock(return_value=mock_session)
    mcp_toolset._session_manager = mock_session_manager
    session = await mcp_toolset._initialize()
    assert session == mock_session
    mock_session_manager.create_session.assert_called_once()

def test_is_selected_no_filter():
    """Test _is_selected when tool_filter is None."""
    mock_connection_params = MagicMock()
    mcp_toolset = MCPToolset(connection_params=mock_connection_params)
    mock_tool = MagicMock()
    assert mcp_toolset._is_selected(mock_tool, None) is True

def test_is_selected_tool_predicate():
    """Test _is_selected when tool_filter is a ToolPredicate."""
    mock_connection_params = MagicMock()
    tool_filter = MagicMock(spec=ToolPredicate)
    tool_filter.return_value = True
    mcp_toolset = MCPToolset(connection_params=mock_connection_params, tool_filter=tool_filter)
    mock_tool = MagicMock()
    assert mcp_toolset._is_selected(mock_tool, None) is True
    tool_filter.assert_called_once_with(mock_tool, None)

def test_is_selected_tool_list():
    """Test _is_selected when tool_filter is a list."""
    mock_connection_params = MagicMock()
    tool_filter = ["tool1", "tool2"]
    mcp_toolset = MCPToolset(connection_params=mock_connection_params, tool_filter=tool_filter)
    mock_tool = MagicMock()
    mock_tool.name = "tool1"
    assert mcp_toolset._is_selected(mock_tool, None) is True
    mock_tool.name = "tool3"
    assert mcp_toolset._is_selected(mock_tool, None) is False

@pytest.mark.asyncio
async def test_get_tools_initializes_session():
    """Test that get_tools initializes the session if it's None."""
    mock_connection_params = MagicMock(spec=StdioServerParameters)
    mock_connection_params.command = "test_command"
    mock_connection_params.args = []  # Add the missing 'args' attribute
    mcp_toolset = MCPToolset(connection_params=mock_connection_params)
    mock_session = AsyncMock()
    mock_mcp_tool = MagicMock()
    mock_mcp_tool.name = "test_tool"
    mock_mcp_tool.description = "test_description"
    mock_session.list_tools = AsyncMock(return_value=MagicMock(tools=[mock_mcp_tool]))
    mcp_toolset._session_manager = MagicMock()
    mcp_toolset._session_manager.create_session = AsyncMock(return_value=mock_session)

    mcp_toolset._initialize = AsyncMock()  # Mock with AsyncMock
    mcp_toolset._initialize.side_effect = lambda: setattr(mcp_toolset, '_session', mock_session) or mock_session # Set session as side effect

    mcp_toolset._session = None
    tools = await mcp_toolset.get_tools()
    mcp_toolset._initialize.assert_called_once()
=======
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

from io import StringIO
import sys
import unittest
from unittest.mock import AsyncMock
from unittest.mock import Mock
from unittest.mock import patch

from google.adk.auth.auth_credential import AuthCredential
import pytest

# Skip all tests in this module if Python version is less than 3.10
pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 10), reason="MCP tool requires Python 3.10+"
)

# Import dependencies with version checking
try:
  from google.adk.tools.mcp_tool.mcp_session_manager import MCPSessionManager
  from google.adk.tools.mcp_tool.mcp_session_manager import SseConnectionParams
  from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
  from google.adk.tools.mcp_tool.mcp_session_manager import StreamableHTTPConnectionParams
  from google.adk.tools.mcp_tool.mcp_tool import MCPTool
  from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset
  from mcp import StdioServerParameters
except ImportError as e:
  if sys.version_info < (3, 10):
    # Create dummy classes to prevent NameError during test collection
    # Tests will be skipped anyway due to pytestmark
    class DummyClass:
      pass

    class StdioServerParameters:

      def __init__(self, command="test_command", args=None):
        self.command = command
        self.args = args or []

    MCPSessionManager = DummyClass
    SseConnectionParams = DummyClass
    StdioConnectionParams = DummyClass
    StreamableHTTPConnectionParams = DummyClass
    MCPTool = DummyClass
    MCPToolset = DummyClass
  else:
    raise e


class MockMCPTool:
  """Mock MCP Tool for testing."""

  def __init__(self, name, description="Test tool description"):
    self.name = name
    self.description = description
    self.inputSchema = {
        "type": "object",
        "properties": {"param": {"type": "string"}},
    }


class MockListToolsResult:
  """Mock ListToolsResult for testing."""

  def __init__(self, tools):
    self.tools = tools


class TestMCPToolset:
  """Test suite for MCPToolset class."""

  def setup_method(self):
    """Set up test fixtures."""
    self.mock_stdio_params = StdioServerParameters(
        command="test_command", args=[]
    )
    self.mock_session_manager = Mock(spec=MCPSessionManager)
    self.mock_session = AsyncMock()
    self.mock_session_manager.create_session = AsyncMock(
        return_value=self.mock_session
    )

  def test_init_basic(self):
    """Test basic initialization with StdioServerParameters."""
    toolset = MCPToolset(connection_params=self.mock_stdio_params)

    # Note: StdioServerParameters gets converted to StdioConnectionParams internally
    assert toolset._errlog == sys.stderr
    assert toolset._auth_scheme is None
    assert toolset._auth_credential is None

  def test_init_with_stdio_connection_params(self):
    """Test initialization with StdioConnectionParams."""
    stdio_params = StdioConnectionParams(
        server_params=self.mock_stdio_params, timeout=10.0
    )
    toolset = MCPToolset(connection_params=stdio_params)

    assert toolset._connection_params == stdio_params

  def test_init_with_sse_connection_params(self):
    """Test initialization with SseConnectionParams."""
    sse_params = SseConnectionParams(
        url="https://example.com/mcp", headers={"Authorization": "Bearer token"}
    )
    toolset = MCPToolset(connection_params=sse_params)

    assert toolset._connection_params == sse_params

  def test_init_with_streamable_http_params(self):
    """Test initialization with StreamableHTTPConnectionParams."""
    http_params = StreamableHTTPConnectionParams(
        url="https://example.com/mcp",
        headers={"Content-Type": "application/json"},
    )
    toolset = MCPToolset(connection_params=http_params)

    assert toolset._connection_params == http_params

  def test_init_with_tool_filter_list(self):
    """Test initialization with tool filter as list."""
    tool_filter = ["tool1", "tool2"]
    toolset = MCPToolset(
        connection_params=self.mock_stdio_params, tool_filter=tool_filter
    )

    # The tool filter is stored in the parent BaseToolset class
    # We can verify it by checking the filtering behavior in get_tools
    assert toolset._is_tool_selected is not None

  def test_init_with_auth(self):
    """Test initialization with authentication."""
    # Create real auth scheme instances
    from fastapi.openapi.models import OAuth2

    auth_scheme = OAuth2(flows={})
    from google.adk.auth.auth_credential import OAuth2Auth

    auth_credential = AuthCredential(
        auth_type="oauth2",
        oauth2=OAuth2Auth(client_id="test_id", client_secret="test_secret"),
    )

    toolset = MCPToolset(
        connection_params=self.mock_stdio_params,
        auth_scheme=auth_scheme,
        auth_credential=auth_credential,
    )

    assert toolset._auth_scheme == auth_scheme
    assert toolset._auth_credential == auth_credential

  def test_init_missing_connection_params(self):
    """Test initialization with missing connection params raises error."""
    with pytest.raises(ValueError, match="Missing connection params"):
      MCPToolset(connection_params=None)

  @pytest.mark.asyncio
  async def test_get_tools_basic(self):
    """Test getting tools without filtering."""
    # Mock tools from MCP server
    mock_tools = [
        MockMCPTool("tool1"),
        MockMCPTool("tool2"),
        MockMCPTool("tool3"),
    ]
    self.mock_session.list_tools = AsyncMock(
        return_value=MockListToolsResult(mock_tools)
    )

    toolset = MCPToolset(connection_params=self.mock_stdio_params)
    toolset._mcp_session_manager = self.mock_session_manager

    tools = await toolset.get_tools()

    assert len(tools) == 3
    for tool in tools:
      assert isinstance(tool, MCPTool)
    assert tools[0].name == "tool1"
    assert tools[1].name == "tool2"
    assert tools[2].name == "tool3"

  @pytest.mark.asyncio
  async def test_get_tools_with_list_filter(self):
    """Test getting tools with list-based filtering."""
    # Mock tools from MCP server
    mock_tools = [
        MockMCPTool("tool1"),
        MockMCPTool("tool2"),
        MockMCPTool("tool3"),
    ]
    self.mock_session.list_tools = AsyncMock(
        return_value=MockListToolsResult(mock_tools)
    )

    tool_filter = ["tool1", "tool3"]
    toolset = MCPToolset(
        connection_params=self.mock_stdio_params, tool_filter=tool_filter
    )
    toolset._mcp_session_manager = self.mock_session_manager

    tools = await toolset.get_tools()

    assert len(tools) == 2
    assert tools[0].name == "tool1"
    assert tools[1].name == "tool3"

  @pytest.mark.asyncio
  async def test_get_tools_with_function_filter(self):
    """Test getting tools with function-based filtering."""
    # Mock tools from MCP server
    mock_tools = [
        MockMCPTool("read_file"),
        MockMCPTool("write_file"),
        MockMCPTool("list_directory"),
    ]
    self.mock_session.list_tools = AsyncMock(
        return_value=MockListToolsResult(mock_tools)
    )

    def file_tools_filter(tool, context):
      """Filter for file-related tools only."""
      return "file" in tool.name

    toolset = MCPToolset(
        connection_params=self.mock_stdio_params, tool_filter=file_tools_filter
    )
    toolset._mcp_session_manager = self.mock_session_manager

    tools = await toolset.get_tools()

    assert len(tools) == 2
    assert tools[0].name == "read_file"
    assert tools[1].name == "write_file"

  @pytest.mark.asyncio
  async def test_close_success(self):
    """Test successful cleanup."""
    toolset = MCPToolset(connection_params=self.mock_stdio_params)
    toolset._mcp_session_manager = self.mock_session_manager

    await toolset.close()

    self.mock_session_manager.close.assert_called_once()

  @pytest.mark.asyncio
  async def test_close_with_exception(self):
    """Test cleanup when session manager raises exception."""
    toolset = MCPToolset(connection_params=self.mock_stdio_params)
    toolset._mcp_session_manager = self.mock_session_manager

    # Mock close to raise an exception
    self.mock_session_manager.close = AsyncMock(
        side_effect=Exception("Cleanup error")
    )

    custom_errlog = StringIO()
    toolset._errlog = custom_errlog

    # Should not raise exception
    await toolset.close()

    # Should log the error
    error_output = custom_errlog.getvalue()
    assert "Warning: Error during MCPToolset cleanup" in error_output
    assert "Cleanup error" in error_output

  @pytest.mark.asyncio
  async def test_get_tools_retry_decorator(self):
    """Test that get_tools has retry decorator applied."""
    toolset = MCPToolset(connection_params=self.mock_stdio_params)

    # Check that the method has the retry decorator
    assert hasattr(toolset.get_tools, "__wrapped__")
