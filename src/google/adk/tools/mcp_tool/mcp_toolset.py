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

import logging
import sys
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Mapping
from typing import Optional
from typing import TextIO
from typing import Tuple
from typing import Union

from ...agents.readonly_context import ReadonlyContext
from ...auth.auth_credential import AuthCredential
from ...auth.auth_schemes import AuthScheme
from ..base_tool import BaseTool
from ..base_toolset import BaseToolset
from ..base_toolset import ToolPredicate
from .mcp_session_manager import MCPSessionManager
from .mcp_session_manager import retry_on_closed_resource
from .mcp_session_manager import SseConnectionParams
from .mcp_session_manager import StdioConnectionParams
from .mcp_session_manager import StreamableHTTPConnectionParams

# Attempt to import MCP Tool from the MCP library, and hints user to upgrade
# their Python version to 3.10 if it fails.
try:
  from mcp import StdioServerParameters
  from mcp.types import ListToolsResult
except ImportError as e:
  import sys

  if sys.version_info < (3, 10):
    raise ImportError(
        "MCP Tool requires Python 3.10 or above. Please upgrade your Python"
        " version."
    ) from e
  else:
    raise e

from .mcp_tool import MCPTool

logger = logging.getLogger("google_adk." + __name__)

# Type definition for auth extraction callback
GetAuthFromContextCallback = Callable[
    [Dict[str, Any]], Tuple[Optional[AuthScheme], Optional[AuthCredential]]
]

# Type definition for environment extraction callback
GetEnvFromContextCallback = Callable[[Mapping[str, Any]], Dict[str, str]]


class AuthExtractionError(Exception):
  """Exception raised when auth extraction from context fails."""

  pass


class MCPToolset(BaseToolset):
  """Connects to a MCP Server, and retrieves MCP Tools into ADK Tools.

  This toolset manages the connection to an MCP server and provides tools
  that can be used by an agent. It properly implements the BaseToolset
  interface for easy integration with the agent framework.

  Usage::

    toolset = MCPToolset(
        connection_params=StdioServerParameters(
            command='npx',
            args=["-y", "@modelcontextprotocol/server-filesystem"],
        ),
        tool_filter=['read_file', 'list_directory']  # Optional: filter specific tools
    )

    # Use in an agent
    agent = LlmAgent(
        model='gemini-2.0-flash',
        name='enterprise_assistant',
        instruction='Help user accessing their file systems',
        tools=[toolset],
    )

    # Cleanup is handled automatically by the agent framework
    # But you can also manually close if needed:
    # await toolset.close()
  """

  def __init__(
      self,
      *,
      connection_params: Union[
          StdioServerParameters,
          StdioConnectionParams,
          SseConnectionParams,
          StreamableHTTPConnectionParams,
      ],
      tool_filter: Optional[Union[ToolPredicate, List[str]]] = None,
      get_auth_from_context_fn: Optional[GetAuthFromContextCallback] = None,
      get_env_from_context_fn: Optional[GetEnvFromContextCallback] = None,
      errlog: TextIO = sys.stderr,
      auth_scheme: Optional[AuthScheme] = None,
      auth_credential: Optional[AuthCredential] = None,
  ):
    """Initializes the MCPToolset.

    Args:
      connection_params: The connection parameters to the MCP server. Can be:
        ``StdioConnectionParams`` for using local mcp server (e.g. using ``npx`` or
        ``python3``); or ``SseConnectionParams`` for a local/remote SSE server; or
        ``StreamableHTTPConnectionParams`` for local/remote Streamable http
        server. Note, ``StdioServerParameters`` is also supported for using local
        mcp server (e.g. using ``npx`` or ``python3`` ), but it does not support
        timeout, and we recommend to use ``StdioConnectionParams`` instead when
        timeout is needed.
      tool_filter: Optional filter to select specific tools. Can be either: - A
        list of tool names to include - A ToolPredicate function for custom
        filtering logic
      get_auth_from_context_fn: Optional callback function to extract auth data
        from ReadonlyContext.state into AuthScheme and AuthCredential. Must
        return a tuple of (AuthScheme, AuthCredential). If None, the toolset
        will use the auth_scheme and auth_credential provided in __init__.
        If provided, the callback must return valid AuthScheme and
        AuthCredential objects - None values are not allowed.
      get_env_from_context_fn: Optional callback function to transform session
        state into environment variables for the MCP connection. Takes a
        dictionary of session state and returns a dictionary of environment
        variables to be injected into the MCP connection.
      errlog: TextIO stream for error logging.
      auth_scheme: The auth scheme of the tool for tool calling
      auth_credential: The auth credential of the tool for tool calling
    """
    super().__init__(tool_filter=tool_filter)

    if not connection_params:
      raise ValueError("Missing connection params in MCPToolset.")

    self._connection_params = connection_params
    self._get_auth_from_context_fn = get_auth_from_context_fn
    self._get_env_from_context_fn = get_env_from_context_fn
    self._errlog = errlog

    # Create the session manager that will handle the MCP connection
    self._mcp_session_manager = MCPSessionManager(
        connection_params=self._connection_params,
        errlog=self._errlog,
    )
    self._auth_scheme = auth_scheme
    self._auth_credential = auth_credential

    # self._session = None

  def _extract_env_from_context(
      self, readonly_context: Optional[ReadonlyContext]
  ) -> Dict[str, str]:
    """Extracts environment variables from readonly context using callback.

    Args:
        readonly_context: The readonly context containing state information.

    Returns:
        Dictionary of environment variables to inject.
    """
    if not self._get_env_from_context_fn or not readonly_context:
      return {}

    try:
      # Get state from readonly context if available
      if hasattr(readonly_context, "state") and readonly_context.state:
        # Pass readonly state directly - no need to copy for read-only access
        return self._get_env_from_context_fn(readonly_context.state)
      else:
        return {}
    except Exception as e:
      logger.warning(f"Context to env mapper callback failed: {e}")
      return {}

  def _inject_env_vars(self, env_vars: Dict[str, str]) -> Union[
      StdioServerParameters,
      StdioConnectionParams,
      SseConnectionParams,
      StreamableHTTPConnectionParams,
  ]:
    """Injects environment variables into StdioConnectionParams.

    Args:
        env_vars: Dictionary of environment variables to inject.

    Returns:
        Updated connection params with injected environment variables.
    """
    if not env_vars or not isinstance(
        self._connection_params, StdioConnectionParams
    ):
      return self._connection_params

    # Get existing env vars from connection params
    existing_env = (
        getattr(self._connection_params.server_params, "env", None) or {}
    )

    # Merge existing and new env vars (new ones take precedence)
    merged_env = {**existing_env, **env_vars}

    # Create new server params with merged environment variables
    from mcp import StdioServerParameters

    new_server_params = StdioServerParameters(
        command=self._connection_params.server_params.command,
        args=self._connection_params.server_params.args,
        env=merged_env,
        cwd=getattr(self._connection_params.server_params, "cwd", None),
        encoding=getattr(
            self._connection_params.server_params, "encoding", None
        )
        or "utf-8",
        encoding_error_handler=getattr(
            self._connection_params.server_params,
            "encoding_error_handler",
            None,
        )
        or "strict",
    )

    # Create new connection params with updated server params
    return StdioConnectionParams(
        server_params=new_server_params,
        timeout=self._connection_params.timeout,
    )

  def _extract_auth_from_context(
      self, readonly_context: Optional[ReadonlyContext]
  ) -> Tuple[Optional[AuthScheme], Optional[AuthCredential]]:
    """Extracts auth scheme and credential from readonly context.

    Args:
        readonly_context: The readonly context containing state information.

    Returns:
        Tuple of (AuthScheme, AuthCredential) or (None, None) if not found.

    Raises:
        AuthExtractionError: If callback is provided but returns invalid types
            or if callback execution fails.
    """
    # If no context provided, return init values
    if not readonly_context:
      return self._auth_scheme, self._auth_credential

    # Get state from readonly context if available
    if hasattr(readonly_context, "state") and readonly_context.state:
      try:
        # Handle both real ReadonlyContext (state is MappingProxyType)
        # and test mocks (state might be a callable returning dict)
        if callable(readonly_context.state):
          state_dict = readonly_context.state()
        else:
          state_dict = dict(readonly_context.state)
      except (TypeError, ValueError) as e:
        if self._get_auth_from_context_fn:
          raise AuthExtractionError(
              f"Failed to extract state from readonly context: {e}"
          ) from e
        else:
          # If no callback, just return init values on state extraction failure
          return self._auth_scheme, self._auth_credential
    else:
      return self._auth_scheme, self._auth_credential

    # If callback is provided, use it and validate return
    if self._get_auth_from_context_fn:
      try:
        auth_result = self._get_auth_from_context_fn(state_dict)
      except Exception as e:
        raise AuthExtractionError(
            f"Auth extraction callback failed: {e}"
        ) from e

      # Validate callback return type
      if not isinstance(auth_result, tuple) or len(auth_result) != 2:
        raise AuthExtractionError(
            "Auth extraction callback must return a tuple of (AuthScheme,"
            f" AuthCredential), got {type(auth_result)}"
        )

      auth_scheme, auth_credential = auth_result

      # Validate that returned values are correct types (allow None)
      if auth_scheme is not None and not isinstance(auth_scheme, AuthScheme):
        raise AuthExtractionError(
            "Auth extraction callback returned invalid auth_scheme type: "
            f"expected AuthScheme or None, got {type(auth_scheme)}"
        )

      if auth_credential is not None and not isinstance(
          auth_credential, AuthCredential
      ):
        raise AuthExtractionError(
            "Auth extraction callback returned invalid auth_credential type: "
            f"expected AuthCredential or None, got {type(auth_credential)}"
        )

      return auth_scheme, auth_credential

    # If no callback, look for auth data directly in state (fallback behavior)
    auth_scheme = state_dict.get("auth_scheme", self._auth_scheme)
    auth_credential = state_dict.get("auth_credential", self._auth_credential)

    # Validate types - only use state values if they are correct types
    if not isinstance(auth_scheme, AuthScheme):
      auth_scheme = self._auth_scheme
    if not isinstance(auth_credential, AuthCredential):
      auth_credential = self._auth_credential

    return auth_scheme, auth_credential

  @retry_on_closed_resource
  async def get_tools(
      self,
      readonly_context: Optional[ReadonlyContext] = None,
  ) -> List[BaseTool]:
    """Return all tools in the toolset based on the provided context.

    Args:
        readonly_context: Context used to filter tools available to the agent.
            If None, all tools in the toolset are returned. The context may
            also contain auth information in its state.

    Returns:
        List[BaseTool]: A list of tools available under the specified context.
    """
    # Extract environment variables from context and inject them
    env_vars = self._extract_env_from_context(readonly_context)
    if env_vars:
      # Update connection params with environment variables
      updated_connection_params = self._inject_env_vars(env_vars)
      # Update the session manager with new connection params
      self._mcp_session_manager.update_connection_params(
          updated_connection_params
      )

    # Get session from session manager
    session = await self._mcp_session_manager.create_session()

    # Extract auth information from context
    auth_scheme, auth_credential = self._extract_auth_from_context(
        readonly_context
    )

    # Fetch available tools from the MCP server
    tools_response: ListToolsResult = await session.list_tools()

    # Apply filtering based on context and tool_filter
    tools = []
    for tool in tools_response.tools:
      mcp_tool = MCPTool(
          mcp_tool=tool,
          mcp_session_manager=self._mcp_session_manager,
          auth_scheme=auth_scheme,
          auth_credential=auth_credential,
      )

      if self._is_tool_selected(mcp_tool, readonly_context):
        tools.append(mcp_tool)
    return tools

  def _is_tool_selected(
      self, tool: BaseTool, readonly_context: Optional[ReadonlyContext]
  ) -> bool:
    """Override to handle None readonly_context."""
    if not self.tool_filter:
      return True

    if isinstance(self.tool_filter, ToolPredicate):
      return self.tool_filter(tool, readonly_context)

    if isinstance(self.tool_filter, list):
      return tool.name in self.tool_filter

    return False

  async def close(self) -> None:
    """Performs cleanup and releases resources held by the toolset.

    This method closes the MCP session and cleans up all associated resources.
    It's designed to be safe to call multiple times and handles cleanup errors
    gracefully to avoid blocking application shutdown.
    """
    try:
      await self._mcp_session_manager.close()
    except Exception as e:
      # Log the error but don't re-raise to avoid blocking shutdown
      print(f"Warning: Error during MCPToolset cleanup: {e}", file=self._errlog)
