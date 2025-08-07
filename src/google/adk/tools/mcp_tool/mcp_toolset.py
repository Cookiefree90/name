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
from typing import List
from typing import Optional
from typing import TextIO
from typing import Union

from fastapi.openapi.models import OAuth2
from ...agents.readonly_context import ReadonlyContext
from ...auth.auth_credential import AuthCredential
from ...auth.auth_schemes import AuthScheme
from ...auth.oauth2_discovery_util import create_oauth_scheme_from_discovery
from ..base_tool import BaseTool
from ..base_toolset import BaseToolset
from ..base_toolset import ToolPredicate
from .mcp_auth_discovery import MCPAuthDiscovery
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


class MCPToolset(BaseToolset):
  """Connects to a MCP Server, and retrieves MCP Tools into ADK Tools.

  This toolset manages the connection to an MCP server and provides tools
  that can be used by an agent. It properly implements the BaseToolset
  interface for easy integration with the agent framework.

  **OAuth Discovery by Default**: MCPToolset automatically attempts OAuth2 discovery
  for HTTP-based connections (StreamableHTTP, SSE) unless explicitly disabled. This
  provides seamless authentication setup without manual configuration.

  Usage::

    # Basic usage with automatic OAuth discovery (default behavior)
    toolset = MCPToolset(
        connection_params=StreamableHTTPConnectionParams(
            url='http://localhost:9204/mcp/',
        ),
        auth_credential=oauth2_credential,  # OAuth discovery will find token endpoint
        tool_filter=['read_file', 'list_directory']  # Optional: filter specific tools
    )

    # Explicit OAuth discovery configuration (overrides default)
    toolset = MCPToolset(
        connection_params=StreamableHTTPConnectionParams(
            url='http://localhost:9204/mcp/',
        ),
        auth_credential=oauth2_credential,
        auth_discovery=MCPAuthDiscovery(
            base_url='http://custom-auth-server:9205',  # Different auth server
            timeout=15.0
        ),
        tool_filter=['read_file', 'list_directory']
    )

    # Disable OAuth discovery completely
    toolset = MCPToolset(
        connection_params=StreamableHTTPConnectionParams(
            url='http://localhost:9204/mcp/',
        ),
        auth_credential=oauth2_credential,
        auth_discovery=MCPAuthDiscovery(enabled=False),  # Explicitly disabled
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
      errlog: TextIO = sys.stderr,
      auth_scheme: Optional[AuthScheme] = None,
      auth_credential: Optional[AuthCredential] = None,
      auth_discovery: Optional[MCPAuthDiscovery] = None,
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
      errlog: TextIO stream for error logging.
      auth_scheme: The auth scheme of the tool for tool calling. If not provided
        and OAuth discovery succeeds, a discovered scheme will be used.
      auth_credential: The auth credential of the tool for tool calling
      auth_discovery: Optional OAuth discovery configuration. If not provided,
        automatic OAuth discovery will be enabled for HTTP-based connections using
        the server's base URL. Set to MCPAuthDiscovery(enabled=False) to disable.
    """
    super().__init__(tool_filter=tool_filter)

    if not connection_params:
      raise ValueError("Missing connection params in MCPToolset.")

    self._connection_params = connection_params
    self._errlog = errlog

    # Create the session manager that will handle the MCP connection
    self._mcp_session_manager = MCPSessionManager(
        connection_params=self._connection_params,
        errlog=self._errlog,
    )
    self._auth_scheme = auth_scheme
    self._auth_credential = auth_credential
    
    # Default OAuth discovery behavior: Auto-enable for HTTP connections
    if auth_discovery is None:
      auth_discovery = self._create_default_auth_discovery()
      
    self._auth_discovery = auth_discovery
    self._oauth_discovery_attempted = False

  def _create_default_auth_discovery(self) -> MCPAuthDiscovery:
    """Create default OAuth discovery configuration from connection parameters.
    
    Returns:
        MCPAuthDiscovery instance configured for the connection, or disabled
        if the connection type doesn't support OAuth discovery.
    """
    # Extract base URL from HTTP-based connection parameters
    base_url = None
    
    if isinstance(self._connection_params, StreamableHTTPConnectionParams):
      # Extract server root from MCP URL
      full_url = self._connection_params.url
      from urllib.parse import urlparse
      parsed = urlparse(full_url)
      base_url = f"{parsed.scheme}://{parsed.netloc}"
      logger.debug(f"Auto-detected OAuth discovery base URL: {base_url} from MCP URL: {full_url}")
      
    elif isinstance(self._connection_params, SseConnectionParams):
      # Extract server root from SSE URL  
      full_url = self._connection_params.url
      from urllib.parse import urlparse
      parsed = urlparse(full_url)
      base_url = f"{parsed.scheme}://{parsed.netloc}"
      logger.debug(f"Auto-detected OAuth discovery base URL: {base_url} from SSE URL: {full_url}")
    
    if base_url:
      logger.debug(f"✅ Enabling default OAuth discovery for HTTP connection at: {base_url}")
      return MCPAuthDiscovery(
          base_url=base_url,
          timeout=10.0,
          enabled=True
      )
    else:
      # For Stdio connections, OAuth discovery is not applicable
      logger.debug("❌ Disabling OAuth discovery for non-HTTP connection (Stdio)")
      return MCPAuthDiscovery(
          enabled=False
      )

  async def _perform_oauth_discovery(self) -> None:
    """Perform OAuth discovery if enabled and not already attempted."""
    logger.debug("🔍 _perform_oauth_discovery() called")
    logger.debug(f"🔍 auth_discovery: {self._auth_discovery}")
    logger.debug(f"🔍 current auth_scheme: {self._auth_scheme}")
    
    if (
        not self._auth_discovery or not self._auth_discovery.is_enabled
        or self._oauth_discovery_attempted
    ):
      logger.debug("❌ OAuth discovery skipped (not enabled or already attempted)")
      return
      
    # Check if we need discovery even when auth_scheme is provided
    needs_discovery = False
    
    if self._auth_scheme is None:
      # No auth scheme provided - definitely need discovery
      needs_discovery = True
      logger.debug("🎯 OAuth discovery needed: no auth scheme provided")
    elif isinstance(self._auth_scheme, OAuth2):
      # Check if OAuth2 scheme has client credentials flow with empty/invalid tokenUrl
      if (self._auth_scheme.flows and 
          self._auth_scheme.flows.clientCredentials and 
          (not self._auth_scheme.flows.clientCredentials.tokenUrl or 
           self._auth_scheme.flows.clientCredentials.tokenUrl.strip() == "")):
        needs_discovery = True
        logger.debug("🎯 OAuth discovery needed: empty tokenUrl in existing scheme")
    
    if not needs_discovery:
      logger.debug("❌ OAuth discovery not needed")
      return
      
    self._oauth_discovery_attempted = True
    logger.debug("🚀 Starting OAuth discovery process")
    
    # Determine the discovery base URL
    if self._auth_discovery.base_url:
      # Use explicitly configured base URL
      base_url = self._auth_discovery.base_url
      logger.debug(f"Using explicitly configured discovery base URL: {base_url}")
    else:
      # Auto-extract base URL from connection parameters (same logic as _create_default_auth_discovery)
      base_url = None
      if isinstance(self._connection_params, StreamableHTTPConnectionParams):
        # Extract server root from HTTP URL
        full_url = self._connection_params.url
        from urllib.parse import urlparse
        parsed = urlparse(full_url)
        base_url = f"{parsed.scheme}://{parsed.netloc}"
        logger.debug(f"Auto-extracted OAuth discovery base URL: {base_url} from MCP URL: {full_url}")
        
      elif isinstance(self._connection_params, SseConnectionParams):
        # Extract server root from SSE URL  
        full_url = self._connection_params.url
        from urllib.parse import urlparse
        parsed = urlparse(full_url)
        base_url = f"{parsed.scheme}://{parsed.netloc}"
        logger.debug(f"Auto-extracted OAuth discovery base URL: {base_url} from SSE URL: {full_url}")
      
      if not base_url:
        logger.debug("❌ Cannot auto-extract base URL for OAuth discovery (non-HTTP connection)")
        return
      
    try:
      logger.debug(f"🔍 Attempting OAuth discovery at server root: {base_url}")
      
      # Extract scopes from existing auth scheme if available
      discovery_scopes = None
      if (isinstance(self._auth_scheme, OAuth2) and 
          self._auth_scheme.flows and 
          self._auth_scheme.flows.clientCredentials and
          self._auth_scheme.flows.clientCredentials.scopes):
        # Use scopes from existing auth scheme
        discovery_scopes = list(self._auth_scheme.flows.clientCredentials.scopes.keys())
        logger.debug(f"Using scopes from auth scheme: {discovery_scopes}")
      
      discovered_scheme = await create_oauth_scheme_from_discovery(
          base_url=base_url,
          scopes=discovery_scopes,
          timeout=self._auth_discovery.timeout,
          verify_ssl=self._auth_discovery.verify_ssl
      )
      
      if discovered_scheme:
        if self._auth_scheme is None:
          # No existing scheme - use discovered scheme entirely
          logger.debug("✅ OAuth discovery successful - using discovered configuration")
          self._auth_scheme = discovered_scheme
        else:
          # Existing scheme with empty tokenUrl - merge discovered tokenUrl
          logger.debug("✅ OAuth discovery successful - updating tokenUrl in existing scheme")
          if (isinstance(self._auth_scheme, OAuth2) and 
              self._auth_scheme.flows and 
              self._auth_scheme.flows.clientCredentials and
              isinstance(discovered_scheme, OAuth2) and
              discovered_scheme.flows and
              discovered_scheme.flows.clientCredentials):
            # Update the tokenUrl with discovered value
            self._auth_scheme.flows.clientCredentials.tokenUrl = discovered_scheme.flows.clientCredentials.tokenUrl
            logger.debug(f"Updated tokenUrl to: {discovered_scheme.flows.clientCredentials.tokenUrl}")
      else:
        logger.debug("❌ OAuth discovery failed - no valid configuration found")
        
    except Exception as e:
      logger.warning(f"❌ OAuth discovery failed with error: {e}")
      
    logger.debug(f"✅ OAuth discovery completed. Final auth_scheme: {self._auth_scheme}")

  @retry_on_closed_resource
  async def get_tools(
      self,
      readonly_context: Optional[ReadonlyContext] = None,
  ) -> List[BaseTool]:
    """Return all tools in the toolset based on the provided context.

    Args:
        readonly_context: Context used to filter tools available to the agent.
            If None, all tools in the toolset are returned.

    Returns:
        List[BaseTool]: A list of tools available under the specified context.
    """
    # Perform OAuth discovery if needed
    await self._perform_oauth_discovery()
    
    # Perform OAuth token exchange before session creation if we have auth
    session_headers = None
    if self._auth_scheme and self._auth_credential:
      logger.debug("🔐 Performing OAuth token exchange for session authentication")
      
      # Get verify_ssl setting from auth_discovery configuration
      verify_ssl = True
      if self._auth_discovery and hasattr(self._auth_discovery, 'verify_ssl'):
        verify_ssl = self._auth_discovery.verify_ssl
      
      # Create a temporary CredentialManager to exchange tokens
      from ...auth.auth_tool import AuthConfig
      from ...auth.credential_manager import CredentialManager
      
      auth_config = AuthConfig(
          auth_scheme=self._auth_scheme,
          raw_auth_credential=self._auth_credential
      )
      
      credential_manager = CredentialManager(auth_config)
      
      # Create a dummy callback context for token exchange
      # This is a simplified approach - in a full implementation this would come from the agent
      class DummyCallbackContext:
        def __init__(self):
          from ...agents.readonly_context import ReadonlyContext
          self._invocation_context = type('obj', (object,), {
            'credential_service': None,
            'app_name': 'mcp_toolset',
            'user_id': 'system'
          })()
          
        def get_auth_response(self, auth_config):
          """Return None since we're using raw credentials for client credentials flow."""
          return None
          
        async def load_credential(self, auth_config):
          """Return None since no stored credentials."""
          return None
          
        async def save_credential(self, auth_config):
          """No-op for dummy context."""
          pass
      
      dummy_context = DummyCallbackContext()
      
      try:
        # Exchange credentials to get access token with SSL verification setting
        exchanged_credential = await credential_manager.get_auth_credential(dummy_context, verify_ssl)
        
        if exchanged_credential and exchanged_credential.oauth2 and exchanged_credential.oauth2.access_token:
          logger.debug(f"✅ Successfully obtained access token for session")
          session_headers = {"Authorization": f"Bearer {exchanged_credential.oauth2.access_token}"}
        else:
          logger.debug("❌ Failed to obtain access token for session")
      except Exception as e:
        logger.debug(f"❌ OAuth token exchange failed: {e}")
    
    # Get session from session manager with OAuth headers
    session = await self._mcp_session_manager.create_session(headers=session_headers)

    # Fetch available tools from the MCP server
    logger.debug("🔍 Calling session.list_tools()")
    tools_response: ListToolsResult = await session.list_tools()
    logger.debug(f"✅ Retrieved {len(tools_response.tools)} tools from MCP server")

    # Apply filtering based on context and tool_filter
    tools = []
    logger.debug(f"🔍 Creating MCPTools with auth_scheme: {self._auth_scheme}")
    logger.debug(f"🔍 Auth credential: {self._auth_credential}")
    
    for tool in tools_response.tools:
      mcp_tool = MCPTool(
          mcp_tool=tool,
          mcp_session_manager=self._mcp_session_manager,
          auth_scheme=self._auth_scheme,
          auth_credential=self._auth_credential,
      )
      
      logger.debug(f"✅ Created MCPTool '{tool.name}' with auth_config: {mcp_tool._credentials_manager is not None}")

      # Handle None readonly_context for _is_tool_selected method
      if readonly_context is None:
        # When no context provided, include tool if tool_filter allows it
        if not self.tool_filter or (isinstance(self.tool_filter, list) and mcp_tool.name in self.tool_filter):
          tools.append(mcp_tool)
      elif self._is_tool_selected(mcp_tool, readonly_context):
        tools.append(mcp_tool)
    return tools

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
