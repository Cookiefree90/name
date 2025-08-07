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


import os

from google.adk.agents.llm_agent import LlmAgent
from google.adk.tools.mcp_tool import StdioConnectionParams, StreamableHTTPConnectionParams
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset
from mcp import StdioServerParameters

# Example 1: Basic MCP Toolset with Stdio (existing example)
_allowed_path = os.path.dirname(os.path.abspath(__file__))

basic_mcp_toolset = MCPToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command='npx',
            args=[
                '-y',  # Arguments for the command
                '@modelcontextprotocol/server-filesystem',
                _allowed_path,
            ],
        ),
        timeout=5,
    ),
    # don't want agent to do write operation
    # you can also do below
    # tool_filter=lambda tool, ctx=None: tool.name
    # not in [
    #     'write_file',
    #     'edit_file',
    #     'create_directory',
    #     'move_file',
    # ],
    tool_filter=[
        'read_file',
        'read_multiple_files',
        'list_directory',
        'directory_tree',
        'search_files',
        'get_file_info',
        'list_allowed_directories',
    ],
)

# Example 2: MCP Toolset with OAuth2 Client Credentials + Auto Discovery (new functionality!)
def create_oauth2_auto_discovery_mcp_toolset():
    """
    Example of creating an MCPToolset with automatic OAuth discovery.
    
    This demonstrates the powerful new OAuth discovery feature added to ADK.
    No manual token URL configuration needed!
    """
    from google.adk.auth.auth_credential import AuthCredential, AuthCredentialTypes, OAuth2Auth

    # Configuration - replace with your actual values
    CLIENT_ID = os.getenv("OAUTH_CLIENT_ID", "your_client_id_here")
    CLIENT_SECRET = os.getenv("OAUTH_CLIENT_SECRET", "your_client_secret_here")
    SERVER_URL = "https://your-mcp-server.com"  # Your OAuth-protected MCP server

    # Create auth credential (no auth scheme needed for auto-discovery!)
    auth_credential = AuthCredential(
        auth_type=AuthCredentialTypes.OAUTH2,
        oauth2=OAuth2Auth(
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET,
        )
    )

    # Create MCPToolset with automatic OAuth discovery
    # ADK will automatically:
    # 1. Query .well-known/oauth-protected-resource 
    # 2. Query .well-known/oauth-authorization-server
    # 3. Extract token endpoints and create OAuth2 scheme
    # 4. Handle token exchange and refresh seamlessly
    toolset = MCPToolset(
        connection_params=StreamableHTTPConnectionParams(
            url=SERVER_URL
        ),
        auth_credential=auth_credential,
        auto_discover_oauth=True,           # Enable automatic discovery!
        discovery_timeout=10.0,             # Discovery timeout (optional)
        discovery_scopes=["read", "write"], # Scopes to request (optional)
        tool_filter=["list_tools", "call_tool"]  # Optional tool filtering
    )
    
    return toolset

# Example 3: MCP Toolset with Manual OAuth2 Client Credentials (traditional approach)
def create_oauth2_manual_mcp_toolset():
    """
    Example of creating an MCPToolset with manual OAuth configuration.
    
    Use this approach when you need explicit control over OAuth endpoints
    or when the server doesn't support discovery.
    """
    from fastapi.openapi.models import OAuth2, OAuthFlows, OAuthFlowClientCredentials
    from google.adk.auth.auth_credential import AuthCredential, AuthCredentialTypes, OAuth2Auth

    # Configuration - replace with your actual values
    CLIENT_ID = os.getenv("OAUTH_CLIENT_ID", "your_client_id_here")
    CLIENT_SECRET = os.getenv("OAUTH_CLIENT_SECRET", "your_client_secret_here")
    TOKEN_URL = "https://auth.example.com/token"
    SERVER_URL = "https://your-mcp-server.com"

    # Manually create OAuth2 auth scheme
    auth_scheme = OAuth2(
        flows=OAuthFlows(
            clientCredentials=OAuthFlowClientCredentials(
                tokenUrl=TOKEN_URL,
                scopes={
                    "read": "Read access",
                    "write": "Write access"
                }
            )
        )
    )

    # Create auth credential
    auth_credential = AuthCredential(
        auth_type=AuthCredentialTypes.OAUTH2,
        oauth2=OAuth2Auth(
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET,
        )
    )

    # Create MCPToolset with explicit OAuth configuration
    toolset = MCPToolset(
        connection_params=StreamableHTTPConnectionParams(
            url=SERVER_URL
        ),
        auth_scheme=auth_scheme,        # Explicit auth scheme
        auth_credential=auth_credential,
        auto_discover_oauth=False,      # Disable discovery when using explicit scheme
    )
    
    return toolset

# Create the authenticated toolset (will be None if not configured)
authenticated_toolset = create_oauth2_auto_discovery_mcp_toolset()

# Create agent with both file system and (optionally) authenticated MCP toolset
if authenticated_toolset:
    # Both file system and authenticated toolsets
    toolsets = [basic_mcp_toolset, authenticated_toolset]
else:
    # Just file system toolset
    toolsets = [basic_mcp_toolset]

agent = LlmAgent(
    model='gemini-2.0-flash-exp',
    name='file_mcp_agent',
    instruction="""
You are a helpful assistant with access to file system operations via MCP tools.
You can read files, list directories, and perform other file operations as requested.

If OAuth-protected MCP tools are available, you can also access authenticated services.
    """,
    tools=toolsets,
)
