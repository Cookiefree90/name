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

"""
OAuth2 Client Credentials Flow with Automatic Discovery Sample

This sample demonstrates the OAuth2 client credentials authentication flow
with automatic OAuth discovery for MCP servers. It shows multiple scenarios
from basic automatic discovery to custom configurations.
"""

import os
from fastapi.openapi.models import OAuth2, OAuthFlows, OAuthFlowClientCredentials

from google.adk.agents.llm_agent import LlmAgent
from google.adk.auth.auth_credential import AuthCredential, AuthCredentialTypes, OAuth2Auth
from google.adk.tools.mcp_tool.mcp_auth_discovery import MCPAuthDiscovery
from google.adk.tools.mcp_tool.mcp_session_manager import StreamableHTTPConnectionParams
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset


def create_oauth2_credential(client_id: str, client_secret: str) -> AuthCredential:
    """Helper function to create OAuth2 credentials."""
    return AuthCredential(
        auth_type=AuthCredentialTypes.OAUTH2,
        oauth2=OAuth2Auth(
            client_id=client_id,
            client_secret=client_secret
        )
    )


def create_oauth2_scheme(token_url: str, scopes: dict[str, str]) -> OAuth2:
    """Helper function to create OAuth2 auth scheme."""
    return OAuth2(
        flows=OAuthFlows(
            clientCredentials=OAuthFlowClientCredentials(
                tokenUrl=token_url,
                scopes=scopes
            )
        )
    )


# =============================================================================
# Scenario 1: Automatic OAuth Discovery (Simplest Case)
# =============================================================================

# This is the simplest way to use OAuth2 client credentials with MCP.
# MCPToolset automatically:
# 1. Extracts base URL from the MCP connection (http://localhost:9204)
# 2. Enables OAuth discovery for HTTP-based connections
# 3. Discovers OAuth endpoints via RFC 8414 two-stage process
# 4. Exchanges client credentials for access tokens
# 5. Authenticates all MCP requests

automatic_discovery_agent = LlmAgent(
    model='gemini-2.0-flash',
    name='oauth2_automatic_agent',
    instruction="""
You are an assistant that can access secured MCP tools using OAuth2 authentication.
The OAuth2 discovery and authentication happens automatically behind the scenes.
Help users by calling the available MCP tools as needed.
    """,
    tools=[
        MCPToolset(
            # MCP server connection
            connection_params=StreamableHTTPConnectionParams(
                url='http://localhost:9204/mcp/',
            ),
            # OAuth2 credentials (client_id and client_secret)
            auth_credential=create_oauth2_credential(
                client_id=os.getenv('OAUTH2_CLIENT_ID', 'demo_client_id'),
                client_secret=os.getenv('OAUTH2_CLIENT_SECRET', 'demo_client_secret')
            ),
            # Optional: Define scopes in auth_scheme (will be used during discovery)
            auth_scheme=create_oauth2_scheme(
                token_url="",  # Empty - will be automatically discovered
                scopes={
                    "api:read": "Read access to API",
                    "api:write": "Write access to API"
                }
            ),
            # No auth_discovery parameter needed - automatic discovery enabled!
        )
    ],
)


# =============================================================================
# Scenario 2: Custom OAuth Discovery Configuration
# =============================================================================

# Sometimes you need to customize the OAuth discovery process:
# - Use a different OAuth server than the MCP server
# - Adjust discovery timeout
# - Specify exact discovery endpoints

custom_discovery_agent = LlmAgent(
    model='gemini-2.0-flash',
    name='oauth2_custom_discovery_agent',
    instruction="""
You are an assistant using custom OAuth2 discovery configuration.
This demonstrates how to override default discovery behavior.
    """,
    tools=[
        MCPToolset(
            connection_params=StreamableHTTPConnectionParams(
                url='http://localhost:9204/mcp/',
            ),
            auth_credential=create_oauth2_credential(
                client_id=os.getenv('OAUTH2_CLIENT_ID', 'demo_client_id'),
                client_secret=os.getenv('OAUTH2_CLIENT_SECRET', 'demo_client_secret')
            ),
            auth_scheme=create_oauth2_scheme(
                token_url="",  # Empty - will be discovered
                scopes={"api:read": "Read access"}
            ),
            # Custom OAuth discovery configuration
            auth_discovery=MCPAuthDiscovery(
                base_url='http://auth-server.example.com:9205',  # Custom auth server
                timeout=15.0,  # Custom timeout
                enabled=True
            ),
        )
    ],
)


# =============================================================================
# Scenario 3: No Auth Scheme (Discovery Creates Complete Scheme)
# =============================================================================

# When no auth_scheme is provided, OAuth discovery will create a complete
# OAuth2 scheme with discovered endpoints and default scopes

discovery_only_agent = LlmAgent(
    model='gemini-2.0-flash',
    name='oauth2_discovery_only_agent',
    instruction="""
You are an assistant where OAuth2 discovery creates the complete auth scheme.
This shows the most minimal configuration possible.
    """,
    tools=[
        MCPToolset(
            connection_params=StreamableHTTPConnectionParams(
                url='http://localhost:9204/mcp/',
            ),
            auth_credential=create_oauth2_credential(
                client_id=os.getenv('OAUTH2_CLIENT_ID', 'demo_client_id'),
                client_secret=os.getenv('OAUTH2_CLIENT_SECRET', 'demo_client_secret')
            ),
            # No auth_scheme - discovery will create one
            # No auth_discovery - automatic discovery enabled
        )
    ],
)


# =============================================================================
# Scenario 4: Disabled OAuth Discovery (Manual Configuration)
# =============================================================================

# Sometimes you want to disable automatic discovery and provide
# the complete OAuth configuration manually

manual_config_agent = LlmAgent(
    model='gemini-2.0-flash',
    name='oauth2_manual_config_agent',
    instruction="""
You are an assistant using manual OAuth2 configuration without discovery.
This demonstrates traditional OAuth2 setup without automatic discovery.
    """,
    tools=[
        MCPToolset(
            connection_params=StreamableHTTPConnectionParams(
                url='http://localhost:9204/mcp/',
            ),
            auth_credential=create_oauth2_credential(
                client_id=os.getenv('OAUTH2_CLIENT_ID', 'demo_client_id'),
                client_secret=os.getenv('OAUTH2_CLIENT_SECRET', 'demo_client_secret')
            ),
            # Complete OAuth2 scheme with known token endpoint
            auth_scheme=create_oauth2_scheme(
                token_url="http://localhost:9204/token",  # Known token endpoint
                scopes={"api:read": "Read access", "api:write": "Write access"}
            ),
            # Explicitly disable OAuth discovery
            auth_discovery=MCPAuthDiscovery(
                base_url="http://localhost:9204",
                enabled=False  # Discovery disabled
            ),
        )
    ],
)


# =============================================================================
# Scenario 5: Multiple MCP Servers with Different Auth Configurations
# =============================================================================

# This demonstrates using multiple MCP servers with different OAuth setups

multi_server_agent = LlmAgent(
    model='gemini-2.0-flash',
    name='oauth2_multi_server_agent',
    instruction="""
You are an assistant that can access multiple secured MCP servers,
each with their own OAuth2 configuration. Choose the appropriate
toolset based on the user's needs.
    """,
    tools=[
        # Server 1: Automatic discovery
        MCPToolset(
            connection_params=StreamableHTTPConnectionParams(
                url='http://server1.example.com:9204/mcp/',
            ),
            auth_credential=create_oauth2_credential(
                client_id=os.getenv('SERVER1_CLIENT_ID', 'server1_client'),
                client_secret=os.getenv('SERVER1_CLIENT_SECRET', 'server1_secret')
            ),
            tool_filter=['list_tools', 'get_info'],  # Simple string list filter
        ),
        
        # Server 2: Custom discovery server
        MCPToolset(
            connection_params=StreamableHTTPConnectionParams(
                url='http://server2.example.com:8080/api/mcp/',
            ),
            auth_credential=create_oauth2_credential(
                client_id=os.getenv('SERVER2_CLIENT_ID', 'server2_client'),
                client_secret=os.getenv('SERVER2_CLIENT_SECRET', 'server2_secret')
            ),
            auth_discovery=MCPAuthDiscovery(
                base_url='http://auth.server2.example.com:9000',
                timeout=20.0
            ),
            tool_filter=['query_data', 'update_records'],  # Simple string list filter
        ),
    ],
)


# =============================================================================
# Scenario 6: Self-Signed Certificates (SSL Verification Disabled)
# =============================================================================

# This scenario demonstrates using OAuth2 discovery with self-signed certificates
# by disabling SSL certificate verification - useful for development environments
# Note: base_url is auto-extracted from connection_params, only verify_ssl is overridden

self_signed_ssl_agent = LlmAgent(
    model='gemini-2.0-flash',
    name='oauth2_self_signed_ssl_agent',
    instruction="""
You are an assistant that can access MCP servers using self-signed SSL certificates.
SSL certificate verification is disabled for development environments.
    """,
    tools=[
        MCPToolset(
            connection_params=StreamableHTTPConnectionParams(
                url='https://localhost:9204/mcp/',  # HTTPS with self-signed cert
            ),
            auth_credential=create_oauth2_credential(
                client_id=os.getenv('OAUTH2_CLIENT_ID', 'demo_client_id'),
                client_secret=os.getenv('OAUTH2_CLIENT_SECRET', 'demo_client_secret')
            ),
            # Override just SSL verification - base_url auto-extracted from connection_params
            auth_discovery=MCPAuthDiscovery(
                verify_ssl=False,  # Only override SSL verification
                # base_url auto-extracted as "https://localhost:9204" from connection_params
            ),
        )
    ],
)


# =============================================================================
# Scenario 7: Custom Settings Without Base URL Override
# =============================================================================

# This scenario shows overriding multiple discovery settings while letting
# MCPToolset auto-extract the base_url from connection parameters

custom_settings_agent = LlmAgent(
    model='gemini-2.0-flash',
    name='oauth2_custom_settings_agent',
    instruction="""
You are an assistant with custom OAuth discovery settings.
The base URL is automatically extracted from the MCP connection.
    """,
    tools=[
        MCPToolset(
            connection_params=StreamableHTTPConnectionParams(
                url='http://localhost:9204/mcp/',
            ),
            auth_credential=create_oauth2_credential(
                client_id=os.getenv('OAUTH2_CLIENT_ID', 'demo_client_id'),
                client_secret=os.getenv('OAUTH2_CLIENT_SECRET', 'demo_client_secret')
            ),
            # Override multiple settings - base_url auto-extracted
            auth_discovery=MCPAuthDiscovery(
                timeout=15.0,      # Custom timeout
                verify_ssl=True,   # Explicit SSL verification (default)
                enabled=True       # Explicit enabled (default)
                # base_url auto-extracted as "http://localhost:9204" from connection_params
            ),
        )
    ],
)


# =============================================================================
# Default agent for the sample (most commonly used scenario)
# =============================================================================

# The default agent demonstrates the most common usage: automatic OAuth discovery
root_agent = automatic_discovery_agent 