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

"""Tests for MCPToolset OAuth discovery functionality."""

import pytest
import sys
from unittest.mock import AsyncMock, Mock, patch

# Skip all tests in this module if Python version is less than 3.10
pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 10), reason="MCP tool requires Python 3.10+"
)

try:
    from fastapi.openapi.models import OAuth2, OAuthFlows, OAuthFlowClientCredentials
    from google.adk.auth.auth_credential import AuthCredential, AuthCredentialTypes, OAuth2Auth
    from google.adk.tools.mcp_tool.mcp_auth_discovery import MCPAuthDiscovery
    from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset
    from google.adk.tools.mcp_tool.mcp_session_manager import (
        StreamableHTTPConnectionParams,
        SseConnectionParams,
        StdioConnectionParams
    )
    from mcp import StdioServerParameters
except ImportError:
    if sys.version_info >= (3, 10):
        raise
    # Create dummy classes for older Python versions
    class OAuth2:
        pass
    class OAuthFlows:
        pass
    class OAuthFlowClientCredentials:
        pass
    class AuthCredential:
        pass
    class AuthCredentialTypes:
        OAUTH2 = "oauth2"
    class OAuth2Auth:
        pass
    class MCPAuthDiscovery:
        pass
    class MCPToolset:
        pass
    class StreamableHTTPConnectionParams:
        pass
    class SseConnectionParams:
        pass
    class StdioConnectionParams:
        pass
    class StdioServerParameters:
        pass


class TestMCPToolsetOAuthDiscovery:
    """Test suite for MCPToolset OAuth discovery functionality."""

    def test_default_auth_discovery_streamable_http(self):
        """Test that MCPToolset creates default MCPAuthDiscovery for StreamableHTTP connections."""
        connection_params = StreamableHTTPConnectionParams(
            url="http://localhost:9204/mcp/"
        )
        
        toolset = MCPToolset(connection_params=connection_params)
        
        # Verify default auth discovery was created
        assert toolset._auth_discovery is not None
        assert toolset._auth_discovery.base_url == "http://localhost:9204"
        assert toolset._auth_discovery.enabled is True
        assert toolset._auth_discovery.timeout == 10.0

    def test_default_auth_discovery_sse(self):
        """Test that MCPToolset creates default MCPAuthDiscovery for SSE connections."""
        connection_params = SseConnectionParams(
            url="http://server.example.com:8080/sse/"
        )
        
        toolset = MCPToolset(connection_params=connection_params)
        
        # Verify default auth discovery was created
        assert toolset._auth_discovery is not None
        assert toolset._auth_discovery.base_url == "http://server.example.com:8080"
        assert toolset._auth_discovery.enabled is True

    def test_default_auth_discovery_stdio_disabled(self):
        """Test that MCPToolset disables OAuth discovery for Stdio connections."""
        connection_params = StdioConnectionParams(
            server_params=StdioServerParameters(command="test", args=[])
        )
        
        toolset = MCPToolset(connection_params=connection_params)
        
        # Verify auth discovery is disabled for stdio
        assert toolset._auth_discovery is not None
        assert toolset._auth_discovery.enabled is False

    def test_explicit_auth_discovery_overrides_default(self):
        """Test that explicit auth_discovery parameter overrides default behavior."""
        connection_params = StreamableHTTPConnectionParams(
            url="http://localhost:9204/mcp/"
        )
        
        custom_discovery = MCPAuthDiscovery(
            base_url="http://custom-auth.example.com",
            timeout=15.0
        )
        
        toolset = MCPToolset(
            connection_params=connection_params,
            auth_discovery=custom_discovery
        )
        
        # Verify custom discovery is used
        assert toolset._auth_discovery is custom_discovery
        assert toolset._auth_discovery.base_url == "http://custom-auth.example.com"
        assert toolset._auth_discovery.timeout == 15.0

    def test_disabled_auth_discovery_override(self):
        """Test that explicitly disabled auth_discovery overrides default enabling."""
        connection_params = StreamableHTTPConnectionParams(
            url="http://localhost:9204/mcp/"
        )
        
        disabled_discovery = MCPAuthDiscovery(
            base_url="http://localhost:9204",
            enabled=False
        )
        
        toolset = MCPToolset(
            connection_params=connection_params,
            auth_discovery=disabled_discovery
        )
        
        # Verify discovery is disabled despite HTTP connection
        assert toolset._auth_discovery.enabled is False
        assert toolset._auth_discovery.is_enabled is False

    @patch("google.adk.tools.mcp_tool.mcp_toolset.create_oauth_scheme_from_discovery")
    @pytest.mark.asyncio
    async def test_oauth_discovery_not_attempted_when_disabled(self, mock_discovery):
        """Test that OAuth discovery is not attempted when disabled."""
        connection_params = StreamableHTTPConnectionParams(
            url="http://localhost:9204/mcp/"
        )
        
        toolset = MCPToolset(
            connection_params=connection_params,
            auth_discovery=MCPAuthDiscovery(
                base_url="http://localhost:9204",
                enabled=False
            )
        )
        
        # Call the discovery method
        await toolset._perform_oauth_discovery()
        
        # Verify discovery was not attempted
        mock_discovery.assert_not_called()
        assert toolset._oauth_discovery_attempted is False

    @patch("google.adk.tools.mcp_tool.mcp_toolset.create_oauth_scheme_from_discovery")
    @pytest.mark.asyncio
    async def test_oauth_discovery_with_empty_token_url(self, mock_discovery):
        """Test OAuth discovery when auth_scheme has empty tokenUrl."""
        mock_discovery.return_value = OAuth2(
            flows=OAuthFlows(
                clientCredentials=OAuthFlowClientCredentials(
                    tokenUrl="http://localhost:9204/token",
                    scopes={"api:read": "Read access"}
                )
            )
        )
        
        connection_params = StreamableHTTPConnectionParams(
            url="http://localhost:9204/mcp/"
        )
        
        auth_scheme = OAuth2(
            flows=OAuthFlows(
                clientCredentials=OAuthFlowClientCredentials(
                    tokenUrl="",  # Empty tokenUrl triggers discovery
                    scopes={"api:read": "Read access"}
                )
            )
        )
        
        toolset = MCPToolset(
            connection_params=connection_params,
            auth_scheme=auth_scheme
        )
        
        # Call the discovery method
        await toolset._perform_oauth_discovery()
        
        # Verify discovery was attempted with correct parameters
        mock_discovery.assert_called_once_with(
            base_url="http://localhost:9204",
            scopes=["api:read"],
            timeout=10.0
        )
        
        # Verify tokenUrl was updated
        assert isinstance(toolset._auth_scheme, OAuth2)
        assert toolset._auth_scheme.flows.clientCredentials.tokenUrl == "http://localhost:9204/token"
        assert toolset._oauth_discovery_attempted is True

    @patch("google.adk.tools.mcp_tool.mcp_toolset.create_oauth_scheme_from_discovery")
    @pytest.mark.asyncio
    async def test_oauth_discovery_with_no_auth_scheme(self, mock_discovery):
        """Test OAuth discovery when no auth_scheme is provided."""
        mock_discovery.return_value = OAuth2(
            flows=OAuthFlows(
                clientCredentials=OAuthFlowClientCredentials(
                    tokenUrl="http://localhost:9204/token",
                    scopes={}
                )
            )
        )
        
        connection_params = StreamableHTTPConnectionParams(
            url="http://localhost:9204/mcp/"
        )
        
        toolset = MCPToolset(connection_params=connection_params)
        
        # Call the discovery method
        await toolset._perform_oauth_discovery()
        
        # Verify discovery was attempted
        mock_discovery.assert_called_once_with(
            base_url="http://localhost:9204",
            scopes=None,
            timeout=10.0
        )
        
        # Verify auth_scheme was set from discovery
        assert toolset._auth_scheme is not None
        assert isinstance(toolset._auth_scheme, OAuth2)
        assert toolset._auth_scheme.flows.clientCredentials.tokenUrl == "http://localhost:9204/token"

    @patch("google.adk.tools.mcp_tool.mcp_toolset.create_oauth_scheme_from_discovery")
    @pytest.mark.asyncio
    async def test_oauth_discovery_failure_handling(self, mock_discovery):
        """Test handling of OAuth discovery failures."""
        mock_discovery.return_value = None  # Discovery fails
        
        connection_params = StreamableHTTPConnectionParams(
            url="http://localhost:9204/mcp/"
        )
        
        toolset = MCPToolset(connection_params=connection_params)
        
        # Call the discovery method
        await toolset._perform_oauth_discovery()
        
        # Verify discovery was attempted but auth_scheme remains None
        mock_discovery.assert_called_once()
        assert toolset._auth_scheme is None
        assert toolset._oauth_discovery_attempted is True

    @patch("google.adk.tools.mcp_tool.mcp_toolset.create_oauth_scheme_from_discovery")
    @pytest.mark.asyncio
    async def test_oauth_discovery_exception_handling(self, mock_discovery):
        """Test handling of exceptions during OAuth discovery."""
        mock_discovery.side_effect = Exception("Discovery failed")
        
        connection_params = StreamableHTTPConnectionParams(
            url="http://localhost:9204/mcp/"
        )
        
        toolset = MCPToolset(connection_params=connection_params)
        
        # Call the discovery method - should not raise
        await toolset._perform_oauth_discovery()
        
        # Verify discovery was attempted and exception was handled
        mock_discovery.assert_called_once()
        assert toolset._oauth_discovery_attempted is True

    @pytest.mark.asyncio
    async def test_oauth_discovery_only_attempted_once(self):
        """Test that OAuth discovery is only attempted once even with multiple calls."""
        connection_params = StreamableHTTPConnectionParams(
            url="http://localhost:9204/mcp/"
        )
        
        toolset = MCPToolset(connection_params=connection_params)
        
        with patch("google.adk.tools.mcp_tool.mcp_toolset.create_oauth_scheme_from_discovery") as mock_discovery:
            mock_discovery.return_value = None
            
            # Call discovery multiple times
            await toolset._perform_oauth_discovery()
            await toolset._perform_oauth_discovery()
            await toolset._perform_oauth_discovery()
            
            # Verify discovery was only called once
            assert mock_discovery.call_count == 1
            assert toolset._oauth_discovery_attempted is True

    def test_url_parsing_with_complex_paths(self):
        """Test URL parsing with complex paths extracts correct base URL."""
        connection_params = StreamableHTTPConnectionParams(
            url="https://api.example.com:8443/services/v1/mcp/stream"
        )
        
        toolset = MCPToolset(connection_params=connection_params)
        
        # Verify correct base URL extraction
        assert toolset._auth_discovery.base_url == "https://api.example.com:8443"

    def test_url_parsing_with_query_parameters(self):
        """Test URL parsing ignores query parameters and fragments."""
        connection_params = StreamableHTTPConnectionParams(
            url="http://localhost:9204/mcp/?version=1.0&debug=true#section"
        )
        
        toolset = MCPToolset(connection_params=connection_params)
        
        # Verify query parameters and fragments are ignored
        assert toolset._auth_discovery.base_url == "http://localhost:9204"

    @patch("google.adk.auth.credential_manager.CredentialManager.get_auth_credential")
    @patch("google.adk.tools.mcp_tool.mcp_session_manager.MCPSessionManager.create_session")
    @pytest.mark.asyncio
    async def test_token_exchange_before_session_creation(self, mock_create_session, mock_get_credential):
        """Test that token exchange happens before session creation."""
        # Setup mocks
        mock_credential = Mock()
        mock_credential.oauth2.access_token = "test_access_token"
        mock_get_credential.return_value = mock_credential
        
        mock_session = AsyncMock()
        mock_session.list_tools.return_value = Mock(tools=[])
        mock_create_session.return_value = mock_session
        
        connection_params = StreamableHTTPConnectionParams(
            url="http://localhost:9204/mcp/"
        )
        
        auth_scheme = OAuth2(
            flows=OAuthFlows(
                clientCredentials=OAuthFlowClientCredentials(
                    tokenUrl="http://localhost:9204/token",
                    scopes={"api:read": "Read access"}
                )
            )
        )
        
        auth_credential = AuthCredential(
            auth_type=AuthCredentialTypes.OAUTH2,
            oauth2=OAuth2Auth(client_id="test_id", client_secret="test_secret")
        )
        
        toolset = MCPToolset(
            connection_params=connection_params,
            auth_scheme=auth_scheme,
            auth_credential=auth_credential
        )
        
        # Call get_tools to trigger token exchange and session creation
        with patch.object(toolset, '_perform_oauth_discovery'):
            await toolset.get_tools(readonly_context=None)
        
        # Verify token exchange was called
        mock_get_credential.assert_called_once()
        
        # Verify session was created with Authorization header
        mock_create_session.assert_called_once()
        call_args = mock_create_session.call_args
        headers = call_args.kwargs.get('headers', {})
        assert headers.get('Authorization') == 'Bearer test_access_token' 