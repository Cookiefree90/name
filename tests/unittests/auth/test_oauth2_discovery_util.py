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

"""Tests for OAuth2 discovery utilities."""

import json
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest
from fastapi.openapi.models import OAuth2, OAuthFlows, OAuthFlowClientCredentials

# Import from the correct path
import sys
sys.path.insert(0, '../../../src')

from google.adk.auth.oauth2_discovery_util import (
    discover_oauth_configuration,
    create_oauth_scheme_from_discovery,
    _validate_oauth_discovery_response,
    OAUTH_PROTECTED_RESOURCE_DISCOVERY,
    OAUTH_AUTHORIZATION_SERVER_DISCOVERY,
)


class TestDiscoverOAuthConfiguration:
    """Test suite for discover_oauth_configuration function."""

    @patch("google.adk.auth.oauth2_discovery_util.httpx.AsyncClient")
    @pytest.mark.asyncio
    async def test_discovery_oauth_protected_resource_success(self, mock_async_client):
        """Test successful OAuth discovery using oauth-protected-resource endpoint."""
        # Setup mock response
        mock_response = Mock()
        mock_response.json.return_value = {
            "authorization_servers": ["https://auth.example.com"]
        }
        mock_response.raise_for_status = Mock()
        
        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_async_client.return_value.__aenter__.return_value = mock_client
        
        # Test discovery
        result = await discover_oauth_configuration("https://api.example.com")
        
        # Verify result
        assert result == {"authorization_servers": ["https://auth.example.com"]}
        
        # Verify first endpoint was called
        mock_client.get.assert_called_with(
            f"https://api.example.com/{OAUTH_PROTECTED_RESOURCE_DISCOVERY}"
        )

    @patch("google.adk.auth.oauth2_discovery_util.httpx.AsyncClient")
    @pytest.mark.asyncio
    async def test_discovery_oauth_authorization_server_success(self, mock_async_client):
        """Test successful OAuth discovery using oauth-authorization-server endpoint."""
        # Setup mock to fail on first endpoint, succeed on second
        mock_response1 = Mock()
        mock_response1.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Not found", request=Mock(), response=Mock(status_code=404)
        )
        
        mock_response2 = Mock()
        mock_response2.json.return_value = {
            "token_endpoint": "https://auth.example.com/token"
        }
        mock_response2.raise_for_status = Mock()
        
        mock_client = AsyncMock()
        mock_client.get.side_effect = [mock_response1, mock_response2]
        mock_async_client.return_value.__aenter__.return_value = mock_client
        
        # Test discovery
        result = await discover_oauth_configuration("https://api.example.com")
        
        # Verify result
        assert result == {"token_endpoint": "https://auth.example.com/token"}
        
        # Verify both endpoints were called
        assert mock_client.get.call_count == 2

    @patch("google.adk.auth.oauth2_discovery_util.httpx.AsyncClient")
    @pytest.mark.asyncio
    async def test_discovery_failure_all_endpoints(self, mock_async_client):
        """Test OAuth discovery failure when all endpoints fail."""
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Not found", request=Mock(), response=Mock(status_code=404)
        )
        
        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_async_client.return_value.__aenter__.return_value = mock_client
        
        # Test discovery
        result = await discover_oauth_configuration("https://api.example.com")
        
        # Verify result
        assert result is None
        
        # Verify both endpoints were tried
        assert mock_client.get.call_count == 2

    @patch("google.adk.auth.oauth2_discovery_util.httpx.AsyncClient")
    @pytest.mark.asyncio
    async def test_discovery_invalid_response(self, mock_async_client):
        """Test OAuth discovery with invalid response format."""
        mock_response = Mock()
        mock_response.json.return_value = {"invalid": "response"}
        mock_response.raise_for_status = Mock()
        
        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_async_client.return_value.__aenter__.return_value = mock_client
        
        # Test discovery
        result = await discover_oauth_configuration("https://api.example.com")
        
        # Verify result is None due to validation failure
        assert result is None

    @patch("google.adk.auth.oauth2_discovery_util.httpx.AsyncClient")
    @pytest.mark.asyncio
    async def test_discovery_json_decode_error(self, mock_async_client):
        """Test OAuth discovery with JSON decode error."""
        mock_response = Mock()
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        mock_response.raise_for_status = Mock()
        
        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_async_client.return_value.__aenter__.return_value = mock_client
        
        # Test discovery
        result = await discover_oauth_configuration("https://api.example.com")
        
        # Verify result is None due to JSON error
        assert result is None

    @pytest.mark.asyncio
    async def test_discovery_with_custom_timeout(self):
        """Test OAuth discovery with custom timeout."""
        with patch("google.adk.auth.oauth2_discovery_util.httpx.AsyncClient") as mock_async_client:
            mock_client = AsyncMock()
            mock_async_client.return_value.__aenter__.return_value = mock_client
            
            await discover_oauth_configuration("https://api.example.com", timeout=5.0)
            
            # Verify timeout was passed to AsyncClient
            mock_async_client.assert_called_with(timeout=5.0)


class TestValidateOAuthDiscoveryResponse:
    """Test suite for _validate_oauth_discovery_response function."""

    def test_validate_oauth_protected_resource_valid(self):
        """Test validation of valid oauth-protected-resource response."""
        config = {"authorization_servers": ["https://auth.example.com"]}
        assert _validate_oauth_discovery_response(config) is True

    def test_validate_oauth_protected_resource_empty_list(self):
        """Test validation of oauth-protected-resource response with empty server list."""
        config = {"authorization_servers": []}
        assert _validate_oauth_discovery_response(config) is False

    def test_validate_oauth_authorization_server_valid(self):
        """Test validation of valid oauth-authorization-server response."""
        config = {"token_endpoint": "https://auth.example.com/token"}
        assert _validate_oauth_discovery_response(config) is True

    def test_validate_oauth_authorization_server_empty_endpoint(self):
        """Test validation of oauth-authorization-server response with empty endpoint."""
        config = {"token_endpoint": ""}
        assert _validate_oauth_discovery_response(config) is False

    def test_validate_invalid_response(self):
        """Test validation of invalid response format."""
        config = {"invalid": "response"}
        assert _validate_oauth_discovery_response(config) is False

    def test_validate_mixed_valid_response(self):
        """Test validation prioritizes oauth-protected-resource over authorization-server."""
        config = {
            "authorization_servers": ["https://auth.example.com"],
            "token_endpoint": "https://auth.example.com/token"
        }
        # Should return True because authorization_servers is checked first
        assert _validate_oauth_discovery_response(config) is True


class TestCreateOAuthSchemeFromDiscovery:
    """Test suite for create_oauth_scheme_from_discovery function."""

    @patch("google.adk.auth.oauth2_discovery_util.discover_oauth_configuration")
    @pytest.mark.asyncio
    async def test_create_scheme_oauth_protected_resource(self, mock_discover):
        """Test OAuth scheme creation from oauth-protected-resource discovery."""
        # Setup mock discovery response
        mock_discover.return_value = {
            "authorization_servers": ["https://auth.example.com"]
        }
        
        # Mock nested discovery for auth server
        with patch("google.adk.auth.oauth2_discovery_util.discover_oauth_configuration") as mock_nested_discover:
            mock_nested_discover.side_effect = [
                {"authorization_servers": ["https://auth.example.com"]},  # First call
                {"token_endpoint": "https://auth.example.com/token"}       # Nested call for auth server
            ]
            
            result = await create_oauth_scheme_from_discovery("https://api.example.com")
            
            # Verify result
            assert isinstance(result, OAuth2)
            assert result.flows.clientCredentials.tokenUrl == "https://auth.example.com/token"
            assert "read" in result.flows.clientCredentials.scopes
            assert "write" in result.flows.clientCredentials.scopes

    @patch("google.adk.auth.oauth2_discovery_util.discover_oauth_configuration")
    @pytest.mark.asyncio
    async def test_create_scheme_oauth_authorization_server(self, mock_discover):
        """Test OAuth scheme creation from oauth-authorization-server discovery."""
        mock_discover.return_value = {
            "token_endpoint": "https://auth.example.com/token"
        }
        
        result = await create_oauth_scheme_from_discovery("https://api.example.com")
        
        # Verify result
        assert isinstance(result, OAuth2)
        assert result.flows.clientCredentials.tokenUrl == "https://auth.example.com/token"
        assert "read" in result.flows.clientCredentials.scopes
        assert "write" in result.flows.clientCredentials.scopes

    @patch("google.adk.auth.oauth2_discovery_util.discover_oauth_configuration")
    @pytest.mark.asyncio
    async def test_create_scheme_custom_scopes(self, mock_discover):
        """Test OAuth scheme creation with custom scopes."""
        mock_discover.return_value = {
            "token_endpoint": "https://auth.example.com/token"
        }
        
        custom_scopes = ["admin", "user:read"]
        result = await create_oauth_scheme_from_discovery(
            "https://api.example.com",
            scopes=custom_scopes
        )
        
        # Verify custom scopes
        assert isinstance(result, OAuth2)
        assert "admin" in result.flows.clientCredentials.scopes
        assert "user:read" in result.flows.clientCredentials.scopes
        assert "read" not in result.flows.clientCredentials.scopes

    @patch("google.adk.auth.oauth2_discovery_util.discover_oauth_configuration")
    @pytest.mark.asyncio
    async def test_create_scheme_discovery_failure(self, mock_discover):
        """Test OAuth scheme creation when discovery fails."""
        mock_discover.return_value = None
        
        result = await create_oauth_scheme_from_discovery("https://api.example.com")
        
        # Verify result is None
        assert result is None

    @patch("google.adk.auth.oauth2_discovery_util.discover_oauth_configuration")
    @pytest.mark.asyncio
    async def test_create_scheme_fallback_token_endpoint(self, mock_discover):
        """Test OAuth scheme creation with fallback token endpoint."""
        # Setup mock to fail on nested discovery
        mock_discover.side_effect = [
            {"authorization_servers": ["https://auth.example.com"]},  # First call
            None  # Nested call fails
        ]
        
        result = await create_oauth_scheme_from_discovery("https://api.example.com")
        
        # Verify fallback endpoint is used
        assert isinstance(result, OAuth2)
        assert result.flows.clientCredentials.tokenUrl == "https://auth.example.com/token"

    @patch("google.adk.auth.oauth2_discovery_util.discover_oauth_configuration")
    @pytest.mark.asyncio
    async def test_create_scheme_no_token_endpoint(self, mock_discover):
        """Test OAuth scheme creation when no token endpoint can be determined."""
        mock_discover.return_value = {"invalid": "response"}
        
        result = await create_oauth_scheme_from_discovery("https://api.example.com")
        
        # Verify result is None
        assert result is None 


class TestOAuth2DiscoverySSLVerification:
    """Test suite for SSL verification functionality in OAuth2 discovery."""

    @patch("google.adk.auth.oauth2_discovery_util.httpx.AsyncClient")
    @pytest.mark.asyncio
    async def test_discovery_ssl_verification_enabled(self, mock_async_client):
        """Test OAuth discovery with SSL verification enabled (default)."""
        # Setup mock client
        mock_client = AsyncMock()
        mock_async_client.return_value.__aenter__.return_value = mock_client
        
        # Setup mock response for oauth-protected-resource
        mock_response = Mock()
        mock_response.json.return_value = {
            "authorization_servers": ["https://auth.example.com"]
        }
        mock_response.raise_for_status = Mock()
        mock_client.get.return_value = mock_response
        
        result = await create_oauth_scheme_from_discovery(
            "https://api.example.com",
            scopes=["read"],
            timeout=10.0,
            verify_ssl=True
        )
        
        # Verify AsyncClient was created with SSL verification enabled
        mock_async_client.assert_called_with(timeout=10.0, verify=True)
        
        # Verify discovery was attempted
        mock_client.get.assert_called()

    @patch("google.adk.auth.oauth2_discovery_util.httpx.AsyncClient")
    @pytest.mark.asyncio
    async def test_discovery_ssl_verification_disabled(self, mock_async_client):
        """Test OAuth discovery with SSL verification disabled for self-signed certificates."""
        # Setup mock client
        mock_client = AsyncMock()
        mock_async_client.return_value.__aenter__.return_value = mock_client
        
        # Setup mock response for oauth-protected-resource
        mock_response = Mock()
        mock_response.json.return_value = {
            "authorization_servers": ["https://localhost:9204"]
        }
        mock_response.raise_for_status = Mock()
        mock_client.get.return_value = mock_response
        
        result = await create_oauth_scheme_from_discovery(
            "https://localhost:9204",  # Self-signed SSL scenario
            scopes=["read"],
            timeout=10.0,
            verify_ssl=False  # Disable SSL verification
        )
        
        # Verify AsyncClient was created with SSL verification disabled
        mock_async_client.assert_called_with(timeout=10.0, verify=False)
        
        # Verify discovery was attempted
        mock_client.get.assert_called()

    @patch("google.adk.auth.oauth2_discovery_util.httpx.AsyncClient")
    @pytest.mark.asyncio
    async def test_discovery_ssl_verification_default_true(self, mock_async_client):
        """Test that SSL verification defaults to True when not specified."""
        # Setup mock client
        mock_client = AsyncMock()
        mock_async_client.return_value.__aenter__.return_value = mock_client
        
        # Setup mock response for oauth-protected-resource
        mock_response = Mock()
        mock_response.json.return_value = {
            "authorization_servers": ["https://auth.example.com"]
        }
        mock_response.raise_for_status = Mock()
        mock_client.get.return_value = mock_response
        
        # Call without verify_ssl parameter - should default to True
        result = await create_oauth_scheme_from_discovery(
            "https://api.example.com",
            scopes=["read"],
            timeout=10.0
        )
        
        # Verify AsyncClient was created with SSL verification enabled by default
        mock_async_client.assert_called_with(timeout=10.0, verify=True)
        
        # Verify discovery was attempted
        mock_client.get.assert_called()

    @patch("google.adk.auth.oauth2_discovery_util.httpx.AsyncClient")
    @pytest.mark.asyncio
    async def test_discovery_two_stage_ssl_verification(self, mock_async_client):
        """Test that SSL verification is applied to both stages of OAuth discovery."""
        # Setup mock client
        mock_client = AsyncMock()
        mock_async_client.return_value.__aenter__.return_value = mock_client
        
        # Setup mock responses for two-stage discovery
        protected_resource_response = Mock()
        protected_resource_response.json.return_value = {
            "authorization_servers": ["https://localhost:9204"]
        }
        protected_resource_response.raise_for_status = Mock()
        
        auth_server_response = Mock()
        auth_server_response.json.return_value = {
            "token_endpoint": "https://localhost:9204/token"
        }
        auth_server_response.raise_for_status = Mock()
        
        # Mock both calls in sequence
        mock_client.get.side_effect = [protected_resource_response, auth_server_response]
        
        result = await create_oauth_scheme_from_discovery(
            "https://localhost:9204",
            scopes=["read"],
            timeout=10.0,
            verify_ssl=False  # Disable SSL verification for self-signed certs
        )
        
        # Verify AsyncClient was created with SSL verification disabled for both stages
        # Should be called twice - once for each stage of discovery
        assert mock_async_client.call_count >= 2
        for call in mock_async_client.call_args_list:
            args, kwargs = call
            assert kwargs.get('verify') is False
        
        # Verify both discovery calls were made
        assert mock_client.get.call_count == 2
        
        # Verify result
        assert isinstance(result, OAuth2)
        assert result.flows.clientCredentials.tokenUrl == "https://localhost:9204/token" 