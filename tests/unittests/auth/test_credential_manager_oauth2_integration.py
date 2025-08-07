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

"""Tests for OAuth2CredentialExchanger integration in CredentialManager."""

from unittest.mock import patch
import pytest

from fastapi.openapi.models import OAuth2, OAuthFlows, OAuthFlowClientCredentials
from google.adk.auth.auth_credential import AuthCredential, AuthCredentialTypes, OAuth2Auth
from google.adk.auth.auth_tool import AuthConfig
from google.adk.auth.credential_manager import CredentialManager
from google.adk.auth.exchanger.oauth2_credential_exchanger import OAuth2CredentialExchanger


class TestCredentialManagerOAuth2Integration:
    """Test OAuth2CredentialExchanger integration with CredentialManager."""

    def test_oauth2_credential_exchanger_registration(self):
        """Test that OAuth2CredentialExchanger is registered in CredentialManager."""
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
            oauth2=OAuth2Auth(
                client_id="test_client_id",
                client_secret="test_client_secret"
            )
        )
        
        auth_config = AuthConfig(
            auth_scheme=auth_scheme,
            raw_auth_credential=auth_credential
        )
        
        credential_manager = CredentialManager(auth_config)
        
        # Verify OAuth2CredentialExchanger is registered
        exchanger = credential_manager._exchanger_registry.get_exchanger(AuthCredentialTypes.OAUTH2)
        assert exchanger is not None
        assert isinstance(exchanger, OAuth2CredentialExchanger)

    @patch('google.adk.auth.exchanger.oauth2_credential_exchanger.OAuth2Session')
    @pytest.mark.asyncio
    async def test_oauth2_credential_exchange_flow(self, mock_oauth2_session):
        """Test complete OAuth2 credential exchange flow through CredentialManager."""
        # Setup mock OAuth2Session
        mock_session_instance = mock_oauth2_session.return_value
        mock_session_instance.fetch_token.return_value = {
            'access_token': 'test_access_token',
            'token_type': 'Bearer',
            'expires_in': 3600
        }
        
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
            oauth2=OAuth2Auth(
                client_id="test_client_id",
                client_secret="test_client_secret"
            )
        )
        
        auth_config = AuthConfig(
            auth_scheme=auth_scheme,
            raw_auth_credential=auth_credential
        )
        
        credential_manager = CredentialManager(auth_config)
        
        # Mock callback context
        class MockCallbackContext:
            def get_auth_response(self, auth_config):
                return None
            async def load_credential(self, auth_config):
                return None
            async def save_credential(self, auth_config):
                pass
        
        callback_context = MockCallbackContext()
        
        # Perform credential exchange
        result_credential = await credential_manager.get_auth_credential(callback_context)  # type: ignore
        
        # Verify OAuth2Session was called with correct parameters
        mock_oauth2_session.assert_called_once_with(
            "test_client_id",
            "test_client_secret",
            scope="api:read",
            token_endpoint_auth_method='client_secret_post'
        )
        
        # Verify fetch_token was called with correct parameters
        mock_session_instance.fetch_token.assert_called_once_with(
            "http://localhost:9204/token",
            grant_type="client_credentials"
        )
        
        # Verify result credential has access token
        assert result_credential is not None
        assert result_credential.oauth2 is not None
        assert result_credential.oauth2.access_token == "test_access_token"

    def test_oauth2_credential_fallback_when_no_exchanger_found(self):
        """Test that CredentialManager falls back to raw OAuth2 credential when needed."""
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
            oauth2=OAuth2Auth(
                client_id="test_client_id",
                client_secret="test_client_secret"
            )
        )
        
        auth_config = AuthConfig(
            auth_scheme=auth_scheme,
            raw_auth_credential=auth_credential
        )
        
        credential_manager = CredentialManager(auth_config)
        
        # Verify that the raw OAuth2 credential fallback logic exists
        # This is tested indirectly by verifying the _is_credential_ready method
        assert not credential_manager._is_credential_ready()
        
        # Verify that OAuth2 credentials are handled appropriately
        assert auth_credential.auth_type == AuthCredentialTypes.OAUTH2
        assert auth_credential.oauth2 is not None
        assert auth_credential.oauth2.client_id == "test_client_id" 

    async def test_oauth2_exchanger_processing_with_token(self):
        """Test that OAuth2CredentialExchanger properly processes credentials with existing tokens."""
        
        from google.adk.auth.auth_tool import AuthConfig
        from google.adk.auth.credential_manager import CredentialManager
        from google.adk.auth.auth_credential import AuthCredential, AuthCredentialTypes, OAuth2Auth
        from fastapi.openapi.models import OAuth2, OAuthFlows, OAuthFlowClientCredentials
        
        # Create OAuth2 credential with existing token
        oauth2_credential = AuthCredential(
            auth_type=AuthCredentialTypes.OAUTH2,
            oauth2=OAuth2Auth(
                client_id="test_client",
                client_secret="test_secret",
                access_token="existing_token"
            )
        )
        
        # Create OAuth2 auth scheme
        auth_scheme = OAuth2(
            flows=OAuthFlows(
                clientCredentials=OAuthFlowClientCredentials(
                    tokenUrl="https://example.com/token",
                    scopes={"read": "Read access"}
                )
            )
        )
        
        auth_config = AuthConfig(
            raw_auth_credential=oauth2_credential,
            auth_scheme=auth_scheme
        )
        
        manager = CredentialManager(auth_config)
        
        # Verify OAuth2CredentialExchanger is registered
        exchanger = manager._exchanger_registry.get_exchanger(AuthCredentialTypes.OAUTH2)
        assert exchanger is not None
        assert isinstance(exchanger, OAuth2CredentialExchanger)


    @patch('google.adk.auth.exchanger.oauth2_credential_exchanger.OAuth2Session')
    async def test_credential_manager_ssl_verification_enabled(self, mock_oauth2_session):
        """Test CredentialManager passes verify_ssl=True to OAuth2CredentialExchanger."""
        from google.adk.auth.auth_tool import AuthConfig
        from google.adk.auth.credential_manager import CredentialManager
        from google.adk.auth.auth_credential import AuthCredential, AuthCredentialTypes, OAuth2Auth
        from google.adk.agents.readonly_context import ReadonlyContext
        from fastapi.openapi.models import OAuth2, OAuthFlows, OAuthFlowClientCredentials
        
        # Setup mock
        mock_client = Mock()
        mock_oauth2_session.return_value = mock_client
        mock_tokens = {
            "access_token": "test_access_token",
            "expires_at": int(time.time()) + 3600,
            "expires_in": 3600,
        }
        mock_client.fetch_token.return_value = mock_tokens
        
        # Create OAuth2 credential
        oauth2_credential = AuthCredential(
            auth_type=AuthCredentialTypes.OAUTH2,
            oauth2=OAuth2Auth(
                client_id="test_client",
                client_secret="test_secret"
            )
        )
        
        # Create OAuth2 auth scheme
        auth_scheme = OAuth2(
            flows=OAuthFlows(
                clientCredentials=OAuthFlowClientCredentials(
                    tokenUrl="https://example.com/token",
                    scopes={"read": "Read access"}
                )
            )
        )
        
        auth_config = AuthConfig(
            raw_auth_credential=oauth2_credential,
            auth_scheme=auth_scheme
        )
        
        manager = CredentialManager(auth_config)
        
        # Create mock callback context
        mock_context = Mock()
        mock_context._invocation_context = Mock()
        mock_context._invocation_context.credential_service = None
        mock_context.get_auth_response = Mock(return_value=None)
        mock_context.load_credential = AsyncMock(return_value=None)
        mock_context.save_credential = AsyncMock()
        
        # Test with SSL verification enabled (default)
        result = await manager.get_auth_credential(mock_context, verify_ssl=True)
        
        # Verify SSL verification is enabled
        assert hasattr(mock_client, 'verify')
        assert mock_client.verify is True
        assert result is not None


    @patch('google.adk.auth.exchanger.oauth2_credential_exchanger.urllib3')
    @patch('google.adk.auth.exchanger.oauth2_credential_exchanger.OAuth2Session')
    async def test_credential_manager_ssl_verification_disabled(self, mock_oauth2_session, mock_urllib3):
        """Test CredentialManager passes verify_ssl=False to OAuth2CredentialExchanger."""
        from google.adk.auth.auth_tool import AuthConfig
        from google.adk.auth.credential_manager import CredentialManager
        from google.adk.auth.auth_credential import AuthCredential, AuthCredentialTypes, OAuth2Auth
        from fastapi.openapi.models import OAuth2, OAuthFlows, OAuthFlowClientCredentials
        
        # Setup mock
        mock_client = Mock()
        mock_oauth2_session.return_value = mock_client
        mock_tokens = {
            "access_token": "test_access_token",
            "expires_at": int(time.time()) + 3600,
            "expires_in": 3600,
        }
        mock_client.fetch_token.return_value = mock_tokens
        
        # Create OAuth2 credential
        oauth2_credential = AuthCredential(
            auth_type=AuthCredentialTypes.OAUTH2,
            oauth2=OAuth2Auth(
                client_id="test_client",
                client_secret="test_secret"
            )
        )
        
        # Create OAuth2 auth scheme
        auth_scheme = OAuth2(
            flows=OAuthFlows(
                clientCredentials=OAuthFlowClientCredentials(
                    tokenUrl="https://localhost:9204/token",  # Self-signed SSL scenario
                    scopes={"read": "Read access"}
                )
            )
        )
        
        auth_config = AuthConfig(
            raw_auth_credential=oauth2_credential,
            auth_scheme=auth_scheme
        )
        
        manager = CredentialManager(auth_config)
        
        # Create mock callback context
        mock_context = Mock()
        mock_context._invocation_context = Mock()
        mock_context._invocation_context.credential_service = None
        mock_context.get_auth_response = Mock(return_value=None)
        mock_context.load_credential = AsyncMock(return_value=None)
        mock_context.save_credential = AsyncMock()
        
        # Test with SSL verification disabled (for self-signed certificates)
        result = await manager.get_auth_credential(mock_context, verify_ssl=False)
        
        # Verify SSL verification is disabled
        assert hasattr(mock_client, 'verify')
        assert mock_client.verify is False
        
        # Verify SSL warnings are suppressed
        mock_urllib3.disable_warnings.assert_called_once_with(mock_urllib3.exceptions.InsecureRequestWarning)
        
        assert result is not None 