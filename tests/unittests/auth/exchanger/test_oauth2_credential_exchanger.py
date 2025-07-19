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

import time
from unittest.mock import Mock
from unittest.mock import patch

from authlib.oauth2.rfc6749 import OAuth2Token
from fastapi.openapi.models import OAuth2
from fastapi.openapi.models import OAuthFlowClientCredentials
from fastapi.openapi.models import OAuthFlowAuthorizationCode
from fastapi.openapi.models import OAuthFlows
from google.adk.auth.auth_credential import AuthCredential
from google.adk.auth.auth_credential import AuthCredentialTypes
from google.adk.auth.auth_credential import OAuth2Auth
from google.adk.auth.auth_schemes import OAuthGrantType
from google.adk.auth.auth_schemes import OpenIdConnectWithConfig
from google.adk.auth.exchanger.base_credential_exchanger import CredentialExchangError
from google.adk.auth.exchanger.oauth2_credential_exchanger import OAuth2CredentialExchanger
import pytest


class TestOAuth2CredentialExchanger:
  """Test suite for OAuth2CredentialExchanger."""

  @pytest.mark.asyncio
  async def test_exchange_with_existing_token(self):
    """Test exchange method when access token already exists."""
    scheme = OpenIdConnectWithConfig(
        type_="openIdConnect",
        openId_connect_url=(
            "https://example.com/.well-known/openid_configuration"
        ),
        authorization_endpoint="https://example.com/auth",
        token_endpoint="https://example.com/token",
        scopes=["openid"],
    )
    credential = AuthCredential(
        auth_type=AuthCredentialTypes.OPEN_ID_CONNECT,
        oauth2=OAuth2Auth(
            client_id="test_client_id",
            client_secret="test_client_secret",
            access_token="existing_token",
        ),
    )

    exchanger = OAuth2CredentialExchanger()
    result = await exchanger.exchange(credential, scheme)

    # Should return the same credential since access token already exists
    assert result == credential
    assert result.oauth2.access_token == "existing_token"

  @patch("google.adk.auth.oauth2_credential_util.OAuth2Session")
  @pytest.mark.asyncio
  async def test_exchange_success(self, mock_oauth2_session):
    """Test successful token exchange."""
    # Setup mock
    mock_client = Mock()
    mock_oauth2_session.return_value = mock_client
    mock_tokens = OAuth2Token({
        "access_token": "new_access_token",
        "refresh_token": "new_refresh_token",
        "expires_at": int(time.time()) + 3600,
        "expires_in": 3600,
    })
    mock_client.fetch_token.return_value = mock_tokens

    scheme = OpenIdConnectWithConfig(
        type_="openIdConnect",
        openId_connect_url=(
            "https://example.com/.well-known/openid_configuration"
        ),
        authorization_endpoint="https://example.com/auth",
        token_endpoint="https://example.com/token",
        scopes=["openid"],
    )
    credential = AuthCredential(
        auth_type=AuthCredentialTypes.OPEN_ID_CONNECT,
        oauth2=OAuth2Auth(
            client_id="test_client_id",
            client_secret="test_client_secret",
            auth_response_uri="https://example.com/callback?code=auth_code",
            auth_code="auth_code",
        ),
    )

    exchanger = OAuth2CredentialExchanger()
    result = await exchanger.exchange(credential, scheme)

    # Verify token exchange was successful
    assert result.oauth2.access_token == "new_access_token"
    assert result.oauth2.refresh_token == "new_refresh_token"
    mock_client.fetch_token.assert_called_once()

  @pytest.mark.asyncio
  async def test_exchange_missing_auth_scheme(self):
    """Test exchange with missing auth_scheme raises ValueError."""
    credential = AuthCredential(
        auth_type=AuthCredentialTypes.OPEN_ID_CONNECT,
        oauth2=OAuth2Auth(
            client_id="test_client_id",
            client_secret="test_client_secret",
        ),
    )

    exchanger = OAuth2CredentialExchanger()
    try:
      await exchanger.exchange(credential, None)
      assert False, "Should have raised ValueError"
    except CredentialExchangError as e:
      assert "auth_scheme is required" in str(e)

  @patch("google.adk.auth.oauth2_credential_util.OAuth2Session")
  @pytest.mark.asyncio
  async def test_exchange_no_session(self, mock_oauth2_session):
    """Test exchange when OAuth2Session cannot be created."""
    # Mock to return None for create_oauth2_session
    mock_oauth2_session.return_value = None

    scheme = OpenIdConnectWithConfig(
        type_="openIdConnect",
        openId_connect_url=(
            "https://example.com/.well-known/openid_configuration"
        ),
        authorization_endpoint="https://example.com/auth",
        token_endpoint="https://example.com/token",
        scopes=["openid"],
    )
    credential = AuthCredential(
        auth_type=AuthCredentialTypes.OPEN_ID_CONNECT,
        oauth2=OAuth2Auth(
            client_id="test_client_id",
            # Missing client_secret to trigger session creation failure
        ),
    )

    exchanger = OAuth2CredentialExchanger()
    result = await exchanger.exchange(credential, scheme)

    # Should return original credential when session creation fails
    assert result == credential
    assert result.oauth2.access_token is None

  @patch("google.adk.auth.oauth2_credential_util.OAuth2Session")
  @pytest.mark.asyncio
  async def test_exchange_fetch_token_failure(self, mock_oauth2_session):
    """Test exchange when fetch_token fails."""
    # Setup mock to raise exception during fetch_token
    mock_client = Mock()
    mock_oauth2_session.return_value = mock_client
    mock_client.fetch_token.side_effect = Exception("Token fetch failed")

    scheme = OpenIdConnectWithConfig(
        type_="openIdConnect",
        openId_connect_url=(
            "https://example.com/.well-known/openid_configuration"
        ),
        authorization_endpoint="https://example.com/auth",
        token_endpoint="https://example.com/token",
        scopes=["openid"],
    )
    credential = AuthCredential(
        auth_type=AuthCredentialTypes.OPEN_ID_CONNECT,
        oauth2=OAuth2Auth(
            client_id="test_client_id",
            client_secret="test_client_secret",
            auth_response_uri="https://example.com/callback?code=auth_code",
            auth_code="auth_code",
        ),
    )

    exchanger = OAuth2CredentialExchanger()
    result = await exchanger.exchange(credential, scheme)

    # Should return original credential when fetch_token fails
    assert result == credential
    assert result.oauth2.access_token is None
    mock_client.fetch_token.assert_called_once()

  @pytest.mark.asyncio
  async def test_exchange_authlib_not_available(self):
    """Test exchange when authlib is not available."""
    scheme = OpenIdConnectWithConfig(
        type_="openIdConnect",
        openId_connect_url=(
            "https://example.com/.well-known/openid_configuration"
        ),
        authorization_endpoint="https://example.com/auth",
        token_endpoint="https://example.com/token",
        scopes=["openid"],
    )
    credential = AuthCredential(
        auth_type=AuthCredentialTypes.OPEN_ID_CONNECT,
        oauth2=OAuth2Auth(
            client_id="test_client_id",
            client_secret="test_client_secret",
            auth_response_uri="https://example.com/callback?code=auth_code",
            auth_code="auth_code",
        ),
    )

    exchanger = OAuth2CredentialExchanger()

    # Mock AUTHLIB_AVAILABLE to False
    with patch(
        "google.adk.auth.exchanger.oauth2_credential_exchanger.AUTHLIB_AVAILABLE",
        False,
    ):
      result = await exchanger.exchange(credential, scheme)

    # Should return original credential when authlib is not available
    assert result == credential
    assert result.oauth2.access_token is None


class TestOAuth2CredentialExchangerClientCredentials:
  """Test suite for OAuth2CredentialExchanger client credentials flow."""

  @patch("google.adk.auth.exchanger.oauth2_credential_exchanger.OAuth2Session")
  @pytest.mark.asyncio
  async def test_exchange_client_credentials_success(self, mock_oauth2_session):
    """Test successful client credentials token exchange."""
    from fastapi.openapi.models import OAuth2, OAuthFlows, OAuthFlowClientCredentials
    
    # Setup mock
    mock_client = Mock()
    mock_oauth2_session.return_value = mock_client
    mock_tokens = OAuth2Token({
        "access_token": "client_creds_access_token",
        "expires_at": int(time.time()) + 3600,
        "expires_in": 3600,
    })
    mock_client.fetch_token.return_value = mock_tokens

    # Create OAuth2 scheme with client credentials flow
    scheme = OAuth2(
        flows=OAuthFlows(
            clientCredentials=OAuthFlowClientCredentials(
                tokenUrl="https://example.com/token",
                scopes={"read": "Read access", "write": "Write access"}
            )
        )
    )
    
    credential = AuthCredential(
        auth_type=AuthCredentialTypes.OAUTH2,
        oauth2=OAuth2Auth(
            client_id="test_client_id",
            client_secret="test_client_secret",
        ),
    )

    exchanger = OAuth2CredentialExchanger()
    result = await exchanger.exchange(credential, scheme)

    # Verify token exchange was successful
    assert result.oauth2.access_token == "client_creds_access_token"
    # Client credentials flow doesn't provide refresh tokens
    assert result.oauth2.refresh_token is None or result.oauth2.refresh_token == "None"
    
    # Verify the correct grant type was used
    mock_client.fetch_token.assert_called_once()
    call_args = mock_client.fetch_token.call_args
    assert call_args[1]["grant_type"] == "client_credentials"

  @pytest.mark.asyncio
  async def test_exchange_client_credentials_missing_client_secret(self):
    """Test client credentials exchange with missing client secret."""
    from fastapi.openapi.models import OAuth2, OAuthFlows, OAuthFlowClientCredentials
    
    scheme = OAuth2(
        flows=OAuthFlows(
            clientCredentials=OAuthFlowClientCredentials(
                tokenUrl="https://example.com/token",
                scopes={"read": "Read access"}
            )
        )
    )
    
    credential = AuthCredential(
        auth_type=AuthCredentialTypes.OAUTH2,
        oauth2=OAuth2Auth(
            client_id="test_client_id",
            # Missing client_secret
        ),
    )

    exchanger = OAuth2CredentialExchanger()
    result = await exchanger.exchange(credential, scheme)

    # Should return original credential when client secret is missing
    assert result == credential
    assert result.oauth2.access_token is None

  @patch("google.adk.auth.exchanger.oauth2_credential_exchanger.OAuth2Session")
  @pytest.mark.asyncio
  async def test_exchange_client_credentials_token_fetch_failure(self, mock_oauth2_session):
    """Test client credentials exchange when token fetch fails."""
    from fastapi.openapi.models import OAuth2, OAuthFlows, OAuthFlowClientCredentials
    
    # Setup mock to raise exception during fetch_token
    mock_client = Mock()
    mock_oauth2_session.return_value = mock_client
    mock_client.fetch_token.side_effect = Exception("Token fetch failed")

    scheme = OAuth2(
        flows=OAuthFlows(
            clientCredentials=OAuthFlowClientCredentials(
                tokenUrl="https://example.com/token",
                scopes={"read": "Read access"}
            )
        )
    )
    
    credential = AuthCredential(
        auth_type=AuthCredentialTypes.OAUTH2,
        oauth2=OAuth2Auth(
            client_id="test_client_id",
            client_secret="test_client_secret",
        ),
    )

    exchanger = OAuth2CredentialExchanger()
    result = await exchanger.exchange(credential, scheme)

    # Should return original credential when fetch_token fails
    assert result == credential
    assert result.oauth2.access_token is None
    mock_client.fetch_token.assert_called_once()

  def test_get_grant_type_client_credentials(self):
    """Test grant type detection for client credentials flow."""
    from fastapi.openapi.models import OAuth2, OAuthFlows, OAuthFlowClientCredentials
    
    scheme = OAuth2(
        flows=OAuthFlows(
            clientCredentials=OAuthFlowClientCredentials(
                tokenUrl="https://example.com/token",
                scopes={"read": "Read access"}
            )
        )
    )
    
    exchanger = OAuth2CredentialExchanger()
    grant_type = exchanger._get_grant_type(scheme)
    
    assert grant_type == OAuthGrantType.CLIENT_CREDENTIALS

  def test_get_grant_type_authorization_code(self):
    """Test grant type detection for authorization code flow."""
    from fastapi.openapi.models import OAuth2, OAuthFlows, OAuthFlowAuthorizationCode
    
    scheme = OAuth2(
        flows=OAuthFlows(
            authorizationCode=OAuthFlowAuthorizationCode(
                authorizationUrl="https://example.com/auth",
                tokenUrl="https://example.com/token",
                scopes={"read": "Read access"}
            )
        )
    )
    
    exchanger = OAuth2CredentialExchanger()
    grant_type = exchanger._get_grant_type(scheme)
    
    assert grant_type == OAuthGrantType.AUTHORIZATION_CODE

  def test_get_grant_type_mixed_flows_prioritizes_client_credentials(self):
    """Test that client credentials is prioritized when multiple flows are present."""
    from fastapi.openapi.models import OAuth2, OAuthFlows, OAuthFlowClientCredentials, OAuthFlowAuthorizationCode
    
    # Create scheme with both client credentials and authorization code flows
    scheme = OAuth2(
        flows=OAuthFlows(
            clientCredentials=OAuthFlowClientCredentials(
                tokenUrl="https://example.com/token",
                scopes={"read": "Read access"}
            ),
            authorizationCode=OAuthFlowAuthorizationCode(
                authorizationUrl="https://example.com/auth",
                tokenUrl="https://example.com/token",
                scopes={"read": "Read access"}
            )
        )
    )
    
    exchanger = OAuth2CredentialExchanger()
    grant_type = exchanger._get_grant_type(scheme)
    
    # Should prioritize client credentials
    assert grant_type == OAuthGrantType.CLIENT_CREDENTIALS

  def test_get_grant_type_no_flows(self):
    """Test grant type detection when no flows are configured."""
    from fastapi.openapi.models import OAuth2, OAuthFlows
    
    scheme = OAuth2(flows=OAuthFlows())
    
    exchanger = OAuth2CredentialExchanger()
    grant_type = exchanger._get_grant_type(scheme)
    
    assert grant_type is None

  def test_get_grant_type_non_oauth2_scheme(self):
    """Test grant type detection for non-OAuth2 schemes."""
    from google.adk.auth.auth_schemes import OpenIdConnectWithConfig
    
    scheme = OpenIdConnectWithConfig(
        openId_connect_url="https://example.com/.well-known/openid_configuration",
        authorization_endpoint="https://example.com/auth",
        token_endpoint="https://example.com/token",
        scopes=["openid"],
    )
    
    exchanger = OAuth2CredentialExchanger()
    grant_type = exchanger._get_grant_type(scheme)
    
    assert grant_type is None
