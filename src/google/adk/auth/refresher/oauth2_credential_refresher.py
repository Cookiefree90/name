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

"""OAuth2 credential refresher implementation."""

from __future__ import annotations

import json
import logging
from typing import Optional

from fastapi.openapi.models import OAuth2
from google.adk.auth.auth_credential import AuthCredential
from google.adk.auth.auth_schemes import AuthScheme
from google.adk.auth.auth_schemes import OAuthGrantType
from google.adk.auth.oauth2_credential_util import create_oauth2_session
from google.adk.auth.oauth2_credential_util import update_credential_with_tokens
from google.adk.utils.feature_decorator import experimental
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from typing_extensions import override

from .base_credential_refresher import BaseCredentialRefresher

try:
  from authlib.oauth2.rfc6749 import OAuth2Token

  AUTHLIB_AVAILABLE = True
except ImportError:
  AUTHLIB_AVAILABLE = False

logger = logging.getLogger("google_adk." + __name__)


@experimental
class OAuth2CredentialRefresher(BaseCredentialRefresher):
  """Refreshes OAuth2 credentials including Google OAuth2 JSON credentials."""

  @override
  async def is_refresh_needed(
      self,
      auth_credential: AuthCredential,
      auth_scheme: Optional[AuthScheme] = None,
  ) -> bool:
    """Check if the OAuth2 credential needs to be refreshed.

    Args:
        auth_credential: The OAuth2 credential to check.
        auth_scheme: The OAuth2 authentication scheme (optional for Google OAuth2 JSON).

    Returns:
        True if the credential needs to be refreshed, False otherwise.
    """

    # Handle regular OAuth2 credentials
    if auth_credential.oauth2:
      if not AUTHLIB_AVAILABLE:
        return False

      return OAuth2Token({
          "expires_at": auth_credential.oauth2.expires_at,
          "expires_in": auth_credential.oauth2.expires_in,
      }).is_expired()

    return False

  @override
  async def refresh(
      self,
      auth_credential: AuthCredential,
      auth_scheme: Optional[AuthScheme] = None,
  ) -> AuthCredential:
    """Refresh the OAuth2 credential.
    If refresh failed, return the original credential.

    Args:
        auth_credential: The OAuth2 credential to refresh.
        auth_scheme: The OAuth2 authentication scheme (optional for Google OAuth2 JSON).

    Returns:
        The refreshed credential.

    """

    # Handle regular OAuth2 credentials
    if auth_credential.oauth2 and auth_scheme:
      if not AUTHLIB_AVAILABLE:
        return auth_credential

      if not auth_credential.oauth2:
        return auth_credential

      # Check if token is expired
      if OAuth2Token({
          "expires_at": auth_credential.oauth2.expires_at,
          "expires_in": auth_credential.oauth2.expires_in,
      }).is_expired():
        
        # Determine the OAuth2 flow type
        grant_type = self._get_grant_type(auth_scheme)
        
        if grant_type == OAuthGrantType.CLIENT_CREDENTIALS:
          return await self._refresh_client_credentials(auth_credential, auth_scheme)
        elif grant_type == OAuthGrantType.AUTHORIZATION_CODE:
          return await self._refresh_authorization_code(auth_credential, auth_scheme)
        else:
          logger.warning(f"Unsupported OAuth2 grant type for refresh: {grant_type}")
          return auth_credential

    return auth_credential

  def _get_grant_type(self, auth_scheme: AuthScheme) -> Optional[OAuthGrantType]:
    """Determine the OAuth2 grant type from the auth scheme."""
    if isinstance(auth_scheme, OAuth2) and auth_scheme.flows:
      return OAuthGrantType.from_flow(auth_scheme.flows)
    return None

  async def _refresh_client_credentials(
      self, 
      auth_credential: AuthCredential, 
      auth_scheme: AuthScheme
  ) -> AuthCredential:
    """Refresh client credentials by getting a new token (no refresh tokens in client credentials flow)."""
    
    # For client credentials, "refresh" means getting a new token
    # Import here to avoid circular imports
    from ..exchanger.oauth2_credential_exchanger import OAuth2CredentialExchanger
    
    try:
      exchanger = OAuth2CredentialExchanger()
      # Clear the access token to force re-exchange
      if auth_credential.oauth2:
        auth_credential.oauth2.access_token = None
      return await exchanger.exchange(auth_credential, auth_scheme)
    except Exception as e:
      logger.error("Failed to refresh OAuth2 client credentials: %s", e)
      return auth_credential

  async def _refresh_authorization_code(
      self, 
      auth_credential: AuthCredential, 
      auth_scheme: AuthScheme
  ) -> AuthCredential:
    """Refresh authorization code credentials using refresh token."""
    
    client, token_endpoint = create_oauth2_session(
        auth_scheme, auth_credential
    )
    if not client:
      logger.warning("Could not create OAuth2 session for token refresh")
      return auth_credential

    try:
      tokens = client.refresh_token(
          url=token_endpoint,
          refresh_token=auth_credential.oauth2.refresh_token if auth_credential.oauth2 else None,
      )
      update_credential_with_tokens(auth_credential, tokens)
      logger.debug("Successfully refreshed OAuth2 authorization code tokens")
    except Exception as e:
      # TODO reconsider whether we should raise error when refresh failed.
      logger.error("Failed to refresh OAuth2 authorization code tokens: %s", e)
      # Return original credential on failure
      return auth_credential

    return auth_credential
