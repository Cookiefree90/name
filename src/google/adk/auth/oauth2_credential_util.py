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
from typing import Optional
from typing import Tuple

from fastapi.openapi.models import OAuth2

from ..utils.feature_decorator import experimental
from .auth_credential import AuthCredential
from .auth_schemes import AuthScheme
from .auth_schemes import OpenIdConnectWithConfig

try:
  from authlib.integrations.requests_client import OAuth2Session
  from authlib.oauth2.rfc6749 import OAuth2Token

  AUTHLIB_AVAILABLE = True
except ImportError:
  AUTHLIB_AVAILABLE = False


logger = logging.getLogger("google_adk." + __name__)


@experimental
def create_oauth2_session(
    auth_scheme: AuthScheme,
    auth_credential: AuthCredential,
) -> Tuple[Optional[OAuth2Session], Optional[str]]:
  """Create an OAuth2 session for token operations.

  Args:
      auth_scheme: The authentication scheme configuration.
      auth_credential: The authentication credential.

  Returns:
      Tuple of (OAuth2Session, token_endpoint) or (None, None) if cannot create session.
  """
  if isinstance(auth_scheme, OpenIdConnectWithConfig):
    if not hasattr(auth_scheme, "token_endpoint"):
      return None, None
    token_endpoint = auth_scheme.token_endpoint
    scopes = auth_scheme.scopes or []
  elif isinstance(auth_scheme, OAuth2):
    # Handle client credentials flow
    if auth_scheme.flows.clientCredentials:
      token_endpoint = auth_scheme.flows.clientCredentials.tokenUrl
      scopes = list(auth_scheme.flows.clientCredentials.scopes.keys()) if auth_scheme.flows.clientCredentials.scopes else []
    # Handle authorization code flow  
    elif auth_scheme.flows.authorizationCode:
      token_endpoint = auth_scheme.flows.authorizationCode.tokenUrl
      scopes = list(auth_scheme.flows.authorizationCode.scopes.keys()) if auth_scheme.flows.authorizationCode.scopes else []
    else:
      return None, None
  else:
    return None, None

  if (
      not auth_credential
      or not auth_credential.oauth2
      or not auth_credential.oauth2.client_id
      or not auth_credential.oauth2.client_secret
  ):
    return None, None

  # For client credentials flow, we don't need redirect_uri or state
  if isinstance(auth_scheme, OAuth2) and auth_scheme.flows.clientCredentials:
    return (
        OAuth2Session(
            auth_credential.oauth2.client_id,
            auth_credential.oauth2.client_secret,
            scope=" ".join(scopes),
        ),
        token_endpoint,
    )
  else:
    # For authorization code flow, include redirect_uri and state
    return (
        OAuth2Session(
            auth_credential.oauth2.client_id,
            auth_credential.oauth2.client_secret,
            scope=" ".join(scopes),
            redirect_uri=auth_credential.oauth2.redirect_uri,
            state=auth_credential.oauth2.state,
        ),
        token_endpoint,
    )


@experimental
def update_credential_with_tokens(
    auth_credential: AuthCredential, tokens: OAuth2Token
) -> None:
  """Update the credential with new tokens.

  Args:
      auth_credential: The authentication credential to update.
      tokens: The OAuth2Token object containing new token information.
  """
  if not auth_credential.oauth2:
    return
    
  # Cast token values to appropriate types
  access_token = tokens.get("access_token")
  auth_credential.oauth2.access_token = str(access_token) if access_token else None
  
  refresh_token = tokens.get("refresh_token")
  auth_credential.oauth2.refresh_token = str(refresh_token) if refresh_token else None
  
  expires_at = tokens.get("expires_at")
  try:
    auth_credential.oauth2.expires_at = int(expires_at) if expires_at is not None else None
  except (ValueError, TypeError):
    auth_credential.oauth2.expires_at = None
  
  expires_in = tokens.get("expires_in")
  try:
    auth_credential.oauth2.expires_in = int(expires_in) if expires_in is not None else None
  except (ValueError, TypeError):
    auth_credential.oauth2.expires_in = None
