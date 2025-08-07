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

from enum import Enum
from typing import List
from typing import Optional
from typing import Union

from fastapi.openapi.models import OAuthFlows
from fastapi.openapi.models import SecurityBase
from fastapi.openapi.models import SecurityScheme
from fastapi.openapi.models import SecuritySchemeType
from pydantic import Field


class OpenIdConnectWithConfig(SecurityBase):
  type_: SecuritySchemeType = Field(
      default=SecuritySchemeType.openIdConnect, alias="type"
  )
  authorization_endpoint: str
  token_endpoint: str
  userinfo_endpoint: Optional[str] = None
  revocation_endpoint: Optional[str] = None
  token_endpoint_auth_methods_supported: Optional[List[str]] = None
  grant_types_supported: Optional[List[str]] = None
  scopes: Optional[List[str]] = None


# AuthSchemes contains SecuritySchemes from OpenAPI 3.0 and an extra flattened OpenIdConnectWithConfig.
AuthScheme = Union[SecurityScheme, OpenIdConnectWithConfig]


class OAuthGrantType(str, Enum):
  """Represents the OAuth2 flow (or grant type)."""

  CLIENT_CREDENTIALS = "client_credentials"
  AUTHORIZATION_CODE = "authorization_code"
  IMPLICIT = "implicit"
  PASSWORD = "password"

  @staticmethod
  def from_flow(flow: OAuthFlows) -> Optional["OAuthGrantType"]:
    """Converts an OAuthFlows object to a OAuthGrantType.
    
    Determines the OAuth2 grant type based on which flow is configured
    in the OAuthFlows object. Prioritizes client credentials as it's the
    most specific flow for machine-to-machine authentication.
    
    Args:
        flow: The OAuthFlows object containing flow configurations.
        
    Returns:
        The corresponding OAuthGrantType, or None if no recognized flow is found.
    """
    # Prioritize client credentials for machine-to-machine authentication
    if flow.clientCredentials:
      return OAuthGrantType.CLIENT_CREDENTIALS
    # Authorization code flow for interactive user authentication
    if flow.authorizationCode:
      return OAuthGrantType.AUTHORIZATION_CODE
    # Implicit flow (less secure, deprecated in OAuth 2.1)
    if flow.implicit:
      return OAuthGrantType.IMPLICIT
    # Password flow (not recommended for security reasons)
    if flow.password:
      return OAuthGrantType.PASSWORD
    # No recognized flow found
    return None


# AuthSchemeType re-exports SecuritySchemeType from OpenAPI 3.0.
AuthSchemeType = SecuritySchemeType
