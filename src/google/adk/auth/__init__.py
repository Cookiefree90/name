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

"""Auth configurations for Google ADK."""

from .auth_credential import AuthCredential
from .auth_credential import AuthCredentialTypes
from .auth_credential import OAuth2Auth
from .auth_credential import ServiceAccount
from .auth_handler import AuthHandler
from .auth_schemes import AuthScheme
from .auth_schemes import OAuthGrantType
from .auth_schemes import OpenIdConnectWithConfig
from .credential_manager import CredentialManager

# OAuth discovery utilities - imported conditionally to avoid circular imports
try:
    from .oauth2_discovery_util import create_oauth_scheme_from_discovery
    _discovery_available = True
except ImportError:
    _discovery_available = False

__all__ = [
    "AuthCredential",
    "AuthCredentialTypes",
    "AuthHandler",
    "AuthScheme",
    "CredentialManager",
    "OAuthGrantType",
    "OAuth2Auth",
    "OpenIdConnectWithConfig", 
    "ServiceAccount",
]

# Add discovery utilities to __all__ if available
if _discovery_available:
    __all__.extend([
        "create_oauth_scheme_from_discovery",
    ])
