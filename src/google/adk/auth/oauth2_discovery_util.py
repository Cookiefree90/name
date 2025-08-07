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

"""OAuth2 discovery utilities for automatic configuration discovery."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

import httpx
from fastapi.openapi.models import OAuth2
from fastapi.openapi.models import OAuthFlowClientCredentials
from fastapi.openapi.models import OAuthFlows
from google.adk.utils.feature_decorator import experimental

logger = logging.getLogger(__name__)

# OAuth Discovery Constants
OAUTH_PROTECTED_RESOURCE_DISCOVERY = ".well-known/oauth-protected-resource"
OAUTH_AUTHORIZATION_SERVER_DISCOVERY = ".well-known/oauth-authorization-server"


@experimental
def _validate_oauth_discovery_response(config: Dict[str, Any]) -> bool:
  """
  Validate OAuth discovery response contains required fields.
  
  Args:
      config: The discovered OAuth configuration
      
  Returns:
      True if configuration is valid, False otherwise
  """
  # For oauth-protected-resource discovery
  if "authorization_servers" in config:
    return isinstance(config["authorization_servers"], list) and bool(config["authorization_servers"])
  
  # For oauth-authorization-server discovery  
  if "token_endpoint" in config:
    return isinstance(config["token_endpoint"], str) and bool(config["token_endpoint"])
    
  return False


@experimental
async def create_oauth_scheme_from_discovery(
    base_url: str,
    scopes: Optional[list[str]] = None,
    timeout: float = 10.0,
    verify_ssl: bool = True
) -> Optional[OAuth2]:
  """
  Create an OAuth2 auth scheme by automatically discovering OAuth configuration.
  
  Implements RFC 8414 two-stage discovery:
  1. Query .well-known/oauth-protected-resource to find authorization server
  2. Query authorization server's .well-known/oauth-authorization-server for token endpoint
  
  Args:
      base_url: The base URL to discover OAuth configuration for
      scopes: List of OAuth scopes to request
      timeout: Discovery request timeout in seconds
      verify_ssl: Whether to verify SSL certificates (default: True).
          Set to False for self-signed certificates.
      
  Returns:
      OAuth2 auth scheme with discovered configuration, or None if discovery fails
  """
  # Stage 1: Try to find authorization server from protected resource endpoint
  protected_resource_config = await _query_oauth_endpoint(
      base_url, OAUTH_PROTECTED_RESOURCE_DISCOVERY, timeout, verify_ssl
  )
  
  token_endpoint = None
  
  if protected_resource_config and "authorization_servers" in protected_resource_config:
    # Stage 2: Query the authorization server's oauth-authorization-server endpoint
    auth_servers = protected_resource_config["authorization_servers"]
    if auth_servers:
      auth_server_url = auth_servers[0]
      logger.debug(f"Found authorization server: {auth_server_url}")
      
      # Specifically query the authorization server's oauth-authorization-server endpoint
      auth_server_config = await _query_oauth_endpoint(
          auth_server_url, OAUTH_AUTHORIZATION_SERVER_DISCOVERY, timeout, verify_ssl
      )
      
      if auth_server_config and "token_endpoint" in auth_server_config:
        token_endpoint = auth_server_config["token_endpoint"]
        logger.debug(f"Discovered token endpoint: {token_endpoint}")
      else:
        logger.warning(f"Authorization server {auth_server_url} did not provide token_endpoint")
        # Fallback: assume standard /token endpoint
        token_endpoint = f"{auth_server_url.rstrip('/')}/token"
        logger.debug(f"Using fallback token endpoint: {token_endpoint}")
  else:
    # Fallback: Try direct authorization server discovery at base URL
    logger.debug(f"No oauth-protected-resource found, trying direct authorization server discovery at {base_url}")
    auth_server_config = await _query_oauth_endpoint(
        base_url, OAUTH_AUTHORIZATION_SERVER_DISCOVERY, timeout, verify_ssl
    )
    
    if auth_server_config and "token_endpoint" in auth_server_config:
      token_endpoint = auth_server_config["token_endpoint"]
      logger.debug(f"Discovered token endpoint via direct discovery: {token_endpoint}")
  
  if not token_endpoint:
    logger.warning("Could not determine token endpoint from OAuth discovery")
    return None
    
  # Create scopes dictionary
  scopes = scopes or ["read", "write"]
  scopes_dict = {scope: f"Access to {scope}" for scope in scopes}
  
  # Create OAuth2 scheme with client credentials flow
  return OAuth2(
    flows=OAuthFlows(
      clientCredentials=OAuthFlowClientCredentials(
        tokenUrl=token_endpoint,
        scopes=scopes_dict
      )
    )
  )


@experimental
async def _query_oauth_endpoint(
    base_url: str,
    endpoint_path: str, 
    timeout: float,
    verify_ssl: bool = True
) -> Optional[Dict[str, Any]]:
  """
  Query a specific OAuth discovery endpoint.
  
  Args:
      base_url: The base URL of the server
      endpoint_path: The discovery endpoint path (e.g., ".well-known/oauth-protected-resource")
      timeout: Request timeout in seconds
      verify_ssl: Whether to verify SSL certificates (default: True).
          Set to False for self-signed certificates.
      
  Returns:
      Dictionary containing the discovery response, or None if failed
  """
  discovery_url = f"{base_url.rstrip('/')}/{endpoint_path}"
  
  async with httpx.AsyncClient(timeout=timeout, verify=verify_ssl) as client:
    try:
      logger.debug(f"Querying OAuth endpoint: {discovery_url} (verify_ssl={verify_ssl})")
      
      response = await client.get(discovery_url)
      response.raise_for_status()
      
      config = response.json()
      logger.debug(f"Successfully got response from {discovery_url}")
      
      # Validate response has expected structure
      if _validate_oauth_discovery_response(config):
        return config
      else:
        logger.warning(f"Invalid OAuth discovery response from {discovery_url}")
        return None
        
    except httpx.HTTPStatusError as e:
      logger.debug(f"OAuth endpoint {discovery_url} returned HTTP {e.response.status_code}")
      return None
    except (httpx.RequestError, json.JSONDecodeError) as e:
      logger.debug(f"Failed to query OAuth endpoint {discovery_url}: {e}")
      return None
    except Exception as e:
      logger.warning(f"Unexpected error querying OAuth endpoint {discovery_url}: {e}")
      return None 