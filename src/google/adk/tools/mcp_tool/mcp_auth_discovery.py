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

"""OAuth discovery configuration for MCP tools."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class MCPAuthDiscovery:
    """Configuration for OAuth2 discovery in MCP tools.
    
    This class encapsulates parameters needed for automatic OAuth2 discovery,
    providing a clean API for configuring how MCPToolset should discover 
    OAuth2 token endpoints.
    
    Attributes:
        base_url: The base server URL for OAuth discovery endpoints (e.g., "http://server:9204").
            If None, MCPToolset will automatically extract the base URL from connection parameters.
            OAuth .well-known endpoints will be queried at this URL root.
        timeout: Timeout in seconds for OAuth discovery requests (default: 10.0).
        enabled: Whether OAuth discovery is enabled (default: True).
        verify_ssl: Whether to verify SSL certificates during discovery (default: True).
            Set to False for self-signed certificates or development environments.
        
    Note:
        OAuth scopes should be specified in the auth_scheme parameter of MCPToolset,
        not in this discovery configuration. Discovery only finds token endpoints.
            
    Examples:
        >>> # Override just SSL verification (base_url auto-extracted)
        >>> discovery = MCPAuthDiscovery(verify_ssl=False)
        >>> toolset = MCPToolset(
        ...     connection_params=StreamableHTTPConnectionParams(url="https://localhost:9204/mcp/"),
        ...     auth_credential=credential,
        ...     auth_discovery=discovery  # base_url will be auto-extracted as "https://localhost:9204"
        ... )
        
        >>> # Override multiple settings
        >>> discovery = MCPAuthDiscovery(
        ...     verify_ssl=False,
        ...     timeout=15.0
        ... )
        
        >>> # Explicit base_url (override auto-extraction)
        >>> discovery = MCPAuthDiscovery(
        ...     base_url="https://auth-server.example.com",
        ...     verify_ssl=False
        ... )
        
        >>> # For production with valid SSL certificates
        >>> discovery = MCPAuthDiscovery(
        ...     base_url="https://api.example.com"
        ... )
        
        >>> # For development with self-signed certificates
        >>> discovery = MCPAuthDiscovery(
        ...     base_url="https://localhost:9204",
        ...     verify_ssl=False  # Disable SSL verification
        ... )
        
        >>> # Scopes go in auth scheme, not discovery config
        >>> auth_scheme = OAuth2(
        ...     flows=OAuthFlows(
        ...         clientCredentials=OAuthFlowClientCredentials(
        ...             tokenUrl="",  # Will be discovered
        ...             scopes={"read": "Read access", "write": "Write access"}
        ...         )
        ...     )
        ... )
        >>> toolset = MCPToolset(
        ...     connection_params=StreamableHTTPConnectionParams(url="https://localhost:9204/mcp/"),
        ...     auth_scheme=auth_scheme,
        ...     auth_credential=credential,
        ...     auth_discovery=discovery
        ... )
    """
    
    base_url: Optional[str] = None
    timeout: float = 10.0
    enabled: bool = True
    verify_ssl: bool = True
    
    def __post_init__(self):
        """Validate and normalize configuration after initialization."""
        # Normalize base URL - remove trailing slash (if provided)
        if self.base_url:
            self.base_url = self.base_url.rstrip('/')
            
        # Validate timeout
        if self.timeout <= 0:
            raise ValueError("Discovery timeout must be positive")
    
    @property
    def is_enabled(self) -> bool:
        """Check if OAuth discovery is enabled and properly configured.
        
        Returns True if discovery is enabled. Note that base_url can be None
        since MCPToolset will auto-extract it from connection parameters.
        """
        return self.enabled 
