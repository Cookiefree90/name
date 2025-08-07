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

"""Tests for MCP Auth Discovery configuration."""

import pytest
import sys

# Skip all tests in this module if Python version is less than 3.10
pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 10), reason="MCP tool requires Python 3.10+"
)

try:
    from google.adk.tools.mcp_tool.mcp_auth_discovery import MCPAuthDiscovery
except ImportError:
    if sys.version_info >= (3, 10):
        raise
    # Create dummy class for older Python versions
    class MCPAuthDiscovery:
        pass


class TestMCPAuthDiscovery:
    """Test suite for MCPAuthDiscovery configuration class."""

    def test_basic_initialization(self):
        """Test basic MCPAuthDiscovery initialization with required parameters."""
        discovery = MCPAuthDiscovery(
            base_url="http://localhost:9204",
            timeout=10.0,
            enabled=True
        )
        
        assert discovery.base_url == "http://localhost:9204"
        assert discovery.timeout == 10.0
        assert discovery.enabled is True
        assert discovery.is_enabled is True

    def test_mcp_auth_discovery_defaults(self):
        """Test MCPAuthDiscovery with default values."""
        discovery = MCPAuthDiscovery(base_url="http://localhost:9204")
        
        assert discovery.base_url == "http://localhost:9204"
        assert discovery.timeout == 10.0
        assert discovery.enabled is True
        assert discovery.verify_ssl is True
        assert discovery.is_enabled is True


    def test_mcp_auth_discovery_custom_values(self):
        """Test MCPAuthDiscovery with custom values."""
        discovery = MCPAuthDiscovery(
            base_url="https://custom-server:8080/",
            timeout=15.0,
            enabled=False,
            verify_ssl=False
        )
        
        assert discovery.base_url == "https://custom-server:8080"  # Trailing slash removed
        assert discovery.timeout == 15.0
        assert discovery.enabled is False
        assert discovery.verify_ssl is False
        assert discovery.is_enabled is False  # Disabled


    def test_mcp_auth_discovery_self_signed_ssl(self):
        """Test MCPAuthDiscovery configured for self-signed SSL certificates."""
        discovery = MCPAuthDiscovery(
            base_url="https://localhost:9204",
            verify_ssl=False,  # Disable SSL verification for self-signed certs
            timeout=5.0
        )
        
        assert discovery.base_url == "https://localhost:9204"
        assert discovery.verify_ssl is False
        assert discovery.timeout == 5.0
        assert discovery.enabled is True
        assert discovery.is_enabled is True

    def test_url_normalization(self):
        """Test that base URLs are properly normalized (trailing slash removed)."""
        discovery = MCPAuthDiscovery(base_url="http://localhost:9204/")
        
        assert discovery.base_url == "http://localhost:9204"  # Trailing slash removed

    def test_complex_url_normalization(self):
        """Test URL normalization with complex paths."""
        discovery = MCPAuthDiscovery(base_url="http://server.example.com:8080/api/v1/")
        
        assert discovery.base_url == "http://server.example.com:8080/api/v1"

    def test_disabled_discovery(self):
        """Test MCPAuthDiscovery when explicitly disabled."""
        discovery = MCPAuthDiscovery(
            base_url="http://localhost:9204",
            enabled=False
        )
        
        assert discovery.enabled is False
        assert discovery.is_enabled is False

    def test_empty_base_url_validation(self):
        """Test that empty base URL raises ValueError."""
        with pytest.raises(ValueError, match="Discovery base_url is required and cannot be empty"):
            MCPAuthDiscovery(base_url="")

    def test_whitespace_base_url_validation(self):
        """Test that whitespace-only base URL raises ValueError."""
        with pytest.raises(ValueError, match="Discovery base_url is required and cannot be empty"):
            MCPAuthDiscovery(base_url="   ")

    def test_none_base_url_validation(self):
        """Test that None base URL raises ValueError."""
        with pytest.raises(ValueError, match="Discovery base_url is required and cannot be empty"):
            MCPAuthDiscovery(base_url=None)  # type: ignore

    def test_negative_timeout_validation(self):
        """Test that negative timeout raises ValueError."""
        with pytest.raises(ValueError, match="Discovery timeout must be positive"):
            MCPAuthDiscovery(base_url="http://localhost:9204", timeout=-1.0)

    def test_zero_timeout_validation(self):
        """Test that zero timeout raises ValueError."""
        with pytest.raises(ValueError, match="Discovery timeout must be positive"):
            MCPAuthDiscovery(base_url="http://localhost:9204", timeout=0.0)

    def test_is_enabled_property_with_empty_url(self):
        """Test is_enabled property returns False when base_url is effectively empty."""
        # This test verifies the property works correctly even if validation is bypassed
        discovery = MCPAuthDiscovery.__new__(MCPAuthDiscovery)
        discovery.base_url = ""
        discovery.enabled = True
        
        assert discovery.is_enabled is False

    def test_valid_timeout_values(self):
        """Test various valid timeout values."""
        test_timeouts = [0.1, 1.0, 5.0, 30.0, 120.0]
        
        for timeout in test_timeouts:
            discovery = MCPAuthDiscovery(
                base_url="http://localhost:9204",
                timeout=timeout
            )
            assert discovery.timeout == timeout

    def test_https_url(self):
        """Test MCPAuthDiscovery with HTTPS URL."""
        discovery = MCPAuthDiscovery(base_url="https://secure.example.com")
        
        assert discovery.base_url == "https://secure.example.com"
        assert discovery.is_enabled is True

    def test_url_with_port(self):
        """Test MCPAuthDiscovery with URL including port."""
        discovery = MCPAuthDiscovery(base_url="http://localhost:8080")
        
        assert discovery.base_url == "http://localhost:8080"
        assert discovery.is_enabled is True

    def test_dataclass_equality(self):
        """Test that MCPAuthDiscovery instances with same values are equal."""
        discovery1 = MCPAuthDiscovery(
            base_url="http://localhost:9204",
            timeout=10.0,
            enabled=True
        )
        discovery2 = MCPAuthDiscovery(
            base_url="http://localhost:9204",
            timeout=10.0,
            enabled=True
        )
        
        assert discovery1 == discovery2

    def test_dataclass_inequality(self):
        """Test that MCPAuthDiscovery instances with different values are not equal."""
        discovery1 = MCPAuthDiscovery(base_url="http://localhost:9204")
        discovery2 = MCPAuthDiscovery(base_url="http://localhost:9205")
        
        assert discovery1 != discovery2

    def test_string_representation(self):
        """Test string representation of MCPAuthDiscovery."""
        discovery = MCPAuthDiscovery(
            base_url="http://localhost:9204",
            timeout=15.0,
            enabled=True
        )
        
        repr_str = repr(discovery)
        assert "MCPAuthDiscovery" in repr_str
        assert "http://localhost:9204" in repr_str
        assert "15.0" in repr_str
        assert "True" in repr_str 


def test_mcp_auth_discovery_optional_base_url():
    """Test MCPAuthDiscovery with optional base_url (None)."""
    discovery = MCPAuthDiscovery(
        verify_ssl=False,
        timeout=15.0
    )
    
    assert discovery.base_url is None
    assert discovery.verify_ssl is False
    assert discovery.timeout == 15.0
    assert discovery.enabled is True
    assert discovery.is_enabled is True  # Should be enabled even without base_url


def test_mcp_auth_discovery_override_ssl_only():
    """Test MCPAuthDiscovery overriding only SSL verification."""
    discovery = MCPAuthDiscovery(verify_ssl=False)
    
    assert discovery.base_url is None
    assert discovery.verify_ssl is False
    assert discovery.timeout == 10.0  # Default
    assert discovery.enabled is True   # Default
    assert discovery.is_enabled is True


def test_mcp_auth_discovery_multiple_overrides():
    """Test MCPAuthDiscovery overriding multiple settings without base_url."""
    discovery = MCPAuthDiscovery(
        timeout=20.0,
        verify_ssl=False,
        enabled=True
    )
    
    assert discovery.base_url is None
    assert discovery.timeout == 20.0
    assert discovery.verify_ssl is False
    assert discovery.enabled is True
    assert discovery.is_enabled is True 