# OAuth2 Client Credentials Flow with Automatic Discovery Sample

## Overview

This sample demonstrates the **OAuth2 client credentials authentication flow** with **automatic OAuth discovery** for MCP (Model Context Protocol) servers. It showcases enterprise-grade authentication that "just works" out of the box while providing flexibility for custom configurations.

## Key Features

- 🚀 **Automatic OAuth Discovery**: Zero-configuration OAuth2 setup for HTTP-based MCP connections
- 🔧 **RFC 8414 Compliance**: Two-stage OAuth discovery following industry standards
- 🔐 **Complete Client Credentials Flow**: Full token exchange using the authlib library
- 📝 **Production-Ready**: Appropriate logging and error handling
- 🎯 **Multiple Scenarios**: From simple automatic setup to advanced custom configurations

## How It Works

The ADK MCPToolset automatically:

1. **Extracts base URL** from MCP connection parameters (e.g., `http://localhost:9204/mcp/` → `http://localhost:9204`)
2. **Enables OAuth discovery** for HTTP-based connections (StreamableHTTP, SSE)
3. **Discovers OAuth endpoints** via RFC 8414 two-stage process:
   - Query `.well-known/oauth-protected-resource` to find authorization server
   - Query authorization server's `.well-known/oauth-authorization-server` for token endpoint
4. **Exchanges client credentials** for access tokens using the discovered token endpoint
5. **Authenticates all MCP requests** with the obtained access tokens

## Sample Scenarios

### 1. Automatic OAuth Discovery (Simplest Case)

The most common usage - everything happens automatically:

```python
MCPToolset(
    connection_params=StreamableHTTPConnectionParams(
        url='http://localhost:9204/mcp/',
    ),
    auth_credential=create_oauth2_credential(
        client_id='your_client_id',
        client_secret='your_client_secret'
    ),
    # Optional: Define scopes (will be used during discovery)
    auth_scheme=create_oauth2_scheme(
        token_url="",  # Empty - will be automatically discovered
        scopes={"api:read": "Read access", "api:write": "Write access"}
    ),
    # No auth_discovery parameter needed - automatic discovery enabled!
)
```

### 2. Custom OAuth Discovery Configuration

Override default discovery behavior:

```python
MCPToolset(
    connection_params=StreamableHTTPConnectionParams(
        url='http://localhost:9204/mcp/',
    ),
    auth_credential=create_oauth2_credential(
        client_id='your_client_id',
        client_secret='your_client_secret'
    ),
    # Custom OAuth discovery configuration
    auth_discovery=MCPAuthDiscovery(
        base_url='http://auth-server.example.com:9205',  # Different auth server
        timeout=15.0,  # Custom timeout
        enabled=True
    ),
)
```

### 3. No Auth Scheme (Minimal Configuration)

Let discovery create the complete OAuth scheme:

```python
MCPToolset(
    connection_params=StreamableHTTPConnectionParams(
        url='http://localhost:9204/mcp/',
    ),
    auth_credential=create_oauth2_credential(
        client_id='your_client_id',
        client_secret='your_client_secret'
    ),
    # No auth_scheme - discovery will create one
    # No auth_discovery - automatic discovery enabled
)
```

### 4. Disabled OAuth Discovery (Manual Configuration)

Traditional OAuth2 setup without automatic discovery:

```python
MCPToolset(
    connection_params=StreamableHTTPConnectionParams(
        url='http://localhost:9204/mcp/',
    ),
    auth_credential=create_oauth2_credential(
        client_id='your_client_id',
        client_secret='your_client_secret'
    ),
    # Complete OAuth2 scheme with known token endpoint
    auth_scheme=create_oauth2_scheme(
        token_url="http://localhost:9204/token",  # Known endpoint
        scopes={"api:read": "Read access"}
    ),
    # Explicitly disable OAuth discovery
    auth_discovery=MCPAuthDiscovery(
        base_url="http://localhost:9204",
        enabled=False
    ),
)
```

### 5. Multiple MCP Servers

Access multiple secured MCP servers with different configurations:

```python
tools=[
    # Server 1: Automatic discovery
    MCPToolset(
        connection_params=StreamableHTTPConnectionParams(
            url='http://server1.example.com:9204/mcp/',
        ),
        auth_credential=create_oauth2_credential(
            client_id='server1_client',
            client_secret='server1_secret'
        ),
    ),
    
    # Server 2: Custom discovery server
    MCPToolset(
        connection_params=StreamableHTTPConnectionParams(
            url='http://server2.example.com:8080/api/mcp/',
        ),
        auth_credential=create_oauth2_credential(
            client_id='server2_client',
            client_secret='server2_secret'
        ),
        auth_discovery=MCPAuthDiscovery(
            base_url='http://auth.server2.example.com:9000',
        ),
    ),
]
```

### 6. Self-Signed Certificates (SSL Verification Disabled)

For development environments using self-signed SSL certificates:

```python
MCPToolset(
    connection_params=StreamableHTTPConnectionParams(
        url='https://localhost:9204/mcp/',  # HTTPS with self-signed cert
    ),
    auth_credential=create_oauth2_credential(
        client_id='your_client_id',
        client_secret='your_client_secret'
    ),
    # Override just SSL verification - base_url auto-extracted from connection_params
    auth_discovery=MCPAuthDiscovery(
        verify_ssl=False,  # Only override SSL verification
        # base_url auto-extracted as "https://localhost:9204" from connection_params
    ),
)
```

⚠️ **Security Warning**: Only disable SSL verification in development environments with self-signed certificates. Never disable SSL verification in production!

### 7. Custom Settings Without Base URL Override

Override multiple discovery settings while letting MCPToolset auto-extract the base_url:

```python
MCPToolset(
    connection_params=StreamableHTTPConnectionParams(
        url='http://localhost:9204/mcp/',
    ),
    auth_credential=create_oauth2_credential(
        client_id='your_client_id',
        client_secret='your_client_secret'
    ),
    # Override multiple settings - base_url auto-extracted
    auth_discovery=MCPAuthDiscovery(
        timeout=15.0,      # Custom timeout
        verify_ssl=True,   # Explicit SSL verification (default)
        enabled=True       # Explicit enabled (default)
        # base_url auto-extracted as "http://localhost:9204" from connection_params
    ),
)
```

## Key Benefits

1. **Zero Configuration**: OAuth discovery works out-of-the-box for HTTP connections
2. **Smart Auto-Extraction**: Base URL automatically extracted from MCP connection parameters
3. **Override Only What You Need**: No need to duplicate base_url when overriding other settings
4. **Standards Compliant**: Follows RFC 8414 OAuth2 Authorization Server Metadata
5. **Production Ready**: Appropriate logging levels and comprehensive error handling
6. **Backwards Compatible**: Existing OAuth2 configurations continue to work
7. **Flexible**: From automatic discovery to complete manual control
8. **Secure**: Uses industry-standard OAuth2 client credentials flow

## Prerequisites

### OAuth2 Server Setup

You need an OAuth2 server that supports:

1. **RFC 8414 OAuth discovery endpoints**:
   - `/.well-known/oauth-protected-resource`
   - `/.well-known/oauth-authorization-server`

2. **Client credentials grant type** (`client_credentials`)

3. **client_secret_post** authentication method (credentials in form body)

### Self-Signed SSL Certificates

If you're using self-signed SSL certificates for development:

1. **Generate self-signed certificate**:
   ```bash
   # Generate private key
   openssl genrsa -out server.key 2048
   
   # Generate certificate
   openssl req -new -x509 -key server.key -out server.crt -days 365 \
     -subj "/C=US/ST=CA/L=SF/O=Dev/CN=localhost"
   ```

2. **Configure MCPAuthDiscovery**:
   ```python
   auth_discovery=MCPAuthDiscovery(
       base_url='https://localhost:9204',
       verify_ssl=False,  # Disable for self-signed certs
   )
   ```

3. **Test manually**:
   ```bash
   # Test with curl (skip SSL verification)
   curl -k https://localhost:9204/.well-known/oauth-protected-resource
   ```

### Environment Variables

Set your OAuth2 credentials:

```bash
export OAUTH2_CLIENT_ID="your_client_id"
export OAUTH2_CLIENT_SECRET="your_client_secret"

# For multi-server setup:
export SERVER1_CLIENT_ID="server1_client_id"
export SERVER1_CLIENT_SECRET="server1_client_secret"
export SERVER2_CLIENT_ID="server2_client_id"
export SERVER2_CLIENT_SECRET="server2_client_secret"
```

## Running the Sample

1. **Start the Mock OAuth2 Server** (included in this sample):
   
   **For HTTP (simple testing):**
   ```bash
   python mock_oauth_server.py
   ```
   
   **For HTTPS with self-signed certificates:**
   ```bash
   # First generate self-signed certificates
   openssl genrsa -out server.key 2048
   openssl req -new -x509 -key server.key -out server.crt -days 365 \
     -subj "/C=US/ST=CA/L=SF/O=Dev/CN=localhost"
   
   # Then start server with SSL
   python mock_oauth_server.py --ssl-keyfile server.key --ssl-certfile server.crt
   ```
   
   This starts a test OAuth2 server with:
   - RFC 8414 discovery endpoints
   - Client credentials grant support
   - Demo client credentials (see server output)
   - Optional HTTPS support for testing SSL configurations

2. **Set environment variables** with OAuth2 credentials:
   ```bash
   export OAUTH2_CLIENT_ID="demo_client_id"
   export OAUTH2_CLIENT_SECRET="demo_client_secret"
   ```

3. **Run the agent**:
   ```bash
   python -m google.adk.cli.chatbot --agent-module contributing.samples.mcp_oauth2_client_credentials_agent
   ```

4. **Try different agents** by modifying the `root_agent` variable in `agent.py`:
   - `automatic_discovery_agent` (default)
   - `custom_discovery_agent`
   - `discovery_only_agent`
   - `manual_config_agent`
   - `multi_server_agent`
   - `self_signed_ssl_agent`
   - `custom_settings_agent`

## Testing with the Mock OAuth2 Server

The included `mock_oauth_server.py` provides a complete OAuth2 server for testing:

### Server Features

- **RFC 8414 Discovery Endpoints**:
  - `/.well-known/oauth-protected-resource`
  - `/.well-known/oauth-authorization-server`
- **OAuth2 Token Endpoint**: `/token`
- **Token Validation**: `/validate` (for debugging)
- **Health Check**: `/health`

### Demo Clients

The mock server includes these demo clients:

| Client ID | Client Secret | Scopes |
|-----------|---------------|--------|
| `demo_client_id` | `demo_client_secret` | `api:read`, `api:write` |
| `server1_client` | `server1_secret` | `api:read` |
| `server2_client` | `server2_secret` | `api:read`, `api:write` |

### Testing the OAuth Flow Manually

You can test the OAuth flow manually using curl:

1. **Discover OAuth Configuration**:
   ```bash
   curl http://localhost:9204/.well-known/oauth-protected-resource
   curl http://localhost:9204/.well-known/oauth-authorization-server
   ```

2. **Request Access Token**:
   ```bash
   curl -X POST http://localhost:9204/token \
     -d "grant_type=client_credentials" \
     -d "client_id=demo_client_id" \
     -d "client_secret=demo_client_secret" \
     -d "scope=api:read api:write"
   ```

3. **Validate Token**:
   ```bash
   curl -H "Authorization: Bearer YOUR_TOKEN" \
     http://localhost:9204/validate
   ```

## Debugging OAuth Discovery

Enable debug logging to trace the OAuth discovery process:

```python
import logging
logging.getLogger("google_adk").setLevel(logging.DEBUG)
```

You'll see detailed logs like:
```
DEBUG:google_adk: 🚀 Starting OAuth discovery process
DEBUG:google_adk: 🔍 Attempting OAuth discovery at server root: http://localhost:9204
DEBUG:google_adk: ✅ OAuth discovery successful - updating tokenUrl in existing scheme
DEBUG:google_adk: 🔐 Performing OAuth token exchange for session authentication
DEBUG:google_adk: ✅ Successfully obtained access token for session
```

## Troubleshooting

### SSL Certificate Issues

If you encounter SSL certificate verification errors with self-signed certificates:

```
ssl.SSLCertVerificationError: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: self-signed certificate
```

**Solution**: Set `verify_ssl=False` in your MCPAuthDiscovery configuration:

```python
auth_discovery=MCPAuthDiscovery(
    verify_ssl=False  # Disables SSL verification for self-signed certificates
)
```

### OAuth2Session Constructor Errors

If you encounter OAuth2Session constructor conflicts:

```
authlib.oauth2.client.OAuth2Client.__init__() got multiple values for keyword argument 'session'
```

This issue has been resolved in the latest version. The ADK now properly sets SSL verification on the OAuth2Session object without constructor conflicts.

### Authentication Failures

If OAuth token exchange fails:

1. **Check client credentials**: Ensure your `client_id` and `client_secret` are correct
2. **Verify discovery endpoints**: Confirm your server supports RFC 8414 discovery
3. **Test manually**: Use curl to test OAuth discovery and token exchange
4. **Enable debug logging**: Check detailed OAuth flow logs for specific errors

### Testing Your Setup

You can test the complete OAuth flow manually:

1. **Test Discovery Endpoints**:
   ```bash
   # Test discovery (use -k for self-signed certificates)
   curl -k https://localhost:9204/.well-known/oauth-protected-resource
   curl -k https://localhost:9204/.well-known/oauth-authorization-server
   ```

2. **Test Token Exchange**:
   ```bash
   # Test client credentials flow (use -k for self-signed certificates)
   curl -k -X POST https://localhost:9204/token \
     -d "grant_type=client_credentials" \
     -d "client_id=demo_client_id" \
     -d "client_secret=demo_client_secret" \
     -d "scope=api:read api:write"
   ```

3. **Verify Token**:
   ```bash
   # Test token validation (replace YOUR_TOKEN with actual token)
   curl -k -H "Authorization: Bearer YOUR_TOKEN" \
     https://localhost:9204/validate
   ```

## Benefits

1. **Zero Configuration**: OAuth discovery works out-of-the-box for HTTP connections
2. **Standards Compliant**: Follows RFC 8414 OAuth2 Authorization Server Metadata
3. **Production Ready**: Appropriate logging levels and comprehensive error handling
4. **Backwards Compatible**: Existing OAuth2 configurations continue to work
5. **Flexible**: From automatic discovery to complete manual control
6. **Secure**: Uses industry-standard OAuth2 client credentials flow

## Common Use Cases

- **Enterprise API Integration**: Secure access to internal MCP services
- **Multi-tenant Applications**: Different OAuth configurations per tenant
- **Development vs Production**: Automatic discovery in dev, manual config in prod
- **Microservices Architecture**: Multiple MCP servers with centralized auth
- **Third-party Integrations**: Secure access to external MCP providers

## Next Steps

- Explore the different agent configurations in `agent.py`
- Try connecting to your own OAuth2-enabled MCP server
- Experiment with custom discovery configurations
- Implement your own MCP tools with OAuth2 authentication

This sample demonstrates the power of **convention over configuration** while maintaining full flexibility for advanced use cases. The OAuth2 client credentials flow with automatic discovery makes enterprise authentication simple and reliable! 🚀 