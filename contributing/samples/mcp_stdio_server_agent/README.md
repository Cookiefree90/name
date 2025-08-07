# MCP Server Agent Sample

This sample demonstrates how to use MCPToolset in ADK, including both basic file system access and OAuth2 client credentials authentication for external APIs.

## Features

### 1. Basic MCP Integration (Stdio)
- File system access using the filesystem MCP server
- Read-only operations (configured with tool filters)
- Local directory access

### 2. OAuth2 Client Credentials Authentication (New!)
- Machine-to-machine authentication for external APIs
- Automatic token exchange and refresh
- Secure access to protected MCP servers

## Setup

### Basic File System Access

The basic example works out of the box:

```bash
# Install dependencies
npm install -g @modelcontextprotocol/server-filesystem

# Run the agent
python agent.py
```

### OAuth2 Client Credentials Setup

To use the OAuth2 client credentials functionality:

1. **Get OAuth2 Credentials**: Obtain client ID and secret from your OAuth provider
2. **Configure Environment Variables**:
   ```bash
   export OAUTH_CLIENT_ID="your_client_id_here"
   export OAUTH_CLIENT_SECRET="your_client_secret_here"
   ```

3. **Update Configuration**: Uncomment and configure the OAuth2 section in `agent.py`:

```python
# Configuration - replace with your actual values
CLIENT_ID = os.getenv("OAUTH_CLIENT_ID", "your_client_id_here")
CLIENT_SECRET = os.getenv("OAUTH_CLIENT_SECRET", "your_client_secret_here")
TOKEN_URL = "https://your-oauth-provider.com/token"
MCP_SERVER_URL = "https://your-mcp-server.com"
SCOPES = ["read", "write"]

# Define OAuth2 client credentials scheme
auth_scheme = OAuth2(
    flows=OAuthFlows(
        clientCredentials=OAuthFlowClientCredentials(
            tokenUrl=TOKEN_URL,
            scopes={
                "read": "Read access to resources",
                "write": "Write access to resources"
            }
        )
    )
)

# Provide client credentials (ADK will automatically handle token exchange)
auth_credential = AuthCredential(
    auth_type=AuthCredentialTypes.OAUTH2,
    oauth2=OAuth2Auth(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET
    )
)

# Create MCPToolset with OAuth2 client credentials authentication
return MCPToolset(
    connection_params=StreamableHTTPConnectionParams(
        url=MCP_SERVER_URL,
        timeout=30
    ),
    auth_scheme=auth_scheme,
    auth_credential=auth_credential
)
```

## OAuth2 Client Credentials Flow

### How It Works

1. **No User Interaction**: Client credentials flow is designed for machine-to-machine authentication
2. **Automatic Token Exchange**: ADK automatically exchanges your client ID/secret for access tokens
3. **Token Refresh**: ADK handles token expiration and refresh automatically
4. **Secure Headers**: Access tokens are automatically added to MCP server requests

### Flow Diagram

```
Client Application
       ↓
   ADK Framework
       ↓ (client_id, client_secret)
   OAuth2 Provider
       ↓ (access_token)
   MCP Server
       ↓ (API response)
   Client Application
```

### Key Advantages

- **Secure**: No user credentials stored, only OAuth2 tokens
- **Automatic**: No manual token management required
- **Standards Compliant**: Uses standard OAuth2 client credentials flow
- **Flexible**: Works with any OAuth2-compliant provider

## Usage Examples

### Basic File Operations

```python
# The agent can access local files
agent.run("List the files in the current directory")
agent.run("Read the content of README.md")
```

### OAuth2-Protected API Access

```python
# Once OAuth2 is configured, the agent can access protected APIs
agent.run("Get my user profile from the API")
agent.run("List my resources")
agent.run("Create a new item")
```

## Supported OAuth2 Providers

This implementation works with any OAuth2 provider that supports client credentials flow:

- **Google Cloud Platform**: For accessing Google APIs
- **Microsoft Azure**: For accessing Microsoft Graph and other APIs
- **Auth0**: For custom applications
- **Okta**: For enterprise applications
- **Custom OAuth2 Servers**: Any RFC 6749 compliant server

## Security Best Practices

1. **Environment Variables**: Store client secrets in environment variables, not code
2. **Least Privilege**: Request only the scopes you actually need
3. **Token Rotation**: Let ADK handle automatic token refresh
4. **HTTPS Only**: Always use HTTPS for token endpoints and MCP servers
5. **Secret Management**: Use proper secret management systems in production

## Troubleshooting

### Common Issues

1. **Invalid Client Credentials**
   - Verify your client ID and secret
   - Check that the client is authorized for the specified scopes

2. **Token Exchange Failures**
   - Verify the token URL is correct
   - Check network connectivity to the OAuth provider

3. **MCP Server Authentication Errors**
   - Ensure the MCP server expects Bearer token authentication
   - Verify the server accepts your access token format

### Debug Logging

Enable debug logging to see OAuth2 flow details:

```python
import logging
logging.getLogger("google_adk").setLevel(logging.DEBUG)
```

## Advanced Configuration

### Custom Scopes

```python
auth_scheme = OAuth2(
    flows=OAuthFlows(
        clientCredentials=OAuthFlowClientCredentials(
            tokenUrl=TOKEN_URL,
            scopes={
                "read:users": "Read user information",
                "write:data": "Write application data",
                "admin:settings": "Manage application settings"
            }
        )
    )
)
```

### Multiple MCP Servers

```python
# Create multiple toolsets with different authentication
public_mcp_toolset = MCPToolset(connection_params=public_params)
authenticated_mcp_toolset = MCPToolset(
    connection_params=auth_params,
    auth_scheme=auth_scheme,
    auth_credential=auth_credential
)

# Use both in the same agent
agent = LlmAgent(
    tools=[public_mcp_toolset, authenticated_mcp_toolset]
)
```

## Migration from Manual Token Management

If you were previously managing OAuth2 tokens manually:

### Before (Manual)
```python
# Manual token management (error-prone)
access_token = get_access_token_manually(client_id, client_secret)
headers = {"Authorization": f"Bearer {access_token}"}
# Need to handle refresh manually
```

### After (ADK Automatic)
```python
# ADK handles everything automatically
mcp_toolset = MCPToolset(
    connection_params=connection_params,
    auth_scheme=oauth2_scheme,
    auth_credential=oauth2_credential
)
# Tokens are managed automatically
```

## Contributing

Found an issue or want to improve OAuth2 support? Please:

1. Check existing issues in the ADK repository
2. Create a detailed bug report or feature request
3. Include your OAuth2 provider details (if relevant)
4. Provide minimal reproduction examples 