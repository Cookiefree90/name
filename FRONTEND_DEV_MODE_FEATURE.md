# ADK FastAPI Frontend Development Mode

## Overview

The `--frontend-dev-mode` option is a new CLI flag for ADK FastAPI servers that significantly enhances the development experience when building frontend applications that connect to ADK agents. This feature automatically configures common development settings and provides additional debugging capabilities.

## Features

### 1. Automatic CORS Configuration
When `--frontend-dev-mode` is enabled, the server automatically adds CORS origins for common frontend development servers:

- `http://localhost:3000` - React, Next.js default
- `http://localhost:3001` - React alternative port
- `http://localhost:4200` - Angular CLI default
- `http://localhost:5173` - Vite default
- `http://localhost:8080` - Vue CLI default
- Plus `127.0.0.1` variants of all above

These are added in addition to any origins specified via `--allow_origins`.

### 2. Enhanced Error Responses
In frontend dev mode, error responses include detailed debugging information:

```json
{
  "error": "Session not found",
  "type": "HTTPException", 
  "traceback": "Traceback (most recent call last):\n  File \"...\", line 123...",
  "request_info": {
    "method": "GET",
    "url": "http://localhost:8000/apps/my-app/users/user123/sessions/session456",
    "headers": {...}
  }
}
```

### 3. Development Headers
All responses include helpful headers for frontend development:

- `X-Frontend-Dev-Mode: true` - Indicates dev mode is active
- `X-Server-Time: <timestamp>` - Server timestamp for debugging
- Enhanced CORS headers for preflight requests

### 4. Development Info Endpoint
A new `/dev-info` endpoint provides configuration information:

```json
{
  "frontend_dev_mode": true,
  "cors_origins": ["http://localhost:3000", "..."],
  "server_time": 1234567890,
  "features": {
    "detailed_errors": true,
    "dev_headers": true,
    "auto_cors": true
  }
}
```

## Usage

### Basic Usage

```bash
# Enable frontend dev mode with web UI
adk web --frontend-dev-mode /path/to/agents

# Enable frontend dev mode with API server only  
adk api_server --frontend-dev-mode --port 8000 /path/to/agents
```

### Combined with Custom CORS Origins

```bash
# Add custom origins in addition to auto-configured dev origins
adk web --frontend-dev-mode --allow_origins https://myapp.com --allow_origins https://staging.myapp.com /path/to/agents
```

### Production vs Development

```bash
# Development - with all dev features enabled
adk web --frontend-dev-mode --reload /path/to/agents

# Production - clean configuration without dev features
adk web --allow_origins https://myapp.com --no-reload /path/to/agents
```

## Implementation Details

### Files Modified

1. **`src/google/adk/cli/cli_tools_click.py`**
   - Added `--frontend-dev-mode` flag to `fast_api_common_options()`
   - Updated `cli_web()` and `cli_api_server()` functions to accept and pass the parameter

2. **`src/google/adk/cli/fast_api.py`**
   - Modified `get_fast_api_app()` to accept `frontend_dev_mode` parameter
   - Added automatic CORS origin configuration logic
   - Implemented development middleware for enhanced error handling and headers
   - Added `/dev-info` endpoint

3. **`tests/unittests/cli/test_fast_api.py`**
   - Added comprehensive tests for the new functionality

### Key Implementation Features

- **Backwards Compatible**: The feature is opt-in via CLI flag
- **Additive CORS**: Combines user-specified origins with auto-configured dev origins
- **Conditional Middleware**: Development features only active when flag is enabled
- **Error Safety**: Enhanced error handling doesn't break existing error responses

## Benefits for Frontend Developers

1. **Zero Configuration CORS**: No need to manually configure CORS for common dev servers
2. **Better Debugging**: Detailed error responses help identify issues quickly
3. **Development Visibility**: Headers and endpoints provide insight into server state
4. **Faster Iteration**: Automatic reload and dev-friendly settings speed up development

## Security Considerations

- **Development Only**: This feature should only be used in development environments
- **Detailed Errors**: Error responses include sensitive debugging information
- **Permissive CORS**: Automatically allows common localhost origins
- **Production Safety**: Feature is disabled by default and requires explicit opt-in

## Frontend Integration Examples

### React/Next.js (localhost:3000)
```javascript
// Automatically allowed when --frontend-dev-mode is enabled
const response = await fetch('http://localhost:8000/apps/my-app/users/user123/sessions');
```

### Angular (localhost:4200)
```typescript
// Automatically allowed when --frontend-dev-mode is enabled
this.http.get('http://localhost:8000/dev-info').subscribe(info => {
  console.log('Server dev mode:', info.frontend_dev_mode);
});
```

### Vue/Vite (localhost:5173)
```javascript
// Check if server is in dev mode
const devInfo = await fetch('http://localhost:8000/dev-info').then(r => r.json());
if (devInfo.frontend_dev_mode) {
  console.log('Server is in development mode');
}
```

## Migration Guide

### Existing Projects
No changes required for existing projects. The feature is opt-in and backwards compatible.

### New Projects
Consider using `--frontend-dev-mode` during development:

```bash
# Old way - manual CORS configuration
adk web --allow_origins http://localhost:3000 /path/to/agents

# New way - automatic dev configuration
adk web --frontend-dev-mode /path/to/agents
```

## Troubleshooting

### CORS Issues
- Verify `--frontend-dev-mode` is enabled
- Check `/dev-info` endpoint to see configured origins
- Ensure your frontend dev server is using a supported port

### Missing Debug Information
- Confirm `--frontend-dev-mode` flag is set
- Check response headers for `X-Frontend-Dev-Mode: true`
- Verify error responses include `traceback` field

### Performance Concerns
- Development middleware adds minimal overhead
- Use `--no-reload` in production environments
- Disable `--frontend-dev-mode` for production deployments
