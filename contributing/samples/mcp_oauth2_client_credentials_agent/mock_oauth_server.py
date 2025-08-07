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

"""
Mock OAuth2 Server for Testing OAuth2 Client Credentials Flow

This is a minimal OAuth2 server implementation that supports:
- RFC 8414 OAuth discovery endpoints
- Client credentials grant type
- Testing the ADK OAuth2 functionality

DO NOT use this in production - it's for demonstration purposes only!
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import JSONResponse
import uvicorn


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Mock OAuth2 Server for ADK Testing")

# Simple in-memory client store (DO NOT use in production!)
CLIENTS = {
    "demo_client_id": {
        "client_secret": "demo_client_secret",
        "scopes": ["api:read", "api:write"]
    },
    "server1_client": {
        "client_secret": "server1_secret",
        "scopes": ["api:read"]
    },
    "server2_client": {
        "client_secret": "server2_secret", 
        "scopes": ["api:read", "api:write"]
    }
}

# Simple token store (DO NOT use in production!)
TOKENS: Dict[str, Dict[str, Any]] = {}


@app.get("/.well-known/oauth-protected-resource")
async def oauth_protected_resource(request: Request):
    """
    RFC 8414 OAuth Protected Resource Discovery
    
    This endpoint tells clients where to find the authorization server.
    """
    base_url = f"{request.url.scheme}://{request.url.netloc}"
    
    logger.info(f"🔍 OAuth protected resource discovery requested from {request.client.host if request.client else 'unknown'}")
    
    return JSONResponse({
        "authorization_servers": [base_url]
    })


@app.get("/.well-known/oauth-authorization-server")
async def oauth_authorization_server(request: Request):
    """
    RFC 8414 OAuth Authorization Server Metadata
    
    This endpoint provides metadata about the authorization server capabilities.
    """
    base_url = f"{request.url.scheme}://{request.url.netloc}"
    
    logger.info(f"🔍 OAuth authorization server metadata requested from {request.client.host if request.client else 'unknown'}")
    
    return JSONResponse({
        "issuer": base_url,
        "token_endpoint": f"{base_url}/token",
        "grant_types_supported": ["client_credentials"],
        "token_endpoint_auth_methods_supported": ["client_secret_post"],
        "scopes_supported": ["api:read", "api:write"],
        "response_types_supported": ["token"]
    })


@app.post("/token")
async def token_endpoint(
    request: Request,
    grant_type: str = Form(...),
    client_id: str = Form(...),
    client_secret: str = Form(...),
    scope: str = Form(None)
):
    """
    OAuth2 Token Endpoint - Client Credentials Grant
    
    This endpoint exchanges client credentials for access tokens.
    """
    logger.info(f"🔐 Token request from {request.client.host if request.client else 'unknown'}")
    logger.info(f"   Grant Type: {grant_type}")
    logger.info(f"   Client ID: {client_id}")
    logger.info(f"   Scopes: {scope or 'default'}")
    
    # Validate grant type
    if grant_type != "client_credentials":
        logger.warning(f"❌ Unsupported grant type: {grant_type}")
        raise HTTPException(
            status_code=400,
            detail={
                "error": "unsupported_grant_type",
                "error_description": "Only client_credentials grant type is supported"
            }
        )
    
    # Validate client credentials
    if client_id not in CLIENTS:
        logger.warning(f"❌ Unknown client ID: {client_id}")
        raise HTTPException(
            status_code=401,
            detail={
                "error": "invalid_client",
                "error_description": "Unknown client ID"
            }
        )
    
    client = CLIENTS[client_id]
    if client["client_secret"] != client_secret:
        logger.warning(f"❌ Invalid client secret for client: {client_id}")
        raise HTTPException(
            status_code=401,
            detail={
                "error": "invalid_client",
                "error_description": "Invalid client secret"
            }
        )
    
    # Validate scopes (if provided)
    requested_scopes = scope.split() if scope else client["scopes"]
    for requested_scope in requested_scopes:
        if requested_scope not in client["scopes"]:
            logger.warning(f"❌ Unauthorized scope '{requested_scope}' for client: {client_id}")
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "invalid_scope",
                    "error_description": f"Scope '{requested_scope}' not authorized for client"
                }
            )
    
    # Generate access token (simple approach for demo)
    access_token = f"demo_token_{client_id}_{int(time.time())}"
    expires_in = 3600  # 1 hour
    
    # Store token (in production, use proper token storage)
    TOKENS[access_token] = {
        "client_id": client_id,
        "scopes": requested_scopes,
        "expires_at": time.time() + expires_in
    }
    
    logger.info(f"✅ Successfully issued token for client: {client_id}")
    logger.info(f"   Token: {access_token[:20]}...")
    logger.info(f"   Scopes: {' '.join(requested_scopes)}")
    
    return JSONResponse({
        "access_token": access_token,
        "token_type": "Bearer",
        "expires_in": expires_in,
        "scope": " ".join(requested_scopes)
    })


@app.get("/validate")
async def validate_token(request: Request):
    """
    Token validation endpoint (for debugging purposes)
    
    This endpoint allows you to validate tokens issued by this server.
    """
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    
    token = auth_header[7:]  # Remove "Bearer " prefix
    
    if token not in TOKENS:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    token_info = TOKENS[token]
    
    # Check if token is expired
    if time.time() > token_info["expires_at"]:
        del TOKENS[token]  # Clean up expired token
        raise HTTPException(status_code=401, detail="Token expired")
    
    logger.info(f"✅ Valid token for client: {token_info['client_id']}")
    
    return JSONResponse({
        "valid": True,
        "client_id": token_info["client_id"],
        "scopes": token_info["scopes"],
        "expires_at": token_info["expires_at"]
    })


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return JSONResponse({"status": "healthy", "server": "Mock OAuth2 Server"})


@app.get("/")
async def root():
    """Root endpoint with server information."""
    return JSONResponse({
        "server": "Mock OAuth2 Server for ADK Testing",
        "endpoints": {
            "discovery": "/.well-known/oauth-protected-resource",
            "metadata": "/.well-known/oauth-authorization-server", 
            "token": "/token",
            "validate": "/validate",
            "health": "/health"
        },
        "demo_clients": {
            "demo_client_id": "demo_client_secret",
            "server1_client": "server1_secret",
            "server2_client": "server2_secret"
        },
        "note": "This is a demo server - DO NOT use in production!"
    })


async def main():
    """Run the mock OAuth2 server."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Mock OAuth2 Server for ADK Testing")
    parser.add_argument("--ssl-keyfile", help="SSL private key file for HTTPS")
    parser.add_argument("--ssl-certfile", help="SSL certificate file for HTTPS")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=9204, help="Port to bind to (default: 9204)")
    args = parser.parse_args()
    
    protocol = "https" if args.ssl_keyfile and args.ssl_certfile else "http"
    
    print("🚀 Starting Mock OAuth2 Server for ADK Testing")
    print(f"📍 Server will be available at: {protocol}://{args.host}:{args.port}")
    print(f"🔍 Discovery endpoint: {protocol}://{args.host}:{args.port}/.well-known/oauth-protected-resource")
    print(f"🔐 Token endpoint: {protocol}://{args.host}:{args.port}/token")
    
    if protocol == "https":
        print("🔒 HTTPS mode enabled with SSL certificates")
        print(f"   SSL Key: {args.ssl_keyfile}")
        print(f"   SSL Cert: {args.ssl_certfile}")
        print("⚠️  If using self-signed certificates, set verify_ssl=False in MCPAuthDiscovery")
    else:
        print("🔓 HTTP mode (no SSL)")
        
    print("⚠️  This is for testing only - DO NOT use in production!")
    print()
    print("Demo clients:")
    for client_id, client_info in CLIENTS.items():
        print(f"  • {client_id} / {client_info['client_secret']} (scopes: {', '.join(client_info['scopes'])})")
    print()
    
    config = uvicorn.Config(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
        ssl_keyfile=args.ssl_keyfile,
        ssl_certfile=args.ssl_certfile
    )
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Mock OAuth2 Server stopped") 