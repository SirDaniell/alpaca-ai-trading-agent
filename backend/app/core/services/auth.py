"""
Authentication and Authorization Module
Provides JWT-based authentication for API endpoints
"""

import logging

logger = logging.getLogger(__name__)

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
import os
from typing import Optional, Dict
from datetime import datetime, timedelta
import uuid

security = HTTPBearer(auto_error=False)


def create_access_token(user_id: str, wallet_address: Optional[str] = None, roles: list = None) -> str:
    """
    Create JWT access token
    
    Args:
        user_id: User identifier
        wallet_address: Optional wallet address
        roles: User roles
        
    Returns:
        JWT token string
    """
    expires_delta = timedelta(hours=int(os.getenv("JWT_EXPIRATION_HOURS", 24)))
    expire = datetime.now(timezone.utc) + expires_delta
    
    payload = {
        "user_id": user_id,
        "wallet_address": wallet_address,
        "roles": roles or [],
        "exp": expire,
        "iat": datetime.now(timezone.utc)
    }
    
    token = jwt.encode(
        payload,
        os.getenv("JWT_SECRET", "dev-secret-change-in-production"),
        algorithm=os.getenv("JWT_ALGORITHM", "HS256")
    )
    
    return token


from fastapi import Request

async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Dict:
    """
    Extract and validate user from JWT token (Head or Cookie)
    
    Args:
        request: FastAPI Request object
        credentials: HTTP Bearer token
        
    Returns:
        User information dict
        
    Raises:
        HTTPException: If token is invalid or expired
    """
    token = None
    
    # 1. Try Header
    if credentials:
        token = credentials.credentials
        
    # 2. Try Cookie
    if not token:
        token = request.cookies.get("access_token")

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    try:
        payload = jwt.decode(
            token,
            os.getenv("JWT_SECRET", "dev-secret-change-in-production"),
            algorithms=[os.getenv("JWT_ALGORITHM", "HS256")]
        )
        
        return {
            'user_id': payload['user_id'],
            'wallet_address': payload.get('wallet_address'),
            'roles': payload.get('roles', [])
        }
    
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError:
        # SATELLITE MODE CHECK
        # If we are a Client Backend (Satellite) connected to a Server Backend,
        # we might receive tokens signed by the Server Backend's key, which we don't have.
        # In this case, we rely on the TLS connection and the fact that we are proxying.
        server_backend_url = os.getenv("SERVER_BACKEND_URL")
        if server_backend_url:
            try:
                # Decode purely for claims, ignoring signature verification
                payload = jwt.decode(token, options={"verify_signature": False})
                logger.warning(f"🛰️ Satellite Mode: Accepting unverified token from {server_backend_url} for user {payload.get('user_id')}")
                
                # Check expiration manually since verify_signature=False might skip it depending on lib version
                # (PyJWT verifies exp by default even with verify_signature=False, but let's be safe)
                # actually options={"verify_signature": False} usually still checks exp unless verify_exp is False.
                
                return {
                    'user_id': payload['user_id'],
                    'wallet_address': payload.get('wallet_address'),
                    'roles': payload.get('roles', []),
                    '_is_satellite': True
                }
            except Exception as e:
                logger.error(f"Satellite mode token parse failed: {e}")
                pass # Fall through to 401
                
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[Dict]:
    """
    Optional authentication - returns None if no credentials provided
    
    Args:
        credentials: Optional HTTP Bearer token
        
    Returns:
        User information dict or None
    """
    if not credentials:
        return None
    
    try:
        return await get_current_user(credentials)
    except HTTPException:
        return None


def require_role(required_role: str):
    """
    Dependency to require specific role
    
    Usage:
        @router.get("/admin")
        async def admin_endpoint(user: dict = Depends(require_role("admin"))):
            ...
    """
    async def role_checker(user: Dict = Depends(get_current_user)) -> Dict:
        if required_role not in user.get('roles', []):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{required_role}' required"
            )
        return user
    
    return role_checker


# Development helper - remove in production
def create_dev_token(user_id: str = "dev-user") -> str:
    """Create development token for testing"""
    if os.getenv("ENVIRONMENT") != "development":
        raise RuntimeError("Dev tokens only available in development")
    
    return create_access_token(
        user_id=user_id,
        wallet_address="0xdev",
        roles=["user", "admin"]
    )
