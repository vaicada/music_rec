"""
Authentication Module for Music Recommender.

Handles:
    - Password hashing/verification (bcrypt via passlib)
    - JWT token creation/verification (python-jose)
    - FastAPI dependency injection for auth (required + optional)

Usage:
    from auth import get_current_user, get_optional_user, create_access_token, hash_password, verify_password
"""

import os
from datetime import datetime, timedelta
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
from sqlalchemy.orm import Session

from database import User, get_db

# =============================================================================
# Bcrypt Compatibility Shim
# (passlib 1.7.4 fails to detect __about__ in bcrypt >= 4.x)
# =============================================================================
try:
    import bcrypt
    if not hasattr(bcrypt, "__about__"):
        # Create a minimal shim so passlib can read the version
        class _About:
            __version__ = bcrypt.__version__
        bcrypt.__about__ = _About()
except Exception:
    pass

# =============================================================================
# Configuration
# =============================================================================

SECRET_KEY = os.getenv("JWT_SECRET_KEY", "music-recommender-secret-key-change-in-production-2026")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "10080"))  # 7 days default

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
bearer_scheme = HTTPBearer(auto_error=False)


# =============================================================================
# Pydantic Schemas
# =============================================================================

class UserPublic(BaseModel):
    """Public-safe user representation (no password)."""
    id: int
    username: str
    email: str

    class Config:
        from_attributes = True


class TokenPayload(BaseModel):
    """JWT token payload."""
    sub: str  # user_id as string
    username: str
    exp: Optional[datetime] = None


# =============================================================================
# Password Helpers
# =============================================================================

def hash_password(plain_password: str) -> str:
    """Hash a plain-text password using bcrypt."""
    return pwd_context.hash(plain_password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plain-text password against a bcrypt hash."""
    return pwd_context.verify(plain_password, hashed_password)


# =============================================================================
# JWT Helpers
# =============================================================================

def create_access_token(user_id: int, username: str) -> str:
    """Create a signed JWT access token for the given user."""
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {
        "sub": str(user_id),
        "username": username,
        "exp": expire,
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> Optional[TokenPayload]:
    """Decode and validate a JWT token. Returns None if invalid."""
    try:
        data = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return TokenPayload(**data)
    except JWTError:
        return None


# =============================================================================
# FastAPI Dependencies
# =============================================================================

def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> User:
    """
    Require authenticated user. Raises 401 if token missing or invalid.

    Usage::

        @app.get("/api/protected")
        async def protected(user: User = Depends(get_current_user)):
            return {"hello": user.username}
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = decode_token(credentials.credentials)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = db.query(User).filter(User.id == int(payload.sub)).first()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user


def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> Optional[User]:
    """
    Optionally authenticate user. Returns None if no token provided.
    Raises 401 only if a token IS provided but is invalid.

    Usage::

        @app.get("/api/search")
        async def search(user: Optional[User] = Depends(get_optional_user)):
            if user:
                # save to history
                pass
    """
    if credentials is None:
        return None

    payload = decode_token(credentials.credentials)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return db.query(User).filter(User.id == int(payload.sub)).first()
