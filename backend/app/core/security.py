"""
Security utilities: JWT creation/verification and password hashing.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional

import bcrypt
from jose import JWTError, jwt

from app.core.config import settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)


# ── Password Utilities ────────────────────────────────────────────────────────

def hash_password(plain_password: str) -> str:
    """Hash a plain-text password using bcrypt."""
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(plain_password.encode("utf-8"), salt).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plain-text password against a stored bcrypt hash."""
    return bcrypt.checkpw(
        plain_password.encode("utf-8"), hashed_password.encode("utf-8")
    )


# ── JWT Utilities ─────────────────────────────────────────────────────────────

def create_access_token(
    subject: str,
    expires_delta: Optional[timedelta] = None,
) -> str:
    """
    Create a signed JWT access token.

    Args:
        subject: The token subject (typically user_id as string).
        expires_delta: Optional custom expiry. Defaults to settings value.

    Returns:
        Encoded JWT string.
    """
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(
            minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
        )

    payload = {
        "sub": subject,
        "exp": expire,
        "iat": datetime.now(timezone.utc),
        "type": "access",
    }

    token = jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    logger.debug("access_token_created", subject=subject, expires_at=expire.isoformat())
    return token


def decode_access_token(token: str) -> Optional[str]:
    """
    Decode and verify a JWT access token.

    Args:
        token: The encoded JWT string.

    Returns:
        The subject (user_id) if valid, or None if invalid/expired.
    """
    try:
        payload = jwt.decode(
            token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM]
        )
        subject: Optional[str] = payload.get("sub")
        token_type: Optional[str] = payload.get("type")

        if subject is None or token_type != "access":
            logger.warning("invalid_token_payload", payload=payload)
            return None

        return subject
    except JWTError as exc:
        logger.warning("jwt_decode_error", error=str(exc))
        return None
