import os
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
from typing import Optional
import uuid
import logging

logger = logging.getLogger(__name__)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# It is crucial to set this in your environment variables for production
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 4320  # 72 hours for offline access
REFRESH_TOKEN_EXPIRE_DAYS = 7

logger.info(
    f"🔑 AuthService initialized with SECRET_KEY: {SECRET_KEY[:20]}... (length: {len(SECRET_KEY)})"
)


class AuthService:
    @staticmethod
    def hash_password(password: str) -> str:
        return pwd_context.hash(password)

    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        # Handle cases where hashed_password might be None or not a string
        if not isinstance(hashed_password, str):
            return False
        return pwd_context.verify(plain_password, hashed_password)

    @staticmethod
    def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
        to_encode = data.copy()
        expire = datetime.now(timezone.utc) + (
            expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        )
        to_encode.update(
            {
                "exp": expire,
                "iat": datetime.now(timezone.utc),  # Issued at timestamp
                "jti": str(uuid.uuid4()),  # Unique token ID for revocation
                "type": "access",
            }
        )
        return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

    @staticmethod
    def create_refresh_token(data: dict):
        to_encode = data.copy()
        expire = datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
        to_encode.update(
            {
                "exp": expire,
                "iat": datetime.now(timezone.utc),  # Issued at timestamp
                "jti": str(uuid.uuid4()),  # Unique token ID for revocation
                "type": "refresh",
            }
        )
        return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

    @staticmethod
    def decode_token(token: str) -> Optional[dict]:
        logger.debug(
            f"🔓 Attempting to decode token with SECRET_KEY: {SECRET_KEY[:20]}..."
        )
        logger.debug(f"🔓 Token (first 30 chars): {token[:30]}...")
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            logger.debug(
                f"✅ JWT decode successful! Payload keys: {list(payload.keys())}"
            )
            return payload
        except JWTError as e:
            # Satellite Mode Logic
            server_backend_url = os.getenv("SERVER_BACKEND_URL")
            if server_backend_url:
                try:
                    # Try decoding without signature verification
                    payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM], options={"verify_signature": False})
                    logger.warning(f"🛰️ AuthService: Satellite Mode - Payload extracted without signature verification.")
                    return payload
                except Exception as satellite_error:
                    logger.error(f"❌ Satellite decode failed: {satellite_error}")
            
            logger.error(f"❌ JWT decode failed: {type(e).__name__}: {str(e)}")
            return None
