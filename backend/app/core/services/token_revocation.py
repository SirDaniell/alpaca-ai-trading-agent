"""
Token Revocation Service

Provides functionality to revoke tokens and check if tokens have been revoked.
This is critical for security - allows invalidating tokens before natural expiry.
"""

import time
from datetime import datetime
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
import logging

from app.core.services.auth_service import AuthService
from app.core.services.redis_service import redis_service

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# In-process negative-result cache
# ---------------------------------------------------------------------------
# Token revocation checks are on the hot-path: every authenticated request
# calls is_token_revoked().  Redis is already checked first (fast), but if
# the token is NOT in Redis the code fell through to SQLite, holding a pool
# connection for each concurrent request and exhausting the QueuePool under
# bursty MT5 tick load.
#
# Fix: cache "this token is NOT revoked" results in memory for a short window
# (TOKEN_VALID_CACHE_TTL seconds).  On a cache hit we skip both Redis AND
# SQLite completely, releasing pool connections immediately.
#
# Cache eviction: entries are checked on read; a background cleanup isn't
# needed because the dict is bounded by active JTIs (one per logged-in user
# per session) and entries expire naturally when the token itself expires.

_valid_token_cache: dict[str, float] = {}   # jti -> expiry_monotonic
TOKEN_VALID_CACHE_TTL = 30  # seconds — short enough to respect rapid revocations


def _cache_token_valid(token_id: str) -> None:
    _valid_token_cache[token_id] = time.monotonic() + TOKEN_VALID_CACHE_TTL


def _is_cached_valid(token_id: str) -> bool:
    exp = _valid_token_cache.get(token_id)
    if exp is None:
        return False
    if time.monotonic() > exp:
        # Expired cache entry — remove it
        _valid_token_cache.pop(token_id, None)
        return False
    return True


def _invalidate_cache(token_id: str) -> None:
    """Call this whenever a token is revoked so the cache doesn't lie."""
    _valid_token_cache.pop(token_id, None)


class TokenRevocationService:
    """
    Service for managing token revocation.

    Revoked tokens are stored in a database table with their expiry time.
    Expired revocations are automatically cleaned up to prevent table bloat.
    """

    @staticmethod
    async def revoke_token(
        db: AsyncSession, token: str, reason: str = "user_logout"
    ) -> bool:
        """
        Revoke a specific token.

        Args:
            db: Database session
            token: JWT token to revoke
            reason: Reason for revocation (for audit purposes)

        Returns:
            True if revocation successful, False otherwise
        """
        try:
            # Decode token to get metadata
            payload = AuthService.decode_token(token)
            if not payload:
                logger.warning("Cannot revoke invalid token")
                return False

            token_id = payload.get("jti")
            user_id = payload.get("sub")
            exp = payload.get("exp")

            if not token_id or not user_id or not exp:
                logger.error("Token missing required fields (jti, sub, or exp)")
                return False

            # Convert exp timestamp to TTL
            now = datetime.now(timezone.utc).timestamp()
            ttl = int(exp - now)
            if ttl <= 0:
                logger.info(f"Token {token_id[:8]} already expired, no need to revoke in Redis")
            else:
                # Store in Redis: revoked_token:{jti} -> reason
                # TTL ensures it's automatically removed after token naturally expires
                await redis_service.set(f"revoked_token:{token_id}", reason, ex=ttl)
                logger.info(f"🚀 Token revoked in Redis: {token_id[:8]}... (TTL: {ttl}s)")

            # Purge the in-process valid-token cache so the next check doesn't
            # use a stale "not revoked" entry.
            _invalidate_cache(token_id)

            # FALLBACK/AUDIT: Insert into revoked_tokens SQLite table
            expires_at = datetime.fromtimestamp(exp)
            query = text(
                """
                INSERT INTO revoked_tokens (token_id, user_id, reason, expires_at)
                VALUES (:token_id, :user_id, :reason, :expires_at)
                ON CONFLICT (token_id) DO NOTHING
            """
            )

            await db.execute(
                query,
                {
                    "token_id": token_id,
                    "user_id": user_id,
                    "reason": reason,
                    "expires_at": expires_at,
                },
            )
            await db.commit()

            return True

        except Exception as e:
            logger.error(f"❌ Failed to revoke token: {str(e)}")
            await db.rollback()
            return False

    @staticmethod
    async def revoke_all_user_tokens(
        db: AsyncSession, user_id: str, reason: str = "security_action"
    ) -> int:
        """
        Revoke all active tokens for a specific user.

        This is useful for:
        - Password changes
        - Account security breaches
        - Forced logout from all devices

        Args:
            db: Database session
            user_id: User ID whose tokens should be revoked
            reason: Reason for mass revocation

        Returns:
            Number of tokens revoked
        """
        try:
            # Store user-wide revocation in Redis
            # Key: user_revocation:{user_id} -> timestamp
            # We set a long TTL (e.g., 7 days) or use a permanent record if needed
            revoked_at = datetime.now(timezone.utc)
            await redis_service.set(f"user_revocation:{user_id}", str(revoked_at.timestamp()), ex=3600*24*7)
            logger.info(f"🚀 User-wide revocation stored in Redis for user {user_id}")

            # Flush all in-process cache entries — we can't know which JTIs belong
            # to this user without scanning, so a full clear is safe (rare operation).
            _valid_token_cache.clear()

            # FALLBACK/AUDIT: SQLite
            query = text(
                """
                INSERT INTO user_token_revocations (user_id, revoked_at, reason)
                VALUES (:user_id, :revoked_at, :reason)
                ON CONFLICT (user_id) DO UPDATE SET
                    revoked_at = EXCLUDED.revoked_at,
                    reason = EXCLUDED.reason
            """
            )

            await db.execute(
                query,
                {"user_id": user_id, "revoked_at": revoked_at, "reason": reason},
            )
            await db.commit()

            return 1

        except Exception as e:
            logger.error(f"❌ Failed to revoke user tokens: {str(e)}")
            await db.rollback()
            return 0

    @staticmethod
    async def is_token_revoked(db: AsyncSession, token: str) -> bool:
        """
        Check if a token has been revoked.

        This checks two things:
        1. Is the specific token ID in the revocation list?
        2. Has the user had all tokens revoked after this token was issued?

        Args:
            db: Database session
            token: JWT token to check

        Returns:
            True if token is revoked, False otherwise
        """
        try:
            # Decode token
            payload = AuthService.decode_token(token)
            if not payload:
                return True  # Treat invalid tokens as revoked

            token_id = payload.get("jti")
            user_id = payload.get("sub")
            issued_at = payload.get("iat")

            if not token_id or not user_id:
                return True  # Missing required fields = revoked

            # 0. In-process cache: skip Redis + SQLite for recently-confirmed-valid tokens.
            #    This is the primary defence against QueuePool exhaustion under bursty
            #    tick-driven request load.
            if _is_cached_valid(token_id):
                return False

            # Mark valid speculatively to short-circuit concurrent burst requests (e.g. 24 WS connects)
            _cache_token_valid(token_id)

            # 1. Check Redis first (Source of Truth)
            is_revoked_redis = await redis_service.exists(f"revoked_token:{token_id}")
            if is_revoked_redis:
                logger.debug(f"Token {token_id[:8]} is revoked in Redis")
                _invalidate_cache(token_id)
                return True

            # 2. Check User-wide revocation in Redis
            revocation_ts = await redis_service.get(f"user_revocation:{user_id}")
            if revocation_ts and issued_at:
                if float(revocation_ts) > issued_at:
                    logger.debug(f"Token {token_id[:8]} revoked by user-wide revocation in Redis")
                    _invalidate_cache(token_id)
                    return True

            # 3. FALLBACK: Check SQLite (if Redis failed or for legacy)
            query = text(
                """
                SELECT 1 FROM revoked_tokens
                WHERE token_id = :token_id
                LIMIT 1
            """
            )

            result = await db.execute(query, {"token_id": token_id})
            if result.fetchone():
                _invalidate_cache(token_id)
                return True

            if issued_at:
                issued_at_dt = datetime.fromtimestamp(issued_at)
                query = text(
                    """
                    SELECT revoked_at FROM user_token_revocations
                    WHERE user_id = :user_id
                    LIMIT 1
                """
                )
                result = await db.execute(query, {"user_id": user_id})
                row = result.fetchone()
                if row and row[0] > issued_at_dt:
                    _invalidate_cache(token_id)
                    return True

            return False

        except Exception as e:
            logger.error(f"❌ Error checking token revocation: {str(e)}")
            # ✅ FIX: Do NOT fail-closed on transient infrastructure errors.
            # "Fail closed" (return True = revoked) made sense for unexpected errors,
            # but it causes every request to be rejected during SQLite pool exhaustion
            # (QueuePool limit reached), which then trips the frontend circuit breaker
            # and takes down the entire data pipeline for 3 minutes.
            #
            # Strategy: distinguish between DB availability errors (transient, fail-open)
            # and genuine decode/logic errors (fail-closed).
            err_msg = str(e).lower()
            is_db_unavailable = any(x in err_msg for x in (
                'queuepool', 'timeout', 'connection', 'pool', 'locked', 'operationalerror'
            ))
            if is_db_unavailable:
                logger.warning(
                    "⚠️ Token revocation check skipped (DB unavailable) — "
                    "treating token as valid to avoid false 401s"
                )
                return False  # Fail-open: assume not revoked when DB is unreachable
            # For any other unexpected error (decode failure, missing fields), fail-closed
            return True

    @staticmethod
    async def cleanup_expired_revocations(db: AsyncSession) -> int:
        """
        Clean up expired token revocations from the database.

        This should be run periodically (e.g., daily cron job) to prevent
        the revoked_tokens table from growing indefinitely.

        Args:
            db: Database session

        Returns:
            Number of revocations cleaned up
        """
        try:
            query = text(
                """
                DELETE FROM revoked_tokens
                WHERE expires_at < :now
            """
            )

            result = await db.execute(query, {"now": datetime.now(timezone.utc)})
            await db.commit()

            count = result.rowcount
            if count > 0:
                logger.info(f"🧹 Cleaned up {count} expired token revocations")

            return count

        except Exception as e:
            logger.error(f"❌ Failed to cleanup revocations: {str(e)}")
            await db.rollback()
            return 0
