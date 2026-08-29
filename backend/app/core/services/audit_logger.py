"""
Audit Logging Service for Security Events

Logs important security events like:
- Login attempts (success/failure)
- Token refresh
- Logout events
- Token revocation
- Password changes
- Failed authentication attempts
"""

from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
import logging
import json

logger = logging.getLogger(__name__)


class AuditLogger:
    """Service for logging security-related events"""

    @staticmethod
    async def log_event(
        db: AsyncSession,
        event_type: str,
        user_id: str = None,
        ip_address: str = None,
        user_agent: str = None,
        details: dict = None,
        success: bool = True,
    ):
        """
        Log a security event to the database

        Args:
            db: Database session
            event_type: Type of event (login, logout, token_refresh, etc.)
            user_id: User ID if applicable
            ip_address: IP address of the request
            user_agent: User agent string
            details: Additional details as JSON
            success: Whether the event was successful
        """
        try:
            query = text(
                """
                INSERT INTO audit_logs 
                (event_type, user_id, ip_address, user_agent, details, success, created_at)
                VALUES 
                (:event_type, :user_id, :ip_address, :user_agent, :details, :success, :created_at)
            """
            )

            await db.execute(
                query,
                {
                    "event_type": event_type,
                    "user_id": user_id,
                    "ip_address": ip_address,
                    "user_agent": user_agent,
                    "details": json.dumps(details) if details else None,
                    "success": success,
                    "created_at": datetime.now(timezone.utc),
                },
            )
            await db.commit()

            logger.info(
                f"Audit log: {event_type} - User: {user_id} - Success: {success}"
            )
        except Exception as e:
            logger.error(f"Failed to log audit event: {e}")
            # Don't fail the main operation if audit logging fails
            await db.rollback()

    @staticmethod
    async def log_login_attempt(
        db: AsyncSession,
        username: str,
        ip_address: str,
        user_agent: str,
        success: bool,
        user_id: str = None,
        failure_reason: str = None,
    ):
        """Log a login attempt"""
        details = {"username": username}
        if failure_reason:
            details["failure_reason"] = failure_reason

        await AuditLogger.log_event(
            db=db,
            event_type="login_attempt",
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            details=details,
            success=success,
        )

    @staticmethod
    async def log_logout(
        db: AsyncSession,
        user_id: str,
        ip_address: str,
        user_agent: str,
        logout_type: str = "user_initiated",
    ):
        """Log a logout event"""
        await AuditLogger.log_event(
            db=db,
            event_type="logout",
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            details={"logout_type": logout_type},
            success=True,
        )

    @staticmethod
    async def log_token_refresh(
        db: AsyncSession, user_id: str, ip_address: str, user_agent: str, success: bool
    ):
        """Log a token refresh event"""
        await AuditLogger.log_event(
            db=db,
            event_type="token_refresh",
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            success=success,
        )

    @staticmethod
    async def log_token_revocation(
        db: AsyncSession,
        user_id: str,
        ip_address: str,
        user_agent: str,
        reason: str,
        revocation_type: str = "single",
    ):
        """Log a token revocation event"""
        await AuditLogger.log_event(
            db=db,
            event_type="token_revocation",
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            details={"reason": reason, "revocation_type": revocation_type},
            success=True,
        )

    @staticmethod
    async def log_password_change(
        db: AsyncSession, user_id: str, ip_address: str, user_agent: str, success: bool
    ):
        """Log a password change event"""
        await AuditLogger.log_event(
            db=db,
            event_type="password_change",
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            success=success,
        )

    @staticmethod
    async def log_failed_authentication(
        db: AsyncSession,
        ip_address: str,
        user_agent: str,
        reason: str,
        attempted_user: str = None,
    ):
        """Log a failed authentication attempt"""
        await AuditLogger.log_event(
            db=db,
            event_type="failed_authentication",
            ip_address=ip_address,
            user_agent=user_agent,
            details={"reason": reason, "attempted_user": attempted_user},
            success=False,
        )
