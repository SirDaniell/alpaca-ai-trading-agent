"""
Request context management for request ID tracking.
Provides context variables and logging utilities for correlating requests across logs.
"""

import contextvars
import logging
import uuid
from typing import Optional

# Priority 5: Context variable for request ID tracking
request_id_context: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    'request_id', default=None
)


def get_request_id() -> str:
    """Get the current request ID from context, or generate a new one if not set."""
    request_id = request_id_context.get()
    if not request_id:
        request_id = str(uuid.uuid4())
        request_id_context.set(request_id)
    return request_id


def set_request_id(request_id: str) -> None:
    """Set the request ID for the current context."""
    request_id_context.set(request_id)


class RequestIDFormatFilter(logging.Filter):
    """
    Logging filter that adds request ID to all log records.
    Integrates seamlessly with existing logging formatters.
    """
    
    def filter(self, record):
        """Add request_id to the log record for use in format strings."""
        record.request_id = get_request_id()
        return True
