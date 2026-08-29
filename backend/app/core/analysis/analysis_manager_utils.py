"""
AnalysisManager Utility Functions & Types

Helper classes and types used by AnalysisManager:
- ErrorHandling (ErrorCategory, ErrorContext)
- Request/Response models (DataSourceRequest, AnalysisRequest)
- Caching (SessionDataCache)
- Performance monitoring (ResourceMonitor)
- Request deduplication (RequestDeduplicator)

These components provide:
- Safe error handling with retry logic
- Clear request structures
- Session data caching
- Performance tracking
- Request deduplication

"""

import asyncio
import time
import hashlib
from typing import Dict, Optional, Any, List, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import pandas as pd
from datetime import datetime, date, timezone


class ErrorCategory(Enum):
    """Classification for error handling and retry logic."""
    
    # Recoverable - safe to retry
    NETWORK_TIMEOUT = "network_timeout"          # MT5/DB connection timeout
    CONNECTION_RESET = "connection_reset"        # Mid-transfer reset
    TEMPORARY_UNAVAILABLE = "temporary_unavailable"  # Service temporarily down
    LOCK_CONTENTION = "lock_contention"         # DB lock, contention
    
    # Recoverable with delay - need exponential backoff
    RATE_LIMITED = "rate_limited"                # Too many requests
    RESOURCE_EXHAUSTED = "resource_exhausted"   # Memory/CPU spike
    
    # Fatal - don't retry
    INVALID_DATA_FORMAT = "invalid_data_format"  # Malformed data
    MISSING_REQUIRED_FIELD = "missing_required_field"
    VALIDATION_FAILED = "validation_failed"      # Schema mismatch
    AUTHENTICATION_FAILED = "auth_failed"        # Invalid credentials
    PERMISSION_DENIED = "permission_denied"
    DATA_CORRUPTION = "data_corruption"          # Checksum mismatch
    
    # Unknown
    UNKNOWN = "unknown"


class ErrorContext:
    """Rich error context for debugging and recovery."""
    
    def __init__(
        self,
        category: ErrorCategory,
        message: str,
        step: str,
        exception: Optional[Exception] = None,
        retry_count: int = 0,
        max_retries: int = 3,
    ):
        self.category = category
        self.message = message
        self.step = step
        self.exception = exception
        self.retry_count = retry_count
        self.max_retries = max_retries
        self.timestamp = datetime.utcnow()
        self.retryable = self._is_retryable()
        self.backoff_seconds = self._calculate_backoff()
    
    def _is_retryable(self) -> bool:
        """Determine if error is retryable."""
        retryable_categories = {
            ErrorCategory.NETWORK_TIMEOUT,
            ErrorCategory.CONNECTION_RESET,
            ErrorCategory.TEMPORARY_UNAVAILABLE,
            ErrorCategory.LOCK_CONTENTION,
            ErrorCategory.RATE_LIMITED,
            ErrorCategory.RESOURCE_EXHAUSTED,
        }
        return self.category in retryable_categories
    
    def _calculate_backoff(self) -> float:
        """Calculate exponential backoff for retry."""
        if not self.retryable:
            return 0
        # 0.5s, 1.0s, 2.0s base, then add jitter
        base_delay = min(0.5 * (2 ** self.retry_count), 10)
        jitter = np.random.uniform(0, base_delay * 0.1)
        return base_delay + jitter
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for serialization."""
        return {
            'category': self.category.value,
            'message': self.message,
            'step': self.step,
            'retryable': self.retryable,
            'retry_count': self.retry_count,
            'max_retries': self.max_retries,
            'backoff_seconds': self.backoff_seconds,
        }


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class DataSourceRequest:
    """Request parameters for data fetching."""
    source: str  # 'mt5', 'database', 'csv'
    symbol: str
    timeframe: str
    
    # MT5 params
    count: Optional[int] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    method: str = "copy_rates_from"
    
    # Database params
    limit: Optional[int] = None
    page: Optional[int] = None
    min_close: Optional[float] = None
    max_close: Optional[float] = None
    min_volume: Optional[float] = None
    max_volume: Optional[float] = None
    
    # CSV params
    df: Optional[pd.DataFrame] = None


@dataclass
class AnalysisRequest:
    """Request for complete pipeline (data fetch + analysis)."""
    data_source: DataSourceRequest
    analysis_steps: List[str] = None  # ['technical', 'snr', 'astronomical']
    analysis_configs: Optional[Dict[str, Any]] = None
    user_id: Optional[str] = None
    skip_duplicate_check: bool = False


# ============================================================================
# SESSION DATA CACHE
# ============================================================================

class SessionDataCache:
    """Wrapper for cached session data with expiration tracking."""
    
    def __init__(self, data: Union[List[Dict], 'pd.DataFrame'], source_step: str, ttl_seconds: int = 1800):
        """
        Args:
            data: List of dicts or DataFrame (session data records)
            source_step: Step that produced this data ('data_source', 'technical_analysis', etc.)
            ttl_seconds: Time-to-live in seconds (default 30 minutes)
        """
        self.data = data
        self.source_step = source_step
        self.cached_at = time.time()
        self.ttl_seconds = ttl_seconds
        # ⚠️ CRITICAL: Use isinstance() — never `if data` — because DataFrames raise
        # "The truth value of a DataFrame is ambiguous" when evaluated as bool.
        if isinstance(data, pd.DataFrame):
            self.n_rows = len(data)
        elif isinstance(data, (list, tuple)):
            self.n_rows = len(data)
        else:
            self.n_rows = 0
    
    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        elapsed = time.time() - self.cached_at
        return elapsed > self.ttl_seconds
    
    def get_age_seconds(self) -> float:
        """Get age of cached data in seconds."""
        return time.time() - self.cached_at


class SessionMetadataBuilder:
    """Build complete DataSession metadata from fetched data."""
    
    @staticmethod
    def build(
        source: str,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
        record_count: int,
        data_checksum: Optional[str] = None,
        session_id: Optional[str] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Build complete DataSession metadata.
        
        Args:
            source: Data source ('mt5', 'database', 'csv')
            symbol: Trading symbol
            timeframe: Timeframe
            start_date: ISO 8601 start date (from fetcher)
            end_date: ISO 8601 end date (from fetcher)
            record_count: Number of records
            data_checksum: Optional SHA-256 checksum
            session_id: Optional UUID (or generate new)
            name: Optional user-friendly session name
            description: Optional session description
            
        Returns:
            Complete metadata dict ready for DataSession creation
        """
        final_session_id = session_id or str(uuid.uuid4())
        
        # Generate default name if not provided
        if not name:
            # Format: "EURUSD H1 - Dec 15, 2024"
            try:
                start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
                formatted_date = start_dt.strftime("%b %d, %Y")
                name = f"{symbol} {timeframe} - {formatted_date}"
            except:
                name = f"{symbol} {timeframe} Analysis"
        
        return {
            'session_id': final_session_id,
            'name': name,
            'description': description or "",
            'data_source': source,
            'symbol': symbol,
            'timeframe': timeframe,
            'start_date': start_date,  
            'end_date': end_date,    
            'record_count': record_count,
            'data_checksum': data_checksum,
            'status': 'active',
            'created_at': datetime.utcnow().isoformat(),
        }

