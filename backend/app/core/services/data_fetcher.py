"""
DataFetcher: Unified data source handler for MT5, Database, and CSV.

Extracts data from any source and returns standardized format:
  (df, start_date, end_date, record_count)

Ensures all session metadata is available BEFORE session creation,
fixing the NULL constraint violation in data_sessions.start_date/end_date.

Usage:
    fetcher = DataFetcher(mt5_service=mt5_service)
    df, start_date, end_date, record_count = await fetcher.fetch(
        source='mt5',
        symbol='EURUSD',
        timeframe='D1',
        count=34000
    )
"""

import gc
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Tuple, Optional, List, Any, Dict

import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.mt5_service import MT5Service
from app.core.services.data_utils import normalize_dataframe_columns
from app.database.connection import DbConfig
from app.core.processing.progress_reporter import ProgressReporter

logger = logging.getLogger(__name__)


# ============================================================================
# DATA FETCHER BASE CLASS
# ============================================================================

class BaseFetcher(ABC):
    """Abstract base for all data fetchers."""
    
    @abstractmethod
    async def fetch(self, **kwargs) -> Tuple[pd.DataFrame, str, str, int]:
        """
        Fetch data from source.
        
        Returns:
            (df, start_date_iso, end_date_iso, record_count)
        """
        pass
    
    def _extract_dates_from_dataframe(
        self, 
        df: pd.DataFrame,
        date_column: str = 'Time'
    ) -> Tuple[str, str]:
        """
        Extract start/end dates from OHLC dataframe.
        
        Uses min/max instead of first/last for robustness against unordered data.
        Validates chronological order and warns if data is not sorted.
        
        Args:
            df: DataFrame with OHLC data
            date_column: Name of date column (usually 'Time')
            
        Returns:
            (start_date_iso, end_date_iso) as ISO 8601 strings
        """
        if df.empty:
            raise ValueError("Cannot extract dates from empty dataframe")
        
        if date_column not in df.columns:
            raise ValueError(f"Date column '{date_column}' not found in dataframe")
        
        # Convert to datetime for robust min/max extraction
        date_series = pd.to_datetime(df[date_column])
        
        # Use min/max instead of first/last (robust to unordered data)
        start_date = date_series.min()
        end_date = date_series.max()
        
        # Check if data is chronologically sorted (warn if not)
        is_sorted = date_series.is_monotonic_increasing or date_series.is_monotonic_decreasing
        if not is_sorted:
            logger.warning(
                f"⚠️ Date column '{date_column}' is not monotonically sorted. "
                f"Data may be out of order: {date_series.iloc[0]} → {date_series.iloc[-1]}"
            )
        
        # Convert to ISO 8601
        start_iso = start_date.isoformat()
        end_iso = end_date.isoformat()
        
        logger.info(f"Extracted date range: {start_iso} to {end_iso}")
        return start_iso, end_iso


# ============================================================================
# MT5 FETCHER
# ============================================================================

class MT5Fetcher(BaseFetcher):
    """Fetch OHLC data from MT5 terminal."""
    
    def __init__(self, mt5_service: MT5Service):
        self.mt5_service = mt5_service
    
    async def fetch(
        self,
        symbol: str,
        timeframe: str = "H1",
        count: Optional[int] = 1000,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        method: str = "copy_rates_from",
        reporter: Optional[ProgressReporter] = None,
    ) -> Tuple[pd.DataFrame, str, str, int]:
        """
        Fetch OHLC data from MT5.
        
        Args:
            symbol: Trading symbol (e.g., 'EURUSD')
            timeframe: Timeframe (e.g., 'D1', 'H1')
            count: Number of bars to fetch
            date_from: Start date (ISO format, optional)
            date_to: End date (ISO format, optional)
            method: Fetch method ('copy_rates_from', etc.)
            reporter: Optional progress reporter for status updates
            
        Returns:
            (df, start_date_iso, end_date_iso, record_count)
        """
        try:
            logger.info(f"MT5Fetcher: Fetching {symbol} {timeframe} (count={count})")
            
            if reporter:
                await reporter.report_async(
                    progress=20,
                    message="Requesting MT5 Data",
                    message2=f"Requesting {count:,} bars for {symbol} ({timeframe}) from MT5 terminal..."
                )

            params = {
                "symbol": symbol,
                "timeframe": timeframe,
                "method": method,
                "reporter": reporter,
            }
            
            # Add date range if provided
            if date_from and date_to:
                params["date_from"] = datetime.fromisoformat(
                    date_from.replace("Z", "+00:00")
                )
                params["date_to"] = datetime.fromisoformat(
                    date_to.replace("Z", "+00:00")
                )
            else:
                params["count"] = count or 1000
            
            # Fetch from MT5
            ohlc_data = await self.mt5_service.fetch_ohlc_data_v2(**params)
            
            # Handle error response
            if isinstance(ohlc_data, dict) and (
                ohlc_data.get("error") or ohlc_data.get("success") is False
            ):
                raise ValueError(
                    ohlc_data.get("error") or ohlc_data.get("message") or 
                    "Failed to retrieve OHLC data"
                )
            
            # Unwrap data if nested in 'data' key
            final_data = ohlc_data
            if isinstance(ohlc_data, dict) and "data" in ohlc_data:
                final_data = ohlc_data["data"]
            
            if not isinstance(final_data, list) or not final_data:
                raise ValueError(
                    f"Invalid MT5 response format: {type(ohlc_data)}"
                )
            
            # ✅ VALIDATION: Detect partial fetch (shortfall >20%)
            expected_count = count or 1000
            actual_count = len(final_data)
            shortfall_pct = ((expected_count - actual_count) / expected_count * 100) if expected_count > 0 else 0
            
            if actual_count < expected_count * 0.8:  # >20% shortfall
                logger.warning(
                    f"⚠️ MT5 Partial Fetch Detected: "
                    f"Expected {expected_count}, got {actual_count} ({shortfall_pct:.1f}% shortfall). "
                    f"This may indicate timeout or data completeness issue."
                )
            
            if reporter:
                await reporter.report_async(
                    progress=60,
                    message="Normalizing Data",
                    message2=f"Received {actual_count:,} records. Normalizing and formatting..."
                )

            # Convert to DataFrame
            df = pd.DataFrame(final_data)
            df = normalize_dataframe_columns(df)
            
            # ✅ Convert Time column from Unix seconds to datetime (same as old mt5.py endpoint)
            # This matches the frontend's expectation: timestamps as seconds, multiplied by 1000 for Date()
            if 'Time' in df.columns:
                if not pd.api.types.is_datetime64_any_dtype(df['Time']):
                    df['Time'] = pd.to_datetime(df['Time'], unit='s', errors='coerce')
            
            # Extract dates
            start_iso, end_iso = self._extract_dates_from_dataframe(df)
            record_count = len(df)
            
            logger.info(
                f"✅ MT5Fetcher: {record_count} bars, "
                f"{start_iso} to {end_iso}"
            )
            
            return df, start_iso, end_iso, record_count
            
        except Exception as e:
            # Extract bridge error detail when available so the log is self-describing
            bridge_detail = ""
            if hasattr(e, "response"):
                try:
                    body = e.response.json()
                    bridge_detail = f" | bridge: {body.get('detail') or body.get('error') or body}"
                except Exception:
                    bridge_detail = f" | bridge body: {getattr(e.response, 'text', '')}"
            logger.error(
                f"❌ MT5Fetcher error: symbol={symbol} timeframe={timeframe} "
                f"count={count}{bridge_detail} | {e}"
            )
            raise


# ============================================================================
# DATABASE FETCHER
# ============================================================================

class DatabaseFetcher(BaseFetcher):
    """Fetch OHLC data from database."""
    
    def __init__(self, db_config: DbConfig):
        self.db_config = db_config
    
    async def fetch(
        self,
        symbol: str,
        timeframe: str,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        limit: Optional[int] = None,
        page: Optional[int] = None,
        min_close: Optional[float] = None,
        max_close: Optional[float] = None,
        min_volume: Optional[float] = None,
        max_volume: Optional[float] = None,
        reporter: Optional[ProgressReporter] = None,
    ) -> Tuple[pd.DataFrame, str, str, int]:
        """
        Query OHLC data from database with optional filters and pagination.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            date_from: Start date (ISO format, optional)
            date_to: End date (ISO format, optional)
            limit: Max rows per page (page size)
            page: Page number for pagination (1-based)
            min_close: Minimum close price filter
            max_close: Maximum close price filter
            min_volume: Minimum volume filter
            max_volume: Maximum volume filter
            reporter: Optional progress reporter for status updates
            
        Returns:
            (df, start_date_iso, end_date_iso, record_count)
        """
        try:
            logger.info(
                f"DatabaseFetcher: Querying {symbol} {timeframe} "
                f"({date_from} to {date_to}) "
                f"filters: close[{min_close}-{max_close}] volume[{min_volume}-{max_volume}]"
            )
            
            if reporter:
                await reporter.report_async(
                    progress=30,
                    message="Querying Database",
                    message2=f"Executing SQL query for {symbol} ({timeframe}) in local database..."
                )

            from app.database.connection import get_db_connection
            
            async with get_db_connection(self.db_config) as conn:
                # Build parameterized query (prevents SQL injection)
                query_parts = [
                    "SELECT time, open, high, low, close, volume",
                    "FROM market_data_ohlcv",
                    "WHERE symbol = $1 AND timeframe = $2"
                ]
                params = [symbol, timeframe]
                param_index = 3
                
                # Add date range filters
                if date_from:
                    query_parts.append(f"AND time >= ${param_index}")
                    params.append(date_from)
                    param_index += 1
                
                if date_to:
                    query_parts.append(f"AND time <= ${param_index}")
                    params.append(date_to)
                    param_index += 1
                
                # Add price filters
                if min_close is not None:
                    query_parts.append(f"AND close >= ${param_index}")
                    params.append(min_close)
                    param_index += 1
                
                if max_close is not None:
                    query_parts.append(f"AND close <= ${param_index}")
                    params.append(max_close)
                    param_index += 1
                
                # Add volume filters
                if min_volume is not None:
                    query_parts.append(f"AND volume >= ${param_index}")
                    params.append(min_volume)
                    param_index += 1
                
                if max_volume is not None:
                    query_parts.append(f"AND volume <= ${param_index}")
                    params.append(max_volume)
                    param_index += 1
                
                query_parts.append("ORDER BY time ASC")
                
                # Add pagination
                page_num = max(1, page or 1)
                page_size = limit or 100000  # Default page size
                offset = (page_num - 1) * page_size
                
                query_parts.append(f"OFFSET ${param_index}")
                params.append(offset)
                param_index += 1
                
                query_parts.append(f"LIMIT ${param_index}")
                params.append(page_size)
                
                query = "\n".join(query_parts)
                
                logger.debug(f"Executing parameterized query with params: {params}")
                rows = await conn.fetch(query, *params)
                
                if not rows:
                    raise ValueError(
                        f"No data found for {symbol} {timeframe} with specified filters"
                    )
                
                if reporter:
                    await reporter.report_async(
                        progress=70,
                        message="Processing Results",
                        message2=f"Retrieved {len(rows):,} rows from database. Formatting..."
                    )

                # Convert to DataFrame
                df = pd.DataFrame([dict(row) for row in rows])
                df = normalize_dataframe_columns(df)
                
                # Extract dates
                start_iso, end_iso = self._extract_dates_from_dataframe(df)
                record_count = len(df)
                
                logger.info(
                    f"✅ DatabaseFetcher: {record_count} rows (page {page_num}), "
                    f"{start_iso} to {end_iso}"
                )
                
                return df, start_iso, end_iso, record_count
                
        except Exception as e:
            logger.error(f"❌ DatabaseFetcher error: {e}")
            raise


# ============================================================================
# CSV FETCHER
# ============================================================================

class CSVFetcher(BaseFetcher):
    """Fetch OHLC data from CSV file (already loaded as DataFrame)."""
    
    async def fetch(
        self,
        df: pd.DataFrame,
        reporter: Optional[ProgressReporter] = None,
        **kwargs
    ) -> Tuple[pd.DataFrame, str, str, int]:
        """
        Process CSV data (already loaded as DataFrame).
        
        Args:
            df: Already-loaded CSV data as DataFrame
            reporter: Optional progress reporter for status updates
            **kwargs: Additional parameters (ignored, for interface consistency)
            
        Returns:
            (df, start_date_iso, end_date_iso, record_count)
        """
        try:
            logger.info(f"CSVFetcher: Processing {len(df)} rows")
            
            if reporter:
                await reporter.report_async(
                    progress=40,
                    message="Processing CSV",
                    message2=f"Validating and normalizing {len(df):,} records from uploaded CSV..."
                )

            if df.empty:
                raise ValueError("CSV data is empty")
            
            # ✅ VALIDATION: Check for required columns BEFORE processing
            current_columns = set(df.columns)
            required_columns = {'time', 'open', 'high', 'low', 'close', 'volume'}
            
            # Also check for uppercase variants (before normalization)
            uppercase_variants = {'Time', 'Open', 'High', 'Low', 'Close', 'Volume'}
            
            has_required = (
                required_columns.issubset(current_columns.union({c.lower() for c in current_columns})) or
                uppercase_variants.issubset(current_columns)
            )
            
            if not has_required:
                missing = required_columns - {c.lower() for c in current_columns}
                raise ValueError(
                    f"CSV validation failed. Missing columns: {missing}. "
                    f"Found: {list(current_columns)}"
                )
            
            # Normalize columns
            df = normalize_dataframe_columns(df)
            
            # Extract dates (CSV should have 'Time' column after normalization)
            start_iso, end_iso = self._extract_dates_from_dataframe(df)
            record_count = len(df)
            
            logger.info(
                f"✅ CSVFetcher: {record_count} rows, "
                f"{start_iso} to {end_iso}"
            )
            
            return df, start_iso, end_iso, record_count
            
        except Exception as e:
            logger.error(f"❌ CSVFetcher error: {e}")
            raise


# ============================================================================
# DATA FETCHER FACTORY
# ============================================================================

class DataFetcher:
    """
    Unified interface for all data sources.
    
    Usage:
        fetcher = DataFetcher(mt5_service=mt5_service)
        df, start_date, end_date, record_count = await fetcher.fetch(
            source='mt5',
            symbol='EURUSD',
            timeframe='D1',
            count=34000
        )
    """
    
    def __init__(
        self,
        mt5_service: Optional[MT5Service] = None,
        db_config: Optional[DbConfig] = None,
    ):
        self._mt5_service = mt5_service
        self._db_config = db_config
        self._mt5_fetcher = MT5Fetcher(mt5_service) if mt5_service else None
        self._db_fetcher = DatabaseFetcher(db_config) if db_config else None
        self._csv_fetcher = CSVFetcher()
    
    @property
    def mt5_service(self) -> Optional[MT5Service]:
        return self._mt5_service
    
    @mt5_service.setter
    def mt5_service(self, value: Optional[MT5Service]):
        self._mt5_service = value
        if value:
            self._mt5_fetcher = MT5Fetcher(value)
        else:
            self._mt5_fetcher = None

    @property
    def db_config(self) -> Optional[DbConfig]:
        return self._db_config
    
    @db_config.setter
    def db_config(self, value: Optional[DbConfig]):
        self._db_config = value
        if value:
            self._db_fetcher = DatabaseFetcher(value)
        else:
            self._db_fetcher = None

    
    async def fetch(
        self,
        source: str,
        reporter: Optional[ProgressReporter] = None,
        **kwargs
    ) -> Tuple[pd.DataFrame, str, str, int]:
        """
        Fetch data from specified source.
        
        Args:
            source: 'mt5', 'database', or 'csv'
            reporter: Optional progress reporter for status updates
            **kwargs: Source-specific parameters
            
        Returns:
            (df, start_date_iso, end_date_iso, record_count)
            
        Raises:
            ValueError: If source not supported or fetch fails
        """
        try:
            if source == 'mt5':
                if not self._mt5_fetcher:
                    raise ValueError("MT5Service not initialized")
                return await self._mt5_fetcher.fetch(reporter=reporter, **kwargs)
            
            elif source == 'database':
                if not self._db_fetcher:
                    raise ValueError("DB config not initialized")
                return await self._db_fetcher.fetch(reporter=reporter, **kwargs)
            
            elif source == 'csv':
                return await self._csv_fetcher.fetch(reporter=reporter, **kwargs)
            
            else:
                raise ValueError(f"Unknown data source: {source}")
        
        finally:
            # Cleanup memory
            gc.collect()
    
    def cleanup(self):
        """Cleanup resources."""
        self._mt5_fetcher = None
        self._db_fetcher = None
        self._csv_fetcher = None
        gc.collect()
