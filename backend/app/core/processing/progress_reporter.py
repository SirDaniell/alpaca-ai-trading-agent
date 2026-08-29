"""
Unified Progress Reporter for All Analysis Steps

Single source of truth for progress updates with intelligent throttling.
Replaces ad-hoc progress_store.update_task() calls throughout the codebase.

Architecture:
- ProgressEvent: Structured event object
- ProgressReporter: Main reporter class with throttling
- Throttling: Intelligent 500ms OR 5% delta logic
- Resilience: Graceful failure handling for WebSocket issues
"""

import logging
import time
import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


# ============================================================================
# THROTTLING STRATEGY
# ============================================================================

class ThrottlingStrategy(Enum):
    """Different throttling approaches"""
    TIME_BASED = "time"          # Minimum time between updates
    PROGRESS_BASED = "progress"  # Minimum progress delta
    HYBRID = "hybrid"            # Both time and progress (stricter)
    NONE = "none"                # No throttling


# ============================================================================
# PROGRESS EVENT
# ============================================================================

@dataclass
class ProgressEvent:
    """
    Structured progress update event.
    
    Consolidates all progress information in a single object for unified handling.
    """
    
    task_id: str
    progress: int  # 0-100
    message: str
    stage: str  # "start", "processing", "validating", "saving", "end", "error"
    message2: str = ""  # ✅ FIXED: Default values must follow non-defaults
    
    # Metrics
    current: int = 0  # Items processed
    total: int = 100  # Total items
    
    # ✅ NEW: Technical Analysis specific fields
    processed_bars: int = 0  # Bars/rows processed so far
    total_bars: int = 0  # Total bars/rows in dataset
    current_indicator: str = ""  # Current indicator being calculated (e.g., "SMA_50")
    strategy: str = ""  # Processing strategy (Sequential/Parallel/Streaming)
    
    # Metadata
    extra: Optional[Dict[str, Any]] = field(default_factory=dict)
    timestamp: Optional[float] = None
    
    def __post_init__(self):
        """Validate and initialize"""
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc).timestamp()
        
        # Clamp progress to 0-100
        self.progress = round(max(0, min(100, self.progress)), 2)
        
        # Validate stage
        valid_stages = ["start", "processing", "validating", "saving", "end", "error"]
        if self.stage not in valid_stages:
            raise ValueError(f"Invalid stage: {self.stage}. Expected one of: {valid_stages}")
        
        # Ensure extra is dict
        if self.extra is None:
            self.extra = {}
    
    @property
    def is_complete(self) -> bool:
        """Whether this event represents completion"""
        return self.progress == 100 or self.stage in ["end", "error"]
    
    def to_dict(self, include_extra: bool = True) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = {
            "task_id": self.task_id,
            "progress": self.progress,
            "message": self.message,
            "message2": self.message2,  # ✅ Included in dict conversion
            "stage": self.stage,
            "status": self.stage,  # ✅ Map stage to status for frontend discovery
            "current": self.current,
            "total": self.total,
            "timestamp": self.timestamp,
            # ✅ NEW: Technical Analysis fields
            "processed_bars": self.processed_bars,
            "total_bars": self.total_bars,
            "current_indicator": self.current_indicator,
            "strategy": self.strategy,
        }
        
        if include_extra and self.extra:
            result.update(self.extra)
        
        return result


# ============================================================================
# THROTTLER
# ============================================================================

class ProgressThrottler:
    """
    Intelligent throttling for progress updates.
    
    Balances:
    - Real-time responsiveness (updates flow naturally)
    - Load reduction (fewer WebSocket messages)
    - Visibility (no missing updates during long phases)
    
    Strategy HYBRID (recommended):
    - Wait 500ms AND progress changes by 5%
    - Both conditions must be satisfied
    - Always broadcasts 0% and 100%
    """
    
    def __init__(
        self,
        strategy: ThrottlingStrategy = ThrottlingStrategy.HYBRID,
        min_time_ms: int = 500,      # Minimum 500ms between updates
        min_progress_delta: int = 5,  # Minimum 5% progress change
    ):
        self.strategy = strategy
        self.min_time_ms = min_time_ms
        self.min_progress_delta = min_progress_delta
        
        # Tracking per task
        self.last_broadcast_time: Dict[str, float] = {}
        self.last_broadcast_progress: Dict[str, int] = {}
        self.max_progress_seen: Dict[str, int] = {}  # ✅ NEW: Track max progress for monotonic enforcement
    
    def should_broadcast(self, task_id: str, progress: int) -> bool:
        """
        Determine if progress should be broadcast to WebSocket.
        
        Always broadcast boundaries (0%, 100%).
        For intermediate: check time/progress constraints.
        
        Args:
            task_id: Unique task identifier
            progress: Current progress (0-100)
            
        Returns:
            bool: True if should broadcast, False if throttled
        """
        
        # ✅ SAFETY: Handle None progress
        if progress is None:
            progress = 0
        
        # ✅ MONOTONIC: Never go backwards - silently clamp to max seen
        max_seen = self.max_progress_seen.get(task_id, 0)
        if progress < max_seen:
            return False  # Silently drop regressive progress
        
        # Track max seen regardless of whether we broadcast
        self.max_progress_seen[task_id] = max(max_seen, progress)
        
        # Always broadcast completion and start
        if progress in [0, 100]:
            return True
        
        # Get previous state
        now = time.time() * 1000  # milliseconds
        last_time = self.last_broadcast_time.get(task_id, 0)
        last_progress = self.last_broadcast_progress.get(task_id, 0)
        
        time_elapsed = now - last_time
        progress_delta = abs(progress - last_progress)
        
        if self.strategy == ThrottlingStrategy.TIME_BASED:
            return time_elapsed >= self.min_time_ms
        
        elif self.strategy == ThrottlingStrategy.PROGRESS_BASED:
            return progress_delta >= self.min_progress_delta
        
        elif self.strategy == ThrottlingStrategy.HYBRID:
            # EITHER condition met (more responsive for fast tasks)
            time_ok = time_elapsed >= self.min_time_ms
            progress_ok = progress_delta >= self.min_progress_delta
            return time_ok or progress_ok
        
        else:  # NONE
            return True
    
    def record_broadcast(self, task_id: str, progress: int) -> None:
        """Record that progress was broadcast"""
        now = time.time() * 1000
        self.last_broadcast_time[task_id] = now
        self.last_broadcast_progress[task_id] = progress
        # ✅ Track max for monotonic enforcement
        self.max_progress_seen[task_id] = max(self.max_progress_seen.get(task_id, 0), progress)
    
    def cleanup(self, task_id: str) -> None:
        """Clean up tracking for completed task"""
        self.last_broadcast_time.pop(task_id, None)
        self.last_broadcast_progress.pop(task_id, None)
        self.max_progress_seen.pop(task_id, None)


# ============================================================================
# UNIFIED PROGRESS REPORTER
# ============================================================================

class ProgressReporter:
    """
    ✅ UNIFIED: Single entry point for all progress updates across all analysis steps.
    
    Responsibilities:
    1. Consolidate all progress updates into single interface
    2. Update task store (local state)
    3. Emit WebSocket messages (frontend UI)
    4. Apply intelligent throttling (load management)
    5. Handle failures gracefully (resilience)
    6. Track metrics (monitoring/debugging)
    
    Usage:
        reporter = ProgressReporter(
            task_store=task_store,
            connection_manager=ws_manager,
            user_id="user123",
            throttling_strategy=ThrottlingStrategy.HYBRID
        )
        
        await reporter.update(ProgressEvent(
            task_id=task_id,
            progress=25,
            message="Processing...",
            stage="processing",
            current=250,
            total=1000,
            extra={"signals": 5}
        ))
    """
    
    def __init__(
        self,
        # BACKWARD COMPATIBILITY: Old signature support (positional args)
        task_id: Optional[str] = None,
        progress_store: Optional[Any] = None,
        slice_context: Optional[Dict[str, Any]] = None,
        # NEW: Keyword-only arguments
        task_store: Optional[Any] = None,
        connection_manager: Optional[Any] = None,
        user_id: str = "unknown",
        throttling_strategy: ThrottlingStrategy = ThrottlingStrategy.NONE,
    ):
        """
        ✅ UNIFIED: Single entry point for all progress updates across all analysis steps.
        
        Supports both old and new calling patterns:
        - Old: ProgressReporter(task_id, progress_store, slice_context)
        - New: ProgressReporter(task_store=store, connection_manager=mgr, user_id="user")
        
        Args:
            task_id: Task identifier (old pattern)
            progress_store: Progress store (old pattern, mapped to task_store if provided)
            slice_context: Slice context (old pattern, contains chunk_id/total_chunks for auto-scaling)
            task_store: Task store for local state (new pattern)
            connection_manager: WebSocket manager (new pattern)
            user_id: User identifier (new pattern)
            throttling_strategy: Throttling approach (new pattern)
        """
        self.logger = logger  # 🔒 INITIALIZE IMMEDIATELY
        
        # ✅ Robust task_id prioritization
        # Check positional first, then keyword, then context
        self.task_id = task_id or (slice_context.get("task_id") if isinstance(slice_context, dict) else None)
        
        # ✅ Store slice_context for automatic progress scaling
        self._slice_context = slice_context if isinstance(slice_context, dict) else None
        
        # Handle backward compatibility: Mapping progress_store to task_store if positional
        if progress_store is not None:
            task_store = progress_store
        
        # ✅ FALLBACK: Only resolve manager if we are in an async context (main process)
        if connection_manager is None:
            try:
                # Check if loop is running before attempting fallback
                asyncio.get_running_loop()
                from app.core.services.websocket_manager import manager as global_manager
                connection_manager = global_manager
                self.logger.debug(f"[ProgressReporter] Resolved fallback global connection_manager for task {self.task_id}")
            except (ImportError, RuntimeError, Exception):
                # RuntimeError means no loop is running (expected in workers)
                pass
        
        self.task_store = task_store
        self.connection_manager = connection_manager
        
        # ✅ Unified user_id extraction (supports positional slice_context from workers)
        self.user_id = user_id
        if self.user_id == "unknown" and isinstance(slice_context, dict):
            self.user_id = slice_context.get("user_id", "unknown")
        
        # ✅ Thread Safety: Capture loop at creation time
        try:
            self.loop = asyncio.get_running_loop()
        except RuntimeError:
            self.loop = None
        
        # Log status (DEBUG for workers, WARNING only for main process)
        if not self.connection_manager:
            if self.loop and self.loop.is_running():
                # We are in the main process but have no manager
                self.logger.warning(
                    f"⚠️ [ProgressReporter] Created without connection_manager for user '{self.user_id}'. "
                    "WebSocket progress updates will be DISABLED."
                )
            else:
                # We are in a worker process (no loop) - this is normal
                self.logger.debug(f"[ProgressReporter] Worker initialized for user '{self.user_id}' (Proxy mode)")
        
        # Throttling
        self.throttler = ProgressThrottler(throttling_strategy)
        
        # Metrics
        self.total_updates_received = 0  # All updates
        self.total_updates_broadcast = 0  # Updates actually sent to WebSocket
        self.total_errors = 0
        self.total_throttled = 0

    def __getstate__(self):
        """
        ✅ PRODUCTION: Pickling support for cross-process transmission.
        Drops process-locked resources (connection_manager, loop, logger) before serialization.
        """
        state = self.__dict__.copy()
        # These objects cannot be pickled as they are tied to the main process memory/thread
        state['connection_manager'] = None
        state['loop'] = None
        state['logger'] = None
        return state

    def __setstate__(self, state):
        """
        ✅ PRODUCTION: Restoration after cross-process transmission.
        Re-initializes process-safe resources in the worker process.
        """
        self.__dict__.update(state)
        # Re-initialize logger for the new process
        self.logger = logging.getLogger(__name__)
        # Ensure loop is None in workers (handled by __init__ logic too, but good to be explicit)
        self.loop = None
        self.logger.debug(f"[ProgressReporter] Restored serializable state for task {self.task_id} in worker")
    
    async def update(self, event: ProgressEvent) -> bool:
        """
        ✅ SINGLE UPDATE METHOD: Process a progress event through unified pipeline.
        
        Processing pipeline:
        1. Validate event structure
        2. Update task store (always)
        3. Check throttling logic
        4. Broadcast to WebSocket (if not throttled)
        5. Record metrics and cleanup
        
        Args:
            event: ProgressEvent with all update information
            
        Returns:
            bool: True if broadcast occurred, False if throttled or failed
        """
        self.total_updates_received += 1
        
        try:
            # 1. Validate event
            if event.progress < 0 or event.progress > 100:
                raise ValueError(f"Invalid progress: {event.progress}, expected 0-100")
            
            # 2. Always update task store (local state persists even if WebSocket fails)
            self._update_task_store(event)
            
            # 3. Check if should broadcast based on throttling strategy
            should_broadcast = self.throttler.should_broadcast(event.task_id, event.progress)
            
            if not should_broadcast:
                self.total_throttled += 1
                self.logger.debug(
                    f"[ProgressReporter] Throttled {event.task_id}: {event.progress}% "
                    f"(last: {self.throttler.last_broadcast_progress.get(event.task_id, 0)}%)"
                )
                return False
            
            # 4. Broadcast to WebSocket (with timeout and error handling)
            broadcast_success = await self._broadcast_to_websocket(event)
            
            # 5. Record broadcast if successful
            if broadcast_success:
                self.throttler.record_broadcast(event.task_id, event.progress)
                self.total_updates_broadcast += 1
                self.logger.debug(f"[ProgressReporter] Broadcasted {event.task_id}: {event.progress}% ✓")
                return True
            else:
                self.logger.warning(
                    f"[ProgressReporter] Broadcast failed for {event.task_id} {event.progress}%"
                )
                return False
        
        except Exception as err:
            self.total_errors += 1
            self.logger.error(f"[ProgressReporter] Update failed: {err}", exc_info=False)
            return False
        
        finally:
            # Clean up completed tasks from throttler
            if event.is_complete:
                self.throttler.cleanup(event.task_id)
    
    def report(self, progress: int, message: str = "", message2: str = "", **kwargs) -> None:
        """
        ✅ BACKWARD COMPATIBILITY: Legacy report() method for sync handlers.
        Wraps update_sync() for existing code that calls reporter.report().
        
        ✅ AUTOMATIC PROGRESS SCALING: If slice_context is provided, automatically
        scales progress to account for chunking/slicing:
        - Chunk 1 of 4: progress 0-100 maps to global 0-25%
        - Chunk 2 of 4: progress 0-100 maps to global 25-50%
        - etc.
        
        Args:
            progress: Progress percentage (0-100) within current chunk/slice
            message: Primary message (short, ≤30 chars)
            message2: Secondary details message (technical context)
            **kwargs: Additional metadata (processed_bars, total_bars, current_indicator, strategy)
        """
        if not self.task_id:
            return
        
        # ✅ SAFETY: Ensure progress is not None
        if progress is None:
            if self.logger:
                self.logger.warning(f"[ProgressReporter] progress is None for task {self.task_id}, defaulting to 0")
            progress = 0
        
        # ✅ AUTO-SCALE: Calculate global progress if in chunk/slice context
        scaled_progress = progress
        if hasattr(self, '_slice_context') and self._slice_context:
            chunk_id = self._slice_context.get('chunk_id', 0)
            total_chunks = self._slice_context.get('total_chunks', 1)
            
            if total_chunks > 1:
                # Calculate this chunk's progress range
                chunk_size_pct = 100.0 / total_chunks
                chunk_base = chunk_id * chunk_size_pct
                
                # Scale local progress (0-100) to chunk's allocated range
                scaled_progress = chunk_base + (progress * chunk_size_pct / 100.0)
                scaled_progress = int(scaled_progress)
                
                # Debug log for verification
                if progress in [0, 50, 100]:  # Log at key milestones
                    if self.logger:
                        self.logger.debug(
                            f"[Progress] Chunk {chunk_id+1}/{total_chunks}: "
                            f"local {progress}% → global {scaled_progress}%"
                        )
        
        # ✅ Extract technical analysis fields from kwargs
        processed_bars = kwargs.pop('processed_bars', 0)
        total_bars = kwargs.pop('total_bars', 0)
        current_indicator = kwargs.pop('current_indicator', '')
        strategy = kwargs.pop('strategy', '')
        
        self.update_sync(ProgressEvent(
            task_id=self.task_id,
            progress=scaled_progress,  # ✅ Use scaled progress
            message=message,
            message2=message2,
            stage="processing",
            processed_bars=processed_bars,
            total_bars=total_bars,
            current_indicator=current_indicator,
            strategy=strategy,
            extra=kwargs
        ))
    
    async def report_async(self, progress: float, message: str = "", message2: str = "", **kwargs) -> None:
        """
        ✅ ASYNC HELPER: Like report() but awaitable for async handlers.
        """
        if not self.task_id:
            self.logger.error(
                f"❌ [ProgressReporter] report_async() called without task_id! "
                f"Progress update DROPPED: {progress}% - {message}. "
                f"This indicates ProgressReporter was not initialized with task_id."
            )
            return
        
        await self.update(ProgressEvent(
            task_id=self.task_id,
            progress=progress,
            message=message,      # ✅ Use direct message parameter
            message2=message2,    # ✅ Use direct message2 parameter
            stage="processing",
            extra=kwargs
        ))
    
    def check_cancellation(self) -> None:
        """
        ✅ BACKWARD COMPATIBILITY: Check if task was cancelled.
        No-op in ProgressReporter (task store handles this).
        """
        pass
    
    def report_loop(self, current: int, total: int, message: str = "", message2: str = "", 
                   base_progress: float = 0.0, progress_range: float = 10.0, **kwargs) -> None:
        """
        ✅ RESTORED: Loop reporting - keeps long-running calculations sending updates.
        
        Used during intensive loops (volume analysis, indicator calculation, etc.) to:
        1. Prevent WebSocket timeout during long operations
        2. Provide per-iteration feedback
        3. Keep circuit breaker alive
        
        This is REQUIRED for infrastructure stability during 1000+ iteration loops.
        Without these periodic updates, the connection times out and handler fails.
        
        Args:
            current: Current iteration number
            total: Total iterations
            message: Base update message
            message2: Template with {current} and {total} placeholders
            base_progress: Base progress value (when part of larger operation)
            progress_range: Range of progress movement during this loop
            **kwargs: Additional metadata
        """
        if total <= 0:
            return
        
        # Calculate progress within the loop's allocated range
        pct_through_loop = current / total if total > 0 else 0.0
        loop_progress = base_progress + (pct_through_loop * progress_range)
        
        # Format message2 if template provided
        formatted_message2 = message2
        if "{current}" in formatted_message2:
            formatted_message2 = formatted_message2.replace("{current}", str(current))
            formatted_message2 = formatted_message2.replace("{total}", str(total))
        
        # ✅ Extract technical analysis fields from kwargs
        processed_bars = kwargs.pop('processed_bars', current)  # Default to current iteration
        total_bars = kwargs.pop('total_bars', total)  # Default to total iterations
        current_indicator = kwargs.pop('current_indicator', '')
        strategy = kwargs.pop('strategy', '')
        
        # Create and send event
        event = ProgressEvent(
            task_id=self.task_id,
            progress=int(max(0, min(100, loop_progress))),
            message=message or f"Processing {current}/{total}",
            message2=formatted_message2,
            stage="processing",
            current=current,
            total=total,
            processed_bars=processed_bars,
            total_bars=total_bars,
            current_indicator=current_indicator,
            strategy=strategy,
            extra=kwargs
        )
        
        self.update_sync(event)
    
    async def error(self, task_id: str, message: str) -> None:
        """
        ✅ ERROR REPORTING: Mark task as failed and broadcast error.
        """
        event = ProgressEvent(
            task_id=task_id,
            progress=0,
            message=message,
            stage="error"
        )
        await self.update(event)
    
    def _update_task_store(self, event: ProgressEvent) -> None:
        """
        Update task store with progress event information.
        
        Always updates (even if WebSocket fails), ensuring local state is accurate.
        """
        if not self.task_store:
            return
        
        update_dict = {
            "progress": event.progress,
            "message": event.message,
            "stage": event.stage,
            "current_index": event.current,
            "total": event.total,
        }
        
        if event.extra:
            # ✅ FIX: Remove task_id from extra to avoid duplicate parameter
            extra_filtered = {k: v for k, v in event.extra.items() if k != 'task_id'}
            update_dict.update(extra_filtered)
        
        try:
            self.task_store.update_task(event.task_id, **update_dict)
        except Exception as e:
            self.logger.error(f"[ProgressReporter] Failed to update task store: {e}")
    
    async def _broadcast_to_websocket(self, event: ProgressEvent) -> bool:
        """
        Broadcast progress event to WebSocket subscribers.
        
        Uses asyncio.wait_for with 5-second timeout (not fire-and-forget).
        Gracefully handles timeouts and connection errors.
        
        Args:
            event: ProgressEvent to broadcast
            
        Returns:
            bool: True if broadcast succeeded, False if timeout/error
        """
        if not self.connection_manager:
            return True  # No manager configured, "succeed" silently
        
        # 🔴 FIX: Nest extra data inside 'data' object for frontend unpacking
        # Frontend expects: msg.data.message2, msg.data.metrics, msg.data.snrMetrics
        # ✅ DYNAMIC TYPE DETECTION: Frontend relies on 'type' to hide progress bars
        msg_type = "progress"
        if event.stage == "error":
            msg_type = "error"
        elif event.is_complete:
            msg_type = "complete"

        payload = {
            "type": msg_type,
            "progress": event.progress,
            "message": event.message,
            "message2": event.message2,  # ✅ Top-level first-class citizen
            "stage": event.stage,
            "status": event.stage,  # Map stage to status for frontend discovery
            "current": event.current,
            "total": event.total,
            "timestamp": event.timestamp,
            "user_id": self.user_id,
            "data": event.extra or {}  # Nest extra data for frontend unpacking
        }
        
        try:
            # ✅ Use wait_for with proper timeout (5s not 2s)
            # Timeout is acceptable for progress (processing can continue)
            await asyncio.wait_for(
                self.connection_manager.send_progress_update(
                    event.task_id,
                    payload,
                    user_id=self.user_id
                ),
                timeout=5.0
            )
            return True
        
        except asyncio.TimeoutError:
            # Timeout is acceptable for progress
            self.logger.debug(
                f"[ProgressReporter] WebSocket timeout for {event.task_id} "
                f"(progress={event.progress}%), continuing..."
            )
            return False  # Signal timeout occurred (for backpressure tracking)
        
        except asyncio.CancelledError:
            self.logger.warning(f"[ProgressReporter] WebSocket send cancelled for {event.task_id}")
            raise
        
        except Exception as e:
            # Other errors (connection closed, serialization error, etc.)
            self.total_errors += 1
            self.logger.error(
                f"[ProgressReporter] WebSocket send failed: {e}",
                exc_info=False
            )
            return False
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get reporter metrics for monitoring and debugging.
        
        Returns:
            dict: Statistics about progress reporting
        """
        broadcast_rate = (
            self.total_updates_broadcast / self.total_updates_received
            if self.total_updates_received > 0
            else 0
        )
        
        throttle_rate = (
            self.total_throttled / self.total_updates_received
            if self.total_updates_received > 0
            else 0
        )
        
        return {
            "total_updates_received": self.total_updates_received,
            "total_updates_broadcast": self.total_updates_broadcast,
            "total_updates_throttled": self.total_throttled,
            "broadcast_rate_pct": broadcast_rate * 100,
            "throttle_rate_pct": throttle_rate * 100,
            "total_errors": self.total_errors,
            "active_tasks": len(self.throttler.last_broadcast_time),
        }
    
    def update_sync(self, event: ProgressEvent) -> bool:
        """
        ✅ SYNCHRONOUS VERSION: For use in sync handlers (signal_generator, etc).
        
        Updates task store immediately (always) and emits async WebSocket broadcast.
        Does NOT await the WebSocket send - starts it in background.
        
        Args:
            event: ProgressEvent with update information
            
        Returns:
            bool: True if task store was updated
        """
        self.total_updates_received += 1
        
        try:
            # Validate event
            if event.progress < 0 or event.progress > 100:
                raise ValueError(f"Invalid progress: {event.progress}, expected 0-100")
            
            # 1. Always update task store immediately (synchronous)
            self._update_task_store(event)
            
            # 2. Check if should broadcast based on throttling
            should_broadcast = self.throttler.should_broadcast(event.task_id, event.progress)
            
            if not should_broadcast:
                self.total_throttled += 1
                self.logger.debug(
                    f"[ProgressReporter] Throttled {event.task_id}: {event.progress}% "
                    f"(last: {self.throttler.last_broadcast_progress.get(event.task_id, 0)}%)"
                )
                return False
            
            # 3. Emit async WebSocket broadcast (fire-and-forget, non-blocking)
            # This allows sync handlers to proceed immediately without waiting for WebSocket
            if self.connection_manager and self.loop and self.loop.is_running():
                # ✅ Schedule broadcast ONLY if loop is available
                try:
                    # Thread-safe call to schedule the broadcast task
                    self.loop.call_soon_threadsafe(
                        lambda: asyncio.create_task(self._broadcast_to_websocket(event))
                    )
                except Exception as e:
                    self.logger.debug(f"[ProgressReporter] Failed to schedule sync broadcast: {e}")
            elif self.connection_manager:
                # Log only once to avoid spamming
                self.logger.debug(f"[ProgressReporter] Skipping WebSocket broadcast for {event.task_id} (no running loop)")
            
            self.throttler.record_broadcast(event.task_id, event.progress)
            self.total_updates_broadcast += 1
            self.logger.debug(f"[ProgressReporter] Queued broadcast {event.task_id}: {event.progress}% ✓")
            return True
        
        except Exception as err:
            self.total_errors += 1
            self.logger.error(f"[ProgressReporter] Sync update failed: {err}", exc_info=False)
            return False
        
        finally:
            # Clean up completed tasks from throttler
            if event.is_complete:
                self.throttler.cleanup(event.task_id)
