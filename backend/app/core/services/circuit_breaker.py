"""
Circuit Breaker Pattern Implementation

Implements explicit state machine (CLOSED → OPEN → HALF_OPEN) for resilience.

States:
- CLOSED: Normal operation, allowing requests
- OPEN: Rejecting requests after threshold failures, external service likely down
- HALF_OPEN: Testing if service recovered, accepting limited requests

This prevents cascading failures and allows graceful degradation.
"""

from enum import Enum
from datetime import datetime, timedelta
import asyncio
import logging
from typing import Callable, Any, Optional

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"          # Normal operation
    OPEN = "open"              # Reject requests, external service likely down
    HALF_OPEN = "half_open"    # Testing if service recovered


class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior"""
    
    def __init__(
        self,
        failure_threshold: int = 5,      # Failures before opening
        success_threshold: int = 2,      # Successes before closing
        timeout_seconds: int = 60,       # Time in open state before half_open
        window_seconds: int = 30         # Time window for failure counting
    ):
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.timeout_seconds = timeout_seconds
        self.window_seconds = window_seconds


class CircuitBreakerOpen(Exception):
    """Raised when circuit breaker is open"""
    pass


class CircuitBreaker:
    """
    Explicit state machine for handling transient failures in backend services.
    
    Prevents cascading failures by:
    - Opening circuit after repeated failures
    - Rejecting requests with clear error
    - Testing recovery with HALF_OPEN state
    - Reopening if recovery fails
    """
    
    def __init__(self, name: str, config: CircuitBreakerConfig = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        
        # State machine
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.failure_times = []  # Track when failures occurred
        self.opened_at = None
        self.lock = asyncio.Lock()
        
        logger.info(
            f"🔵 Circuit breaker '{name}' created: "
            f"failure_threshold={self.config.failure_threshold}, "
            f"timeout={self.config.timeout_seconds}s"
        )
    
    async def call(
        self,
        fn: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute function with circuit breaker protection.
        
        Args:
            fn: Async function to execute
            *args: Positional arguments for function
            **kwargs: Keyword arguments for function
            
        Returns:
            Result from function
            
        Raises:
            CircuitBreakerOpen: When circuit is open
            Any exception from wrapped function
        """
        
        # Check state and update if needed
        async with self.lock:
            # Check for timeout if in OPEN state
            if self.state == CircuitState.OPEN:
                time_since_open = (datetime.now(timezone.utc) - self.opened_at).total_seconds()
                
                if time_since_open > self.config.timeout_seconds:
                    # Try HALF_OPEN state
                    logger.info(
                        f"🟡 '{self.name}': OPEN → HALF_OPEN (testing recovery)"
                    )
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                
                else:
                    # Still in open state
                    remaining = self.config.timeout_seconds - time_since_open
                    logger.warning(
                        f"🔴 '{self.name}': Circuit OPEN, "
                        f"retry after {remaining:.1f}s"
                    )
                    raise CircuitBreakerOpen(
                        f"Circuit breaker open for '{self.name}', "
                        f"retry after {remaining:.1f} seconds"
                    )
        
        # Execute function outside of lock
        try:
            result = await fn(*args, **kwargs)
            
            # Success - update state
            async with self.lock:
                if self.state == CircuitState.HALF_OPEN:
                    self.success_count += 1
                    logger.info(
                        f"🟡 '{self.name}': HALF_OPEN success {self.success_count}/"
                        f"{self.config.success_threshold}"
                    )
                    
                    if self.success_count >= self.config.success_threshold:
                        logger.info(
                            f"🟢 '{self.name}': HALF_OPEN → CLOSED (recovered)"
                        )
                        self.state = CircuitState.CLOSED
                        self.failure_count = 0
                        self.success_count = 0
                        self.failure_times = []
                
                else:
                    # Normal operation, reset failure count
                    self.failure_count = 0
                    self.failure_times = []
            
            return result
        
        except Exception as e:
            # Failure - update state
            async with self.lock:
                current_time = datetime.now(timezone.utc)
                self.failure_times.append(current_time)
                
                # Remove old failures outside the window
                cutoff = current_time - timedelta(seconds=self.config.window_seconds)
                self.failure_times = [t for t in self.failure_times if t > cutoff]
                
                self.failure_count = len(self.failure_times)
                
                logger.warning(
                    f"⚠️ '{self.name}': Failure {self.failure_count}/"
                    f"{self.config.failure_threshold}"
                )
                
                if self.failure_count >= self.config.failure_threshold:
                    logger.error(
                        f"🔴 '{self.name}': {self.failure_count} failures, "
                        f"CLOSED → OPEN"
                    )
                    self.state = CircuitState.OPEN
                    self.opened_at = datetime.now(timezone.utc)
            
            raise
    
    def get_status(self) -> dict:
        """
        Get current circuit breaker status.
        
        Returns:
            Status dict with state, failure count, etc.
        """
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "opened_at": self.opened_at.isoformat() if self.opened_at else None,
            "config": {
                "failure_threshold": self.config.failure_threshold,
                "success_threshold": self.config.success_threshold,
                "timeout_seconds": self.config.timeout_seconds,
                "window_seconds": self.config.window_seconds
            }
        }
    
    def record_failure(self) -> None:
        """
        Manually record a failure for the circuit breaker.
        Used when failures occur outside the call() context.
        
        Opens circuit when failure threshold is reached.
        """
        try:
            current_time = datetime.now(timezone.utc)
            self.failure_times.append(current_time)
            
            # Remove old failures outside the window
            cutoff = current_time - timedelta(seconds=self.config.window_seconds)
            self.failure_times = [t for t in self.failure_times if t > cutoff]
            
            self.failure_count = len(self.failure_times)
            
            logger.warning(
                f"⚠️ '{self.name}': Manual failure recorded {self.failure_count}/"
                f"{self.config.failure_threshold}"
            )
            
            if self.failure_count >= self.config.failure_threshold:
                logger.error(
                    f"🔴 '{self.name}': {self.failure_count} failures, "
                    f"CLOSED → OPEN"
                )
                self.state = CircuitState.OPEN
                self.opened_at = datetime.now(timezone.utc)
        except Exception as e:
            logger.error(f"Error recording failure for '{self.name}': {e}")
    
    def record_success(self) -> None:
        """
        Manually record a success for the circuit breaker.
        Used when successes occur outside the call() context.
        
        Transitions HALF_OPEN to CLOSED when success threshold reached.
        """
        try:
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                logger.info(
                    f"🟡 '{self.name}': Manual success {self.success_count}/"
                    f"{self.config.success_threshold}"
                )
                
                if self.success_count >= self.config.success_threshold:
                    logger.info(
                        f"🟢 '{self.name}': HALF_OPEN → CLOSED (recovered)"
                    )
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                    self.success_count = 0
                    self.failure_times = []
            
            elif self.state == CircuitState.CLOSED:
                # Reset failure count on success during normal operation
                self.failure_count = 0
                self.failure_times = []
        except Exception as e:
            logger.error(f"Error recording success for '{self.name}': {e}")
            
    def call_allowed(self) -> bool:
        """
        Check if a call is allowed based on the current state.
        
        Returns:
            bool: True if call is allowed (CLOSED or HALF_OPEN states), False otherwise.
        """
        if self.state == CircuitState.OPEN:
            time_since_open = (datetime.now(timezone.utc) - self.opened_at).total_seconds()
            if time_since_open > self.config.timeout_seconds:
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
                return True
            return False
        return True
