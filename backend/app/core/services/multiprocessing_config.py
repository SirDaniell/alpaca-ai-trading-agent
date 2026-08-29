"""
Multiprocessing Configuration - Safe Initialization

Handles safe 'spawn' method initialization for multiprocessing to prevent
deadlocks with other libraries (sklearn KMeans, joblib, loky, etc.).

CRITICAL: This must be imported BEFORE any multiprocessing.Pool creation.
Strategy: Set spawn method ONCE at module import, never override after.

Why 'spawn':
- 'fork': Child processes inherit parent's thread pools → deadlock on first KMeans call
- 'spawn': Fresh processes with no inherited state (safe, but slower startup)
- 'forkserver': Not available on macOS/Windows

Usage:
    from app.core.services.multiprocessing_config import init_spawn_method
    
    class ProcessingManager:
        def __init__(self):
            init_spawn_method()  # Call before Pool creation
"""

import multiprocessing
import logging
import threading
import os

# --- THREAD LIMITING: CRITICAL for CPU Throttling ---
# Prevent scientific libraries from spawning their own thread pools
# which multiplies the CPU usage and ignores the worker limit.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
# Ensure OpenBLAS uses 1 thread specifically for numpy
os.environ["OPENBLAS_MAIN_FREE"] = "1"


logger = logging.getLogger(__name__)

# Module-level lock to ensure thread-safe initialization
_init_lock = threading.Lock()
_initialized = False


def init_spawn_method() -> None:
    """
    Initialize multiprocessing.get_start_method to 'spawn' (safe for forked threads).
    
    Safe guard against:
    1. sklearn KMeans using joblib with loky threads
    2. Multiple libraries each trying to set the method
    3. Deadlocks from inherited thread pools in child processes
    
    IDEMPOTENT: Safe to call multiple times (only sets once).
    """
    global _initialized
    
    if _initialized:
        return  # Already initialized, no-op
    
    with _init_lock:
        if _initialized:
            return  # Double-check after acquiring lock
        
        try:
            current_method = multiprocessing.get_start_method(allow_none=True)
            
            if current_method is None:
                # Method not set yet, safe to set it
                multiprocessing.set_start_method('spawn', force=False)
                logger.info("✅ Multiprocessing start method set to 'spawn' (fresh processes)")
                _initialized = True
            elif current_method == 'spawn':
                # Already spawn, good!
                logger.debug("✓ Multiprocessing already using 'spawn' method")
                _initialized = True
            else:
                # Method already set to something else (fork/forkserver)
                logger.warning(
                    f"⚠️ Multiprocessing already using '{current_method}' method. "
                    f"Cannot override after initialization. "
                    f"This may cause deadlocks with sklearn KMeans. "
                    f"Consider setting environment variable PYTHONMULTIPROCESSING_START_METHOD=spawn"
                )
                _initialized = True  # Mark as initialized even if not spawn
        except RuntimeError as e:
            # Already set by some earlier code - this is OK, just log and move on
            logger.warning(
                f"⚠️ Multiprocessing start method already set (possibly by earlier import): {e}. "
                f"Schema initialized, proceeding."
            )
            _initialized = True
        except Exception as e:
            logger.error(f"❌ Failed to initialize multiprocessing: {e}", exc_info=True)
            # Don't raise - let caller handle Pool creation failure if it happens
            _initialized = True


# Initialize on module import (runs once per Python process)
logger.debug("🔧 Multiprocessing config module loaded, initializing spawn method...")
init_spawn_method()
