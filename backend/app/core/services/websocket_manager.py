import json
import logging
import time
import asyncio
from typing import Dict, Set, List, Union, Any, Optional
from fastapi import WebSocket, WebSocketDisconnect
import uuid
from app.core.data.serializers import to_serializable

logger = logging.getLogger(__name__)


# ============================================================================
# CIRCUIT BREAKER PATTERN - Prevents cascade failures
# ============================================================================

class CircuitBreaker:
    """
    🔥 RESILIENCE FEATURE: Circuit breaker for WebSocket failures
    
    Prevents rapid-fire retry attempts that overwhelm the backend.
    After N failures within a timeout window, "opens" the circuit and
    rejects new sends until the service recovers.
    
    States:
    - CLOSED: Normal operation, all sends attempted
    - OPEN: Too many failures, sends are dropped
    - HALF_OPEN: Service recovering, test sends enabled
    """
    
    def __init__(self, failure_threshold: int = 10, timeout_seconds: int = 30, recovery_timeout_seconds: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.recovery_timeout_seconds = recovery_timeout_seconds
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = "CLOSED"  # CLOSED | OPEN | HALF_OPEN
        self.opened_at: Optional[float] = None
    
    def call_allowed(self) -> bool:
        """Check if a send operation should be attempted."""
        current_time = time.time()
        
        if self.state == "CLOSED":
            # Normal operation
            return True
        elif self.state == "OPEN":
            # Check if recovery window has passed
            if self.opened_at and (current_time - self.opened_at) > self.recovery_timeout_seconds:
                self.state = "HALF_OPEN"
                self.failure_count = 0
                logger.info("🔌 [CB] Circuit HALF_OPEN - testing recovery")
                return True
            return False
        elif self.state == "HALF_OPEN":
            # In recovery mode, allow attempts
            return True
        return False
    
    def record_success(self):
        """Record successful send."""
        if self.state == "HALF_OPEN":
            self.state = "CLOSED"
            self.failure_count = 0
            self.last_failure_time = None
            logger.info("✅ [CB] Circuit CLOSED - service recovered")
        elif self.state == "CLOSED":
            # Reset failure count on success
            self.failure_count = max(0, self.failure_count - 1)
    
    def record_failure(self):
        """Record failed send."""
        current_time = time.time()
        
        # Reset counter if outside timeout window
        if self.last_failure_time and (current_time - self.last_failure_time) > self.timeout_seconds:
            self.failure_count = 0
        
        self.failure_count += 1
        self.last_failure_time = current_time
        
        # If threshold exceeded, open circuit
        if self.failure_count >= self.failure_threshold:
            if self.state != "OPEN":
                self.state = "OPEN"
                self.opened_at = current_time
                logger.warning(f"🔴 [CB] Circuit OPEN - {self.failure_count} failures in {self.timeout_seconds}s")
    
    def get_state(self) -> Dict[str, Any]:
        """Get current circuit state for monitoring."""
        return {
            "state": self.state,
            "failure_count": self.failure_count,
            "since": self.last_failure_time or "never"
        }


class ConnectionManager:
    def __init__(self):
        # Store active connections: {user_id: {websocket1, websocket2, ...}}
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        # Store room subscriptions: {room_name: {user_id1, user_id2, ...}}
        self.rooms: Dict[str, Set[str]] = {}
        # Task subscription tracking
        self.task_connections: Dict[str, Set[str]] = {}  # task_id -> connection_ids
        self.connection_tasks: Dict[str, Set[str]] = {}  # connection_id -> task_ids
        # 🔥 RESILIENCE: Circuit breakers per user to prevent cascade failures
        self.user_circuit_breakers: Dict[str, CircuitBreaker] = {}

    async def connect(self, websocket: WebSocket, user_id: str):
        await websocket.accept()
        if user_id not in self.active_connections:
            self.active_connections[user_id] = set()
        self.active_connections[user_id].add(websocket)
        logger.info(
            f"WebSocket connected for user {user_id}. Total connections for user: {len(self.active_connections[user_id])}"
        )

    def disconnect(self, websocket: WebSocket, user_id: str):
        if user_id in self.active_connections:
            self.active_connections[user_id].discard(websocket)
            if not self.active_connections[user_id]:
                del self.active_connections[user_id]
                # Clean up rooms if user has no active connections
                for room in list(self.rooms.keys()):
                    if user_id in self.rooms[room]:
                        self.rooms[room].discard(user_id)
                        if not self.rooms[room]:
                            del self.rooms[room]
            logger.info(
                f"WebSocket disconnected for user {user_id}. Remaining connections for user: {len(self.active_connections.get(user_id, set()))}"
            )


    async def subscribe_to_room(self, room_name: str, user_id: str):
        if room_name not in self.rooms:
            self.rooms[room_name] = set()
        self.rooms[room_name].add(user_id)
        logger.info(f"User {user_id} subscribed to room '{room_name}'.")

    async def unsubscribe_from_room(self, room_name: str, user_id: str):
        if room_name in self.rooms:
            self.rooms[room_name].discard(user_id)
            if not self.rooms[room_name]:
                del self.rooms[room_name]
            logger.info(f"User {user_id} unsubscribed from room '{room_name}'.")

    async def send_personal_message(self, message: Union[str, Dict[str, Any]], user_id: str):
        if isinstance(message, dict):
            try:
                # Use to_serializable to handle numpy types before json.dumps
                serializable_message = to_serializable(message)
                message = json.dumps(serializable_message)
            except Exception as e:
                logger.error(f"❌ Failed to serialize personal message for user {user_id}: {e}", exc_info=True)
                logger.debug(f"Message that failed to serialize: {message}")
                return

        if user_id in self.active_connections:
            # CRITICAL FIX: Iterate over a copy to prevent "Set changed size during iteration" errors
            connections_copy = list(self.active_connections[user_id])
            dead_connections = []
            
            for connection in connections_copy:
                try:
                    await connection.send_text(message)
                except WebSocketDisconnect as e:
                    # ✅ Explicit handling for FastAPI WebSocket disconnect
                    logger.debug(f"[WS] WebSocketDisconnect for user {user_id}, marking for cleanup")
                    dead_connections.append(connection)
                except (RuntimeError, Exception) as e:
                    # Specific check for "send after close" or general broken connections
                    error_str = str(e).lower()
                    is_closed = any(msg in error_str for msg in [
                        "asgi message", "websocket.send", "close message has been sent", 
                        "connection closed", "closed", "invalid", "state enum", 
                        "client has closed", "1000", "1006"
                    ])
                    
                    if is_closed:
                        logger.debug(f"[WS] Connection closed for user {user_id}, marking for cleanup")
                        dead_connections.append(connection)
                    else:
                        logger.error(
                            f"[WS] Error sending personal message to user {user_id}: {type(e).__name__}: {e}",
                            exc_info=False
                        )
                        dead_connections.append(connection)
            
            # Clean up dead connections
            for conn in dead_connections:
                self.active_connections[user_id].discard(conn)
            
            # Remove user if no connections left
            if not self.active_connections[user_id]:
                del self.active_connections[user_id]
                logger.debug(f"[WS] Removed user {user_id} - no active connections")

    async def broadcast_to_room(self, room_name: str, message: Union[str, Dict[str, Any]]):
        if isinstance(message, dict):
            try:
                # Use to_serializable to handle numpy types before json.dumps
                serializable_message = to_serializable(message)
                message = json.dumps(serializable_message)
            except Exception as e:
                logger.error(f"❌ Failed to serialize room broadcast for room {room_name}: {e}", exc_info=True)
                logger.debug(f"Message that failed to serialize: {message}")
                return

        if room_name in self.rooms:
            user_ids = list(self.rooms[room_name])  # Copy to prevent set change during iteration
            for user_id in user_ids:
                if user_id in self.active_connections:
                    connections_copy = list(self.active_connections[user_id])
                    dead_connections = []
                    
                    for connection in connections_copy:
                        try:
                            await connection.send_text(message)
                        except WebSocketDisconnect:
                            logger.debug(f"[WS] Client disconnected during room broadcast to user {user_id} in room '{room_name}'")
                            dead_connections.append(connection)
                        except Exception as e:
                            error_str = str(e).lower()
                            is_closed = any(msg in error_str for msg in [
                                "asgi message", "websocket.send", "close message has been sent",
                                "connection closed", "closed", "invalid", "state enum",
                                "client has closed", "1000", "1006", "websocketdisconnect",
                            ])

                            if is_closed:
                                logger.debug(f"[WS] Connection closed in room '{room_name}' for user {user_id}")
                                dead_connections.append(connection)
                            else:
                                logger.error(
                                    f"[WS] Error broadcasting to room '{room_name}' for user {user_id}: {type(e).__name__}: {e}",
                                    exc_info=False
                                )
                                dead_connections.append(connection)
                    
                    # Clean up dead connections
                    for conn in dead_connections:
                        self.active_connections[user_id].discard(conn)
                    
                    if not self.active_connections[user_id]:
                        del self.active_connections[user_id]

    async def broadcast(self, message: Union[str, Dict[str, Any]]):
        if isinstance(message, dict):
            try:
                # Use to_serializable to handle numpy types before json.dumps
                serializable_message = to_serializable(message)
                message = json.dumps(serializable_message)
            except Exception as e:
                logger.error(f"❌ Failed to serialize general broadcast: {e}", exc_info=True)
                logger.debug(f"Message that failed to serialize: {message}")
                return

        # Track dead connections to clean up afterwards
        dead_connections = {}  # user_id -> list of dead connections
        
        # Iterate over a copy to prevent "Set changed size during iteration" errors
        for user_id, connections in list(self.active_connections.items()):
            dead_connections[user_id] = []
            for connection in list(connections):
                try:
                    await connection.send_text(message)
                except WebSocketDisconnect:
                    # Client disconnected cleanly — not an error, just clean up.
                    logger.debug(f"[WS] Client disconnected during broadcast to user {user_id}, removing connection")
                    dead_connections[user_id].append(connection)
                except (RuntimeError, Exception) as e:
                    error_str = str(e).lower()
                    # Identify connection state from error message
                    is_closed = any(msg in error_str for msg in [
                        "asgi message", "websocket.send", "close message has been sent",
                        "connection closed", "closed", "invalid", "state enum",
                        "client has closed", "1000", "1006", "websocketdisconnect",
                    ])

                    if is_closed:
                        logger.debug(f"[WS] Connection closed for user {user_id}, marking for cleanup: {error_str}")
                        dead_connections[user_id].append(connection)
                    else:
                        logger.error(
                            f"[WS] Error during general broadcast to user {user_id}: {type(e).__name__}: {e}",
                            exc_info=False
                        )
                        dead_connections[user_id].append(connection)
        
        # Clean up dead connections
        for user_id, dead_conns in dead_connections.items():
            for conn in dead_conns:
                if user_id in self.active_connections:
                    self.active_connections[user_id].discard(conn)
                    if not self.active_connections[user_id]:
                        del self.active_connections[user_id]
                        logger.debug(f"[WS] Removed user {user_id} - no active connections")

    async def subscribe_to_task(self, connection_id: str, task_id: str):
        """Subscribe a connection to task updates."""
        if task_id not in self.task_connections:
            self.task_connections[task_id] = set()
        self.task_connections[task_id].add(connection_id)
        
        if connection_id not in self.connection_tasks:
            self.connection_tasks[connection_id] = set()
        self.connection_tasks[connection_id].add(task_id)
    
    async def cleanup_task_connections(self, task_id: str) -> int:
        """Clean up all connections subscribed to a specific task."""
        if task_id not in self.task_connections:
            return 0
        
        connection_ids = self.task_connections[task_id].copy()
        
        # Remove task from connection mappings
        for conn_id in connection_ids:
            if conn_id in self.connection_tasks:
                self.connection_tasks[conn_id].discard(task_id)
                if not self.connection_tasks[conn_id]:
                    del self.connection_tasks[conn_id]
        
        # Remove task mapping
        del self.task_connections[task_id]
        
        logger.info(f"Cleaned up {len(connection_ids)} connections for task {task_id}")
        return len(connection_ids)

    async def send_progress_update(self, task_id: str, progress_data: dict, user_id: Optional[str] = None) -> None:
        """
        Send progress update via WebSocket to the relevant client(s).
        
        🔥 RESILIENCE: Uses circuit breaker to prevent cascade failures
        - If too many failures detected, temporarily stops sending
        - Allows recovery time for backend
        - Resumes once service health improves
        """
        try:
            # Extract user_id from progress_data if not provided explicitly
            effective_user_id = user_id or progress_data.get("user_id")
            
            # ✅ CIRCUIT BREAKER CHECK: Should we attempt to send?
            if effective_user_id and effective_user_id != "unknown":
                if effective_user_id not in self.user_circuit_breakers:
                    self.user_circuit_breakers[effective_user_id] = CircuitBreaker()
                
                cb = self.user_circuit_breakers[effective_user_id]
                if not cb.call_allowed():
                    # Circuit is OPEN - silently drop this message
                    logger.debug(f"[CB] Dropping progress update for user {effective_user_id} (circuit {cb.state})")
                    return
            
            # Standardize message root: task_id, type, and user_id should be easily accessible
            message = {
                "type": progress_data.get("type", "progress"),
                "task_id": task_id,
                "user_id": effective_user_id or "unknown",
                **progress_data,
            }
            
            # Enforce personal messages over broadcasts whenever a user_id is available
            if effective_user_id and effective_user_id != "unknown":
                if effective_user_id in self.active_connections:
                    try:
                        await self.send_personal_message(message, effective_user_id)
                        # ✅ Record success for circuit breaker
                        if effective_user_id in self.user_circuit_breakers:
                            self.user_circuit_breakers[effective_user_id].record_success()
                    except Exception as send_err:
                        # ✅ Record failure for circuit breaker
                        if effective_user_id in self.user_circuit_breakers:
                            self.user_circuit_breakers[effective_user_id].record_failure()
                        raise
                else:
                    logger.debug(f"⚠️ [send_progress_update] User {effective_user_id} NOT found in active connections")
            else:
                # Fallback to general broadcast only if absolutely necessary
                try:
                    logger.debug(f"📤 [send_progress_update] Broadcasting to all users (no specific user_id)")
                    await self.broadcast(message)
                except Exception as broadcast_err:
                    # Don't update circuit breaker for broadcast errors (no specific user)
                    raise
                
        except Exception as e:
            logger.error(f"Error sending progress update for task {task_id}: {e}")


# Global instance
manager = ConnectionManager()


def get_websocket_manager():
    return manager
