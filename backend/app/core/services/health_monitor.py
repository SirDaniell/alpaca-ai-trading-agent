"""
Backend Health Monitor Service

Monitors Backend health and ServerBackend connectivity.
Integrates with existing rate limiters and provides WebSocket updates.
"""

import asyncio
import time
import httpx
import os
from typing import Dict, Any, Optional
from datetime import datetime, timezone
from fastapi import WebSocket
from collections import deque
import logging

logger = logging.getLogger(__name__)


class HealthMetrics:
    """Track health metrics for a service"""

    def __init__(self, window_size: int = 100):
        self.request_count = 0
        self.success_count = 0
        self.error_count = 0
        self.latencies = deque(maxlen=window_size)
        self.start_time = time.time()
        self.last_check = None

    def record_request(self, success: bool, latency: float):
        """Record a request"""
        self.request_count += 1
        if success:
            self.success_count += 1
        else:
            self.error_count += 1
        self.latencies.append(latency)
        self.last_check = datetime.now(timezone.utc).isoformat()

    def get_error_rate(self) -> float:
        """Calculate error rate"""
        if self.request_count == 0:
            return 0.0
        return self.error_count / self.request_count

    def get_avg_latency(self) -> float:
        """Calculate average latency"""
        if not self.latencies:
            return 0.0
        return sum(self.latencies) / len(self.latencies)

    def get_uptime(self) -> float:
        """Get uptime in seconds"""
        return time.time() - self.start_time

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "uptime": self.get_uptime(),
            "request_count": self.request_count,
            "error_rate": self.get_error_rate(),
            "avg_latency": self.get_avg_latency(),
            "last_check": self.last_check or datetime.now(timezone.utc).isoformat(),
        }


class HealthMonitor:
    """
    Centralized health monitoring service

    Monitors:
    - Backend's own health (database, services)
    - ServerBackend connectivity
    - Request metrics
    """

    def __init__(self, server_backend_url: str = None):
        # Read from environment variable, with proper defaults for dev vs prod
        if server_backend_url is None:
            server_backend_url = os.getenv(
                "SERVER_BACKEND_URL",
                "http://localhost:8214"  # Dev default: localhost:8214
            )
        self.server_backend_url = server_backend_url
        self.backend_metrics = HealthMetrics()
        self.server_backend_metrics = HealthMetrics()

        # WebSocket connections
        self.active_connections: set[WebSocket] = set()

        # Health check task
        self.check_task: Optional[asyncio.Task] = None
        self.check_interval = 30  # seconds

        # Status cache
        self.backend_status = "unknown"
        self.server_backend_status = "unknown"
        self.dependencies_status = {}

    async def start(self):
        """Start health monitoring"""
        if self.check_task is None or self.check_task.done():
            self.check_task = asyncio.create_task(self._periodic_health_check())
            logger.info("Health monitor started")

    async def stop(self):
        """Stop health monitoring"""
        if self.check_task and not self.check_task.done():
            self.check_task.cancel()
            try:
                await self.check_task
            except asyncio.CancelledError:
                pass
            logger.info("Health monitor stopped")

    async def _periodic_health_check(self):
        """Periodic health check loop"""
        while True:
            try:
                await self.check_all_services()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health check: {e}")
                await asyncio.sleep(self.check_interval)

    async def check_all_services(self):
        """Check health of all services"""
        # Check ServerBackend
        await self.check_server_backend()

        # Check local dependencies (database, etc.)
        await self.check_dependencies()

        # Broadcast update
        await self.broadcast_health_update()

    async def check_server_backend(self):
        """Check ServerBackend health"""
        start_time = time.time()

        try:
            async with httpx.AsyncClient() as client:
                try:
                    response = await client.get(
                        f"{self.server_backend_url}/api/health", timeout=5.0
                    )

                    latency = time.time() - start_time

                    if response.status_code == 200:
                        self.server_backend_metrics.record_request(True, latency)
                        data = response.json()
                        self.server_backend_status = data.get("status", "healthy")
                    else:
                        self.server_backend_metrics.record_request(False, latency)
                        self.server_backend_status = "degraded"
                except (httpx.ConnectError, httpx.TimeoutException) as e:
                    # Expected errors when service is down
                    latency = time.time() - start_time
                    self.server_backend_metrics.record_request(False, latency)
                    self.server_backend_status = "offline"
                    # Log at debug level to avoid spamming logs
                    logger.debug(f"ServerBackend health check failed (expected): {e}")
                except Exception as e:
                    # Unexpected errors
                    latency = time.time() - start_time
                    self.server_backend_metrics.record_request(False, latency)
                    self.server_backend_status = "offline"
                    logger.error(f"ServerBackend health check failed (unexpected): {e}")

        except Exception as e:
            # Catch-all for any other errors (e.g. client init)
            latency = time.time() - start_time
            self.server_backend_metrics.record_request(False, latency)
            self.server_backend_status = "offline"
            logger.error(f"Critical error in health check: {e}")

    async def check_dependencies(self):
        """Check local dependencies using async-safe operations only."""
        # ✅ FIX: The old code used a synchronous `with engine.connect()` inside
        # an async function. SQLAlchemy's synchronous connect() blocks the entire
        # uvicorn event loop for the duration of the DB round-trip, which is why
        # the health endpoint was timing out on the frontend (5 s abort) whenever
        # the DB was under load from concurrent auth sessions.
        #
        # Correct approach: run the sync call in a thread via run_in_executor so
        # the event loop stays free.
        try:
            from app.models.database import engine
            from sqlalchemy import text
            import asyncio

            loop = asyncio.get_running_loop()
            def _check_db():
                with engine.connect() as conn:
                    conn.execute(text("SELECT 1"))

            await loop.run_in_executor(None, _check_db)
            self.dependencies_status["database"] = "connected"
        except Exception as e:
            self.dependencies_status["database"] = "disconnected"
            logger.error(f"Database check failed: {e}")

        self.dependencies_status["mt5"] = "unknown"

    def record_request(self, success: bool, latency: float):
        """Record a request to Backend"""
        self.backend_metrics.record_request(success, latency)

    def get_backend_health(self) -> Dict[str, Any]:
        """Get Backend health status"""
        error_rate = self.backend_metrics.get_error_rate()

        # Determine status
        if error_rate < 0.05:
            status = "healthy"
        elif error_rate < 0.2:
            status = "degraded"
        else:
            status = "offline"

        return {
            "service": "backend",
            "status": status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metrics": self.backend_metrics.to_dict(),
            "dependencies": self.dependencies_status,
        }

    def get_server_backend_health(self) -> Dict[str, Any]:
        """Get ServerBackend health status"""
        return {
            "service": "server_backend",
            "status": self.server_backend_status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metrics": self.server_backend_metrics.to_dict(),
        }

    def get_full_health(self) -> Dict[str, Any]:
        """Get full health status"""
        backend_health = self.get_backend_health()
        server_backend_health = self.get_server_backend_health()

        # Determine overall status
        if (
            backend_health["status"] == "healthy"
            and server_backend_health["status"] == "healthy"
        ):
            overall_status = "healthy"
        elif (
            backend_health["status"] == "offline"
            or server_backend_health["status"] == "offline"
        ):
            overall_status = "critical"
        else:
            overall_status = "degraded"

        return {
            "overall_status": overall_status,
            "backend": backend_health,
            "server_backend": server_backend_health,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    async def connect_websocket(self, websocket: WebSocket):
        """Connect a WebSocket client"""
        await websocket.accept()
        self.active_connections.add(websocket)

        # Send initial health status
        try:
            await websocket.send_json(
                {"type": "health_update", **self.get_full_health()}
            )
        except Exception as e:
            logger.warning(f"Failed to send initial health update: {e}")
            # Don't remove from active_connections here, let the caller handle disconnect
            # or let the next broadcast clean it up.

    def disconnect_websocket(self, websocket: WebSocket):
        """Disconnect a WebSocket client"""
        self.active_connections.discard(websocket)

    async def broadcast_health_update(self):
        """Broadcast health update to all connected WebSocket clients"""
        if not self.active_connections:
            return

        message = {"type": "health_update", **self.get_full_health()}

        # Send to all connected clients
        disconnected = set()
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error sending to WebSocket: {e}")
                disconnected.add(connection)

        # Remove disconnected clients
        self.active_connections -= disconnected


# Global health monitor instance
health_monitor = HealthMonitor()
