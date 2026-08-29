import os
import httpx
import asyncio
import logging
import time
from typing import List, Dict, Optional
import random
from app.services.code_integrity_service import get_code_checksum

logger = logging.getLogger(__name__)

class SmartRouter:
    """
    Smart Router for ServerBackend.
    Handles load balancing, session stickiness, and latency-based selection.
    """
    def __init__(self):
        self.servers: List[str] = self._load_servers()
        self.server_stats: Dict[str, Dict] = {
            server: {"latency": float('inf'), "active_sessions": 0, "status": "unknown"}
            for server in self.servers
        }
        self.session_map: Dict[str, str] = {}  # session_id -> server_url
        self._loop_task = None

    def _load_servers(self) -> List[str]:
        """Load server URLs from environment variable."""
        urls_str = os.getenv("SERVER_BACKEND_URLS", "http://localhost:9786")
        logger.info(f"🔧 [SMART_ROUTER] Loading servers from SERVER_BACKEND_URLS: {urls_str}")
        logger.info(f"🔧 [SMART_ROUTER] Also checking SERVER_BACKEND_URL: {os.getenv('SERVER_BACKEND_URL')}")
        urls = [url.strip() for url in urls_str.split(",") if url.strip()]
        if not urls:
            logger.warning("No SERVER_BACKEND_URLS configured, defaulting to localhost:9786")
            return ["http://localhost:9786"]
        logger.info(f"🔧 [SMART_ROUTER] Loaded servers: {urls}")
        return urls

    async def start_monitoring(self):
        """Start background monitoring task."""
        if self._loop_task is None:
            self._loop_task = asyncio.create_task(self._monitor_loop())

    async def stop_monitoring(self):
        """Stop background monitoring task."""
        if self._loop_task:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
            self._loop_task = None

    async def _monitor_loop(self):
        """Periodically check server health and latency."""
        while True:
            await self._check_servers()
            await asyncio.sleep(60)  # Check every minute

    async def _check_servers(self):
        """Ping all servers to update latency and status."""
        async with httpx.AsyncClient(timeout=5.0) as client:
            for server in self.servers:
                try:
                    start_time = time.time()
                    response = await client.get(f"{server}/api/health")
                    latency = (time.time() - start_time) * 1000  # ms
                    
                    if response.status_code == 200:
                        self.server_stats[server]["latency"] = latency
                        self.server_stats[server]["status"] = "healthy"
                    else:
                        self.server_stats[server]["status"] = "degraded"
                        self.server_stats[server]["latency"] = float('inf')
                except Exception as e:
                    logger.warning(f"Health check failed for {server}: {e}")
                    self.server_stats[server]["status"] = "offline"
                    self.server_stats[server]["latency"] = float('inf')

    def get_server(self, session_id: Optional[str] = None) -> str:
        """
        Get the best server for the request.
        If session_id is provided, returns the assigned server (stickiness).
        Otherwise, returns the best available server based on latency and load.
        """
        # 1. Session Stickiness
        if session_id and session_id in self.session_map:
            server = self.session_map[session_id]
            # Verify server is still healthy, if not, reassign
            if self.server_stats.get(server, {}).get("status") == "healthy":
                return server
            else:
                logger.info(f"Session {session_id} server {server} is unhealthy, reassigning.")
                del self.session_map[session_id]

        # 2. Select Best Server
        healthy_servers = [
            s for s in self.servers 
            if self.server_stats[s]["status"] == "healthy"
        ]
        
        if not healthy_servers:
            # If no healthy servers known (e.g. startup), try all
            # Or if all are offline, return one to try anyway
            candidates = self.servers
        else:
            candidates = healthy_servers

        # Sort by latency (simple approach)
        # In a real scenario, we'd also consider active_sessions (load)
        candidates.sort(key=lambda s: self.server_stats[s]["latency"])
        
        # Pick top candidate (or random among top 3 to avoid thundering herd)
        best_server = candidates[0]
        
        # 3. Assign Session
        if session_id:
            self.session_map[session_id] = best_server
            self.server_stats[best_server]["active_sessions"] += 1

        return best_server

    async def get_integrity_headers(self) -> Dict[str, str]:
        """Get headers required for code integrity verification by ServerBackend."""
        try:
            checksum = await get_code_checksum()
            return {"X-Code-Checksum": checksum}
        except Exception as e:
            logger.warning(f"Failed to get code checksum for integrity headers: {e}")
            return {}

# Global instance
_smart_router: Optional[SmartRouter] = None

def get_smart_router() -> SmartRouter:
    global _smart_router
    if _smart_router is None:
        _smart_router = SmartRouter()
    return _smart_router

smart_router = get_smart_router()
