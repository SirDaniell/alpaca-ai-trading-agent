import redis.asyncio as redis
import os
import logging
from app.core.config import settings

logger = logging.getLogger(__name__)

class RedisService:
    _instance = None
    _client = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RedisService, cls).__new__(cls)
        return cls._instance

    async def get_client(self):
        if self._client is None:
            redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")
            logger.info(f"🔌 Connecting to Redis at {redis_url}")
            try:
                self._client = await redis.from_url(
                    redis_url, 
                    decode_responses=True,
                    socket_timeout=5,
                    retry_on_timeout=True
                )
            except Exception as e:
                logger.error(f"❌ Failed to connect to Redis: {e}")
                raise e
        return self._client

    async def set(self, key: str, value: str, ex: int = None):
        client = await self.get_client()
        return await client.set(key, value, ex=ex)

    async def get(self, key: str):
        client = await self.get_client()
        return await client.get(key)

    async def delete(self, key: str):
        client = await self.get_client()
        return await client.delete(key)

    async def exists(self, key: str):
        client = await self.get_client()
        return await client.exists(key)

redis_service = RedisService()
