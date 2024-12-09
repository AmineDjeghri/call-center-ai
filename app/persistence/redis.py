import hashlib
from datetime import timedelta
from uuid import uuid4

from opentelemetry.instrumentation.redis import RedisInstrumentor
from redis.asyncio import Redis
from redis.asyncio.retry import Retry
from redis.backoff import ExponentialBackoff
from redis.exceptions import (
    BusyLoadingError,
    ConnectionError as RedisConnectionError,
    RedisError,
)

from app.helpers.config_models.cache import RedisModel
from app.helpers.logging import logger
from app.models.readiness import ReadinessEnum
from app.persistence.icache import ICache

# Instrument redis
RedisInstrumentor().instrument()

_retry = Retry(backoff=ExponentialBackoff(), retries=3)


class RedisCache(ICache):
    _client: Redis
    _config: RedisModel

    def __init__(self, config: RedisModel):
        logger.info("Using Redis cache %s:%s", config.host, config.port)
        self._config = config
        self._client = Redis(
            # Database location
            db=config.database,
            # Reliability
            health_check_interval=10,  # Check the health of the connection every 10 secs
            retry_on_error=[BusyLoadingError, RedisConnectionError],
            retry_on_timeout=True,
            retry=_retry,
            socket_connect_timeout=5,  # Give the system sufficient time to connect even under higher CPU conditions
            socket_timeout=1,  # Respond quickly or abort, this is a cache
            # Deployment
            host=config.host,
            port=config.port,
            ssl=config.ssl,
            # Authentication
            password=config.password.get_secret_value(),
        )  # Redis manage by itself a low level connection pool with asyncio, but be warning to not use a generator while consuming the connection, it will close it

    async def readiness(self) -> ReadinessEnum:
        """
        Check the readiness of the Redis cache.

        This will validate the ACID properties of the database: Create, Read, Update, Delete.
        """
        test_name = str(uuid4())
        test_value = "test"
        try:
            # Test the item does not exist
            assert await self._client.get(test_name) is None
            # Create a new item
            await self._client.set(test_name, test_value)
            # Test the item is the same
            assert (await self._client.get(test_name)).decode() == test_value
            # Delete the item
            await self._client.delete(test_name)
            # Test the item does not exist
            assert await self._client.get(test_name) is None
            return ReadinessEnum.OK
        except AssertionError:
            logger.exception("Readiness test failed")
        except RedisError:
            logger.exception("Error requesting Redis")
        except Exception:
            logger.exception("Unknown error while checking Redis readiness")
        return ReadinessEnum.FAIL

    async def get(self, key: str) -> bytes | None:
        """
        Get a value from the cache.

        If the key does not exist or if the key exists but the value is empty, return `None`.

        Catch errors for a maximum of 3 times, then raise the error.
        """
        sha_key = self._key_to_hash(key)
        res = None
        try:
            res = await self._client.get(sha_key)
        except RedisError:
            logger.exception("Error getting value")
        return res

    async def set(self, key: str, value: str | bytes | None, ttl_sec: int) -> bool:
        """
        Set a value in the cache.

        If the value is `None`, set an empty string.

        Catch errors for a maximum of 3 times, then raise the error.
        """
        sha_key = self._key_to_hash(key)
        try:
            await self._client.set(
                ex=timedelta(seconds=ttl_sec),
                name=sha_key,
                value=value if value else "",
            )
        except RedisError:
            logger.exception("Error setting value")
            return False
        return True

    async def delete(self, key: str) -> bool:
        """
        Delete a value from the cache.

        Catch errors for a maximum of 3 times, then raise the error.
        """
        sha_key = self._key_to_hash(key)
        try:
            await self._client.delete(sha_key)
        except RedisError:
            logger.exception("Error deleting value")
            return False
        return True

    @staticmethod
    def _key_to_hash(key: str) -> bytes:
        """
        Transform the key into a hash.

        SHA-256 lower the collision probability. Plus, it reduce the key size, which is useful for memory usage.
        """
        return hashlib.sha256(key.encode(), usedforsecurity=False).digest()