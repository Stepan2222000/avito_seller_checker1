"""
Database connection management using asyncpg.
Provides connection pooling and database initialization.
"""

import asyncpg
from typing import Optional
import logging

from .schema import INIT_DATABASE

logger = logging.getLogger(__name__)

# Global connection pool
_pool: Optional[asyncpg.Pool] = None


async def create_pool(
    host: str,
    port: int,
    database: str,
    user: str,
    password: str,
    min_size: int = 5,
    max_size: int = 20,
) -> asyncpg.Pool:
    """
    Create and return an asyncpg connection pool.

    Args:
        host: Database host
        port: Database port
        database: Database name
        user: Database user
        password: Database password
        min_size: Minimum pool size
        max_size: Maximum pool size

    Returns:
        asyncpg.Pool instance
    """
    global _pool

    if _pool is not None:
        logger.warning("Connection pool already exists. Returning existing pool.")
        return _pool

    logger.info(f"Creating asyncpg connection pool to {host}:{port}/{database}")

    try:
        _pool = await asyncpg.create_pool(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password,
            min_size=min_size,
            max_size=max_size,
            command_timeout=60,
        )
        logger.info(f"Connection pool created successfully (min_size={min_size}, max_size={max_size})")
        return _pool
    except Exception as e:
        logger.error(f"Failed to create connection pool: {e}")
        raise


async def init_database(pool: asyncpg.Pool) -> None:
    """
    Initialize database schema (create tables and indexes if they don't exist).

    Args:
        pool: asyncpg connection pool
    """
    logger.info("Initializing database schema...")

    try:
        async with pool.acquire() as conn:
            # Execute schema initialization in a transaction
            async with conn.transaction():
                await conn.execute(INIT_DATABASE)

        logger.info("Database schema initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database schema: {e}")
        raise


async def close_pool() -> None:
    """
    Close the connection pool gracefully.
    """
    global _pool

    if _pool is None:
        logger.warning("No connection pool to close")
        return

    logger.info("Closing connection pool...")

    try:
        await _pool.close()
        _pool = None
        logger.info("Connection pool closed successfully")
    except Exception as e:
        logger.error(f"Error closing connection pool: {e}")
        raise


def get_pool() -> asyncpg.Pool:
    """
    Get the global connection pool.

    Returns:
        asyncpg.Pool instance

    Raises:
        RuntimeError: If pool hasn't been created yet
    """
    if _pool is None:
        raise RuntimeError(
            "Connection pool not initialized. Call create_pool() first."
        )
    return _pool
