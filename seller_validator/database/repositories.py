"""
Database repositories for tasks, results, and proxies.
All methods use raw SQL queries through asyncpg.
"""

import asyncpg
from typing import Optional, List, Dict, Any
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)


class TaskRepository:
    """Repository for managing tasks in the database."""

    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool

    async def get_next_task(self) -> Optional[asyncpg.Record]:
        """
        Get the next pending task for processing using row-level locking.
        Uses FOR UPDATE SKIP LOCKED for safe concurrent access.

        Returns:
            Task record or None if no tasks available
        """
        query = """
        UPDATE tasks
        SET state = 'in_progress',
            started_at = NOW(),
            updated_at = NOW()
        WHERE id = (
            SELECT id
            FROM tasks
            WHERE state = 'pending'
            ORDER BY enqueued_at ASC
            LIMIT 1
            FOR UPDATE SKIP LOCKED
        )
        RETURNING *;
        """

        async with self.pool.acquire() as conn:
            return await conn.fetchrow(query)

    async def mark_completed(self, task_id: str) -> None:
        """Mark a task as completed."""
        query = """
        UPDATE tasks
        SET state = 'completed',
            completed_at = NOW(),
            updated_at = NOW()
        WHERE id = $1;
        """

        async with self.pool.acquire() as conn:
            await conn.execute(query, task_id)

    async def mark_failed(
        self,
        task_id: str,
        increment_attempt: bool = True,
        max_attempts: int = 4
    ) -> str:
        """
        Mark a task as failed or return to pending for retry.

        Args:
            task_id: Task UUID
            increment_attempt: Whether to increment attempt counter
            max_attempts: Maximum number of attempts before abandoning

        Returns:
            New state of the task ('pending' or 'abandoned')
        """
        if increment_attempt:
            query = """
            UPDATE tasks
            SET state = CASE
                    WHEN attempt + 1 >= $2 THEN 'abandoned'
                    ELSE 'pending'
                END,
                attempt = attempt + 1,
                updated_at = NOW()
            WHERE id = $1
            RETURNING state;
            """
            async with self.pool.acquire() as conn:
                new_state = await conn.fetchval(query, task_id, max_attempts)
                return new_state or 'unknown'
        else:
            query = """
            UPDATE tasks
            SET state = 'pending',
                updated_at = NOW()
            WHERE id = $1
            RETURNING state;
            """
            async with self.pool.acquire() as conn:
                new_state = await conn.fetchval(query, task_id)
                return new_state or 'unknown'

    async def update_last_proxy(self, task_id: str, proxy: str) -> None:
        """Update the last used proxy for a task."""
        query = """
        UPDATE tasks
        SET last_proxy = $2,
            updated_at = NOW()
        WHERE id = $1;
        """

        async with self.pool.acquire() as conn:
            await conn.execute(query, task_id, proxy)

    async def add_tasks(
        self,
        urls: List[str],
        payload: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Add multiple tasks to the database.
        Ignores duplicates (based on unique URL constraint).

        Args:
            urls: List of URLs to add
            payload: Optional metadata for tasks

        Returns:
            Number of tasks actually inserted
        """
        if not urls:
            return 0

        payload_json = json.dumps(payload or {})

        query = """
        INSERT INTO tasks (url, payload)
        VALUES ($1, $2)
        ON CONFLICT (url) DO NOTHING;
        """

        inserted_count = 0
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                for url in urls:
                    result = await conn.execute(query, url, payload_json)
                    # Parse result string like "INSERT 0 1" to get row count
                    if result.split()[-1] == "1":
                        inserted_count += 1

        logger.info(f"Inserted {inserted_count}/{len(urls)} new tasks")
        return inserted_count

    async def clear_pending_tasks(self) -> int:
        """
        Clear all pending tasks (used for safe replace operation).
        Does NOT touch in_progress, completed, failed, or abandoned tasks.

        Returns:
            Number of tasks deleted
        """
        query = """
        DELETE FROM tasks
        WHERE state = 'pending'
        RETURNING id;
        """

        async with self.pool.acquire() as conn:
            records = await conn.fetch(query)
            deleted_count = len(records)

        logger.info(f"Deleted {deleted_count} pending tasks")
        return deleted_count

    async def get_task_by_url(self, url: str) -> Optional[asyncpg.Record]:
        """Get a task by its URL."""
        query = """
        SELECT * FROM tasks
        WHERE url = $1;
        """

        async with self.pool.acquire() as conn:
            return await conn.fetchrow(query, url)

    async def count_by_state(self) -> Dict[str, int]:
        """Get count of tasks grouped by state."""
        query = """
        SELECT state, COUNT(*) as count
        FROM tasks
        GROUP BY state;
        """

        async with self.pool.acquire() as conn:
            records = await conn.fetch(query)

        return {record['state']: record['count'] for record in records}


class ResultRepository:
    """Repository for managing validation results."""

    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool

    async def save_result(
        self,
        task_id: str,
        seller_url: str,
        verdict: str,
        reason: str,
        confidence: float,
        flags: List[str],
        items: List[Dict[str, Any]],
        llm_attempts: int = 1
    ) -> str:
        """
        Save a validation result to the database.

        Args:
            task_id: Task UUID
            seller_url: Seller profile URL
            verdict: Validation verdict (passed/failed/error)
            reason: Explanation
            confidence: Confidence score (0.0-1.0)
            flags: List of flags
            items: List of item decisions
            llm_attempts: Number of LLM attempts made

        Returns:
            Result UUID
        """
        query = """
        INSERT INTO results (
            task_id, seller_url, verdict, reason, confidence, flags, items, llm_attempts
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        RETURNING id;
        """

        items_json = json.dumps(items)

        async with self.pool.acquire() as conn:
            result_id = await conn.fetchval(
                query,
                task_id,
                seller_url,
                verdict,
                reason,
                confidence,
                flags,
                items_json,
                llm_attempts
            )

        logger.debug(f"Saved result {result_id} for task {task_id}")
        return str(result_id)

    async def get_results_by_verdict(self, verdict: str) -> List[asyncpg.Record]:
        """Get all results with a specific verdict."""
        query = """
        SELECT * FROM results
        WHERE verdict = $1
        ORDER BY created_at DESC;
        """

        async with self.pool.acquire() as conn:
            return await conn.fetch(query, verdict)

    async def count_by_verdict(self) -> Dict[str, int]:
        """Get count of results grouped by verdict."""
        query = """
        SELECT verdict, COUNT(*) as count
        FROM results
        GROUP BY verdict;
        """

        async with self.pool.acquire() as conn:
            records = await conn.fetch(query)

        return {record['verdict']: record['count'] for record in records}


class ProxyRepository:
    """Repository for managing proxies."""

    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool

    async def get_available_proxy(self, exclude_last: Optional[str] = None) -> Optional[str]:
        """
        Get an available (non-blocked, not in use) proxy.

        Args:
            exclude_last: Proxy to exclude (last used proxy for retry)

        Returns:
            Proxy address or None if no proxies available
        """
        if exclude_last:
            query = """
            SELECT address
            FROM proxies
            WHERE is_blocked = FALSE
              AND in_use = FALSE
              AND address != $1
            ORDER BY COALESCE(last_used_at, '1970-01-01'::timestamp) ASC
            LIMIT 1;
            """
            async with self.pool.acquire() as conn:
                record = await conn.fetchrow(query, exclude_last)
        else:
            query = """
            SELECT address
            FROM proxies
            WHERE is_blocked = FALSE
              AND in_use = FALSE
            ORDER BY COALESCE(last_used_at, '1970-01-01'::timestamp) ASC
            LIMIT 1;
            """
            async with self.pool.acquire() as conn:
                record = await conn.fetchrow(query)

        return record['address'] if record else None

    async def mark_in_use(self, address: str, in_use: bool) -> None:
        """Mark a proxy as in use or not in use."""
        query = """
        UPDATE proxies
        SET in_use = $2,
            last_used_at = CASE WHEN $2 THEN NOW() ELSE last_used_at END,
            updated_at = NOW()
        WHERE address = $1;
        """

        async with self.pool.acquire() as conn:
            await conn.execute(query, address, in_use)

    async def mark_blocked(
        self,
        address: str,
        reason: Optional[str] = None
    ) -> None:
        """Mark a proxy as blocked."""
        query = """
        UPDATE proxies
        SET is_blocked = TRUE,
            blocked_reason = $2,
            blocked_at = NOW(),
            in_use = FALSE,
            updated_at = NOW()
        WHERE address = $1;
        """

        async with self.pool.acquire() as conn:
            await conn.execute(query, address, reason)

        logger.info(f"Marked proxy {address} as blocked: {reason}")

    async def load_proxies(
        self,
        proxies: List[str],
        replace: bool = False
    ) -> int:
        """
        Load proxies into the database.

        Args:
            proxies: List of proxy addresses
            replace: If True, delete all existing proxies first

        Returns:
            Number of proxies inserted
        """
        if not proxies:
            return 0

        async with self.pool.acquire() as conn:
            async with conn.transaction():
                if replace:
                    # Delete all proxies
                    await conn.execute("DELETE FROM proxies;")
                    logger.info("Deleted all existing proxies")

                # Insert new proxies
                query = """
                INSERT INTO proxies (address)
                VALUES ($1)
                ON CONFLICT (address) DO NOTHING;
                """

                inserted_count = 0
                for proxy in proxies:
                    result = await conn.execute(query, proxy)
                    if result.split()[-1] == "1":
                        inserted_count += 1

        logger.info(f"Inserted {inserted_count}/{len(proxies)} proxies (replace={replace})")
        return inserted_count

    async def get_all_proxies(self) -> List[str]:
        """Get all proxy addresses."""
        query = """
        SELECT address
        FROM proxies
        ORDER BY created_at ASC;
        """

        async with self.pool.acquire() as conn:
            records = await conn.fetch(query)

        return [record['address'] for record in records]

    async def count_proxies(self) -> Dict[str, int]:
        """Get count of proxies by status."""
        query = """
        SELECT
            COUNT(*) FILTER (WHERE is_blocked = FALSE) as available,
            COUNT(*) FILTER (WHERE is_blocked = TRUE) as blocked,
            COUNT(*) as total
        FROM proxies;
        """

        async with self.pool.acquire() as conn:
            record = await conn.fetchrow(query)

        return {
            'available': record['available'],
            'blocked': record['blocked'],
            'total': record['total']
        }
