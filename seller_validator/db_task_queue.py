"""
Database-backed TaskQueue implementation.
Provides the same interface as avito_library TaskQueue but uses PostgreSQL for persistence.
"""

import asyncpg
from typing import Optional, Hashable, Any, Iterable, Tuple
from avito_library.reuse_utils.task_queue import ProcessingTask
import logging

from .database.repositories import TaskRepository
from .db_utils import record_to_processing_task

logger = logging.getLogger(__name__)


class DatabaseTaskQueue:
    """
    Database-backed task queue using PostgreSQL.
    Compatible interface with avito_library.reuse_utils.task_queue.TaskQueue.
    """

    def __init__(self, pool: asyncpg.Pool, max_attempts: int = 4):
        """
        Initialize database task queue.

        Args:
            pool: asyncpg connection pool
            max_attempts: Maximum number of attempts per task
        """
        self.pool = pool
        self._max_attempts = max_attempts
        self.task_repo = TaskRepository(pool)

    async def put_many(self, items: Iterable[Tuple[Hashable, Any]]) -> int:
        """
        Add multiple tasks to the queue.

        Note: All tasks will share the same payload (the first one encountered).
        This matches the behavior of loading tasks from files where all tasks
        have the same metadata (source, enqueued_at).

        Args:
            items: Iterable of (task_key, payload) tuples

        Returns:
            Number of tasks actually inserted
        """
        urls = []
        # Use the first payload for all tasks (file mode compatibility)
        # In database mode, tasks are typically loaded via scripts with identical metadata
        payload = None

        for task_key, task_payload in items:
            urls.append(str(task_key))
            if payload is None:
                payload = task_payload

        return await self.task_repo.add_tasks(urls, payload)

    async def get(self) -> Optional[ProcessingTask]:
        """
        Get the next task from the queue.
        Uses FOR UPDATE SKIP LOCKED for safe concurrent access.

        Returns:
            ProcessingTask or None if no tasks available
        """
        record = await self.task_repo.get_next_task()

        if record is None:
            return None

        # Convert database record to ProcessingTask
        task = record_to_processing_task(record)

        # Store the database ID for later operations
        # We'll store it in the payload as '_db_task_id'
        if isinstance(task.payload, dict):
            task.payload['_db_task_id'] = str(record['id'])
        else:
            # If payload is not a dict, wrap it
            task.payload = {
                '_original_payload': task.payload,
                '_db_task_id': str(record['id'])
            }

        return task

    async def mark_done(self, task_key: Hashable) -> None:
        """
        Mark a task as completed.

        Args:
            task_key: Task key (URL)
        """
        # Get task by URL to get its ID
        record = await self.task_repo.get_task_by_url(str(task_key))
        if record:
            await self.task_repo.mark_completed(str(record['id']))
        else:
            logger.warning(f"Task not found for mark_done: {task_key}")

    async def retry(
        self,
        task_key: Hashable,
        *,
        last_proxy: Optional[str] = None
    ) -> bool:
        """
        Return a task to the queue for retry.

        Args:
            task_key: Task key (URL)
            last_proxy: Last used proxy (optional)

        Returns:
            True if task was returned for retry, False if max attempts exceeded
        """
        # Get task by URL
        record = await self.task_repo.get_task_by_url(str(task_key))
        if not record:
            logger.warning(f"Task not found for retry: {task_key}")
            return False

        task_id = str(record['id'])

        # Update last proxy if provided
        if last_proxy:
            await self.task_repo.update_last_proxy(task_id, last_proxy)

        # Mark as failed (this will increment attempt and return new state)
        new_state = await self.task_repo.mark_failed(
            task_id,
            increment_attempt=True,
            max_attempts=self._max_attempts
        )

        # Check if task was abandoned
        if new_state == 'abandoned':
            logger.info(f"Task {task_key} abandoned after reaching max attempts")
            return False

        return True

    async def abandon(self, task_key: Hashable) -> None:
        """
        Remove a task from the queue without retry.

        Args:
            task_key: Task key (URL)
        """
        record = await self.task_repo.get_task_by_url(str(task_key))
        if record:
            # Mark as failed without incrementing attempts
            # Then manually set state to abandoned
            query = """
            UPDATE tasks
            SET state = 'abandoned',
                updated_at = NOW()
            WHERE id = $1;
            """
            async with self.pool.acquire() as conn:
                await conn.execute(query, record['id'])
        else:
            logger.warning(f"Task not found for abandon: {task_key}")

    async def pending_count(self) -> int:
        """
        Get the count of pending tasks.

        Returns:
            Number of pending tasks
        """
        counts = await self.task_repo.count_by_state()
        return counts.get('pending', 0)

    async def pause(self, *, reason: str) -> bool:
        """
        Pause the queue (not implemented for database queue).

        Note: In database mode, pause/resume is not necessary as tasks
        are managed at the database level. This is a no-op for compatibility.

        Returns:
            False (not supported)
        """
        logger.warning(f"Queue pause requested (reason: {reason}) but not supported in database mode")
        return False

    async def resume(self, *, reason: str) -> bool:
        """
        Resume the queue (not implemented for database queue).

        Note: In database mode, pause/resume is not necessary as tasks
        are managed at the database level. This is a no-op for compatibility.

        Returns:
            False (not supported)
        """
        logger.warning(f"Queue resume requested (reason: {reason}) but not supported in database mode")
        return False
