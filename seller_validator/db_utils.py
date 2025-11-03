"""
Utility functions for database operations.
Provides helpers for converting between asyncpg records and ProcessingTask objects.
"""

import asyncpg
from typing import Optional
from datetime import datetime
import json
from avito_library.reuse_utils.task_queue import ProcessingTask, TaskState


def record_to_processing_task(record: asyncpg.Record) -> ProcessingTask:
    """
    Convert an asyncpg Record from the tasks table to a ProcessingTask object.

    Args:
        record: Database record from tasks table

    Returns:
        ProcessingTask instance
    """
    # Parse state from database string to TaskState enum
    state_mapping = {
        'pending': TaskState.PENDING,
        'in_progress': TaskState.IN_PROGRESS,
        'returned': TaskState.RETURNED,
        # Database has more states (completed, failed, abandoned) but they won't be in queue
    }
    state = state_mapping.get(record['state'], TaskState.PENDING)

    # Parse payload from JSONB
    payload = json.loads(record['payload']) if isinstance(record['payload'], str) else record['payload']

    return ProcessingTask(
        task_key=record['url'],  # URL is the task key
        payload=payload,
        attempt=record['attempt'],
        state=state,
        last_proxy=record['last_proxy'],
        enqueued_at=record['enqueued_at'],
        updated_at=record['updated_at'],
        last_result=None,  # Not stored in database
    )


def get_task_id_from_url(url: str) -> str:
    """
    Helper to extract task_id from a URL-keyed task.
    In our case, we need to query the database to get the UUID.

    Args:
        url: Task URL (task_key)

    Returns:
        Task UUID as string
    """
    # This is a placeholder - actual implementation requires database query
    # The caller should use TaskRepository.get_task_by_url() instead
    raise NotImplementedError("Use TaskRepository.get_task_by_url() to get task_id")


def serialize_result_for_db(result_data: dict) -> dict:
    """
    Serialize a result dictionary for database storage.
    Ensures all fields are properly typed for PostgreSQL.

    Args:
        result_data: Result data from LLM analyzer

    Returns:
        Serialized result data ready for database insertion
    """
    return {
        'seller_url': result_data.get('seller_url', ''),
        'verdict': result_data.get('verdict', 'error'),
        'reason': result_data.get('reason', ''),
        'confidence': float(result_data.get('confidence', 0.0)),
        'flags': result_data.get('flags', []),
        'items': result_data.get('items', []),
    }


def ensure_datetime(value: Optional[datetime | str]) -> Optional[datetime]:
    """
    Ensure a value is a datetime object.

    Args:
        value: Datetime or ISO string

    Returns:
        Datetime object or None
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        return datetime.fromisoformat(value)
    return None
