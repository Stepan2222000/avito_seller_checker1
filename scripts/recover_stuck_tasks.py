#!/usr/bin/env python3
"""
Script to recover stuck tasks that are in 'in_progress' state for too long.
This can happen when a worker crashes or is forcefully terminated.
"""

import asyncio
import sys
from pathlib import Path
from datetime import timedelta

# Add parent directory to path to import seller_validator
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from seller_validator.config import Config
from seller_validator.database import create_pool, init_database, close_pool


async def recover_stuck_tasks(timeout_minutes: int = 60, dry_run: bool = True):
    """
    Recover tasks that have been in 'in_progress' state for longer than timeout.

    Args:
        timeout_minutes: Consider tasks stuck if in_progress for this many minutes
        dry_run: If True, only show what would be recovered without making changes
    """
    config = Config()

    print("=" * 60)
    print("Stuck Tasks Recovery Tool")
    print("=" * 60)
    print(f"Database: {config.db_host}:{config.db_port}/{config.db_name}")
    print(f"Timeout: {timeout_minutes} minutes")
    print(f"Mode: {'DRY RUN (no changes)' if dry_run else 'LIVE (will update tasks)'}")
    print()

    try:
        pool = await create_pool(
            host=config.db_host,
            port=config.db_port,
            database=config.db_name,
            user=config.db_user,
            password=config.db_password,
            min_size=config.db_pool_min_size,
            max_size=config.db_pool_max_size,
        )

        await init_database(pool)

        # Find stuck tasks
        query = """
        SELECT id, url, updated_at, attempt,
               NOW() - updated_at as stuck_duration
        FROM tasks
        WHERE state = 'in_progress'
          AND updated_at < NOW() - INTERVAL '%s minutes'
        ORDER BY updated_at ASC;
        """

        async with pool.acquire() as conn:
            stuck_tasks = await conn.fetch(query % timeout_minutes)

        if not stuck_tasks:
            print("✅ No stuck tasks found!")
            return

        print(f"Found {len(stuck_tasks)} stuck tasks:")
        print()
        print(f"{'URL':<60} {'Stuck For':<20} {'Attempt':<10}")
        print("-" * 90)

        for task in stuck_tasks:
            url = task['url'][:57] + '...' if len(task['url']) > 60 else task['url']
            duration = str(task['stuck_duration']).split('.')[0]  # Remove microseconds
            print(f"{url:<60} {duration:<20} {task['attempt']:<10}")

        print()

        if dry_run:
            print("⚠️  DRY RUN MODE - No changes will be made")
            print()
            print("To actually recover these tasks, run:")
            print(f"  python scripts/recover_stuck_tasks.py --timeout {timeout_minutes} --apply")
            return

        # Confirm recovery
        print()
        confirm = input(f"Recover {len(stuck_tasks)} stuck tasks? (yes/no): ").strip().lower()
        if confirm not in ['yes', 'y']:
            print("Recovery cancelled.")
            return

        # Recover tasks
        update_query = """
        UPDATE tasks
        SET state = 'pending',
            updated_at = NOW()
        WHERE state = 'in_progress'
          AND updated_at < NOW() - INTERVAL '%s minutes'
        RETURNING id;
        """

        async with pool.acquire() as conn:
            recovered = await conn.fetch(update_query % timeout_minutes)

        print()
        print("=" * 60)
        print(f"✅ Successfully recovered {len(recovered)} tasks!")
        print("=" * 60)

        # Show current statistics
        stats_query = """
        SELECT state, COUNT(*) as count
        FROM tasks
        GROUP BY state
        ORDER BY state;
        """

        async with pool.acquire() as conn:
            stats = await conn.fetch(stats_query)

        print()
        print("Current task statistics:")
        for row in stats:
            print(f"  {row['state']}: {row['count']}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await close_pool()


def main():
    """Main function with argument parsing."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Recover stuck tasks that have been in 'in_progress' state for too long"
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=60,
        help='Consider tasks stuck after this many minutes (default: 60)'
    )
    parser.add_argument(
        '--apply',
        action='store_true',
        help='Actually recover tasks (default is dry-run mode)'
    )

    args = parser.parse_args()

    asyncio.run(recover_stuck_tasks(
        timeout_minutes=args.timeout,
        dry_run=not args.apply
    ))


if __name__ == "__main__":
    main()
