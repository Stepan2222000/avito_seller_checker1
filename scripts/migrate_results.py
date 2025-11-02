#!/usr/bin/env python3
"""
Script to migrate existing results from results.json to the database.
Creates tasks with 'completed' state and associated results.
"""

import asyncio
import sys
import json
from pathlib import Path

# Add parent directory to path to import seller_validator
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from seller_validator.config import Config
from seller_validator.database import create_pool, init_database, close_pool, TaskRepository, ResultRepository


async def load_results_from_file(file_path: Path) -> list[dict]:
    """
    Load results from results.json file.

    Args:
        file_path: Path to the results.json file

    Returns:
        List of result dictionaries
    """
    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        return []

    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            results = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            return []

    print(f"Loaded {len(results)} results from {file_path}")
    return results


async def migrate_result(
    result: dict,
    task_repo: TaskRepository,
    result_repo: ResultRepository,
    pool
) -> tuple[bool, str]:
    """
    Migrate a single result to the database.

    Args:
        result: Result dictionary
        task_repo: Task repository
        result_repo: Result repository
        pool: Database connection pool

    Returns:
        Tuple of (success, message)
    """
    seller_url = result.get('seller_url', '')
    if not seller_url:
        return False, "Missing seller_url"

    # Check if task already exists
    existing_task = await task_repo.get_task_by_url(seller_url)
    if existing_task:
        return False, f"Task already exists for {seller_url}"

    # Create task with completed state
    # We need to do this with raw SQL since add_tasks doesn't support setting state
    query = """
    INSERT INTO tasks (url, payload, state, attempt, max_attempts, completed_at)
    VALUES ($1, $2, 'completed', 1, 1, NOW())
    RETURNING id;
    """

    try:
        async with pool.acquire() as conn:
            task_id = await conn.fetchval(query, seller_url, '{}')

        # Create result
        await result_repo.save_result(
            task_id=str(task_id),
            seller_url=seller_url,
            verdict=result.get('verdict', 'error'),
            reason=result.get('reason', ''),
            confidence=float(result.get('confidence', 0.0)),
            flags=result.get('flags', []),
            items=result.get('items', []),
            llm_attempts=1
        )

        return True, f"Migrated {seller_url}"

    except Exception as e:
        return False, f"Error migrating {seller_url}: {e}"


async def main():
    """Main function to migrate results."""
    # Parse command line arguments
    if len(sys.argv) > 1:
        file_path = Path(sys.argv[1])
    else:
        config = Config()
        file_path = config.results_json

    print("=" * 60)
    print("Results Migration Tool for Avito Seller Checker")
    print("=" * 60)
    print(f"Source file: {file_path}")
    print()

    # Load results from file
    results = await load_results_from_file(file_path)
    if not results:
        print("No results to migrate. Exiting.")
        return

    print(f"Found {len(results)} results to migrate")
    print()

    # Confirm migration
    confirm = input("Proceed with migration? (yes/no): ").strip().lower()
    if confirm not in ['yes', 'y']:
        print("Migration cancelled.")
        return

    # Connect to database
    config = Config()
    print()
    print(f"Connecting to database at {config.db_host}:{config.db_port}/{config.db_name}...")

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

        # Initialize database schema
        await init_database(pool)

        # Create repositories
        task_repo = TaskRepository(pool)
        result_repo = ResultRepository(pool)

        # Migrate results
        print()
        print("Migrating results...")
        print()

        success_count = 0
        skip_count = 0
        error_count = 0

        for i, result in enumerate(results, 1):
            success, message = await migrate_result(result, task_repo, result_repo, pool)

            if success:
                success_count += 1
                if success_count % 10 == 0:  # Progress indicator every 10 items
                    print(f"  Progress: {i}/{len(results)} ({success_count} migrated, {skip_count} skipped, {error_count} errors)")
            elif "already exists" in message:
                skip_count += 1
            else:
                error_count += 1
                print(f"  Error: {message}")

        print()
        print("=" * 60)
        print("Migration completed!")
        print(f"  Total results: {len(results)}")
        print(f"  Successfully migrated: {success_count}")
        print(f"  Skipped (duplicates): {skip_count}")
        print(f"  Errors: {error_count}")
        print("=" * 60)

        # Show statistics
        print()
        print("Current database statistics:")
        print()
        print("Tasks by state:")
        task_counts = await task_repo.count_by_state()
        for state, count in sorted(task_counts.items()):
            print(f"  {state}: {count}")

        print()
        print("Results by verdict:")
        result_counts = await result_repo.count_by_verdict()
        for verdict, count in sorted(result_counts.items()):
            print(f"  {verdict}: {count}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await close_pool()


if __name__ == "__main__":
    asyncio.run(main())
