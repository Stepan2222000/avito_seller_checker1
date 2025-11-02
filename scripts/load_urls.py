#!/usr/bin/env python3
"""
Script to load URLs from a file into the database.
Supports add and replace modes with interactive prompt.
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path to import seller_validator
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from seller_validator.config import Config
from seller_validator.database import create_pool, init_database, close_pool, TaskRepository


async def load_urls_from_file(file_path: Path) -> list[str]:
    """
    Read URLs from a text file (one per line).

    Args:
        file_path: Path to the URLs file

    Returns:
        List of URLs (stripped and filtered)
    """
    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        return []

    with open(file_path, 'r', encoding='utf-8') as f:
        urls = [line.strip() for line in f if line.strip()]

    print(f"Read {len(urls)} URLs from {file_path}")
    return urls


async def main():
    """Main function to load URLs into the database."""
    # Parse command line arguments
    if len(sys.argv) > 1:
        file_path = Path(sys.argv[1])
    else:
        config = Config()
        file_path = config.urls_file

    print("=" * 60)
    print("URL Loader for Avito Seller Checker")
    print("=" * 60)
    print(f"Source file: {file_path}")
    print()

    # Load URLs from file
    urls = await load_urls_from_file(file_path)
    if not urls:
        print("No URLs to load. Exiting.")
        return

    # Ask user for mode
    print("Choose operation mode:")
    print("  [a] Add - Add URLs to existing tasks (skip duplicates)")
    print("  [r] Replace - Clear pending tasks and load new URLs")
    print()

    mode = input("Enter mode (a/r): ").strip().lower()

    if mode not in ['a', 'r', 'add', 'replace']:
        print("Invalid mode. Please enter 'a' or 'r'.")
        return

    is_replace = mode in ['r', 'replace']

    # Confirm replace mode
    if is_replace:
        print()
        print("WARNING: Replace mode will delete all PENDING tasks!")
        print("         (In-progress, completed, and failed tasks will not be touched)")
        confirm = input("Are you sure you want to continue? (yes/no): ").strip().lower()
        if confirm not in ['yes', 'y']:
            print("Operation cancelled.")
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

        # Create repository
        task_repo = TaskRepository(pool)

        # Execute operation
        if is_replace:
            print()
            print("Clearing pending tasks...")
            deleted_count = await task_repo.clear_pending_tasks()
            print(f"Deleted {deleted_count} pending tasks")

        print()
        print(f"{'Replacing' if is_replace else 'Adding'} {len(urls)} URLs...")
        inserted_count = await task_repo.add_tasks(urls)

        print()
        print("=" * 60)
        print("Operation completed successfully!")
        print(f"  Mode: {'Replace' if is_replace else 'Add'}")
        print(f"  Total URLs processed: {len(urls)}")
        print(f"  New tasks inserted: {inserted_count}")
        print(f"  Duplicates skipped: {len(urls) - inserted_count}")
        print("=" * 60)

        # Show task statistics
        print()
        print("Current task statistics:")
        counts = await task_repo.count_by_state()
        for state, count in sorted(counts.items()):
            print(f"  {state}: {count}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await close_pool()


if __name__ == "__main__":
    asyncio.run(main())
