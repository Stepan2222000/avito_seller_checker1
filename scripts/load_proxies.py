#!/usr/bin/env python3
"""
Script to load proxies from a file into the database.
Supports add and replace modes with interactive prompt.
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path to import seller_validator
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from seller_validator.config import Config
from seller_validator.database import create_pool, init_database, close_pool, ProxyRepository


async def load_proxies_from_file(file_path: Path) -> list[str]:
    """
    Read proxies from a text file (one per line).

    Args:
        file_path: Path to the proxies file

    Returns:
        List of proxy addresses (stripped and filtered)
    """
    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        return []

    with open(file_path, 'r', encoding='utf-8') as f:
        proxies = [line.strip() for line in f if line.strip()]

    print(f"Read {len(proxies)} proxies from {file_path}")
    return proxies


async def main():
    """Main function to load proxies into the database."""
    # Parse command line arguments
    if len(sys.argv) > 1:
        file_path = Path(sys.argv[1])
    else:
        config = Config()
        file_path = config.proxies_file

    print("=" * 60)
    print("Proxy Loader for Avito Seller Checker")
    print("=" * 60)
    print(f"Source file: {file_path}")
    print()

    # Load proxies from file
    proxies = await load_proxies_from_file(file_path)
    if not proxies:
        print("No proxies to load. Exiting.")
        return

    # Ask user for mode
    print("Choose operation mode:")
    print("  [a] Add - Add proxies to existing list (skip duplicates)")
    print("  [r] Replace - Delete all proxies and load new ones")
    print()

    mode = input("Enter mode (a/r): ").strip().lower()

    if mode not in ['a', 'r', 'add', 'replace']:
        print("Invalid mode. Please enter 'a' or 'r'.")
        return

    is_replace = mode in ['r', 'replace']

    # Confirm replace mode
    if is_replace:
        print()
        print("WARNING: Replace mode will delete ALL proxies (including blocked ones)!")
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
        proxy_repo = ProxyRepository(pool)

        # Show current statistics before operation
        if not is_replace:
            print()
            print("Current proxy statistics:")
            counts = await proxy_repo.count_proxies()
            for key, value in sorted(counts.items()):
                print(f"  {key}: {value}")

        # Execute operation
        print()
        print(f"{'Replacing' if is_replace else 'Adding'} {len(proxies)} proxies...")
        inserted_count = await proxy_repo.load_proxies(proxies, replace=is_replace)

        print()
        print("=" * 60)
        print("Operation completed successfully!")
        print(f"  Mode: {'Replace' if is_replace else 'Add'}")
        print(f"  Total proxies processed: {len(proxies)}")
        print(f"  New proxies inserted: {inserted_count}")
        print(f"  Duplicates skipped: {len(proxies) - inserted_count}")
        print("=" * 60)

        # Show proxy statistics after operation
        print()
        print("Current proxy statistics:")
        counts = await proxy_repo.count_proxies()
        for key, value in sorted(counts.items()):
            print(f"  {key}: {value}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await close_pool()


if __name__ == "__main__":
    asyncio.run(main())
