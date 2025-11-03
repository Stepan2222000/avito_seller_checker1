"""
Database module for avito_seller_checker.
Provides asyncpg-based connection pooling and repositories.
"""

from .connection import create_pool, init_database, close_pool
from .repositories import TaskRepository, ResultRepository, ProxyRepository

__all__ = [
    'create_pool',
    'init_database',
    'close_pool',
    'TaskRepository',
    'ResultRepository',
    'ProxyRepository',
]
