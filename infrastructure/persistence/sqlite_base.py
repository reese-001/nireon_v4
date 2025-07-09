# nireon_v4/infrastructure/persistence/sqlite_base.py
import logging
import sqlite3
import json
from typing import Any, Dict, Optional, List, TypeVar, Generic
from pathlib import Path
from contextlib import contextmanager
from threading import Lock
import time
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

T = TypeVar('T')

class SQLiteConfig:
    """Shared configuration for SQLite-based repositories"""
    def __init__(self, config: Dict[str, Any]):
        self.db_path = Path(config.get('db_path', 'runtime/nireon.db'))
        self.enable_wal_mode = config.get('enable_wal_mode', True)
        self.timeout = config.get('timeout', 30.0)
        self.retry_attempts = config.get('retry_attempts', 3)
        self.retry_delay = config.get('retry_delay', 0.1)
        self.pool_size = config.get('pool_size', 5)
        
        # Performance settings
        self.page_size = config.get('page_size', 4096)
        self.cache_size = config.get('cache_size', 2000)
        self.mmap_size = config.get('mmap_size', 268435456)  # 256MB
        self.synchronous = config.get('synchronous', 'NORMAL')
        self.temp_store = config.get('temp_store', 'MEMORY')

class SQLiteBaseRepository(ABC, Generic[T]):
    """
    Base class for SQLite-based repositories with:
    - Connection pooling
    - Retry logic
    - Common optimizations
    - Thread safety
    - Shared configuration
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = SQLiteConfig(config)
        self.operation_count = 0
        self._lock = Lock()
        
        # Connection pool
        self._conn_pool: List[sqlite3.Connection] = []
        self._pool_lock = Lock()
        
        # Initialize database
        self._ensure_database_exists()
        self._init_database_schema()
        
        logger.info(f'{self.__class__.__name__} initialized with db: {self.config.db_path}')

    def _ensure_database_exists(self):
        """Ensure database directory exists"""
        self.config.db_path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection from pool or create new one"""
        conn = None
        try:
            # Try to get from pool
            with self._pool_lock:
                if self._conn_pool:
                    conn = self._conn_pool.pop()
                    
            if conn is None:
                conn = self._create_connection()
                
            yield conn
            
        finally:
            if conn:
                # Return to pool if space available
                with self._pool_lock:
                    if len(self._conn_pool) < self.config.pool_size:
                        self._conn_pool.append(conn)
                    else:
                        conn.close()

    def _create_connection(self) -> sqlite3.Connection:
        """Create a new database connection with optimizations"""
        conn = sqlite3.connect(
            self.config.db_path, 
            timeout=self.config.timeout,
            check_same_thread=False
        )
        
        # Enable optimizations
        if self.config.enable_wal_mode:
            conn.execute('PRAGMA journal_mode=WAL')
        
        conn.execute('PRAGMA foreign_keys=ON')
        conn.execute(f'PRAGMA page_size={self.config.page_size}')
        conn.execute(f'PRAGMA cache_size={self.config.cache_size}')
        conn.execute(f'PRAGMA synchronous={self.config.synchronous}')
        conn.execute(f'PRAGMA temp_store={self.config.temp_store}')
        conn.execute(f'PRAGMA mmap_size={self.config.mmap_size}')
        
        # Set row factory for dict-like access
        conn.row_factory = sqlite3.Row
        
        return conn

    def _execute_with_retry(self, func, *args, **kwargs):
        """Execute a function with retry logic for database operations"""
        last_error = None
        for attempt in range(self.config.retry_attempts):
            try:
                return func(*args, **kwargs)
            except sqlite3.OperationalError as e:
                last_error = e
                if "database is locked" in str(e) and attempt < self.config.retry_attempts - 1:
                    time.sleep(self.config.retry_delay * (attempt + 1))
                    continue
                raise
        raise last_error

    def _increment_operation_count(self, count: int = 1):
        """Thread-safe operation count increment"""
        with self._lock:
            self.operation_count += count

    @abstractmethod
    def _init_database_schema(self):
        """Initialize database schema - must be implemented by subclasses"""
        pass

    def execute_query(self, query: str, params: tuple = None):
        """Execute a query with retry logic"""
        def _execute():
            with self._get_connection() as conn:
                cursor = conn.execute(query, params or ())
                return cursor.fetchall()
        
        return self._execute_with_retry(_execute)

    def execute_write(self, query: str, params: tuple = None):
        """Execute a write operation with retry logic"""
        def _execute():
            with self._get_connection() as conn:
                conn.execute(query, params or ())
                conn.commit()
                
        self._execute_with_retry(_execute)
        self._increment_operation_count()

    def execute_many(self, query: str, params_list: List[tuple]):
        """Execute many write operations in a single transaction"""
        if not params_list:
            return
            
        def _execute():
            with self._get_connection() as conn:
                conn.executemany(query, params_list)
                conn.commit()
                
        self._execute_with_retry(_execute)
        self._increment_operation_count(len(params_list))

    def vacuum_database(self) -> bool:
        """Vacuum the database (optimization operation)"""
        try:
            # Vacuum requires exclusive access
            with self._pool_lock:
                for conn in self._conn_pool:
                    conn.close()
                self._conn_pool.clear()
                
            with sqlite3.connect(self.config.db_path, timeout=self.config.timeout) as conn:
                conn.execute('VACUUM')
                conn.execute('ANALYZE')
                
            logger.info(f'{self.__class__.__name__}: Database vacuum completed')
            return True
            
        except Exception as e:
            logger.error(f'Error vacuuming database: {e}')
            return False

    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        stats = {
            'operation_count': self.operation_count,
            'database_path': str(self.config.db_path),
            'pool_size': len(self._conn_pool),
            'wal_mode_enabled': self.config.enable_wal_mode,
        }
        
        # Get database size
        if self.config.db_path.exists():
            stats['database_size_bytes'] = self.config.db_path.stat().st_size
            
            # Get page stats
            with self._get_connection() as conn:
                cursor = conn.execute('PRAGMA page_count')
                stats['page_count'] = cursor.fetchone()[0]
                
                cursor = conn.execute('PRAGMA page_size')
                stats['page_size'] = cursor.fetchone()[0]
                
        return stats

    def close(self):
        """Close all database connections and cleanup resources"""
        with self._pool_lock:
            for conn in self._conn_pool:
                conn.close()
            self._conn_pool.clear()
            
        logger.info(f'{self.__class__.__name__} closed')

    def __del__(self):
        """Cleanup on deletion"""
        try:
            self.close()
        except:
            pass

# Utility functions for common SQL operations

def json_array_contains(column: str, value: str) -> str:
    """SQL to check if JSON array contains a value"""
    return f"EXISTS (SELECT 1 FROM json_each({column}) WHERE value = ?)"

def json_array_append(column: str, value: str) -> str:
    """SQL to append to JSON array if not exists"""
    return f"""
        json_insert(
            {column},
            '$[#]', ?
        )
        WHERE NOT {json_array_contains(column, '?')}
    """

def build_in_clause(count: int) -> str:
    """Build IN clause with proper number of placeholders"""
    return f"({','.join('?' * count)})"