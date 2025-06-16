import logging
import sqlite3
import json
from typing import List, Optional, Dict, Any
from pathlib import Path
from datetime import datetime, timezone

from domain.ports.idea_repository_port import IdeaRepositoryPort
from domain.ideas.idea import Idea

logger = logging.getLogger(__name__)

class IdeaRepository(IdeaRepositoryPort):
    """Generic Idea Repository for NIREON V4 that supports multiple database backends"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.provider = self.config.get('provider', 'sqlite')
        self.operation_count = 0
        
        # Initialize based on provider
        if self.provider == 'sqlite':
            self._init_sqlite()
        elif self.provider == 'postgres':
            self._init_postgres()
        elif self.provider == 'memory':
            self._init_memory()
        else:
            logger.warning(f"Unknown provider '{self.provider}', falling back to memory")
            self._init_memory()
            
        logger.info(f'IdeaRepository initialized (provider: {self.provider})')

    def _init_sqlite(self):
        """Initialize SQLite backend"""
        self.db_path = Path(self.config.get('db_path', 'runtime/ideas.db'))
        self.enable_wal_mode = self.config.get('enable_wal_mode', True)
        self.timeout = self.config.get('timeout', 30.0)
        
        self._init_sqlite_database()
        logger.info(f'SQLite backend initialized (db: {self.db_path})')

    def _init_postgres(self):
        """Initialize PostgreSQL backend (placeholder)"""
        self.connection_string = self.config.get('connection_string', 'postgresql://localhost/nireon')
        self.pool_size = self.config.get('pool_size', 5)
        self.timeout = self.config.get('timeout', 30.0)
        
        # For now, fallback to memory until PostgreSQL is fully implemented
        logger.warning('PostgreSQL backend not fully implemented, using memory backend')
        self._init_memory()

    def _init_memory(self):
        """Initialize in-memory backend"""
        self._ideas: Dict[str, Idea] = {}
        self._child_relationships: Dict[str, List[str]] = {}
        self._world_facts: Dict[str, List[str]] = {}
        self._backend = 'memory'
        logger.info('In-memory backend initialized')

    def _init_sqlite_database(self):
        """Initialize SQLite database with proper schema"""
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            
            with sqlite3.connect(self.db_path, timeout=self.timeout) as conn:
                if self.enable_wal_mode:
                    conn.execute('PRAGMA journal_mode=WAL')
                conn.execute('PRAGMA foreign_keys=ON')
                
                # Create main ideas table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS ideas (
                        idea_id TEXT PRIMARY KEY,
                        text TEXT NOT NULL,
                        parent_ids TEXT DEFAULT '[]',  -- JSON array
                        children TEXT DEFAULT '[]',    -- JSON array
                        world_facts TEXT DEFAULT '[]', -- JSON array
                        timestamp TEXT NOT NULL,       -- ISO format
                        step INTEGER DEFAULT -1,
                        method TEXT DEFAULT 'manual',
                        metadata TEXT DEFAULT '{}',    -- JSON object
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create relationships table for efficient querying
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS idea_relationships (
                        parent_id TEXT NOT NULL,
                        child_id TEXT NOT NULL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (parent_id, child_id),
                        FOREIGN KEY (parent_id) REFERENCES ideas(idea_id) ON DELETE CASCADE,
                        FOREIGN KEY (child_id) REFERENCES ideas(idea_id) ON DELETE CASCADE
                    )
                """)
                
                # Create world facts table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS idea_world_facts (
                        idea_id TEXT NOT NULL,
                        fact_id TEXT NOT NULL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (idea_id, fact_id),
                        FOREIGN KEY (idea_id) REFERENCES ideas(idea_id) ON DELETE CASCADE
                    )
                """)
                
                # Create indexes for performance
                conn.execute('CREATE INDEX IF NOT EXISTS idx_ideas_timestamp ON ideas(timestamp)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_ideas_step ON ideas(step)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_ideas_method ON ideas(method)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_relationships_parent ON idea_relationships(parent_id)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_relationships_child ON idea_relationships(child_id)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_world_facts_idea ON idea_world_facts(idea_id)')
                
                # Create trigger to update timestamp
                conn.execute("""
                    CREATE TRIGGER IF NOT EXISTS update_ideas_timestamp 
                    AFTER UPDATE ON ideas
                    BEGIN
                        UPDATE ideas SET updated_at = CURRENT_TIMESTAMP WHERE idea_id = NEW.idea_id;
                    END
                """)
                
                conn.commit()
                logger.debug('SQLite database schema initialized successfully')
                
        except Exception as e:
            logger.error(f'Error initializing SQLite database: {e}')
            logger.warning('Falling back to memory backend')
            self._init_memory()

    def save(self, idea: Idea) -> None:
        """Save an idea to the repository"""
        self.operation_count += 1
        
        if hasattr(self, '_backend') and self._backend == 'memory':
            self._save_memory(idea)
        else:
            self._save_sqlite(idea)

    def _save_memory(self, idea: Idea) -> None:
        """Save idea to memory backend"""
        logger.debug(f"IdeaRepository: save idea '{idea.idea_id}' (memory, operation #{self.operation_count})")
        self._ideas[idea.idea_id] = idea

    def _save_sqlite(self, idea: Idea) -> None:
        """Save idea to SQLite backend"""
        try:
            with sqlite3.connect(self.db_path, timeout=self.timeout) as conn:
                timestamp_str = idea.timestamp.isoformat()
                parent_ids_json = json.dumps(idea.parent_ids)
                children_json = json.dumps(idea.children)
                world_facts_json = json.dumps(idea.world_facts)
                metadata_json = json.dumps(idea.metadata)
                
                conn.execute("""
                    INSERT OR REPLACE INTO ideas 
                    (idea_id, text, parent_ids, children, world_facts, timestamp, step, method, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (idea.idea_id, idea.text, parent_ids_json, children_json, 
                      world_facts_json, timestamp_str, idea.step, idea.method, metadata_json))
                
                # Update relationships
                conn.execute('DELETE FROM idea_relationships WHERE child_id = ?', (idea.idea_id,))
                for parent_id in idea.parent_ids:
                    conn.execute("""
                        INSERT OR IGNORE INTO idea_relationships (parent_id, child_id)
                        VALUES (?, ?)
                    """, (parent_id, idea.idea_id))
                
                for child_id in idea.children:
                    conn.execute("""
                        INSERT OR IGNORE INTO idea_relationships (parent_id, child_id)
                        VALUES (?, ?)
                    """, (idea.idea_id, child_id))
                
                # Update world facts
                conn.execute('DELETE FROM idea_world_facts WHERE idea_id = ?', (idea.idea_id,))
                for fact_id in idea.world_facts:
                    conn.execute("""
                        INSERT OR IGNORE INTO idea_world_facts (idea_id, fact_id)
                        VALUES (?, ?)
                    """, (idea.idea_id, fact_id))
                
                conn.commit()
                logger.debug(f"Saved idea '{idea.idea_id}' to SQLite (op #{self.operation_count})")
                
        except Exception as e:
            logger.error(f"Error saving idea '{idea.idea_id}': {e}")
            raise

    def get_by_id(self, idea_id: str) -> Optional[Idea]:
        """Retrieve an idea by ID"""
        self.operation_count += 1
        
        if hasattr(self, '_backend') and self._backend == 'memory':
            return self._get_by_id_memory(idea_id)
        else:
            return self._get_by_id_sqlite(idea_id)

    def _get_by_id_memory(self, idea_id: str) -> Optional[Idea]:
        """Get idea from memory backend"""
        idea = self._ideas.get(idea_id)
        logger.debug(f"IdeaRepository: get_by_id '{idea_id}' -> {'FOUND' if idea else 'None'} (memory, op #{self.operation_count})")
        return idea

    def _get_by_id_sqlite(self, idea_id: str) -> Optional[Idea]:
        """Get idea from SQLite backend"""
        try:
            with sqlite3.connect(self.db_path, timeout=self.timeout) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("""
                    SELECT idea_id, text, parent_ids, children, world_facts, 
                           timestamp, step, method, metadata
                    FROM ideas WHERE idea_id = ?
                """, (idea_id,))
                
                row = cursor.fetchone()
                if not row:
                    return None
                
                idea = self._row_to_idea(row)
                logger.debug(f"Retrieved idea '{idea_id}' from SQLite (op #{self.operation_count})")
                return idea
                
        except Exception as e:
            logger.error(f"Error retrieving idea '{idea_id}': {e}")
            return None

    def get_all(self) -> List[Idea]:
        """Retrieve all ideas"""
        self.operation_count += 1
        
        if hasattr(self, '_backend') and self._backend == 'memory':
            return self._get_all_memory()
        else:
            return self._get_all_sqlite()

    def _get_all_memory(self) -> List[Idea]:
        """Get all ideas from memory backend"""
        all_ideas = list(self._ideas.values())
        logger.debug(f'IdeaRepository: get_all -> {len(all_ideas)} ideas (memory, op #{self.operation_count})')
        return all_ideas

    def _get_all_sqlite(self) -> List[Idea]:
        """Get all ideas from SQLite backend"""
        try:
            with sqlite3.connect(self.db_path, timeout=self.timeout) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("""
                    SELECT idea_id, text, parent_ids, children, world_facts, 
                           timestamp, step, method, metadata
                    FROM ideas ORDER BY timestamp
                """)
                
                ideas = []
                for row in cursor.fetchall():
                    idea = self._row_to_idea(row)
                    ideas.append(idea)
                
                logger.debug(f'Retrieved {len(ideas)} ideas from SQLite (op #{self.operation_count})')
                return ideas
                
        except Exception as e:
            logger.error(f'Error retrieving all ideas: {e}')
            return []

    def get_by_parent_id(self, parent_id: str) -> List[Idea]:
        """Retrieve ideas by parent ID"""
        self.operation_count += 1
        
        if hasattr(self, '_backend') and self._backend == 'memory':
            return self._get_by_parent_id_memory(parent_id)
        else:
            return self._get_by_parent_id_sqlite(parent_id)

    def _get_by_parent_id_memory(self, parent_id: str) -> List[Idea]:
        """Get ideas by parent ID from memory backend"""
        children_ids = self._child_relationships.get(parent_id, [])
        children = [self._ideas[cid] for cid in children_ids if cid in self._ideas]
        logger.debug(f"IdeaRepository: get_by_parent_id '{parent_id}' -> {len(children)} ideas (memory, op #{self.operation_count})")
        return children

    def _get_by_parent_id_sqlite(self, parent_id: str) -> List[Idea]:
        """Get ideas by parent ID from SQLite backend"""
        try:
            with sqlite3.connect(self.db_path, timeout=self.timeout) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("""
                    SELECT i.idea_id, i.text, i.parent_ids, i.children, i.world_facts, 
                           i.timestamp, i.step, i.method, i.metadata
                    FROM ideas i
                    INNER JOIN idea_relationships r ON i.idea_id = r.child_id
                    WHERE r.parent_id = ?
                    ORDER BY i.timestamp
                """, (parent_id,))
                
                ideas = []
                for row in cursor.fetchall():
                    idea = self._row_to_idea(row)
                    ideas.append(idea)
                
                logger.debug(f"Retrieved {len(ideas)} children for '{parent_id}' from SQLite (op #{self.operation_count})")
                return ideas
                
        except Exception as e:
            logger.error(f"Error retrieving children for '{parent_id}': {e}")
            return []

    def add_child_relationship(self, parent_id: str, child_id: str) -> bool:
        """Add a parent-child relationship between ideas"""
        self.operation_count += 1
        
        if hasattr(self, '_backend') and self._backend == 'memory':
            return self._add_child_relationship_memory(parent_id, child_id)
        else:
            return self._add_child_relationship_sqlite(parent_id, child_id)

    def _add_child_relationship_memory(self, parent_id: str, child_id: str) -> bool:
        """Add relationship in memory backend"""
        if parent_id not in self._ideas or child_id not in self._ideas:
            logger.warning(f"IdeaRepository: add_child_relationship failed. Parent '{parent_id}' or child '{child_id}' not found.")
            return False

        if parent_id not in self._child_relationships:
            self._child_relationships[parent_id] = []
        if child_id not in self._child_relationships[parent_id]:
            self._child_relationships[parent_id].append(child_id)

        # Update idea objects
        if hasattr(self._ideas[parent_id], 'children') and isinstance(self._ideas[parent_id].children, list):
            if child_id not in self._ideas[parent_id].children:
                self._ideas[parent_id].children.append(child_id)

        if hasattr(self._ideas[child_id], 'parent_ids') and isinstance(self._ideas[child_id].parent_ids, list):
            if parent_id not in self._ideas[child_id].parent_ids:
                self._ideas[child_id].parent_ids.append(parent_id)

        logger.debug(f"IdeaRepository: add_child_relationship parent='{parent_id}', child='{child_id}' -> True (memory, op #{self.operation_count})")
        return True

    def _add_child_relationship_sqlite(self, parent_id: str, child_id: str) -> bool:
        """Add relationship in SQLite backend"""
        try:
            with sqlite3.connect(self.db_path, timeout=self.timeout) as conn:
                # Check if both ideas exist
                cursor = conn.execute('SELECT COUNT(*) FROM ideas WHERE idea_id IN (?, ?)', (parent_id, child_id))
                if cursor.fetchone()[0] != 2:
                    logger.warning(f"Cannot add relationship: one or both ideas don't exist ({parent_id}, {child_id})")
                    return False

                # Add relationship
                conn.execute("""
                    INSERT OR IGNORE INTO idea_relationships (parent_id, child_id)
                    VALUES (?, ?)
                """, (parent_id, child_id))

                # Update JSON arrays in ideas table
                conn.execute("""
                    UPDATE ideas 
                    SET children = json_insert(
                        CASE WHEN json_valid(children) THEN children ELSE '[]' END,
                        '$[#]', ?
                    )
                    WHERE idea_id = ? AND NOT json_extract(children, '$') LIKE '%' || ? || '%'
                """, (child_id, parent_id, child_id))

                conn.execute("""
                    UPDATE ideas 
                    SET parent_ids = json_insert(
                        CASE WHEN json_valid(parent_ids) THEN parent_ids ELSE '[]' END,
                        '$[#]', ?
                    )
                    WHERE idea_id = ? AND NOT json_extract(parent_ids, '$') LIKE '%' || ? || '%'
                """, (parent_id, child_id, parent_id))

                conn.commit()
                logger.debug(f'Added relationship {parent_id}->{child_id} in SQLite (op #{self.operation_count})')
                return True
                
        except Exception as e:
            logger.error(f'Error adding relationship {parent_id}->{child_id}: {e}')
            return False

    def add_world_fact(self, idea_id: str, fact_id: str) -> bool:
        """Add a world fact association to an idea"""
        self.operation_count += 1
        
        if hasattr(self, '_backend') and self._backend == 'memory':
            return self._add_world_fact_memory(idea_id, fact_id)
        else:
            return self._add_world_fact_sqlite(idea_id, fact_id)

    def _add_world_fact_memory(self, idea_id: str, fact_id: str) -> bool:
        """Add world fact in memory backend"""
        if idea_id not in self._ideas:
            logger.warning(f"IdeaRepository: add_world_fact failed. Idea '{idea_id}' not found.")
            return False

        if idea_id not in self._world_facts:
            self._world_facts[idea_id] = []
        if fact_id not in self._world_facts[idea_id]:
            self._world_facts[idea_id].append(fact_id)

        # Update idea object
        if hasattr(self._ideas[idea_id], 'world_facts') and isinstance(self._ideas[idea_id].world_facts, list):
            if fact_id not in self._ideas[idea_id].world_facts:
                self._ideas[idea_id].world_facts.append(fact_id)

        logger.debug(f"IdeaRepository: add_world_fact idea='{idea_id}', fact='{fact_id}' -> True (memory, op #{self.operation_count})")
        return True

    def _add_world_fact_sqlite(self, idea_id: str, fact_id: str) -> bool:
        """Add world fact in SQLite backend"""
        try:
            with sqlite3.connect(self.db_path, timeout=self.timeout) as conn:
                # Check if idea exists
                cursor = conn.execute('SELECT COUNT(*) FROM ideas WHERE idea_id = ?', (idea_id,))
                if cursor.fetchone()[0] == 0:
                    logger.warning(f"Cannot add world fact: idea '{idea_id}' doesn't exist")
                    return False

                # Add world fact
                conn.execute("""
                    INSERT OR IGNORE INTO idea_world_facts (idea_id, fact_id)
                    VALUES (?, ?)
                """, (idea_id, fact_id))

                # Update JSON array in ideas table
                conn.execute("""
                    UPDATE ideas 
                    SET world_facts = json_insert(
                        CASE WHEN json_valid(world_facts) THEN world_facts ELSE '[]' END,
                        '$[#]', ?
                    )
                    WHERE idea_id = ? AND NOT json_extract(world_facts, '$') LIKE '%' || ? || '%'
                """, (fact_id, idea_id, fact_id))

                conn.commit()
                logger.debug(f'Added world fact {fact_id} to {idea_id} in SQLite (op #{self.operation_count})')
                return True
                
        except Exception as e:
            logger.error(f'Error adding world fact {fact_id} to {idea_id}: {e}')
            return False

    def vacuum_database(self) -> bool:
        """Vacuum the database (optimization operation)"""
        if hasattr(self, '_backend') and self._backend == 'memory':
            logger.debug('Memory backend does not require vacuum operation')
            return True
            
        try:
            with sqlite3.connect(self.db_path, timeout=self.timeout) as conn:
                conn.execute('VACUUM')
                logger.info('Database vacuum completed successfully')
                return True
        except Exception as e:
            logger.error(f'Error vacuuming database: {e}')
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get repository statistics"""
        if hasattr(self, '_backend') and self._backend == 'memory':
            return self._get_stats_memory()
        else:
            return self._get_stats_sqlite()

    def _get_stats_memory(self) -> Dict[str, Any]:
        """Get stats for memory backend"""
        return {
            'operation_count': self.operation_count,
            'total_ideas': len(self._ideas),
            'total_relationships': sum(len(children) for children in self._child_relationships.values()),
            'total_world_facts': sum(len(facts) for facts in self._world_facts.values()),
            'repository_type': 'memory',
            'provider': self.provider
        }

    def _get_stats_sqlite(self) -> Dict[str, Any]:
        """Get stats for SQLite backend"""
        try:
            with sqlite3.connect(self.db_path, timeout=self.timeout) as conn:
                cursor = conn.execute('SELECT COUNT(*) FROM ideas')
                total_ideas = cursor.fetchone()[0]
                
                cursor = conn.execute('SELECT COUNT(*) FROM idea_relationships')
                total_relationships = cursor.fetchone()[0]
                
                cursor = conn.execute('SELECT COUNT(*) FROM idea_world_facts')
                total_world_facts = cursor.fetchone()[0]
                
                cursor = conn.execute('SELECT method, COUNT(*) FROM ideas GROUP BY method')
                method_distribution = dict(cursor.fetchall())
                
                cursor = conn.execute('SELECT step, COUNT(*) FROM ideas WHERE step >= 0 GROUP BY step ORDER BY step')
                step_distribution = dict(cursor.fetchall())
                
                db_size = self.db_path.stat().st_size if self.db_path.exists() else 0
                
                return {
                    'operation_count': self.operation_count,
                    'total_ideas': total_ideas,
                    'total_relationships': total_relationships,
                    'total_world_facts': total_world_facts,
                    'method_distribution': method_distribution,
                    'step_distribution': step_distribution,
                    'database_path': str(self.db_path),
                    'database_size_bytes': db_size,
                    'wal_mode_enabled': self.enable_wal_mode,
                    'repository_type': 'sqlite',
                    'provider': self.provider
                }
        except Exception as e:
            logger.error(f'Error getting repository stats: {e}')
            return {
                'operation_count': self.operation_count,
                'error': str(e),
                'repository_type': 'sqlite',
                'provider': self.provider
            }

    def _row_to_idea(self, row: sqlite3.Row) -> Idea:
        """Convert SQLite row to Idea object"""
        try:
            parent_ids = json.loads(row['parent_ids']) if row['parent_ids'] else []
        except (json.JSONDecodeError, TypeError):
            parent_ids = []

        try:
            children = json.loads(row['children']) if row['children'] else []
        except (json.JSONDecodeError, TypeError):
            children = []

        try:
            world_facts = json.loads(row['world_facts']) if row['world_facts'] else []
        except (json.JSONDecodeError, TypeError):
            world_facts = []

        try:
            metadata = json.loads(row['metadata']) if row['metadata'] else {}
        except (json.JSONDecodeError, TypeError):
            metadata = {}

        try:
            timestamp = datetime.fromisoformat(row['timestamp'].replace('Z', '+00:00'))
        except (ValueError, AttributeError):
            timestamp = datetime.now(timezone.utc)

        return Idea(
            idea_id=row['idea_id'],
            text=row['text'],
            parent_ids=parent_ids,
            children=children,
            world_facts=world_facts,
            timestamp=timestamp,
            step=row['step'] if row['step'] is not None else -1,
            method=row['method'] or 'manual',
            metadata=metadata
        )