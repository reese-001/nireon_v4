import logging
import json
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
from threading import Lock

from domain.ports.idea_repository_port import IdeaRepositoryPort
from domain.ideas.idea import Idea


logger = logging.getLogger(__name__)

# Import the shared SQLite base only when needed
try:
    from infrastructure.persistence.sqlite_base import SQLiteBaseRepository, json_array_contains, build_in_clause
except ImportError:
    SQLiteBaseRepository = None
    logger.warning("SQLiteBaseRepository not available, SQLite backend will use inline implementation")


class IdeaRepository(IdeaRepositoryPort):
    """Generic Idea Repository for NIREON V4 that supports multiple database backends
    
    Now uses SQLiteBaseRepository for SQLite operations to reduce code duplication
    and ensure consistent optimizations across all SQLite-based repositories.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.provider = self.config.get('provider', 'sqlite')
        
        # Initialize based on provider
        if self.provider == 'sqlite' and SQLiteBaseRepository:
            # Use the optimized SQLite repository
            self._impl = SQLiteIdeaRepository(self.config)
        elif self.provider == 'memory':
            self._impl = InMemoryIdeaRepository(self.config)
        elif self.provider == 'postgres':
            logger.warning('PostgreSQL backend not implemented, using memory')
            self._impl = InMemoryIdeaRepository(self.config)
        else:
            logger.warning(f"Unknown provider '{self.provider}', using memory")
            self._impl = InMemoryIdeaRepository(self.config)
            
        logger.info(f'IdeaRepository initialized (provider: {self.provider})')

    # Delegate all methods to implementation
    def save(self, idea: Idea) -> None:
        self._impl.save(idea)

    def save_batch(self, ideas: List[Idea]) -> None:
        self._impl.save_batch(ideas)

    def get_by_id(self, idea_id: str) -> Optional[Idea]:
        return self._impl.get_by_id(idea_id)

    def get_by_ids(self, idea_ids: List[str]) -> List[Idea]:
        return self._impl.get_by_ids(idea_ids)

    def get_all(self) -> List[Idea]:
        return self._impl.get_all()

    def get_by_parent_id(self, parent_id: str) -> List[Idea]:
        return self._impl.get_by_parent_id(parent_id)
        
    def get_high_trust_ideas(self, min_trust: float, limit: int) -> List[Idea]:
        return self._impl.get_high_trust_ideas(min_trust, limit)

    def update_scores(self, idea_id: str, trust_score: float, novelty_score: Optional[float], is_stable: Optional[bool]) -> bool:
        return self._impl.update_scores(idea_id, trust_score, novelty_score, is_stable)

    def add_child_relationship(self, parent_id: str, child_id: str) -> bool:
        return self._impl.add_child_relationship(parent_id, child_id)

    def add_world_fact(self, idea_id: str, fact_id: str) -> bool:
        return self._impl.add_world_fact(idea_id, fact_id)

    def vacuum_database(self) -> bool:
        return self._impl.vacuum_database()

    def get_stats(self) -> Dict[str, Any]:
        return self._impl.get_stats()

    def close(self):
        if hasattr(self._impl, 'close'):
            self._impl.close()

    @property
    def operation_count(self) -> int:
        return getattr(self._impl, 'operation_count', 0)


class InMemoryIdeaRepository:
    """In-memory implementation of IdeaRepository"""
    
    def __init__(self, config: Dict[str, Any]):
        self._ideas: Dict[str, Idea] = {}
        self._child_relationships: Dict[str, List[str]] = {}
        self._world_facts: Dict[str, List[str]] = {}
        self._lock = Lock()
        self.operation_count = 0
        logger.info('In-memory idea repository initialized')

    def save(self, idea: Idea) -> None:
        with self._lock:
            self.operation_count += 1
            self._ideas[idea.idea_id] = idea
            
            # Update relationships
            for parent_id in idea.parent_ids:
                if parent_id not in self._child_relationships:
                    self._child_relationships[parent_id] = []
                if idea.idea_id not in self._child_relationships[parent_id]:
                    self._child_relationships[parent_id].append(idea.idea_id)
                    
            logger.debug(f"Saved idea '{idea.idea_id}' (memory, op #{self.operation_count})")

    def save_batch(self, ideas: List[Idea]) -> None:
        with self._lock:
            self.operation_count += len(ideas)
            for idea in ideas:
                self._ideas[idea.idea_id] = idea
                
                # Update relationships
                for parent_id in idea.parent_ids:
                    if parent_id not in self._child_relationships:
                        self._child_relationships[parent_id] = []
                    if idea.idea_id not in self._child_relationships[parent_id]:
                        self._child_relationships[parent_id].append(idea.idea_id)

    def get_by_id(self, idea_id: str) -> Optional[Idea]:
        with self._lock:
            self.operation_count += 1
            return self._ideas.get(idea_id)

    def get_by_ids(self, idea_ids: List[str]) -> List[Idea]:
        with self._lock:
            self.operation_count += 1
            return [self._ideas[id] for id in idea_ids if id in self._ideas]

    def get_all(self) -> List[Idea]:
        with self._lock:
            self.operation_count += 1
            return list(self._ideas.values())

    def get_by_parent_id(self, parent_id: str) -> List[Idea]:
        with self._lock:
            self.operation_count += 1
            children_ids = self._child_relationships.get(parent_id, [])
            return [self._ideas[cid] for cid in children_ids if cid in self._ideas]

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {'operation_count': self.operation_count, 'total_ideas': len(self._ideas), 'total_relationships': sum((len(children) for children in self._child_relationships.values())), 'total_world_facts': sum((len(facts) for facts in self._world_facts.values())), 'repository_type': 'memory', 'provider': 'memory'}

    def get_high_trust_ideas(self, min_trust: float, limit: int) -> List[Idea]:
        with self._lock:
            self.operation_count += 1
            high_trust_ideas = []
            for idea in self._ideas.values():
                if hasattr(idea, 'trust_score') and idea.trust_score is not None and idea.trust_score >= min_trust:
                    high_trust_ideas.append(idea)
            high_trust_ideas.sort(key=lambda i: i.trust_score, reverse=True)
            return high_trust_ideas[:limit]
        
    def update_scores(self, idea_id: str, trust_score: float, novelty_score: Optional[float], is_stable: Optional[bool]) -> bool:
        with self._lock:
            if idea_id in self._ideas:
                self.operation_count += 1
                idea = self._ideas[idea_id]
                idea.trust_score = trust_score
                idea.novelty_score = novelty_score
                idea.is_stable = is_stable
                logger.debug(f"Updated scores for idea '{idea_id}' (memory)")
                return True
            return False

    def add_child_relationship(self, parent_id: str, child_id: str) -> bool:
        with self._lock:
            self.operation_count += 1
            
            if parent_id not in self._ideas or child_id not in self._ideas:
                return False

            if parent_id not in self._child_relationships:
                self._child_relationships[parent_id] = []
            if child_id not in self._child_relationships[parent_id]:
                self._child_relationships[parent_id].append(child_id)

            # Update idea objects
            if child_id not in self._ideas[parent_id].children:
                self._ideas[parent_id].children.append(child_id)
            if parent_id not in self._ideas[child_id].parent_ids:
                self._ideas[child_id].parent_ids.append(parent_id)

            return True

    def add_world_fact(self, idea_id: str, fact_id: str) -> bool:
        with self._lock:
            self.operation_count += 1
            
            if idea_id not in self._ideas:
                return False

            if idea_id not in self._world_facts:
                self._world_facts[idea_id] = []
            if fact_id not in self._world_facts[idea_id]:
                self._world_facts[idea_id].append(fact_id)

            # Update idea object
            if fact_id not in self._ideas[idea_id].world_facts:
                self._ideas[idea_id].world_facts.append(fact_id)

            return True

    def vacuum_database(self) -> bool:
        logger.debug('Memory backend does not require vacuum operation')
        return True
        
from infrastructure.persistence.sqlite_base import SQLiteBaseRepository, json_array_contains, build_in_clause  


class SQLiteIdeaRepository(SQLiteBaseRepository[Idea]):
    def _init_database_schema(self):
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ideas (
                    idea_id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    parent_ids TEXT DEFAULT '[]',
                    children TEXT DEFAULT '[]',
                    world_facts TEXT DEFAULT '[]',
                    timestamp TEXT NOT NULL,
                    step INTEGER DEFAULT -1,
                    method TEXT DEFAULT 'manual',
                    metadata TEXT DEFAULT '{}',
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    -- ADDED COLUMNS
                    trust_score REAL,
                    novelty_score REAL,
                    is_stable BOOLEAN
                )
            """)
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
            conn.execute("""
                CREATE TABLE IF NOT EXISTS idea_world_facts (
                    idea_id TEXT NOT NULL,
                    fact_id TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (idea_id, fact_id),
                    FOREIGN KEY (idea_id) REFERENCES ideas(idea_id) ON DELETE CASCADE
                )
            """)
            conn.execute('CREATE INDEX IF NOT EXISTS idx_ideas_timestamp ON ideas(timestamp)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_ideas_step ON ideas(step)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_ideas_method ON ideas(method)')
            # ADDED INDEXES
            conn.execute('CREATE INDEX IF NOT EXISTS idx_ideas_trust_score ON ideas(trust_score DESC) WHERE trust_score IS NOT NULL')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_ideas_is_stable ON ideas(is_stable) WHERE is_stable IS NOT NULL')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_relationships_parent ON idea_relationships(parent_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_relationships_child ON idea_relationships(child_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_world_facts_idea ON idea_world_facts(idea_id)')
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS update_ideas_timestamp 
                AFTER UPDATE ON ideas
                BEGIN
                    UPDATE ideas SET updated_at = CURRENT_TIMESTAMP WHERE idea_id = NEW.idea_id;
                END
            """)
            conn.commit()

    def save(self, idea: Idea) -> None:
        timestamp_str = idea.timestamp.isoformat()
        parent_ids_json = json.dumps(idea.parent_ids)
        children_json = json.dumps(idea.children)
        world_facts_json = json.dumps(idea.world_facts)
        metadata_json = json.dumps(idea.metadata)
        
        # ADDED score handling
        trust_score = getattr(idea, 'trust_score', None)
        novelty_score = getattr(idea, 'novelty_score', None)
        is_stable = getattr(idea, 'is_stable', None)

        def _save():
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO ideas 
                    (idea_id, text, parent_ids, children, world_facts, timestamp, step, method, metadata, trust_score, novelty_score, is_stable)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    idea.idea_id, idea.text, parent_ids_json, children_json, 
                    world_facts_json, timestamp_str, idea.step, idea.method, 
                    metadata_json, trust_score, novelty_score, is_stable
                ))
                conn.execute('DELETE FROM idea_relationships WHERE child_id = ?', (idea.idea_id,))
                if idea.parent_ids:
                    conn.executemany("""
                        INSERT OR IGNORE INTO idea_relationships (parent_id, child_id)
                        VALUES (?, ?)
                    """, [(pid, idea.idea_id) for pid in idea.parent_ids])
                if idea.children:
                    conn.executemany("""
                        INSERT OR IGNORE INTO idea_relationships (parent_id, child_id)
                        VALUES (?, ?)
                    """, [(idea.idea_id, cid) for cid in idea.children])
                conn.execute('DELETE FROM idea_world_facts WHERE idea_id = ?', (idea.idea_id,))
                if idea.world_facts:
                    conn.executemany("""
                        INSERT OR IGNORE INTO idea_world_facts (idea_id, fact_id)
                        VALUES (?, ?)
                    """, [(idea.idea_id, fid) for fid in idea.world_facts])
                conn.commit()

        self._execute_with_retry(_save)
        self._increment_operation_count()
        logger.debug(f"Saved idea '{idea.idea_id}'")

    def save_batch(self, ideas: List[Idea]) -> None:
        if not ideas:
            return
        ideas_data = []
        relationships_data = []
        world_facts_data = []
        for idea in ideas:
            timestamp_str = idea.timestamp.isoformat()
            parent_ids_json = json.dumps(idea.parent_ids)
            children_json = json.dumps(idea.children)
            world_facts_json = json.dumps(idea.world_facts)
            metadata_json = json.dumps(idea.metadata)
            
            # ADDED score handling for batch
            trust_score = getattr(idea, 'trust_score', None)
            novelty_score = getattr(idea, 'novelty_score', None)
            is_stable = getattr(idea, 'is_stable', None)

            ideas_data.append((
                idea.idea_id, idea.text, parent_ids_json, children_json, world_facts_json,
                timestamp_str, idea.step, idea.method, metadata_json,
                trust_score, novelty_score, is_stable
            ))
            for parent_id in idea.parent_ids:
                relationships_data.append((parent_id, idea.idea_id))
            for child_id in idea.children:
                relationships_data.append((idea.idea_id, child_id))
            for fact_id in idea.world_facts:
                world_facts_data.append((idea.idea_id, fact_id))

        self.execute_many("""
            INSERT OR REPLACE INTO ideas 
            (idea_id, text, parent_ids, children, world_facts, timestamp, step, method, metadata, trust_score, novelty_score, is_stable)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, ideas_data)
        
        if relationships_data:
            self.execute_many("""
                INSERT OR IGNORE INTO idea_relationships (parent_id, child_id)
                VALUES (?, ?)
            """, relationships_data)
        if world_facts_data:
            self.execute_many("""
                INSERT OR IGNORE INTO idea_world_facts (idea_id, fact_id)
                VALUES (?, ?)
            """, world_facts_data)
        logger.debug(f'Batch saved {len(ideas)} ideas')

    def get_by_id(self, idea_id: str) -> Optional[Idea]:
        """Retrieve an idea by ID"""
        rows = self.execute_query("""
            SELECT idea_id, text, parent_ids, children, world_facts, 
                   timestamp, step, method, metadata, trust_score, novelty_score, is_stable
            FROM ideas WHERE idea_id = ?
        """, (idea_id,))
        
        if not rows:
            return None
            
        return self._row_to_idea(rows[0])

    def get_by_ids(self, idea_ids: List[str]) -> List[Idea]:
        """Retrieve multiple ideas by IDs efficiently"""
        if not idea_ids:
            return []
            
        placeholders = build_in_clause(len(idea_ids))
        rows = self.execute_query(f"""
            SELECT idea_id, text, parent_ids, children, world_facts, 
                   timestamp, step, method, metadata, trust_score, novelty_score, is_stable
            FROM ideas WHERE idea_id IN {placeholders}
        """, tuple(idea_ids))
        
        return [self._row_to_idea(row) for row in rows]

    def get_all(self) -> List[Idea]:
        """Retrieve all ideas"""
        rows = self.execute_query("""
            SELECT idea_id, text, parent_ids, children, world_facts, 
                   timestamp, step, method, metadata, trust_score, novelty_score, is_stable
            FROM ideas ORDER BY timestamp
        """)
        
        return [self._row_to_idea(row) for row in rows]

    def get_by_parent_id(self, parent_id: str) -> List[Idea]:
        """Retrieve ideas by parent ID"""
        rows = self.execute_query("""
            SELECT i.idea_id, i.text, i.parent_ids, i.children, i.world_facts, 
                   i.timestamp, i.step, i.method, i.metadata, i.trust_score, i.novelty_score, i.is_stable
            FROM ideas i
            INNER JOIN idea_relationships r ON i.idea_id = r.child_id
            WHERE r.parent_id = ?
            ORDER BY i.timestamp
        """, (parent_id,))
        
        return [self._row_to_idea(row) for row in rows]

    def add_child_relationship(self, parent_id: str, child_id: str) -> bool:
        """Add a parent-child relationship between ideas"""
        def _add_relationship():
            with self._get_connection() as conn:
                # Check if both ideas exist
                cursor = conn.execute('SELECT COUNT(*) FROM ideas WHERE idea_id IN (?, ?)', 
                                    (parent_id, child_id))
                if cursor.fetchone()[0] != 2:
                    return False

                # Add relationship
                conn.execute("""
                    INSERT OR IGNORE INTO idea_relationships (parent_id, child_id)
                    VALUES (?, ?)
                """, (parent_id, child_id))

                # Update JSON arrays using proper JSON functions
                conn.execute(f"""
                    UPDATE ideas 
                    SET children = json_insert(children, '$[#]', ?)
                    WHERE idea_id = ? 
                    AND NOT {json_array_contains('children', '?')}
                """, (child_id, parent_id, child_id))

                conn.execute(f"""
                    UPDATE ideas 
                    SET parent_ids = json_insert(parent_ids, '$[#]', ?)
                    WHERE idea_id = ? 
                    AND NOT {json_array_contains('parent_ids', '?')}
                """, (parent_id, child_id, parent_id))

                conn.commit()
                return True
                
        result = self._execute_with_retry(_add_relationship)
        self._increment_operation_count()
        return result

    def add_world_fact(self, idea_id: str, fact_id: str) -> bool:
        """Add a world fact association to an idea"""
        def _add_fact():
            with self._get_connection() as conn:
                # Check if idea exists
                cursor = conn.execute('SELECT COUNT(*) FROM ideas WHERE idea_id = ?', (idea_id,))
                if cursor.fetchone()[0] == 0:
                    return False

                # Add world fact
                conn.execute("""
                    INSERT OR IGNORE INTO idea_world_facts (idea_id, fact_id)
                    VALUES (?, ?)
                """, (idea_id, fact_id))

                # Update JSON array
                conn.execute(f"""
                    UPDATE ideas 
                    SET world_facts = json_insert(world_facts, '$[#]', ?)
                    WHERE idea_id = ? 
                    AND NOT {json_array_contains('world_facts', '?')}
                """, (fact_id, idea_id, fact_id))

                conn.commit()
                return True
                
        result = self._execute_with_retry(_add_fact)
        self._increment_operation_count()
        return result

    def get_stats(self) -> Dict[str, Any]:
        """Get repository statistics"""
        base_stats = self.get_database_stats()
        
        # Get idea-specific stats
        with self._get_connection() as conn:
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
        
        base_stats.update({
            'total_ideas': total_ideas,
            'total_relationships': total_relationships,
            'total_world_facts': total_world_facts,
            'method_distribution': method_distribution,
            'step_distribution': step_distribution,
            'repository_type': 'sqlite',
            'provider': 'sqlite'
        })
        
        return base_stats

    def _row_to_idea(self, row) -> Idea:
        """Convert database row to Idea object"""
        # Parse JSON fields
        parent_ids = json.loads(row['parent_ids']) if row['parent_ids'] else []
        children = json.loads(row['children']) if row['children'] else []
        world_facts = json.loads(row['world_facts']) if row['world_facts'] else []
        metadata = json.loads(row['metadata']) if row['metadata'] else {}

        # Parse timestamp
        timestamp_str = row['timestamp']
        if timestamp_str:
            try:
                if '+' in timestamp_str or 'Z' in timestamp_str:
                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                else:
                    timestamp = datetime.fromisoformat(timestamp_str).replace(tzinfo=timezone.utc)
            except (ValueError, AttributeError) as e:
                logger.warning(f"Invalid timestamp format: {timestamp_str}, using current time. Error: {e}")
                timestamp = datetime.now(timezone.utc)
        else:
            timestamp = datetime.now(timezone.utc)

        idea = Idea(
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

        if 'trust_score' in row.keys() and row['trust_score'] is not None:
            idea.trust_score = row['trust_score']
        if 'novelty_score' in row.keys() and row['novelty_score'] is not None:
            idea.novelty_score = row['novelty_score']
        if 'is_stable' in row.keys() and row['is_stable'] is not None:
            idea.is_stable = bool(row['is_stable'])

        return idea

    def get_high_trust_ideas(self, min_trust: float, limit: int) -> List[Idea]:
        """Retrieve ideas with trust score above threshold, sorted by trust score"""
        rows = self.execute_query("""
            SELECT idea_id, text, parent_ids, children, world_facts,
                   timestamp, step, method, metadata, trust_score, novelty_score, is_stable
            FROM ideas
            WHERE trust_score IS NOT NULL AND trust_score >= ?
            ORDER BY trust_score DESC
            LIMIT ?
        """, (min_trust, limit))
        return [self._row_to_idea(row) for row in rows]

    def update_scores(self, idea_id: str, trust_score: float, novelty_score: Optional[float], is_stable: Optional[bool]) -> bool:
        def _update():
            with self._get_connection() as conn:
                cursor = conn.execute("""
                    UPDATE ideas
                    SET trust_score = ?, novelty_score = ?, is_stable = ?
                    WHERE idea_id = ?
                """, (trust_score, novelty_score, is_stable, idea_id))
                conn.commit()
                return cursor.rowcount > 0
        
        result = self._execute_with_retry(_update)
        if result:
            self._increment_operation_count()
            logger.debug(f"Updated scores for idea '{idea_id}' (sqlite)")
        return result