# infrastructure/persistence/event_repository_enhanced.py

import json
import logging
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timezone
from dataclasses import dataclass
from pydantic import BaseModel

from .sqlite_base import SQLiteBaseRepository, build_in_clause

logger = logging.getLogger(__name__)


@dataclass
class EventRecord:
    """Enhanced event record for database storage with trust assessment fields"""
    event_id: str
    signal_type: str
    source_node_id: Optional[str]
    timestamp: datetime
    payload: Dict[str, Any]
    trust_score: Optional[float] = None
    novelty_score: Optional[float] = None
    parent_signal_ids: List[str] = None
    context_tags: Dict[str, str] = None
    # Add explicit fields for trust assessment
    is_stable: Optional[bool] = None
    target_id: Optional[str] = None
    target_type: Optional[str] = None
    assessment_rationale: Optional[str] = None


class EventRepository(SQLiteBaseRepository[EventRecord]):
    """
    Enhanced event repository that properly extracts and stores trust assessment data
    """
    
    def __init__(self, config: Dict[str, Any]):
        if 'db_path' not in config:
            config['db_path'] = config.get('shared_db_path', 'runtime/nireon_ideas.db')
        
        super().__init__(config)
        
        self.retention_days = config.get('event_retention_days', 30)
        self.batch_size = config.get('event_batch_size', 100)
        
        logger.info(f'EnhancedEventRepository initialized with shared db: {self.config.db_path}')
    
    def _init_database_schema(self):
        """Initialize enhanced event tables with trust assessment fields"""
        with self._get_connection() as conn:
            # Enhanced events table with trust assessment columns
            conn.execute('''
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL UNIQUE,
                    signal_type TEXT NOT NULL,
                    source_node_id TEXT,
                    timestamp TEXT NOT NULL,
                    payload TEXT,  -- JSON
                    trust_score REAL,
                    novelty_score REAL,
                    parent_signal_ids TEXT,  -- JSON array
                    context_tags TEXT,  -- JSON object
                    created_at REAL NOT NULL,
                    
                    -- Trust assessment specific fields
                    is_stable BOOLEAN,
                    target_id TEXT,
                    target_type TEXT,
                    assessment_rationale TEXT,
                    
                    -- Foreign key to ideas if event references an idea
                    related_idea_id TEXT,
                    FOREIGN KEY (related_idea_id) REFERENCES ideas(idea_id) ON DELETE SET NULL
                )
            ''')
            
            # Create indexes
            conn.execute('CREATE INDEX IF NOT EXISTS idx_events_signal_type ON events(signal_type)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_events_source_node ON events(source_node_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_events_created_at ON events(created_at)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_events_related_idea ON events(related_idea_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_events_trust_score ON events(trust_score) WHERE trust_score IS NOT NULL')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_events_is_stable ON events(is_stable) WHERE is_stable IS NOT NULL')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_events_target_id ON events(target_id) WHERE target_id IS NOT NULL')
            
            # Composite indexes for common queries
            conn.execute('CREATE INDEX IF NOT EXISTS idx_events_type_time ON events(signal_type, created_at DESC)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_events_trust_stable ON events(trust_score, is_stable) WHERE signal_type = "TrustAssessmentSignal"')
            
            # Event subscriptions and aggregations tables remain the same
            conn.execute('''
                CREATE TABLE IF NOT EXISTS event_subscriptions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    signal_type TEXT NOT NULL,
                    subscriber_id TEXT,
                    subscriber_info TEXT,
                    created_at REAL NOT NULL,
                    active BOOLEAN DEFAULT 1
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS event_aggregations (
                    signal_type TEXT NOT NULL,
                    date TEXT NOT NULL,
                    hour INTEGER NOT NULL,
                    count INTEGER DEFAULT 0,
                    avg_trust_score REAL,
                    avg_novelty_score REAL,
                    high_trust_stable_count INTEGER DEFAULT 0,
                    PRIMARY KEY (signal_type, date, hour)
                )
            ''')
            
            conn.commit()
    
    def save_event(self, event: Union[EventRecord, BaseModel, Dict[str, Any]]) -> None:
        """Save a single event with proper trust assessment extraction"""
        processed_event = self._process_event_data(event)
        self._save_events([processed_event])
    
    def save_events_batch(self, events: List[Union[EventRecord, BaseModel, Dict[str, Any]]]) -> None:
        """Save multiple events in a single transaction"""
        processed_events = [self._process_event_data(event) for event in events]
        self._save_events(processed_events)
    
    def _process_event_data(self, event_data: Union[EventRecord, BaseModel, Dict[str, Any]]) -> EventRecord:
        """Process various event data formats and extract trust assessment fields"""
        # Handle Pydantic models
        if hasattr(event_data, 'model_dump'):
            event_dict = event_data.model_dump()
        elif hasattr(event_data, 'dict'):
            event_dict = event_data.dict()
        elif isinstance(event_data, dict):
            event_dict = event_data
        elif isinstance(event_data, EventRecord):
            return event_data
        else:
            raise ValueError(f"Unsupported event data type: {type(event_data)}")
        
        # Extract base fields
        event_id = event_dict.get('signal_id', event_dict.get('event_id', f"evt_{datetime.now().timestamp()}"))
        signal_type = event_dict.get('signal_type', 'UnknownSignal')
        source_node_id = event_dict.get('source_node_id')
        
        # Handle timestamp
        timestamp = event_dict.get('timestamp')
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        elif not isinstance(timestamp, datetime):
            timestamp = datetime.now(timezone.utc)
        
        # Extract payload
        payload = event_dict.get('payload', {})
        
        # Extract trust assessment specific fields
        trust_score = event_dict.get('trust_score')
        novelty_score = event_dict.get('novelty_score')
        is_stable = None
        target_id = event_dict.get('target_id')
        target_type = event_dict.get('target_type')
        assessment_rationale = event_dict.get('assessment_rationale')
        
        # For TrustAssessmentSignal, extract from payload if needed
        if signal_type == 'TrustAssessmentSignal':
            # Trust score might be at the top level or in payload
            if trust_score is None and payload:
                trust_score = payload.get('trust_score')
            
            # Extract stability from payload
            if payload:
                is_stable = payload.get('is_stable')
                # Also check in assessment_details
                assessment_details = payload.get('assessment_details', {})
                if is_stable is None and assessment_details:
                    is_stable = assessment_details.get('is_stable')
                
                # Extract other fields from payload if not at top level
                if not target_id:
                    target_id = payload.get('target_id') or payload.get('idea_id')
                if not target_type:
                    target_type = payload.get('target_type')
                if not assessment_rationale:
                    assessment_rationale = payload.get('assessment_rationale')
        
        # Extract parent signal IDs and context tags
        parent_signal_ids = event_dict.get('parent_signal_ids', [])
        context_tags = event_dict.get('context_tags', {})
        
        return EventRecord(
            event_id=event_id,
            signal_type=signal_type,
            source_node_id=source_node_id,
            timestamp=timestamp,
            payload=payload,
            trust_score=trust_score,
            novelty_score=novelty_score,
            parent_signal_ids=parent_signal_ids,
            context_tags=context_tags,
            is_stable=is_stable,
            target_id=target_id,
            target_type=target_type,
            assessment_rationale=assessment_rationale
        )
    
    def _save_events(self, events: List[EventRecord]) -> None:
        """Internal method to save events with trust assessment fields"""
        if not events:
            return
        
        events_data = []
        aggregation_updates = {}
        
        for event in events:
            timestamp_str = event.timestamp.isoformat()
            created_at = datetime.now(timezone.utc).timestamp()
            
            # Extract related idea ID
            related_idea_id = event.target_id if event.target_type == 'Idea' else None
            if not related_idea_id and isinstance(event.payload, dict):
                related_idea_id = event.payload.get('idea_id') or event.payload.get('target_idea_id')
            
            events_data.append((
                event.event_id,
                event.signal_type,
                event.source_node_id,
                timestamp_str,
                json.dumps(event.payload, default=str),
                event.trust_score,
                event.novelty_score,
                json.dumps(event.parent_signal_ids) if event.parent_signal_ids else None,
                json.dumps(event.context_tags) if event.context_tags else None,
                created_at,
                event.is_stable,
                event.target_id,
                event.target_type,
                event.assessment_rationale,
                related_idea_id
            ))
            
            # Prepare aggregation data
            date_key = event.timestamp.strftime('%Y-%m-%d')
            hour_key = event.timestamp.hour
            agg_key = (event.signal_type, date_key, hour_key)
            
            if agg_key not in aggregation_updates:
                aggregation_updates[agg_key] = {
                    'count': 0,
                    'trust_scores': [],
                    'novelty_scores': [],
                    'high_trust_stable_count': 0
                }
            
            aggregation_updates[agg_key]['count'] += 1
            if event.trust_score is not None:
                aggregation_updates[agg_key]['trust_scores'].append(event.trust_score)
            if event.novelty_score is not None:
                aggregation_updates[agg_key]['novelty_scores'].append(event.novelty_score)
            
            # Count high trust stable ideas
            if event.signal_type == 'TrustAssessmentSignal' and event.trust_score and event.trust_score > 6.0 and event.is_stable:
                aggregation_updates[agg_key]['high_trust_stable_count'] += 1
        
        # Execute batch insert
        self.execute_many('''
            INSERT OR REPLACE INTO events (
                event_id, signal_type, source_node_id, timestamp,
                payload, trust_score, novelty_score, parent_signal_ids,
                context_tags, created_at, is_stable, target_id,
                target_type, assessment_rationale, related_idea_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', events_data)
        
        # Update aggregations
        self._update_aggregations(aggregation_updates)
        
        logger.debug(f"Saved {len(events)} events to database")
    
    def _update_aggregations(self, aggregation_updates: Dict[tuple, Dict]):
        """Update hourly aggregations with high trust stable count"""
        if not aggregation_updates:
            return
        
        with self._get_connection() as conn:
            for (signal_type, date, hour), data in aggregation_updates.items():
                avg_trust = sum(data['trust_scores']) / len(data['trust_scores']) if data['trust_scores'] else None
                avg_novelty = sum(data['novelty_scores']) / len(data['novelty_scores']) if data['novelty_scores'] else None
                
                conn.execute('''
                    INSERT INTO event_aggregations (
                        signal_type, date, hour, count, 
                        avg_trust_score, avg_novelty_score, high_trust_stable_count
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(signal_type, date, hour) DO UPDATE SET
                        count = count + ?,
                        avg_trust_score = CASE 
                            WHEN avg_trust_score IS NULL THEN ?
                            ELSE (avg_trust_score * count + ? * ?) / (count + ?)
                        END,
                        avg_novelty_score = CASE
                            WHEN avg_novelty_score IS NULL THEN ?
                            ELSE (avg_novelty_score * count + ? * ?) / (count + ?)
                        END,
                        high_trust_stable_count = high_trust_stable_count + ?
                ''', (
                    signal_type, date, hour, data['count'], 
                    avg_trust, avg_novelty, data['high_trust_stable_count'],
                    data['count'], 
                    avg_trust, avg_trust or 0, data['count'], data['count'],
                    avg_novelty, avg_novelty or 0, data['count'], data['count'],
                    data['high_trust_stable_count']
                ))
            
            conn.commit()
    
    def get_high_trust_stable_events(self, min_trust: float = 6.0, limit: int = 100) -> List[EventRecord]:
        """Get events with high trust scores that are stable"""
        rows = self.execute_query('''
            SELECT event_id, signal_type, source_node_id, timestamp,
                   payload, trust_score, novelty_score, parent_signal_ids,
                   context_tags, is_stable, target_id, target_type,
                   assessment_rationale
            FROM events
            WHERE signal_type = 'TrustAssessmentSignal'
              AND trust_score > ?
              AND is_stable = 1
            ORDER BY trust_score DESC, created_at DESC
            LIMIT ?
        ''', (min_trust, limit))
        
        return [self._row_to_event(row) for row in rows]
    
    def get_trust_assessment_summary(self) -> Dict[str, Any]:
        """Get summary statistics for trust assessments"""
        with self._get_connection() as conn:
            # Overall stats
            cursor = conn.execute('''
                SELECT 
                    COUNT(*) as total_assessments,
                    AVG(trust_score) as avg_trust_score,
                    MIN(trust_score) as min_trust_score,
                    MAX(trust_score) as max_trust_score,
                    SUM(CASE WHEN is_stable = 1 THEN 1 ELSE 0 END) as stable_count,
                    SUM(CASE WHEN trust_score > 6.0 THEN 1 ELSE 0 END) as high_trust_count,
                    SUM(CASE WHEN trust_score > 6.0 AND is_stable = 1 THEN 1 ELSE 0 END) as high_trust_stable_count
                FROM events
                WHERE signal_type = 'TrustAssessmentSignal'
                  AND trust_score IS NOT NULL
            ''')
            
            stats = cursor.fetchone()
            
            # Distribution by stability
            cursor = conn.execute('''
                SELECT 
                    is_stable,
                    COUNT(*) as count,
                    AVG(trust_score) as avg_trust_score
                FROM events
                WHERE signal_type = 'TrustAssessmentSignal'
                  AND trust_score IS NOT NULL
                GROUP BY is_stable
            ''')
            
            stability_distribution = {}
            for row in cursor:
                stability_key = 'stable' if row[0] else 'unstable' if row[0] is not None else 'unknown'
                stability_distribution[stability_key] = {
                    'count': row[1],
                    'avg_trust_score': row[2]
                }
            
            return {
                'total_assessments': stats[0] or 0,
                'avg_trust_score': stats[1],
                'min_trust_score': stats[2],
                'max_trust_score': stats[3],
                'stable_count': stats[4] or 0,
                'high_trust_count': stats[5] or 0,
                'high_trust_stable_count': stats[6] or 0,
                'stability_distribution': stability_distribution
            }
    
    def _row_to_event(self, row) -> EventRecord:
        """Convert database row to EventRecord with trust assessment fields"""
        timestamp = datetime.fromisoformat(row['timestamp'].replace('Z', '+00:00'))
        
        return EventRecord(
            event_id=row['event_id'],
            signal_type=row['signal_type'],
            source_node_id=row['source_node_id'],
            timestamp=timestamp,
            payload=json.loads(row['payload']) if row['payload'] else {},
            trust_score=row['trust_score'],
            novelty_score=row['novelty_score'],
            parent_signal_ids=json.loads(row['parent_signal_ids']) if row['parent_signal_ids'] else None,
            context_tags=json.loads(row['context_tags']) if row['context_tags'] else None,
            is_stable=row.get('is_stable'),
            target_id=row.get('target_id'),
            target_type=row.get('target_type'),
            assessment_rationale=row.get('assessment_rationale')
        )


# Migration script to update existing database
def migrate_events_table(db_path: str):
    """Add trust assessment columns to existing events table"""
    import sqlite3
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check if columns already exist
    cursor.execute("PRAGMA table_info(events)")
    columns = {row[1] for row in cursor.fetchall()}
    
    # Add missing columns
    if 'is_stable' not in columns:
        cursor.execute('ALTER TABLE events ADD COLUMN is_stable BOOLEAN')
    if 'target_id' not in columns:
        cursor.execute('ALTER TABLE events ADD COLUMN target_id TEXT')
    if 'target_type' not in columns:
        cursor.execute('ALTER TABLE events ADD COLUMN target_type TEXT')
    if 'assessment_rationale' not in columns:
        cursor.execute('ALTER TABLE events ADD COLUMN assessment_rationale TEXT')
    
    # Update existing TrustAssessmentSignal records
    cursor.execute('''
        UPDATE events 
        SET is_stable = json_extract(payload, '$.is_stable'),
            target_id = COALESCE(json_extract(payload, '$.target_id'), json_extract(payload, '$.idea_id')),
            target_type = json_extract(payload, '$.target_type')
        WHERE signal_type = 'TrustAssessmentSignal'
          AND is_stable IS NULL
    ''')
    
    conn.commit()
    conn.close()
    
    logger.info(f"Migration completed for {db_path}")