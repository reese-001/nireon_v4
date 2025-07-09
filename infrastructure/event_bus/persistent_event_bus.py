# infrastructure/event_bus/persistent_event_bus.py

import logging
import threading
from typing import Any, Callable, Dict, List, Optional
from datetime import datetime, timezone
from queue import Queue, Empty

from domain.ports.event_bus_port import EventBusPort
from infrastructure.persistence.event_repository import EventRepository, EventRecord
from .memory_event_bus import MemoryEventBus

logger = logging.getLogger(__name__)


class PersistentEventBus(MemoryEventBus):
    """
    Event bus that persists all events to the shared SQLite database
    while maintaining in-memory pub/sub functionality.
    
    Uses EventRepository to share the same database as IdeaRepository.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        config = config or {}
        
        # Initialize event repository with shared database
        repo_config = {
            'shared_db_path': config.get('shared_db_path', 'runtime/nireon_ideas.db'),
            'event_retention_days': config.get('retention_days', 30),
            'event_batch_size': config.get('batch_size', 100),
            **config  # Allow override of any setting
        }
        
        self.event_repository = EventRepository(repo_config)
        
        # Async write queue for better performance
        self.write_queue = Queue(maxsize=1000)
        self.batch_size = config.get('batch_size', 100)
        self.flush_interval = config.get('flush_interval_seconds', 5)
        
        # Start background writer thread
        self._stop_writer = threading.Event()
        self._writer_thread = threading.Thread(target=self._write_worker, daemon=True)
        self._writer_thread.start()
        
        # Stats
        self._total_events_published = 0
        self._total_events_persisted = 0
        
        logger.info(f"PersistentEventBus initialized with shared database")
    
    def publish(self, signal_type: str, data: Any) -> None:
        """Publish event to subscribers and queue for persistence"""
        # First, do the in-memory publish
        super().publish(signal_type, data)
        
        self._total_events_published += 1
        
        # Queue for persistence
        try:
            event_record = self._create_event_record(signal_type, data)
            self.write_queue.put_nowait(event_record)
        except:
            logger.error(f"Failed to queue event {signal_type} for persistence")
    
    def _create_event_record(self, signal_type: str, data: Any) -> EventRecord:
        """Convert signal data to EventRecord"""
        # Extract fields from signal object
        event_id = getattr(data, 'signal_id', None) or str(data.get('signal_id', f'evt_{datetime.now().timestamp()}'))
        source_node_id = getattr(data, 'source_node_id', None) or data.get('source_node_id')
        timestamp = getattr(data, 'timestamp', None) or datetime.now(timezone.utc)
        trust_score = getattr(data, 'trust_score', None)
        novelty_score = getattr(data, 'novelty_score', None)
        parent_signal_ids = getattr(data, 'parent_signal_ids', None) or []
        context_tags = getattr(data, 'context_tags', None) or {}
        
        # Get payload
        if hasattr(data, 'model_dump'):
            payload = data.model_dump(mode='json')
        elif hasattr(data, 'dict'):
            payload = data.dict()
        elif isinstance(data, dict):
            payload = data
        else:
            payload = {'raw_data': str(data)}
        
        return EventRecord(
            event_id=event_id,
            signal_type=signal_type,
            source_node_id=source_node_id,
            timestamp=timestamp,
            payload=payload,
            trust_score=trust_score,
            novelty_score=novelty_score,
            parent_signal_ids=parent_signal_ids,
            context_tags=context_tags
        )
    
    def _write_worker(self):
        """Background thread that batches and writes events to database"""
        batch = []
        last_flush = datetime.now()
        
        while not self._stop_writer.is_set():
            try:
                # Try to get an event with timeout
                timeout = max(0.1, self.flush_interval - (datetime.now() - last_flush).total_seconds())
                event = self.write_queue.get(timeout=timeout)
                batch.append(event)
                
                # Flush if batch is full
                if len(batch) >= self.batch_size:
                    self._flush_batch(batch)
                    batch = []
                    last_flush = datetime.now()
                    
            except Empty:
                # Timeout - check if we need to flush
                if batch and (datetime.now() - last_flush).total_seconds() >= self.flush_interval:
                    self._flush_batch(batch)
                    batch = []
                    last_flush = datetime.now()
            except Exception as e:
                logger.error(f"Error in event writer thread: {e}")
        
        # Final flush on shutdown
        if batch:
            self._flush_batch(batch)
    
    def _flush_batch(self, batch: List[EventRecord]):
        """Write a batch of events to database"""
        if not batch:
            return
        
        try:
            self.event_repository.save_events_batch(batch)
            self._total_events_persisted += len(batch)
            logger.debug(f"Persisted batch of {len(batch)} events")
        except Exception as e:
            logger.error(f"Failed to persist event batch: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get event bus statistics"""
        base_stats = {
            'total_published': self._total_events_published,
            'total_persisted': self._total_events_persisted,
            'queue_size': self.write_queue.qsize(),
            'subscribers': {k: len(v) for k, v in self._subscribers.items()}
        }
        
        # Add repository stats
        try:
            repo_stats = self.event_repository.get_stats()
            base_stats['repository'] = repo_stats
            
            # Get recent event stats
            event_stats = self.event_repository.get_event_stats(hours=1)
            base_stats['last_hour'] = event_stats
        except:
            pass
        
        return base_stats
    
    def close(self):
        """Shutdown the event bus cleanly"""
        logger.info("Shutting down PersistentEventBus...")
        
        # Stop writer thread
        self._stop_writer.set()
        self._writer_thread.join(timeout=10)
        
        # Final flush
        remaining = []
        try:
            while not self.write_queue.empty():
                remaining.append(self.write_queue.get_nowait())
        except:
            pass
            
        if remaining:
            self._flush_batch(remaining)
        
        # Close repository
        self.event_repository.close()
        
        logger.info("PersistentEventBus shutdown complete")
    
    def __del__(self):
        """Cleanup on deletion"""
        try:
            self.close()
        except:
            pass