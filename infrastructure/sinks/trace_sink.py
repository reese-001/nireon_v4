# nireon_v4/infrastructure/sinks/trace_sink.py
import logging
import sqlite3
import asyncio
import json  # NEW IMPORT
from pathlib import Path
from typing import Any, Dict
import math

from core.base_component import NireonBaseComponent
from core.lifecycle import ComponentMetadata
from core.results import ProcessResult
from core.tracing import BlockTrace
from domain.context import NireonExecutionContext
from domain.ports.event_bus_port import EventBusPort
from signals.core import TraceEmittedSignal
from .metadata import TRACE_SINK_METADATA


logger = logging.getLogger(__name__)

class TraceSink(NireonBaseComponent):
    METADATA_DEFINITION = TRACE_SINK_METADATA

    def __init__(self, config: Dict[str, Any], metadata_definition: ComponentMetadata):
        super().__init__(config, metadata_definition)
        db_path = self.config.get("db_path", "runtime/training_traces.db")
        self.db_path = Path(db_path)
        self.conn = None

    def _create_table(self):
        # MODIFIED: Added new columns for frame_id and interpreter_set
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS block_traces (
            trace_id TEXT PRIMARY KEY, session_id TEXT, parent_idea_id TEXT, 
            parent_trust_score REAL, parent_depth INTEGER, planner_policy_id TEXT,
            chosen_action TEXT, chosen_mechanism_id TEXT, generated_idea_id TEXT, 
            generated_trust_score REAL, reward REAL, duration_ms REAL, 
            llm_calls INTEGER, timestamp TEXT, frame_id TEXT, interpreter_set TEXT
        );
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute(create_table_sql)
            self.conn.commit()
        except sqlite3.Error as e:
            logger.error(f"[{self.component_id}] Database error on table creation: {e}")
            raise

    async def _initialize_impl(self, context: NireonExecutionContext) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._create_table()
        
        event_bus = context.component_registry.get_service_instance(EventBusPort)
        event_bus.subscribe(TraceEmittedSignal.__name__, self._handle_trace_event)
        context.logger.info(f"[{self.component_id}] Initialized and subscribed to TraceEmittedSignal. DB at {self.db_path}")

    async def _handle_trace_event(self, signal: TraceEmittedSignal):
        if not isinstance(signal, TraceEmittedSignal):
            logger.warning(f"[{self.component_id}] Received non-TraceEmittedSignal: {type(signal)}")
            return
            
        trace = signal.trace
        
        # MODIFIED: Implement the new composite reward function.
        parent_trust = trace.parent_trust_score if trace.parent_trust_score is not None else 5.0
        duration_sec = max(trace.duration_ms / 1000.0, 0.001)
        
        delta_trust = trace.generated_trust_score - parent_trust
        diversity_bonus = trace.trace_metadata.get("frame_variance", 0.0)
        
        # This shaping function can now be tuned over time.
        reward_raw = (delta_trust + (0.3 * diversity_bonus)) / duration_sec
        
        # Clip reward to prevent skew from large outliers.
        reward = max(-1.0, min(1.0, reward_raw))
        
        insert_sql = """
        INSERT INTO block_traces (
            trace_id, session_id, parent_idea_id, parent_trust_score, parent_depth, 
            planner_policy_id, chosen_action, chosen_mechanism_id, generated_idea_id, 
            generated_trust_score, reward, duration_ms, llm_calls, timestamp,
            frame_id, interpreter_set
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        params = (
            str(trace.trace_id), trace.session_id, trace.parent_idea_id, 
            trace.parent_trust_score, trace.parent_depth, trace.planner_policy_id,
            trace.chosen_action, trace.chosen_mechanism_id, trace.generated_idea_id, 
            trace.generated_trust_score, reward, trace.duration_ms, trace.llm_calls, 
            trace.timestamp.isoformat(), trace.frame_id, 
            json.dumps(trace.interpreter_set) if trace.interpreter_set else None
        )

        # FIX: Run blocking DB I/O in a separate thread to not block the event loop.
        try:
            await asyncio.to_thread(self._execute_db_write, insert_sql, params)
            logger.debug(f"[{self.component_id}] Persisted trace {trace.trace_id} with reward {reward:.4f}")
        except Exception as e:
            logger.error(f"[{self.component_id}] Failed to insert trace {trace.trace_id}: {e}")

    def _execute_db_write(self, sql: str, params: tuple):
        try:
            with self.conn: # Use a context manager for commits/rollbacks
                self.conn.execute(sql, params)
        except sqlite3.Error as e:
            logger.error(f"[{self.component_id}] DB write error: {e}")
            raise

    async def _process_impl(self, data: Any, context: NireonExecutionContext) -> ProcessResult:
        return ProcessResult(success=True, message="TraceSink is a reactive subscriber.")

    async def _shutdown_impl(self, context: NireonExecutionContext) -> None:
        if self.conn:
            self.conn.close()
        context.logger.info(f"[{self.component_id}] Shutdown complete, DB connection closed.")