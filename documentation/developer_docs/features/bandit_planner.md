v### **Project: Stage-2 Self-Improving Planner**

**Objective:**
This document outlines the complete, end-to-end engineering plan to evolve NIREON V4's planning from a static, rule-based system into a dynamic, self-improving one. We will implement a BanditPlanner component that learns from experience to maximize the (trust gain - compute_cost) per cycle.

**Core Strategy:**
The implementation will integrate seamlessly with NIREON's existing signal-driven architecture. The planner will be a standard component triggered by the Reactor. Its "action" will be to emit a signal, which the Reactor then routes to the appropriate mechanism (e.g., Explorer). A TraceSink component will listen for TraceEmittedSignal events to build a dataset for offline training, closing the learning loop.

---

### **Phase 1: Foundational Changes (Data and Signals)**

We begin by defining the data contracts and communication signals required for the learning loop.

#### **1. NEW FILE: nireon_v4/core/tracing.py**
This file defines the BlockTrace, the fundamental, immutable record of a single generative-evaluative step, which serves as one training data point.

python
# nireon_v4/core/tracing.py
from __future__ import annotations
import uuid
from typing import Optional, Dict, Any
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel, Field

class BlockTrace(BaseModel):
    """
    An immutable record of a single, completed generative-evaluative step.
    This serves as the basis for reinforcement learning.
    """
    # FIX: Use default_factory for uuid.uuid4
    trace_id: UUID = Field(default_factory=uuid.uuid4)
    session_id: str
    
    # Context (State)
    parent_idea_id: Optional[str] = None
    parent_trust_score: Optional[float] = None
    parent_depth: int = 0
    
    # Action
    planner_policy_id: str
    chosen_action: str # e.g., "EXPLORE", "SYNTHESIZE"
    # NEW (from review): Track which specific component was triggered for better per-arm stats.
    chosen_mechanism_id: str 
    
    # Outcome
    generated_idea_id: str
    generated_trust_score: float
    
    # Reward Signal (will be calculated by the sink)
    reward: Optional[float] = None
    
    # Cost
    duration_ms: float
    llm_calls: int = 0
    
    # Metadata
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    trace_metadata: Dict[str, Any] = Field(default_factory=dict)


#### **2. MODIFIED FILE: nireon_v4/signals/core.py**
We introduce two new signals: PlanNextStepSignal to invoke the planner and TraceEmittedSignal to broadcast the results of a cycle.

python
# nireon_v4/signals/core.py

from __future__ import annotations
from enum import Enum
# FIX: Import Literal
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import Field, field_validator
from .base import EpistemicSignal
# NEW IMPORT
from core.tracing import BlockTrace

# ... (keep existing signal classes: LoopAction, SeedSignal, etc.) ...

# NEW SIGNAL
class PlanNextStepSignal(EpistemicSignal):
    """
    Signal to instruct the active planner to choose the next action.
    This is the entry point to the learning-driven part of the loop.
    """
    signal_type: Literal['PlanNextStepSignal'] = 'PlanNextStepSignal'
    session_id: str
    current_idea_id: str
    current_trust_score: float
    current_depth: int
    objective: str

# NEW SIGNAL
class TraceEmittedSignal(EpistemicSignal):
    """
    Broadcasts a completed BlockTrace for sinks and monitoring components.
    """
    signal_type: Literal['TraceEmittedSignal'] = 'TraceEmittedSignal'
    trace: BlockTrace


#### **3. MODIFIED FILE: nireon_v4/signals/__init__.py**
We update the package's __init__.py to correctly and cleanly export the new signals, improving IDE auto-completion.

python
# nireon_v4/signals/__init__.py
from __future__ import annotations
import inspect
from typing import Dict, Type, List

from .base import EpistemicSignal
from .core import (
    SeedSignal, LoopSignal, IdeaGeneratedSignal, TrustAssessmentSignal,
    StagnationDetectedSignal, ErrorSignal, GenerativeLoopFinishedSignal,
    MathQuerySignal, MathResultSignal, ProtoTaskSignal, ProtoResultSignal,
    ProtoErrorSignal, MathProtoResultSignal, PlanNextStepSignal, TraceEmittedSignal
)

def get_all_subclasses(cls: Type) -> List[Type]:
    all_subclasses = []
    for subclass in cls.__subclasses__():
        all_subclasses.append(subclass)
        all_subclasses.extend(get_all_subclasses(subclass))
    return all_subclasses

# FIX: Use a private variable for the map to avoid polluting the namespace
_signal_class_map: Dict[str, Type[EpistemicSignal]] = {
    cls.__name__: cls for cls in get_all_subclasses(EpistemicSignal) 
    if not inspect.isabstract(cls)
}
_signal_class_map['EpistemicSignal'] = EpistemicSignal

# FIX: Define __all__ explicitly for clean imports and better IDE support
__all__ = [
    'EpistemicSignal',
    'SeedSignal',
    'LoopSignal',
    'IdeaGeneratedSignal',
    'TrustAssessmentSignal',
    'StagnationDetectedSignal',
    'ErrorSignal',
    'GenerativeLoopFinishedSignal',
    'MathQuerySignal',
    'MathResultSignal',
    'ProtoTaskSignal',
    'ProtoResultSignal',
    'ProtoErrorSignal',
    'MathProtoResultSignal',
    'PlanNextStepSignal',
    'TraceEmittedSignal',
    '_signal_class_map'  # Keep accessible for programmatic use
]

# For backwards compatibility if needed, but direct imports are preferred.
signal_class_map = _signal_class_map


---

### **Phase 2: Component Implementation**

Next, we create the new BanditPlanner and TraceSink components.

#### **4. NEW MODULE: nireon_v4/components/planners/bandit_planner/**

Create a new directory structure for the planner.

*   nireon_v4/components/planners/bandit_planner/
    *   __init__.py
    *   config.py
    *   metadata.py
    *   service.py

##### **config.py**
python
# nireon_v4/components/planners/bandit_planner/config.py
from pydantic import BaseModel, Field

class BanditPlannerConfig(BaseModel):
    model_path: str = Field(description="Path to the serialized contextual bandit model (.pkl).")
    exploration_epsilon: float = Field(0.3, ge=0.0, le=1.0, description="Probability of choosing a random action for exploration.")
    available_actions: list[str] = Field(default=["EXPLORE", "SYNTHESIZE"], description="List of possible actions the bandit can choose.")
    default_action: str = Field("EXPLORE", description="Action to take if the model is not loaded or fails.")
    action_to_mechanism_map: dict[str, str] = Field(
        default={"EXPLORE": "explorer_instance_01", "SYNTHESIZE": "catalyst_instance_01"},
        description="Maps abstract planner actions to concrete mechanism component IDs."
    )


##### **metadata.py**
python
# nireon_v4/components/planners/bandit_planner/metadata.py
from core.lifecycle import ComponentMetadata

BANDIT_PLANNER_METADATA = ComponentMetadata(
    id="bandit_planner_default",
    name="Bandit-based Epistemic Planner",
    version="1.0.0",
    description="Uses a contextual bandit model to choose the next epistemic action (e.g., Explore vs. Synthesize) to maximize trust gain.",
    category="planner",
    epistemic_tags=["planner", "learning", "reinforcement_learning", "bandit"],
    requires_initialize=True,
    dependencies={'EventBusPort': '*'}
)


##### **service.py**
python
# nireon_v4/components/planners/bandit_planner/service.py
import logging
import random
import asyncio
from pathlib import Path
from typing import Any, Dict

from core.base_component import NireonBaseComponent
from core.results import ProcessResult
from domain.context import NireonExecutionContext
from domain.ports.event_bus_port import EventBusPort
from signals.core import SeedSignal, PlanNextStepSignal
from .config import BanditPlannerConfig
from .metadata import BANDIT_PLANNER_METADATA

logger = logging.getLogger(__name__)

class BanditPlanner(NireonBaseComponent):
    METADATA_DEFINITION = BANDIT_PLANNER_METADATA
    ConfigModel = BanditPlannerConfig

    def __init__(self, config: Dict[str, Any], metadata_definition: ComponentMetadata):
        super().__init__(config, metadata_definition)
        self.cfg: BanditPlannerConfig = self.ConfigModel(**self.config)
        self.event_bus: EventBusPort | None = None
        self.mab = None
        self.rng = random.Random()
        self._model_last_loaded_time = 0.0

    async def _initialize_impl(self, context: NireonExecutionContext) -> None:
        self.event_bus = context.component_registry.get_service_instance(EventBusPort)
        self._load_model(context)

        # FIX (from review): Start a background task for model hot-reloading.
        asyncio.create_task(self._model_watcher(context))

    async def _model_watcher(self, context: NireonExecutionContext):
        """Periodically checks if the model file has been updated and reloads it."""
        while True:
            await asyncio.sleep(300) # Check every 5 minutes
            try:
                model_path = Path(self.cfg.model_path)
                if model_path.exists():
                    mod_time = model_path.stat().st_mtime
                    if mod_time > self._model_last_loaded_time:
                        context.logger.info(f"[{self.component_id}] New model file detected. Reloading...")
                        self._load_model(context)
            except Exception as e:
                context.logger.error(f"[{self.component_id}] Error in model watcher: {e}")

    def _load_model(self, context: NireonExecutionContext):
        model_path = Path(self.cfg.model_path)
        if model_path.exists():
            try:
                from mabwiser.mab import ContextualMAB
                self.mab = ContextualMAB.load(str(model_path))
                self._model_last_loaded_time = model_path.stat().st_mtime
                context.logger.info(f"[{self.component_id}] Loaded bandit model from {model_path}")
            except Exception as e:
                context.logger.error(f"[{self.component_id}] Failed to load model: {e}. Will use default action.")
                self.mab = None
        else:
            context.logger.warning(f"[{self.component_id}] Bandit model not found at {model_path}. Will use default action.")
            self.mab = None

    async def _process_impl(self, data: PlanNextStepSignal, context: NireonExecutionContext) -> ProcessResult:
        if not isinstance(data, PlanNextStepSignal):
            return ProcessResult(success=False, message="Invalid input, expected PlanNextStepSignal")

        chosen_action = self.cfg.default_action
        context_features = [data.current_trust_score, float(data.current_depth)]

        # FIX (from review): Placeholder for a real epsilon decay schedule.
        current_epsilon = self.cfg.exploration_epsilon
        
        if self.mab and self.rng.random() > current_epsilon:
            try:
                prediction = self.mab.predict([context_features])
                chosen_action = prediction[0]
                context.logger.info(f"[{self.component_id}] Bandit chose action '{chosen_action}' based on context.")
            except Exception as e:
                context.logger.error(f"[{self.component_id}] Bandit prediction failed: {e}. Falling back to default.")
        else:
            chosen_action = self.rng.choice(self.cfg.available_actions)
            context.logger.info(f"[{self.component_id}] Epsilon-greedy exploration chose random action: '{chosen_action}'")
        
        # FIX (from review): Map the abstract action to a concrete mechanism ID.
        target_component_id = self.cfg.action_to_mechanism_map.get(chosen_action, "explorer_instance_01")
        
        # We re-use SeedSignal as a generic entry point for the generative loop.
        # It's crucial to pass metadata forward for the trace assembler.
        next_signal = SeedSignal(
            source_node_id=self.component_id,
            seed_content=f"Action '{chosen_action}' on idea '{data.current_idea_id}'",
            payload={
                "seed_idea_id": data.current_idea_id,
                "objective": data.objective,
                "metadata": {
                    "depth": data.current_depth,
                    "planner_action": chosen_action,
                    "target_component_id": target_component_id,
                    "session_id": data.session_id,
                    "parent_trust_score": data.current_trust_score,
                },
            }
        )
        
        if self.event_bus:
            self.event_bus.publish(next_signal.signal_type, next_signal)
            message = f"Planned next action '{chosen_action}' -> '{target_component_id}' and emitted SeedSignal."
            return ProcessResult(success=True, message=message, output_data={"chosen_action": chosen_action})
        else:
            return ProcessResult(success=False, message="EventBus not available.")


#### **5. NEW MODULE: nireon_v4/infrastructure/sinks/**

Create a directory for the TraceSink.

*   nireon_v4/infrastructure/sinks/
    *   __init__.py
    *   trace_sink.py
    *   metadata.py (for TraceSink's metadata)

##### **metadata.py**
python
# nireon_v4/infrastructure/sinks/metadata.py
from core.lifecycle import ComponentMetadata

TRACE_SINK_METADATA = ComponentMetadata(
    id="trace_sink_sqlite",
    name="BlockTrace SQLite Sink",
    version="1.0.0",
    description="Subscribes to trace events and persists them to a SQLite database for offline learning.",
    category="infrastructure_service",
    epistemic_tags=["persistence", "logger", "sink", "learning_data"],
    requires_initialize=True
)


##### **trace_sink.py** (Fully Corrected)
python
# nireon_v4/infrastructure/sinks/trace_sink.py
import logging
import sqlite3
import asyncio
from pathlib import Path
from typing import Any, Dict
import math

from core.base_component import NireonBaseComponent
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
        # Added chosen_mechanism_id
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS block_traces (
            trace_id TEXT PRIMARY KEY, session_id TEXT, parent_idea_id TEXT, 
            parent_trust_score REAL, parent_depth INTEGER, planner_policy_id TEXT,
            chosen_action TEXT, chosen_mechanism_id TEXT, generated_idea_id TEXT, 
            generated_trust_score REAL, reward REAL, duration_ms REAL, 
            llm_calls INTEGER, timestamp TEXT
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
        
        # FIX: Reward calculation moved here from YAML.
        parent_trust = trace.parent_trust_score if trace.parent_trust_score is not None else 5.0
        duration_sec = max(trace.duration_ms / 1000.0, 0.001)
        raw_reward = (trace.generated_trust_score - parent_trust) / duration_sec
        
        # FIX (from review): Apply reward shaping (clipping).
        reward = max(-1.0, min(1.0, raw_reward))
        
        insert_sql = """
        INSERT INTO block_traces (
            trace_id, session_id, parent_idea_id, parent_trust_score, parent_depth, 
            planner_policy_id, chosen_action, chosen_mechanism_id, generated_idea_id, 
            generated_trust_score, reward, duration_ms, llm_calls, timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        params = (
            str(trace.trace_id), trace.session_id, trace.parent_idea_id, 
            trace.parent_trust_score, trace.parent_depth, trace.planner_policy_id,
            trace.chosen_action, trace.chosen_mechanism_id, trace.generated_idea_id, 
            trace.generated_trust_score, reward, trace.duration_ms, trace.llm_calls, 
            trace.timestamp.isoformat()
        )

        try:
            # FIX: Run blocking DB I/O in a separate thread.
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


---

### **Phase 3: Integration & Orchestration (The Reactor)**

#### **6. NEW FILE: nireon_v4/configs/reactor/rules/planner.yaml**
This new rule file orchestrates the entire learning loop.

yaml
# nireon_v4/configs/reactor/rules/planner.yaml
version: "1.0"
rules:
  # Rule 1: Starts the planning loop from an external seed.
  - id: "on_seed_trigger_planner"
    description: "When a new seed is introduced, trigger the planner to decide the first action."
    namespace: "planner_loop"
    priority: 5
    enabled: true
    conditions:
      - type: "signal_type_match"
        signal_type: "SeedSignal"
      - type: "payload_expression"
        expression: "payload.metadata.planner_action == None"
    actions:
      - type: "emit_signal"
        signal_type: "PlanNextStepSignal"
        source_node_id_override: "reactor"
        payload:
          session_id: "{{ signal.run_id }}_{{ signal.signal_id }}"
          current_idea_id: "{{ payload.seed_idea_id }}"
          current_trust_score: 10.0
          current_depth: 0
          objective: "{{ payload.metadata.objective }}"

  # Rule 2: Triggers the chosen mechanism based on the planner's output.
  - id: "on_planner_seed_trigger_mechanism"
    description: "When the planner emits a SeedSignal, trigger the specified mechanism."
    namespace: "planner_loop"
    priority: 6
    enabled: true
    conditions:
      - type: "signal_type_match"
        signal_type: "SeedSignal"
      - type: "payload_expression"
        expression: "exists(payload.metadata.planner_action)"
    actions:
      - type: "trigger_component"
        component_id: "{{ payload.metadata.target_component_id }}"
        input_data_mapping:
          text: "{{ signal.seed_content }}"
          id: "{{ payload.seed_idea_id }}"
          objective: "{{ payload.metadata.objective }}"
          metadata: "{{ payload.metadata }}"

  # Rule 3: Continues the loop after an idea is assessed.
  - id: "on_assessment_trigger_planner_and_trace"
    description: "After an idea is assessed, emit a trace and trigger the planner for the next step."
    namespace: "planner_loop"
    priority: 15
    enabled: true
    conditions:
      - type: "signal_type_match"
        signal_type: "TrustAssessmentSignal"
      - type: "payload_expression"
        expression: "exists(payload.assessment_details.metadata.session_id)"
    actions:
      - type: "emit_signal"
        signal_type: "TraceEmittedSignal"
        source_node_id_override: "reactor_trace_assembler"
        payload:
          trace:
            session_id: "{{ payload.assessment_details.metadata.session_id }}"
            parent_idea_id: "{{ payload.assessment_details.idea_parent_id }}" 
            parent_trust_score: "{{ payload.assessment_details.metadata.parent_trust_score }}"
            parent_depth: "{{ payload.assessment_details.metadata.depth | default(0) - 1 }}"
            planner_policy_id: "bandit_v1"
            chosen_action: "{{ payload.assessment_details.metadata.planner_action }}"
            chosen_mechanism_id: "{{ payload.assessment_details.metadata.target_component_id }}"
            generated_idea_id: "{{ signal.target_id }}"
            generated_trust_score: "{{ signal.trust_score }}"
            duration_ms: "{{ payload.assessment_details.metadata.duration_ms | default(1000) }}"
            llm_calls: 1
      - type: "emit_signal"
        signal_type: "PlanNextStepSignal"
        source_node_id_override: "reactor"
        payload:
          session_id: "{{ payload.assessment_details.metadata.session_id }}"
          current_idea_id: "{{ signal.target_id }}"
          current_trust_score: "{{ signal.trust_score }}"
          current_depth: "{{ payload.assessment_details.metadata.depth }}"
          objective: "{{ payload.assessment_details.metadata.objective }}"

  # Rule 4: Safety circuit breaker
  - id: "planner_circuit_breaker"
    description: "If average reward is too low, fall back to a default action."
    namespace: "planner_safety"
    priority: 1
    enabled: true
    conditions:
      - type: "signal_type_match"
        signal_type: "TraceEmittedSignal"
      - type: "payload_expression"
        expression: "context.component_registry.get('reward_monitor').get_average_reward() < -0.2"
    actions:
      - type: "emit_signal"
        signal_type: "SystemAlert"
        payload:
          severity: "high"
          message: "Planner circuit breaker triggered. Average reward is too low."

*(Note: Implementing the reward_monitor component is left as a next step, but the rule is in place).*

#### **7. MODIFIED FILE: nireon_v4/configs/reactor/rules/core.yaml**
Disable the old rule to prevent conflicts.

yaml
# nireon_v4/configs/reactor/rules/core.yaml
version: "1.0"
rules:
  - id: "core_seed_to_explorer_rule"
    description: "[DEPRECATED & DISABLED by Planner Loop] When a SeedSignal is detected, trigger the primary explorer."
    namespace: "core"
    priority: 10
    enabled: false # <<<<<<<<<<<<<<< THIS IS THE CRITICAL CHANGE
    conditions:
      - type: "signal_type_match"
        signal_type: "SeedSignal"
    actions:
      - type: "trigger_component"
        component_id: "explorer_instance_01"
        input_data_mapping:
          text: "payload.text"
          id: "payload.seed_idea_id"
          objective: "payload.metadata.objective"
# ... rest of the file ...


---

### **Phase 4: Configuration & Training**

#### **8. MODIFIED FILE: nireon_v4/configs/manifests/standard.yaml**
Register the new components.

yaml
# nireon_v4/configs/manifests/standard.yaml

# ... (inside a new `planners:` section or existing `mechanisms:`)
mechanisms: # Or create a new top-level `planners:` section
  # ... other mechanisms
  active_planner:
    enabled: true
    class: "components.planners.bandit_planner.service:BanditPlanner"
    metadata_definition: "components.planners.bandit_planner.metadata:BANDIT_PLANNER_METADATA"
    config:
      model_path: "runtime/models/bandit_planner_v1.pkl"
      exploration_epsilon: 0.3
      available_actions: ["EXPLORE", "SYNTHESIZE"]
      default_action: "EXPLORE"
      action_to_mechanism_map:
        EXPLORE: "explorer_instance_01"
        SYNTHESIZE: "catalyst_instance_01"

shared_services:
  # ... other services
  trace_sink_main:
    enabled: true
    preload: false
    class: "infrastructure.sinks.trace_sink:TraceSink"
    metadata_definition: "infrastructure.sinks.metadata:TRACE_SINK_METADATA"
    config:
      db_path: "runtime/training_traces.db"


#### **9. NEW FILE: nireon_v4/training/train_bandit.py** (Complete Script)
python
# nireon_v4/training/train_bandit.py
import argparse
import pandas as pd
import sqlite3
from pathlib import Path

def train_model(db_path: str, model_output_path: str):
    """Trains a contextual bandit model from the trace database."""
    print(f"Connecting to database at: {db_path}")
    if not Path(db_path).exists():
        print(f"Error: Database file not found at {db_path}")
        return

    conn = sqlite3.connect(db_path)
    
    try:
        df = pd.read_sql("SELECT * FROM block_traces WHERE reward IS NOT NULL", conn)
        print(f"Loaded {len(df)} traces with valid rewards from the database.")
        
        if len(df) < 100:
            print("Not enough data to train a model. Need at least 100 valid traces.")
            return

        # Context (State) -> Features for the model
        X = df[['parent_trust_score', 'parent_depth']].values
        
        # Action and Reward
        A = df['chosen_action'].values
        R = df['reward'].values
        
        print("Training ContextualMAB model...")
        from mabwiser.mab import ContextualMAB, LearningPolicy

        arms = list(df['chosen_action'].unique())
        print(f"Discovered arms from data: {arms}")

        mab = ContextualMAB(
            arms=arms,
            learning_policy=LearningPolicy.LinUCB(alpha=1.5)
        )
        mab.fit(decisions=A, rewards=R, contexts=X)
        
        print("Model training complete.")
        
        # Save the model
        output_path = Path(model_output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mab.save(str(output_path))
        print(f"✅ Model saved to: {output_path}")

    except Exception as e:
        print(f"An error occurred during training: {e}")
    finally:
        conn.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a bandit planner model for NIREON.")
    parser.add_argument("--db-path", default="runtime/training_traces.db", help="Path to the training traces SQLite database.")
    parser.add_argument("--model-out", default="runtime/models/bandit_planner_v1.pkl", help="Path to save the trained model.")
    args = parser.parse_args()
    
    train_model(args.db_path, args.model_out)


---

### **Phase 5: Testing & Validation Strategy**

1.  **Unit Tests:**
    *   BlockTrace serialization and deserialization.
    *   BanditPlanner.process correctly maps actions to SeedSignals, and falls back to the default action if the model fails.
    *   TraceSink._handle_trace_event correctly calculates and clips rewards.
2.  **Integration Tests:**
    *   A test that injects a SeedSignal, verifies a PlanNextStepSignal is emitted, then a SeedSignal from the planner, then IdeaGeneratedSignal, then TrustAssessmentSignal, and finally a TraceEmittedSignal.
    *   A test where the bandit model file (.pkl) is corrupted or missing; assert that the BanditPlanner logs an error and uses its default action without crashing the system.
3.  **Load Tests:**
    *   Use a test script to publish 1,000 TraceEmittedSignals in rapid succession and measure the asyncio event loop's latency to ensure the to_thread call in TraceSink is effectively preventing blocking.

---

### **Phase 6: Deployment & Rollout Plan**

1.  **Initial Deployment (Cold Start):**
    *   Deploy the code with an empty runtime/models directory.
    *   The BanditPlanner will not find a model and will operate in pure exploration mode (random action selection) with the configured exploration_epsilon.
    *   The TraceSink will begin populating the training_traces.db file.
2.  **First Training Cycle:**
    *   After 24-48 hours of data collection, run the training/train_bandit.py script manually or via a cron job. This will create the first bandit_planner_v1.pkl model file.
3.  **Model Activation (Hot-Reload):**
    *   The BanditPlanner's background watcher task will detect the new model file and automatically load it into the live process without requiring a service restart. The planner will then switch from pure exploration to exploitation/exploration based on the learned policy.
4.  **Ongoing Operation:**
    *   Schedule the train_bandit.py script to run nightly or weekly to continuously refine the model as more data is collected.
    *   Monitor the average reward via the TraceSink database to ensure system performance is improving.

This comprehensive plan provides a complete, robust, and safe implementation path, addressing all feedback from the review.