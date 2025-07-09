# Fixed training script for MABWiser 2.7.4
# training/train_bandit.py

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
        
        if len(df) < 10:  # Lower threshold for testing
            print("Not enough data to train a model. Need at least 10 valid traces.")
            return

        # Context (State) -> Features for the model
        X = df[['parent_trust_score', 'parent_depth']].values
        
        # Action and Reward
        A = df['chosen_action'].values
        R = df['reward'].values
        
        print("Training MAB model with LinUCB...")
        
        # CORRECTED IMPORT for MABWiser 2.7.4
        from mabwiser.mab import MAB, LearningPolicy

        arms = list(df['chosen_action'].unique())
        print(f"Discovered arms from data: {arms}")

        # CORRECTED: Use MAB instead of ContextualMAB, and LinUCB for contextual bandits
        mab = MAB(
            arms=arms,
            learning_policy=LearningPolicy.LinUCB(alpha=1.25)
        )
        
        # For contextual bandits, fit with contexts
        mab.fit(decisions=A, rewards=R, contexts=X)
        
        print("Model training complete.")
        
        # Save the model
        output_path = Path(model_output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mab.save(str(output_path))
        print(f"✅ Model saved to: {output_path}")

    except Exception as e:
        print(f"An error occurred during training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        conn.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a bandit planner model for NIREON.")
    parser.add_argument("--db-path", default="runtime/training_traces.db", 
                       help="Path to the training traces SQLite database.")
    parser.add_argument("--model-out", default="runtime/models/bandit_planner_v1.pkl", 
                       help="Path to save the trained model.")
    args = parser.parse_args()

    train_model(args.db_path, args.model_out)


# =====================================================================================
# Updated BanditPlanner service.py to work with MABWiser 2.7.4
# components/planners/bandit_planner/service.py
# =====================================================================================

import logging
import random
import asyncio
from pathlib import Path
from typing import Any, Dict

from core.base_component import NireonBaseComponent
from core.lifecycle import ComponentMetadata
from core.results import ProcessResult
from domain.context import NireonExecutionContext
from domain.ports.event_bus_port import EventBusPort
from signals.core import SeedSignal, PlanNextStepSignal
from components.planners import BanditPlannerConfig
from components.planners.bandit_planner import BANDIT_PLANNER_METADATA

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
        # Start background task for model hot-reloading
        asyncio.create_task(self._model_watcher(context))

    async def _model_watcher(self, context: NireonExecutionContext):
        """Periodically checks if the model file has been updated and reloads it."""
        while True:
            await asyncio.sleep(300)  # Check every 5 minutes
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
                # CORRECTED IMPORT for MABWiser 2.7.4
                from mabwiser.mab import MAB
                self.mab = MAB.load(str(model_path))
                self._model_last_loaded_time = model_path.stat().st_mtime
                context.logger.info(f"[{self.component_id}] Loaded bandit model from {model_path}")
            except Exception as e:
                context.logger.error(f"[{self.component_id}] Failed to load model: {e}. Will use default action.")
                self.mab = None
        else:
            context.logger.warning(f"[{self.component_id}] Bandit model not found at {model_path}. Will use default action.")
            self.mab = None

    async def _process_impl(self, data: Dict[str, Any], context: NireonExecutionContext) -> ProcessResult:
        try:
            signal = PlanNextStepSignal(**data)
        except Exception as e:
            msg = f"Failed to create PlanNextStepSignal from input data: {e}. Data: {data}"
            logger.error(f"[{self.component_id}] {msg}")
            return ProcessResult(
                success=False, 
                component_id=self.component_id, 
                message=msg, 
                error_code="INVALID_INPUT_DATA"
            )

        chosen_action = self.cfg.default_action
        context_features = [signal.current_trust_score, float(signal.current_depth)]
        
        # Epsilon-greedy exploration
        current_epsilon = self.cfg.exploration_epsilon
        
        if self.mab and self.rng.random() > current_epsilon:
            try:
                # CORRECTED: Use predict with context for MABWiser 2.7.4
                prediction = self.mab.predict([context_features])
                chosen_action = prediction[0]
                context.logger.info(f"[{self.component_id}] Bandit chose action '{chosen_action}' based on context.")
            except Exception as e:
                context.logger.error(f"[{self.component_id}] Bandit prediction failed: {e}. Falling back to default.")
        else:
            chosen_action = self.rng.choice(self.cfg.available_actions)
            context.logger.info(f"[{self.component_id}] Epsilon-greedy exploration chose random action: '{chosen_action}'")
        
        # Map abstract action to concrete mechanism ID
        target_component_id = self.cfg.action_to_mechanism_map.get(
            chosen_action, 
            self.cfg.action_to_mechanism_map.get(self.cfg.default_action)
        )
        
        # Create the next signal with all necessary metadata for tracing
        next_signal = SeedSignal(
            source_node_id=self.component_id,
            seed_content=signal.current_idea_text,
            payload={
                "seed_idea_id": signal.current_idea_id,
                "text": signal.current_idea_text,
                "objective": signal.objective,
                "depth": signal.current_depth,
                "planner_action": chosen_action,
                "target_component_id": target_component_id,
                "session_id": signal.session_id,
                "parent_trust_score": signal.current_trust_score,
                "frame_id": context.metadata.get('current_frame_id'),
            }
        )
        
        if self.event_bus:
            self.event_bus.publish(next_signal.signal_type, next_signal)
            message = f"Planned next action '{chosen_action}' -> '{target_component_id}' and emitted SeedSignal."
            return ProcessResult(
                success=True, 
                component_id=self.component_id, 
                message=message, 
                output_data={"chosen_action": chosen_action}
            )
        else:
            return ProcessResult(
                success=False, 
                component_id=self.component_id, 
                message="EventBus not available."
            )


# =====================================================================================
# Quick test script to verify MABWiser works
# test_mabwiser.py
# =====================================================================================

def test_mabwiser():
    """Quick test to verify MABWiser 2.7.4 works correctly"""
    try:
        from mabwiser.mab import MAB, LearningPolicy
        print("✅ MABWiser imports successful!")
        
        # Create test data
        arms = ['EXPLORE', 'SYNTHESIZE']
        decisions = ['EXPLORE', 'EXPLORE', 'SYNTHESIZE', 'EXPLORE', 'SYNTHESIZE']
        rewards = [0.1, 0.3, 0.2, 0.4, -0.1]
        contexts = [[5.0, 0], [7.0, 1], [3.0, 2], [8.0, 1], [4.0, 3]]
        
        # Create and train model
        mab = MAB(arms=arms, learning_policy=LearningPolicy.LinUCB(alpha=1.25))
        mab.fit(decisions=decisions, rewards=rewards, contexts=contexts)
        print("✅ Model training successful!")
        
        # Test prediction
        test_contexts = [[6.0, 1], [2.0, 4]]
        predictions = mab.predict(test_contexts)
        print(f"✅ Predictions: {predictions}")
        
        # Test save/load
        from pathlib import Path
        test_path = "runtime/models/test_mab.pkl"
        Path(test_path).parent.mkdir(parents=True, exist_ok=True)
        mab.save(test_path)
        loaded_mab = MAB.load(test_path)
        test_predictions = loaded_mab.predict(test_contexts)
        print(f"✅ Save/load successful! Test predictions: {test_predictions}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_mabwiser()


# =====================================================================================
# Quick bootstrap script to create initial model
# bootstrap_model.py
# =====================================================================================

def create_initial_model():
    """Create an initial bandit model for cold start"""
    try:
        from mabwiser.mab import MAB, LearningPolicy
        from pathlib import Path
        
        # Minimal seed data for cold start
        arms = ['EXPLORE', 'SYNTHESIZE']
        decisions = ['EXPLORE', 'EXPLORE', 'SYNTHESIZE', 'EXPLORE']
        rewards = [0.1, 0.2, 0.15, 0.25]
        contexts = [[5.0, 0], [7.0, 1], [3.0, 2], [6.0, 1]]
        
        # Create model
        mab = MAB(arms=arms, learning_policy=LearningPolicy.LinUCB(alpha=1.25))
        mab.fit(decisions=decisions, rewards=rewards, contexts=contexts)
        
        # Save model
        model_path = Path("runtime/models/bandit_planner_v1.pkl")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        mab.save(str(model_path))
        
        print(f"✅ Initial bandit model created at {model_path}")
        print("🚀 Your bandit planner is now ready to learn!")
        
    except Exception as e:
        print(f"❌ Failed to create initial model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    create_initial_model()