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