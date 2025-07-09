Bandit Planner Subsystem
Description: A machine learning-driven planner that optimizes the epistemic workflow of the system. It uses a contextual multi-armed bandit model (mabwiser) to decide the most promising next action (e.g., 'EXPLORE' or 'SYNTHESIZE') based on the current context, such as an idea's trust score and depth. The model is trained on historical BlockTrace data, allowing the system to learn which actions lead to the greatest increase in trust over time. This creates a reinforcement learning loop that continually improves the system's strategic decision-making.
Public API / Contracts
components.planners.bandit_planner.service.BanditPlanner: The core component that receives planning requests and uses the trained model to predict the next best action.
03_bandit_runner/train.py: A standalone script used to train the bandit model from the training_traces.db SQLite database, producing a bandit_planner_v1.pkl model file.
Accepted Signals: PlanNextStepSignal, which contains the context features (trust, depth) for the bandit model.
Produced Signals: SeedSignal, which is targeted at the mechanism chosen by the bandit's action (e.g., ExplorerMechanism for an 'EXPLORE' action).
Dependencies (Imports From)
Event_and_Signal_System
Domain_Model
Kernel
Directory Layout (Conceptual)
Generated mermaid
graph TD
    subgraph BanditPlanner [components/planners/bandit_planner]
        A(service.py)
        B(config.py)
        C(metadata.py)
    end
    subgraph Training [03_bandit_runner]
        D(train.py) -- reads --> E[SQLite DB];
        D -- writes --> F[bandit_planner_v1.pkl];
    end
    A -- reads --> F;