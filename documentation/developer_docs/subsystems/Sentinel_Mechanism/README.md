# Sentinel Mechanism Subsystem

**Description:** The Sentinel is a critical evaluative agent responsible for quality control and gatekeeping within the NIREON idea ecosystem. Its primary function is to perform a multi-faceted assessment of a given `Idea` to determine its "trustworthiness" and "stability." It does this by scoring the idea against multiple axes—primarily **alignment** with objectives, **feasibility**, and **novelty**. These scores are then combined into a single, weighted `trust_score`.

If an idea's trust score and individual axis scores meet configurable thresholds, it is deemed "stable" and allowed to progress. If not, it is rejected with a detailed explanation. The Sentinel operates within a structured `Frame`, leverages LLMs for nuanced qualitative assessment, and incorporates adaptive scoring adjustments based on an idea's history and context.

---

## Core Concepts & Functionality

The Sentinel's design incorporates several key architectural patterns to ensure it performs robust, consistent, and context-aware evaluations.

-   **Frame-Based Assessment**: Similar to the Explorer, every assessment task is encapsulated within its own dedicated `Frame`, created via the `FrameFactoryService`. This provides a clean, auditable context for each evaluation, complete with its own resource budget (e.g., max 2 LLM calls per assessment) and a specific LLM policy tailored for critical evaluation (e.g., low temperature for deterministic scoring).

-   **Multi-Axis Evaluation (`assessment_core.py`)**: The Sentinel does not rely on a single metric. Its `AssessmentCore` evaluates an idea along three primary axes:
    1.  **Alignment**: How well the idea aligns with a given `objective`.
    2.  **Feasibility**: The likelihood that the idea could be practically implemented.
    3.  **Novelty**: How semantically different the idea is from its parent and sibling ideas.
    The alignment and feasibility scores are determined by an LLM, while the novelty score is calculated quantitatively using vector embeddings.

-   **Stage-Aware Evaluation**: The Sentinel is aware of the system's `EpistemicStage` (e.g., `EXPLORATION`, `CRITIQUE`). It uses the `StageEvaluationService` to resolve different assessment parameters (like axis weights) depending on the current stage. For example, during `EXPLORATION`, it might place a higher weight on novelty, whereas during a later stage, it might prioritize feasibility.

-   **Weighted Scoring & Stability Threshold**: The scores from each axis are combined into a single `trust_score` using a configurable set of `weights` (e.g., `[0.4, 0.3, 0.3]` for alignment, feasibility, and novelty). An idea is only considered "stable" if this final `trust_score` is above the `trust_threshold` **and** each individual axis score is above the `min_axis_score`. This prevents an idea with a fatal flaw in one area (e.g., very low feasibility) from passing just because it scored highly in others.

-   **Quantitative Novelty Calculation (`novelty_calculator.py`)**: The novelty score is not subjective. The `NoveltyCalculator` uses the `EmbeddingPort` to get vector representations of the target idea and its references (parents/siblings). It then calculates the maximum cosine similarity between the target and its references. The final novelty score is a function of this similarity (`novelty = (1 - max_similarity) * 9 + 1`), providing a quantitative, reproducible measure of an idea's originality.

-   **Adaptive Scoring Adjustments (`scoring_adjustment.py`)**: The base trust score can be dynamically adjusted based on contextual factors to provide a more nuanced evaluation:
    -   **Length Penalty**: Excessively long or verbose ideas are penalized to encourage concise and clear communication.
    -   **Progression Bonus**: Ideas that contain keywords indicating iterative progress or detailed planning (e.g., "implementation timeline," "metrics for success") can receive a trust bonus, rewarding development and refinement.
    -   **Edge Trust**: If an idea's parent has a high trust score, the child idea receives a small "support boost," acknowledging its foundation in a previously validated concept.

-   **Robust LLM Parsing with Circuit Breaker**: The Sentinel relies on an LLM to provide alignment and feasibility scores in a structured JSON format. The `llm_response_parser.py` is designed to robustly parse this output. If parsing fails repeatedly, a circuit breaker pattern is activated, causing the Sentinel to temporarily use default scores instead of making further failing LLM calls, preventing system stalls.

---

## Detailed Feature Breakdown

-   **LLM-Powered Axis Scoring**: Uses a carefully engineered prompt (built by `StageEvaluationService`) to ask an LLM to score an idea's alignment and feasibility on a 1-10 scale and provide a brief explanation.
-   **Quantitative Novelty Scoring**: Employs vector embeddings to calculate a novelty score based on cosine similarity, making this axis objective and data-driven.
-   **Configurable Scoring Weights**: The relative importance of alignment, feasibility, and novelty can be tuned via the `weights` parameter in the `SentinelMechanismConfig`.
-   **Dual Thresholds**: Enforces both an overall `trust_threshold` and a `min_axis_score` for each dimension, ensuring well-rounded, high-quality ideas.
-   **Contextual Adjustments**: The final score is not static; it's adjusted based on idea length, evidence of progressive development, and trust inherited from parent ideas.
-   **Stage-Specific Evaluation**: Integrates with `StageEvaluationService` to adapt its evaluation criteria to the current epistemic goals of the system.
-   **Comprehensive Assessment Object**: The final output is a rich `IdeaAssessment` object containing the final score, stability status, a detailed rejection reason (if any), all individual axis scores, and extensive metadata about how the assessment was performed.

---

## Public API / Contracts

-   **`components.mechanisms.sentinel.service.SentinelMechanism`**: The main component class that implements the `ProcessorMechanism` interface.
-   **`components.mechanisms.sentinel.config.SentinelMechanismConfig`**: The Pydantic model defining all tunable parameters, such as thresholds, weights, and penalty factors.
-   **Accepted Signals**:
    -   `IdeaGeneratedSignal`: Typically triggered by a Reactor rule when a new idea is created, initiating an assessment. The signal payload contains the `target_idea_id`.
-   **Produced Signals**:
    -   `TrustAssessmentSignal`: Emitted after an evaluation is complete. It contains the full `IdeaAssessment` object, including the trust score, stability, and rationale.

---

## Dependencies (Imports From)

-   `Mechanism_Gateway` (via `MechanismGatewayPort` for all external service calls)
-   `Application_Services` (for `IdeaService`, `FrameFactoryService`, `StageEvaluationService`)
-   `Event_and_Signal_System` (for signal definitions)
-   `Domain_Model` (for `Idea`, `IdeaAssessment`, and various `Ports`)
-   `Kernel` (for `NireonBaseComponent` and `ProcessResult`)

---

## Directory & Module Breakdown

-   `service.py`: The main `SentinelMechanism` class. It serves as the orchestrator, initializing all helper components, receiving the trigger signal, and driving the assessment process. It holds the final state (like current weights) and implements the component lifecycle methods (`analyze`, `react`, `adapt`).
-   `config.py`: Defines the `SentinelMechanismConfig` Pydantic model, which is the single source of truth for all configurable parameters and their validation rules.
-   `assessment_core.py`: Contains the `AssessmentCore` class, which manages the main assessment workflow. It creates the assessment `Frame`, calls the `StageEvaluationService` to get parameters, invokes the LLM for qualitative scores, calls the `NoveltyCalculator`, and applies scoring adjustments to produce the final `IdeaAssessment` object.
-   `novelty_calculator.py`: Implements the `NoveltyCalculator` class. Its sole responsibility is to compute the novelty score for an idea by fetching reference ideas, encoding them into vectors, and calculating the maximum similarity.
-   `scoring_adjustment.py`: Implements the `ScoringAdjustments` class. This module contains the logic for applying all contextual scoring modifications, such as the length penalty, progression bonus, and edge trust boost/decay.
-   `service_helpers/`: A directory of helper classes that abstract specific functionalities to keep the main service file clean.
    -   `initialization.py`: `InitializationHelper` handles dependency resolution and validation at startup.
    -   `processing.py`: `ProcessingHelper` contains the logic for parsing the input data, fetching the idea, and orchestrating the call to the `AssessmentCore`.
    -   `analysis.py` & `adaptation.py`: These helpers implement the logic for the `analyze`, `react`, and `adapt` lifecycle methods, separating self-monitoring and adaptation logic from the core processing workflow.
    -   `events.py`: The `EventPublisher` class abstracts the process of creating and publishing the final `TrustAssessmentSignal` through the `MechanismGateway`.
-   `metadata.py`: Defines the canonical `SENTINEL_METADATA` object for the component.
-   `errors.py`: Defines custom, specific exception types like `SentinelAssessmentError` and `SentinelLLMParsingError`.
-   `constants.py`: Holds shared constants used throughout the subsystem, such as default weights and event names.

---

## Configuration (`SentinelMechanismConfig`)

The Sentinel's evaluation behavior is highly tunable:

-   `trust_threshold`: (float, 0-10) The final score an idea must meet or exceed to be `stable`.
-   `min_axis_score`: (float, 0-10) The minimum score an idea must achieve on *every* axis.
-   `weights`: (list of 3 floats) The weights for [alignment, feasibility, novelty]. Must sum to 1.0.
-   `enable_length_penalty`: (bool) Enables/disables the penalty for overly long ideas.
-   `length_penalty_threshold`: (int) The character count above which the length penalty starts to apply.
-   `enable_edge_trust`: (bool) Enables/disables the trust bonus/decay based on an idea's graph neighbors.
-   `enable_progression_adjustment`: (bool) Enables/disables the trust bonus for ideas showing iterative development.
-   `objective_override`: (str, optional) If set, this objective will be used for all assessments, ignoring any objective from the incoming context.
-   `default_llm_score_on_error`: (float, 1-10) The score to use for alignment and feasibility if the LLM call fails or returns un-parsable text, ensuring system resilience.