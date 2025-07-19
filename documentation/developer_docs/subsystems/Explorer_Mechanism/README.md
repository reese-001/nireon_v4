# Explorer Mechanism Subsystem

**Description:** The Explorer is a generative agent designed for systematic idea space exploration and variation. Its primary function is to take a "seed" `Idea` and autonomously generate a multitude of novel, divergent variations. Unlike a simple generator, the Explorer operates within a structured, frame-based execution model. Each exploration task is encapsulated within a dedicated `Frame`, complete with its own resource budget, cognitive policies, and an internal audit trail.

The mechanism employs configurable strategies (e.g., depth-first, breadth-first) to navigate the conceptual space around the seed idea. It leverages the system's `MechanismGateway` to make managed calls to LLMs for text generation and can interact with the embedding subsystem to ensure novelty. Furthermore, it is an adaptive component that analyzes its own performance, proposing changes to its configuration to improve the quality and diversity of its output over time.

---

## Core Concepts & Functionality

The Explorer's design is built on several sophisticated architectural patterns that make it a robust and integral part of the NIREON system.

-   **Frame-Based Execution Model**: At the heart of the Explorer's operation is its use of `Frames`. When a request is received, the Explorer doesn't just process it immediately. Instead, it uses the `FrameFactoryService` to create a new, dedicated `Frame` for the exploration task. This provides several key advantages:
    -   **Isolation**: Each exploration run is isolated from others.
    -   **Resource Management**: Each frame is assigned a resource budget (e.g., max LLM calls, CPU time) defined in the Explorer's configuration, preventing runaway generation.
    -   **Contextualization**: The `Frame` holds all the context for the task, including the objective, seed idea ID, and current exploration depth.
    -   **Auditing**: The Explorer actively logs a detailed audit trail of its actions (LLM calls, idea generation, errors) directly into the `Frame`'s `context_tags`, making post-mortem analysis of a specific run trivial.

-   **Asynchronous Background Processing**: To avoid blocking the main system loop, the core exploration logic is launched as a non-blocking background task (`asyncio.create_task`). The initial `process()` call returns immediately with a `task_id` and `frame_id`, allowing the caller to monitor the task's progress asynchronously while the system continues other operations.

-   **Structured Exploration Strategies**: The mechanism's search pattern is not random. It is guided by the `exploration_strategy` parameter in its configuration. This allows it to perform more systematic and predictable exploration of the idea space:
    -   `depth_first`: Focuses on iterating deeply from a single variation.
    -   `breadth_first`: Generates a wide array of variations at each level before going deeper.
    -   `random`: Applies more stochastic transformations.
    -   `llm_guided`: (Implied) Could leverage the LLM to choose the next exploration path.

-   **LLM-Powered Variation**: The generation of new idea text is powered by an LLM. The Explorer constructs detailed prompts (`_build_llm_prompt`) that include the seed idea, the overall objective, and a "creativity factor" to guide the LLM's output. All LLM calls are routed through the `MechanismGateway` as `CognitiveEvent`s, ensuring they are subject to system-wide policies and budgeting.

-   **Adaptive Lifecycle (`analyze`, `react`, `adapt`)**: The Explorer is a dynamic component that self-monitors and self-improves.
    -   **`analyze`**: It periodically analyzes statistics from recent exploration frames to calculate its own performance metrics, such as variation generation efficiency, LLM success rate, and frame error rates.
    -   **`react`**: It listens for system signals, most notably `EmbeddingComputedSignal`. This allows it to asynchronously request embeddings for its generated ideas and react when the results are available, correlating them back to the correct `Frame`. It also handles request timeouts and cleans up aged-out requests.
    -   **`adapt`**: Based on the insights from its `analyze` step, it can propose `AdaptationAction`s. For example, if it detects low variation quality, it might propose increasing its `divergence_strength` or reducing the LLM `temperature` to improve reliability.

-   **Service-Oriented Design**: The Explorer strictly adheres to the Ports and Adapters pattern. It does not import concrete infrastructure implementations directly. Instead, it depends on abstract ports like `MechanismGatewayPort` and `FrameFactoryService`, which are resolved at runtime using the `ServiceResolutionMixin`.

---

## Detailed Feature Breakdown

-   **Configurable Exploration Parameters**: The `ExplorerConfig` allows for deep customization of `max_depth`, `max_variations_per_level`, and the core `exploration_strategy`.
-   **Diversity Filtering**: Can be configured to automatically reject newly generated variations if their semantic distance to the seed idea is below a `diversity_threshold`, ensuring a minimum level of novelty.
-   **Semantic Exploration**: When enabled, the mechanism can use vector embeddings and semantic distance to guide its exploration, moving into conceptually novel areas rather than just performing textual transformations.
-   **Asynchronous Embedding & Back-pressure**: The Explorer can request embeddings for its new ideas. To prevent overwhelming the embedding service, it has a `max_pending_embedding_requests` cap and a timeout for responses, making it a well-behaved citizen in a high-throughput system.
-   **Novelty-Seeking Retries**: The mechanism can be configured to re-attempt idea generation with an increased "perturbation" if the initial attempt doesn't meet a `min_novelty_threshold_for_acceptance`.
-   **Resource Budgeting**: Every exploration task is constrained by the `default_resource_budget_for_exploration` defined in its configuration, which is applied to the `Frame` it creates.

---

## Public API / Contracts

-   **`components.mechanisms.explorer.service.ExplorerMechanism`**: The main component class.
-   **`components.mechanisms.explorer.config.ExplorerConfig`**: The comprehensive Pydantic model for configuring the mechanism's behavior.
-   **Accepted Signals**:
    -   `SEED_SIGNAL`, `EXPLORATION_REQUEST`: Triggers an exploration task. The payload can contain a seed idea, objective, and other contextual metadata.
-   **Produced Signals**:
    -   `IdeaGeneratedSignal`: Emitted via the `ExplorerEventHelper` for each valid new idea variation that is created and persisted.
    -   `GenerativeLoopFinishedSignal`: Emitted when an entire exploration task within a frame is complete.
    -   `TrustAssessmentSignal`: Can be emitted to request a trust evaluation of the newly generated ideas.
    -   `EmbeddingRequestSignal`: Emitted to request a vector embedding for a new idea variation.

---

## Dependencies (Imports From)

-   `Mechanism_Gateway` (via `MechanismGatewayPort`)
-   `Application_Services` (via `FrameFactoryService`)
--   `Event_and_Signal_System` (for signal definitions)
-   `Domain_Model` (for `Idea`, `Frame`, and various `Ports`)
-   `Kernel` (for `NireonBaseComponent`, `ProcessResult`, `ComponentMetadata`)

---

## Directory & Module Breakdown

-   `service.py`: The main component implementation. It contains the `ExplorerMechanism` class, orchestrates the creation of exploration frames, launches background tasks, and implements the `analyze`/`react`/`adapt` lifecycle methods.
-   `config.py`: Defines the `ExplorerConfig` Pydantic model. This is the canonical source for all tunable parameters, providing strict validation and clear definitions for each setting.
-   `service_helpers/explorer_event_helper.py`: A helper class that abstracts away the boilerplate of publishing signals and creating new `Idea` instances through the `IdeaService`. This keeps the main `service.py` focused on the core exploration logic.
-   `errors.py`: Defines the `ExplorerErrorCode` enum, providing a structured set of error codes for more precise error reporting and handling.

---

## Configuration (`ExplorerConfig`)

The Explorer's behavior is highly configurable through the `ExplorerConfig` model. Key parameters include:

-   **Exploration Control**:
    -   `exploration_strategy`: The core algorithm to use (`depth_first`, `breadth_first`, `random`).
    -   `max_depth`: The maximum number of recursive steps the exploration can take.
    -   `max_variations_per_level`: How many new ideas to generate from a single parent idea at each depth.
    -   `application_rate`: The probability of applying exploration to a given idea.

-   **Novelty and Diversity**:
    -   `divergence_strength`: The base strength of the mutation applied to an idea's vector during semantic exploration.
    -   `enable_diversity_filter`: A boolean to enable or disable the filtering of non-novel ideas.
    -   `diversity_threshold`: The semantic distance below which a new variation is considered a duplicate and discarded.
    -   `min_novelty_threshold_for_acceptance`: The minimum novelty score required for a perturbed idea to be accepted.
    -   `max_retries_for_novelty`: The number of attempts the mechanism will make to generate an idea that meets the novelty threshold.

-   **LLM & Resource Management**:
    -   `enable_llm_enhancement`: A master switch to allow the LLM to refine or expand on generated ideas.
    -   `creativity_factor`: A `temperature`-like setting (0.0 to 1.0) passed to the LLM to control the creativity of its output.
    -   `default_resource_budget_for_exploration`: The default budget (LLM calls, etc.) assigned to each exploration `Frame`.
    -   `max_parallel_llm_calls_per_frame`: A concurrency limit to prevent overwhelming the LLM service from a single exploration task.

-   **Asynchronous Service Integration**:
    -   `request_embeddings_for_variations`: If `True`, the mechanism will emit `EmbeddingRequestSignal`s for new ideas.
    -   `max_pending_embedding_requests`: A back-pressure mechanism to limit the number of outstanding embedding requests.
    -   `embedding_response_timeout_s`: The time to wait for an `EmbeddingComputedSignal` before assuming the request has failed.