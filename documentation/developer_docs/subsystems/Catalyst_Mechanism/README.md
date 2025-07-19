# Catalyst Mechanism Subsystem

**Description:** A synthesis-focused generative mechanism designed to foster creativity by blending concepts from disparate domains. The Catalyst takes an existing `Idea` and "injects" influence from a specified cross-domain vector, performing vector arithmetic to create a novel, hybrid semantic representation. This new vector is then used to prompt an LLM to regenerate the idea's text, resulting in a synthesized concept that bridges disciplinary boundaries.

Beyond simple generation, the Catalyst is an adaptive agent. It monitors its own output for semantic diversity and duplication. If its creativity stagnates, it can dynamically increase its blend strength or activate "anti-constraints" in its prompts to force more divergent thinking, making it a powerful engine for interdisciplinary exploration and creative breakthroughs.

---

## Core Concepts & Functionality

The Catalyst mechanism operates on several core principles that make it a sophisticated component within the NIREON system.

-   **Frame-Based Processing**: Every catalysis task is executed within a dedicated `Frame` created by the `FrameFactoryService`. This provides an isolated execution context with its own resource budget (e.g., max LLM calls), cognitive policies (LLM temperature, model preference), and a clear link to the overarching objective. This ensures that even highly creative tasks are properly managed and accounted for.

-   **Vector Blending**: The heart of the Catalyst is its use of vector arithmetic (`vector.py`). It does not simply concatenate text. Instead, it blends the semantic vector of an existing idea (`original_vector`) with a pre-encoded vector from a different knowledge domain (`domain_vector`). The core operation is a weighted average: `new_vector = (1 - α) * original_vector + α * domain_vector`, where `α` is the `blend_strength`. The resulting vector is a true semantic hybrid.

-   **LLM-Powered Synthesis**: After a new hybrid vector is created, the mechanism doesn't just stop there. It uses the LLM to translate this abstract semantic representation back into a coherent, human-readable idea. The `prompt_builder.py` constructs a detailed prompt that includes the original idea's text, the name of the inspiration domain, and potentially a set of anti-constraints, guiding the LLM to perform the final creative synthesis.

-   **Adaptive Behavior**: The Catalyst is not a static generator. It includes adaptive loops (`adaptation.py`) to maintain creative output over time:
    -   **Duplication Handling**: It can probabilistically check for semantic duplication in its outputs. If detected, it increases its `blend_range` (the min/max values for `α`), pushing it to be more aggressive in its blending. This effect has a configurable cooldown period.
    -   **Anti-Constraint Activation**: It maintains a running history of the semantic distance of its outputs. If the average distance drops below a configured threshold (i.e., ideas are becoming too similar), it can activate "anti-constraints" in its prompts, explicitly telling the LLM to avoid certain themes and forcing more divergent output.

-   **Configuration-Driven**: The mechanism's behavior is extensively controlled by the `CatalystMechanismConfig` (`config.py`). This Pydantic model allows for fine-tuning of nearly every aspect, from the base application rate and blend strengths to the aggressiveness of its adaptive behaviors.

-   **Service-Oriented Integration**: The Catalyst is a well-behaved component that relies on standard system contracts (`ports`). It uses the `MechanismGatewayPort` for all external calls (LLM, events), the `EmbeddingPort` to ensure ideas have vectors, and the `IdeaService` for persistence, all resolved via the `ServiceResolutionMixin`.

---

## Detailed Feature Breakdown

-   **Cross-Domain Injection**: The core capability. Blends an idea from one context with a vector representing an entire domain (e.g., "Biomimicry," "Renaissance Art," "Quantum Physics").
-   **Configurable Blend Strength**: The `blend_low` and `blend_high` parameters define the range from which a random blend strength is chosen, allowing control over how much influence the cross-domain vector has.
-   **Duplication Detection & Cooldown**: To prevent creative stagnation, the mechanism can detect when it's producing ideas that are too similar to existing ones. When this happens, it adaptively increases the blend strength to force more novelty. After a set number of `duplication_cooldown_steps`, the blend strength will reset to its base level.
-   **Adaptive Anti-Constraints**: When semantic diversity drops, the mechanism can be configured to generate and apply a list of "anti-constraints" to its LLM prompts. This forces the LLM to avoid specific concepts, pushing it into new areas of the solution space. These constraints have a configurable expiry step.
-   **LLM Prompt Engineering**: The `prompt_builder.py` module constructs sophisticated prompts that provide clear instructions to the LLM, including the seed idea, inspiration domain, objective, and the dynamic anti-constraint block.
-   **Component Lifecycle Integration**: As a full `NireonBaseComponent`, it implements `analyze`, `react`, and `adapt` methods, allowing it to self-monitor, perform periodic maintenance (like resetting blend cooldowns), and propose configuration changes to improve its own performance.

---

## Public API / Contracts

-   **`components.mechanisms.catalyst.service.CatalystMechanism`**: The main component class that implements the `ProducerMechanism` interface.
-   **`components.mechanisms.catalyst.config.CatalystMechanismConfig`**: A Pydantic model defining all tunable parameters for the mechanism. This is a key part of its public contract for configuration.
-   **Accepted Signals**:
    -   `CATALYSIS_REQUEST`: Triggers a catalysis process on a batch of ideas.
    -   `CROSS_DOMAIN_INJECTION_REQUEST`: A more specific trigger for a single idea.
-   **Produced Signals**:
    -   `IdeaCatalyzedSignal`: Emitted upon successful creation of a hybrid idea, containing the new idea object, original idea ID, blend strength, and semantic distance metrics.
    -   `CrossDomainBlendSignal`, `HybridConceptGeneratedSignal`: Other potential signals indicating stages of the process.

---

## Dependencies (Imports From)

-   `Mechanism_Gateway` (via `MechanismGatewayPort`)
-   `Application_Services` (via `FrameFactoryService`)
-   `Event_and_Signal_System` (for signal definitions)
-   `Domain_Model` (for `Idea`, `Vector`, and various `Ports`)
-   `Kernel` (for `NireonBaseComponent`, `ProcessResult`)

---

## Directory & Module Breakdown

The subsystem is organized into modules with clear responsibilities:

-   `service.py`: The primary component entry point. It orchestrates the entire catalysis process, manages state (like blend ranges and anti-constraints), and integrates with other NIREON services via the `MechanismGatewayPort`. It implements the core `_process_impl`, `analyze`, `react`, and `adapt` methods.
-   `config.py`: Defines the `CatalystMechanismConfig` Pydantic model. This file serves as the single source of truth for all configurable parameters, including their types, default values, and validation rules.
-   `processing.py`: Contains pure, stateless helper functions for core data processing tasks, such as validating domain vectors, selecting which ideas to apply catalysis to, and computing post-blend metrics.
-   `vector.py`: Implements the low-level vector mathematics. `VectorOperations` class handles the blending of vectors, normalization, and calculation of semantic distance and interdisciplinary scores.
-   `prompt_builder.py`: Responsible for constructing the final text prompt sent to the LLM. It intelligently assembles the prompt from a template, incorporating the seed idea, inspiration domain, and the dynamic anti-constraint block when needed.
-   `adaptation.py`: Contains the logic for the mechanism's adaptive loops. It defines how the blend range is adjusted in response to duplication and how anti-constraints expire over time.
-   `service_helpers/catalyst_event_helper.py`: A helper class that abstracts away the details of creating and persisting new ideas and publishing signals to the event bus, keeping the main `service.py` cleaner.
-   `metadata.py`: Defines the canonical `CATALYST_METADATA` object, which describes the component's identity, capabilities, dependencies, and epistemic role to the rest of the system.
-   `errors.py`: Defines custom exception types specific to the Catalyst, allowing for more granular error handling (e.g., `VectorBlendError`, `DuplicationError`).
-   `types.py`: Defines simple type aliases and constants used throughout the subsystem for clarity and consistency (e.g., `BlendRange`, `DomainVectors`).

---

## Configuration (`CatalystMechanismConfig`)

The mechanism's behavior can be extensively tuned via the following parameters:

-   `application_rate`: (float, 0-1) The probability that the Catalyst will be applied to any given idea.
-   `blend_low` / `blend_high`: (float, 0-1) The minimum and maximum strength for the cross-domain vector's influence (`α`). A random value is chosen from this range for each catalysis.
-   `max_blend_low` / `max_blend_high`: (float, 0-1) The maximum values the blend range can be adapted to when handling duplication. This acts as a ceiling to prevent runaway adaptation.
-   `duplication_check_enabled`: (bool) If `True`, enables the adaptive duplication handling behavior.
-   `duplication_check_probability`: (float, 0-1) The probability of performing a duplication check for any given catalyzed idea.
-   `duplication_cooldown_steps`: (int) The number of steps to wait after a duplication-driven adaptation before the blend range can reset to its base values.
-   `duplication_aggressiveness`: (float) A multiplier for how much the blend range increases upon detecting duplication. Higher values cause more aggressive adaptation.
-   `anti_constraints_enabled`: (bool) If `True`, enables the adaptive anti-constraint behavior to combat low diversity.
-   `anti_constraints_count`: (int) The maximum number of anti-constraint themes to include in the prompt.
-   `anti_constraints_diversity_threshold`: (float, 0-1) The average semantic distance threshold. If diversity falls below this, anti-constraints are activated.
-   `prompt_template`: (str, optional) Allows overriding the default LLM prompt template for custom synthesis instructions.
-   `default_llm_policy_for_catalysis`: (dict) Default LLM settings (temperature, max_tokens, etc.) for the `Frame` created by this mechanism.
-   `default_resource_budget_for_catalysis`: (dict) Default resource budget (LLM calls, CPU time) for the `Frame` created by this mechanism.