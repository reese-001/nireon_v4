## **9. Glossary & Quick Reference**

This final section serves as a quick reference for the key concepts, terminology, and syntax used throughout the NIREON V4 system.

---

### **9.1. Glossary of Core Concepts**

*   **Adapter:** A concrete implementation of a `Port` that connects NIREON's domain logic to a specific external technology (e.g., `SentenceTransformerAdapter` is an adapter for the `EmbeddingPort`). Resides in the `infrastructure/` layer.

*   **Agent:** An autonomous component that performs cognitive work, such as a `Mechanism`. It is the primary initiator of `CognitiveEvents`.

*   **A➜F➜CE (Agent ➜ Frame ➜ Cognitive Event):** The core architectural ontology of NIREON V4. It mandates that any action (`CognitiveEvent`) by an `Agent` must occur within a defined `Frame`. This model is enforced by the `MechanismGateway`.

*   **Bootstrap:** The process of initializing the entire NIREON V4 system from configuration files. It is orchestrated by the `BootstrapOrchestrator` and executed in a series of `BootstrapPhase`s.

*   **Cognitive Event (CE):** A structured, atomic, and traceable record of a cognitive act (e.g., an LLM call, a signal publication). CEs are created by `Agents` and processed by the `MechanismGateway`.

*   **Component:** The fundamental building block of NIREON. A component can be a service, mechanism, observer, or manager. All active components inherit from `NireonBaseComponent`.

*   **ComponentRegistry:** A central, singleton service that holds all instantiated `Component`s and their `ComponentMetadata`. It enables service discovery and system introspection.

*   **Domain:** The architectural layer containing the core business logic and concepts of the system, such as the `Idea` and `Frame` data models and the abstract `Port` interfaces. It is completely independent of any specific infrastructure.

*   **Epistemic Signal:** A data structure representing an event that has occurred within the system. Signals are the primary means of asynchronous communication, published to the `EventBus` and processed by the `Reactor`.

*   **Frame:** A bounded, interpretive context for cognitive work. A `Frame` defines goals, resource budgets, and policies that govern the `CognitiveEvents` occurring within it.

*   **FrameFactoryService:** The application service responsible for creating, managing, and retrieving `Frames`.

*   **Infrastructure:** The architectural layer that contains concrete implementations of `Ports` (i.e., `Adapters`). This layer deals with external technologies like databases, file systems, and third-party APIs.

*   **Kernel (`core/`):** The foundational layer of the system. It defines the absolute core abstractions like `NireonBaseComponent`, `ComponentRegistry`, and `ProcessResult`, and has no dependencies on other NIREON subsystems.

*   **LLM Router:** The central component within the LLM Subsystem responsible for receiving all LLM requests, applying parameters, and routing them to the appropriate backend model based on configuration.

*   **Mechanism:** A specialized type of `Component` that directly participates in the idea evolution lifecycle (e.g., `Explorer`, `Sentinel`, `Catalyst`).

*   **Mechanism Gateway:** A critical façade that mediates all interactions between `Mechanisms` and core system services. It enforces the A➜F➜CE model.

*   **Port:** An abstract interface defined in the `domain/` layer that specifies a contract for a service (e.g., `LLMPort`, `IdeaRepositoryPort`). This allows the application to be decoupled from specific implementations.

*   **Proto-Block:** A declarative, YAML-defined task that can be executed in a secure, sandboxed environment by the `ProtoEngine`.

*   **Reactor:** The rule-based engine that serves as the system's central nervous system. It listens for all `EpistemicSignals` and triggers `Component` actions based on a set of declarative YAML rules.

*   **Rule Expression Language (REL):** A simple, safe, and sandboxed expression language used in `Reactor` rules to evaluate conditions against a signal's payload (e.g., `payload.trust_score > 0.8`).

---

### **9.2. Quick Reference: Rule Expression Language (REL)**

Used in the `payload_expression` condition of a Reactor rule.

| Category         | Examples                                                  | Notes                                                              |
| :--------------- | :-------------------------------------------------------- | :----------------------------------------------------------------- |
| **Context Access** | `payload.is_stable` <br> `signal.trust_score`               | Access data from the triggering signal using dot notation.         |
| **Operators**      | `==`, `!=`, `>`, `<`, `>=`, `<=`, `and`, `or`, `not`, `in` | Standard Python operators.                                         |
| **Functions**      | `exists(payload.optional_field)`<br>`lower(payload.text)` | `exists()` checks if a value is not None. `lower()` is for strings. |

**Example:**
```yaml
# This expression checks three conditions on a TrustAssessmentSignal
expression: "payload.is_stable == True and signal.trust_score > 0.8 and 'business' in lower(payload.idea_text)"
```

---

### **9.3. Quick Reference: Core Signals**

These are some of the most important signals that drive the main epistemic loop.

| Signal Name                  | Emitter(s)                                 | Purpose                                                            |
| :--------------------------- | :----------------------------------------- | :----------------------------------------------------------------- |
| `SeedSignal`                 | External, Test Scripts                     | To inject a new seed idea and kick off an exploration cycle.       |
| `IdeaGeneratedSignal`        | `ExplorerMechanism`                        | To announce that a new idea variation has been created.            |
| `TrustAssessmentSignal`      | `SentinelMechanism`                        | To announce the result of an idea's evaluation.                    |
| `ProtoTaskSignal`            | `ProtoGenerator`, `QuantifierAgent`        | To request the execution of a sandboxed `Proto-Block`.             |
| `ProtoResultSignal`          | `ProtoEngine`                              | To announce the successful result of a `Proto-Block` execution.    |
| `MathResultSignal`           | `PrincipiaAgent`                           | To announce the result and explanation of a math computation.      |
| `GenerativeLoopFinishedSignal` | `QuantifierAgent`, `Reactor` Rules       | To signal that a branch of the reasoning process has concluded.    |

---

### **9.4. System Architecture Diagram**

This diagram provides a high-level visual summary of the key subsystems and their interactions in NIREON V4.

```mermaid
graph TD
    subgraph User/External
        A[External Trigger / User]
    end

    subgraph Declarative Plane
        B(Bootstrap)
        C(YAML Configs)
        D(Reactor Rules)
        B -- reads --> C
        B -- loads --> D
    end
    
    subgraph Application Core
        E(Component Registry)
        F(Event Bus)
        G(Reactor Engine)
        B -- populates --> E
        G -- listens to --> F
        G -- reads --> D
    end

    subgraph Mechanisms
        H(Explorer)
        I(Sentinel)
        J(Quantifier / Principia)
        K(Catalyst)
    end

    subgraph Gateway & Services
        L(Mechanism Gateway)
        M(Frame Factory)
        N(LLM Subsystem)
        O(Persistence Layer)
        P(Embedding Subsystem)
        L -- uses --> M & N & O & P & F
    end

    A --> F
    H & I & J & K -- "sends CognitiveEvent" --> L
    L -- "publishes Signal" --> F
    F -- "notifies" --> G
    G -- "triggers Component" --> H & I & J & K

    classDef core fill:#E8EAF6,stroke:#3F51B5,stroke-width:2px;
    classDef mech fill:#E0F2F1,stroke:#00796B,stroke-width:2px;
    classDef decl fill:#FFF3E0,stroke:#F57C00,stroke-width:2px;
    classDef gateway fill:#FCE4EC,stroke:#D81B60,stroke-width:2px;
    
    class B,C,D decl;
    class E,F,G core;
    class H,I,J,K mech;
    class L,M,N,O,P gateway;
