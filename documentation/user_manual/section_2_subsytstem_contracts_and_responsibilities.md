## **2. NIREON V4 Subsystem Contracts & Responsibilities**

This section defines the public contracts, responsibilities, and architectural role for every major subsystem in the NIREON V4 repository. It is the authoritative reference for what each module does, what it exposes, and how it interacts with the rest of the system.

---

### **2.1. Kernel**

*   **Description:** The absolute core of the Nireon V4 system. Defines fundamental abstractions, the component lifecycle, the central component registry, and result objects. As the foundational layer, it has no dependencies on other project subsystems.
*   **🔑 Key Concepts:** `NireonBaseComponent`, `ComponentLifecycle`, `ComponentMetadata`, `ProcessResult`, `ComponentHealth`, `ComponentRegistry`.
*   **📄 Key Files:**
    *   `nireon_v4/core/base_component.py`
    *   `nireon_v4/core/lifecycle.py`
    *   `nireon_v4/core/registry/component_registry.py`
    *   `nireon_v4/core/results.py`
*   **🔗 Depends On:** None.

---

### **2.2. Domain Model**

*   **Description:** Defines the core business logic concepts (e.g., `Idea`, `Frame`, `CognitiveEvent`) and the abstract interfaces (`Ports`) that decouple the application from the infrastructure. This layer ensures the application's core is independent of specific technologies.
*   **🔑 Key Concepts:** `Idea`, `Frame`, `CognitiveEvent`, `LLMRequestPayload`, `NireonExecutionContext`, and all `...Port` protocols (e.g., `LLMPort`, `EventBusPort`, `MechanismGatewayPort`).
*   **📄 Key Files:**
    *   `nireon_v4/domain/ideas/idea.py`
    *   `nireon_v4/domain/frames.py`
    *   `nireon_v4/domain/cognitive_events.py`
    *   `nireon_v4/domain/context.py`
    *   `nireon_v4/domain/ports/` (entire directory)
*   **🔗 Depends On:** `Kernel`.

---

### **2.3. Event & Signal System**

*   **Description:** Manages the communication backbone of the system. It defines the hierarchy of signals (`EpistemicSignal`) that flow between components and provides the concrete event bus implementation for asynchronous, pub/sub-style communication.
*   **🔑 Key Concepts:** `EpistemicSignal` (and all subclasses like `IdeaGeneratedSignal`, `TrustAssessmentSignal`), `EventBusPort`, `MemoryEventBus`.
*   **📄 Key Files:**
    *   `nireon_v4/signals/base.py`, `nireon_v4/signals/core.py`
    *   `nireon_v4/infrastructure/event_bus/memory_event_bus.py`
*   **🔗 Depends On:** `Domain Model`.

---

### **2.4. Application Services**

*   **Description:** Contains high-level services that orchestrate domain logic and provide core application capabilities. These services act as the connective tissue between the abstract domain model and the concrete mechanisms, often encapsulating business rules that are not specific to any single mechanism.
*   **🔑 Key Concepts:** `FrameFactoryService`, `IdeaService`, `EmbeddingService`, `BudgetManagerPort`.
*   **📄 Key Files:**
    *   `nireon_v4/application/services/frame_factory_service.py`
    *   `nireon_v4/application/services/idea_service.py`
    *   `nireon_v4/application/services/embedding_service.py`
    *   `nireon_v4/application/services/budget_manager.py`
    *   `nireon_v4/application/services/stage_evaluation_service.py`
*   **🔗 Depends On:** `Kernel`, `Domain Model`, `Event_and_Signal_System`.

---

### **2.5. LLM Subsystem**

*   **Description:** A comprehensive subsystem for managing all interactions with Large Language Models. It is responsible for routing requests to different backends, applying context-specific parameters, handling failures with circuit breakers, and collecting performance metrics.
*   **🔑 Key Concepts:** `LLMRouter`, `ParameterService`, `CircuitBreaker`, `GenericHttpLLM`.
*   **📄 Key Files:**
    *   `nireon_v4/infrastructure/llm/router.py`
    *   `nireon_v4/infrastructure/llm/parameter_service.py`
    *   `nireon_v4/infrastructure/llm/factory.py`
    *   `nireon_v4/infrastructure/llm/generic_http.py`
*   **🔗 Depends On:** `Domain Model`, `Kernel`.
*   **🧠 LLM Guidance:** When reasoning about this subsystem, understand that its primary goal is resilience and flexibility. The router abstracts away specific LLM backends, so changes should focus on routing logic, parameter resolution, or circuit-breaking policies, not on hardcoding provider-specific details.

---

### **2.6. Mechanism Gateway**

*   **Description:** A crucial architectural façade that provides a single, controlled entry point for all cognitive mechanisms to interact with core system services. By routing all external calls through this gateway, we can enforce policies (like budgets), manage context (via Frames), and maintain a clean separation between mechanism logic and infrastructure concerns.
*   **🔑 Key Concepts:** `MechanismGatewayPort`, `MechanismGateway`, `CognitiveEvent`. This is the primary interface used by all mechanisms.
*   **📄 Key Files:**
    *   `nireon_v4/infrastructure/gateway/mechanism_gateway.py`
    *   `nireon_v4/domain/ports/mechanism_gateway_port.py`
*   **🔗 Depends On:** `LLM_Subsystem`, `Application_Services`, `Event_and_Signal_System`, `Domain_Model`.

---

### **2.7. Reactor Subsystem**

*   **Description:** A declarative, rule-based engine that forms the central nervous system of Nireon. It listens for all signals on the event bus and triggers component actions based on a set of conditions defined in YAML rule files. This allows for complex, emergent behaviors to be defined without changing core component code.
*   **🔑 Key Concepts:** `ReactorEngine`, `ReactorRule`, `RuleLoader`, `REL (Rule Expression Language)`.
*   **📄 Key Files:**
    *   `nireon_v4/reactor/engine/main.py`
    *   `nireon_v4/reactor/loader.py`
    *   `nireon_v4/reactor/core_rules.py`
    *   `nireon_v4/configs/reactor/rules/` (directory)
*   **🔗 Depends On:** `Event_and_Signal_System`, `Kernel`.

---

### **2.8. Explorer Mechanism**

*   **Description:** A generative agent focused on creating novel variations of existing ideas. The Explorer's primary goal is to increase the diversity of the idea space by applying mutations and transformations, effectively "exploring" the conceptual neighborhood around a seed idea. It operates within a `Frame` and uses the `MechanismGateway` for all external interactions.
*   **🔑 Key Concepts:** `ExplorerMechanism`, `ExplorerConfig`, `IdeaGeneratedSignal`.
*   **📄 Key Files:**
    *   `nireon_v4/components/mechanisms/explorer/service.py`
    *   `nireon_v4/components/mechanisms/explorer/config.py`
*   **🔗 Depends On:** `Mechanism_Gateway`, `Application_Services`, `Event_and_Signal_System`.
*   **💡 LLM Guidance:** The core intent is to generate novel and divergent variations. When analyzing or refactoring, prioritize logic that enhances creativity and exploration breadth/depth. Avoid adding logic that is overly critical or evaluative, as that is the Sentinel's role.

---

### **2.9. Sentinel Mechanism**

*   **Description:** An evaluative agent responsible for quality control. The Sentinel assesses ideas against multiple axes—such as alignment with objectives, feasibility, and novelty—to produce a `trust_score`. This score determines whether an idea is "stable" enough to proceed or should be rejected. It is the primary gatekeeper in the idea lifecycle.
*   **🔑 Key Concepts:** `SentinelMechanism`, `IdeaAssessment`, `TrustAssessmentSignal`.
*   **📄 Key Files:**
    *   `nireon_v4/components/mechanisms/sentinel/service.py`
    *   `nireon_v4/components/mechanisms/sentinel/assessment_core.py`
    *   `nireon_v4/components/mechanisms/sentinel/config.py`
*   **🔗 Depends On:** `Mechanism_Gateway`, `Application_Services`, `Event_and_Signal_System`.
*   **💡 LLM Guidance:** The core intent is critical, objective evaluation against defined axes. The LLM's role here is to provide structured, parseable JSON output, not creative text.

---

### **2.10. Catalyst Mechanism**

*   **Description:** A synthesis-focused agent designed to foster creativity by blending concepts from different domains. The Catalyst takes an existing idea and "injects" influence from a specified cross-domain vector, creating a novel, hybrid concept. Its goal is to bridge disciplinary boundaries and spark interdisciplinary thinking.
*   **🔑 Key Concepts:** `CatalystMechanism`, `CatalystConfig`, Vector Blending.
*   **📄 Key Files:**
    *   `nireon_v4/components/mechanisms/catalyst/service.py`
    *   `nireon_v4/components/mechanisms/catalyst/vector.py`
    *   `nireon_v4/components/mechanisms/catalyst/config.py`
*   **🔗 Depends On:** `Mechanism_Gateway`, `Application_Services`, `Event_and_Signal_System`.
*   **💡 LLM Guidance:** The core intent is creative synthesis and the blending of disparate concepts. The goal is a surprising but coherent hybrid idea. When analyzing, focus on the effectiveness of the vector blending and the quality of the LLM-generated hybrid text.

---

### **2.11. Math Engine (Principia)**

*   **Description:** A specialized subsystem for performing deterministic, symbolic mathematical computations. It is orchestrated by the `PrincipiaAgent`, which receives a `MathQuerySignal`, offloads the computation to a `MathPort` implementation (like the `SymPyAdapter`) via the Mechanism Gateway, and then uses an LLM to generate a human-readable explanation of the result, which is published as a `MathResultSignal`.
*   **🔑 Key Concepts:** `PrincipiaAgent`, `MathPort`, `SymPyAdapter`, `MathQuerySignal`, `MathResultSignal`.
*   **📄 Key Files:**
    *   `nireon_v4/components/mechanisms/math_agents/principia_agent.py`
    *   `nireon_v4/infrastructure/math/sympy_adapter.py`
*   **🔗 Depends On:** `Mechanism_Gateway`, `Event_and_Signal_System`.
*   **💡 LLM Guidance:** The LLM's role in this subsystem is strictly for explanation. It translates the structured, symbolic output from the math computation into a clear, human-readable, step-by-step explanation. It should not perform any computation itself.

---

### **2.12. Proto-Plane (Execution & Generation)**

*   **Description:** A powerful subsystem for executing arbitrary, declarative tasks ("Proto blocks") in a secure, sandboxed environment. It consists of the `ProtoGenerator`, which translates natural language into a YAML Proto block, and the `ProtoEngine`, which executes these blocks. This allows for dynamic, on-the-fly creation of complex computational tasks.
*   **🔑 Key Concepts:** `ProtoBlock`, `ProtoEngine`, `ProtoGenerator`, `ProtoTaskSignal`, `ProtoResultSignal`.
*   **📄 Key Files:**
    *   `nireon_v4/proto_engine/service.py`
    *   `nireon_v4/proto_engine/executors/docker.py`
    *   `nireon_v4/proto_generator/service.py`
    *   `nireon_v4/domain/proto/base_schema.py`
*   **🔗 Depends On:** `LLM_Subsystem`, `Event_and_Signal_System`, `Application_Services`.

---

### **2.13. Security & RBAC**

*   **Description:** The Role-Based Access Control (RBAC) system for Nireon. It includes the policy engine responsible for evaluating permissions and the decorators used to protect sensitive functions and methods. Policies are defined in `bootstrap_rbac.yaml` and loaded during the bootstrap process.
*   **🔑 Key Concepts:** `RBACPolicyEngine`, `RBACRule`, `@requires_permission` decorator.
*   **📄 Key Files:**
    *   `nireon_v4/security/rbac_engine.py`
    *   `nireon_v4/security/decorators.py`
    *   `nireon_v4/configs/default/bootstrap_rbac.yaml`
*   **🔗 Depends On:** `Kernel`.

---

### **2.14. Bootstrap System**

*   **Description:** This subsystem is responsible for the entire startup sequence of the Nireon application. It orchestrates a series of well-defined phases to load configurations, instantiate all components, wire up their dependencies, and bring the system online in a predictable and reliable state.
*   **🔑 Key Concepts:** `BootstrapOrchestrator`, `BootstrapPhase`, `BootstrapResult`, `ManifestProcessor`.
*   **📄 Key Files:**
    *   `nireon_v4/bootstrap/core/main.py`
    *   `nireon_v4/bootstrap/phases/` (entire directory)
    *   `nireon_v4/bootstrap/processors/` (entire directory)
    *   `nireon_v4/bootstrap/context/bootstrap_context_builder.py`
*   **🔗 Depends On:** All other subsystems. It is the master orchestrator.

---

### **2.15. Testing & Runners**

*   **Description:** A collection of scripts, test cases, and fixtures used for testing, debugging, and running specific parts of the system. This subsystem is not part of the production runtime and contains all developer-facing tools for validation.
*   **🔑 Key Concepts:** Unit tests, integration tests, standalone runners for specific subsystems.
*   **📄 Key Files:**
    *   `nireon_v4/tests/` (entire directory)
    *   `nireon_v4/01_math_runner/` (entire directory)
    *   `nireon_v4/02_proto_runner/` (entire directory)
    *   `nireon_v4/run_explorer_test.py`
*   **🔗 Depends On:** All other subsystems (for testing purposes).