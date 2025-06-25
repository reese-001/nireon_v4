## **4. Developing NIREON V4 Components**

This section outlines the standard process for implementing a component (mechanism, observer, manager, or core service) that participates in the NIREON V4 runtime, aligns with its philosophy, and integrates with the bootstrap, gateway, and reactor systems.

---

### **4.1. Core Principles of a NIREON Component**

Every component in NIREON V4 is designed around a set of core principles that ensure consistency, testability, and architectural integrity.

*   **Standard Lifecycle:** All components inherit from `NireonBaseComponent`, which provides a consistent set of lifecycle methods (`initialize`, `process`, `analyze`, etc.). This guarantees predictable behavior during bootstrap and runtime.
*   **Centralized Configuration:** All configuration is managed through Pydantic models and loaded from YAML files. Components should never hardcode parameters; they should be configurable.
*   **Gateway-Mediated Interactions:** Mechanisms do not directly call services like LLMs or the event bus. Instead, they create a `CognitiveEvent` and send it to the `MechanismGateway`, which enforces policies (like budgets) and handles the actual service interaction.
*   **Declarative Registration:** Components are not manually instantiated in code. They are declared in a manifest file (`standard.yaml`), and the bootstrap process is responsible for creating and registering them.
*   **Self-Description:** Each component is associated with a `ComponentMetadata` object that describes its purpose, capabilities, and dependencies.

---

### **4.2. The `NireonBaseComponent` Lifecycle Contract**

By inheriting from `nireon_v4/core/base_component.py:NireonBaseComponent`, your component automatically adheres to the NIREON lifecycle. You only need to implement the `_..._impl` methods for the behaviors you need.

```python
# In your component's service.py
from core.base_component import NireonBaseComponent
from core.results import ProcessResult
from domain.context import NireonExecutionContext

class MyMechanism(NireonBaseComponent):
    # This is called by the bootstrap process after the component is created.
    # Use it to resolve dependencies from the registry.
    async def _initialize_impl(self, context: NireonExecutionContext) -> None:
        self.gateway = context.component_registry.get_service_instance(...)
        # ... other initial setup
    
    # This is the primary method for processing data.
    # The 'data' payload is typically determined by the Reactor rule that triggers it.
    async def _process_impl(self, data: Any, context: NireonExecutionContext) -> ProcessResult:
        # ... your core logic here ...
        return ProcessResult(success=True, message="Processing complete.")

    # Other optional lifecycle methods you can override:
    async def analyze(self, context: NireonExecutionContext) -> AnalysisResult: ...
    async def react(self, context: NireonExecutionContext) -> List[SystemSignal]: ...
    async def adapt(self, context: NireonExecutionContext) -> List[AdaptationAction]: ...
    async def health_check(self, context: NireonExecutionContext) -> ComponentHealth: ...
    async def shutdown(self, context: NireonExecutionContext) -> None: ...
```

---

### **4.3. The Mechanism Gateway Pattern**

A key architectural pattern in NIREON V4 is that **mechanisms do not directly call services**. Instead, they formulate a `CognitiveEvent` and pass it to the `MechanismGateway`.

**Why?**

*   **Policy Enforcement:** The Gateway can check budgets, apply rate limits, and enforce other policies before allowing the call to proceed.
*   **Context Management:** The Gateway ensures all interactions happen within the context of a `Frame`, providing traceability and resource scoping.
*   **Decoupling:** Mechanisms only need to know how to "ask" the Gateway for an action, not how to implement that action. This makes mechanisms simpler and more reusable.

**Example: Requesting an LLM Response**

```python
# Inside a mechanism's _process_impl method
from domain.cognitive_events import CognitiveEvent, LLMRequestPayload
from domain.epistemic_stage import EpistemicStage

# 1. Create the LLM payload
llm_payload = LLMRequestPayload(
    prompt="Generate a creative variation of this idea...",
    stage=EpistemicStage.EXPLORATION,
    role="idea_generator"
)

# 2. Create the CognitiveEvent, specifying the Frame ID and the service call type
cognitive_event = CognitiveEvent(
    frame_id=current_frame.id, # The active frame for this task
    owning_agent_id=self.component_id,
    service_call_type='LLM_ASK',
    payload=llm_payload
)

# 3. Send the event to the Gateway and await the response
# 'self.gateway' should be resolved during initialization
llm_response = await self.gateway.process_cognitive_event(cognitive_event, context)

# 4. Use the result
new_idea_text = llm_response.text
```

---

### **4.4. Step-by-Step: Creating a New Mechanism**

Let's create a new mechanism called `Synthesizer`.

#### **✅ Step 1: Scaffold the Component's Module**

Create the directory and files for your new component.

```sh
mkdir -p nireon_v4/components/mechanisms/synthesizer
cd nireon_v4/components/mechanisms/synthesizer
touch __init__.py service.py config.py metadata.py
```

*   `__init__.py`: Exports key classes.
*   `service.py`: Contains the `SynthesizerMechanism` class.
*   `config.py`: Defines `SynthesizerConfig` using Pydantic.
*   `metadata.py`: Defines the `SYNTHESIZER_METADATA` object.

#### **✅ Step 2: Define Configuration (`config.py`)**

Use Pydantic to define the configuration schema.

```python
# nireon_v4/components/mechanisms/synthesizer/config.py
from pydantic import BaseModel, Field

class SynthesizerConfig(BaseModel):
    """Configuration for the Synthesizer mechanism."""
    synthesis_depth: int = Field(default=2, ge=1, le=5, description="How many ideas to combine.")
    enable_novelty_boost: bool = Field(default=True, description="Boost trust for highly novel synthesized ideas.")

    class Config:
        extra = "forbid"
```

#### **✅ Step 3: Define Metadata (`metadata.py`)**

Create a `ComponentMetadata` instance. This object makes the component self-describing.

```python
# nireon_v4/components/mechanisms/synthesizer/metadata.py
from core.lifecycle import ComponentMetadata

SYNTHESIZER_METADATA = ComponentMetadata(
    id="synthesizer_mechanism_default",
    name="Synthesizer Mechanism",
    version="1.0.0",
    description="Combines multiple ideas to create a novel synthesis.",
    category="mechanism",
    epistemic_tags=["synthesizer", "combiner", "integrator"],
    requires_initialize=True,
    dependencies={'MechanismGatewayPort': '*'} # Declares a dependency
)
```

#### **✅ Step 4: Implement the Component (`service.py`)**

Inherit from `NireonBaseComponent` and implement the required logic.

```python
# nireon_v4/components/mechanisms/synthesizer/service.py
from core.base_component import NireonBaseComponent
from core.results import ProcessResult
from domain.context import NireonExecutionContext
from domain.ports.mechanism_gateway_port import MechanismGatewayPort
from .config import SynthesizerConfig
from .metadata import SYNTHESIZER_METADATA

class SynthesizerMechanism(NireonBaseComponent):
    METADATA_DEFINITION = SYNTHESIZER_METADATA
    ConfigModel = SynthesizerConfig

    def __init__(self, config, metadata_definition, **kwargs):
        super().__init__(config, metadata_definition)
        self.cfg: SynthesizerConfig = self.ConfigModel(**self.config)
        self.gateway: MechanismGatewayPort | None = None

    async def _initialize_impl(self, context: NireonExecutionContext) -> None:
        self.gateway = context.component_registry.get_service_instance(MechanismGatewayPort)
        context.logger.info(f"Synthesizer '{self.component_id}' initialized.")

    async def _process_impl(self, data: Any, context: NireonExecutionContext) -> ProcessResult:
        self.logger.info(f"Synthesizer '{self.component_id}' processing data...")
        # Your logic here, using self.gateway to interact with other services
        # e.g., create a CognitiveEvent and call self.gateway.process_cognitive_event(...)
        return ProcessResult(success=True, message="Synthesis complete.")
```

#### **✅ Step 5: Add to Manifest (`standard.yaml`)**

Define an instance of your new component in `configs/manifests/standard.yaml`.

```yaml
# In configs/manifests/standard.yaml, under the 'mechanisms' section:
mechanisms:
  # ... other mechanisms
  synthesizer_instance_01:
    enabled: true
    class: "components.mechanisms.synthesizer.service:SynthesizerMechanism"
    metadata_definition: "components.mechanisms.synthesizer.metadata:SYNTHESIZER_METADATA"
    config: "configs/default/mechanisms/{id}.yaml" # Path to default config
    config_override: # Optional inline overrides
      synthesis_depth: 3 
```

#### **✅ Step 6: Create a Default Config File**

Create the default configuration file referenced in the manifest.

```yaml
# In configs/default/mechanisms/synthesizer_instance_01.yaml
synthesis_depth: 2
enable_novelty_boost: true
```

#### **✅ Step 7: Add a Reactor Rule to Trigger It**

In a file like `configs/reactor/rules/core.yaml`, add a rule to trigger your new mechanism.

```yaml
# In configs/reactor/rules/core.yaml
- id: "high_trust_to_synthesizer"
  description: "When a high-trust idea is assessed, trigger the Synthesizer."
  namespace: "core"
  priority: 30
  enabled: true
  conditions:
    - type: "signal_type_match"
      signal_type: "TrustAssessmentSignal"
    - type: "payload_expression"
      expression: "payload.trust_score > 7.5 and payload.is_stable == True"
  actions:
    - type: "trigger_component"
      component_id: "synthesizer_instance_01" # Matches the ID from the manifest
      input_data_mapping:
        idea_to_synthesize_id: "payload.idea_id"
```

#### **✅ Step 8: Write Tests**

Create unit and integration tests for your component in the `tests/` directory to ensure its logic is correct and it integrates properly with the system. Your tests should mock dependencies like the `MechanismGateway` to isolate the component's logic.