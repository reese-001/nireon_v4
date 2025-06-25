## **5. Core Systems Implementation Details**

This section provides a detailed look into the implementation of NIREON V4's foundational subsystems: the Bootstrap process, the Reactor engine, the Mechanism Gateway, and the LLM Subsystem.

---

### **5.1. Bootstrap System**

The NIREON V4 bootstrap component, located in `nireon_v4/bootstrap/`, is responsible for initializing the entire system from configuration files into a ready-to-run state.

*   **🔑 Key Concepts:** `BootstrapOrchestrator`, `BootstrapPhase`, `BootstrapResult`, `ManifestProcessor`.
*   **📄 Key Files:**
    *   `bootstrap/core/main.py`: The main entry point and orchestrator.
    *   `bootstrap/phases/`: Directory containing all individual startup phases.
    *   `bootstrap/processors/manifest_processor.py`: Logic for parsing `standard.yaml`.
    *   `bootstrap/result_builder.py`: Constructs the final `BootstrapResult`.

#### **5.1.1. Phased Execution**

Bootstrap operates through a sequence of well-defined phases, ensuring a predictable and ordered startup. The `BootstrapOrchestrator` executes these phases sequentially:

1.  **`AbiogenesisPhase`:** The very first phase. Its critical role is to perform "preloading" of essential services declared in the manifest with `preload: true`. This ensures that foundational services like the `EventBusPort` are available to all subsequent phases.
2.  **`ContextFormationPhase`:** Establishes the core `BootstrapContext`, which provides a shared environment (registry, health reporter, etc.) for all subsequent phases.
3.  **`RegistrySetupPhase`:** Prepares the `ComponentRegistry` for operation, enabling features like metadata tracking and self-certification.
4.  **`FactorySetupPhase`:** Instantiates and registers core factories and services that are required by mechanisms but are not defined in the manifest, such as the `FrameFactoryService` and `MechanismGateway`.
5.  **`ManifestProcessingPhase`:** The main component loading phase. It parses the `standard.yaml` manifest, discovers all declared components (services, mechanisms, etc.), and uses processors to instantiate and register them.
6.  **`ComponentInitializationPhase`:** Iterates through all registered components that have `requires_initialize: true` and calls their `.initialize()` method, allowing them to resolve dependencies and set up their internal state.
7.  **`InterfaceValidationPhase`:** Performs a validation pass on all initialized components to ensure they conform to their declared contracts and metadata.
8.  **`RBACSetupPhase`:** Loads Role-Based Access Control policies from `bootstrap_rbac.yaml` and sets up the `RBACPolicyEngine`.
9.  **`ReactorSetupPhase`:** Loads all rule definitions from `configs/reactor/rules/` and initializes the `MainReactorEngine`. It also bridges the Reactor to the `EventBusPort`, subscribing it to all relevant signals.
10. **`LateRebindingPhase`:** A final "cleanup" phase that iterates through components to replace any placeholder dependencies (like a `PlaceholderLLMPort`) with the real instances that were loaded during the bootstrap process.

---

### **5.2. Reactor Engine & Rules**

The Reactor is the central nervous system of NIREON, orchestrating component interactions in a declarative, event-driven manner.

*   **🔑 Key Concepts:** `MainReactorEngine`, `ConditionalRule`, `Action`, `Rule Expression Language (REL)`.
*   **📄 Key Files:**
    *   `reactor/engine/main.py`: The core engine implementation.
    *   `reactor/loader.py`: Loads and parses YAML rule files.
    *   `reactor/core_rules.py`: Defines the Python class for rules.
    *   `reactor/expressions/rel_engine.py`: The simple expression parser.

#### **5.2.1. Rule-Based Architecture**

Instead of components calling each other directly, they emit **signals**. The Reactor listens for these signals and triggers actions based on **rules** defined in YAML. This decouples components and allows for complex, emergent behavior to be configured without changing code.

#### **5.2.2. Rule Anatomy (YAML)**

All rules are defined in YAML files within `configs/reactor/rules/`. A rule consists of two main parts: `conditions` and `actions`.

```yaml
# In configs/reactor/rules/core.yaml
- id: "idea_generated_to_trust_eval"
  description: "Evaluate trust for newly generated ideas"
  priority: 20
  conditions:
    # 1. The signal must be of this type.
    - type: "signal_type_match"
      signal_type: "IdeaGeneratedSignal"
  actions:
    # 2. If it matches, trigger this component.
    - type: "trigger_component"
      component_id: "sentinel_instance_01"
      # 3. Map data from the signal to the component's input.
      input_data_mapping:
        target_idea_id: "signal.payload.id" 
```

#### **5.2.3. Rule Expression Language (REL)**

For more complex conditions, the Reactor uses a simple, safe expression language (REL) in the `payload_expression` condition.

*   **Syntax:** Python-like expressions.
*   **Context:** You can access the signal's data using dot notation (e.g., `signal.trust_score`, `payload.is_stable`).
*   **Safety:** The engine is sandboxed and only allows safe operations (comparisons, basic math, boolean logic).

**Example using REL:**

```yaml
- id: "route_business_idea_to_quantifier_fixed"
  conditions:
    - type: "signal_type_match"
      signal_type: "TrustAssessmentSignal"
    - type: "payload_expression"
      # This entire string is evaluated by the REL engine.
      expression: >
        payload.is_stable == True and
        payload.trust_score > 6.0 and
        'business' in lower(payload.idea_text)
```

---

### **5.3. Mechanism Gateway & The A➜F➜CE Framework**

The `MechanismGateway` is a critical façade that standardizes how all mechanisms interact with the rest of the system, enforcing the **Agent ➜ Frame ➜ Cognitive Event (A➜F➜CE)** model.

*   **🔑 Key Concepts:** `MechanismGateway`, `Frame`, `CognitiveEvent`.
*   **📄 Key Files:**
    *   `infrastructure/gateway/mechanism_gateway.py`: The concrete implementation.
    *   `domain/frames.py`: The `Frame` data structure.
    *   `domain/cognitive_events.py`: The `CognitiveEvent` data structure.

#### **5.3.1. The A➜F➜CE Ontology**

1.  **Agent:** An active component (like a Mechanism) that performs cognitive work.
2.  **Frame:** A bounded context for that work. A Frame contains goals, rules, resource budgets, and an audit trail. All significant actions must occur within a Frame.
3.  **Cognitive Event (CE):** A record of a single, atomic action taken by an Agent within a Frame.

Instead of a mechanism calling an LLM directly, it does this:

1.  **Gets a Frame:** It requests a new, specific `Frame` from the `FrameFactoryService`.
2.  **Creates a CE:** It packages its request (e.g., an LLM prompt) into a `CognitiveEvent`, referencing the Frame's ID.
3.  **Submits to Gateway:** It sends the `CognitiveEvent` to the `MechanismGateway`.

The Gateway then validates the event against the Frame's budget and policies before dispatching it to the appropriate service (e.g., the `LLMRouter`). This ensures all actions are contextualized, traceable, and governed.

---

### **5.4. LLM Subsystem**

The LLM subsystem is designed for flexibility, resilience, and context-awareness.

*   **🔑 Key Concepts:** `LLMRouter`, `ParameterService`, `GenericHttpLLM`, `CircuitBreaker`.
*   **📄 Key Files:**
    *   `infrastructure/llm/router.py`: The main routing logic.
    *   `infrastructure/llm/parameter_service.py`: Resolves LLM settings based on context.
    *   `infrastructure/llm/generic_http.py`: A flexible adapter for calling any HTTP-based LLM API.
    *   `configs/default/llm_config.yaml`: The central configuration file for all models and routes.

#### **5.4.1. Configuration-Driven Routing**

The `llm_config.yaml` file defines everything:

*   **Models:** Defines specific backends (e.g., `nano_default`), their provider, API endpoint, payload template, and authentication method.
*   **Routes:** Creates logical aliases (e.g., `chat_fast`, `sentinel_axis_scorer`) that map to specific models. This allows system behavior to be changed without modifying component code.
*   **Parameters:** Specifies default `temperature`, `max_tokens`, etc., with overrides based on the `EpistemicStage` of the request.

#### **5.4.2. Call Flow**

1.  A mechanism creates a `CognitiveEvent` for an `LLM_ASK`.
2.  The `MechanismGateway` receives it.
3.  The Gateway consults the `ParameterService` to resolve the final LLM settings based on the event's `stage` and `role`, as well as the active `Frame`'s `llm_policy`.
4.  The Gateway calls the `LLMRouter`'s `call_llm_async` method.
5.  The `LLMRouter` uses the `route` specified in the settings to select the correct backend (e.g., `nano_default`).
6.  It calls the backend instance (e.g., a `GenericHttpLLM` adapter).
7.  The adapter makes the actual HTTP request to the LLM provider and returns the `LLMResponse`.


### **5.5. Proto-Plane Subsystem**

The Proto-Plane is a powerful, specialized subsystem within NIREON V4 designed for the secure, sandboxed execution of declarative, on-the-fly computational tasks. It allows the system to generate and run code to perform complex analysis, simulations, or visualizations that go beyond the capabilities of the core mechanisms.

*   **🔑 Key Concepts:** `ProtoBlock`, `ProtoEngine`, `ProtoGenerator`, `ProtoTaskSignal`.
*   **📄 Key Files:**
    *   `proto_engine/service.py`: The `ProtoEngine` and `ProtoGateway` implementations.
    *   `proto_generator/service.py`: The `ProtoGenerator` implementation.
    *   `domain/proto/base_schema.py`: Pydantic models for `ProtoBlock` and its dialects.
    *   `proto_engine/executors/`: Directory containing the sandboxed executors (`DockerExecutor`, `SubprocessExecutor`).
    *   `02_proto_runner/run_proto_test.py`: The primary test script for this subsystem.

#### **5.5.1. The Proto-Plane Philosophy: Declarative & Sandboxed Execution**

The core idea of the Proto-Plane is to enable the system to **write code to solve its own problems**. Instead of pre-programming every possible analytical function, NIREON can define a task in natural language, have an LLM translate it into a self-contained, executable script (a `ProtoBlock`), and then run that script in a secure sandbox.

This provides two major benefits:

1.  **Extensibility:** The system can perform new types of analysis (e.g., financial modeling, graph analysis, scientific simulation) without requiring changes to the core Python codebase.
2.  **Security:** All generated code is executed in a sandboxed environment (like a Docker container) with strict resource limits (CPU, memory) and no access to the host file system or network, preventing malicious or poorly-written code from impacting the main application.

#### **5.5.2. Architecture & Workflow**

The Proto-Plane consists of two main components that work in tandem:

1.  **`ProtoGenerator`:** An agent that takes a natural language request (e.g., "Plot the impact of a 25% tariff on a business") and uses an LLM to generate a valid YAML `ProtoBlock`.
2.  **`ProtoGateway` & `ProtoEngine`:**
    *   The `ProtoGateway` is a simple router that listens for `ProtoTaskSignal`s.
    *   It routes the task to the appropriate `ProtoEngine` based on the `ProtoBlock`'s `eidos` (dialect), such as `'math'`.
    *   The `ProtoEngine` takes the `ProtoBlock` and uses an **Executor** (`DockerExecutor` or `SubprocessExecutor`) to run the code in a sandbox.
    *   Upon completion, the `ProtoEngine` emits either a `ProtoResultSignal` (on success) or a `ProtoErrorSignal` (on failure).

**High-Level Workflow:**

```mermaid
graph TD
    A(1. High-Trust Idea) -->|Triggers QuantifierAgent| B(2. ProtoGenerator);
    B -->|Generates YAML| C(3. ProtoBlock);
    C -->|Emits ProtoTaskSignal| D[4. ProtoGateway];
    D -->|Routes by Dialect| E[5. ProtoEngine (e.g., Math Engine)];
    E -->|Uses Executor| F(6. Sandboxed Execution);
    F -- Success --> G(7. ProtoResultSignal);
    F -- Failure --> H(8. ProtoErrorSignal);
```

#### **5.5.3. The `ProtoBlock` Schema**

A `ProtoBlock` is a declarative YAML structure that defines a complete, executable task.

**Example `ProtoBlock`:**
```yaml
# In examples/math_proto_example.yaml
schema_version: proto/1.0
id: proto_math_plot_sine_wave
eidos: math # The dialect determines which engine and validator to use.
description: "Generates and saves a plot of a sine wave."
objective: "Visualize the sine function over one period."
function_name: plot_sine_wave # The entry point function within the code.
inputs: # Parameters passed to the function.
  output_filename: "sine_wave.png"
code: | # The Python code to be executed.
  import numpy as np
  import matplotlib.pyplot as plt
  def plot_sine_wave(output_filename: str):
      # ... function logic ...
      plt.savefig(output_filename)
      plt.close()
      return {"status": "success"}
requirements: ["numpy", "matplotlib"] # Pip requirements for the sandbox.
limits: # Resource limits for the sandbox.
  timeout_sec: 15
  memory_mb: 256
```

#### **5.5.4. Execution and Sandboxing**

The `ProtoEngine` uses an **Executor** to run the code.

*   **`DockerExecutor` (Recommended for Production):**
    1.  Creates a temporary workspace directory.
    2.  Writes the `code` from the `ProtoBlock` into an `execute.py` script.
    3.  Writes the `inputs` into an `inputs.json` file.
    4.  If `requirements` are specified, it writes them to `requirements.txt`.
    5.  It spins up a Docker container from a pre-built image (e.g., `nireon-proto-math:latest`), mounting the workspace.
    6.  It runs a command inside the container that first installs requirements and then executes the Python script.
    7.  It captures the `stdout` from the container, which contains a special `RESULT_JSON:` line with the execution output.
    8.  It copies any generated files (artifacts) out of the workspace into a persistent `artifacts/` directory before cleaning up the workspace.

*   **`SubprocessExecutor` (For Local Development):**
    *   Follows a similar process but runs the `execute.py` script in a local `subprocess` instead of a Docker container.
    *   **Warning:** This mode is less secure and does not enforce memory limits on Windows. It is intended for development and testing where Docker may not be available.

This robust execution model ensures that even dynamically generated code runs in a safe, predictable, and resource-constrained environment.

### **5.6. Developer's Guide to Proto-Blocks**

This guide provides a practical, step-by-step process for developers to create, test, and integrate new `ProtoBlock` tasks into the NIREON V4 system.

#### **5.6.1. When to Use a Proto-Block**

A Proto-Block is the right tool for the job when you need to perform a task that is:

*   **Computational & Deterministic:** Ideal for tasks involving mathematics, data analysis, simulations, or complex calculations that have a clear input and a structured output.
*   **Self-Contained:** The logic can be expressed in a single Python script with well-defined dependencies.
*   **Sandboxed:** The task requires external libraries (`numpy`, `matplotlib`, etc.) that you don't want to install in the main NIREON environment, or you want to run it with strict resource limits for security and stability.
*   **Declarative:** You want to define the *what* (the task) in a simple YAML file, leaving the *how* (the execution environment) to the `ProtoEngine`.

It is **not** the right tool for tasks that require complex interactions with multiple, live NIREON services or long-running stateful processes.

#### **5.6.2. Step-by-Step: Creating a New Proto-Block**

Let's create a new `ProtoBlock` to calculate and plot the Mandelbrot set, a classic computational task perfect for this system.

**Step 1: Define the Goal**

Our goal is to create a Proto-Block that:
1.  Calculates the Mandelbrot set for a given image size and iteration count.
2.  Saves the resulting image as a PNG artifact.
3.  Returns statistics about the computation (e.g., standard deviation of iterations).

**Step 2: Write the Python Code**

First, write the core logic as a self-contained Python script. The key is to have a single entry-point function that takes all necessary parameters as arguments.

```python
# This is the code that will eventually go inside the Proto-Block's `code` field.

import numpy as np
import matplotlib.pyplot as plt

def generate_mandelbrot(width, height, max_iterations):
    """
    Generates the Mandelbrot set and saves it as an image.
    """
    x = np.linspace(-2.025, 0.6, width)
    y = np.linspace(-1.125, 1.125, height)
    c = x[:, np.newaxis] + 1j * y[np.newaxis, :]
    z = c
    
    # The escape-time algorithm
    escape_times = np.full(c.shape, max_iterations, dtype=np.int32)
    for i in range(max_iterations):
        z = z**2 + c
        diverged = np.abs(z) > 2.0
        # Update escape times for newly diverged points
        escape_times[diverged & (escape_times == max_iterations)] = i
        z[diverged] = 2.0 # Prevent overflow

    # Create and save the plot
    plt.figure(figsize=(width / 100, height / 100))
    plt.imshow(escape_times.T, cmap='magma', extent=[-2.025, 0.6, -1.125, 1.125])
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig('mandelbrot_set.png', dpi=100)
    plt.close()

    # Return structured results
    return {
        "status": "success",
        "message": "Mandelbrot set generated and saved.",
        "image_size": f"{width}x{height}",
        "boundary_stats": {
            "mean_iterations": float(np.mean(escape_times)),
            "std_dev_iterations": float(np.std(escape_times)),
            "boundary_pixels": int(np.sum(escape_times < max_iterations))
        }
    }
```

**Step 3: Create the Proto-Block YAML File**

Now, wrap this code and its metadata into a YAML file. We can save this as `mandelbrot_proto_example.yaml` in the `examples/` directory for testing.

```yaml
# In examples/mandelbrot_proto_example.yaml
schema_version: proto/1.0
id: proto_mandelbrot_high_res
eidos: math # Use the 'math' dialect for its pre-installed libraries.
description: "Generates a high-resolution image of the Mandelbrot set."
objective: "Visualize the complexity of the Mandelbrot set boundary."
function_name: generate_mandelbrot # Must match the function name in the code.
inputs:
  width: 1200
  height: 800
  max_iterations: 150
code: |
  # --- Paste the entire Python script from Step 2 here ---
  import numpy as np
  import matplotlib.pyplot as plt

  def generate_mandelbrot(width, height, max_iterations):
      """
      Generates the Mandelbrot set and saves it as an image.
      """
      x = np.linspace(-2.025, 0.6, width)
      y = np.linspace(-1.125, 1.125, height)
      c = x[:, np.newaxis] + 1j * y[np.newaxis, :]
      z = c
      
      escape_times = np.full(c.shape, max_iterations, dtype=np.int32)
      for i in range(max_iterations):
          z = z**2 + c
          diverged = np.abs(z) > 2.0
          escape_times[diverged & (escape_times == max_iterations)] = i
          z[diverged] = 2.0

      plt.figure(figsize=(width / 100, height / 100))
      plt.imshow(escape_times.T, cmap='magma', extent=[-2.025, 0.6, -1.125, 1.125])
      plt.axis('off')
      plt.tight_layout(pad=0)
      plt.savefig('mandelbrot_set.png', dpi=100)
      plt.close()

      return {
          "status": "success",
          "message": "Mandelbrot set generated and saved.",
          "image_size": f"{width}x{height}",
          "boundary_stats": {
              "mean_iterations": float(np.mean(escape_times)),
              "std_dev_iterations": float(np.std(escape_times)),
              "boundary_pixels": int(np.sum(escape_times < max_iterations))
          }
      }
requirements:
  # numpy and matplotlib are already in the 'math' dialect's Docker image,
  # but it's good practice to list them.
  - numpy
  - matplotlib
limits:
  timeout_sec: 25 # This might take longer than the default 10 seconds.
  memory_mb: 512  # Use more memory for the large numpy array.
```

**Step 4: Test the Proto-Block with the Runner Script**

The easiest way to test a new Proto-Block is with the dedicated runner script. This bootstraps the system and executes the task directly.

Run this command from the `nireon_v4/` root directory:

```sh
python -m 02_proto_runner.run_proto_test --proto examples/mandelbrot_proto_example.yaml
```

**Expected Output:**

The script will:
1.  Bootstrap the NIREON system.
2.  Publish a `ProtoTaskSignal` containing your YAML block.
3.  The `ProtoEngine` will pick it up and execute it.
4.  You will see a "SUCCESS" message with the JSON result from your Python function.
5.  Crucially, it will list the **artifact** that was created: `mandelbrot_set.png`.
6.  You can find this generated image inside the `runtime/proto/artifacts/` directory.

#### **5.6.3. Integrating with the Reactor**

Once your Proto-Block is tested and working, you can create a `Reactor` rule to trigger it automatically.

For example, a rule in `configs/reactor/rules/advanced.yaml` could trigger our Mandelbrot generator:

```yaml
- id: "trigger_mandelbrot_on_complexity_query"
  description: "When an idea mentions 'fractal complexity', generate a Mandelbrot set."
  namespace: "analysis_triggers"
  priority: 60
  enabled: true
  conditions:
    - type: "signal_type_match"
      signal_type: "IdeaGeneratedSignal"
    - type: "payload_expression"
      expression: "'fractal' in lower(payload.idea_content) and 'complexity' in lower(payload.idea_content)"
  actions:
    - type: "emit_signal"
      signal_type: "ProtoTaskSignal"
      payload:
        # Here, you can either embed the whole block or, more cleanly,
        # have a system to load it from a file. For simplicity, we embed.
        proto_block:
          schema_version: proto/1.0
          id: proto_mandelbrot_from_rule
          eidos: math
          description: "Auto-triggered Mandelbrot set generation."
          objective: "Visualize fractal complexity based on an incoming idea."
          function_name: generate_mandelbrot
          # ... rest of the proto block from Step 3 ...
```
With this rule in place, any `IdeaGeneratedSignal` containing the right keywords will automatically trigger the execution of your Proto-Block, demonstrating the power of combining the Reactor with the Proto-Plane.