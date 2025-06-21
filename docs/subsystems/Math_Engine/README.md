# Math Engine Subsystem

**Description:** A specialized subsystem for performing deterministic, symbolic mathematical computations. It is orchestrated by the `PrincipiaAgent`, which receives a `MathQuerySignal`, offloads the computation to a `MathPort` implementation (like the `SymPyAdapter`) via the Mechanism Gateway, and then uses an LLM to generate a human-readable explanation of the result, which is published as a `MathResultSignal`.

---

## Public API / Contracts

- **`components.mechanisms.math_agents.principia_agent.PrincipiaAgent`**: The agent that orchestrates the math task. It is the primary consumer of `MathQuerySignal`.
- **`infrastructure.math.sympy_adapter.SymPyAdapter`**: The concrete implementation of the `MathPort`, using the SymPy library for symbolic math.
- **Accepted Signals:** `MathQuerySignal` (defined in `signals/core.py`). This is the entry point for requesting a computation.
- **Produced Signals:** `MathResultSignal` (defined in `signals/core.py`). This is the final output containing the result and explanation.
- **Interface Port:** `domain.ports.math_port.MathPort`. This is the abstract contract that any math computation backend must implement.

---

## Dependencies (Imports From)

- `Mechanism_Gateway`: Used to make external calls for both computation (`MATH_COMPUTE`) and LLM-based explanation (`LLM_ASK`).
- `Event_and_Signal_System`: For defining and receiving the input/output signals (`MathQuerySignal`, `MathResultSignal`).
- `Domain_Model`: For the `MathPort` interface and `CognitiveEvent` structure.
- `Kernel`: For the `NireonBaseComponent` class.

---

## Directory Layout (Conceptual)

```mermaid
graph TD
    subgraph MathEngine [Math Engine]
        direction LR
        A[principia_agent.py] -- uses --> B((Gateway));
        B -- requests compute --> C[sympy_adapter.py];
        B -- requests explanation --> D((LLM Subsystem));
    end

    subgraph Signals
        E[MathQuerySignal] --> A;
        A --> F[MathResultSignal];
    end

    subgraph Domain
        G[MathPort]
    end

    C -- implements --> G;