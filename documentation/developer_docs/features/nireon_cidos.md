# NIREON CIDOS Math-Engine Integration Guide

> **Status** - Draft v0.1  |  Last updated 2025-06-19

---

## 1  Purpose & Scope

This document turns the conceptual discussion around **CIDOS** (Cognitive Intent Dialect Of System) into a concrete implementation roadmap. It covers:

1. Why CIDOS exists and how it unifies runtime intent across NIREON.
2. A minimal **base schema** plus the **math dialect** needed today.
3. A lightweight **sub-process sandbox** runner (no Docker) for executing generated Python + matplotlib + LaTeX tasks.
4. How CIDOS artifacts flow through the existing **Reactor** and component ecosystem.
5. Guard-rails: cost budgeting, validation, security, and prompt-caching.
6. Milestones for a staged rollout that stays cheap but future-proof.

---

## 2  Terminology

| Term                    | Meaning                                                                     |
| ----------------------- | --------------------------------------------------------------------------- |
| **CIDOS**               | YAML-based meta-language for declarative intent in NIREON.                  |
| **Dialect**             | A specialization of CIDOS (e.g. `reactor_rule`, `math`).                    |
| **CIDOS Block**         | A YAML document that follows any CIDOS dialect schema.                      |
| **Math-Task**           | A CIDOS block with `eidos: math`, containing code, inputs, etc.             |
| **Sub-process sandbox** | A short-lived Python interpreter launched with resource limits.             |
| **Runner**              | Component that loads a math-task block, executes, emits `MathResultSignal`. |

---

## 3  High-Level Vision

```mermaid
flowchart LR
    Idea -->|"How can math enhance?"| LLM_Plan
    LLM_Plan -->|CIDOS math block| MathRunner
    MathRunner -->|Result (plot + latex + numeric)| EventBus
    EventBus --> Reactor
    Reactor --> DownstreamAgents
```

* **Agent** asks the enhancement question.
* **High-tier LLM** returns a *CIDOS math block*.
* **MathRunner** executes it in a safe sub-process → produces graph(s), LaTeX, numbers.
* Results re-enter the DAG via `MathResultSignal` for trust evaluation, reporting, etc.

---

## 4  CIDOS Base Schema (v1.0)

```yaml
schema_version: cidos/1.0
id: MATH_001                # unique
description: ...            # human summary
metadata:
  created_by: mirror_agent
  created_at: "2025-06-19T21:00:00Z"
context:
  frame_id: "F-ROOT-..."
  run_id: dag_run_123
# dialect-specific fields follow ↓
```

*Required*: `schema_version`, `id`, `description`, `context.frame_id`
*Optional*: anything under `metadata`

---

## 5  `eidos: math` Dialect

```yaml
eidos: math           # identifies the dialect
id: MATH_011
objective: Simulate cost under tariff policy
inputs:
  tariff: 0.07
  volume: 1000
function_name: compute_cost
code: |
  import matplotlib.pyplot as plt
  def compute_cost(tariff, volume):
      base_cost = 15
      return (base_cost + tariff * 100) * volume

  def plot_result():
      tariffs = [0.01 * i for i in range(1, 10)]
      results = [compute_cost(t, 1000) for t in tariffs]
      plt.plot(tariffs, results)
      plt.xlabel("Tariff Rate")
      plt.ylabel("Total Cost")
      plt.title("Cost vs. Tariff Rate")
      plt.savefig("cost_plot.png")
equation_latex: "C = (15 + 100 \\cdot t) \\cdot V"
requirements: ["matplotlib"]     # future extensibility
limits:
  timeout_sec: 10
  memory_mb: 256
```

*Validator rules*

* `code` must define `function_name`.
* `inputs` keys must match function params or be optional.
* `limits` default to `(10 s, 256 MB)` if omitted.

---

## 6  Sandbox Execution Layer (No Docker)

1. **Launch** `python -I -` as a **sub-process** (isolated site, no user site).
2. Pass the `code` block via stdin; run under `resource.setrlimit()` caps.
3. Inject `inputs` dict before calling `function_name(**inputs)`.
4. Capture:

   * return value → `numeric_result`
   * any file outputs (PNG, PDF) saved in `runtime/math/artifacts/` → paths
   * printed stdout/stderr for logs.
5. Terminate sub-process and wipe temp dir.

> **Why not Docker yet?**
> *Fast to implement*, zero external deps, leverages CPython `-I` isolation + `resource` limits.

---

## 7  Signals & Reactor Integration

| Signal                | Purpose                                                     |
| --------------------- | ----------------------------------------------------------- |
| `MathQuerySignal`     | (existing) used when components ask for math help.          |
| `MathTaskCIDOSSignal` | *new* — carries a CIDOS math block to `MathRunner`.         |
| `MathResultSignal`    | (existing) delivers numeric, plot paths, LaTeX, commentary. |
| `MathErrorSignal`     | *new* — sandbox or validation failure.                      |

`reactor_rule` example to route CIDOS blocks:

```yaml
id: math_task_router
eidos: reactor_rule
conditions:
  - type: signal_type_match
    signal_type: MathTaskCIDOSSignal
actions:
  - type: trigger_component
    component_id: dynamic_math_runner
    input_data_mapping:
      cidos_block: "signal.payload.cidos_yaml"
```

---

## 8  Resource & Cost Controls

* **BudgetManager** already tracks component CPU/LLM budget.
  Extend it with a `math_cpu_ms` dimension.
* Reject tasks where `limits.timeout_sec > 30` unless `override=true`.
* Log high-tier LLM usage under `math_task.plan_cost_tokens` field.

---

## 9  Prompt-Caching Strategy

1. Store the CIDOS base schema + math examples in **Frame memory** at startup.
2. On first math-task invocation per session, include full schema in the prompt.
3. Afterwards send only the tag:
   *“Respond using **CIDOS math v1.0** format.”*
4. Cache successful CIDOS templates in Redis (keyed by objective hash).

---

## 10  Governance & Security

| Measure          | Detail                                                         |
| ---------------- | -------------------------------------------------------------- |
| Static guards    | Regex deny list for `import os`, `import subprocess`, `open(`. |
| Runtime caps     | `resource.setrlimit` for CPU & RSS.                            |
| Exception filter | Only whitelisted traceback lines returned to user.             |
| Audit log        | Persist CIDOS, stdout, stderr, time, memory.                   |

---

## 11  CIDOS Versioning & Validation

* `schema_version:` prefix (e.g. `cidos/1.0`).
* JSON-Schema per dialect stored in `configs/cidos/schemas/`.
* `cidolib.validate(cidos_yaml)` raises before execution.

---

## 12  Extensibility Roadmap

1. **v1.1** - add `steps:` array for multi-phase tasks.
2. **v1.2** - add `backend:` selector (`sympy`, `sage`, `numpy_gpu`).
3. **v2.0** - multi-task pipelines & reusable modules, CIDOS-generated charts auto-embedded in reports.

---

## 13  Implementation Milestones

| Phase | Deliverable                                 | Owner | Target Date |
| ----- | ------------------------------------------- | ----- | ----------- |
|  0    | Agree CIDOS v1.0 schema                     | —     | 2025-06-24  |
|  1    | `cidolib` validator + JSON-Schema           | —     | 06-28       |
|  2    | `dynamic_math_runner` (sub-process sandbox) | —     | 07-05       |
|  3    | New signals + reactor rule wiring           | —     | 07-07       |
|  4    | Prompt templates and cache service          | —     | 07-10       |
|  5    | Governance tests & performance tuning       | —     | 07-14       |

---

## 14  Appendix A - Minimal Runner Stub

```python
import resource, subprocess, textwrap, uuid, json
from pathlib import Path

def run_cidos_math(cidos):
    code = cidos['code']
    limits = cidos.get('limits', {})
    timeout = limits.get('timeout_sec', 10)
    mem_mb = limits.get('memory_mb', 256)

    script_path = Path(f"/tmp/{uuid.uuid4()}.py")
    script_path.write_text(textwrap.dedent(code))

    def preexec():
        resource.setrlimit(resource.RLIMIT_AS, (mem_mb * 1024 ** 2,)*2)
    try:
        proc = subprocess.run(["python", "-I", script_path], capture_output=True, text=True,
                              preexec_fn=preexec, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"error": "timeout"}
    return {
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "returncode": proc.returncode,
    }
```

---

*End of v0.1*
