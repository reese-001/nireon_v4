# CIDOS User Manual

> Version 0.1  •  Last updated 2025-06-19

---

## About This Manual

This manual teaches **analysts, data scientists, and power-users** how to leverage **CIDOS** (Cognitive Intent Dialect Of System) inside NIREON to add mathematical insight, visualisations, and structured logic to any idea-flow. No back-end coding required—everything is declared in YAML.

> **Target Audience** - Users comfortable editing simple text files, basic Python syntax, and reading YAML.

---

## 1  What Is CIDOS?

CIDOS is NIREON’s declarative language for expressing *what* you want the system to do—math simulations, rule triggers, critique plans—without telling it *how* in imperative code. Think of CIDOS blocks as **self-contained instruction cards** that agents can pick up and execute.

### 1.1  Dialects

| Dialect        | Purpose                                    |
| -------------- | ------------------------------------------ |
| `reactor_rule` | Signal routing & automation                |
| `math`         | Executable mathematics with plots & LaTeX  |
| *(future)*     | `synthesis_plan`, `evaluation_strategy`, … |

---

## 2  Quick-Start (5 Minutes)

1. **Install prerequisites**

   ```bash
   pip install matplotlib pyyaml
   ```
2. **Create a file** `tariff_cost.yaml` with:

   ```yaml
   schema_version: cidos/1.0
   eidos: math
   id: COST_SIM_001
   description: Cost vs. tariff sensitivity
   inputs:
     tariff: 0.05
     volume: 1000
   function_name: compute_cost
   code: |
     import matplotlib.pyplot as plt
     def compute_cost(tariff, volume):
         return (15 + tariff * 100) * volume
     def plot_result():
         tariffs = [0.01*i for i in range(1,10)]
         costs   = [compute_cost(t, 1000) for t in tariffs]
         plt.plot(tariffs, costs)
         plt.xlabel('Tariff Rate')
         plt.ylabel('Total Cost')
         plt.title('Cost vs Tariff')
         plt.savefig('cost_plot.png')
   equation_latex: "C = (15 + 100 t) V"
   ```
3. **Drop the file** into `runtime/cidos/`.
4. **Run NIREON** as usual—`dynamic_math_runner` auto-detects the new CIDOS block, executes it, and emits a `MathResultSignal`.
5. **Check output**
   \* PNG plot in `runtime/math/artifacts/`
   \* Numeric result in the final report
   \* LaTeX equation rendered in HTML/PDF dashboards (if enabled).

---

## 3  Authoring CIDOS Math Blocks

### 3.1  Mandatory Fields

| Field            | Description                                 |
| ---------------- | ------------------------------------------- |
| `schema_version` | Always `cidos/1.0` for now                  |
| `eidos`          | Must be `math` to invoke math runner        |
| `id`             | Unique string, letters + `_` + digits       |
| `description`    | One-sentence summary                        |
| `function_name`  | Entry-point function defined in `code`      |
| `code`           | Valid Python 3; may define helper functions |

### 3.2  Optional Enhancements

* **`inputs`** - Key/value pairs auto-injected as kwargs.
* **`equation_latex`** - Display-ready formula for the report.
* **`requirements`** - Extra pip packages (`numpy`, `sympy`, …).
* **`limits`** - Safety caps (`timeout_sec`, `memory_mb`).

### 3.3  Best Practices

* Keep code idempotent; avoid global state.
* Save plots with **relative paths**—runner rewrites to artifact dir.
* Don’t import `os`, `subprocess`, or network libraries (blocked).
* Use `inputs` for parameters instead of hard-coding constants.

---

## 4  Running & Scheduling Tasks

### 4.1  On-Demand

Place CIDOS YAML into `runtime/cidos/`. The file watcher component publishes a `MathTaskCIDOSSignal` which triggers the runner.

### 4.2  Triggered from Ideas

Agents or reactor rules can generate CIDOS blocks on the fly and emit the same signal—ideal for “math-assist” enrichment.

---

## 5  Understanding the Results

A successful run emits `MathResultSignal` with payload:

```json
{
  "numeric_result": 75000.0,
  "artifacts": ["cost_plot.png"],
  "latex": "C = (15 + 100 t) V",
  "stdout": "",
  "execution_ms": 123
}
```

*Plots* are auto-embedded in HTML/PDF post-run reports.
*LaTeX* renders with MathJax.
*Numeric values* feed into trust or fitness scoring.

---

## 6  Troubleshooting

| Symptom                    | Likely Cause                 | Fix                                                  |
| -------------------------- | ---------------------------- | ---------------------------------------------------- |
| `MathErrorSignal: timeout` | Long-running loop            | Increase `limits.timeout_sec` (≤30) or optimise code |
| `ImportError: matplotlib`  | Missing dep                  | Add `requirements: ["matplotlib"]`                   |
| Plot file missing          | `plot_result()` never called | Ensure code saves PNG under same directory           |
| YAML validation error      | Missing required field       | Run `cidolib validate file.yaml`                     |

---

## 7  FAQ

**Q - Is Docker required?**  No, the default runner uses Python sub-process isolation with resource caps.

**Q - Can I import heavy libs like TensorFlow?**  Discouraged; resource caps will likely kill the job.

**Q - Where are artifacts stored?**  `runtime/math/artifacts/` with run-scoped UUID sub-folders.

---

## 8  Glossary

| Term                 | Definition                                      |
| -------------------- | ----------------------------------------------- |
| **CIDOS**            | YAML dialect for declarative intent             |
| **Runner**           | Component that validates & executes math blocks |
| **Artifact**         | Generated file (PNG, CSV, PDF) saved by runner  |
| **MathResultSignal** | Signal carrying numeric + artifact results      |

---

## 9  Appendix A - CIDOS Math Cheat-Sheet

```yaml
# Minimal block
schema_version: cidos/1.0
eidos: math
id: EXAMPLE_01
description: Quick multiply test
inputs:
  x: 7
  y: 6
function_name: multiply
code: |
  def multiply(x, y):
      return x * y
```

---

*End of User Manual v0.1*
