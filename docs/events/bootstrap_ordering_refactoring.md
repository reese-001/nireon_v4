# NIREON Bootstrap Phase Refactor – Implementation Guide (v1.1)

> **Goal:** Eliminate placeholder‑service warnings by pre‑loading the real shared
> services *before* any other bootstrap logic **while keeping “Abiogenesis” as
> the very first phase**.
>
> This revision swaps the order of the first two steps to avoid filename
> conflicts and clarifies that only the **bootstrap** directory and the manifest
> are touched.

---

## Table of Contents

- [NIREON Bootstrap Phase Refactor – Implementation Guide (v1.1)](#nireon-bootstrap-phase-refactor--implementation-guide-v11)
  - [Table of Contents](#tableofcontents)
  - [1  Prerequisites \& assumptions](#1prerequisites--assumptions)
  - [2  File‑tree snapshot **before** the change](#2filetree-snapshot-before-the-change)
  - [3  Step 1 – Rename existing `AbiogenesisPhase` to `ContextFormationPhase`](#3step1--rename-existing-abiogenesisphase-to-contextformationphase)
  - [4  Step 2 – Create the *new* `AbiogenesisPhase` (shared‑service preload)](#4step2--create-the-new-abiogenesisphase-sharedservice-preload)
  - [5  Step 3 – Register phase order](#5step3--register-phase-order)
  - [6  Step 4 – Mark services for preload in manifest](#6step4--mark-services-for-preload-in-manifest)
  - [7  Step 5 – Adjust imports \& logs](#7step5--adjust-imports--logs)
  - [8  Step 6 – Run \& verify](#8step6--run--verify)
  - [9  Completion criteria](#9completion-criteria)
  - [10  Rollback plan](#10rollback-plan)
  - [Appendix A – Source for new `AbiogenesisPhase`](#appendixa--source-for-new-abiogenesisphase)
  - [Appendix B – Minimal diff for renamed phase](#appendixb--minimal-diff-for-renamed-phase)

---

<a name="prereqs"></a>

## 1  Prerequisites & assumptions

* Working copy of NIREON v4.
* All code changes reside under **`bootstrap/`**. The only file outside is
  `configs/manifests/standard.yaml` where we add `preload: true` flags.

---

<a name="tree-before"></a>

## 2  File‑tree snapshot **before** the change

```
bootstrap/
 ├── core/main.py
 ├── phases/
 │   ├── abiogenesis_phase.py   (old logic – will be renamed)
 │   └── …
configs/
 └── manifests/standard.yaml
```

---

<a name="step1"></a>

## 3  Step 1 – Rename existing `AbiogenesisPhase` to `ContextFormationPhase`

1. **Move** the file:

   ```bash
   mv bootstrap/phases/abiogenesis_phase.py \
      bootstrap/phases/context_formation_phase.py
   ```
2. **Edit** the moved file:

   * Rename the class to `ContextFormationPhase`.
   * Change `PHASE_NAME = "ContextFormationPhase"`.
   * Keep its original `EXECUTION_ORDER` (e.g., `-4`).
3. **Commit**:

   ```bash
   git add bootstrap/phases/context_formation_phase.py
   git commit -m "refactor(bootstrap): rename AbiogenesisPhase→ContextFormationPhase"
   ```

---

<a name="step2"></a>

## 4  Step 2 – Create the *new* `AbiogenesisPhase` (shared‑service preload)

1. **Add** `bootstrap/phases/abiogenesis_phase.py` with this content:

```python
"""bootstrap.phases.abiogenesis_phase – pre‑loads essential shared services."""
from __future__ import annotations
import logging, importlib
from bootstrap.processors.component_processor import _instantiate_service
from bootstrap.health.reporter import ComponentStatus

logger = logging.getLogger(__name__)

class AbiogenesisPhase:
    PHASE_NAME = "AbiogenesisPhase"
    EXECUTION_ORDER = -10  # earliest

    def __init__(self, context, manifest_specs):
        self._ctx = context
        self._manifests = manifest_specs

    async def execute(self):
        reg = self._ctx.component_registry
        for manifest in self._manifests:
            for cid, spec in manifest.get("shared_services", {}).items():
                if not (spec.get("enabled", True) and spec.get("preload", False)):
                    continue
                if reg.contains(cid):
                    continue
                instance = await _instantiate_service(cid, spec, self._ctx)
                reg.register(cid, instance)
                # optional interface registration
                if port := spec.get("port_type"):
                    mod, attr = port.split(":") if ":" in port else port.rsplit(".", 1)
                    iface = getattr(importlib.import_module(mod), attr)
                    reg.register_service_instance(iface, instance)
                logger.info("[Abiogenesis] Pre‑loaded shared service '%s'", cid)
        if self._ctx.health_reporter:
            self._ctx.health_reporter.add_phase_status(
                self.PHASE_NAME, ComponentStatus.COMPLETED,
                "Shared‑services pre‑loaded"
            )
```

2. **Commit**:

```bash
git add bootstrap/phases/abiogenesis_phase.py
git commit -m "feat(bootstrap): new AbiogenesisPhase for service preload"
```

---

<a name="step3"></a>

## 5  Step 3 – Register phase order

Edit `bootstrap/core/main.py`:

```python
from bootstrap.phases.abiogenesis_phase import AbiogenesisPhase
from bootstrap.phases.context_formation_phase import ContextFormationPhase
...
PHASES = [
    AbiogenesisPhase,
    ContextFormationPhase,
    RegistrySetupPhase,
    FactorySetupPhase,
    ...
]
```

Commit.

---

<a name="step4"></a>

## 6  Step 4 – Mark services for preload in manifest

Add `preload: true` plus optional `port_type` to the four shared services in
`configs/manifests/standard.yaml`.

---

<a name="step5"></a>

## 7  Step 5 – Adjust imports & logs

Search for old import paths or phase‑name strings and update accordingly.

---

<a name="step6"></a>

## 8  Step 6 – Run & verify

1. `python -m bootstrap --diagnose configs/manifests/standard.yaml`
   *Expect:* lines `[Abiogenesis] Pre-loaded ...`, **no** `PLACEHOLDER` warnings.
2. Full DAG run (`python run_explorer_test.py`) passes without fallback messages.

---

<a name="complete"></a>

## 9  Completion criteria

✔ Bootstrap shows zero placeholder warnings.
✔ First phase log prefix is `[Abiogenesis]`.
✔ Registry lookup by interface returns the concrete instance.
✔ Full NIREON run completes cleanly.

---

<a name="rollback"></a>

## 10  Rollback plan

`git revert` the three commits or switch branches; remove `preload:` flags; restart bootstrap.

---

<a name="appendixa"></a>

## Appendix A – Source for new `AbiogenesisPhase`

*(identical to listing in Step 2)*

---

<a name="appendixb"></a>

## Appendix B – Minimal diff for renamed phase

```diff
-class AbiogenesisPhase:
-    PHASE_NAME = "AbiogenesisPhase"
+class ContextFormationPhase:
+    PHASE_NAME = "ContextFormationPhase"
```
