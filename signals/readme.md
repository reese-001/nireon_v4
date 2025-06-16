# NIREON Runtime Signals

This directory defines **runtime signals** used by the NIREON engine and its mechanisms.

---

## Why Are There Two Signal Modules?

NIREON uses two distinct signal directories to enforce a clear boundary between system initialization and active reasoning:

### 1. [`nireon_v4/bootstrap/signals/`](../bootstrap/signals/)
- **Purpose**: Contains **bootstrap-time signals** used during system startup.
- **Examples**: `COMPONENT_REGISTERED`, `FACTORY_SETUP_PHASE_COMPLETE`
- **Audience**: Developers, SREs, CI tools.
- **Rule**: Runtime components (e.g., Reactor, Mechanisms, Observers) **must not import** from this module.

### 2. [`nireon_v4/signals/`](.)
- **Purpose**: Houses **runtime (epistemic) signals** used during the actual operation of the system.
- **Examples**: `SeedSignal`, `IdeaGenerated`, `TrustAssessment`
- **Audience**: Reactor, Mechanisms, Observers.
- **Rule**: This module is **dependency-free** and serves as a core event contract across the runtime system.

---

## Architectural Principle

> Runtime logic must never depend on bootstrap logic.

This separation improves modularity, import hygiene, testability, and long-term maintainability.

---

## EpistemicSignal Base

All runtime signals inherit from `EpistemicSignal`, a typed, immutable message structure that includes:
- `signal_type`: e.g. `"IdeaGenerated"`
- `source_node_id`: the emitting component
- `payload`: event-specific data
- `signal_id`: unique UUID for the event
- `timestamp`: UTC datetime

See `base.py` and `core.py` in this directory for signal definitions.
