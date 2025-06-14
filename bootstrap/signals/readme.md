# NIREON Bootstrap Signals

This directory defines **bootstrap-time signals** used during system initialization.

---

## Why Are Bootstrap Signals Separate?

NIREON enforces a strict separation between **bootstrap-time** and **runtime** signals to avoid coupling initialization logic with runtime reasoning. This is a foundational architectural principle.

---

## Role of This Module

This module emits signals related to the **birth and setup** of the system.

### Examples:
- `COMPONENT_REGISTERED`
- `FACTORY_SETUP_PHASE_COMPLETE`
- `CONFIG_VALIDATION_STARTED`

These signals help track the initialization process, assist with CI/CD introspection, and allow developers or tools to observe system readiness.

---

## Architectural Boundary

> Runtime components (e.g., Reactor, Mechanisms, Observers) **must not import** from this directory.

This constraint ensures:
- Clean separation of concerns  
- Strict import hygiene (see Documentation §2.12)  
- Minimal side effects during runtime  

---

## When to Use These Signals

Use signals from this module **only inside** the bootstrap flow:
- Component registration
- Factory instantiation
- Dependency validation
- Initial system configuration

---

## BootstrapSignalEmitter

All bootstrap signals are emitted using the `BootstrapSignalEmitter`, which is confined to the bootstrap lifecycle and is **not available to runtime components**.

---

## Related Module

For runtime reasoning signals, see [`nireon_v4/signals/`](../../signals/), which defines `EpistemicSignal` and its subclasses like `SeedSignal` and `IdeaGeneratedSignal`.
