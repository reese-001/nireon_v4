"""Namespace package for custom Action subclasses.

The core system currently defines only two concrete Pydantic models
(:class:`~reactor.models.TriggerComponentAction`,
 :class:`~reactor.models.EmitSignalAction`), but placing them in *models.py*
keeps circular‑import risk low.  New action types can be added here without
modifying the engine.

Nothing is imported at package level to avoid side effects.
"""
from __future__ import annotations

__all__: list[str] = []  # populated dynamically by plugins
