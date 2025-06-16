"""Reactor sub‑package.

The *reactor* orchestrates run‑time signal/response behaviour for NIREON.
Importing this package performs **no heavy work**: rules are discovered lazily
via :pymeth:`reactor.loader.RuleLoader.load_rules_from_directory`.

Public re‑exports are limited to keep the import surface minimal.
"""
from __future__ import annotations

__all__ = ["loader", "engine", "rules", "expressions"]

# Semantic version is set by the build/release pipeline.
__version__: str = "0.4.1"
