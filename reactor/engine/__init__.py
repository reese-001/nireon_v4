"""Reactor engines.

`MainReactorEngine` is the default asynchronous rule‑evaluation loop; alternate
engines can live beside it and be selected via configuration.

Example
-------
>>> from reactor.engine import MainReactorEngine
"""
from __future__ import annotations

from .main import MainReactorEngine  # re‑export

__all__ = ["MainReactorEngine"]
