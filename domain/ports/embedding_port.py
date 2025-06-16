"""Slim, behaviour‑only contract for embedding providers."""
from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from domain.embeddings.vector import Vector


@runtime_checkable
class EmbeddingPort(Protocol):
    """Minimal interface: *encode* only.  Policy lives elsewhere."""

    
    def encode(self, text: str) -> Vector:  # noqa: D401
        """Return a *normalised* Vector embedding for *text*."""

    # Optional bulk helper – *may* raise NotImplementedError
    def encode_batch(self, texts: Sequence[str]) -> list[Vector]: ...