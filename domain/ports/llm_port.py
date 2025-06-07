from __future__ import annotations

import abc
import logging
from typing import Any, Mapping, MutableMapping, Optional, Protocol, runtime_checkable

from domain.context import NireonExecutionContext

logger = logging.getLogger(__name__)

__all__ = [
    "LLMPort",
    "LLMResponse",
]


class LLMResponse(MutableMapping[str, Any]):
    """A minimally‑structured response returned by any :class:`LLMPort` implementation.

    Concrete adapters are free to extend this mapping with provider‑specific
    keys (e.g. *prompt_tokens*, *completion_tokens*, *model_latency_ms*).
    Mechanisms should rely only on the *text* field unless they have a
    provider‑agnostic reason to inspect the raw payload.
    """

    # Mandatory, provider‑agnostic keys – kept deliberately small.
    TEXT_KEY: str = "text"

    # ---------------------------------------------------------------------
    # The class behaves like a ``dict`` for convenience.  A full ``TypedDict``
    # is avoided here to keep run‑time overhead negligible and prevent cross‑
    # version typing headaches.  Adapter tests enforce required keys instead.
    # ---------------------------------------------------------------------

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: D401 – simple delegate
        self._data: dict[str, Any] = dict(*args, **kwargs)

    # Mapping protocol – pass‑through to *self._data* ---------------------
    def __getitem__(self, key: str) -> Any:  # noqa: D401 – mapping helper
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:  # noqa: D401 – mapping helper
        self._data[key] = value

    def __delitem__(self, key: str) -> None:  # noqa: D401 – mapping helper
        del self._data[key]

    def __iter__(self):  # noqa: D401 – mapping helper
        return iter(self._data)

    def __len__(self) -> int:  # noqa: D401 – mapping helper
        return len(self._data)

    # Convenience helpers --------------------------------------------------
    @property
    def text(self) -> str:
        """Return the primary textual output of the model ("completion")."""

        return self._data.get(self.TEXT_KEY, "")

    def __repr__(self) -> str:  # noqa: D401 – developer aid
        return f"LLMResponse({self._data!r})"


@runtime_checkable
class LLMPort(Protocol):
    """Public contract for all LLM adapters.

    Implementations *must* be compatible with ``asyncio``.  The synchronous helper is
    provided for convenience but should delegate to :py:meth:`call_llm_async` internally.
    """

    # NOTE: Using string annotations avoids importing heavy domain modules.
    async def call_llm_async(
        self,
        prompt: str,
        *,
        stage: "EpistemicStage", # type: ignore
        role: str,
        context: "NireonExecutionContext",
        settings: Optional[Mapping[str, Any]] = None,
    ) -> LLMResponse:  # noqa: D401 – core protocol method
        """Asynchronously invoke the underlying LLM service.

        Parameters
        ----------
        prompt:
            Fully‑rendered prompt text.
        stage:
            The current :class:`~domain.epistemic.EpistemicStage` (e.g., GENERATE, CRITIQUE).
        role:
            Logical role of the caller (e.g., *Explorer*, *Catalyst*, *MirrorAgent*).
        context:
            Execution context carrying tracing / run‑id / shared metadata.
        settings:
            Optional, provider‑agnostic overrides such as *temperature*, *top_p*,
            *max_tokens*.  The adapter is responsible for mapping these to the
            provider‑specific schema.
        """
        ...  # pragma: no cover – Protocol stub

    def call_llm_sync(
        self,
        prompt: str,
        *,
        stage: "EpistemicStage", # type: ignore
        role: str,
        context: "NireonExecutionContext",
        settings: Optional[Mapping[str, Any]] = None,
    ) -> LLMResponse:  # noqa: D401 – convenience wrapper
        """Synchronously invoke the LLM.

        This helper *should* await :py:meth:`call_llm_async` internally, e.g. using
        :pyfunc:`asyncio.run` when not already in an event loop.
        """
        raise NotImplementedError
