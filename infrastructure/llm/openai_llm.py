"""
OpenAI LLM Adapter
──────────────────
Thin wrapper around the `/chat/completions` endpoint that conforms to
`LLMPort`.  Sync & async calls share common helpers for payload construction,
logging, and error handling.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Mapping, Optional, Tuple

import httpx

from domain.ports.llm_port import LLMPort, LLMResponse
from domain.context import NireonExecutionContext
from domain.epistemic_stage import EpistemicStage

logger = logging.getLogger(__name__)


class OpenAILLMAdapter(LLMPort):
    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    def __init__(self, config: Dict[str, Any] | None = None, model_name: str | None = None) -> None:
        cfg = config or {}

        self.api_key: Optional[str] = os.getenv(cfg.get("api_key_env", "OPENAI_API_KEY") or "") or cfg.get("api_key")
        self.model: str = model_name or cfg.get("model", "gpt-4")
        self.base_url: str = cfg.get("base_url", "https://api.openai.com/v1")
        self.timeout: float | int = cfg.get("timeout", 30)
        self.max_retries: int = cfg.get("max_retries", 3)

        if not self.api_key:
            logger.warning("OpenAI API key not provided; adapter will return mock responses.")

        # Create one sync client for reuse (cheaper sockets); async calls use disposable clients
        self._sync_client: Optional[httpx.Client] = (
            httpx.Client(
                timeout=self.timeout,
                headers=_build_headers(self.api_key),
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
            )
            if self.api_key
            else None
        )

        self.call_count: int = 0
        logger.info("OpenAI LLM Adapter ready (model=%s)", self.model)

    # ------------------------------------------------------------------ #
    # LLMPort required methods
    # ------------------------------------------------------------------ #
    async def call_llm_async(
        self,
        prompt: str,
        *,
        stage: EpistemicStage,
        role: str,
        context: NireonExecutionContext,
        settings: Optional[Mapping[str, Any]] = None,
    ) -> LLMResponse:
        self.call_count += 1
        if not self.api_key:
            return self._mock_response(prompt, async_mode=True)

        payload = _build_payload(prompt, self.model, settings or {})
        url = f"{self.base_url}/chat/completions"

        try:
            async with httpx.AsyncClient(timeout=self.timeout, headers=_build_headers(self.api_key)) as client:
                raw = await _send_request_async(client, url, payload)
            return _build_llm_response(raw, self.model)
        except Exception as exc:
            return _error_response(exc, self.model, async_mode=True)

    def call_llm_sync(
        self,
        prompt: str,
        *,
        stage: EpistemicStage,
        role: str,
        context: NireonExecutionContext,
        settings: Optional[Mapping[str, Any]] = None,
    ) -> LLMResponse:
        self.call_count += 1
        if not self.api_key or not self._sync_client:
            return self._mock_response(prompt, async_mode=False)

        payload = _build_payload(prompt, self.model, settings or {})
        url = f"{self.base_url}/chat/completions"

        try:
            raw = _send_request_sync(self._sync_client, url, payload)
            return _build_llm_response(raw, self.model)
        except Exception as exc:
            return _error_response(exc, self.model, async_mode=False)

    # Convenience wrappers (kept for backward compatibility)
    def generate(self, prompt: str, **kwargs) -> str:
        mock_ctx = NireonExecutionContext(run_id="sync_generate")
        resp = self.call_llm_sync(
            prompt, stage=EpistemicStage.DEFAULT, role="default", context=mock_ctx, settings=kwargs
        )
        return resp.text

    async def generate_async(self, prompt: str, **kwargs) -> str:
        mock_ctx = NireonExecutionContext(run_id="async_generate")
        resp = await self.call_llm_async(
            prompt, stage=EpistemicStage.DEFAULT, role="default", context=mock_ctx, settings=kwargs
        )
        return resp.text

    # Metrics
    def get_stats(self) -> Dict[str, Any]:
        return {
            "call_count": self.call_count,
            "model": self.model,
            "has_api_key": bool(self.api_key),
            "base_url": self.base_url,
        }

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _mock_response(self, prompt: str, *, async_mode: bool) -> LLMResponse:
        mode = "async" if async_mode else "sync"
        logger.warning("Returning mock %s response; API key missing.", mode)
        return LLMResponse({LLMResponse.TEXT_KEY: f"Mock {mode} OpenAI response to: {prompt[:50]}…"})

# --------------------------------------------------------------------------- #
# Module‑level utility functions
# --------------------------------------------------------------------------- #
def _build_headers(api_key: str | None) -> Dict[str, str]:
    return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"} if api_key else {}


def _build_payload(prompt: str, model: str, settings: Mapping[str, Any]) -> Dict[str, Any]:
    defaults = {"temperature": 0.7, "max_tokens": 1024}
    cfg = {**defaults, **settings}
    sys_prompt = cfg.pop("system_prompt", "You are a helpful assistant.")

    return {
        "model": model,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt},
        ],
        **cfg,
    }


def _send_request_sync(client: httpx.Client, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    resp = client.post(url, json=payload)
    resp.raise_for_status()
    return resp.json()


async def _send_request_async(client: httpx.AsyncClient, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    resp = await client.post(url, json=payload)
    resp.raise_for_status()
    return resp.json()


def _extract_content(raw: Dict[str, Any]) -> str:
    return raw.get("choices", [{}])[0].get("message", {}).get("content", "")


def _build_llm_response(raw: Dict[str, Any], model: str) -> LLMResponse:
    return LLMResponse({LLMResponse.TEXT_KEY: _extract_content(raw), **raw, "model": model})


def _error_response(exc: Exception, model: str, *, async_mode: bool) -> LLMResponse:
    mode = "async" if async_mode else "sync"
    logger.error("OpenAI %s call failed: %s", mode, exc, exc_info=True)
    return LLMResponse(
        {LLMResponse.TEXT_KEY: f"Error calling OpenAI API {mode}: {exc}", "error": str(exc), "model": model}
    )
