import json
import logging
import os
import re
from string import Template
from typing import Any, Dict, Mapping, Optional

import httpx

# Optional dependency ---------------------------------------------------------
try:
    from jsonpath_ng import parse as jsonpath_parse  # type: ignore
    JSONPATH_AVAILABLE = True
except ImportError:  # pragma: no cover
    JSONPATH_AVAILABLE = False
    jsonpath_parse = None
# -----------------------------------------------------------------------------


from domain.ports.llm_port import LLMPort, LLMResponse
from domain.context import NireonExecutionContext
from domain.epistemic_stage import EpistemicStage

logger = logging.getLogger(__name__)


class GenericHttpLLM(LLMPort):
    """
    Generic HTTP‑based LLM adapter.

    * **Compatible** with existing callers (class name, ctor signature,
      `call_llm_sync`, `call_llm_async`, `get_stats` unchanged).
    * Eliminates duplicated code (shared helpers for request/response).
    * Pre‑compiles the payload template & JSONPath for efficiency.
    * Reduces log noise; keeps critical diagnostics.
    """

    # --------------------------------------------------------------------- #
    # Construction helpers
    # --------------------------------------------------------------------- #
    _BRACE_RE = re.compile(r"\{\{\s*(\w+)\s*\}\}")

    def __init__(self, config: Dict[str, Any] | None = None, model_name: str | None = None, **kwargs):
        self.config: Dict[str, Any] = {**(config or {}), **kwargs}

        # Internal vs. provider model naming
        self.internal_model_name = model_name or "default"
        self.model_name = self.config.get("model_name_for_api", self.internal_model_name)

        # HTTP / auth
        self.method: str = self.config.get("method", "POST").upper()
        self.base_url: str = self.config.get("base_url", "").rstrip("/")
        self.endpoint: str = self.config.get("endpoint", "")
        self.timeout: float | int = self.config.get("timeout", 30)

        self.auth_style: str = self.config.get("auth_style", "bearer")  # bearer | header_key | query_param | none
        self.auth_token_env: str = self.config.get("auth_token_env", "")
        self.auth_header_name: str = self.config.get("auth_header_name", "Authorization")
        self.auth_token: str | None = (
            None if self.auth_style == "none" else os.getenv(self.auth_token_env) or None
        )
        if self.auth_style != "none" and not self.auth_token:
            logger.warning("Auth token not found in env var '%s'", self.auth_token_env)

        # Template & response extraction
        raw_template = self.config.get("payload_template", "{}")
        compiled_template_str = self._BRACE_RE.sub(lambda m: f"${m.group(1)}", raw_template)
        self.payload_template = Template(compiled_template_str)

        self.response_text_path: str = self.config.get("response_text_path", "$.text")
        self.response_parser = None
        if JSONPATH_AVAILABLE:
            try:
                self.response_parser = jsonpath_parse(self.response_text_path)
            except Exception as e:  # pragma: no cover
                logger.error("Invalid JSONPath '%s': %s", self.response_text_path, e)

        self.call_count: int = 0
        logger.info(
            "GenericHttpLLM ready — internal='%s' provider='%s' %s%s",
            self.internal_model_name,
            self.model_name,
            self.base_url,
            self.endpoint,
        )

    # --------------------------------------------------------------------- #
    # Public LLMPort interface
    # --------------------------------------------------------------------- #
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
        if self._auth_required_but_missing():
            return self._mock_response(prompt, sync=False)

        url, headers, payload = self._prepare_request(prompt, stage, role, context, settings)
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await self._http_send_async(client, url, headers, payload)
            return self._build_llm_response(resp)
        except Exception as e:
            logger.exception("Async call #%s failed: %s", self.call_count, e)
            return self._error_response(str(e))

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
        if self._auth_required_but_missing():
            return self._mock_response(prompt, sync=True)

        url, headers, payload = self._prepare_request(prompt, stage, role, context, settings)
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = self._http_send_sync(client, url, headers, payload)
            return self._build_llm_response(resp)
        except Exception as e:
            logger.exception("Sync call #%s failed: %s", self.call_count, e)
            return self._error_response(str(e))

    # Metrics ---------------------------------------------------------------
    def get_stats(self) -> Dict[str, Any]:
        return {
            "call_count": self.call_count,
            "model": self.model_name,
            "base_url": self.base_url,
            "endpoint": self.endpoint,
            "has_auth_token": bool(self.auth_token),
            "auth_style": self.auth_style,
        }

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #
    def _auth_required_but_missing(self) -> bool:
        if self.auth_style == "none":
            return False
        missing = self.auth_token is None
        if missing:
            logger.warning("Auth required but token missing; returning mock response.")
        return missing

    def _mock_response(self, prompt: str, *, sync: bool) -> LLMResponse:
        mode = "sync" if sync else "async"
        return LLMResponse(
            {
                LLMResponse.TEXT_KEY: f"Mock {mode} response to: {prompt[:50]}…",
                "error": "NoAuthToken",
            }
        )

    # Request / response ----------------------------------------------------
    def _prepare_request(
        self,
        prompt: str,
        stage: EpistemicStage,
        role: str,
        context: NireonExecutionContext,
        settings_raw: Optional[Mapping[str, Any]],
    ) -> tuple[str, Dict[str, str], Dict[str, Any]]:
        settings = dict(settings_raw or {})
        headers = self._build_headers()

        try:
            payload = self._build_payload(prompt, stage, role, settings)
        except Exception as e:  # pragma: no cover
            logger.exception("Payload build failed; using fallback: %s", e)
            payload = self._fallback_payload(prompt, settings)

        url = f"{self.base_url}{self.endpoint}"
        return url, headers, payload

    def _build_headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if self.auth_style == "none":
            return headers
        if self.auth_style == "bearer" and self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        elif self.auth_style == "header_key" and self.auth_token:
            headers[self.auth_header_name] = self.auth_token
        return headers

    def _build_payload(
        self,
        prompt: str,
        stage: EpistemicStage,
        role: str,
        settings: Mapping[str, Any],
    ) -> Dict[str, Any]:
        json_safe_prompt = json.dumps(prompt)[1:-1]  # strip quotes
        json_safe_system = json.dumps(settings.get("system_prompt", "You are a helpful assistant."))[1:-1]

        vars_: Dict[str, Any] = {
            "prompt": json_safe_prompt,
            "role": role,
            "stage": stage.value if isinstance(stage, EpistemicStage) else str(stage),
            "model_name_for_api": self.model_name,
            "system_prompt": json_safe_system,
            "temperature": settings.get("temperature", 0.7),
            "max_tokens": settings.get("max_tokens", 1024),
            "top_p": settings.get("top_p", 1.0),
            **settings,
        }

        try:
            rendered = self.payload_template.safe_substitute(**vars_)
            return json.loads(rendered)
        except Exception as e:  # pragma: no cover
            logger.error("Template render error: %s", e)
            logger.debug("Template: %s", self.payload_template.template)
            logger.debug("Vars: %s", vars_)
            raise

    @staticmethod
    def _fallback_payload(prompt: str, settings: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            "model": settings.get("model_name_for_api"),
            "messages": [
                {"role": "system", "content": settings.get("system_prompt", "You are a helpful assistant.")},
                {"role": "user", "content": prompt},
            ],
            "temperature": settings.get("temperature", 0.7),
            "max_tokens": settings.get("max_tokens", 1024),
        }

    # HTTP send -------------------------------------------------------------
    def _http_send_sync(
        self, client: httpx.Client, url: str, headers: Dict[str, str], payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        resp = (
            client.post(url, json=payload, headers=headers)
            if self.method == "POST"
            else client.get(url, params=payload, headers=headers)
        )
        resp.raise_for_status()
        return resp.json()

    async def _http_send_async(
        self, client: httpx.AsyncClient, url: str, headers: Dict[str, str], payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        resp = (
            await client.post(url, json=payload, headers=headers)
            if self.method == "POST"
            else await client.get(url, params=payload, headers=headers)
        )
        resp.raise_for_status()
        return resp.json()

    # Response handling -----------------------------------------------------
    def _build_llm_response(self, raw: Dict[str, Any]) -> LLMResponse:
        content = self._extract_response_text(raw)
        return LLMResponse(
            {
                LLMResponse.TEXT_KEY: content,
                "raw_response": raw,
                "model": self.model_name,
                "provider": "generic_http",
            }
        )

    def _error_response(self, msg: str) -> LLMResponse:
        return LLMResponse({LLMResponse.TEXT_KEY: f"Error: {msg}", "error": msg, "model": self.model_name})

    # --------------------------------------------------------------------- #
    # Response text extraction
    # --------------------------------------------------------------------- #
    def _extract_response_text(self, data: Dict[str, Any]) -> str:  # noqa: C901
        if JSONPATH_AVAILABLE and self.response_parser:
            try:
                matches = self.response_parser.find(data)
                if matches:
                    return str(matches[0].value)
            except Exception as e:  # pragma: no cover
                logger.error("JSONPath extract error: %s", e)

        # Fallbacks -----------------------------------------------------
        if isinstance(data, dict):
            if (choice := data.get("choices")):  # OpenAI style
                first = choice[0]
                return (first.get("message", {}) or {}).get("content") or first.get("text", "")
            if (cand := data.get("candidates")):  # Gemini style
                parts = cand[0].get("content", {}).get("parts", [])
                if parts and isinstance(parts[0], dict):
                    return parts[0].get("text", "")
            return data.get("text") or data.get("content") or json.dumps(data)
        return str(data)
