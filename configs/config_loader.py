"""
Global & component configuration loader for NIREON (v4.1)

Key guarantees
==============
* **100 % backwards‑compatible API** – `ConfigLoader` and its async methods keep
  exactly the same signatures and behaviour.
* **Pure‐Python + zero new deps**  – still relies only on stdlib + PyYAML.
* **Stricter typing & logging**   – makes IDE autocomplete, type‑checking,
  and trace‑level debugging far more helpful.
* **Lazy IO & helpers**           – YAML files are opened only when needed and
  parsed with small, composable helpers to simplify future extensions.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Final, Mapping, Optional, Sequence

import yaml

# ──────────────────────────────────────────────
# Public re‑exports
# ──────────────────────────────────────────────
from configs.config_utils import ConfigMerger  # noqa: F401 (public dependency)

__all__: Sequence[str] = ("ConfigLoader", "DEFAULT_CONFIG")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants & helpers
# ---------------------------------------------------------------------------

_ENV_DEFAULT: Final[str] = "default"

DEFAULT_CONFIG: Dict[str, Any] = {
    "env": _ENV_DEFAULT,
    "feature_flags": {},
    "storage": {
        "lineage_db_path": "runtime/lineage.db",
        "ideas_path": "runtime/ideas.json",
        "facts_path": "runtime/world_facts.json",
    },
    "logging": {
        "prompt_response_logger": {"output_dir": "runtime/llm_logs"},
        "pipeline_event_logger": {"output_dir": "runtime/pipeline_event_logs"},
    },
}

# `${VAR:-default}` expansion pattern
_RE_ENV_DEFAULT: Final[re.Pattern[str]] = re.compile(
    r"\$\{([A-Za-z_][A-Za-z0-9_]*)[:-](.*?)\}"
)


def _interpolate_env(value: str) -> str:
    """Expand `${VAR:-default}` and `$VAR`/`${VAR}` placeholders in *value*."""
    if not isinstance(value, str):
        return value

    def _repl(match: re.Match[str]) -> str:
        var, default = match.group(1), match.group(2)
        return os.getenv(var, default)

    before = value
    value = _RE_ENV_DEFAULT.sub(_repl, value)
    value = os.path.expandvars(value)  # handles $VAR and ${VAR}

    if before != value:
        logger.debug("Env expand: '%s' → '%s'", before, value)
    elif "${" in before:
        logger.warning("Unresolved env var in '%s'", before)

    return value


def _expand_tree(node: Any) -> Any:
    """Recursively apply environment‑variable interpolation."""
    if isinstance(node, dict):
        return {k: _expand_tree(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_expand_tree(v) for v in node]
    if isinstance(node, str):
        return _interpolate_env(node)
    return node


def _load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML/JSON file safely, returning `{}` on empty/null."""
    try:
        text = path.read_text(encoding="utf-8")
        # Auto‑detect JSON to avoid YAML quirks in extremely large numbers, etc.
        if path.suffix.lower() in {".json", ".json5"}:
            return json.loads(text) or {}
        data = yaml.safe_load(text) or {}
        if not isinstance(data, dict):
            logger.warning("%s does not contain a top‑level mapping – ignored", path)
            return {}
        return data
    except FileNotFoundError:
        logger.debug("Config file not found: %s", path)
        return {}
    except Exception as exc:  # pragma: no cover – defensive
        logger.error("Failed to read %s: %s", path, exc, exc_info=True)
        return {}


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------


class ConfigLoader:
    """
    Resolves layered configuration for the whole system *and* individual
    components.  The resolution order is identical to the legacy v4.0 loader:

    1. Hard‑coded defaults (in‑module `DEFAULT_CONFIG`).
    2. `configs/default/global_app_config.yaml`
    3. `configs/<env>/global_app_config.yaml`
    4. `configs/default/llm_config.yaml`
    5. `configs/<env>/llm_config.yaml`
    6. Post‑processing / feature‑flag enhancement.
    """

    # Resolve project root once; makes unit‑testing easier (can be overridden).
    def __init__(self, package_root: Optional[Path] = None) -> None:
        self._package_root: Path = (
            package_root
            if package_root is not None
            else Path(__file__).resolve().parents[1]
        )

    # ------------------------------------------------------------------#
    # Public API                                                         #
    # ------------------------------------------------------------------#

    async def load_global_config(
        self,
        env: Optional[str] = None,
        provided_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Aggregate global configuration.

        * `env` – logical environment name (e.g. `"prod"`).  Falls back to
          `"default"`.
        * `provided_config` – if given, **short‑circuits** all file loading and
          is used as‑is (after env‑var expansion).
        """
        if provided_config is not None:
            logger.info("Using provided global configuration object.")
            return _expand_tree(provided_config)

        env = env or _ENV_DEFAULT
        logger.info("Loading global configuration for env=%s", env)

        # 1️⃣ start with a deep copy of the constant default
        cfg: Dict[str, Any] = copy.deepcopy(DEFAULT_CONFIG)
        cfg["env"] = env

        # 2️⃣ merge layered YAML files (global_app_config then llm_config)
        cfg = await self._load_global_app_configs(cfg, env)
        cfg = await self._load_llm_configs(cfg, env)

        # 3️⃣ expand environment variables everywhere
        cfg = _expand_tree(cfg)

        # 4️⃣ apply system‑level enhancements / feature‑flag defaults
        cfg = await self._apply_enhancements(cfg, env)

        # 5️⃣ final validation
        self._validate_required_config(cfg, env)

        logger.info("✓ Global configuration loaded for env='%s'", env)
        logger.debug("Resolved global config keys: %s", list(cfg))
        return cfg

    async def load_component_config(
        self,
        component_spec: Mapping[str, Any],
        component_id: str,
        global_config: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """
        Resolve the effective configuration dict for a *single* component.

        Resolution order (earlier layers can be overridden by later ones):

        1. Component default YAML (`{config_path}/{id}.yaml`).
        2. Environment‑specific override (`{config_path.replace('default', env')}`).
        3. Inline `component_spec["inline_config"]`.
        4. Explicit `component_spec["config_override"]`.
        """
        logger.debug("Loading component config for id=%s", component_id)
        cfg: Dict[str, Any] = {}

        # Layer 1
        base = await self._load_default_component_config(component_spec, component_id)
        cfg = ConfigMerger.merge(cfg, base, f"{component_id}_defaults")

        # Layer 2
        env_cfg = await self._load_env_component_config(
            component_spec, component_id, global_config.get("env", _ENV_DEFAULT)
        )
        cfg = ConfigMerger.merge(cfg, env_cfg, f"{component_id}_env_override")

        # Layer 3 & 4
        inline_cfg = component_spec.get("inline_config", {})
        override_cfg = component_spec.get("config_override", {})

        if inline_cfg:
            cfg = ConfigMerger.merge(cfg, inline_cfg, f"{component_id}_inline")
        if override_cfg:
            cfg = ConfigMerger.merge(cfg, override_cfg, f"{component_id}_override")

        # Expand environment vars at the very end
        cfg = _expand_tree(cfg)

        logger.debug("✓ Resolved component %s config keys: %s", component_id, list(cfg))
        return cfg

    # ------------------------------------------------------------------#
    # Internals – Global config helpers                                 #
    # ------------------------------------------------------------------#

    async def _load_global_app_configs(
        self, cfg: Dict[str, Any], env: str
    ) -> Dict[str, Any]:
        """Merge `global_app_config.yaml` (default + env) into *cfg*."""
        tasks = [
            ("DEFAULT_GLOBAL_APP_CONFIG", self._package_root / "configs" / "default" / "global_app_config.yaml"),  # noqa: E501
        ]
        if env != _ENV_DEFAULT:
            tasks.append(
                (
                    f"ENV_GLOBAL_APP_CONFIG ({env})",
                    self._package_root / "configs" / env / "global_app_config.yaml",
                )
            )

        for label, path in tasks:
            data = _load_yaml(path)
            if data:
                cfg = ConfigMerger.merge(cfg, data, label)
                logger.info("Merged %s: %s", label, path)
            elif "ENV_GLOBAL" in label:
                logger.warning("%s not found: %s", label, path)
        return cfg

    async def _load_llm_configs(self, cfg: Dict[str, Any], env: str) -> Dict[str, Any]:
        """
        Merge layered `llm_config.yaml` files into *cfg['llm']*.

        The merging happens *inside* the `'llm'` key and respects precedence:
        default → env‑specific → global_app `llm` section (already present in
        *cfg* when we enter).
        """
        layers: list[tuple[str, Path]] = [
            ("DEFAULT_LLM_CONFIG", self._package_root / "configs" / "default" / "llm_config.yaml"),  # noqa: E501
        ]
        if env != _ENV_DEFAULT:
            layers.append(
                (
                    f"ENV_LLM_CONFIG ({env})",
                    self._package_root / "configs" / env / "llm_config.yaml",
                )
            )

        llm_accum: Dict[str, Any] = {}
        for label, path in layers:
            content = _load_yaml(path)
            llm_part = content.get("llm", {}) if content else {}

            if not isinstance(llm_part, dict):
                logger.warning("%s 'llm' section is not a dict – skipped.", label)
                continue

            llm_accum = ConfigMerger.merge(llm_accum, llm_part, f"llm_layer: {label}")
            if llm_part:
                logger.info("Loaded LLM layer: %s (%s)", label, path)

        if llm_accum:
            cfg["llm"] = ConfigMerger.merge(
                cfg.get("llm", {}), llm_accum, "merged_llm_config_layers"
            )
        elif "llm" not in cfg or not cfg["llm"]:
            raise RuntimeError(
                "No valid LLM configuration found in any source – aborting."
            )
        else:
            logger.info("Using LLM section from global_app_config.yaml unchanged.")

        return cfg

    async def _apply_enhancements(
        self, base_cfg: Dict[str, Any], env: str
    ) -> Dict[str, Any]:
        """
        Inject system‑level defaults (feature flags, bootstrap strictness, …).
        """
        cfg = dict(base_cfg)  # shallow copy fine – we only touch top‑level keys

        cfg.setdefault("bootstrap_strict_mode", True)
        cfg.setdefault("feature_flags", {})

        for flag, default_val in {
            "enable_schema_validation": True,
            "enable_self_certification": True,
        }.items():
            cfg["feature_flags"].setdefault(flag, default_val)

        # Example of env‑specific tweak hook (kept async for future IO)
        await asyncio.sleep(0)  # makes function truly 'await'‑able
        return cfg

    # ------------------------------------------------------------------#
    # Internals – Component config helpers                              #
    # ------------------------------------------------------------------#

    async def _load_default_component_config(
        self, spec: Mapping[str, Any], component_id: str
    ) -> Dict[str, Any]:
        return self._load_component_yaml(spec, component_id, env_override=None)

    async def _load_env_component_config(
        self, spec: Mapping[str, Any], component_id: str, env: str
    ) -> Dict[str, Any]:
        if env == _ENV_DEFAULT:
            return {}
        return self._load_component_yaml(spec, component_id, env_override=env)

    # Shared helper -----------------------------------------------------

    def _load_component_yaml(
        self,
        spec: Mapping[str, Any],
        component_id: str,
        *,
        env_override: Optional[str],
    ) -> Dict[str, Any]:
        """Return YAML dict for a component, or `{}` if file not found/invalid."""
        template = spec.get("config")
        if not template:
            return {}

        path_str = template.replace("{id}", component_id)
        if env_override:
            path_str = path_str.replace("default", env_override)

        full_path = self._package_root / path_str
        data = _load_yaml(full_path)
        if data:
            logger.debug("Loaded component YAML: %s", full_path)
        return data

    # ------------------------------------------------------------------#
    # Validation & utilities                                            #
    # ------------------------------------------------------------------#

    def _validate_required_config(self, cfg: Mapping[str, Any], env: str) -> None:
        """
        Ensure critical sections exist.  Raises `ValueError` for fatal issues.
        """
        llm_cfg = cfg.get("llm")
        if not isinstance(llm_cfg, dict) or not llm_cfg:
            logger.warning("No LLM config present during validation – skipping.")
            return

        if "models" not in llm_cfg:
            raise ValueError(
                f"llm.models missing for env='{env}'.  "
                f"Present keys: {list(llm_cfg)}"
            )

    # Legacy helper retained for external callers ----------------------

    def _get_config_path(self, filename: str, env: str) -> Optional[Path]:
        """Legacy wrapper kept *unchanged* for backwards‑compatibility."""
        # First env‑specific, then default.
        if env != _ENV_DEFAULT:
            env_path = self._package_root / "configs" / env / filename
            if env_path.exists():
                return env_path

        default_path = self._package_root / "configs" / "default" / filename
        return default_path if default_path.exists() else None
