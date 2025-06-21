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

from configs.config_utils import ConfigMerger

__all__: Sequence[str] = ('ConfigLoader', 'DEFAULT_CONFIG')
logger = logging.getLogger(__name__)

_ENV_DEFAULT: Final[str] = 'default'

DEFAULT_CONFIG: Dict[str, Any] = {
    'env': _ENV_DEFAULT,
    'feature_flags': {},
    'storage': {
        'lineage_db_path': 'runtime/lineage.db',
        'ideas_path': 'runtime/ideas.json',
        'facts_path': 'runtime/world_facts.json',
    },
    'logging': {
        'prompt_response_logger': {
            'output_dir': 'runtime/llm_logs'
        },
        'pipeline_event_logger': {
            'output_dir': 'runtime/pipeline_event_logs'
        }
    }
}

# --- START OF FIX: ADDED REGEX AND EXPANSION FUNCTIONS ---
_RE_ENV_DEFAULT: Final[re.Pattern[str]] = re.compile('\\$\\{([A-Za-z_][A-Za-z0-9_]*)[:-](.*?)\\}')

def _interpolate_env(value: str) -> str:
    if not isinstance(value, str):
        return value

    def _repl(match: re.Match[str]) -> str:
        var, default = (match.group(1), match.group(2))
        return os.getenv(var, default)

    before = value
    value = _RE_ENV_DEFAULT.sub(_repl, value)
    value = os.path.expandvars(value)

    if before != value:
        logger.debug("Env expand: '%s' → '%s'", before, value)
    elif '${' in before and ':-' not in before: # Avoid warning for defaults
        logger.warning("Unresolved env var in '%s'", before)

    return value

def _expand_tree(node: Any) -> Any:
    if isinstance(node, dict):
        return {k: _expand_tree(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_expand_tree(v) for v in node]
    if isinstance(node, str):
        return _interpolate_env(node)
    return node
# --- END OF FIX ---


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        text = path.read_text(encoding='utf-8')
        if path.suffix.lower() in {'.json', '.json5'}:
            return json.loads(text) or {}
        
        data = yaml.safe_load(text) or {}
        if not isinstance(data, dict):
            logger.warning('%s does not contain a top‑level mapping – ignored', path)
            return {}
        return data

    except FileNotFoundError:
        logger.debug('Config file not found: %s', path)
        return {}
    except Exception as exc:
        logger.error('Failed to read %s: %s', path, exc, exc_info=True)
        return {}


class ConfigLoader:

    def __init__(self, package_root: Optional[Path]=None) -> None:
        self._package_root: Path = package_root if package_root is not None else Path(__file__).resolve().parents[1]

    async def load_global_config(self, env: Optional[str]=None, provided_config: Optional[Dict[str, Any]]=None) -> Dict[str, Any]:
        if provided_config is not None:
            logger.info('Using provided global configuration object.')
            return _expand_tree(provided_config)

        env = env or _ENV_DEFAULT
        logger.info('Loading global configuration for env=%s', env)
        cfg: Dict[str, Any] = copy.deepcopy(DEFAULT_CONFIG)
        cfg['env'] = env

        cfg = await self._load_global_app_configs(cfg, env)
        cfg = await self._load_llm_configs(cfg, env)
        
        # --- FIX: APPLY EXPANSION ---
        cfg = _expand_tree(cfg)
        
        cfg = await self._apply_enhancements(cfg, env)
        self._validate_required_config(cfg, env)
        
        logger.info("✓ Global configuration loaded for env='%s'", env)
        logger.debug('Resolved global config keys: %s', list(cfg))
        return cfg
        
    async def load_component_config(self, component_spec: Mapping[str, Any], component_id: str, global_config: Mapping[str, Any]) -> Dict[str, Any]:
        logger.debug('Loading component config for id=%s', component_id)
        cfg: Dict[str, Any] = {}

        base = await self._load_default_component_config(component_spec, component_id)
        cfg = ConfigMerger.merge(cfg, base, f'{component_id}_defaults')
        
        env_cfg = await self._load_env_component_config(component_spec, component_id, global_config.get('env', _ENV_DEFAULT))
        cfg = ConfigMerger.merge(cfg, env_cfg, f'{component_id}_env_override')
        
        inline_cfg = component_spec.get('inline_config', {})
        override_cfg = component_spec.get('config_override', {})
        
        if inline_cfg:
            cfg = ConfigMerger.merge(cfg, inline_cfg, f'{component_id}_inline')
        if override_cfg:
            cfg = ConfigMerger.merge(cfg, override_cfg, f'{component_id}_override')
            
        # --- FIX: APPLY EXPANSION ---
        cfg = _expand_tree(cfg)
        
        logger.debug('✓ Resolved component %s config keys: %s', component_id, list(cfg))
        return cfg

    async def _load_global_app_configs(self, cfg: Dict[str, Any], env: str) -> Dict[str, Any]:
        tasks = [
            ('DEFAULT_GLOBAL_APP_CONFIG', self._package_root / 'configs' / 'default' / 'global_app_config.yaml')
        ]
        if env != _ENV_DEFAULT:
            tasks.append(
                (f'ENV_GLOBAL_APP_CONFIG ({env})', self._package_root / 'configs' / env / 'global_app_config.yaml')
            )
        
        for label, path in tasks:
            data = _load_yaml(path)
            if data:
                cfg = ConfigMerger.merge(cfg, data, label)
                logger.info('Merged %s: %s', label, path)
            elif 'ENV_GLOBAL' in label:
                logger.warning('%s not found: %s', label, path)
        return cfg

    async def _load_llm_configs(self, cfg: Dict[str, Any], env: str) -> Dict[str, Any]:
        layers: list[tuple[str, Path]] = [
            ('DEFAULT_LLM_CONFIG', self._package_root / 'configs' / 'default' / 'llm_config.yaml')
        ]
        if env != _ENV_DEFAULT:
            layers.append(
                (f'ENV_LLM_CONFIG ({env})', self._package_root / 'configs' / env / 'llm_config.yaml')
            )
        
        llm_accum: Dict[str, Any] = {}
        for label, path in layers:
            content = _load_yaml(path)
            llm_part = content.get('llm', {}) if content else {}
            if not isinstance(llm_part, dict):
                logger.warning("%s 'llm' section is not a dict – skipped.", label)
                continue
            
            llm_accum = ConfigMerger.merge(llm_accum, llm_part, f'llm_layer: {label}')
            if llm_part:
                logger.info('Loaded LLM layer: %s (%s)', label, path)

        if llm_accum:
            cfg['llm'] = ConfigMerger.merge(cfg.get('llm', {}), llm_accum, 'merged_llm_config_layers')
        elif 'llm' not in cfg or not cfg['llm']:
            raise RuntimeError('No valid LLM configuration found in any source – aborting.')
        else:
            logger.info('Using LLM section from global_app_config.yaml unchanged.')
        
        return cfg

    async def _apply_enhancements(self, base_cfg: Dict[str, Any], env: str) -> Dict[str, Any]:
        cfg = dict(base_cfg)
        cfg.setdefault('bootstrap_strict_mode', True)
        cfg.setdefault('feature_flags', {})
        for flag, default_val in {
            'enable_schema_validation': True,
            'enable_self_certification': True,
        }.items():
            cfg['feature_flags'].setdefault(flag, default_val)
        await asyncio.sleep(0)
        return cfg

    async def _load_default_component_config(self, spec: Mapping[str, Any], component_id: str) -> Dict[str, Any]:
        return self._load_component_yaml(spec, component_id, env_override=None)

    async def _load_env_component_config(self, spec: Mapping[str, Any], component_id: str, env: str) -> Dict[str, Any]:
        if env == _ENV_DEFAULT:
            return {}
        return self._load_component_yaml(spec, component_id, env_override=env)

    def _load_component_yaml(self, spec: Mapping[str, Any], component_id: str, *, env_override: Optional[str]) -> Dict[str, Any]:
        template = spec.get('config')
        if not template:
            return {}
        
        path_str = template.replace('{id}', component_id)
        if env_override:
            path_str = path_str.replace('default', env_override)

        full_path = self._package_root / path_str
        data = _load_yaml(full_path)
        if data:
            logger.debug('Loaded component YAML: %s', full_path)
        return data

    def _validate_required_config(self, cfg: Mapping[str, Any], env: str) -> None:
        llm_cfg = cfg.get('llm')
        if not isinstance(llm_cfg, dict) or not llm_cfg:
            logger.warning('No LLM config present during validation – skipping.')
            return
        
        if 'models' not in llm_cfg:
            raise ValueError(f"llm.models missing for env='{env}'.  Present keys: {list(llm_cfg)}")

    def _get_config_path(self, filename: str, env: str) -> Optional[Path]:
        if env != _ENV_DEFAULT:
            env_path = self._package_root / 'configs' / env / filename
            if env_path.exists():
                return env_path
        
        default_path = self._package_root / 'configs' / 'default' / filename
        return default_path if default_path.exists() else None