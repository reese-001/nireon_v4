from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from domain.ports.event_bus_port import EventBusPort
from core.registry.component_registry import ComponentRegistry


@dataclass
class BootstrapConfig:
    """Configuration for bootstrap process."""
    config_paths: List[Path]
    existing_registry: Optional[ComponentRegistry] = None
    existing_event_bus: Optional[EventBusPort] = None
    manifest_style: str = 'auto'
    replay: bool = False
    env: Optional[str] = None
    global_app_config: Optional[Dict[str, Any]] = None
    initial_strict_mode_param: bool = True
    

    @property
    def effective_strict_mode(self) -> bool:
        """Get effective strict mode from config hierarchy."""
        if self.global_app_config and 'bootstrap_strict_mode' in self.global_app_config:
            return bool(self.global_app_config['bootstrap_strict_mode'])
        return self.initial_strict_mode_param

    @classmethod
    def from_params(cls, config_paths: List[str | Path], **kwargs) -> 'BootstrapConfig':
        """Create BootstrapConfig from parameter dict."""
        return cls(
            config_paths=[Path(p) for p in config_paths],
            existing_registry=kwargs.get('existing_registry'),
            existing_event_bus=kwargs.get('existing_event_bus'),
            manifest_style=kwargs.get('manifest_style', 'auto'),
            replay=kwargs.get('replay', False),
            env=kwargs.get('env'),
            global_app_config=kwargs.get('global_app_config'),
            initial_strict_mode_param=kwargs.get('strict_mode', True)
        )


