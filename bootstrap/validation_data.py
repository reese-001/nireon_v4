# C:\Users\erees\Documents\development\nireon\bootstrap\validation_data.py
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from core.lifecycle import ComponentMetadata
logger = logging.getLogger(__name__)
@dataclass
class ComponentValidationData:
    component_id: str
    original_metadata: ComponentMetadata
    resolved_config: Dict[str, Any]
    manifest_spec: Dict[str, Any]
    def __post_init__(self):
        if not isinstance(self.original_metadata, ComponentMetadata):
            raise TypeError('original_metadata must be a ComponentMetadata instance')
        if not isinstance(self.resolved_config, dict):
            raise TypeError('resolved_config must be a dictionary')
        if not isinstance(self.manifest_spec, dict):
            raise TypeError('manifest_spec must be a dictionary')
class BootstrapValidationData:
    def __init__(self, global_config: Dict[str, Any], run_id: Optional[str] = None): # Added run_id parameter
        self.global_config = global_config or {}
        self.run_id = run_id # Store run_id
        self._component_data: Dict[str, ComponentValidationData] = {}
        self._validation_errors: Dict[str, List[str]] = {}
        self._strict_mode = self.global_config.get('bootstrap_strict_mode', True)
        log_message = f'BootstrapValidationData initialized (strict_mode={self._strict_mode}'
        if self.run_id:
            log_message += f', run_id={self.run_id}'
        log_message += ')'
        logger.info(log_message)
    def store_component_data(self, component_id: str, original_metadata: ComponentMetadata, resolved_config: Dict[str, Any], manifest_spec: Dict[str, Any]) -> None:
        try:
            validation_data = ComponentValidationData(component_id=component_id, original_metadata=original_metadata, resolved_config=resolved_config, manifest_spec=manifest_spec)
            if component_id in self._component_data:
                logger.warning(f"Overwriting validation data for component '{component_id}'")
            self._component_data[component_id] = validation_data
            logger.debug(f"Stored validation data for component '{component_id}'")
        except Exception as e:
            error_msg = f"Failed to store validation data for '{component_id}': {e}"
            logger.error(error_msg)
            if self._strict_mode:
                raise ValueError(error_msg) from e
    def get_validation_data_for_component(self, component_id: str) -> Optional[ComponentValidationData]:
        return self._component_data.get(component_id)
    def get_all_component_ids(self) -> List[str]:
        return list(self._component_data.keys())
    def has_component_data(self, component_id: str) -> bool:
        return component_id in self._component_data
    def add_validation_error(self, component_id: str, error: str) -> None:
        if component_id not in self._validation_errors:
            self._validation_errors[component_id] = []
        self._validation_errors[component_id].append(error)
        logger.warning(f"Validation error for '{component_id}': {error}")
    def get_validation_errors(self, component_id: str) -> List[str]:
        return self._validation_errors.get(component_id, [])
    def get_all_validation_errors(self) -> Dict[str, List[str]]:
        return dict(self._validation_errors)
    def has_validation_errors(self) -> bool:
        return bool(self._validation_errors)
    def get_component_count(self) -> int:
        return len(self._component_data)
    def get_validation_summary(self) -> Dict[str, Any]:
        total_components = len(self._component_data)
        components_with_errors = len(self._validation_errors)
        total_errors = sum((len(errors) for errors in self._validation_errors.values()))
        summary = {
            'total_components': total_components,
            'components_with_errors': components_with_errors,
            'total_validation_errors': total_errors,
            'success_rate': (total_components - components_with_errors) / total_components if total_components > 0 else 1.0,
            'strict_mode': self._strict_mode,
            'global_config_keys': list(self.global_config.keys())
        }
        if self.run_id:
            summary['run_id'] = self.run_id
        return summary
    def clear_validation_errors(self) -> None:
        self._validation_errors.clear()
        logger.debug('Cleared all validation errors')
    def __repr__(self) -> str:
        repr_str = f'BootstrapValidationData(components={len(self._component_data)}, errors={len(self._validation_errors)}, strict_mode={self._strict_mode}'
        if self.run_id:
            repr_str += f', run_id={self.run_id}'
        repr_str += ')'
        return repr_str