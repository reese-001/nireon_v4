
1.  **Consolidate Helper Functions (Issue A & C):**
    *   A new module `nireon_v4/bootstrap/utils/component_utils.py` will be created.
    *   Canonical versions of `get_pydantic_defaults`, `create_component_instance`, `inject_dependencies`, `validate_component_interfaces`, `configure_component_logging`, and `prepare_component_metadata` will be moved there and made public (leading underscore removed from their names).
    *   The `create_component_instance` function from `nireon_v4/bootstrap/processors/enhanced_components.py` (which handles `gateway_params` and `common_deps: Dict`) will be used as the canonical version, as per the feedback.
    *   The `get_pydantic_defaults` function from `nireon_v4/bootstrap/bootstrap_helper/enhanced_components.py` (the more detailed one) will be the canonical one.
    *   Other helper functions (`inject_dependencies`, etc.) will be taken from `nireon_v4/bootstrap/bootstrap_helper/enhanced_components.py`.
    *   Original files (`bootstrap_helper/enhanced_components.py` and `processors/enhanced_components.py`) will be updated to import these functions from the new utility module.
    *   The `init_full_component` function in `bootstrap_helper/enhanced_components.py` will be adapted to pass `common_deps` as a dictionary to the new `create_component_instance`.

2.  **Remove `sys.modules` Monkey-Patching (Issue B):**
    *   The `sys.modules.setdefault('phases', ...)` block and the related loop in `nireon_v4/bootstrap/__init__.py` will be removed.
    *   All imports like `from phases...` or `import phases...` will be changed to fully qualified imports, e.g., `from bootstrap.phases...`. This primarily affects `nireon_v4/bootstrap/core/main.py`.
    *   `from __future__ import absolute_import` will be added to the modified files for clarity, although it's standard in Python 3.

3.  **Minor fix in `__main__.py`:**
    *   The `--version` command will use `from . import __version__` instead of a non-existent `get_version()` function.

Let's proceed with the replacement code files.

--- START OF FILE nireon_v4/bootstrap/utils/__init__.py ---
```python
# nireon_v4/bootstrap/utils/__init__.py
from __future__ import absolute_import

# This file can remain empty or be used to re-export utilities
# from submodules if desired. For now, it marks 'utils' as a package.
```
--- END OF FILE nireon_v4/bootstrap/utils/__init__.py ---

--- START OF FILE nireon_v4/bootstrap/utils/component_utils.py ---
```python
# nireon_v4/bootstrap/utils/component_utils.py
from __future__ import annotations
import dataclasses
import importlib
import inspect
import logging
from typing import Any, Dict, List, Optional, Type, Union, cast

from pydantic import BaseModel

from core.base_component import NireonBaseComponent
from core.lifecycle import ComponentMetadata
# Assuming CommonMechanismDependencies is defined elsewhere and can be imported if needed by callers
# For create_component_instance, common_deps is now Dict[str, Any]
# from factories.dependencies import CommonMechanismDependencies

logger = logging.getLogger(__name__)

__all__ = [
    'get_pydantic_defaults',
    'create_component_instance',
    'inject_dependencies',
    'validate_component_interfaces',
    'configure_component_logging',
    'prepare_component_metadata',
]


def get_pydantic_defaults(component_class: Type, component_name: str) -> Dict[str, Any]:
    """
    Extracts default configuration values from a component's Pydantic ConfigModel
    or a legacy DEFAULT_CONFIG attribute.
    """
    defaults = {}
    try:
        if hasattr(component_class, 'ConfigModel'):
            config_model = component_class.ConfigModel
            if hasattr(config_model, 'model_fields'):  # Pydantic V2
                for field_name, field_info in config_model.model_fields.items():
                    if hasattr(field_info, 'default') and field_info.default is not ...:
                        defaults[field_name] = field_info.default
                    elif hasattr(field_info, 'default_factory') and field_info.default_factory is not None:
                        try:
                            defaults[field_name] = field_info.default_factory()
                        except Exception:
                            logger.debug(f"Error calling default_factory for {component_name}.{field_name}")
                            pass # Keep default as not set
            elif hasattr(config_model, '__fields__'):  # Pydantic V1
                for field_name, field in config_model.__fields__.items():
                    if hasattr(field, 'default') and field.default is not ...: # Pydantic V1
                        defaults[field_name] = field.default
                    elif hasattr(field, 'default_factory') and field.default_factory is not None:
                        try:
                            defaults[field_name] = field.default_factory()
                        except Exception:
                            logger.debug(f"Error calling default_factory for {component_name}.{field_name} (V1)")
                            pass
        # Fallback or supplement with DEFAULT_CONFIG for components not fully on Pydantic
        if hasattr(component_class, 'DEFAULT_CONFIG'):
            if isinstance(component_class.DEFAULT_CONFIG, dict):
                # Pydantic defaults should take precedence
                merged_defaults = component_class.DEFAULT_CONFIG.copy()
                merged_defaults.update(defaults)
                defaults = merged_defaults
            else:
                logger.warning(
                    f"Component {component_name} has a DEFAULT_CONFIG attribute that is not a dict. Skipping."
                )

        if defaults:
            logger.debug(f'Found {len(defaults)} Pydantic/DEFAULT_CONFIG defaults for {component_name}')
        else:
            logger.debug(f'No Pydantic/DEFAULT_CONFIG defaults found for {component_name}')
    except Exception as e:
        logger.debug(f'Error extracting Pydantic/DEFAULT_CONFIG defaults for {component_name}: {e}')
    return defaults


async def create_component_instance(
    component_class: Type[NireonBaseComponent],
    resolved_config_for_instance: Dict[str, Any],
    instance_id: str,
    instance_metadata_object: ComponentMetadata,
    common_deps: Optional[Dict[str, Any]] = None
) -> NireonBaseComponent:
    """
    Creates an instance of a NireonBaseComponent.
    This is the canonical version, adapted from processors/enhanced_components.py
    to handle gateway_params and common_deps as a dictionary.
    """
    try:
        ctor_signature = inspect.signature(component_class.__init__)
        ctor_params = ctor_signature.parameters
        kwargs: Dict[str, Any] = {}

        if 'config' in ctor_params:
            kwargs['config'] = resolved_config_for_instance
        elif 'cfg' in ctor_params: # Support for alternative config naming
            kwargs['cfg'] = resolved_config_for_instance

        if 'metadata_definition' in ctor_params:
            kwargs['metadata_definition'] = instance_metadata_object
        elif 'metadata' in ctor_params: # Support for alternative metadata naming
            kwargs['metadata'] = instance_metadata_object
        
        # Inject common dependencies if provided and expected by constructor
        if common_deps:
            for dep_name, dep_value in common_deps.items():
                if dep_name in ctor_params:
                    kwargs[dep_name] = dep_value

        # Special handling for MechanismGateway, ensuring specific params are present
        # This part was highlighted in the feedback as important.
        if component_class.__name__ == 'MechanismGateway':
            gateway_params = ['llm_router', 'parameter_service', 'frame_factory', 'budget_manager', 'event_bus']
            for param_name in gateway_params:
                if param_name in ctor_params and param_name not in kwargs:
                    # If common_deps provided this param, it would be in kwargs.
                    # If not, and gateway expects it, pass None.
                    # This assumes MechanismGateway's __init__ can handle None for these.
                    kwargs[param_name] = None
                    logger.debug(f"Ensuring '{param_name}' is in kwargs for MechanismGateway, set to None as not in common_deps.")


        logger.debug(f'Attempting to create component {instance_id} ({component_class.__name__}) with kwargs: {list(kwargs.keys())}')
        instance = component_class(**kwargs)

        # Ensure component_id and metadata.id are correctly set on the instance
        # setattr is used to bypass potential property setters if they exist
        if hasattr(instance, 'component_id'):
            if instance.component_id != instance_id:
                logger.debug(f"Updating component_id on instance from '{instance.component_id}' to '{instance_id}'")
                object.__setattr__(instance, 'component_id', instance_id) # Direct set
        elif isinstance(instance, NireonBaseComponent): # If it's a NireonBaseComponent, it should have _component_id
             object.__setattr__(instance, '_component_id', instance_id)


        if hasattr(instance, 'metadata') and isinstance(instance.metadata, ComponentMetadata):
            if instance.metadata.id != instance_id:
                logger.debug(f"Updating metadata.id on instance from '{instance.metadata.id}' to '{instance_id}'")
                # Create a new metadata object with the correct ID if it's a dataclass/immutable
                try:
                    updated_meta = dataclasses.replace(instance.metadata, id=instance_id)
                    object.__setattr__(instance, '_metadata_definition', updated_meta) # Assuming it stores metadata in _metadata_definition
                except TypeError: # Not a dataclass or not replaceable
                     instance.metadata.id = instance_id # Try direct assignment
        elif isinstance(instance, NireonBaseComponent): # If it's a NireonBaseComponent, it should have _metadata_definition
            object.__setattr__(instance, '_metadata_definition', instance_metadata_object)


        logger.info(f'Successfully created component instance: {instance_id} ({component_class.__name__})')
        return cast(NireonBaseComponent, instance)
    except Exception as e:
        logger.error(f'Failed to create component instance {instance_id} ({component_class.__name__}): {e}', exc_info=True)
        # Re-raise as a BootstrapError or a more specific error if appropriate
        from bootstrap.exceptions import ComponentInstantiationError # Local import to avoid circularity if utils is imported early
        raise ComponentInstantiationError(f"Instantiation failed for '{instance_id}': {e}", component_id=instance_id) from e


def inject_dependencies(instance: NireonBaseComponent, dependency_map: Dict[str, Any], registry: Optional[Any]=None) -> None:
    """
    Injects dependencies into a component instance.
    Canonical version from bootstrap_helper/enhanced_components.py.
    """
    try:
        for dep_name, dep_value in dependency_map.items():
            # Prefer private attributes first (e.g., _event_bus)
            private_attr_name = f'_{dep_name}'
            if hasattr(instance, private_attr_name):
                setattr(instance, private_attr_name, dep_value)
                logger.debug(f"Injected dependency '{dep_name}' into {instance.component_id} (as {private_attr_name})")
            elif hasattr(instance, dep_name):
                setattr(instance, dep_name, dep_value)
                logger.debug(f"Injected dependency '{dep_name}' into {instance.component_id}")
            else:
                logger.warning(f"Component {instance.component_id} does not have an attribute '{dep_name}' or '{private_attr_name}' for dependency injection.")
    except Exception as e:
        logger.warning(f'Failed to inject dependencies into {instance.component_id}: {e}')


def validate_component_interfaces(instance: NireonBaseComponent, expected_interfaces: List[Type]) -> List[str]:
    """
    Validates if a component instance implements expected interfaces.
    Canonical version from bootstrap_helper/enhanced_components.py.
    """
    errors: List[str] = []
    try:
        for interface_protocol in expected_interfaces:
            if not isinstance(instance, interface_protocol):
                errors.append(f'Component {instance.component_id} does not implement {interface_protocol.__name__}')
    except Exception as e:
        # This catch-all is broad; specific errors might be better if known
        errors.append(f'Interface validation failed for {instance.component_id}: {e}')
    return errors


def configure_component_logging(instance: NireonBaseComponent, log_level: Optional[str]=None, log_prefix: Optional[str]=None) -> None:
    """
    Configures logging for a component instance.
    Canonical version from bootstrap_helper/enhanced_components.py.
    """
    try:
        # If the component has a specific method for configuring its logging
        if hasattr(instance, '_configure_logging') and callable(instance._configure_logging):
            instance._configure_logging(log_level=log_level, log_prefix=log_prefix)
            logger.debug(f"Called _configure_logging for {instance.component_id}")
        # Else, if it has a standard 'logger' attribute, try to set its level
        elif hasattr(instance, 'logger') and isinstance(instance.logger, logging.Logger):
            if log_level:
                level_to_set = getattr(logging, log_level.upper(), None)
                if level_to_set:
                    instance.logger.setLevel(level_to_set)
                    logger.debug(f"Set log level for {instance.component_id} logger to {log_level.upper()}")
                else:
                    logger.warning(f"Invalid log level '{log_level}' for {instance.component_id}")
            # log_prefix might be harder to apply generically if no _configure_logging method
            if log_prefix and not (hasattr(instance, '_configure_logging') and callable(instance._configure_logging)):
                 logger.debug(f"Log prefix '{log_prefix}' provided for {instance.component_id}, but no generic way to apply it without _configure_logging method.")

    except Exception as e:
        logger.warning(f'Failed to configure logging for {instance.component_id}: {e}')


def prepare_component_metadata(
    base_metadata: ComponentMetadata,
    instance_id: str,
    config_overrides: Optional[Dict[str, Any]] = None
) -> ComponentMetadata:
    """
    Prepares the final ComponentMetadata for an instance, applying overrides.
    Canonical version from bootstrap_helper/enhanced_components.py.
    """
    # Ensure base_metadata is indeed a ComponentMetadata instance
    if not isinstance(base_metadata, ComponentMetadata):
        logger.error(f"Base metadata for {instance_id} is not a ComponentMetadata instance. Type: {type(base_metadata)}")
        # Depending on strictness, either raise error or return a default
        # For now, let's try to proceed if it's dict-like, or create a default
        if isinstance(base_metadata, dict) and 'id' in base_metadata and 'name' in base_metadata: # crude check
             base_metadata = ComponentMetadata(**base_metadata)
        else:
            base_metadata = ComponentMetadata(id=instance_id, name=instance_id, version="0.0.0", category="unknown")


    # Start with a copy of the base_metadata, replacing the id
    instance_metadata = dataclasses.replace(base_metadata, id=instance_id)

    if config_overrides:
        # Assuming config_overrides might contain a specific 'metadata_override' section
        # as seen in some parts of the original code.
        metadata_overrides_from_config = config_overrides.get('metadata_override', {})
        if metadata_overrides_from_config:
            logger.debug(f"Applying metadata_override from config for {instance_id}: {metadata_overrides_from_config}")
            try:
                # Convert current instance_metadata to dict to update, then recreate
                current_meta_dict = dataclasses.asdict(instance_metadata)
                current_meta_dict.update(metadata_overrides_from_config)
                instance_metadata = ComponentMetadata(**current_meta_dict)
            except Exception as e:
                logger.error(f"Error applying metadata_override for {instance_id}: {e}. Using metadata before override attempt.")

    return instance_metadata
```
--- END OF FILE nireon_v4/bootstrap/utils/component_utils.py ---

--- START OF FILE nireon_v4/bootstrap/__init__.py ---
```python
# nireon_v4/bootstrap/__init__.py
from __future__ import annotations
from __future__ import absolute_import

# Removed sys.modules monkey-patching for 'phases'
# Imports will now use fully qualified paths like 'bootstrap.phases.some_phase'

from .exceptions import *
from .core.main import BootstrapOrchestrator, bootstrap_nireon_system, bootstrap, bootstrap_sync
from .core.phase_executor import BootstrapPhaseExecutor, PhaseExecutionResult, PhaseExecutionSummary, execute_bootstrap_phases
from .context.bootstrap_context_builder import BootstrapContextBuilder, create_bootstrap_context
from .context.bootstrap_context import BootstrapContext
from .config.bootstrap_config import BootstrapConfig
from .result_builder import BootstrapResult, BootstrapResultBuilder, build_result_from_context, create_minimal_result
from .validation_data import BootstrapValidationData, ComponentValidationData
from .health.reporter import HealthReporter, ComponentStatus, ComponentHealthRecord
from configs.config_loader import ConfigLoader # Assuming this is correctly located

__version__ = '4.0.0'
__author__ = 'NIREON V4 Bootstrap Team'
__description__ = 'L0 Abiogenesis – Bootstrap Infrastructure'
CURRENT_SCHEMA_VERSION = 'V4-alpha.1.0'

__all__ = [
    'bootstrap_nireon_system', 'bootstrap', 'bootstrap_sync',
    'BootstrapConfig',
    'BootstrapContext', 'BootstrapContextBuilder', 'create_bootstrap_context',
    'BootstrapOrchestrator',
    'BootstrapPhaseExecutor', 'PhaseExecutionResult', 'PhaseExecutionSummary', 'execute_bootstrap_phases',
    'BootstrapResult', 'BootstrapResultBuilder', 'build_result_from_context', 'create_minimal_result',
    'BootstrapValidationData', 'ComponentValidationData',
    'HealthReporter', 'ComponentStatus', 'ComponentHealthRecord',
    'ConfigLoader',
    'CURRENT_SCHEMA_VERSION', '__version__', '__author__', '__description__',
    # Add exception classes if they are directly exposed and part of the public API
    # For example, if BootstrapError is meant to be caught by users:
    'BootstrapError', 'ComponentInstantiationError', 'ComponentInitializationError',
    'ComponentValidationError', 'ManifestProcessingError', 'ConfigurationError',
    'BootstrapTimeoutError', 'BootstrapValidationError', 'BootstrapContextBuildError',
    'DependencyResolutionError', 'FactoryError', 'StepCommandError', 'RegistryError',
    'RBACError', 'HealthReportingError', 'PhaseExecutionError'
]
```
--- END OF FILE nireon_v4/bootstrap/__init__.py ---

--- START OF FILE nireon_v4/bootstrap/__main__.py ---
```python
# nireon_v4/bootstrap/__main__.py
import asyncio
import sys
import logging
from pathlib import Path
from __future__ import absolute_import

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def main_cli_entry():
    if len(sys.argv) < 2:
        print('Usage: python -m bootstrap <manifest_file1.yaml> [<manifest_file2.yaml> ...]')
        print('       python -m bootstrap --smoke-test')
        print('       python -m bootstrap --validate <manifest_file.yaml>')
        print('       python -m bootstrap --version')
        sys.exit(1)

    if sys.argv[1] == '--version':
        from . import __version__ # Use __version__ directly
        print(f'NIREON V4 Bootstrap {__version__}')
        sys.exit(0)

    if sys.argv[1] == '--smoke-test':
        try:
            from .core.main import smoke_test
            success = await smoke_test()
            print('✓ Smoke test passed' if success else '✗ Smoke test failed')
            sys.exit(0 if success else 1)
        except ImportError:
            # Check if smoke_test exists, if not, it means it's not implemented or there's an issue.
            logger.error("Smoke test function not found or import error.", exc_info=True)
            print('✗ Smoke test not available - check bootstrap implementation')
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error during smoke test: {e}", exc_info=True)
            print(f'✗ Smoke test encountered an error: {e}')
            sys.exit(1)


    if sys.argv[1] == '--validate':
        if len(sys.argv) < 3:
            print('Error: --validate requires a manifest file')
            sys.exit(1)
        try:
            from .core.main import validate_bootstrap_config
            result = await validate_bootstrap_config([sys.argv[2]])
            print(f"Validation: {('✓ PASSED' if result['valid'] else '✗ FAILED')}")
            if result['errors']:
                print('Errors:')
                for error in result['errors']:
                    print(f'  - {error}')
            if result['warnings']:
                print('Warnings:')
                for warning in result['warnings']:
                    print(f'  - {warning}')
            sys.exit(0 if result['valid'] else 1)
        except ImportError:
            logger.error("validate_bootstrap_config function not found or import error.", exc_info=True)
            print('✗ Validation not available - check bootstrap implementation')
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error during validation: {e}", exc_info=True)
            print(f'✗ Validation encountered an error: {e}')
            sys.exit(1)


    config_paths = [Path(p) for p in sys.argv[1:]]
    for path in config_paths:
        if not path.exists():
            print(f'Error: Configuration file not found: {path}')
            sys.exit(1)
        if not path.suffix.lower() in ['.yaml', '.yml']:
            # This is a warning, not a fatal error for the CLI.
            print(f'Warning: File {path} does not have .yaml/.yml extension, processing anyway.')

    try:
        from .core.main import bootstrap_nireon_system
        logger.info(f'Starting bootstrap with {len(config_paths)} configuration files: {config_paths}')
        # Assuming strict_mode=True is a desired default for CLI unless specified otherwise
        result = await bootstrap_nireon_system(config_paths, strict_mode=True)
        
        print(f'\n=== Bootstrap Summary (Run ID: {result.run_id}) ===')
        print(f'Success: {result.success}')
        print(f'Components: {result.component_count}')
        print(f'Healthy: {result.healthy_component_count}')
        if hasattr(result, 'bootstrap_duration') and result.bootstrap_duration is not None:
            print(f'Duration: {result.bootstrap_duration:.2f}s')
        
        if not result.success:
            print(f'Critical Failures: {result.critical_failure_count}')
            print('\n--- Health Report (excerpt) ---')
            health_report = result.get_health_report() # This should return a string
            # Basic truncation for very long reports
            if len(health_report) > 2000: # Increased limit for more context
                print(health_report[:2000] + '\n... (report truncated for brevity)')
            else:
                print(health_report)
        sys.exit(0 if result.success else 1)
    except KeyboardInterrupt:
        print('\n✗ Bootstrap interrupted by user')
        sys.exit(130) # Standard exit code for Ctrl+C
    except Exception as e:
        logger.error(f'Bootstrap failed with an unhandled exception: {e}', exc_info=True)
        print(f'\n✗ FATAL BOOTSTRAP ERROR: {e}')
        sys.exit(2) # General error

if __name__ == '__main__':
    try:
        asyncio.run(main_cli_entry())
    except KeyboardInterrupt:
        # This handles KeyboardInterrupt if it happens directly in asyncio.run,
        # though the one inside main_cli_entry is more likely to catch it during await.
        print('\nBootstrap process interrupted by user (top-level).')
        sys.exit(130)
    except Exception as e:
        # Catches errors from asyncio.run itself or if main_cli_entry raises
        # an exception not caught internally (though it should).
        print(f'Fatal error during bootstrap execution: {e}')
        logger.critical(f'Unhandled fatal error in __main__: {e}', exc_info=True)
        sys.exit(1)
```
--- END OF FILE nireon_v4/bootstrap/__main__.py ---

--- START OF FILE nireon_v4/bootstrap/bootstrap_helper/enhanced_components.py ---
```python
# nireon_v4/bootstrap/bootstrap_helper/enhanced_components.py
from __future__ import annotations
from __future__ import absolute_import

import dataclasses
import importlib
import inspect
import json
import logging
import re
import types
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Type, Union, cast

from pydantic import BaseModel # Keep for potential local models if any

# Import the consolidated helper functions
from bootstrap.utils.component_utils import (
    get_pydantic_defaults,
    create_component_instance,
    # The following are not directly used in init_full_component but were in the original file.
    # If init_full_component logic expands to use them, they are available.
    # inject_dependencies,
    # validate_component_interfaces,
    # configure_component_logging,
    # prepare_component_metadata,
)

from bootstrap.bootstrap_helper.placeholders import PlaceholderEventBusImpl # Used by _report_and_raise if reporter is None
from bootstrap.health.reporter import BootstrapHealthReporter, ComponentStatus
from configs.config_utils import ConfigMerger # Assuming this is available
from core.base_component import NireonBaseComponent
from core.lifecycle import ComponentMetadata, ComponentRegistryMissingError
from core.registry import ComponentRegistry
from domain.context import NireonExecutionContext # Not directly used here, but often related
from domain.ports.event_bus_port import EventBusPort
from factories.dependencies import CommonMechanismDependencies # Used in init_full_component
from runtime.utils import import_by_path, load_yaml_robust
from ._exceptions import BootstrapError # Relative import within bootstrap_helper
from .service_resolver import _safe_register_service_instance # Relative import

logger = logging.getLogger(__name__)

# This __all__ should reflect what this module now primarily offers, which is init_full_component
__all__ = (
    'init_full_component',
    # Keep original private-like names if they are part of an implicit API used elsewhere,
    # though ideally, direct consumers would use the public versions from utils.
    # For now, focusing on init_full_component as the main export.
)

# _ID_RE is used by init_full_component locally, so it stays.
_ID_RE = re.compile('^[A-Za-z0-9_\\-\\.]+$')


async def init_full_component(
    cid_manifest: str,
    spec: Mapping[str, Any],
    registry: ComponentRegistry,
    event_bus: EventBusPort, # Assuming this is the actual EventBusPort, not a placeholder type
    run_id: str,
    replay_context: bool, # This parameter is not used in the current implementation
    common_deps: Optional[CommonMechanismDependencies] = None,
    global_app_config: Optional[Dict[str, Any]] = None,
    health_reporter: Optional[BootstrapHealthReporter] = None,
    validation_data_store: Any = None # e.g., BootstrapValidationData
) -> None:
    cfg = global_app_config or {}
    strict = cfg.get('bootstrap_strict_mode', True)

    if not _ID_RE.match(cid_manifest):
        msg = f"Component ID '{cid_manifest}' contains invalid characters. Must match '{_ID_RE.pattern}'."
        _report_and_raise(msg, cid_manifest, health_reporter, strict, ComponentStatus.DEFINITION_ERROR)
        return

    if not spec.get('enabled', True):
        logger.info("Component '%s' disabled in manifest – skipping.", cid_manifest)
        if health_reporter:
            # Create a minimal metadata for reporting disabled status
            minimal_meta = ComponentMetadata(id=cid_manifest, name=cid_manifest, version='N/A', category='disabled')
            health_reporter.add_component_status(cid_manifest, ComponentStatus.DISABLED, minimal_meta)
        return

    class_path = spec.get('class')
    meta_path = spec.get('metadata_definition')

    if not class_path: # meta_path can be optional if class provides METADATA_DEFINITION
        msg = f"Component '{cid_manifest}' missing 'class' path in manifest spec."
        _report_and_raise(msg, cid_manifest, health_reporter, strict, ComponentStatus.DEFINITION_ERROR)
        return

    try:
        comp_cls = _import_class(class_path, cid_manifest)
        
        # Resolve canonical_meta: from path, or from class, or fallback
        if meta_path:
            canonical_meta = _import_metadata(meta_path, cid_manifest)
        elif hasattr(comp_cls, 'METADATA_DEFINITION') and isinstance(comp_cls.METADATA_DEFINITION, ComponentMetadata):
            canonical_meta = comp_cls.METADATA_DEFINITION
            logger.debug(f"Using METADATA_DEFINITION from class {comp_cls.__name__} for '{cid_manifest}'.")
        else:
            msg = (f"Component '{cid_manifest}' (class: {class_path}) missing 'metadata_definition' path in spec "
                   f"and class does not have a 'METADATA_DEFINITION' attribute.")
            _report_and_raise(msg, cid_manifest, health_reporter, strict, ComponentStatus.DEFINITION_ERROR)
            return
            
    except BootstrapError as exc:
        _report_and_raise(str(exc), cid_manifest, health_reporter, strict, ComponentStatus.DEFINITION_ERROR)
        return
    except Exception as exc: # Catch other import errors too
        _report_and_raise(f"Failed to import class/metadata for '{cid_manifest}': {exc}",
                          cid_manifest, health_reporter, strict, ComponentStatus.DEFINITION_ERROR)
        return

    inst_meta = _build_instance_metadata(canonical_meta, cid_manifest, spec)
    
    # Use the consolidated get_pydantic_defaults
    pyd_defaults = get_pydantic_defaults(comp_cls, inst_meta.name)
    
    yaml_cfg = _load_yaml_layer(spec.get('config'), cid_manifest)
    inline_override = spec.get('config_override', {})
    
    merged_cfg = ConfigMerger.merge(pyd_defaults, yaml_cfg, f'{cid_manifest}_pyd_yaml')
    resolved_cfg = ConfigMerger.merge(merged_cfg, inline_override, f'{cid_manifest}_final_config')
    
    logger.info('[%s] Resolved config keys: %s', cid_manifest, list(resolved_cfg.keys()))

    if validation_data_store and hasattr(validation_data_store, 'store_component_data'):
        validation_data_store.store_component_data(
            component_id=cid_manifest,
            original_metadata=inst_meta, # This should be the version before instance overrides
            resolved_config=resolved_cfg,
            manifest_spec=spec
        )
    
    # Adapt common_deps for the new create_component_instance signature
    common_deps_dict: Optional[Dict[str, Any]] = None
    if common_deps:
        common_deps_dict = {
            # Map attributes of CommonMechanismDependencies to expected constructor param names
            'llm_port': getattr(common_deps, 'llm_port', None), # or 'llm' if that's the param name
            'embedding_port': getattr(common_deps, 'embedding_port', None),
            'event_bus': getattr(common_deps, 'event_bus', None),
            'idea_service': getattr(common_deps, 'idea_service', None),
            'component_registry': getattr(common_deps, 'component_registry', None),
            'rng': getattr(common_deps, 'rng', None),
            # Add 'common_deps' itself if some components expect the whole object
            'common_deps': common_deps 
        }

    instance: Optional[NireonBaseComponent] = None
    try:
        # Use the consolidated create_component_instance
        instance = await create_component_instance(
            comp_cls, 
            resolved_cfg, 
            cid_manifest, 
            inst_meta, 
            common_deps=common_deps_dict
        )
    except BootstrapError as exc:
        _report_and_raise(str(exc), cid_manifest, health_reporter, strict, ComponentStatus.INSTANTIATION_ERROR, inst_meta)
        return
    except Exception as exc: # Catch unexpected instantiation errors
        _report_and_raise(f"Unexpected error during instantiation of '{cid_manifest}': {exc}", 
                          cid_manifest, health_reporter, strict, ComponentStatus.INSTANTIATION_ERROR, inst_meta)
        return

    if instance is None: # Should be caught by create_component_instance raising an error, but as a safeguard.
        _report_and_raise(f"Instantiation for '{cid_manifest}' returned None.", 
                          cid_manifest, health_reporter, strict, ComponentStatus.INSTANTIATION_ERROR, inst_meta)
        return

    try:
        # Ensure the instance has its metadata correctly set, especially the ID.
        # create_component_instance should handle this, but double check.
        if not hasattr(instance, 'metadata') or instance.metadata.id != cid_manifest:
             # This suggests an issue in create_component_instance or component's __init__
            logger.warning(f"Metadata ID mismatch for '{cid_manifest}' post-instantiation. Instance has: {getattr(instance, 'metadata', None)}. Attempting to fix.")
            if hasattr(instance, '_metadata_definition'):
                 object.__setattr__(instance, '_metadata_definition', inst_meta)
            elif hasattr(instance, 'metadata'):
                 object.__setattr__(instance, 'metadata', inst_meta)


        _safe_register_service_instance(
            registry,
            type(instance),
            instance,
            instance.component_id, # Use ID from instance, should match cid_manifest
            instance.metadata.category,
            description_for_meta=instance.metadata.description,
            requires_initialize_override=instance.metadata.requires_initialize
        )
        if health_reporter:
            health_reporter.add_component_status(
                instance.component_id,
                ComponentStatus.INSTANCE_REGISTERED_INIT_DEFERRED, # Standard status post-registration
                instance.metadata
            )
        logger.info("✓ Registered component '%s' (type: %s, category: %s, requires_init=%s)",
                    instance.component_id, type(instance).__name__, instance.metadata.category, instance.metadata.requires_initialize)
    except Exception as exc:
        _report_and_raise(f"Error during registration of '{cid_manifest}': {exc}", 
                          cid_manifest, health_reporter, strict, ComponentStatus.REGISTRATION_ERROR, inst_meta)


# Helper functions that were part of the original module, kept for init_full_component
def _import_class(path: str, cid: str) -> Type:
    obj = import_by_path(path)
    if not inspect.isclass(obj):
        raise BootstrapError(f"'{path}' for '{cid}' is not a class.")
    return cast(Type, obj)

def _import_metadata(path: str, cid: str) -> ComponentMetadata:
    obj = import_by_path(path)
    if not isinstance(obj, ComponentMetadata):
        raise BootstrapError(f"'{path}' for '{cid}' is not a ComponentMetadata instance.")
    return obj

def _build_instance_metadata(canonical: ComponentMetadata, cid: str, spec: Mapping[str, Any]) -> ComponentMetadata:
    data = asdict(canonical)
    data['id'] = cid # Ensure ID is set to the manifest component ID
    
    # Apply metadata_override from the spec
    metadata_override = spec.get('metadata_override', {})
    if metadata_override:
        logger.debug(f"Applying metadata_override for '{cid}': {metadata_override}")
        data.update(metadata_override)

    # Specific handling for epistemic_tags if present directly in spec
    if 'epistemic_tags' in spec and isinstance(spec['epistemic_tags'], list):
        data['epistemic_tags'] = list(spec['epistemic_tags'])
    
    # Ensure requires_initialize is correctly sourced
    # Priority: spec override > canonical metadata
    if 'requires_initialize' in metadata_override: # If it was in metadata_override
        data['requires_initialize'] = metadata_override['requires_initialize']
    elif 'requires_initialize' in spec: # If directly in spec (less common but possible)
         data['requires_initialize'] = spec['requires_initialize']
    else: # Fallback to canonical
        data.setdefault('requires_initialize', canonical.requires_initialize)
        
    try:
        return ComponentMetadata(**data)
    except TypeError as e:
        raise BootstrapError(f"Failed to create ComponentMetadata for '{cid}' with data {data}: {e}")


def _load_yaml_layer(path_template: Optional[Union[str, Dict[str, Any]]], cid: str) -> Dict[str, Any]:
    if not path_template:
        return {}
    
    if isinstance(path_template, dict): # Inline YAML config
        logger.debug(f"Using inline YAML config for '{cid}'")
        return path_template

    if isinstance(path_template, str):
        try:
            actual_path_str = path_template.replace('{id}', cid)
            actual_path = Path(actual_path_str)
            if actual_path.exists():
                return load_yaml_robust(actual_path)
            else:
                logger.debug(f"YAML config file not found for '{cid}' at '{actual_path_str}'. Skipping.")
                return {}
        except Exception as exc:
            logger.warning("Failed to load YAML config for '%s' from template '%s': %s", cid, path_template, exc)
            return {}
    
    logger.warning(f"Invalid 'config' type for '{cid}': {type(path_template)}. Expected string path or dict.")
    return {}


def _report_and_raise(
    msg: str,
    cid: str,
    reporter: Optional[BootstrapHealthReporter],
    strict: bool,
    status: ComponentStatus,
    meta: Optional[ComponentMetadata] = None
) -> None:
    logger.error(f"Bootstrap error for '{cid}': {msg}")
    if reporter:
        # If meta is not provided, create a minimal one for reporting
        if meta is None:
            meta = ComponentMetadata(id=cid, name=cid, version='0.0.0', category='unknown_error_state')
        reporter.add_component_status(cid, status, meta, [msg])
    if strict:
        raise BootstrapError(msg, component_id=cid)
```
--- END OF FILE nireon_v4/bootstrap/bootstrap_helper/enhanced_components.py ---

--- START OF FILE nireon_v4/bootstrap/processors/enhanced_components.py ---
```python
# nireon_v4/bootstrap/processors/enhanced_components.py
from __future__ import annotations
from __future__ import absolute_import

# This module is now significantly reduced as its core functionalities
# have been moved to bootstrap.utils.component_utils.

# It might still be imported by other processor modules if they expect
# these names to be available from this specific path.
# For now, we re-export the public versions from the utils module
# to maintain compatibility for any internal imports within the processors package
# that might have been using `from .enhanced_components import ...`.

import logging
from typing import Any, Dict, Type, Optional # Keep necessary typings

from core.base_component import NireonBaseComponent
from core.lifecycle import ComponentMetadata

# Import and re-export the consolidated helper functions
from bootstrap.utils.component_utils import (
    get_pydantic_defaults as _get_pydantic_defaults, #
    create_component_instance as _create_component_instance,
    inject_dependencies as _inject_dependencies,
    validate_component_interfaces as _validate_component_interfaces,
    configure_component_logging as _configure_component_logging,
    prepare_component_metadata as _prepare_component_metadata,
)

logger = logging.getLogger(__name__)

# Expose them with their original "private-like" names if other modules in this package
# were importing them that way. This helps minimize changes in other processor files.
# If no other processor files use these directly, this __all__ can be empty or removed.
__all__ = [
    '_get_pydantic_defaults',
    '_create_component_instance',
    '_inject_dependencies',
    '_validate_component_interfaces',
    '_configure_component_logging',
    '_prepare_component_metadata',
]

# The actual implementations are now in bootstrap.utils.component_utils.
# This file serves as a compatibility layer if other modules in the
# 'processors' package were importing these helpers directly from here.
# Ideally, those other modules would eventually be updated to import
# from bootstrap.utils.component_utils directly.
```
--- END OF FILE nireon_v4/bootstrap/processors/enhanced_components.py ---

--- START OF FILE nireon_v4/bootstrap/core/main.py ---
```python
# nireon_v4/bootstrap/core/main.py
from __future__ import annotations
from __future__ import absolute_import

import asyncio
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
import dataclasses # For BootstrapConfig adaptation

from pydantic import BaseModel, Field, ValidationError

from bootstrap.exceptions import BootstrapError, BootstrapValidationError, BootstrapTimeoutError
from bootstrap.result_builder import BootstrapResult, BootstrapResultBuilder
from bootstrap.context.bootstrap_context_builder import create_bootstrap_context
from bootstrap.config.bootstrap_config import BootstrapConfig
from bootstrap.context.bootstrap_context import BootstrapContext
from core.registry.component_registry import ComponentRegistry

# Direct imports from bootstrap.phases
try:
    from bootstrap.phases.abiogenesis_phase import AbiogenesisPhase
except ImportError as e:
    logger = logging.getLogger(__name__) # Ensure logger is defined before use
    logger.error(f'Failed to import AbiogenesisPhase: {e}')
    AbiogenesisPhase = None # type: ignore

try:
    from bootstrap.phases.registry_setup_phase import RegistrySetupPhase
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.error(f'Failed to import RegistrySetupPhase: {e}')
    RegistrySetupPhase = None # type: ignore

try:
    from bootstrap.phases.factory_setup_phase import FactorySetupPhase
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.error(f'Failed to import FactorySetupPhase: {e}')
    FactorySetupPhase = None # type: ignore

try:
    from bootstrap.phases.manifest_processing_phase import ManifestProcessingPhase
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.error(f'Failed to import ManifestProcessingPhase: {e}')
    ManifestProcessingPhase = None # type: ignore

try:
    from bootstrap.phases.component_initialization_phase import ComponentInitializationPhase
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.error(f'Failed to import ComponentInitializationPhase: {e}')
    ComponentInitializationPhase = None # type: ignore

try:
    # Assuming InterfaceValidationPhase is the intended class name from component_validation_phase
    from bootstrap.phases.component_validation_phase import InterfaceValidationPhase
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.error(f'Failed to import InterfaceValidationPhase from component_validation_phase: {e}')
    InterfaceValidationPhase = None # type: ignore

try:
    from bootstrap.phases.rbac_setup_phase import RBACSetupPhase
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.error(f'Failed to import RBACSetupPhase: {e}')
    RBACSetupPhase = None # type: ignore

try:
    from bootstrap.phases.late_rebinding_phase import LateRebindingPhase
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.error(f'Failed to import LateRebindingPhase: {e}')
    LateRebindingPhase = None # type: ignore


try:
    from dotenv import load_dotenv
    load_dotenv()
    logger = logging.getLogger(__name__) # define logger after potential dotenv load
    logger.debug("Dotenv loaded if .env file exists.")
except ImportError:
    logger = logging.getLogger(__name__)
    logger.debug("Dotenv not installed, .env file (if any) will not be loaded.")
except Exception as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"Error loading dotenv: {e}")


class BootstrapExecutionConfig(BaseModel):
    timeout_seconds: float = Field(default=300.0, ge=1.0, le=3600.0, description='Maximum time allowed for bootstrap execution')
    phase_timeout_seconds: float = Field(default=60.0, ge=1.0, le=300.0, description='Maximum time allowed per bootstrap phase')
    retry_on_failure: bool = Field(default=False, description='Whether to retry bootstrap on failure (non-strict mode only)')
    max_retries: int = Field(default=3, ge=0, le=10, description='Maximum number of retry attempts')
    enable_health_checks: bool = Field(default=True, description='Whether to perform health checks during bootstrap')
    
    class Config:
        extra = 'forbid'
        validate_assignment = True


class BootstrapOrchestrator:
    def __init__(self, config: BootstrapConfig, execution_config: Optional[BootstrapExecutionConfig] = None):
        self.config = config
        self.execution_config = execution_config or BootstrapExecutionConfig()
        self.run_id = self._generate_run_id()
        self.start_time: Optional[datetime] = None
        self._phases: Optional[List[Any]] = None # Holds phase instances
        logger.info(f'BootstrapOrchestrator initialized - run_id: {self.run_id}')
        logger.debug(f'Execution config: {self.execution_config.model_dump_json(indent=2)}') # Use Pydantic's model_dump

    async def execute_bootstrap(self) -> BootstrapResult:
        self.start_time = datetime.now(timezone.utc)
        logger.info('=== NIREON V4 System Bootstrap Starting (L0 Abiogenesis) ===')
        logger.info(f'Run ID: {self.run_id}')
        logger.info(f'Config Paths: {[str(p) for p in self.config.config_paths]}')
        logger.info(f"Environment: {self.config.env or 'default'}")
        
        try:
            bootstrap_task = self._execute_bootstrap_process()
            return await asyncio.wait_for(bootstrap_task, timeout=self.execution_config.timeout_seconds)
        except asyncio.TimeoutError:
            error_msg = f'Bootstrap timed out after {self.execution_config.timeout_seconds}s'
            logger.error(error_msg)
            # Create a minimal result for timeout
            minimal_result_for_timeout = self._create_minimal_result(BootstrapTimeoutError(error_msg))
            # Potentially update health reporter if accessible before raising/returning
            if hasattr(minimal_result_for_timeout, 'health_reporter') and minimal_result_for_timeout.health_reporter:
                minimal_result_for_timeout.health_reporter.add_phase_result(
                    'BootstrapTimeout', 'failed', error_msg, errors=[error_msg]
                )
            # Depending on desired behavior, either raise or return the minimal result
            # Raising is often cleaner for the caller to handle.
            raise BootstrapTimeoutError(error_msg)
        except Exception as e:
            logger.critical(f'Unhandled exception during bootstrap execution: {e}', exc_info=True)
            return await self._handle_bootstrap_failure(e)

    async def _execute_bootstrap_process(self) -> BootstrapResult:
        global_config = await self._load_global_configuration()
        context = await self._create_bootstrap_context(global_config)
        
        logger.info(f'Effective Strict Mode: {context.strict_mode}')
        logger.info(f'Health Checks Enabled: {self.execution_config.enable_health_checks}')
        
        await self._signal_bootstrap_started(context)
        await self._execute_phases(context)
        
        if self.execution_config.enable_health_checks:
            await self._perform_final_health_checks(context)
            
        await self._signal_bootstrap_completion(context)
        return self._build_result(context)

    async def _load_global_configuration(self) -> Dict[str, Any]:
        try:
            logger.debug('Loading global configuration...')
            try:
                from configs.config_loader import ConfigLoader
            except ImportError:
                logger.warning('ConfigLoader not available, using minimal config.')
                return self._get_minimal_global_config()

            config_loader = ConfigLoader()
            # Ensure global_app_config on self.config is updated if it was initially None
            # This makes the loaded global_config authoritative.
            loaded_global_config = await config_loader.load_global_config(
                env=self.config.env, 
                provided_config=self.config.global_app_config # Pass existing if any
            )
            
            if self.config.global_app_config is None or self.config.global_app_config != loaded_global_config:
                 # Update the BootstrapConfig instance's global_app_config
                 # Since BootstrapConfig is a dataclass, we might need to create a new instance or modify if mutable
                if dataclasses.is_dataclass(self.config):
                    self.config = dataclasses.replace(self.config, global_app_config=loaded_global_config)
                else: # Fallback if not a dataclass, direct assignment (less safe)
                    self.config.global_app_config = loaded_global_config


            logger.info(f"Global configuration loaded for env: {self.config.env or 'default'}")
            logger.debug(f'Config keys: {list(loaded_global_config.keys())}')
            return loaded_global_config
        except Exception as e:
            logger.error(f'Failed to load global configuration: {e}', exc_info=True)
            logger.warning('Using minimal configuration due to load failure.')
            minimal_cfg = self._get_minimal_global_config()
            if dataclasses.is_dataclass(self.config):
                self.config = dataclasses.replace(self.config, global_app_config=minimal_cfg)
            else:
                self.config.global_app_config = minimal_cfg
            return minimal_cfg


    def _get_minimal_global_config(self) -> Dict[str, Any]:
        return {
            'env': self.config.env or 'default',
            'bootstrap_strict_mode': self.config.initial_strict_mode_param, # Use the initial param
            'feature_flags': {
                'enable_rbac_bootstrap': False,
                'enable_schema_validation': False,
                'enable_concurrent_initialization': False,
            },
            'llm': {'default_model': 'placeholder', 'timeout_seconds': 30},
            'embedding': {'default_model': 'placeholder', 'dimensions': 384},
            'shared_services': {},
            'mechanisms': {},
            'observers': {}
        }

    async def _create_bootstrap_context(self, global_config: Dict[str, Any]) -> BootstrapContext:
        try:
            logger.debug('Creating bootstrap context...')
            context = await create_bootstrap_context(
                run_id=self.run_id,
                config=self.config, # self.config now has updated global_app_config
                global_config=global_config, # Pass it explicitly too for clarity in create_bootstrap_context
                # create_bootstrap_context will use strict_mode from self.config.effective_strict_mode
            )
            logger.info('Bootstrap context created successfully.')
            return context
        except Exception as e:
            logger.error(f'Failed to create bootstrap context: {e}', exc_info=True)
            raise BootstrapError(f'Context creation failed: {e}') from e

    async def _signal_bootstrap_started(self, context: BootstrapContext) -> None:
        try:
            if hasattr(context, 'signal_emitter') and context.signal_emitter:
                 await context.signal_emitter.emit_bootstrap_started()
                 logger.debug('Bootstrap started signal emitted.')
            else:
                logger.warning("Signal emitter not found on context. Cannot emit bootstrap_started signal.")
        except Exception as e:
            logger.warning(f'Failed to emit bootstrap started signal: {e}')
            if context.strict_mode:
                raise BootstrapError(f'Failed to emit bootstrap started signal: {e}') from e


    async def _execute_phases(self, context: BootstrapContext) -> None:
        phases = self._get_phases() # This now returns instances of phases
        total_phases = len(phases)
        logger.info(f'Executing {total_phases} bootstrap phases...')

        for i, phase_instance in enumerate(phases, 1):
            if phase_instance is None: # Should not happen if _get_phases filters correctly
                logger.warning(f'Phase {i} is None, skipping.')
                continue
            
            phase_name = phase_instance.__class__.__name__
            logger.info(f'Executing Phase {i}/{total_phases}: {phase_name}')
            
            try:
                # Assuming phases have an execute_with_hooks or execute method
                if hasattr(phase_instance, 'execute_with_hooks'):
                    phase_task = phase_instance.execute_with_hooks(context)
                elif hasattr(phase_instance, 'execute'):
                    phase_task = phase_instance.execute(context)
                else:
                    logger.error(f"Phase {phase_name} has no execute or execute_with_hooks method.")
                    if context.strict_mode:
                        raise BootstrapError(f"Phase {phase_name} is not executable.")
                    continue

                # Await the phase result with timeout
                # PhaseResult should be the common return type
                from bootstrap.phases.base_phase import PhaseResult # Ensure type hint
                result: PhaseResult = await asyncio.wait_for(phase_task, timeout=self.execution_config.phase_timeout_seconds)
                
                if not result.success:
                    error_msg = f"Phase {phase_name} failed: {'; '.join(result.errors)}"
                    if context.strict_mode:
                        raise BootstrapError(error_msg)
                    else:
                        logger.warning(f'{error_msg} (continuing in non-strict mode)')
                else:
                    logger.info(f'✓ Phase {phase_name} completed successfully. Message: {result.message}')

            except asyncio.TimeoutError:
                error_msg = f'Phase {phase_name} timed out after {self.execution_config.phase_timeout_seconds}s'
                logger.error(error_msg)
                if context.strict_mode:
                    raise BootstrapError(error_msg) # Re-raise as BootstrapError for consistent handling
                else:
                    logger.warning(f'{error_msg} (continuing in non-strict mode)')
            except BootstrapError: # Re-raise if already a BootstrapError
                raise
            except Exception as e: # Catch other unexpected errors from phase execution
                error_msg = f'Unexpected error in {phase_name}: {e}'
                logger.error(error_msg, exc_info=True)
                if context.strict_mode:
                    raise BootstrapError(error_msg) from e
                else:
                    logger.warning(f'{error_msg} (continuing in non-strict mode)')

    async def _perform_final_health_checks(self, context: BootstrapContext) -> None:
        try:
            logger.info('Performing final health checks...')
            component_count = 0
            if context.registry and hasattr(context.registry, 'list_components'):
                 component_count = len(context.registry.list_components())
            
            if component_count == 0:
                logger.warning('No components registered during bootstrap.')
            else:
                logger.info(f'Registry contains {component_count} components.')

            if hasattr(context, 'health_reporter') and context.health_reporter:
                health_summary = context.health_reporter.generate_summary()
                logger.info(f'Final Health Summary:\n{health_summary}')
            else:
                logger.warning("Health reporter not available on context for final health checks.")
            
            logger.info('Final health checks completed.')
        except Exception as e:
            logger.warning(f'Health checks failed: {e}', exc_info=True)
            if context.strict_mode:
                raise BootstrapError(f'Final health checks failed: {e}') from e

    async def _signal_bootstrap_completion(self, context: BootstrapContext) -> None:
        try:
            component_count = 0
            if context.registry and hasattr(context.registry, 'list_components'):
                 component_count = len(context.registry.list_components())
            duration_seconds = self._get_elapsed_seconds()
            
            if hasattr(context, 'signal_emitter') and context.signal_emitter:
                await context.signal_emitter.emit_bootstrap_completed(
                    component_count=component_count,
                    duration_seconds=duration_seconds
                )
                logger.info(f'Bootstrap completion signaled - {component_count} components, {duration_seconds:.2f}s duration.')
            else:
                 logger.warning("Signal emitter not found on context. Cannot emit bootstrap_completed signal.")
        except Exception as e:
            logger.warning(f'Failed to signal bootstrap completion: {e}')
            if context.strict_mode:
                raise BootstrapError(f'Failed to signal completion: {e}') from e

    def _build_result(self, context: BootstrapContext) -> BootstrapResult:
        try:
            logger.debug('Building bootstrap result...')
            builder = BootstrapResultBuilder(context)
            result = builder.build() # build() now handles marking health_reporter complete
            
            if result.success:
                logger.info(f'✓ NIREON V4 Bootstrap Complete. Run ID: {result.run_id}. Components: {result.component_count}, Duration: {result.bootstrap_duration or 0:.2f}s')
            else:
                logger.error(f'✗ NIREON V4 Bootstrap Failed. Run ID: {result.run_id}. Critical Failures: {result.critical_failure_count}')
            return result
        except Exception as e:
            logger.error(f'Failed to build bootstrap result: {e}', exc_info=True)
            # Fallback to minimal result if building full result fails
            return self._create_minimal_result(e, context_fallback=context)


    async def _handle_bootstrap_failure(self, error: Exception) -> BootstrapResult:
        logger.critical(f'Critical bootstrap failure: {error}', exc_info=True)
        
        # In strict mode, we re-raise the error for the caller to handle.
        # The exception is converted to BootstrapError if it's not already one.
        if self.config.effective_strict_mode:
            if isinstance(error, (BootstrapError, BootstrapTimeoutError, BootstrapValidationError)):
                raise error
            raise BootstrapError(f'Bootstrap system failure: {error}') from error
            
        # In non-strict mode, we attempt to create a minimal result.
        logger.warning('Attempting to create minimal result in non-strict mode due to failure...')
        try:
            return self._create_minimal_result(error)
        except Exception as recovery_error:
            # If even minimal result creation fails, log and raise a generic BootstrapError.
            logger.error(f'Recovery (minimal result creation) failed: {recovery_error}', exc_info=True)
            # This is a last resort.
            final_error = BootstrapError(f'Complete bootstrap failure and recovery failed. Original error: {error}')
            final_error.__cause__ = recovery_error # Chain the recovery error
            raise final_error


    def _create_minimal_result(self, original_error: Exception, context_fallback: Optional[BootstrapContext] = None) -> BootstrapResult:
        logger.info(f"Creating minimal bootstrap result due to error: {original_error}")
        registry = self.config.existing_registry or (context_fallback.registry if context_fallback else None) or ComponentRegistry()
        
        health_reporter = None
        if context_fallback and hasattr(context_fallback, 'health_reporter'):
            health_reporter = context_fallback.health_reporter
        
        if health_reporter is None:
            try:
                from bootstrap.health.reporter import HealthReporter
                health_reporter = HealthReporter(registry)
            except ImportError: # Should not happen based on project structure
                logger.error("Failed to import HealthReporter for minimal result.")
                health_reporter = None # type: ignore
        
        if health_reporter and hasattr(health_reporter, 'add_phase_result'):
            health_reporter.add_phase_result(
                'BootstrapFailure', 
                'failed', 
                f'Bootstrap failed critically: {original_error}', 
                errors=[str(original_error)]
            )
            if hasattr(health_reporter, 'mark_bootstrap_complete'):
                 health_reporter.mark_bootstrap_complete()

        validation_data = None
        if context_fallback and hasattr(context_fallback, 'validation_data_store'):
            validation_data = context_fallback.validation_data_store
        
        if validation_data is None:
            try:
                from bootstrap.validation_data import BootstrapValidationData
                validation_data = BootstrapValidationData(self.config.global_app_config or {}, self.run_id)
            except ImportError:
                logger.error("Failed to import BootstrapValidationData for minimal result.")
                validation_data = None # type: ignore

        return BootstrapResult(
            registry=registry,
            health_reporter=health_reporter,
            validation_data=validation_data,
            run_id=self.run_id,
            bootstrap_duration=self._get_elapsed_seconds(),
            global_config=self.config.global_app_config
        )

    def _get_phases(self) -> List[Any]: # Returns list of phase instances
        if self._phases is None:
            # List of phase classes to instantiate
            phase_classes = [
                AbiogenesisPhase,
                RegistrySetupPhase,
                FactorySetupPhase,
                ManifestProcessingPhase,
                ComponentInitializationPhase,
                InterfaceValidationPhase,
                RBACSetupPhase,
                LateRebindingPhase, # Added late rebinding phase
            ]
            
            self._phases = []
            for phase_cls in phase_classes:
                if phase_cls is not None:
                    try:
                        self._phases.append(phase_cls()) # Instantiate the phase
                    except Exception as e:
                        logger.error(f"Failed to instantiate phase {phase_cls.__name__}: {e}", exc_info=True)
                else:
                    # This case handles if a phase failed to import and is None
                    logger.warning(f"A phase class was None during instantiation, likely due to import error.")
            
            logger.info(f'Initialized {len(self._phases)} bootstrap phase instances.')
        return self._phases

    def _generate_run_id(self) -> str:
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f') # Added microseconds
        process_id = os.getpid()
        # Consider adding a random element for more uniqueness if needed
        return f'nireon_bs_{timestamp}_{process_id}'

    def _get_elapsed_seconds(self) -> float:
        if self.start_time is None:
            return 0.0
        return (datetime.now(timezone.utc) - self.start_time).total_seconds()


async def bootstrap_nireon_system(
    config_paths: Sequence[Union[str, Path]], 
    **kwargs: Any # Accept arbitrary keyword arguments
) -> BootstrapResult:
    try:
        # Separate BootstrapConfig kwargs from BootstrapExecutionConfig kwargs
        bootstrap_config_fields = BootstrapConfig.__annotations__.keys() # Or __dataclass_fields__ if dataclass
        execution_config_fields = BootstrapExecutionConfig.model_fields.keys()

        bs_config_kwargs = {k: v for k, v in kwargs.items() if k in bootstrap_config_fields and k != 'config_paths'}
        exec_config_kwargs = {k: v for k, v in kwargs.items() if k in execution_config_fields}
        
        # Create BootstrapConfig
        config = BootstrapConfig.from_params(list(config_paths), **bs_config_kwargs)

        # Create BootstrapExecutionConfig if any relevant kwargs were passed
        execution_config = None
        if exec_config_kwargs:
            execution_config = BootstrapExecutionConfig(**exec_config_kwargs)
        
        orchestrator = BootstrapOrchestrator(config, execution_config)
        return await orchestrator.execute_bootstrap()
        
    except ValidationError as e: # Pydantic validation error for configs
        logger.error(f'Bootstrap configuration validation failed: {e}', exc_info=True)
        raise BootstrapValidationError(f'Bootstrap configuration validation failed: {e}') from e
    except BootstrapError: # Re-raise known bootstrap errors
        raise
    except Exception as e: # Catch other unexpected errors
        logger.error(f'Unexpected error in bootstrap_nireon_system: {e}', exc_info=True)
        raise BootstrapError(f'System bootstrap failed with an unexpected error: {e}') from e

bootstrap = bootstrap_nireon_system # Alias

def bootstrap_sync(
    config_paths: Sequence[Union[str, Path]], 
    **kwargs: Any
) -> BootstrapResult:
    logger.debug('Running V4 bootstrap in synchronous mode.')
    current_loop: Optional[asyncio.AbstractEventLoop] = None
    new_loop_created = False
    try:
        current_loop = asyncio.get_running_loop()
    except RuntimeError: # No event loop is running
        current_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(current_loop)
        new_loop_created = True
        logger.debug('No running event loop, created a new one for sync bootstrap.')
    
    # If a loop was already running, the feedback suggested creating a *nested* new loop.
    # However, asyncio generally discourages nested loops.
    # A common pattern for sync wrappers around async code is to run_until_complete on the existing loop if possible,
    # or create one if none exists. If true nested execution is required, it's more complex.
    # For now, let's assume if a loop is running, we use it, otherwise, we create one.
    # If the user's feedback *insists* on a new nested loop:
    # if current_loop and not new_loop_created: # A loop was already running
    #    logger.warning('Event loop already running. Standard asyncio does not support nested loops well. '
    #                   'Running on the existing loop. For true nesting, consider libraries like `nest_asyncio`.')

    try:
        # Ensure the future is created within the context of the loop that will run it
        future = bootstrap_nireon_system(config_paths, **kwargs)
        return current_loop.run_until_complete(future)
    except Exception as e:
        logger.error(f'Synchronous bootstrap failed: {e}', exc_info=True)
        # Convert to BootstrapError if not already, to maintain consistency
        if not isinstance(e, BootstrapError):
            raise BootstrapError(f'Synchronous bootstrap execution error: {e}') from e
        raise
    finally:
        if new_loop_created and current_loop:
            current_loop.close()
            logger.debug('Closed the event loop created for sync bootstrap.')
            # Set event loop to None if we closed the one we created and it was the global one.
            # This is to avoid issues if another part of the code tries to get_event_loop later.
            try:
                if asyncio.get_event_loop() is current_loop: # Check if it's still the current loop
                    asyncio.set_event_loop(None)
            except RuntimeError: # If no loop is set after closing
                 pass


# Publicly exposed symbols from this module
__all__ = [
    'BootstrapExecutionConfig', 
    'BootstrapOrchestrator', 
    'bootstrap_nireon_system', 
    'bootstrap', 
    'bootstrap_sync',
    # Added for CLI usage potentially
    'smoke_test', 
    'validate_bootstrap_config' 
]

# Placeholder for smoke_test and validate_bootstrap_config if they were intended for CLI
# and need to be defined or imported here.
async def smoke_test() -> bool:
    """A basic smoke test for the bootstrap system."""
    logger.info("Executing bootstrap smoke test...")
    # This should involve a minimal, controlled bootstrap execution.
    # For now, it's a placeholder.
    # Example: Bootstrap with an empty manifest or a very simple one.
    try:
        # Create a dummy manifest path or use a predefined minimal one
        from tempfile import NamedTemporaryFile
        with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp_manifest:
            tmp_manifest.write("version: 1.0\nmetadata:\n  name: SmokeTestManifest\ncomponents: []\n")
            manifest_path = tmp_manifest.name
        
        logger.debug(f"Smoke test using temporary manifest: {manifest_path}")
        result = await bootstrap_nireon_system([Path(manifest_path)], strict_mode=False)
        os.remove(manifest_path) # Clean up

        if result.success and result.component_count == 0: # Expecting success with 0 components for empty manifest
            logger.info("Bootstrap smoke test passed.")
            return True
        else:
            logger.error(f"Bootstrap smoke test failed. Success: {result.success}, Components: {result.component_count}")
            return False
    except Exception as e:
        logger.error(f"Bootstrap smoke test encountered an exception: {e}", exc_info=True)
        return False

async def validate_bootstrap_config(config_paths: List[str]) -> Dict[str, Any]:
    """Validates bootstrap configuration files."""
    logger.info(f"Validating bootstrap configuration: {config_paths}")
    # This should perform schema validation and basic structural checks on manifests.
    # Placeholder implementation.
    errors: List[str] = []
    warnings: List[str] = []
    is_valid = True

    if not config_paths:
        errors.append("No configuration paths provided for validation.")
        is_valid = False
    
    for path_str in config_paths:
        path = Path(path_str)
        if not path.exists():
            errors.append(f"Configuration file not found: {path}")
            is_valid = False
            continue
        if not path.is_file():
            errors.append(f"Configuration path is not a file: {path}")
            is_valid = False
            continue
        if path.suffix.lower() not in ['.yaml', '.yml']:
            warnings.append(f"File {path} does not have a .yaml/.yml extension.")
        
        # Add more sophisticated validation here (e.g., schema validation)
        # For now, just basic checks.
        try:
            from runtime.utils import load_yaml_robust
            data = load_yaml_robust(path)
            if not isinstance(data, dict):
                errors.append(f"Manifest {path} does not load as a dictionary.")
                is_valid = False
            elif 'version' not in data:
                warnings.append(f"Manifest {path} is missing a 'version' field.")

        except Exception as e:
            errors.append(f"Error loading or parsing manifest {path}: {e}")
            is_valid = False
            
    return {"valid": is_valid, "errors": errors, "warnings": warnings}
```
--- END OF FILE nireon_v4/bootstrap/core/main.py ---

--- START OF FILE nireon_v4/bootstrap/processors/component_processor.py ---
```python
# nireon_v4/bootstrap/processors/component_processor.py
from __future__ import absolute_import

import asyncio
import dataclasses # For working with ComponentMetadata
import inspect
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from core.base_component import NireonBaseComponent
from core.lifecycle import ComponentMetadata, ComponentRegistryMissingError
from core.registry.component_registry import ComponentRegistry
from factories.mechanism_factory import SimpleMechanismFactory # Assuming this is the correct import
from domain.ports.event_bus_port import EventBusPort
from bootstrap.exceptions import BootstrapError
from bootstrap.health.reporter import HealthReporter as BootstrapHealthReporter, ComponentStatus
# from bootstrap.bootstrap_helper.metadata import get_default_metadata # Now from utils
from bootstrap.processors.metadata import get_default_metadata # This was specific to processors

# from bootstrap.bootstrap_helper.service_resolver import _safe_register_service_instance # Now from processors.service_resolver
from bootstrap.processors.service_resolver import _safe_register_service_instance

from runtime.utils import import_by_path, load_yaml_robust

# Import from the new utils module
from bootstrap.utils.component_utils import (
    create_component_instance, # Using the public name
    get_pydantic_defaults,   # Using the public name
)

try:
    from configs.config_utils import ConfigMerger
except ImportError:
    logger = logging.getLogger(__name__) # Define logger if not already defined
    logger.warning("ConfigMerger not found in configs.config_utils, using placeholder.")
    class ConfigMerger: # Basic placeholder
        @staticmethod
        def merge(dict1: Dict, dict2: Dict, context_name: str) -> Dict:
            # Simple merge, dict2 overrides dict1
            logger.debug(f"Using placeholder ConfigMerger for: {context_name}")
            result = dict1.copy()
            result.update(dict2)
            return result

logger = logging.getLogger(__name__)


async def process_simple_component(
    comp_def: Dict[str, Any],
    registry: ComponentRegistry,
    mechanism_factory: Optional[SimpleMechanismFactory], # Made optional as not all types use it
    health_reporter: BootstrapHealthReporter,
    run_id: str, # run_id is not directly used here, but part of the signature
    global_app_config: Dict[str, Any],
    validation_data_store: Any # e.g., BootstrapValidationData
) -> None:
    component_id = comp_def.get('component_id')
    factory_key = comp_def.get('factory_key') # For mechanisms primarily
    component_class_path = comp_def.get('class') # For direct class instantiation
    component_type = comp_def.get('type') # e.g., 'mechanism', 'observer', 'manager', 'composite'
    
    # Config can be a path string, an inline dict, or missing (use defaults)
    yaml_config_source = comp_def.get('config') 
    inline_config_override = comp_def.get('config_override', {}) # Always a dict

    is_enabled = comp_def.get('enabled', True)
    is_strict_mode = global_app_config.get('bootstrap_strict_mode', True)

    # Initial metadata for reporting before full resolution
    current_component_metadata_for_reporting = ComponentMetadata(
        id=component_id or 'unknown_id_simple',
        name=factory_key or component_class_path or 'UnknownNameSimple',
        version='0.0.0', # Placeholder version
        category=component_type or 'unknown_type_simple',
        requires_initialize=True # Default assumption, will be refined
    )

    if not is_enabled:
        logger.info(f"Simple component '{component_id}' is disabled via manifest. Skipping.")
        health_reporter.add_component_status(
            component_id or 'unknown_disabled_component', 
            ComponentStatus.DISABLED, 
            current_component_metadata_for_reporting, 
            ['Disabled in manifest']
        )
        return

    if not component_id or not (factory_key or component_class_path) or not component_type:
        errmsg = (f"Skipping simple component definition due to missing 'component_id', "
                  f"('factory_key' or 'class'), or 'type'. Definition: {comp_def}")
        logger.error(errmsg)
        health_reporter.add_component_status(
            component_id or 'unknown_component_def_error', 
            ComponentStatus.DEFINITION_ERROR, 
            current_component_metadata_for_reporting, 
            [errmsg]
        )
        if is_strict_mode:
            raise BootstrapError(errmsg, component_id=(component_id or 'unknown_component_def_error'))
        return

    target_class_or_factory_key = component_class_path if component_class_path else factory_key
    
    # Attempt to get base metadata (e.g., from a predefined map or class attribute)
    base_metadata_for_class: Optional[ComponentMetadata] = None
    if factory_key: # Try factory_key first for well-known components
        base_metadata_for_class = get_default_metadata(factory_key) # Uses processors.metadata.get_default_metadata

    if not base_metadata_for_class and component_class_path:
        try:
            cls_for_meta = import_by_path(component_class_path)
            if hasattr(cls_for_meta, 'METADATA_DEFINITION') and isinstance(cls_for_meta.METADATA_DEFINITION, ComponentMetadata):
                base_metadata_for_class = cls_for_meta.METADATA_DEFINITION
        except Exception as e:
            logger.debug(f"Could not retrieve METADATA_DEFINITION from class {component_class_path} for '{component_id}': {e}")
    
    if not base_metadata_for_class:
        logger.warning(
            f"No default or class-defined metadata found for '{target_class_or_factory_key}'. "
            f"Using minimal fallback for component '{component_id}'."
        )
        base_metadata_for_class = ComponentMetadata(
            id=component_id, name=component_id, version='0.1.0', category=component_type,
            description=f"Auto-generated metadata for {component_type} '{component_id}'"
        )

    try:
        final_instance_metadata = _build_component_metadata(base_metadata_for_class, component_id, comp_def)
        current_component_metadata_for_reporting = final_instance_metadata # Update for more accurate reporting
    except Exception as e_meta:
        errmsg = f"Error constructing ComponentMetadata for simple component '{component_id}': {e_meta}"
        logger.error(errmsg, exc_info=True)
        health_reporter.add_component_status(
            component_id, 
            ComponentStatus.METADATA_CONSTRUCTION_ERROR, 
            current_component_metadata_for_reporting, # Use what we have
            [errmsg]
        )
        if is_strict_mode:
            raise BootstrapError(errmsg, component_id=component_id) from e_meta
        return
        
    pydantic_class_defaults = {}
    component_class_for_pydantic: Optional[Type] = None
    if component_class_path:
        try:
            component_class_for_pydantic = import_by_path(component_class_path)
            pydantic_class_defaults = get_pydantic_defaults(component_class_for_pydantic, final_instance_metadata.name)
        except Exception as e_pydantic:
            logger.debug(f"Could not get Pydantic defaults for simple component '{component_id}' (class: {component_class_path}): {e_pydantic}")
    
    # Load YAML config from path if 'config' is a string, or use if 'config' is a dict
    yaml_loaded_config = {}
    if isinstance(yaml_config_source, str): # It's a path template
        actual_config_path_str = yaml_config_source.replace('{id}', component_id)
        yaml_loaded_config = load_yaml_robust(Path(actual_config_path_str))
    elif isinstance(yaml_config_source, dict): # It's an inline config dict
        yaml_loaded_config = yaml_config_source

    # Config hierarchy: Pydantic/Class Defaults < YAML File/Inline < Manifest Override
    merged_config_step1 = ConfigMerger.merge(pydantic_class_defaults, yaml_loaded_config, f'{component_id}_defaults_yaml')
    final_config = ConfigMerger.merge(merged_config_step1, inline_config_override, f'{component_id}_final_config')

    if validation_data_store and hasattr(validation_data_store, 'store_component_data'):
        validation_data_store.store_component_data(
            component_id=component_id,
            original_metadata=final_instance_metadata, 
            resolved_config=final_config,
            manifest_spec=comp_def
        )

    instance: Optional[NireonBaseComponent] = None
    try:
        if component_class_path:
            # component_class_for_pydantic would be the same as cls_to_instantiate if path is valid
            cls_to_instantiate = component_class_for_pydantic or import_by_path(component_class_path)
            if not issubclass(cls_to_instantiate, NireonBaseComponent):
                 raise BootstrapError(f"Class {component_class_path} for '{component_id}' is not a NireonBaseComponent.")

            # Common deps for simple components are typically not passed this way,
            # but create_component_instance expects the arg.
            instance = await create_component_instance(cls_to_instantiate, final_config, component_id, final_instance_metadata, common_deps=None)

        elif factory_key and component_type == 'mechanism' and mechanism_factory:
            # Mechanisms might have specific factory logic
            instance = mechanism_factory.create_mechanism(factory_key, final_instance_metadata, final_config)
        else:
            errmsg = (f"Cannot determine instantiation method for simple component '{component_id}'. "
                      f"Need 'class' path or a 'factory_key' for a known type (e.g., mechanism).")
            raise BootstrapError(errmsg, component_id=component_id)

        if instance is None: # Should be caught by create_component_instance error, but safeguard.
            errmsg = f"Component instantiation returned None for simple component '{component_id}' (target: {target_class_or_factory_key})"
            raise BootstrapError(errmsg, component_id=component_id)

        logger.info(f"Instantiated simple component '{component_id}' (Type: {component_type}, Class/FactoryKey: {target_class_or_factory_key})")
        
        # Registration using the helper from processors.service_resolver
        _safe_register_service_instance(
            registry, 
            type(instance), 
            instance, 
            component_id, # This is the canonical ID
            final_instance_metadata.category, # Use the resolved metadata's category
            description_for_meta=final_instance_metadata.description,
            requires_initialize_override=final_instance_metadata.requires_initialize
        )
        health_reporter.add_component_status(component_id, ComponentStatus.INSTANCE_REGISTERED, instance.metadata, [])

    except Exception as e:
        errmsg = f"Error during instantiation/registration of simple component '{component_id}': {e}"
        logger.error(errmsg, exc_info=True)
        health_reporter.add_component_status(
            component_id, 
            ComponentStatus.BOOTSTRAP_ERROR, 
            current_component_metadata_for_reporting, # Use the most accurate metadata we have
            [errmsg]
        )
        if is_strict_mode:
            # Convert to BootstrapError if not already, to ensure consistent error type
            if not isinstance(e, BootstrapError):
                raise BootstrapError(errmsg, component_id=component_id) from e
            raise # Re-raise if already BootstrapError
        # In non-strict mode, we log the error and continue if possible.


def _build_component_metadata(
    base_metadata: ComponentMetadata, 
    component_id_from_manifest: str, 
    manifest_comp_definition: Dict[str, Any]
) -> ComponentMetadata:
    """
    Builds the final ComponentMetadata for an instance, applying overrides from the manifest.
    This version is specific to how simple_component_processor uses it.
    """
    if not isinstance(base_metadata, ComponentMetadata):
        # This should ideally not happen if get_default_metadata always returns ComponentMetadata or None (handled by caller)
        raise TypeError(f"base_metadata for '{component_id_from_manifest}' must be ComponentMetadata, got {type(base_metadata)}")

    # Start with a copy of the base_metadata
    instance_metadata_dict = dataclasses.asdict(base_metadata)
    
    # Crucially, set the ID to the one from the manifest component definition
    instance_metadata_dict['id'] = component_id_from_manifest
    
    # Apply overrides from 'metadata_override' section in manifest
    manifest_meta_override = manifest_comp_definition.get('metadata_override', {})
    if manifest_meta_override:
        logger.debug(f'[{component_id_from_manifest}] Applying manifest metadata_override: {manifest_meta_override}')
        for key, value in manifest_meta_override.items():
            if key in instance_metadata_dict:
                instance_metadata_dict[key] = value
            # Handle 'requires_initialize' specifically if it's a common override point
            elif key == 'requires_initialize' and isinstance(value, bool):
                instance_metadata_dict[key] = value
            else:
                logger.warning(
                    f"[{component_id_from_manifest}] Unknown key '{key}' or invalid type in metadata_override. "
                    f"Key not present in base metadata fields. Ignoring."
                )
    
    # Apply epistemic_tags if directly specified in the component definition (not inside metadata_override)
    if 'epistemic_tags' in manifest_comp_definition:
        tags = manifest_comp_definition['epistemic_tags']
        if isinstance(tags, list) and all(isinstance(t, str) for t in tags):
            instance_metadata_dict['epistemic_tags'] = tags
            logger.debug(f'[{component_id_from_manifest}] Using epistemic_tags from manifest: {tags}')
        else:
            logger.warning(
                f"[{component_id_from_manifest}] Invalid 'epistemic_tags' in manifest, expected list of strings. "
                f"Current tags from base: {instance_metadata_dict.get('epistemic_tags')}"
            )
            
    # Ensure 'requires_initialize' is set, prioritizing manifest override, then base metadata.
    # If 'requires_initialize' was in metadata_override, it's already set.
    # If not, but it's directly in comp_def (less standard), consider it.
    # Otherwise, ensure it defaults to base_metadata.requires_initialize.
    if 'requires_initialize' not in manifest_meta_override: # If not in specific override block
        if 'requires_initialize' in manifest_comp_definition and isinstance(manifest_comp_definition['requires_initialize'], bool):
            instance_metadata_dict['requires_initialize'] = manifest_comp_definition['requires_initialize']
        else: # Default to what base_metadata had (which should already be in instance_metadata_dict)
            instance_metadata_dict.setdefault('requires_initialize', base_metadata.requires_initialize)
            
    try:
        return ComponentMetadata(**instance_metadata_dict)
    except TypeError as e:
        # This can happen if instance_metadata_dict contains keys not in ComponentMetadata
        # or if types are wrong after overrides.
        logger.error(f"Failed to create ComponentMetadata for '{component_id_from_manifest}' due to TypeError: {e}. Data: {instance_metadata_dict}")
        raise BootstrapError(f"Metadata construction error for '{component_id_from_manifest}': {e}", component_id=component_id_from_manifest) from e
```
--- END OF FILE nireon_v4/bootstrap/processors/component_processor.py ---

--- START OF FILE nireon_v4/bootstrap/processors/shared_service_processor.py ---
```python
# nireon_v4/bootstrap/processors/shared_service_processor.py
from __future__ import absolute_import

# Import from component_processing_utils (which should be minimal now or gone)
# or directly the necessary standard library and project imports.
import asyncio
import dataclasses
import inspect
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from core.base_component import NireonBaseComponent
from core.lifecycle import ComponentMetadata, ComponentRegistryMissingError
from core.registry.component_registry import ComponentRegistry
from domain.ports.event_bus_port import EventBusPort
from bootstrap.exceptions import BootstrapError
from bootstrap.health.reporter import HealthReporter as BootstrapHealthReporter, ComponentStatus
# from bootstrap.bootstrap_helper.metadata import get_default_metadata # Now from utils
# from bootstrap.processors.metadata import get_default_metadata # If this was used
from runtime.utils import import_by_path, load_yaml_robust

# Import from the new utils module
from bootstrap.utils.component_utils import (
    create_component_instance, # Using the public name
    get_pydantic_defaults,   # Using the public name
)
# Import from local service_resolver if that's where _safe_register_service_instance_with_port now resides
from .service_resolver import _safe_register_service_instance_with_port


try:
    from configs.config_utils import ConfigMerger
except ImportError:
    logger = logging.getLogger(__name__) # Define logger if not already defined
    logger.warning("ConfigMerger not found in configs.config_utils, using placeholder.")
    class ConfigMerger: # Basic placeholder
        @staticmethod
        def merge(dict1: Dict, dict2: Dict, context_name: str) -> Dict:
            logger.debug(f"Using placeholder ConfigMerger for: {context_name}")
            result = dict1.copy()
            result.update(dict2)
            return result

logger = logging.getLogger(__name__)


async def instantiate_shared_service(
    service_key_in_manifest: str,
    service_spec_from_manifest: Dict[str, Any],
    registry: ComponentRegistry,
    event_bus: EventBusPort, # Actual EventBusPort, not placeholder type
    global_app_config: Dict[str, Any],
    health_reporter: BootstrapHealthReporter,
    validation_data_store: Any # e.g., BootstrapValidationData
) -> None:
    class_path = service_spec_from_manifest.get('class')
    # Config can be a path string, an inline dict, or missing
    config_source = service_spec_from_manifest.get('config')
    inline_config_override = service_spec_from_manifest.get('config_override', {})
    port_type_str = service_spec_from_manifest.get('port_type') # String path to the port type
    is_enabled = service_spec_from_manifest.get('enabled', True)
    is_strict_mode = global_app_config.get('bootstrap_strict_mode', True)

    service_name_for_report = class_path.split(':')[-1] if class_path and ':' in class_path else \
                              class_path.split('.')[-1] if class_path else service_key_in_manifest
    
    # Base metadata for reporting, will be refined.
    base_meta_for_report = ComponentMetadata(
        id=service_key_in_manifest, 
        name=service_name_for_report, 
        category='shared_service', 
        version='0.1.0' # Default, may be overridden
    )

    if not class_path:
        msg = f"Shared service '{service_key_in_manifest}' definition missing 'class' path."
        logger.error(msg)
        health_reporter.add_component_status(service_key_in_manifest, ComponentStatus.DEFINITION_ERROR, base_meta_for_report, [msg])
        if is_strict_mode:
            raise BootstrapError(msg, component_id=service_key_in_manifest)
        return

    if not is_enabled:
        logger.info(f"Shared service '{service_key_in_manifest}' (Class: {class_path}) is disabled. Skipping.")
        health_reporter.add_component_status(service_key_in_manifest, ComponentStatus.DISABLED, base_meta_for_report, ['Disabled in manifest'])
        return

    # Check if already registered (e.g., by AbiogenesisPhase or another manifest entry)
    try:
        existing_instance = registry.get(service_key_in_manifest) # Check by manifest key first
        if existing_instance is not None:
            logger.info(f"Service '{service_key_in_manifest}' (Class: {class_path}) already in registry by key. Skipping manifest instantiation.")
            return
    except (KeyError, ComponentRegistryMissingError): # More specific exception
        logger.debug(f"Service '{service_key_in_manifest}' not found in registry by key. Proceeding with instantiation.")
    
    logger.info(f'-> Instantiating Shared Service from manifest: {service_key_in_manifest} (Class: {class_path})')

    try:
        service_class = import_by_path(class_path)
    except ImportError as e_import:
        msg = f"Failed to import class '{class_path}' for shared service '{service_key_in_manifest}': {e_import}"
        logger.error(msg, exc_info=True)
        health_reporter.add_component_status(service_key_in_manifest, ComponentStatus.DEFINITION_ERROR, base_meta_for_report, [msg])
        if is_strict_mode:
            raise BootstrapError(msg, component_id=service_key_in_manifest) from e_import
        return

    # Get Pydantic/DEFAULT_CONFIG defaults from the class
    pydantic_class_defaults = get_pydantic_defaults(service_class, service_name_for_report)
    
    yaml_loaded_config = {}
    if isinstance(config_source, str): # Path template
        actual_config_path = config_source.replace('{id}', service_key_in_manifest)
        yaml_loaded_config = load_yaml_robust(Path(actual_config_path))
        if not yaml_loaded_config and Path(actual_config_path).is_file() and Path(actual_config_path).read_text(encoding='utf-8').strip():
            # File exists and is not empty but couldn't be parsed
            msg = f"Failed to parse non-empty config YAML '{actual_config_path}' for service '{service_key_in_manifest}'."
            logger.error(msg) # This might be a warning or error depending on strictness for config issues
    elif isinstance(config_source, dict): # Inline config
        yaml_loaded_config = config_source
        logger.debug(f"Using inline config from manifest for service '{service_key_in_manifest}'")
    elif config_source is not None: # Invalid type
        logger.warning(f"Unexpected 'config' type for service '{service_key_in_manifest}': {type(config_source)}. Expected string path or dict. Ignoring.")

    # Config hierarchy: Pydantic/Class Defaults < YAML/Inline Config < Manifest Override (`config_override`)
    merged_config_step1 = ConfigMerger.merge(pydantic_class_defaults, yaml_loaded_config, f'{service_key_in_manifest}_defaults_yaml')
    final_service_config = ConfigMerger.merge(merged_config_step1, inline_config_override, f'{service_key_in_manifest}_final_config')

    # Determine the definitive metadata for the instance
    service_instance_metadata = base_meta_for_report # Start with fallback
    if hasattr(service_class, 'METADATA_DEFINITION') and isinstance(service_class.METADATA_DEFINITION, ComponentMetadata):
        service_instance_metadata = dataclasses.replace(service_class.METADATA_DEFINITION, id=service_key_in_manifest)
    # Allow manifest spec to override parts of this resolved metadata
    manifest_meta_override = service_spec_from_manifest.get('metadata_override', {})
    if manifest_meta_override:
        current_meta_dict = dataclasses.asdict(service_instance_metadata)
        current_meta_dict.update(manifest_meta_override)
        try:
            service_instance_metadata = ComponentMetadata(**current_meta_dict)
        except Exception as e_meta_override:
             logger.warning(f"Error applying metadata_override for '{service_key_in_manifest}': {e_meta_override}. Using metadata before override.")


    if validation_data_store and hasattr(validation_data_store, 'store_component_data'):
        validation_data_store.store_component_data(
            component_id=service_key_in_manifest,
            original_metadata=service_instance_metadata,
            resolved_config=final_service_config,
            manifest_spec=service_spec_from_manifest
        )

    service_instance: Optional[Any] = None
    try:
        common_deps_for_service: Optional[Dict[str, Any]] = None # Shared services usually don't take full CommonMechanismDependencies
        # but might take individual core services like event_bus or registry.
        # create_component_instance handles 'config', 'metadata_definition'.
        # For other deps, they must be explicitly listed in service_class.__init__ signature.
        
        # Build a dictionary of potential dependencies the service might need
        # This is simpler than full CommonMechanismDependencies object
        potential_deps_map = {
            'event_bus': event_bus,
            'registry': registry,
            # Add other specific, commonly injected dependencies for services if any
        }
        
        if issubclass(service_class, NireonBaseComponent):
            service_instance = await create_component_instance(
                component_class=service_class,
                resolved_config_for_instance=final_service_config,
                instance_id=service_key_in_manifest,
                instance_metadata_object=service_instance_metadata,
                common_deps=potential_deps_map # Pass the map of potential deps
            )
        else: # For non-NireonBaseComponent services (plain classes)
            ctor_params = inspect.signature(service_class.__init__).parameters
            kwargs_for_ctor: Dict[str, Any] = {}
            if 'config' in ctor_params:
                kwargs_for_ctor['config'] = final_service_config
            elif 'cfg' in ctor_params: # Alternative name
                kwargs_for_ctor['cfg'] = final_service_config
            
            # Inject other known dependencies if constructor expects them
            for dep_name, dep_instance in potential_deps_map.items():
                if dep_name in ctor_params:
                    kwargs_for_ctor[dep_name] = dep_instance
            
            service_instance = service_class(**kwargs_for_ctor)

    except Exception as e_create:
        msg = f"Instantiation failed for shared service '{service_key_in_manifest}': {e_create}"
        logger.error(msg, exc_info=True)
        health_reporter.add_component_status(service_key_in_manifest, ComponentStatus.INSTANTIATION_ERROR, service_instance_metadata, [msg])
        if is_strict_mode:
            # Convert to BootstrapError if not already
            if not isinstance(e_create, BootstrapError):
                raise BootstrapError(msg, component_id=service_key_in_manifest) from e_create
            raise # Re-raise if already BootstrapError
        return

    if service_instance:
        # Determine requires_initialize: from metadata, or False if not NireonBaseComponent
        final_requires_init = False
        if isinstance(service_instance, NireonBaseComponent):
            final_requires_init = getattr(service_instance.metadata, 'requires_initialize', False)
        
        _safe_register_service_instance_with_port(
            registry,
            service_class,
            service_instance,
            service_key_in_manifest, # Use the manifest key as the primary ID
            service_instance_metadata.category,
            port_type=port_type_str, # String path to port type
            description_for_meta=service_instance_metadata.description,
            requires_initialize_override=final_requires_init
        )
        logger.info(
            f'✓ Shared Service Instantiated and Registered: {service_key_in_manifest} -> {service_class.__name__} '
            f'(Category: {service_instance_metadata.category}, Requires Init: {final_requires_init})'
        )
        health_reporter.add_component_status(service_key_in_manifest, ComponentStatus.INSTANCE_REGISTERED, service_instance_metadata, [])
    else: # Should be caught by create_component_instance raising an error, but as safeguard.
        msg = f"Service instantiation returned None for '{service_key_in_manifest}' using class {service_class.__name__}"
        logger.error(msg)
        health_reporter.add_component_status(service_key_in_manifest, ComponentStatus.INSTANTIATION_ERROR, service_instance_metadata, [msg])
        if is_strict_mode:
            raise BootstrapError(msg, component_id=service_key_in_manifest)
