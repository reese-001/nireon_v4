# C:\Users\erees\Documents\development\nireon_v4\bootstrap\signals\bootstrap_signals.py
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from domain.ports.event_bus_port import EventBusPort
logger = logging.getLogger(__name__)
SYSTEM_BOOTSTRAPPED = 'SYSTEM_BOOTSTRAPPED'
SYSTEM_INITIALIZATION_STARTED = 'SYSTEM_INITIALIZATION_STARTED'
SYSTEM_SHUTDOWN_INITIATED = 'SYSTEM_SHUTDOWN_INITIATED'
L0_ABIOGENESIS_COMPLETE = 'L0_ABIOGENESIS_COMPLETE'
L0_EMERGENCE_STARTED = 'L0_EMERGENCE_STARTED'
CORE_SERVICES_EMERGED = 'CORE_SERVICES_EMERGED'
FEATURE_FLAGS_INITIALIZED = 'FEATURE_FLAGS_INITIALIZED'
COMPONENT_REGISTRY_INITIALIZED = 'COMPONENT_REGISTRY_INITIALIZED'
COMPONENT_REGISTERED = 'COMPONENT_REGISTERED'
COMPONENT_INITIALIZATION_STARTED = 'COMPONENT_INITIALIZATION_STARTED'
COMPONENT_INITIALIZATION_COMPLETE = 'COMPONENT_INITIALIZATION_COMPLETE'
COMPONENT_VALIDATION_STARTED = 'COMPONENT_VALIDATION_STARTED'
COMPONENT_VALIDATION_COMPLETE = 'COMPONENT_VALIDATION_COMPLETE'
COMPONENT_HEALTH_CHECK_COMPLETE = 'COMPONENT_HEALTH_CHECK_COMPLETE'
FACTORIES_INITIALIZED = 'FACTORIES_INITIALIZED'
MECHANISM_FACTORY_READY = 'MECHANISM_FACTORY_READY'
INTERFACE_VALIDATOR_READY = 'INTERFACE_VALIDATOR_READY'
DEPENDENCY_INJECTION_READY = 'DEPENDENCY_INJECTION_READY'
CONFIGURATION_LOADED = 'CONFIGURATION_LOADED'
MANIFEST_PROCESSING_STARTED = 'MANIFEST_PROCESSING_STARTED'
MANIFEST_PROCESSING_COMPLETE = 'MANIFEST_PROCESSING_COMPLETE'
MANIFEST_VALIDATION_COMPLETE = 'MANIFEST_VALIDATION_COMPLETE'
SCHEMA_VALIDATION_COMPLETE = 'SCHEMA_VALIDATION_COMPLETE'
SHARED_SERVICE_INSTANTIATED = 'SHARED_SERVICE_INSTANTIATED'
MECHANISM_INSTANTIATED = 'MECHANISM_INSTANTIATED'
OBSERVER_INSTANTIATED = 'OBSERVER_INSTANTIATED'
MANAGER_INSTANTIATED = 'MANAGER_INSTANTIATED'
ORCHESTRATION_COMMAND_REGISTERED = 'ORCHESTRATION_COMMAND_REGISTERED'
RBAC_POLICIES_LOADED = 'RBAC_POLICIES_LOADED'
RBAC_SETUP_COMPLETE = 'RBAC_SETUP_COMPLETE'
SECURITY_POLICIES_APPLIED = 'SECURITY_POLICIES_APPLIED'
BOOTSTRAP_ERROR_OCCURRED = 'BOOTSTRAP_ERROR_OCCURRED'
BOOTSTRAP_WARNING_ISSUED = 'BOOTSTRAP_WARNING_ISSUED'
COMPONENT_INITIALIZATION_FAILED = 'COMPONENT_INITIALIZATION_FAILED'
COMPONENT_VALIDATION_FAILED = 'COMPONENT_VALIDATION_FAILED'
MANIFEST_SCHEMA_VALIDATION_FAILED = 'MANIFEST_SCHEMA_VALIDATION_FAILED'
BOOTSTRAP_HEALTH_CHECK_STARTED = 'BOOTSTRAP_HEALTH_CHECK_STARTED'
BOOTSTRAP_HEALTH_CHECK_COMPLETE = 'BOOTSTRAP_HEALTH_CHECK_COMPLETE'
COMPONENT_STATUS_UPDATED = 'COMPONENT_STATUS_UPDATED'
BOOTSTRAP_METRICS_COLLECTED = 'BOOTSTRAP_METRICS_COLLECTED'
ABIOGENESIS_PHASE_COMPLETE = 'ABIOGENESIS_PHASE_COMPLETE'
REGISTRY_SETUP_PHASE_COMPLETE = 'REGISTRY_SETUP_PHASE_COMPLETE'
FACTORY_SETUP_PHASE_COMPLETE = 'FACTORY_SETUP_PHASE_COMPLETE'
MANIFEST_PHASE_COMPLETE = 'MANIFEST_PHASE_COMPLETE'
INITIALIZATION_PHASE_COMPLETE = 'INITIALIZATION_PHASE_COMPLETE'
VALIDATION_PHASE_COMPLETE = 'VALIDATION_PHASE_COMPLETE'
RBAC_PHASE_COMPLETE = 'RBAC_PHASE_COMPLETE'
BOOTSTRAP_ADAPTATION_TRIGGERED = 'BOOTSTRAP_ADAPTATION_TRIGGERED'
CONFIGURATION_UPDATED = 'CONFIGURATION_UPDATED'
COMPONENT_RECONFIGURED = 'COMPONENT_RECONFIGURED'
BOOTSTRAP_DEBUG_INFO = 'BOOTSTRAP_DEBUG_INFO'
COMPONENT_METADATA_RESOLVED = 'COMPONENT_METADATA_RESOLVED'
CONFIG_HIERARCHY_RESOLVED = 'CONFIG_HIERARCHY_RESOLVED'
DEPENDENCY_GRAPH_CONSTRUCTED = 'DEPENDENCY_GRAPH_CONSTRUCTED'
TEMPLATE_REGISTERED = 'TEMPLATE_REGISTERED'
RULE_REGISTERED = 'RULE_REGISTERED'
REACTOR_RULES_LOADED = 'REACTOR_RULES_LOADED'
PLUGIN_DISCOVERED = 'PLUGIN_DISCOVERED'
EXTENSION_LOADED = 'EXTENSION_LOADED'
HOT_RELOAD_TRIGGERED = 'HOT_RELOAD_TRIGGERED'
SYSTEM_LIFECYCLE_SIGNALS = {SYSTEM_BOOTSTRAPPED, SYSTEM_INITIALIZATION_STARTED, SYSTEM_SHUTDOWN_INITIATED}
ABIOGENESIS_SIGNALS = {L0_ABIOGENESIS_COMPLETE, L0_EMERGENCE_STARTED, CORE_SERVICES_EMERGED, FEATURE_FLAGS_INITIALIZED}
COMPONENT_LIFECYCLE_SIGNALS = {COMPONENT_REGISTERED, COMPONENT_INITIALIZATION_STARTED, COMPONENT_INITIALIZATION_COMPLETE, COMPONENT_VALIDATION_STARTED, COMPONENT_VALIDATION_COMPLETE, COMPONENT_HEALTH_CHECK_COMPLETE}
ERROR_SIGNALS = {BOOTSTRAP_ERROR_OCCURRED, BOOTSTRAP_WARNING_ISSUED, COMPONENT_INITIALIZATION_FAILED, COMPONENT_VALIDATION_FAILED, MANIFEST_SCHEMA_VALIDATION_FAILED}
PHASE_COMPLETION_SIGNALS = {ABIOGENESIS_PHASE_COMPLETE, REGISTRY_SETUP_PHASE_COMPLETE, FACTORY_SETUP_PHASE_COMPLETE, MANIFEST_PHASE_COMPLETE, INITIALIZATION_PHASE_COMPLETE, VALIDATION_PHASE_COMPLETE, RBAC_PHASE_COMPLETE}
DEBUG_SIGNALS = {BOOTSTRAP_DEBUG_INFO, COMPONENT_METADATA_RESOLVED, CONFIG_HIERARCHY_RESOLVED, DEPENDENCY_GRAPH_CONSTRUCTED}
ALL_BOOTSTRAP_SIGNALS = {SYSTEM_BOOTSTRAPPED, SYSTEM_INITIALIZATION_STARTED, SYSTEM_SHUTDOWN_INITIATED, L0_ABIOGENESIS_COMPLETE, L0_EMERGENCE_STARTED, CORE_SERVICES_EMERGED, FEATURE_FLAGS_INITIALIZED, COMPONENT_REGISTRY_INITIALIZED, COMPONENT_REGISTERED, COMPONENT_INITIALIZATION_STARTED, COMPONENT_INITIALIZATION_COMPLETE, COMPONENT_VALIDATION_STARTED, COMPONENT_VALIDATION_COMPLETE, COMPONENT_HEALTH_CHECK_COMPLETE, FACTORIES_INITIALIZED, MECHANISM_FACTORY_READY, INTERFACE_VALIDATOR_READY, DEPENDENCY_INJECTION_READY, CONFIGURATION_LOADED, MANIFEST_PROCESSING_STARTED, MANIFEST_PROCESSING_COMPLETE, MANIFEST_VALIDATION_COMPLETE, SCHEMA_VALIDATION_COMPLETE, SHARED_SERVICE_INSTANTIATED, MECHANISM_INSTANTIATED, OBSERVER_INSTANTIATED, MANAGER_INSTANTIATED, ORCHESTRATION_COMMAND_REGISTERED, RBAC_POLICIES_LOADED, RBAC_SETUP_COMPLETE, SECURITY_POLICIES_APPLIED, BOOTSTRAP_ERROR_OCCURRED, BOOTSTRAP_WARNING_ISSUED, COMPONENT_INITIALIZATION_FAILED, COMPONENT_VALIDATION_FAILED, MANIFEST_SCHEMA_VALIDATION_FAILED, BOOTSTRAP_HEALTH_CHECK_STARTED, BOOTSTRAP_HEALTH_CHECK_COMPLETE, COMPONENT_STATUS_UPDATED, BOOTSTRAP_METRICS_COLLECTED, ABIOGENESIS_PHASE_COMPLETE, REGISTRY_SETUP_PHASE_COMPLETE, FACTORY_SETUP_PHASE_COMPLETE, MANIFEST_PHASE_COMPLETE, INITIALIZATION_PHASE_COMPLETE, VALIDATION_PHASE_COMPLETE, RBAC_PHASE_COMPLETE, BOOTSTRAP_ADAPTATION_TRIGGERED, CONFIGURATION_UPDATED, COMPONENT_RECONFIGURED, BOOTSTRAP_DEBUG_INFO, COMPONENT_METADATA_RESOLVED, CONFIG_HIERARCHY_RESOLVED, DEPENDENCY_GRAPH_CONSTRUCTED, TEMPLATE_REGISTERED, RULE_REGISTERED, REACTOR_RULES_LOADED, PLUGIN_DISCOVERED, EXTENSION_LOADED, HOT_RELOAD_TRIGGERED}
def is_error_signal(signal_type: str) -> bool:
    return signal_type in ERROR_SIGNALS
def is_lifecycle_signal(signal_type: str) -> bool:
    return signal_type in SYSTEM_LIFECYCLE_SIGNALS or signal_type in COMPONENT_LIFECYCLE_SIGNALS or signal_type in ABIOGENESIS_SIGNALS
def is_phase_completion_signal(signal_type: str) -> bool:
    return signal_type in PHASE_COMPLETION_SIGNALS
def get_signal_category(signal_type: str) -> str:
    if signal_type in SYSTEM_LIFECYCLE_SIGNALS:
        return 'system_lifecycle'
    elif signal_type in ABIOGENESIS_SIGNALS:
        return 'abiogenesis'
    elif signal_type in COMPONENT_LIFECYCLE_SIGNALS:
        return 'component_lifecycle'
    elif signal_type in ERROR_SIGNALS:
        return 'error'
    elif signal_type in PHASE_COMPLETION_SIGNALS:
        return 'phase_completion'
    elif signal_type in DEBUG_SIGNALS:
        return 'debug'
    else:
        return 'unknown'
def validate_signal_type(signal_type: str) -> bool:
    return signal_type in ALL_BOOTSTRAP_SIGNALS
class BootstrapSignalEmitter:
    def __init__(self, event_bus: Optional[EventBusPort], run_id: str):
        self.event_bus = event_bus
        self.run_id = run_id
        self.signal_count = 0
        if not event_bus:
            logger.warning('No event bus provided to BootstrapSignalEmitter - signals will be logged only')
        logger.debug(f'BootstrapSignalEmitter initialized for run_id: {run_id}')
    async def emit_signal(self, signal_type: str, payload: Dict[str, Any]) -> None:
        self.signal_count += 1
        if not validate_signal_type(signal_type):
            logger.warning(f'Unknown signal type: {signal_type}')
        enhanced_payload = {'signal_type': signal_type, 'run_id': self.run_id, 'signal_sequence': self.signal_count, 'category': get_signal_category(signal_type), 'timestamp': datetime.now(timezone.utc).isoformat(), **payload}
        log_level = logging.ERROR if is_error_signal(signal_type) else logging.DEBUG
        logger.log(log_level, f"Bootstrap signal [{signal_type}]: {payload.get('message', '')}")
        if self.event_bus:
            try:
                # Assuming publish is synchronous for now, if it's async, it should be awaited.
                # Based on PlaceholderEventBusImpl, it's synchronous.
                self.event_bus.publish(signal_type, enhanced_payload)
            except Exception as e:
                logger.error(f'Failed to emit signal {signal_type}: {e}')

    async def emit_bootstrap_started(self) -> None: # Added this method
        await self.emit_signal(SYSTEM_INITIALIZATION_STARTED, {'message': f'NIREON V4 system bootstrap process started for run_id: {self.run_id}'})

    async def emit_bootstrap_completed(self, component_count: int, duration_seconds: float) -> None: # Renamed from emit_system_bootstrapped
        await self.emit_signal(SYSTEM_BOOTSTRAPPED, {'component_count': component_count, 'duration_seconds': duration_seconds, 'run_id': self.run_id, 'message': f'NIREON V4 system bootstrap completed with {component_count} components in {duration_seconds:.2f}s'})
    async def emit_component_registered(self, component_id: str, category: str) -> None:
        await self.emit_signal(COMPONENT_REGISTERED, {'component_id': component_id, 'category': category, 'message': f'Component {component_id} registered'})
    async def emit_phase_complete(self, phase_name: str, success: bool, **metadata) -> None:
        phase_signal_map = {'AbiogenesisPhase': ABIOGENESIS_PHASE_COMPLETE, 'RegistrySetupPhase': REGISTRY_SETUP_PHASE_COMPLETE, 'FactorySetupPhase': FACTORY_SETUP_PHASE_COMPLETE, 'ManifestProcessingPhase': MANIFEST_PHASE_COMPLETE, 'ComponentInitializationPhase': INITIALIZATION_PHASE_COMPLETE, 'InterfaceValidationPhase': VALIDATION_PHASE_COMPLETE, 'RBACSetupPhase': RBAC_PHASE_COMPLETE}
        signal_type = phase_signal_map.get(phase_name, 'UNKNOWN_PHASE_COMPLETE') # Should probably have a generic phase complete signal
        if signal_type == 'UNKNOWN_PHASE_COMPLETE' and phase_name: # Log if a new phase name is used
             logger.warning(f"Emitting phase complete for an unknown phase name: {phase_name}. Consider adding it to phase_signal_map.")
        await self.emit_signal(signal_type, {'phase_name': phase_name, 'success': success, 'message': f"Phase {phase_name} {('completed' if success else 'failed')}", **metadata})
    async def emit_error(self, error_type: str, component_id: Optional[str], error_message: str) -> None:
        await self.emit_signal(BOOTSTRAP_ERROR_OCCURRED, {'error_type': error_type, 'component_id': component_id, 'error_message': error_message, 'message': f'Bootstrap error: {error_message}'})
    async def emit_warning(self, warning_type: str, component_id: Optional[str], warning_message: str) -> None:
        await self.emit_signal(BOOTSTRAP_WARNING_ISSUED, {'warning_type': warning_type, 'component_id': component_id, 'warning_message': warning_message, 'message': f'Bootstrap warning: {warning_message}'})
    def get_signal_stats(self) -> Dict[str, Any]:
        return {'total_signals_emitted': self.signal_count, 'run_id': self.run_id, 'event_bus_available': self.event_bus is not None}