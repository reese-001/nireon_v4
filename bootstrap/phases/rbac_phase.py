from __future__ import annotations
import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING
import yaml
from pydantic import BaseModel, Field, validator
from jsonschema import validate as js_validate, ValidationError as SchemaError

from bootstrap.phases.base_phase import BootstrapPhase, PhaseResult
from runtime.utils import load_yaml_robust
from core.base_component import NireonBaseComponent
from core.lifecycle import ComponentMetadata
from core.results import ProcessResult
from domain.context import NireonExecutionContext
from bootstrap.bootstrap_helper.context_helper import create_context_builder, SimpleConfigProvider

# Use TYPE_CHECKING to avoid circular imports
if TYPE_CHECKING:
    from bootstrap.bootstrap_context import BootstrapContext

logger = logging.getLogger(__name__)


class RBACRule(BaseModel):
    id: str = Field(..., pattern=r'^[A-Za-z0-9_\-]+$')
    subjects: List[str]
    resources: List[str]
    actions: List[str]
    effect: str = Field(..., pattern=r'^(allow|deny)$')
    description: Optional[str] = None

    @validator('actions', 'effect', pre=True, always=True)
    def _normalize_lowercase(cls, v):
        return [a.lower() for a in v] if isinstance(v, list) else v.lower()

    @validator('subjects', 'resources', pre=True, always=True)
    def _normalize_lists(cls, v):
        if isinstance(v, str):
            return [v]
        return v if isinstance(v, list) else []


class RBACPolicySet(BaseModel):
    version: str = '1.0'
    metadata: Optional[Dict[str, Any]] = {}
    rules: List[RBACRule] = []
    policies: Optional[List[Dict[str, Any]]] = None

    def __init__(self, **data):
        if 'policies' in data and not data.get('rules'):
            legacy_policies = data.pop('policies', [])
            rules = []
            for i, policy in enumerate(legacy_policies):
                rule_data = {
                    'id': policy.get('id', f'legacy_rule_{i}'),
                    'subjects': [policy.get('role', 'unknown')],
                    'resources': policy.get('resources', ['*']),
                    'actions': policy.get('permissions', ['read']),
                    'effect': policy.get('effect', 'allow'),
                    'description': policy.get('description')
                }
                rules.append(RBACRule(**rule_data))
            data['rules'] = rules
        super().__init__(**data)

    def get_rules_for_subject(self, subject: str) -> List[RBACRule]:
        return self.policy_set.get_rules_for_subject(subject)

    def get_all_subjects(self) -> List[str]:
        subjects = set()
        for rule in self.rules:
            subjects.update(rule.subjects)
        return list(subjects)


class PolicySetComponent(NireonBaseComponent):
    def __init__(self, metadata: ComponentMetadata, policy_set: RBACPolicySet, source_file: Optional[Path] = None):
        super().__init__(config={}, metadata_definition=metadata)
        self.policy_set = policy_set
        self.source_file = source_file
        self.loaded_at = datetime.now(timezone.utc)

    async def _process_impl(self, data: Any, context: 'NireonExecutionContext') -> ProcessResult:
        logger.debug(f"PolicySetComponent '{self.component_id}' received process call, but does not actively process data.")
        return ProcessResult(success=True, component_id=self.component_id, message="PolicySetComponent does not process data.")
    
    async def _initialize_impl(self, context):
        logger.debug(f'Initializing PolicySetComponent {self.component_id} with V2 context')
        return

    async def _shutdown_impl(self, context):
        logger.debug(f'Shutting down PolicySetComponent {self.component_id} with V2 context')
        return

    def get_rules_for_subject(self, subject: str) -> List[RBACRule]:
        return self.policy_set.get_rules_for_subject(subject)

    def get_policy_summary(self) -> Dict[str, Any]:
        return {
            'component_id': self.component_id,
            'version': self.policy_set.version,
            'rule_count': len(self.policy_set.rules),
            'subjects': self.policy_set.get_all_subjects(),
            'source_file': str(self.source_file) if self.source_file else None,
            'loaded_at': self.loaded_at.isoformat(),
            'v2_context_enabled': True
        }


class RBACSetupPhase(BootstrapPhase):
    """Enhanced RBAC Setup Phase with V2 context integration."""
    
    phase_id = 'rbac_setup'
    order = 30

    def __init__(self):
        super().__init__()
        # UPGRADED: V2 context integration
        self.config_provider: Optional[SimpleConfigProvider] = None
        self.rbac_contexts: Dict[str, Any] = {}

    async def run(self, context: BootstrapContext) -> None:
        result = await self.execute(context)
        if not result.success:
            raise RuntimeError(f'RBAC setup failed: {result.message}')

    async def execute(self, context) -> PhaseResult:  # Use Any instead of BootstrapContext to avoid import
        logger.info('Executing Combined RBAC Setup Phase with V2 Context Integration')
        
        # UPGRADED: Initialize enhanced configuration management
        self._initialize_enhanced_config(context)
        
        errors = []
        warnings = []
        loaded_components = []

        try:
            if not self._is_rbac_enabled(context):
                return self._create_disabled_result()

            # UPGRADED: Create main RBAC context
            main_context = self._create_rbac_context(
                context, 
                'rbac_main', 
                {'setup_step': 'initialization'}
            )
            self.rbac_contexts['main'] = main_context

            policy_files = self._locate_rbac_policy_files(context)
            if not policy_files:
                return self._create_no_files_result(context, warnings)

            # UPGRADED: Create policy processing context
            policy_context = self._create_rbac_context(
                context, 
                'policy_processing', 
                {'policy_file_count': len(policy_files)}
            )
            self.rbac_contexts['policy_processing'] = policy_context

            for i, policy_file in enumerate(policy_files):
                try:
                    # UPGRADED: Create file-specific context
                    file_context = self._create_rbac_context(
                        context, 
                        f'policy_file_{i}_{policy_file.stem}', 
                        {'file_path': str(policy_file), 'processing_index': i}
                    )
                    self.rbac_contexts[f'file_{i}'] = file_context

                    component = await self._process_policy_file(policy_file, context)
                    if component:
                        loaded_components.append(component)
                        logger.info(f'✓ Successfully loaded RBAC policies from: {policy_file} with V2 context')
                except Exception as e:
                    error_msg = f'Failed to process RBAC policy file {policy_file}: {e}'
                    errors.append(error_msg)
                    logger.error(error_msg, exc_info=True)

            # Create RBAC engine if we have components
            if loaded_components:
                try:
                    # UPGRADED: Create engine context
                    engine_context = self._create_rbac_context(
                        context, 
                        'rbac_engine', 
                        {'component_count': len(loaded_components)}
                    )
                    self.rbac_contexts['engine'] = engine_context

                    # Import here to avoid circular dependency
                    from bootstrap.security.rbac_engine import RBACPolicyEngine
                    
                    rbac_engine = RBACPolicyEngine(loaded_components)
                    
                    # Register with registry manager
                    if hasattr(context, 'registry_manager'):
                        context.registry_manager.register_service_with_certification(
                            service_type=RBACPolicyEngine,
                            instance=rbac_engine,
                            service_id='rbac_engine',
                            category='security_service',
                            description='RBAC permission evaluation engine (V2 enhanced)',
                            requires_initialize=False
                        )
                    
                    logger.info(f"✓ RBACPolicyEngine registered with V2 context: {rbac_engine.get_stats()['total_rules']} rules")
                except Exception as e:
                    error_msg = f'Failed to create RBAC engine: {e}'
                    errors.append(error_msg)
                    logger.error(error_msg, exc_info=True)

            # Emit policy events with V2 integration
            if loaded_components:
                await self._publish_policy_events(loaded_components, context)

            # UPGRADED: Validate V2 integration
            v2_validation_issues = self._validate_v2_integration()
            if v2_validation_issues:
                warnings.extend(v2_validation_issues)

            success = len(errors) == 0
            total_rules = sum(len(comp.policy_set.rules) for comp in loaded_components)
            message = f'RBAC setup completed with V2 integration - loaded {total_rules} rules from {len(loaded_components)} policy sets'

            # Get engine stats
            engine_stats = {}
            try:
                if loaded_components:
                    # Avoid importing at module level
                    from bootstrap.security.rbac_engine import RBACPolicyEngine
                    engine = context.registry.get_service_instance(RBACPolicyEngine)
                    engine_stats = engine.get_stats()
            except:
                pass

            return PhaseResult(
                success=success,
                message=message,
                errors=errors,
                warnings=warnings,
                metadata={
                    'policy_sets_loaded': len(loaded_components),
                    'total_rules': total_rules,
                    'policy_files_processed': len(policy_files),
                    'rbac_enabled': True,
                    'components': [comp.get_policy_summary() for comp in loaded_components],
                    'rbac_engine_stats': engine_stats,
                    'v2_integration': True,
                    'config_provider_enabled': self.config_provider is not None,
                    'rbac_contexts_created': len(self.rbac_contexts)
                }
            )

        except Exception as e:
            error_msg = f'Critical error in RBAC setup phase: {e}'
            logger.error(error_msg, exc_info=True)
            return PhaseResult(
                success=False,
                message='RBAC setup failed critically',
                errors=[error_msg],
                warnings=warnings,
                metadata={
                    'phase_failed': True,
                    'v2_integration': True,
                    'rbac_contexts_created': len(self.rbac_contexts)
                }
            )

    def _initialize_enhanced_config(self, context) -> None:
        """UPGRADED: Initialize V2 context helper configuration management."""
        try:
            # Extract RBAC-specific configuration
            rbac_config = context.global_app_config.get('rbac', {})
            feature_flags = context.global_app_config.get('feature_flags', {})
            
            # Merge configurations for the config provider
            enhanced_config = {
                **{f"rbac.{k}": v for k, v in rbac_config.items()},
                **{f"feature.{k}": v for k, v in feature_flags.items() if k.startswith('rbac') or k.startswith('enable_rbac')}
            }
            
            self.config_provider = SimpleConfigProvider(enhanced_config)
            logger.debug(f'RBAC setup enhanced config provider initialized with {len(enhanced_config)} configuration entries')
            
        except Exception as e:
            logger.warning(f'Failed to initialize RBAC setup enhanced config provider: {e}')
            self.config_provider = None

    def _is_rbac_enabled(self, context) -> bool:
        """Check if RBAC is enabled in configuration with V2 config support."""
        # UPGRADED: Use config provider if available
        if self.config_provider:
            enabled = self.config_provider.get_config('feature', 'enable_rbac_bootstrap', False)
        else:
            feature_flags = context.global_app_config.get('feature_flags', {})
            enabled = feature_flags.get('enable_rbac_bootstrap', False)
            
        logger.info(f'RBAC bootstrap enabled: {enabled} (V2 config: {self.config_provider is not None})')
        return enabled

    def _create_disabled_result(self) -> PhaseResult:
        """Create result for disabled RBAC with V2 metadata."""
        logger.info('RBAC bootstrap disabled in configuration - skipping policy loading')
        return PhaseResult(
            success=True,
            message='RBAC bootstrap skipped (disabled in config)',
            errors=[],
            warnings=['RBAC bootstrap disabled - policies not loaded'],
            metadata={
                'rbac_enabled': False,
                'v2_integration': True,
                'config_provider_enabled': self.config_provider is not None
            }
        )

    def _create_no_files_result(self, context, warnings: List[str]) -> PhaseResult:
        """Create result when no policy files are found with V2 metadata."""
        message = 'No RBAC policy files found'
        if context.global_app_config.get('env') == 'prod':
            warnings.append('No RBAC policies found - enterprise deployments should include bootstrap_rbac.yaml')
        
        logger.info(message)
        return PhaseResult(
            success=True,
            message=message,
            errors=[],
            warnings=warnings,
            metadata={
                'policy_files_found': 0, 
                'rbac_enabled': True,
                'v2_integration': True,
                'config_provider_enabled': self.config_provider is not None
            }
        )

    def _locate_rbac_policy_files(self, context) -> List[Path]:
        """Locate RBAC policy files to process with V2 config support."""
        policy_files = []

        # UPGRADED: Check for V2 config provider paths first
        if self.config_provider:
            additional_paths = self.config_provider.get_config('rbac', 'additional_policy_files', [])
            for path_str in additional_paths:
                additional_path = Path(path_str)
                if not additional_path.is_absolute():
                    package_root = Path(__file__).resolve().parents[2]
                    additional_path = package_root / additional_path
                if additional_path.exists():
                    policy_files.append(additional_path)
                    logger.debug(f'Found V2 config RBAC policies: {additional_path}')

        # Check for legacy path
        legacy_path = context.global_app_config.get('bootstrap_rbac_path')
        if legacy_path:
            legacy_file = Path(legacy_path)
            if legacy_file.exists():
                policy_files.append(legacy_file)
                logger.debug(f'Found legacy RBAC policy file: {legacy_file}')
                return policy_files

        # Get package root
        package_root = Path(__file__).resolve().parents[2]
        env = context.global_app_config.get('env', 'default')

        # Check environment-specific policy file
        env_policy_path = package_root / 'configs' / env / 'bootstrap_rbac.yaml'
        if env_policy_path.exists():
            policy_files.append(env_policy_path)
            logger.debug(f'Found environment RBAC policies: {env_policy_path}')

        # Fallback to default if no env-specific file
        if not policy_files:
            default_policy_path = package_root / 'configs' / 'default' / 'bootstrap_rbac.yaml'
            if default_policy_path.exists():
                policy_files.append(default_policy_path)
                logger.debug(f'Found default RBAC policies: {default_policy_path}')

        # Check for additional policy files from original config
        if not self.config_provider:  # Only if we don't have V2 config
            rbac_config = context.global_app_config.get('rbac', {})
            additional_paths = rbac_config.get('additional_policy_files', [])
            for path_str in additional_paths:
                additional_path = Path(path_str)
                if not additional_path.is_absolute():
                    additional_path = package_root / additional_path
                if additional_path.exists():
                    policy_files.append(additional_path)
                    logger.debug(f'Found additional RBAC policies: {additional_path}')

        logger.info(f'Located {len(policy_files)} RBAC policy files for env={env} with V2 integration')
        return policy_files

    async def _process_policy_file(self, policy_file: Path, context) -> Optional[PolicySetComponent]:
        """Process a single policy file with V2 context integration."""
        # Load raw policy data
        raw_data = await self._load_raw_policy_data(policy_file)
        if not raw_data:
            return None

        # Validate schema
        await self._validate_schema(raw_data, policy_file)

        # Create policy set
        policy_set = RBACPolicySet(**raw_data)
        if not policy_set.rules:
            logger.warning(f'No valid rules found in policy file: {policy_file}')
            return None

        # Create and register component with V2 context
        component = await self._create_and_register_component(policy_set, policy_file, context)
        return component

    async def _load_raw_policy_data(self, policy_file: Path) -> Optional[Dict[str, Any]]:
        """Load raw policy data from file with V2 context logging."""
        try:
            raw_data = load_yaml_robust(policy_file)
            if not raw_data:
                logger.warning(f'RBAC policy file is empty: {policy_file}')
                return None

            if not isinstance(raw_data, dict):
                raise ValueError('RBAC policy file must contain a dictionary')

            if 'version' not in raw_data:
                logger.warning(f'RBAC policy file missing version, using default: {policy_file}')
                raw_data['version'] = '1.0'

            # UPGRADED: Add V2 context metadata
            raw_data.setdefault('metadata', {})
            raw_data['metadata']['v2_context_processed'] = True
            raw_data['metadata']['config_provider_available'] = self.config_provider is not None

            return raw_data

        except Exception as e:
            logger.error(f'Error loading RBAC policy file {policy_file}: {e}')
            raise

    async def _validate_schema(self, data: Dict[str, Any], policy_file: Path) -> None:
        """Validate policy data against schema with V2 context logging."""
        try:
            schema_path = Path(__file__).parent / '../schemas/rbac_policy.schema.json'
            if schema_path.exists():
                schema = json.loads(schema_path.read_text())
                js_validate(instance=data, schema=schema)
                logger.debug(f'Schema validation passed for: {policy_file} with V2 context')
        except SchemaError as e:
            logger.warning(f'Schema validation failed for {policy_file}: {e}')
        except Exception as e:
            logger.debug(f'Schema validation unavailable for {policy_file}: {e}')

    async def _create_and_register_component(self, policy_set: RBACPolicySet, source_file: Path, context) -> PolicySetComponent:
        """Create and register a policy set component with V2 context integration."""
        component_id = f'rbac_policies_{source_file.stem}'
        
        metadata = ComponentMetadata(
            id=component_id,
            name=f'RBAC Policies from {source_file.name}',
            version=policy_set.version,
            category='security_policy',
            description=f'RBAC policies loaded from {source_file.name} (V2 enhanced)',
            epistemic_tags=['security', 'rbac', 'policy', 'v2_context'],
            requires_initialize=True
        )

        component = PolicySetComponent(metadata, policy_set, source_file)
        
        # Register component
        if hasattr(context, 'registry_manager') and hasattr(context.registry_manager, 'register_with_certification'):
            context.registry_manager.register_with_certification(
                component, 
                metadata,
                additional_cert_data={
                    'policy_count': len(policy_set.rules),
                    'source_file': str(source_file),
                    'rbac_version': policy_set.version,
                    'v2_context_enabled': True,
                    'config_provider_available': self.config_provider is not None
                }
            )

        logger.info(f"Registered RBAC policy component '{component_id}' with {len(policy_set.rules)} rules and V2 context")
        return component

    def _create_rbac_context(self, base_context, context_id: str, metadata: Dict[str, Any]):
        """UPGRADED: Create RBAC-specific context using V2 context builder."""
        try:
            context_builder = create_context_builder(
                component_id=f"rbac_{context_id}",
                run_id=f"{base_context.run_id}_rbac"
            )
            
            # Configure builder
            if hasattr(base_context, 'registry'):
                context_builder.with_registry(base_context.registry)
            
            if hasattr(base_context, 'event_bus'):
                context_builder.with_event_bus(base_context.event_bus)
            
            # Add enhanced metadata
            enhanced_metadata = {
                **metadata,
                'rbac_phase': True,
                'v2_context': True,
                'config_provider_available': self.config_provider is not None
            }
            context_builder.with_metadata(**enhanced_metadata)
            
            # Add feature flags if available
            if hasattr(base_context, 'feature_flags'):
                context_builder.with_feature_flags(base_context.feature_flags)
            
            return context_builder.build()
            
        except Exception as e:
            logger.warning(f"Failed to create V2 RBAC context for {context_id}: {e}")
            return None

    def _validate_v2_integration(self) -> List[str]:
        """UPGRADED: Validate V2 context integration across RBAC setup."""
        issues = []
        
        try:
            # Check config provider
            if self.config_provider is None:
                issues.append("V2 config provider not initialized for RBAC setup")
            
            # Check RBAC contexts
            expected_contexts = ['main']
            for expected in expected_contexts:
                if expected not in self.rbac_contexts:
                    issues.append(f"Missing V2 RBAC context for: {expected}")
            
            # Validate RBAC context functionality
            for context_name, rbac_context in self.rbac_contexts.items():
                if rbac_context is None:
                    issues.append(f"V2 RBAC context for {context_name} is None")
                elif not hasattr(rbac_context, 'metadata'):
                    issues.append(f"V2 RBAC context for {context_name} missing metadata")
            
            if issues:
                logger.warning(f"V2 integration validation for RBAC setup found {len(issues)} issues")
            else:
                logger.debug("V2 integration validation for RBAC setup passed")
                
        except Exception as e:
            issues.append(f"V2 integration validation failed for RBAC setup: {e}")
            
        return issues

    async def _publish_policy_events(self, components: List[PolicySetComponent], context) -> None:
        """Publish policy application events with V2 integration metadata."""
        try:
            # Publish individual component events
            for component in components:
                # Note: Import here to avoid circular dependency issues
                try:
                    from domain.ports.event_bus_port import EventBusPort
                    if hasattr(context, 'event_bus') and context.event_bus:
                        context.event_bus.publish('RBAC_POLICY_APPLIED', {
                            'component_id': component.component_id,
                            'rule_count': len(component.policy_set.rules),
                            'v2_context_enabled': True
                        })
                except Exception as e:
                    logger.debug(f'Could not publish individual policy event: {e}')

            # Publish aggregate signal with V2 metadata
            if hasattr(context, 'signal_emitter'):
                await self._publish_rbac_signal(components, context)

        except Exception as e:
            logger.warning(f'Failed to publish RBAC events: {e}')

    async def _publish_rbac_signal(self, components: List[PolicySetComponent], context) -> None:
        """Publish RBAC policies loaded signal with V2 integration metadata."""
        try:
            from bootstrap.signals.bootstrap_signals import RBAC_POLICIES_LOADED
            
            all_subjects = set()
            total_rules = 0
            for component in components:
                all_subjects.update(component.policy_set.get_all_subjects())
                total_rules += len(component.policy_set.rules)

            # UPGRADED: Add V2 integration metadata to signal
            event_payload = {
                'policy_count': total_rules,
                'component_count': len(components),
                'subjects': list(all_subjects),
                'components': [comp.get_policy_summary() for comp in components],
                'bootstrap_run_id': context.run_id,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'v2_context_integration': True,
                'config_provider_enabled': self.config_provider is not None,
                'rbac_contexts_created': len(self.rbac_contexts)
            }

            await context.signal_emitter.emit_signal(
                signal_type=RBAC_POLICIES_LOADED,
                payload=event_payload
            )
            logger.debug('Published RBAC policies loaded signal with V2 integration metadata')

        except ImportError:
            logger.debug('RBAC signal not available - skipping signal emission')
        except Exception as e:
            logger.warning(f'Failed to publish RBAC signal: {e}')