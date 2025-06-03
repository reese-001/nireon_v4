from __future__ import annotations
from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base_phase import BootstrapPhase, PhaseResult
from bootstrap.bootstrap_helper.utils import load_yaml_robust

logger = logging.getLogger(__name__)

class RBACSetupPhase(BootstrapPhase):
    async def execute(self, context) -> PhaseResult:
        logger.info('Executing RBAC Setup Phase for enterprise policies')
        
        errors = []
        warnings = []
        loaded_policies = []
        
        try:
            # Check if RBAC is enabled
            rbac_enabled = context.global_app_config.get('feature_flags', {}).get('enable_rbac_bootstrap', False)
            
            if not rbac_enabled:
                logger.info('RBAC bootstrap disabled in feature flags - skipping policy loading')
                return PhaseResult(
                    success=True,
                    message='RBAC bootstrap skipped (disabled in config)',
                    errors=[],
                    warnings=['RBAC bootstrap disabled - policies not loaded'],
                    metadata={'rbac_enabled': False}
                )
            
            # Locate RBAC policy files - ensure paths align with configs structure
            policy_files = self._locate_rbac_policy_files(context)
            
            if not policy_files:
                message = 'No RBAC policy files found - enterprise deployments should include bootstrap_rbac.yaml'
                if context.global_app_config.get('env') == 'prod':
                    warnings.append(message)
                logger.info(message)
                return PhaseResult(
                    success=True,
                    message='No RBAC policies to load',
                    errors=[],
                    warnings=warnings,
                    metadata={'policy_files_found': 0}
                )
            
            # Load and process policy files
            for policy_file in policy_files:
                try:
                    policies = await self._load_policy_file(policy_file)
                    if policies:
                        await self._register_policies(policies, policy_file, context)
                        loaded_policies.extend(policies.get('policies', []))
                        logger.info(f'✓ Loaded RBAC policies from: {policy_file}')
                except Exception as e:
                    error_msg = f'Failed to load RBAC policies from {policy_file}: {e}'
                    errors.append(error_msg)
                    logger.error(error_msg)
            
            # Publish RBAC event if policies were loaded
            if loaded_policies:
                await self._publish_rbac_event(loaded_policies, context)
            
            success = len(errors) == 0
            message = f'RBAC setup completed - loaded {len(loaded_policies)} policies from {len(policy_files)} files'
            
            return PhaseResult(
                success=success,
                message=message,
                errors=errors,
                warnings=warnings,
                metadata={
                    'policies_loaded': len(loaded_policies),
                    'policy_files_processed': len(policy_files),
                    'rbac_enabled': True
                }
            )
            
        except Exception as e:
            error_msg = f'Critical error in RBAC setup phase: {e}'
            logger.error(error_msg, exc_info=True)
            return PhaseResult(
                success=False,
                message='RBAC setup failed',
                errors=[error_msg],
                warnings=warnings,
                metadata={'phase_failed': True}
            )

    def _locate_rbac_policy_files(self, context) -> List[Path]:
        """Locate RBAC policy files - updated to align with configs structure"""
        policy_files = []
        
        # Use package root from the bootstrap system
        package_root = Path(__file__).resolve().parents[3]  # Go up to nireon_v4 root
        env = context.global_app_config.get('env', 'default')
        
        # Check environment-specific policy path first: configs/{env}/bootstrap_rbac.yaml
        env_policy_path = package_root / 'configs' / env / 'bootstrap_rbac.yaml'
        if env_policy_path.exists():
            policy_files.append(env_policy_path)
            logger.debug(f'Found environment RBAC policies: {env_policy_path}')
        
        # Fallback to default policy path: configs/default/bootstrap_rbac.yaml
        if not policy_files:
            default_policy_path = package_root / 'configs' / 'default' / 'bootstrap_rbac.yaml'
            if default_policy_path.exists():
                policy_files.append(default_policy_path)
                logger.debug(f'Found default RBAC policies: {default_policy_path}')
        
        # Check for additional policy files specified in config
        additional_paths = context.global_app_config.get('rbac', {}).get('additional_policy_files', [])
        for path_str in additional_paths:
            additional_path = Path(path_str)
            if not additional_path.is_absolute():
                additional_path = package_root / additional_path
            
            if additional_path.exists():
                policy_files.append(additional_path)
                logger.debug(f'Found additional RBAC policies: {additional_path}')
        
        logger.info(f'Located {len(policy_files)} RBAC policy files for env={env}')
        return policy_files

    async def _load_policy_file(self, policy_file: Path) -> Optional[Dict[str, Any]]:
        """Load and validate RBAC policy file"""
        try:
            policies = load_yaml_robust(policy_file)
            
            if not policies:
                logger.warning(f'RBAC policy file is empty: {policy_file}')
                return None
            
            if not isinstance(policies, dict):
                raise ValueError('RBAC policy file must contain a dictionary')
            
            # Validate structure
            if 'version' not in policies:
                logger.warning(f'RBAC policy file missing version: {policy_file}')
            
            if 'policies' not in policies:
                logger.warning(f"RBAC policy file missing 'policies' section: {policy_file}")
                return None
            
            policy_list = policies.get('policies', [])
            if not isinstance(policy_list, list):
                raise ValueError("'policies' section must be a list")
            
            # Validate individual policies
            for i, policy in enumerate(policy_list):
                if not isinstance(policy, dict):
                    raise ValueError(f'Policy {i} must be a dictionary')
                
                if 'role' not in policy:
                    raise ValueError(f"Policy {i} missing required 'role' field")
                
                if 'permissions' not in policy:
                    raise ValueError(f"Policy {i} missing required 'permissions' field")
            
            logger.debug(f'Validated RBAC policy file: {policy_file} ({len(policy_list)} policies)')
            return policies
            
        except Exception as e:
            logger.error(f'Error loading RBAC policy file {policy_file}: {e}')
            raise

    async def _register_policies(self, policies: Dict[str, Any], source_file: Path, context) -> None:
        """Register RBAC policies as a component in the registry"""
        policy_list = policies.get('policies', [])
        rbac_component_id = f'rbac_policies_{source_file.stem}'
        
        # Create metadata for the policy set
        from application.components.lifecycle import ComponentMetadata
        from datetime import datetime, timezone
        
        rbac_metadata = ComponentMetadata(
            id=rbac_component_id,
            name=f'RBAC Policies from {source_file.name}',
            version=policies.get('version', '1.0.0'),
            category='rbac_policy_set',
            description=f'Enterprise RBAC policies loaded from {source_file}',
            epistemic_tags=['security', 'rbac', 'enterprise'],
            created_at=datetime.now(timezone.utc)
        )
        
        # Create policy set wrapper
        class RBACPolicySet:
            def __init__(self, policies: List[Dict[str, Any]], source: Path):
                self.policies = policies
                self.source = source
                self.loaded_at = datetime.now(timezone.utc)
            
            def get_policies_for_role(self, role: str) -> List[Dict[str, Any]]:
                return [p for p in self.policies if p.get('role') == role]
            
            def get_all_roles(self) -> List[str]:
                return list(set(p.get('role') for p in self.policies))
        
        policy_set = RBACPolicySet(policy_list, source_file)
        
        # Register with additional certification data
        context.registry_manager.register_with_certification(
            policy_set, 
            rbac_metadata,
            additional_cert_data={
                'policy_count': len(policy_list),
                'source_file': str(source_file),
                'rbac_version': policies.get('version', 'unknown')
            }
        )
        
        logger.info(f"Registered RBAC policy set '{rbac_component_id}' with {len(policy_list)} policies")

    async def _publish_rbac_event(self, loaded_policies: List[Dict[str, Any]], context) -> None:
        """Publish RBAC policies loaded event"""
        try:
            from signals.bootstrap_signals import RBAC_POLICIES_LOADED
            
            event_payload = {
                'policy_count': len(loaded_policies),
                'roles': list(set(p.get('role') for p in loaded_policies)),
                'bootstrap_run_id': context.run_id,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            await context.signal_emitter.emit_signal(
                signal_type=RBAC_POLICIES_LOADED,
                payload=event_payload
            )
            
        except Exception as e:
            logger.warning(f'Failed to publish RBAC event: {e}')