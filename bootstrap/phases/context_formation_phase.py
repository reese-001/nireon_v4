# nireon_v4\bootstrap\phases\context_formation_phase.py
from __future__ import annotations
import logging
from typing import Any, Dict, List, Tuple, Optional
from enum import Enum
from datetime import datetime, timezone
from .base_phase import BootstrapPhase, PhaseResult
logger = logging.getLogger(__name__)

# --- REMOVED ServiceResolutionMode, ServiceImplementationType, ServiceResolutionResult ---
# This logic is now handled by the Abiogenesis and FactorySetup phases.

class ContextFormationPhase(BootstrapPhase):
    def __init__(self):
        super().__init__()
        # --- REMOVED service_resolutions, resolution_mode, config_provider ---

    async def execute(self, context) -> PhaseResult:
        logger.info('Executing Context Formation Phase (V2 simplified)')
        errors = []
        warnings = []
        created_services = []

        try:
            # The only remaining responsibility is ensuring core context managers are available.
            await self._emerge_feature_flags_manager(context, created_services, errors)
            await self._emerge_component_registry(context, created_services, errors)

            success = len(errors) == 0
            message = f"Context Formation complete - {len(created_services)} core context managers ensured."
            return PhaseResult(success=success, message=message, errors=errors, warnings=warnings, metadata={
                'services_emerged': created_services,
                'v2_integration': True
            })
        except Exception as e:
            error_msg = f'Critical failure during Context Formation: {e}'
            logger.error(error_msg, exc_info=True)
            return PhaseResult.failure_result(message='Context Formation failed', errors=[error_msg], warnings=warnings)

    async def _emerge_feature_flags_manager(self, context, created_services: list, errors: list) -> None:
        try:
            from bootstrap.bootstrap_helper.feature_flags import FeatureFlagsManager
            try:
                # Check if it already exists
                context.registry.get_service_instance(FeatureFlagsManager)
                logger.info('FeatureFlagsManager already exists in registry.')
                return
            except:
                pass # Doesn't exist, so we create it.

            feature_flags_config = context.global_app_config.get('feature_flags', {})
            ff_manager = FeatureFlagsManager(feature_flags_config)
            context.registry_manager.register_service_with_certification(
                service_type=FeatureFlagsManager,
                instance=ff_manager,
                service_id='FeatureFlagsManager',
                category='core_service',
                description='System-wide feature flag management for adaptive behavior.',
                requires_initialize=False
            )
            created_services.append('FeatureFlagsManager')
            logger.info('✓ FeatureFlagsManager emerged.')
        except Exception as e:
            error_msg = f'Failed to emerge FeatureFlagsManager: {e}'
            errors.append(error_msg)
            logger.error(error_msg)

    async def _emerge_component_registry(self, context, created_services: list, errors: list) -> None:
        try:
            # The registry already exists on the context, we just formally register it with itself.
            context.registry_manager.register_service_with_certification(
                service_type=type(context.registry),
                instance=context.registry,
                service_id='ComponentRegistry',
                category='core_service',
                description='Central component registry enabling system self-awareness.',
                requires_initialize=False
            )
            created_services.append('ComponentRegistry')
            logger.info('✓ ComponentRegistry achieved reflexive self-emergence.')
        except Exception as e:
            error_msg = f'Failed to register ComponentRegistry reflexively: {e}'
            errors.append(error_msg)
            logger.error(error_msg)

    def should_skip_phase(self, context) -> tuple[bool, str]:
        return (False, '')