# nireon_v4/components/mechanisms/sentinel/service_helpers/initialization.py
from __future__ import annotations
import logging
from typing import TYPE_CHECKING, List
import numpy as np

from application.services.idea_service import IdeaService
from domain.ports.llm_port import LLMPort
from domain.ports.embedding_port import EmbeddingPort
from domain.ports.event_bus_port import EventBusPort
from domain.ports.mechanism_gateway_port import MechanismGatewayPort
from domain.context import NireonExecutionContext
from ..constants import DEFAULT_WEIGHTS

if TYPE_CHECKING:
    from ..service import SentinelMechanism

logger = logging.getLogger(__name__)

class InitializationHelper:
    def __init__(self, sentinel: 'SentinelMechanism'):
        self.sentinel = sentinel

    async def initialize_impl(self, context: NireonExecutionContext) -> None:
        context.logger.info(f'[{self.sentinel.component_id}] Performing Sentinel-specific initialization.')
        try:
            self.sentinel.gateway = context.component_registry.get_service_instance(MechanismGatewayPort)
            self.sentinel.llm = self.sentinel.gateway # For V4, the gateway is the LLM port
            self.sentinel.embed = context.component_registry.get_service_instance(EmbeddingPort)
            self.sentinel.event_bus = context.component_registry.get_service_instance(EventBusPort)
            self.sentinel.idea_service = context.component_registry.get_service_instance(IdeaService)
            context.logger.info(f'[{self.sentinel.component_id}] All core dependencies resolved.')
        except Exception as e:
            context.logger.error(f'[{self.sentinel.component_id}] Failed to resolve essential dependencies: {e}', exc_info=True)
            raise RuntimeError(f"SentinelMechanism '{self.sentinel.component_id}' failed to initialize dependencies.") from e

        self.initialize_weights()
        self.sentinel.trust_th = self.sentinel.sentinel_cfg.trust_threshold
        self.sentinel.min_axis = self.sentinel.sentinel_cfg.min_axis_score
        
        issues: List[str] = []
        if not self.validate_configuration_internal(issues):
            context.logger.warning(f"[{self.sentinel.component_id}] Configuration issues found during _initialize_impl: {'; '.join(issues)}")
            
        context.logger.info(f'[{self.sentinel.component_id}] Sentinel-specific initialization checks complete.')

    def initialize_weights(self) -> None:
        raw = np.asarray(self.sentinel.sentinel_cfg.weights, dtype=float)
        if raw.shape != (3,):
            logger.error(f'[{self.sentinel.component_id}] Configured weights length {len(raw)} invalid (expected 3). Using defaults {DEFAULT_WEIGHTS}')
            processed = np.array(DEFAULT_WEIGHTS, dtype=float)
        else:
            s = raw.sum()
            if np.isclose(s, 0.0):
                logger.error(f'[{self.sentinel.component_id}] Configured weights sum≈0 (raw={raw.tolist()}). Using defaults {DEFAULT_WEIGHTS}')
                processed = np.array(DEFAULT_WEIGHTS, dtype=float)
            elif not np.isclose(s, 1.0):
                processed = raw / s
                logger.info(f'[{self.sentinel.component_id}] Normalised weights {raw.tolist()} → {processed.tolist()} (sum→1.0)')
            else:
                processed = raw
                logger.debug(f'[{self.sentinel.component_id}] Using configured weights {processed.tolist()}')
        self.sentinel.weights = processed
        if list(self.sentinel.sentinel_cfg.weights) != processed.tolist():
            logger.debug(f'[{self.sentinel.component_id}] Updating sentinel_cfg.weights from {self.sentinel.sentinel_cfg.weights} → {processed.tolist()}')
            self.sentinel.sentinel_cfg.weights = processed.tolist()


    def validate_configuration_internal(self, issues: List[str]) -> bool:
        is_valid = True
        try:
            cfg = self.sentinel.sentinel_cfg

            if not (0.0 <= cfg.trust_threshold <= 10.0):
                issues.append(f'Trust threshold ({cfg.trust_threshold}) out of expected range [0.0, 10.0].')
                is_valid = False

            if not (0.0 <= cfg.min_axis_score <= 10.0):
                issues.append(f'Min axis score ({cfg.min_axis_score}) out of expected range [0.0, 10.0].')
                is_valid = False

            if not hasattr(self.sentinel, 'weights') or self.sentinel.weights is None:
                issues.append('Weights not initialized on sentinel instance.')
                is_valid = False
            elif not np.isclose(np.sum(self.sentinel.weights), 1.0):
                issues.append(f'Initialized weights ({self.sentinel.weights.tolist()}) do not sum to 1.0.')
                is_valid = False
            elif len(self.sentinel.weights) != 3:
                issues.append(f'Initialized weights array has incorrect length ({len(self.sentinel.weights)}), expected 3.')
                is_valid = False

        except Exception as e:
            issues.append(f'Configuration validation error: {e}')
            is_valid = False

        return is_valid

    def acquire_idea_service(self, context: NireonExecutionContext, issues: List[str]) -> bool:
        if self.sentinel.idea_service:
            return True

        if context.component_registry:
            service_names_to_try = ['IdeaService', 'idea_service', IdeaService.__name__]

            for service_name in service_names_to_try:
                try:
                    service_instance = context.component_registry.get(service_name)
                    if isinstance(service_instance, IdeaService):
                        self.sentinel.idea_service = service_instance
                        context.logger.debug(
                            f"[{self.sentinel.component_id}] Successfully acquired '{service_name}' "
                            f"(type: {type(service_instance)})."
                        )
                        return True
                    elif service_instance is not None:
                        context.logger.warning(
                            f"[{self.sentinel.component_id}] Found '{service_name}' but it is not an "
                            f"IdeaService instance (type: {type(service_instance)})."
                        )
                except (KeyError, AttributeError):
                    context.logger.debug(
                        f"[{self.sentinel.component_id}] Service '{service_name}' not found or error acquiring."
                    )
                    continue

            msg = f'IdeaService not found in registry under names: {service_names_to_try}.'
            issues.append(msg)
            context.logger.warning(f"[{self.sentinel.component_id}] {msg}")
            return False

        else:
            msg = 'ComponentRegistry not available in context to acquire IdeaService.'
            issues.append(msg)
            context.logger.warning(f"[{self.sentinel.component_id}] {msg}")
            return False

    async def late_initialize(self, context: NireonExecutionContext) -> None:
        if self.sentinel.idea_service is None:
            context.logger.debug(f'[{self.sentinel.component_id}] Attempting late acquisition of IdeaService.')
            issues: List[str] = []
            if self.acquire_idea_service(context, issues):
                context.logger.info(f'[{self.sentinel.component_id}] Successfully acquired IdeaService during late initialization.')
            else:
                context.logger.warning(
                    f'[{self.sentinel.component_id}] Failed to acquire IdeaService during late initialization. '
                    f'Issues: {"; ".join(issues)}'
                )

        if not self.validate_runtime_dependencies(context):
            context.logger.error(
                f'[{self.sentinel.component_id}] Runtime dependencies validation failed during late initialization. '
                f'Processing may fail.'
            )

    def validate_runtime_dependencies(self, context: NireonExecutionContext) -> bool:
        missing_deps = []

        if self.sentinel.llm is None:
            missing_deps.append('LLMPort (self.sentinel.llm)')

        if self.sentinel.embed is None:
            missing_deps.append('EmbeddingPort (self.sentinel.embed)')

        if self.sentinel.idea_service is None:
            missing_deps.append('IdeaService (self.sentinel.idea_service)')

        if missing_deps:
            context.logger.warning(
                f"[{self.sentinel.component_id}] Missing critical runtime dependencies: {', '.join(missing_deps)}. "
                f"Functionality will be severely limited or fail."
            )
            return False

        return True