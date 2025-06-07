from __future__ import annotations
import asyncio
import importlib
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Mapping
from core.base_component import NireonBaseComponent
from core.lifecycle import ComponentMetadata
from domain.ports.llm_port import LLMPort
from infrastructure import event_bus
from infrastructure.llm.parameter_service import ParameterService
from infrastructure.llm.factory import create_llm_instance
from domain.ports.event_bus_port import EventBusPort

logger = logging.getLogger(__name__)

@dataclass(slots=True)
class _ModelConfig:
    name: str
    provider: str
    backend_class: str
    kwargs: Dict[str, Any] = field(default_factory=dict)

class LLMRouter(NireonBaseComponent):
    def __init__(self, config: dict):
        metadata = ComponentMetadata(
            id="llm_router",  # ✅ Correct ID for LLMRouter
            name="LLMRouter",
            version="1.0.0",
            category="service",
            description="Shared LLM router – resolves logical routes to provider instances.",
            requires_initialize=True,
            epistemic_tags=[],
        )
        super().__init__(config=config, metadata_definition=metadata)
        self._config = config or {}

        self._event_bus: Optional[EventBusPort] = event_bus
        self._default_route: str = self._config.get('default', 'default')
        self._route_map: Dict[str, str] = self._config.get('routes', {})
        self._model_defs: Dict[str, _ModelConfig] = {}
        self._backends: Dict[str, LLMPort] = {}

        # ✅ ParameterService instantiation needs its own metadata
        param_metadata = ComponentMetadata(
            id="parameter_service",
            name="ParameterService",
            version="1.0.0",
            category="service",
            description="Provides dynamic or default LLM parameter resolution.",
            requires_initialize=False,
            epistemic_tags=[],
        )
        self._param_service = ParameterService(config=self._config.get('parameters', {}), metadata_definition=param_metadata)

        # Process model definitions
        for model_name, model_cfg in self._config.get('models', {}).items():
            self._model_defs[model_name] = _ModelConfig(
                name=model_name,
                provider=model_cfg.get('provider', 'unknown'),
                backend_class=model_cfg.get('backend', ''),
                kwargs={k: v for k, v in model_cfg.items() if k not in {'provider', 'backend'}}
            )

    def _process_impl(self, prompt: str, context: dict) -> str:
        """
        Implementation of abstract method from NireonBaseComponent.
        For LLMRouter, this delegates to the appropriate backend.
        """
        try:
            # Extract route from context or use default
            route = context.get('route', self._default_route)
            backend_key = self._route_map.get(route, route)
            
            # This is a synchronous fallback - in practice, should use async methods
            logger.warning("Using synchronous _process_impl fallback - prefer async methods")
            return f"LLMRouter processed prompt via route '{route}' -> backend '{backend_key}'"
        except Exception as e:
            logger.error(f"Error in LLMRouter._process_impl: {e}")
            return f"Error processing prompt: {e}"

    async def initialize_async(self) -> None:
        """Initialize the router and load default backend"""
        await self._lazy_get_backend(self._route_map.get(self._default_route, self._default_route))
        self._register_metadata(self._build_metadata())
        logger.info('LLMRouter initialised – default route: %s', self._default_route)

    def _build_metadata(self) -> ComponentMetadata:
        return ComponentMetadata(
            id=self._component_id,
            category='service',
            version='1.0.0',
            description='Shared LLM router – resolves logical routes to provider instances.'
        )

    async def call_llm_async(
        self, 
        prompt: str, 
        *, 
        route: str | None = None, 
        stage: str | None = None, 
        role: str | None = None, 
        **kwargs: Any
    ) -> Any:
        """Call LLM asynchronously via specified route"""
        logical_route = route or self._default_route
        backend_key = self._route_map.get(logical_route, logical_route)
        backend = await self._lazy_get_backend(backend_key)
        
        # Merge parameters from parameter service
        params = self._param_service.get_parameters(
            stage=stage, 
            role=role, 
            overrides=kwargs
        )
        
        if self._event_bus:
            await self._event_bus.publish_async('LLM_CALL_STARTED', {
                'route': logical_route,
                'backend': backend_key,
                'prompt_chars': len(prompt)
            })
        
        try:
            response = await backend.call_llm_async(
                prompt, 
                stage=stage or 'default', 
                role=role or 'default', 
                **params
            )
            return response
        finally:
            if self._event_bus:
                await self._event_bus.publish_async('LLM_CALL_COMPLETED', {
                    'route': logical_route,
                    'backend': backend_key
                })

    async def _lazy_get_backend(self, model_key: str) -> LLMPort:
        """Lazily create and cache backend instances"""
        if model_key not in self._backends:
            model_cfg = self._model_defs.get(model_key)
            if not model_cfg:
                raise KeyError(f"LLM model '{model_key}' not defined in configuration.")
            
            logger.debug("Creating backend '%s' via factory: %s", model_key, model_cfg.backend_class)
            backend: LLMPort = create_llm_instance(model_key, {
                'backend': model_cfg.backend_class,
                **model_cfg.kwargs
            })
            self._backends[model_key] = backend
        
        return self._backends[model_key]

    def get_available_routes(self) -> Dict[str, str]:
        """Get all available routes"""
        return dict(self._route_map)