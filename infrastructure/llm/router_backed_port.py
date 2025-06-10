# nireon_v4/infrastructure/llm/router_backed_port.py
from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Mapping, Any

from domain.ports.llm_port import LLMPort, LLMResponse
from domain.epistemic_stage import EpistemicStage
from domain.context import NireonExecutionContext

if TYPE_CHECKING:
    from infrastructure.llm.router import LLMRouter
    from core.results import ComponentHealth


class RouterBackedLLMPort(LLMPort):
    def __init__(self, router: LLMRouter):
        self.router = router
        self.component_id = "router_backed_llm_port" # Could be made dynamic if needed

    async def call_llm_async(self, prompt: str, *,
                             stage: EpistemicStage,
                             role: str,
                             context: NireonExecutionContext, 
                             settings: Optional[Mapping[str, Any]] = None, **kwargs) -> LLMResponse:
        return await self.router.call_llm_async(prompt, stage=stage, role=role, context=context, settings=settings, **kwargs)

    def call_llm_sync(self, prompt: str, *,
                        stage: EpistemicStage,
                        role: str,
                        context: NireonExecutionContext, 
                        settings: Optional[Mapping[str, Any]] = None, **kwargs) -> LLMResponse:
        return self.router.call_llm_sync(prompt, stage=stage, role=role, context=context, settings=settings, **kwargs)

    async def health_check(self, context: NireonExecutionContext) -> ComponentHealth:
        """Passthrough health check to the underlying router if available."""
        if hasattr(self.router, 'health_check') and callable(self.router.health_check):
            return await self.router.health_check(context)
        from core.results import ComponentHealth # Local import
        return ComponentHealth(component_id=self.component_id, status="HEALTHY", message="RouterBackedLLMPort is operational, router health unknown.")