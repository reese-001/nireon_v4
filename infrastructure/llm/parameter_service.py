# C:\Users\erees\Documents\development\nireon_v4\infrastructure\llm\parameter_service.py
from __future__ import annotations
import ast
import logging
from typing import Any, Dict, Mapping, MutableMapping, Optional

from core.base_component import NireonBaseComponent
from core.lifecycle import ComponentMetadata 
from domain.context import NireonExecutionContext, EpistemicStage

logger = logging.getLogger(__name__)
__all__ = ['LLMParameterService']


class _SafeEvalVisitor(ast.NodeVisitor):
    SAFE_NODES = {ast.Module, ast.Expr, ast.Load, ast.Name, ast.Attribute, ast.Compare, ast.BoolOp, ast.UnaryOp, ast.BinOp, ast.IfExp, ast.Num, ast.Constant, ast.Subscript, ast.Index, ast.Slice, ast.And, ast.Or, ast.Not, ast.Gt, ast.GtE, ast.Lt, ast.LtE, ast.Eq, ast.NotEq, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow}

    def generic_visit(self, node):
        if type(node) not in self.SAFE_NODES:
            raise ValueError(f'Disallowed AST node in dynamic rule: {type(node).__name__}')
        super().generic_visit(node)


class ParameterService(NireonBaseComponent):
    # Modified __init__ to accept metadata_definition and call super correctly
    def __init__(self, config: Optional[Mapping[str, Any]]=None, metadata_definition: Optional[ComponentMetadata]=None):
        # Ensure metadata_definition is provided or create a default one.
        if metadata_definition is None:
            # This fallback is for cases where ParameterService might be instantiated directly
            # without a pre-defined metadata object (e.g., in tests or other contexts).
            # In the LLMRouter case, param_metadata is explicitly passed.
            default_id = "llm_parameter_service_auto" # A generic default ID
            if config and isinstance(config, Mapping): # Attempt to make ID more specific if possible
                default_id = str(config.get('id', config.get('component_id', default_id)))

            metadata_definition = ComponentMetadata(
                id=default_id,
                name='LLMParameterService',
                category='service', # Aligning with how it's used in LLMRouter
                version='1.0.0',
                description='Central generator-parameter resolver for LLM calls (default metadata).',
                requires_initialize=False # As typically configured
            )
            logger.debug(f"ParameterService.__init__: metadata_definition was None, created default: {metadata_definition.id}")

        # Call NireonBaseComponent's __init__ with config and metadata_definition
        # NireonBaseComponent's __init__ expects config as Dict[str, Any]
        super().__init__(config=dict(config or {}), metadata_definition=metadata_definition)
        
        # self.config is now set by NireonBaseComponent's __init__.
        # These should use self.config.
        self._defaults: Mapping[str, Any] = self.config.get('defaults', {})
        self._by_stage: Mapping[str, Mapping[str, Any]] = self.config.get('by_stage', {})
        self._by_role: Mapping[str, Mapping[str, Any]] = self.config.get('by_role', {})
        self._dynamic_rules: Mapping[str, str] = self.config.get('dynamic_rules', {})

    def _process_impl(self, prompt: str, context: dict) -> str: # context type should be NireonExecutionContext
        # This method seems unused in the typical flow of ParameterService.
        # If it's meant to be part of NireonBaseComponent's lifecycle,
        # its signature and purpose should align.
        # For now, keeping as-is but noting potential mismatch.
        # If ParameterService doesn't "process" data, this could be removed or raise NotImplementedError.
        logger.warning(f"ParameterService '{self.component_id}' _process_impl called, but it's not its primary function.")
        return f'ParameterService processed context with {len(context)} items'

    def get_parameters(self, *, stage: EpistemicStage, role: str, ctx: NireonExecutionContext, overrides: Optional[Mapping[str, Any]]=None) -> Dict[str, Any]:
        resolved: Dict[str, Any] = {}
        for key, val in self._defaults.items():
            resolved.setdefault(key, val)
        
        stage_name = stage.value if isinstance(stage, EpistemicStage) else str(stage)
        stage_cfg = self._by_stage.get(stage_name, {}) # Use stage.value or str(stage) for dict keys
        resolved.update({k: v for k, v in stage_cfg.items() if v is not None})
        
        role_cfg = self._by_role.get(role, {})
        resolved.update({k: v for k, v in role_cfg.items() if v is not None})
        
        dyn_vals = self._eval_dynamic_rules(ctx)
        resolved.update({k: v for k, v in dyn_vals.items() if v is not None})
        
        if overrides:
            resolved.update({k: v for k, v in overrides.items() if v is not None})
        return resolved

    # Corrected initialize method to _initialize_impl
    async def _initialize_impl(self, context: NireonExecutionContext) -> None:
        # This service is typically marked with requires_initialize=False.
        # If any specific initialization actions were needed, they would go here.
        logger.debug(f"ParameterService '{self.component_id}' _initialize_impl: No specific initialization actions required.")
        pass # No custom initialization logic seems necessary for ParameterService

    def _eval_dynamic_rules(self, ctx: NireonExecutionContext) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for param_key, expr in self._dynamic_rules.items():
            try:
                tree = ast.parse(expr, mode='eval')
                _SafeEvalVisitor().visit(tree)
                # Ensure the evaluation context has what the rules expect.
                # 'ctx' is the NireonExecutionContext.
                eval_globals = {'ctx': ctx, 'EpistemicStage': EpistemicStage} # Add EpistemicStage if rules use it
                value = eval(compile(tree, filename='<llm_rule>', mode='eval'), eval_globals)
                out[param_key] = value
            except Exception as exc:
                logger.warning('Dynamic LLM param rule failed for %s – %s', param_key, exc)
        return out

LLMParameterService = ParameterService