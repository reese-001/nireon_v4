"""
Explorer mechanism for NIREON V4
Generates idea variations through systematic exploration and mutation
"""
import logging
from typing import Any, Dict
from application.components.base import NireonBaseComponent
from application.components.lifecycle import ComponentMetadata
from application.components.results import ProcessResult
from application.context import NireonExecutionContext
from .config import ExplorerConfig

logger = logging.getLogger(__name__)

# Canonical metadata definition for Explorer mechanism
EXPLORER_METADATA = ComponentMetadata(
    id="explorer_default_from_code",  # This ID is for the *definition*
    name="Explorer Default",
    version="0.1.0",
    category="mechanism",
    description="Default Explorer Mechanism for idea generation and exploration.",
    epistemic_tags=["generator", "variation", "mutator"],
    capabilities={'generate_ideas', 'explore_variations', 'idea_mutation'},
    accepts=['SEED_SIGNAL', 'EXPLORATION_REQUEST'],
    produces=['IDEA_GENERATED', 'EXPLORATION_COMPLETE'],
    requires_initialize=True
)

class ExplorerMechanism(NireonBaseComponent):
    """
    Explorer mechanism that generates idea variations through systematic exploration
    """
    ConfigModel = ExplorerConfig  # Link to Pydantic model

    def __init__(self, config: Dict[str, Any], metadata_definition: ComponentMetadata, common_deps=None):
        """
        Initialize Explorer mechanism
        
        Args:
            config: Configuration dictionary for the explorer
            metadata_definition: Instance-specific metadata (ID will be from manifest)
            common_deps: Common mechanism dependencies for DI
        """
        # The 'id' in metadata_definition passed here will be the *instance* ID from the manifest
        super().__init__(config=config, metadata_definition=metadata_definition)
        
        # Validate and type the final config using Pydantic model
        self.cfg: ExplorerConfig = ExplorerConfig(**self.config)
        
        # Store dependencies
        self.common_deps = common_deps
        
        # Exploration state
        self._exploration_count = 0
        self._generated_ideas = []
        
        logger.info(f"ExplorerMechanism '{self.component_id}' (instance of {metadata_definition.name} v{metadata_definition.version}) created")
        logger.debug(f"Explorer config: max_depth={self.cfg.max_depth}, application_rate={self.cfg.application_rate}, strategy={self.cfg.exploration_strategy}")

    async def _initialize_impl(self, context: NireonExecutionContext) -> None:
        """Initialize the explorer mechanism"""
        context.logger.info(f"ExplorerMechanism '{self.component_id}' initializing. Max depth: {self.cfg.max_depth}")
        
        # Validate dependencies if provided
        if self.common_deps:
            if not self.common_deps.embedding_port:
                context.logger.warning("No embedding port available - some exploration features may be limited")
            if not self.common_deps.llm_port:
                context.logger.warning("No LLM port available - LLM-based exploration disabled")
        
        # Initialize exploration strategies
        self._init_exploration_strategies(context)
        
        context.logger.info(f"✓ ExplorerMechanism '{self.component_id}' initialized successfully")

    def _init_exploration_strategies(self, context: NireonExecutionContext) -> None:
        """Initialize exploration strategies based on configuration"""
        strategy = self.cfg.exploration_strategy
        
        if strategy == 'depth_first':
            context.logger.debug("Using depth-first exploration strategy")
        elif strategy == 'breadth_first':
            context.logger.debug("Using breadth-first exploration strategy")
        elif strategy == 'random':
            context.logger.debug("Using random exploration strategy")
        else:
            context.logger.warning(f"Unknown exploration strategy '{strategy}', defaulting to depth-first")

    async def _process_impl(self, data: Any, context: NireonExecutionContext) -> ProcessResult:
        """
        Process exploration request and generate idea variations
        
        Args:
            data: Input data (could be seed idea, exploration parameters, etc.)
            context: Execution context
            
        Returns:
            ProcessResult with generated ideas or exploration results
        """
        context.logger.info(f"ExplorerMechanism '{self.component_id}' processing exploration request")
        context.logger.debug(f"Input data type: {type(data)}, preview: {str(data)[:100]}")
        
        self._exploration_count += 1
        
        try:
            # Extract or create seed idea
            seed_idea = self._extract_seed_idea(data, context)
            
            # Generate variations based on exploration strategy
            variations = await self._generate_variations(seed_idea, context)
            
            # Store generated ideas
            self._generated_ideas.extend(variations)
            
            # Create result
            result_data = {
                "exploration_id": f"exp_{self.component_id}_{self._exploration_count}",
                "seed_idea": seed_idea,
                "variations_generated": len(variations),
                "variations": variations,
                "exploration_strategy": self.cfg.exploration_strategy,
                "total_explorations": self._exploration_count
            }
            
            context.logger.info(f"✓ Explorer '{self.component_id}' generated {len(variations)} variations")
            
            return ProcessResult(
                success=True,
                component_id=self.component_id,
                output_data=result_data,
                message=f"Generated {len(variations)} idea variations using {self.cfg.exploration_strategy} strategy"
            )
            
        except Exception as e:
            context.logger.error(f"Explorer '{self.component_id}' processing failed: {e}", exc_info=True)
            return ProcessResult(
                success=False,
                component_id=self.component_id,
                message=f"Exploration failed: {e}",
                error_code='EXPLORATION_ERROR'
            )

    def _extract_seed_idea(self, data: Any, context: NireonExecutionContext) -> str:
        """Extract or create seed idea from input data"""
        if isinstance(data, str):
            return data
        elif isinstance(data, dict):
            if 'text' in data:
                return data['text']
            elif 'seed' in data:
                return data['seed']
            elif 'idea' in data:
                return data['idea']
        
        # Fallback: create default seed
        default_seed = "Explore new possibilities and generate innovative ideas"
        context.logger.debug(f"Using default seed idea: {default_seed}")
        return default_seed

    async def _generate_variations(self, seed_idea: str, context: NireonExecutionContext) -> list:
        """Generate idea variations using the configured strategy"""
        variations = []
        
        try:
            if self.cfg.exploration_strategy == 'depth_first':
                variations = await self._depth_first_exploration(seed_idea, context)
            elif self.cfg.exploration_strategy == 'breadth_first':
                variations = await self._breadth_first_exploration(seed_idea, context)
            elif self.cfg.exploration_strategy == 'random':
                variations = await self._random_exploration(seed_idea, context)
            else:
                # Fallback to simple variations
                variations = await self._simple_variations(seed_idea, context)
            
        except Exception as e:
            context.logger.error(f"Variation generation failed: {e}")
            # Return simple fallback variations
            variations = await self._simple_variations(seed_idea, context)
        
        return variations

    async def _depth_first_exploration(self, seed_idea: str, context: NireonExecutionContext) -> list:
        """Generate variations using depth-first exploration"""
        variations = []
        current_depth = 0
        
        while current_depth < self.cfg.max_depth and len(variations) < 10:  # Limit for demo
            if current_depth == 0:
                # First level variations
                variations.extend([
                    f"Enhanced version: {seed_idea} with advanced capabilities",
                    f"Alternative approach: {seed_idea} through different methodology",
                    f"Simplified form: Core essence of '{seed_idea}'"
                ])
            else:
                # Deeper variations based on previous level
                base_variation = variations[-1] if variations else seed_idea
                variations.append(f"Deep exploration level {current_depth + 1}: {base_variation} with recursive improvement")
            
            current_depth += 1
        
        context.logger.debug(f"Depth-first exploration generated {len(variations)} variations at depth {current_depth}")
        return variations

    async def _breadth_first_exploration(self, seed_idea: str, context: NireonExecutionContext) -> list:
        """Generate variations using breadth-first exploration"""
        variations = []
        
        # Generate multiple variations at each level
        base_variations = [
            f"Expanded scope: {seed_idea}",
            f"Refined focus: {seed_idea}",
            f"Cross-domain application: {seed_idea}",
            f"Inverse perspective: {seed_idea}"
        ]
        
        variations.extend(base_variations)
        
        # Second level variations
        for base in base_variations[:2]:  # Limit expansion
            variations.append(f"Further development of: {base}")
        
        context.logger.debug(f"Breadth-first exploration generated {len(variations)} variations")
        return variations

    async def _random_exploration(self, seed_idea: str, context: NireonExecutionContext) -> list:
        """Generate variations using random exploration"""
        import random
        
        variation_templates = [
            "Random mutation: {seed} with unexpected twist",
            "Chance combination: {seed} meets serendipity",
            "Stochastic enhancement: {seed} through probability",
            "Emergent property: {seed} with chaotic elements",
            "Quantum variation: {seed} in superposition"
        ]
        
        # Randomly select and apply templates
        selected_templates = random.sample(variation_templates, min(3, len(variation_templates)))
        variations = [template.format(seed=seed_idea) for template in selected_templates]
        
        context.logger.debug(f"Random exploration generated {len(variations)} variations")
        return variations

    async def _simple_variations(self, seed_idea: str, context: NireonExecutionContext) -> list:
        """Generate simple fallback variations"""
        variations = [
            f"Basic variation 1: {seed_idea} with improvement",
            f"Basic variation 2: {seed_idea} with modification",
            f"Basic variation 3: Alternative to {seed_idea}"
        ]
        
        context.logger.debug(f"Simple exploration generated {len(variations)} variations")
        return variations

    async def analyze(self, context: NireonExecutionContext):
        """Analyze exploration performance and patterns"""
        from application.components.results import AnalysisResult
        
        metrics = {
            'total_explorations': self._exploration_count,
            'total_ideas_generated': len(self._generated_ideas),
            'average_variations_per_exploration': len(self._generated_ideas) / max(1, self._exploration_count),
            'exploration_strategy': self.cfg.exploration_strategy,
            'max_configured_depth': self.cfg.max_depth
        }
        
        return AnalysisResult(
            success=True,
            component_id=self.component_id,
            metrics=metrics,
            confidence=0.8,
            message=f"Explorer analysis: {self._exploration_count} explorations completed"
        )

    async def health_check(self, context: NireonExecutionContext):
        """Check explorer health and readiness"""
        from application.components.results import ComponentHealth
        
        status = 'HEALTHY'
        messages = []
        
        if not self.is_initialized:
            status = 'UNHEALTHY'
            messages.append('Explorer not initialized')
        
        if self.error_count > 0:
            status = 'DEGRADED' if status == 'HEALTHY' else status
            messages.append(f'Explorer has {self.error_count} errors')
        
        if self._exploration_count == 0:
            messages.append('No explorations performed yet')
        
        return ComponentHealth(
            component_id=self.component_id,
            status=status,
            message='; '.join(messages) if messages else 'Explorer operational'
        )