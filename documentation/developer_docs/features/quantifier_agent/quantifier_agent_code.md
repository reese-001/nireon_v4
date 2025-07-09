components/mechanisms/analyst_agents/quantifier_agent/
├── __init__.py           # Package exports
├── service.py           # Main QuantifierAgent class (your entry point)
├── config.py            # Pydantic configuration schema
├── metadata.py          # Component metadata definition
├── prompts.py           # All LLM prompt templates
├── analysis_engine.py   # Core analysis logic
└── utils.py             # Helper utilities

# FILE: components/mechanisms/analyst_agents/quantifier_agent/__init__.py
"""
QuantifierAgent subsystem for converting qualitative ideas into quantitative analyses.
"""

from .service import QuantifierAgent
from .config import QuantifierConfig
from .metadata import QUANTIFIER_METADATA

__all__ = ['QuantifierAgent', 'QuantifierConfig', 'QUANTIFIER_METADATA']

# ============================================================================
# FILE: components/mechanisms/analyst_agents/quantifier_agent/config.py
"""
Configuration schema for the QuantifierAgent.
"""

from typing import Dict, List, Literal
from pydantic import BaseModel, Field

class QuantifierConfig(BaseModel):
    """Configuration for the QuantifierAgent mechanism."""
    
    class Config:
        extra = "forbid"
        validate_assignment = True
    
    # Visualization approach
    max_visualizations: int = Field(
        default=1, 
        ge=1, 
        le=3,
        description="Maximum number of visualizations to generate per idea"
    )
    
    llm_approach: Literal["single_call", "iterative"] = Field(
        default="single_call",
        description="LLM call strategy: single comprehensive call vs multiple iterative calls"
    )
    
    # Library constraints
    available_libraries: Dict[str, List[str]] = Field(
        default={
            "core_data": ["numpy", "pandas", "scipy"],
            "visualization": ["matplotlib", "seaborn", "plotly"], 
            "specialized_viz": ["networkx", "wordcloud", "graphviz"],
            "analysis": ["scikit-learn", "statsmodels"]
        },
        description="Curated libraries available for analysis"
    )
    
    # Quality thresholds
    min_request_length: int = Field(
        default=100,
        description="Minimum length for generated Proto requests"
    )
    
    viability_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0, 
        description="Confidence threshold for visualization viability"
    )
    
    # Timeouts and limits
    llm_timeout_seconds: int = Field(
        default=30,
        description="Timeout for individual LLM calls"
    )
    
    enable_mermaid_output: bool = Field(
        default=True,
        description="Allow Mermaid diagram generation as alternative to Python visualizations"
    )

# ============================================================================
# FILE: components/mechanisms/analyst_agents/quantifier_agent/metadata.py
"""
Metadata definition for the QuantifierAgent.
"""

from typing import Final
from core.lifecycle import ComponentMetadata

QUANTIFIER_METADATA: Final = ComponentMetadata(
    id='quantifier_agent_primary',
    name='Quantifier Agent',
    version='3.0.0',
    category='mechanism',
    description='Converts qualitative ideas into quantitative analyses using curated visualization libraries',
    epistemic_tags=['analyzer', 'translator', 'modeler', 'quantifier', 'visualizer'],
    accepts=['TrustAssessmentSignal'],
    produces=['ProtoTaskSignal', 'GenerativeLoopFinishedSignal'],
    requires_initialize=True,
    dependencies={
        'ProtoGenerator': '*', 
        'IdeaService': '*', 
        'MechanismGatewayPort': '*'
    }
)

# ============================================================================
# FILE: components/mechanisms/analyst_agents/quantifier_agent/prompts.py
"""
Prompt templates for the QuantifierAgent LLM interactions.
"""

from typing import Dict, Any

class QuantifierPrompts:
    """Collection of prompt templates for QuantifierAgent."""
    
    @staticmethod
    def comprehensive_analysis(idea_text: str, available_libs: Dict[str, Any]) -> str:
        """Single comprehensive prompt for complete analysis."""
        
        lib_descriptions = []
        for category, libs in available_libs.items():
            category_name = category.replace('_', ' ').title()
            lib_list = ', '.join(libs)
            lib_descriptions.append(f"{category_name}: {lib_list}")
        
        libraries_text = '\n'.join(lib_descriptions)
        
        return f"""You are a visualization expert. Analyze this concept and create a complete implementation plan.

Concept: "{idea_text}"

Available tools:
{libraries_text}

Alternative: Mermaid diagrams (for process flows, relationships, hierarchies)

Complete this analysis:

1. VIABILITY: Can this concept be visualized or analyzed quantitatively? (YES/NO and brief reason)

2. APPROACH: If YES, what's the best approach?
   - Traditional charts (matplotlib/seaborn): For business metrics, comparisons, trends
   - Interactive visualization (plotly): For exploratory data, multi-dimensional views
   - Network analysis (networkx): For relationships, connections, systems
   - Process flow (Mermaid): For workflows, decision trees, organizational structures
   - Text analysis (wordcloud): For content analysis, sentiment, themes
   - Statistical modeling (scikit-learn + visualization): For predictions, clustering, classification

3. IMPLEMENTATION: If viable, write a complete natural language request for the ProtoGenerator that includes:
   - Specific libraries to use from the available set
   - Function name and parameters
   - Computational logic and steps
   - Expected outputs and file names
   - Realistic data assumptions or sample data to use
   - Critical plotting requirements (e.g., numeric positioning for bar charts)

If not viable for quantitative analysis, explain why and stop here.

Your complete analysis:"""

    @staticmethod
    def mermaid_generation(idea_text: str, viz_type: str) -> str:
        """Generate Mermaid diagram syntax."""
        
        return f"""Create a Mermaid diagram for this concept: "{idea_text}"

Suggested diagram type: {viz_type}

Available Mermaid diagram types:
- flowchart: Process flows, decision trees, workflows
- graph: Network relationships, connections, systems
- classDiagram: Object relationships, hierarchies, structures
- sequenceDiagram: Time-based interactions, processes
- gantt: Project timelines, schedules, planning
- gitgraph: Version control, branching processes
- mindmap: Concept relationships, brainstorming
- timeline: Historical events, progression, evolution

Generate valid Mermaid syntax that best represents this concept.
Make it comprehensive and insightful.

Output only the raw Mermaid code:"""

    @staticmethod
    def viability_quick_check(idea_text: str) -> str:
        """Quick viability assessment for filtering."""
        
        return f"""Can this concept be meaningfully visualized, analyzed, or modeled quantitatively?

Concept: "{idea_text}"

Consider:
- Can it be represented with charts, graphs, or diagrams?
- Are there measurable aspects or relationships?
- Could it benefit from data analysis or modeling?
- Would visualization help understand or explore it?

Respond with exactly "YES" or "NO" on the first line, followed by a brief explanation.

Response:"""

# ============================================================================
# FILE: components/mechanisms/analyst_agents/quantifier_agent/analysis_engine.py
"""
Core analysis logic for the QuantifierAgent.
"""

import logging
import re
from typing import Dict, List, Optional, Any, Tuple
from domain.context import NireonExecutionContext
from domain.ports.llm_port import LLMResponse
from .prompts import QuantifierPrompts
from .config import QuantifierConfig

logger = logging.getLogger(__name__)

class AnalysisResult:
    """Result of quantification analysis."""
    
    def __init__(self, viable: bool, approach: str = "", implementation_request: str = "", 
                 libraries: List[str] = None, use_mermaid: bool = False, 
                 mermaid_content: str = "", confidence: float = 0.0):
        self.viable = viable
        self.approach = approach
        self.implementation_request = implementation_request
        self.libraries = libraries or []
        self.use_mermaid = use_mermaid
        self.mermaid_content = mermaid_content
        self.confidence = confidence

class QuantificationAnalysisEngine:
    """Core engine for analyzing ideas and generating visualization strategies."""
    
    def __init__(self, config: QuantifierConfig):
        self.config = config
        self.prompts = QuantifierPrompts()
        
    async def analyze_idea(self, idea_text: str, gateway, context: NireonExecutionContext) -> Optional[AnalysisResult]:
        """Main entry point for idea analysis."""
        
        if self.config.llm_approach == "single_call":
            return await self._single_call_analysis(idea_text, gateway, context)
        else:
            return await self._iterative_analysis(idea_text, gateway, context)
    
    async def _single_call_analysis(self, idea_text: str, gateway, context: NireonExecutionContext) -> Optional[AnalysisResult]:
        """Single comprehensive LLM call analysis."""
        
        logger.info(f"Starting single-call analysis for idea: {idea_text[:80]}...")
        
        prompt = self.prompts.comprehensive_analysis(idea_text, self.config.available_libraries)
        
        response = await self._call_llm(gateway, prompt, "comprehensive_analyst", context)
        if not response:
            return None
            
        return self._parse_comprehensive_response(response.text, idea_text, gateway, context)
    
    async def _iterative_analysis(self, idea_text: str, gateway, context: NireonExecutionContext) -> Optional[AnalysisResult]:
        """Multi-step iterative analysis (fallback approach)."""
        
        logger.info(f"Starting iterative analysis for idea: {idea_text[:80]}...")
        
        # Step 1: Quick viability check
        viability_prompt = self.prompts.viability_quick_check(idea_text)
        viability_response = await self._call_llm(gateway, viability_prompt, "viability_checker", context)
        
        if not viability_response or not self._is_viable_response(viability_response.text):
            logger.info("Idea determined not viable in quick check")
            return AnalysisResult(viable=False, confidence=0.8)
        
        # Step 2: Detailed analysis
        detailed_prompt = self.prompts.comprehensive_analysis(idea_text, self.config.available_libraries)
        detailed_response = await self._call_llm(gateway, detailed_prompt, "detailed_analyst", context)
        
        if not detailed_response:
            return None
            
        return self._parse_comprehensive_response(detailed_response.text, idea_text, gateway, context)
    
    async def _parse_comprehensive_response(self, response_text: str, idea_text: str, 
                                         gateway, context: NireonExecutionContext) -> Optional[AnalysisResult]:
        """Parse the comprehensive LLM response."""
        
        # Extract viability decision
        viable, confidence = self._extract_viability(response_text)
        if not viable:
            return AnalysisResult(viable=False, confidence=confidence)
        
        # Extract approach and implementation
        approach = self._extract_approach(response_text)
        implementation = self._extract_implementation(response_text)
        
        # Check if Mermaid is recommended
        if self._should_use_mermaid(response_text, approach):
            if self.config.enable_mermaid_output:
                mermaid_content = await self._generate_mermaid_diagram(idea_text, approach, gateway, context)
                if mermaid_content:
                    mermaid_request = self._create_mermaid_proto_request(idea_text, mermaid_content)
                    return AnalysisResult(
                        viable=True,
                        approach=approach,
                        implementation_request=mermaid_request,
                        use_mermaid=True,
                        mermaid_content=mermaid_content,
                        confidence=confidence
                    )
        
        # Validate implementation request
        if not implementation or len(implementation) < self.config.min_request_length:
            logger.warning(f"Implementation request too short: {len(implementation)} chars")
            return AnalysisResult(viable=False, confidence=0.3)
        
        # Extract libraries
        libraries = self._extract_libraries(response_text)
        
        return AnalysisResult(
            viable=True,
            approach=approach,
            implementation_request=implementation,
            libraries=libraries,
            confidence=confidence
        )
    
    def _extract_viability(self, response: str) -> Tuple[bool, float]:
        """Extract viability decision and confidence."""
        
        lines = response.split('\n')
        for line in lines:
            if 'VIABILITY:' in line.upper():
                line_upper = line.upper()
                if 'YES' in line_upper:
                    return True, 0.9
                elif 'NO' in line_upper:
                    return False, 0.9
                break
        
        # Fallback: analyze full text for positive/negative indicators
        response_lower = response.lower()
        positive_indicators = ['can be visualized', 'visualization', 'quantified', 'analyzed', 'charted']
        negative_indicators = ['cannot', 'not possible', 'abstract', 'philosophical', 'no way']
        
        positive_count = sum(1 for indicator in positive_indicators if indicator in response_lower)
        negative_count = sum(1 for indicator in negative_indicators if indicator in response_lower)
        
        if positive_count > negative_count:
            return True, 0.6
        else:
            return False, 0.6
    
    def _extract_approach(self, response: str) -> str:
        """Extract the visualization approach."""
        
        lines = response.split('\n')
        for i, line in enumerate(lines):
            if 'APPROACH:' in line.upper():
                # Take this line and the next few lines
                approach_lines = lines[i:i+3]
                return ' '.join(approach_lines).replace('APPROACH:', '').strip()
        
        return "Traditional visualization approach"
    
    def _extract_implementation(self, response: str) -> str:
        """Extract the implementation request."""
        
        lines = response.split('\n')
        implementation_lines = []
        in_implementation = False
        
        for line in lines:
            if 'IMPLEMENTATION:' in line.upper():
                in_implementation = True
                # Include the rest of this line after "IMPLEMENTATION:"
                remainder = line.split('IMPLEMENTATION:', 1)[-1].strip()
                if remainder:
                    implementation_lines.append(remainder)
                continue
            
            if in_implementation:
                if line.strip() and not line.strip().startswith(('1.', '2.', '3.', '4.')):
                    implementation_lines.append(line.strip())
                elif line.strip().startswith(('1.', '2.', '3.', '4.')):
                    implementation_lines.append(line.strip())
        
        return '\n'.join(implementation_lines).strip()
    
    def _extract_libraries(self, response: str) -> List[str]:
        """Extract mentioned libraries from the response."""
        
        all_libs = []
        for category_libs in self.config.available_libraries.values():
            all_libs.extend(category_libs)
        
        response_lower = response.lower()
        found_libs = []
        
        for lib in all_libs:
            if lib.lower() in response_lower:
                found_libs.append(lib)
        
        return list(set(found_libs))  # Remove duplicates
    
    def _should_use_mermaid(self, response: str, approach: str) -> bool:
        """Determine if Mermaid is recommended."""
        
        mermaid_indicators = ['mermaid', 'flowchart', 'process flow', 'workflow', 'decision tree', 'diagram']
        response_lower = response.lower()
        approach_lower = approach.lower()
        
        return any(indicator in response_lower or indicator in approach_lower for indicator in mermaid_indicators)
    
    async def _generate_mermaid_diagram(self, idea_text: str, approach: str, 
                                      gateway, context: NireonExecutionContext) -> Optional[str]:
        """Generate Mermaid diagram content."""
        
        viz_type = self._determine_mermaid_type(approach)
        prompt = self.prompts.mermaid_generation(idea_text, viz_type)
        
        response = await self._call_llm(gateway, prompt, "mermaid_generator", context)
        if response and response.text.strip():
            return response.text.strip()
        
        return None
    
    def _determine_mermaid_type(self, approach: str) -> str:
        """Determine the best Mermaid diagram type."""
        
        approach_lower = approach.lower()
        
        if any(word in approach_lower for word in ['process', 'workflow', 'flow']):
            return "flowchart"
        elif any(word in approach_lower for word in ['network', 'relationship', 'connection']):
            return "graph"
        elif any(word in approach_lower for word in ['hierarchy', 'structure', 'organization']):
            return "classDiagram"
        elif any(word in approach_lower for word in ['timeline', 'sequence', 'time']):
            return "timeline"
        else:
            return "flowchart"
    
    def _create_mermaid_proto_request(self, idea_text: str, mermaid_content: str) -> str:
        """Create a Proto request for Mermaid diagram output."""
        
        return f"""Create a Python function that generates a Mermaid diagram for the concept: "{idea_text}"

The function should:
1. Output the following Mermaid diagram syntax
2. Save it to a file called 'diagram.mmd'
3. Also save a copy to 'mermaid_output.txt' with rendering instructions
4. Print instructions for how to render the diagram

Mermaid diagram content:
```
{mermaid_content}
```

Create a complete Python function that handles this output and provides user instructions for rendering the diagram."""
    
    def _is_viable_response(self, response: str) -> bool:
        """Check if a response indicates viability."""
        
        first_line = response.split('\n')[0].strip().upper()
        return 'YES' in first_line
    
    async def _call_llm(self, gateway, prompt: str, role: str, context: NireonExecutionContext) -> Optional[LLMResponse]:
        """Helper method for LLM calls."""
        
        if not gateway:
            logger.error("Gateway not available for LLM calls")
            return None
        
        from domain.cognitive_events import CognitiveEvent
        from domain.epistemic_stage import EpistemicStage
        
        frame_id = context.metadata.get('current_frame_id', f"quantifier_{context.run_id}")
        
        try:
            event = CognitiveEvent.for_llm_ask(
                frame_id=frame_id,
                owning_agent_id="quantifier_agent",
                prompt=prompt,
                stage=EpistemicStage.SYNTHESIS,
                role=role
            )
            
            response = await gateway.process_cognitive_event(event, context)
            
            if isinstance(response, LLMResponse):
                return response
            else:
                logger.error(f"Unexpected response type from gateway: {type(response)}")
                return None
                
        except Exception as e:
            logger.error(f"Error calling LLM via gateway: {e}")
            return None

# ============================================================================
# FILE: components/mechanisms/analyst_agents/quantifier_agent/service.py
"""
Main QuantifierAgent service implementation.
"""

from __future__ import annotations
import asyncio
import logging
from typing import Any, Dict, Optional, Final
from core.base_component import NireonBaseComponent
from core.results import ProcessResult
from domain.context import NireonExecutionContext
from signals.core import ProtoTaskSignal, GenerativeLoopFinishedSignal
from .config import QuantifierConfig
from .metadata import QUANTIFIER_METADATA
from .analysis_engine import QuantificationAnalysisEngine

__all__ = ['QuantifierAgent']
logger = logging.getLogger(__name__)

class QuantifierAgent(NireonBaseComponent):
    """
    Advanced quantification agent that converts qualitative ideas into executable analyses.
    
    This agent uses a curated set of visualization libraries and intelligent LLM-driven
    analysis to determine the best way to quantify and visualize abstract concepts.
    """
    
    METADATA_DEFINITION = QUANTIFIER_METADATA
    ConfigModel = QuantifierConfig

    def __init__(self, config: Dict[str, Any], metadata_definition=None) -> None:
        super().__init__(config, metadata_definition or self.METADATA_DEFINITION)
        
        # Initialize configuration
        self.cfg: QuantifierConfig = self.ConfigModel(**self.config)
        
        # Initialize analysis engine
        self.analysis_engine = QuantificationAnalysisEngine(self.cfg)
        
        # Dependencies (resolved during initialization)
        self.proto_generator = None
        self.idea_service = None
        self.gateway = None

    async def _initialize_impl(self, context: NireonExecutionContext) -> None:
        """Initialize dependencies and validate configuration."""
        
        self._resolve_dependencies(context)
        self._validate_dependencies()
        self._log_configuration(context)
        
        context.logger.info(f"QuantifierAgent '{self.component_id}' initialized successfully.")

    async def _process_impl(self, data: Any, context: NireonExecutionContext) -> ProcessResult:
        """
        Main processing logic for quantifying ideas.
        
        Args:
            data: Dictionary containing 'idea_id', 'idea_text', and optional 'assessment_details'
            context: Execution context with registry, logging, etc.
            
        Returns:
            ProcessResult indicating success/failure and any output data
        """
        
        logger.info('=== QUANTIFIER AGENT PROCESSING START ===')
        logger.debug(f"Input data keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
        
        # Extract and validate input
        idea_id = data.get('idea_id')
        idea_text = data.get('idea_text')
        
        if not self._validate_input_data(idea_id, idea_text):
            return self._create_error_result("Missing required input: 'idea_id' or 'idea_text'")

        context.logger.info(f"[{self.component_id}] Analyzing idea '{idea_id}': {idea_text[:100]}...")

        # Perform quantification analysis
        analysis_result = await self.analysis_engine.analyze_idea(idea_text, self.gateway, context)
        
        if not analysis_result or not analysis_result.viable:
            # Idea cannot be quantified - complete the loop gracefully
            return await self._handle_non_quantifiable_idea(data, context)

        # Generate Proto task for quantifiable idea
        return await self._trigger_proto_generation(analysis_result, idea_id, data, context)

    def _resolve_dependencies(self, context: NireonExecutionContext) -> None:
        """Resolve required dependencies from the component registry."""
        
        try:
            # Import here to avoid circular dependencies
            from proto_generator.service import ProtoGenerator
            from application.services.idea_service import IdeaService
            from domain.ports.mechanism_gateway_port import MechanismGatewayPort
            
            if not self.proto_generator:
                self.proto_generator = context.component_registry.get('proto_generator_main')
            if not self.idea_service:
                self.idea_service = context.component_registry.get_service_instance(IdeaService)
            if not self.gateway:
                self.gateway = context.component_registry.get_service_instance(MechanismGatewayPort)
                
        except Exception as e:
            logger.error(f"Failed to resolve dependencies: {e}")
            raise

    def _validate_dependencies(self) -> None:
        """Validate that all required dependencies are available."""
        
        missing_deps = []
        
        if not self.proto_generator:
            missing_deps.append("ProtoGenerator")
        if not self.idea_service:
            missing_deps.append("IdeaService") 
        if not self.gateway:
            missing_deps.append("MechanismGatewayPort")
            
        if missing_deps:
            raise RuntimeError(
                f"QuantifierAgent '{self.component_id}' missing dependencies: {', '.join(missing_deps)}"
            )

    def _log_configuration(self, context: NireonExecutionContext) -> None:
        """Log current configuration for debugging."""
        
        context.logger.info(f"QuantifierAgent configuration:")
        context.logger.info(f"  - LLM approach: {self.cfg.llm_approach}")
        context.logger.info(f"  - Max visualizations: {self.cfg.max_visualizations}")
        context.logger.info(f"  - Mermaid enabled: {self.cfg.enable_mermaid_output}")
        context.logger.info(f"  - Available libraries: {len(sum(self.cfg.available_libraries.values(), []))}")

    async def _handle_non_quantifiable_idea(self, data: Dict[str, Any], 
                                          context: NireonExecutionContext) -> ProcessResult:
        """Handle ideas that cannot be quantified."""
        
        context.logger.info(f"[{self.component_id}] Idea not suitable for quantification")
        
        # Build completion payload
        assessment_details = data.get('assessment_details', {})
        completion_payload = self._build_completion_payload(assessment_details, quantifier_triggered=False)
        
        # Emit completion signal
        completion_signal = GenerativeLoopFinishedSignal(
            source_node_id=self.component_id,
            payload=completion_payload
        )
        
        if context.event_bus:
            await asyncio.to_thread(
                context.event_bus.publish,
                completion_signal.signal_type,
                completion_signal
            )
            
        return self._create_success_result(
            'Idea was not suitable for quantitative analysis - loop completed gracefully'
        )

    async def _trigger_proto_generation(self, analysis_result, idea_id: str, 
                                      original_data: Dict[str, Any], 
                                      context: NireonExecutionContext) -> ProcessResult:
        """Trigger Proto block generation for quantifiable ideas."""
        
        context.logger.info(f"[{self.component_id}] Triggering ProtoGenerator for idea '{idea_id}'")
        context.logger.debug(f"[{self.component_id}] Implementation approach: {analysis_result.approach}")
        
        try:
            # Call ProtoGenerator
            generator_result = await self.proto_generator.process(
                {'natural_language_request': analysis_result.implementation_request},
                context
            )
            
            if generator_result.success:
                # Build success payload
                output_data = {
                    'proto_generation_result': generator_result.output_data,
                    'analysis_approach': analysis_result.approach,
                    'libraries_used': analysis_result.libraries,
                    'uses_mermaid': analysis_result.use_mermaid,
                    'confidence': analysis_result.confidence
                }
                
                return self._create_success_result(
                    f"Successfully triggered quantitative analysis for idea '{idea_id}'",
                    output_data
                )
            else:
                # ProtoGenerator failed - complete loop gracefully
                context.logger.warning(f"ProtoGenerator failed: {generator_result.message}")
                return await self._handle_non_quantifiable_idea(original_data, context)
                
        except Exception as exc:
            logger.exception(f'Error triggering proto generation for idea {idea_id}')
            return self._create_error_result(f'Proto generation failed: {exc}')

    def _build_completion_payload(self, assessment_details: Dict[str, Any], 
                                quantifier_triggered: bool) -> Dict[str, Any]:
        """Build payload for GenerativeLoopFinishedSignal."""
        
        final_depth = assessment_details.get('metadata', {}).get('depth', 0)
        
        return {
            'status': 'completed_one_branch',
            'final_idea_id': assessment_details.get('idea_id'),
            'final_trust_score': assessment_details.get('trust_score'),
            'final_depth': final_depth,
            'quantifier_triggered': quantifier_triggered
        }

    @staticmethod
    def _validate_input_data(idea_id: Any, idea_text: Any) -> bool:
        """Validate required input data."""
        return bool(idea_id and idea_text and isinstance(idea_text, str))

    def _create_error_result(self, message: str) -> ProcessResult:
        """Create an error ProcessResult."""
        return ProcessResult(
            success=False,
            component_id=self.component_id,
            message=message
        )

    def _create_success_result(self, message: str, 
                             output_data: Optional[Dict[str, Any]] = None) -> ProcessResult:
        """Create a success ProcessResult."""
        return ProcessResult(
            success=True,
            component_id=self.component_id,
            message=message,
            output_data=output_data
        )

# ============================================================================
# FILE: components/mechanisms/analyst_agents/quantifier_agent/utils.py
"""
Utility functions for the QuantifierAgent subsystem.
"""

import re
from typing import List, Dict, Any, Set

class LibraryExtractor:
    """Utility for extracting Python library requirements from code or text."""
    
    # Common module to package mappings
    MODULE_TO_PACKAGE = {
        'cv2': 'opencv-python',
        'sklearn': 'scikit-learn', 
        'PIL': 'Pillow',
        'yaml': 'PyYAML',
        'bs4': 'beautifulsoup4',
        'requests': 'requests',
        'np': 'numpy',
        'pd': 'pandas',
        'plt': 'matplotlib',
        'sns': 'seaborn'
    }
    
    # Standard library modules to ignore
    STDLIB_MODULES = {
        'sys', 'os', 'math', 'json', 'time', 'datetime', 'random', 
        'collections', 'itertools', 'functools', 'pathlib', 're',
        'logging', 'typing', 'abc', 'dataclasses', 'enum'
    }
    
    @classmethod
    def extract_from_code(cls, code: str) -> List[str]:
        """Extract library requirements from Python code."""
        
        import ast
        requirements = set()
        
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module = alias.name.split('.')[0]
                        requirements.add(module)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        module = node.module.split('.')[0]
                        requirements.add(module)
        except SyntaxError:
            # Fallback to regex if AST parsing fails
            requirements.update(cls._extract_with_regex(code))
        
        return cls._map_to_packages(requirements)
    
    @classmethod
    def extract_from_text(cls, text: str, available_libraries: Dict[str, List[str]]) -> List[str]:
        """Extract mentioned libraries from text description."""
        
        all_libs = []
        for category_libs in available_libraries.values():
            all_libs.extend(category_libs)
        
        text_lower = text.lower()
        found_libs = []
        
        for lib in all_libs:
            if lib.lower() in text_lower:
                found_libs.append(lib)
        
        return list(set(found_libs))
    
    @classmethod
    def _extract_with_regex(cls, code: str) -> Set[str]:
        """Extract imports using regex as fallback."""
        
        requirements = set()
        
        # Match import statements
        import_patterns = [
            r'^import\s+([a-zA-Z_][a-zA-Z0-9_]*)',
            r'^from\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+import'
        ]
        
        for line in code.split('\n'):
            line = line.strip()
            for pattern in import_patterns:
                match = re.match(pattern, line)
                if match:
                    module = match.group(1).split('.')[0]
                    requirements.add(module)
        
        return requirements
    
    @classmethod
    def _map_to_packages(cls, modules: Set[str]) -> List[str]:
        """Map module names to PyPI package names."""
        
        packages = []
        
        for module in modules:
            if module in cls.STDLIB_MODULES:
                continue
                
            if module in cls.MODULE_TO_PACKAGE:
                packages.append(cls.MODULE_TO_PACKAGE[module])
            else:
                packages.append(module)
        
        return sorted(list(set(packages)))

class ResponseParser:
    """Utility for parsing structured LLM responses."""
    
    @staticmethod
    def extract_section(text: str, section_name: str) -> str:
        """Extract a named section from structured text."""
        
        lines = text.split('\n')
        section_lines = []
        in_section = False
        
        section_pattern = f"{section_name.upper()}:"
        
        for line in lines:
            if section_pattern in line.upper():
                in_section = True
                # Include remainder of this line
                remainder = line.split(':', 1)[-1].strip()
                if remainder:
                    section_lines.append(remainder)
                continue
            
            if in_section:
                # Stop at next section or empty line
                if ':' in line and line.strip().isupper():
                    break
                if line.strip():
                    section_lines.append(line.strip())
        
        return '\n'.join(section_lines).strip()
    
    @staticmethod
    def extract_yes_no_decision(text: str) -> tuple[bool, float]:
        """Extract YES/NO decision with confidence from text."""
        
        first_line = text.split('\n')[0].strip().upper()
        
        # Direct matches
        if 'YES' in first_line:
            return True, 0.9
        elif 'NO' in first_line:
            return False, 0.9
        
        # Analyze full text for indicators
        text_lower = text.lower()
        positive_words = ['possible', 'viable', 'feasible', 'can be', 'visualized', 'analyzed']
        negative_words = ['impossible', 'not possible', 'cannot', 'not viable', 'abstract only']
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            return True, 0.6
        elif negative_count > positive_count:
            return False, 0.6
        else:
            return False, 0.3  # Default to not viable if unclear

class ConfigurationValidator:
    """Utility for validating QuantifierAgent configurations."""
    
    @staticmethod
    def validate_library_config(available_libraries: Dict[str, List[str]]) -> List[str]:
        """Validate library configuration and return any issues."""
        
        issues = []
        
        # Check required categories
        required_categories = ['core_data', 'visualization']
        for category in required_categories:
            if category not in available_libraries:
                issues.append(f"Missing required library category: {category}")
        
        # Check for empty categories
        for category, libs in available_libraries.items():
            if not libs:
                issues.append(f"Empty library category: {category}")
        
        # Check for known problematic combinations
        all_libs = sum(available_libraries.values(), [])
        if 'matplotlib' in all_libs and 'plotly' in all_libs:
            # This is actually fine, just noting
            pass
        
        return issues
    
    @staticmethod
    def estimate_resource_usage(config: 'QuantifierConfig') -> Dict[str, Any]:
        """Estimate resource usage for a given configuration."""
        
        # Rough estimates based on configuration
        calls_per_idea = 1 if config.llm_approach == "single_call" else 3
        tokens_per_call = 2000 if config.llm_approach == "single_call" else 1500
        
        return {
            'llm_calls_per_idea': calls_per_idea,
            'estimated_tokens_per_idea': calls_per_idea * tokens_per_call,
            'visualizations_per_idea': config.max_visualizations,
            'supports_mermaid': config.enable_mermaid_output
        }