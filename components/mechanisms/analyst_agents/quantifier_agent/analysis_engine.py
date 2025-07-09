# nireon_v4/components/mechanisms/analyst_agents/quantifier_agent/analysis_engine.py
import logging
from typing import Dict, List, Optional, Any, Tuple
from domain.context import NireonExecutionContext
from domain.ports.llm_port import LLMResponse
from .prompts import QuantifierPrompts
from .config import QuantifierConfig
# Import the unified parser
from components.common.llm_response_parser import (
    ParserFactory, ParseResult, ParseStatus,
    FieldSpec, BooleanFieldExtractor, TextFieldExtractor, NumericFieldExtractor
)

logger = logging.getLogger(__name__)


class AnalysisResult:
    def __init__(self, viable: bool, approach: str='', implementation_request: str='', 
                 libraries: List[str]=None, use_mermaid: bool=False, mermaid_content: str='', 
                 confidence: float=0.0):
        self.viable = viable
        self.approach = approach
        self.implementation_request = implementation_request
        self.libraries = libraries or []
        self.use_mermaid = use_mermaid
        self.mermaid_content = mermaid_content
        self.confidence = confidence


class QuantificationAnalysisEngine:
    def __init__(self, config: QuantifierConfig):
        self.config = config
        self.prompts = QuantifierPrompts()
        # Create parsers for different types of responses
        self.viability_parser, self.viability_specs = ParserFactory.create_viability_parser()
        self.comprehensive_parser, self.comprehensive_specs = self._create_comprehensive_parser()
    
    def _create_comprehensive_parser(self) -> Tuple:
        """Create a custom parser for comprehensive analysis"""
        from components.common.llm_response_parser import LLMResponseParser, FieldSpec
        
        parser = LLMResponseParser()
        specs = [
            FieldSpec(
                name="viable",
                extractor=BooleanFieldExtractor("viability"),
                default=False,
                required=True
            ),
            FieldSpec(
                name="approach",
                extractor=TextFieldExtractor("approach", min_length=10),
                default="Traditional visualization approach",
                required=False
            ),
            FieldSpec(
                name="implementation",
                extractor=TextFieldExtractor("implementation", min_length=50, multiline=True),
                default="",
                required=False
            ),
            FieldSpec(
                name="confidence",
                extractor=NumericFieldExtractor("confidence", min_val=0.0, max_val=1.0),
                default=0.5,
                required=False
            )
        ]
        return parser, specs
    
    async def analyze_idea(self, idea_text: str, gateway, context: NireonExecutionContext) -> Optional[AnalysisResult]:
        if self.config.llm_approach == 'single_call':
            return await self._single_call_analysis(idea_text, gateway, context)
        else:
            return await self._iterative_analysis(idea_text, gateway, context)
    
    async def _single_call_analysis(self, idea_text: str, gateway, context: NireonExecutionContext) -> Optional[AnalysisResult]:
        logger.info(f'Starting single-call analysis for idea: {idea_text[:80]}...')
        
        prompt = self.prompts.comprehensive_analysis(idea_text, self.config.available_libraries)
        response = await self._call_llm(gateway, prompt, 'comprehensive_analyst', context)
        
        if not response:
            return None
        
        return await self._parse_comprehensive_response(response.text, idea_text, gateway, context)
    
    async def _iterative_analysis(self, idea_text: str, gateway, context: NireonExecutionContext) -> Optional[AnalysisResult]:
        logger.info(f'Starting iterative analysis for idea: {idea_text[:80]}...')
        
        # Quick viability check
        viability_prompt = self.prompts.viability_quick_check(idea_text)
        viability_response = await self._call_llm(gateway, viability_prompt, 'viability_checker', context)
        
        if not viability_response:
            return None
            
        # Parse viability response using unified parser
        viability_result = self.viability_parser.parse(
            viability_response.text, 
            self.viability_specs,
            'quantifier_agent'
        )
        
        if not viability_result.data['viable']:
            logger.info('Idea determined not viable in quick check')
            return AnalysisResult(
                viable=False, 
                confidence=viability_result.data.get('confidence', 0.8)
            )
        
        # Detailed analysis
        detailed_prompt = self.prompts.comprehensive_analysis(idea_text, self.config.available_libraries)
        detailed_response = await self._call_llm(gateway, detailed_prompt, 'detailed_analyst', context)
        
        if not detailed_response:
            return None
        
        return await self._parse_comprehensive_response(detailed_response.text, idea_text, gateway, context)
    
    async def _parse_comprehensive_response(self, response_text: str, idea_text: str, gateway, context: NireonExecutionContext) -> Optional[AnalysisResult]:
        result = self.comprehensive_parser.parse(response_text, self.comprehensive_specs, 'quantifier_agent')
        for warning in result.warnings:
            logger.warning(f'Parse warning: {warning}')
        
        # Use .get() with defaults to safely access the data
        viable = result.data.get('viable', False)
        confidence = result.data.get('confidence', 0.9 if result.is_success else 0.3)
        
        if not viable:
            rejection_reason = self._extract_rejection_reason(response_text)
            logger.info(f'Quantification deemed not viable. Reason: {rejection_reason}')
            return AnalysisResult(viable=False, confidence=confidence)
        
        # ... rest of the method
        
        approach = result.data['approach']
        implementation = result.data['implementation']
        
        # Check if we should use Mermaid
        if self._should_use_mermaid(response_text, approach):
            if self.config.enable_mermaid_output:
                mermaid_content = await self._generate_mermaid_diagram(
                    idea_text, approach, gateway, context
                )
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
        
        # Validate implementation length
        if not implementation or len(implementation) < self.config.min_request_length:
            logger.warning(f'Implementation request too short: {len(implementation)} chars. Full response:\n{response_text}')
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
    
    def _extract_rejection_reason(self, response: str) -> str:
        """Extract rejection reason from response text"""
        # Look for common rejection patterns
        patterns = [
            r'not viable[:\s]+(.+?)(?:\.|$)',
            r'cannot be (?:visualized|quantified)[:\s]+(.+?)(?:\.|$)',
            r'reason[:\s]+(.+?)(?:\.|$)'
        ]
        
        import re
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()
        
        return 'No specific reason provided.'
    
    def _extract_libraries(self, response: str) -> List[str]:
        """Extract mentioned libraries from response text"""
        all_libs = []
        for category_libs in self.config.available_libraries.values():
            all_libs.extend(category_libs)
        
        response_lower = response.lower()
        found_libs = []
        for lib in all_libs:
            if lib.lower() in response_lower:
                found_libs.append(lib)
        
        return list(set(found_libs))
    
    def _should_use_mermaid(self, response: str, approach: str) -> bool:
        """Check if Mermaid diagram should be used"""
        mermaid_indicators = ['mermaid', 'flowchart', 'process flow', 'workflow', 
                             'decision tree', 'diagram']
        response_lower = response.lower()
        approach_lower = approach.lower()
        
        return any(indicator in response_lower or indicator in approach_lower 
                  for indicator in mermaid_indicators)
    
    async def _generate_mermaid_diagram(self, idea_text: str, approach: str, 
                                       gateway, context: NireonExecutionContext) -> Optional[str]:
        """Generate Mermaid diagram"""
        viz_type = self._determine_mermaid_type(approach)
        prompt = self.prompts.mermaid_generation(idea_text, viz_type)
        
        response = await self._call_llm(gateway, prompt, 'mermaid_generator', context)
        if response and response.text.strip():
            return response.text.strip()
        return None
    
    def _determine_mermaid_type(self, approach: str) -> str:
        """Determine Mermaid diagram type from approach"""
        approach_lower = approach.lower()
        
        if any(word in approach_lower for word in ['process', 'workflow', 'flow']):
            return 'flowchart'
        elif any(word in approach_lower for word in ['network', 'relationship', 'connection']):
            return 'graph'
        elif any(word in approach_lower for word in ['hierarchy', 'structure', 'organization']):
            return 'classDiagram'
        elif any(word in approach_lower for word in ['timeline', 'sequence', 'time']):
            return 'timeline'
        else:
            return 'flowchart'
    
    def _create_mermaid_proto_request(self, idea_text: str, mermaid_content: str) -> str:
        """Create Proto request for Mermaid diagram"""
        return f'''Create a Python function that generates a Mermaid diagram for the concept: "{idea_text}"

The function should:
1. Output the following Mermaid diagram syntax
2. Save it to a file called 'diagram.mmd'
3. Also save a copy to 'mermaid_output.txt' with rendering instructions
4. Print instructions for how to render the diagram

Mermaid diagram content:
{mermaid_content}

Create a complete Python function that handles this output and provides user instructions for rendering the diagram.'''
    
    async def _call_llm(self, gateway, prompt: str, role: str, 
                       context: NireonExecutionContext) -> Optional[LLMResponse]:
        """Call LLM via gateway"""
        if not gateway:
            logger.error('Gateway not available for LLM calls')
            return None
        
        from domain.cognitive_events import CognitiveEvent
        from domain.epistemic_stage import EpistemicStage
        
        frame_id = context.metadata.get('current_frame_id', f'quantifier_{context.run_id}')
        
        try:
            event = CognitiveEvent.for_llm_ask(
                frame_id=frame_id,
                owning_agent_id='quantifier_agent',
                prompt=prompt,
                stage=EpistemicStage.SYNTHESIS,
                role=role
            )
            
            response = await gateway.process_cognitive_event(event, context)
            
            if isinstance(response, LLMResponse):
                return response
            else:
                logger.error(f'Unexpected response type from gateway: {type(response)}')
                return None
                
        except Exception as e:
            logger.error(f'Error calling LLM via gateway: {e}')
            return None