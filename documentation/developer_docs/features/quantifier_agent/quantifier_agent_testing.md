"""
Additional tests to ensure NIREON V4 compliance
"""
import pytest
from unittest.mock import Mock, AsyncMock
from domain.context import NireonExecutionContext
from domain.cognitive_events import CognitiveEvent
from signals.core import TrustAssessmentSignal, ProtoTaskSignal

class TestNireonCompliance:
    """Tests for NIREON V4 framework compliance."""
    
    @pytest.fixture
    def mock_context(self):
        """Create a mock NIREON execution context."""
        context = Mock(spec=NireonExecutionContext)
        context.run_id = "test_run_123"
        context.metadata = {'current_frame_id': 'test_frame_001'}
        context.logger = Mock()
        context.component_registry = Mock()
        return context
    
    async def test_cognitive_event_creation(self, mock_context):
        """Test that LLM calls properly create CognitiveEvents."""
        from components.mechanisms.analyst_agents.quantifier_agent.analysis_engine import QuantificationAnalysisEngine
        from components.mechanisms.analyst_agents.quantifier_agent.config import QuantifierConfig
        
        config = QuantifierConfig()
        engine = QuantificationAnalysisEngine(config)
        
        # Mock gateway
        mock_gateway = AsyncMock()
        mock_gateway.process_cognitive_event = AsyncMock()
        
        # Capture the CognitiveEvent passed to gateway
        captured_event = None
        async def capture_event(event, context):
            nonlocal captured_event
            captured_event = event
            # Return a mock LLM response
            from domain.ports.llm_port import LLMResponse
            return LLMResponse(text="VIABILITY: YES\nAPPROACH: Use matplotlib\nIMPLEMENTATION: Create bar chart")
        
        mock_gateway.process_cognitive_event.side_effect = capture_event
        
        # Test the analysis
        await engine.analyze_idea("Test idea", mock_gateway, mock_context)
        
        # Verify CognitiveEvent structure
        assert captured_event is not None
        assert isinstance(captured_event, CognitiveEvent)
        assert captured_event.frame_id == 'test_frame_001'
        assert captured_event.owning_agent_id == 'quantifier_agent'
        assert captured_event.service_call_type == 'LLM_ASK'
        
    async def test_signal_emission_pattern(self, mock_context):
        """Test that signals are emitted correctly."""
        from components.mechanisms.analyst_agents.quantifier_agent.service import QuantifierAgent
        
        # Create agent with mock dependencies
        config = {'llm_approach': 'single_call'}
        agent = QuantifierAgent(config)
        
        # Mock dependencies
        agent.proto_generator = AsyncMock()
        agent.proto_generator.process = AsyncMock(return_value=Mock(success=True, output_data={}))
        agent.idea_service = Mock()
        agent.gateway = AsyncMock()
        agent.event_bus = Mock()
        
        # Mock analysis engine to return viable result
        agent.analysis_engine.analyze_idea = AsyncMock(return_value=Mock(
            viable=True,
            approach="visualization",
            implementation_request="Create chart",
            libraries=["matplotlib"],
            use_mermaid=False,
            confidence=0.9
        ))
        
        # Process a high-trust idea
        data = {
            'idea_id': 'test_idea_001',
            'idea_text': 'Analyze market trends',
            'assessment_details': {
                'trust_score': 8.5,
                'is_stable': True
            }
        }
        
        result = await agent._process_impl(data, mock_context)
        
        # Verify success and proto generation triggered
        assert result.success
        assert agent.proto_generator.process.called
        
    async def test_resource_constraints(self, mock_context):
        """Test that resource constraints are respected."""
        from components.mechanisms.analyst_agents.quantifier_agent.service import QuantifierAgent
        
        # Create agent with conservative config
        config = {
            'llm_timeout_seconds': 5,
            'max_visualizations': 1,
            'min_request_length': 200
        }
        agent = QuantifierAgent(config)
        
        # Verify configuration is applied
        assert agent.cfg.llm_timeout_seconds == 5
        assert agent.cfg.max_visualizations == 1
        assert agent.cfg.min_request_length == 200
        
    def test_metadata_compliance(self):
        """Test that component metadata follows NIREON standards."""
        from components.mechanisms.analyst_agents.quantifier_agent.metadata import QUANTIFIER_METADATA
        
        # Verify all required fields
        assert QUANTIFIER_METADATA.id == 'quantifier_agent_primary'
        assert QUANTIFIER_METADATA.category == 'mechanism'
        assert QUANTIFIER_METADATA.requires_initialize == True
        
        # Verify dependencies are declared
        assert 'ProtoGenerator' in QUANTIFIER_METADATA.dependencies
        assert 'MechanismGatewayPort' in QUANTIFIER_METADATA.dependencies
        
        # Verify signal declarations
        assert 'TrustAssessmentSignal' in QUANTIFIER_METADATA.accepts
        assert 'ProtoTaskSignal' in QUANTIFIER_METADATA.produces
        assert 'GenerativeLoopFinishedSignal' in QUANTIFIER_METADATA.produces
        
    async def test_error_handling_cascade(self, mock_context):
        """Test graceful error handling and fallback behavior."""
        from components.mechanisms.analyst_agents.quantifier_agent.service import QuantifierAgent
        
        config = {'llm_approach': 'single_call'}
        agent = QuantifierAgent(config)
        
        # Setup dependencies
        agent.proto_generator = AsyncMock()
        agent.idea_service = Mock()
        agent.gateway = AsyncMock()
        agent.event_bus = Mock()
        
        # Make analysis fail
        agent.analysis_engine.analyze_idea = AsyncMock(return_value=None)
        
        # Process should handle gracefully
        data = {
            'idea_id': 'test_idea_001',
            'idea_text': 'Test idea',
            'assessment_details': {}
        }
        
        result = await agent._process_impl(data, mock_context)
        
        # Should emit completion signal instead of failing
        assert agent.event_bus.publish.called
        signal_type, signal = agent.event_bus.publish.call_args[0]
        assert signal_type == 'GenerativeLoopFinishedSignal'