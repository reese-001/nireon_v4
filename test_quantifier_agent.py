"""
Standalone test file for QuantifierAgent - can be run directly without pytest
"""

import asyncio
import logging
from unittest.mock import Mock, AsyncMock, MagicMock
from pathlib import Path
import sys

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Test color codes for better output visibility
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    YELLOW = '\033[93m'
    RESET = '\033[0m'

def print_test_header(test_name: str):
    """Print a formatted test header."""
    print(f"\n{Colors.BLUE}{'='*60}")
    print(f"TEST: {test_name}")
    print(f"{'='*60}{Colors.RESET}")

def print_success(message: str):
    """Print success message in green."""
    print(f"{Colors.GREEN}✓ {message}{Colors.RESET}")

def print_failure(message: str):
    """Print failure message in red."""
    print(f"{Colors.RED}✗ {message}{Colors.RESET}")

def print_info(message: str):
    """Print info message in yellow."""
    print(f"{Colors.YELLOW}ℹ {message}{Colors.RESET}")

async def test_quantifier_initialization():
    """Test QuantifierAgent initialization."""
    print_test_header("QuantifierAgent Initialization")
    
    try:
        from components.mechanisms.analyst_agents.quantifier_agent import QuantifierAgent, QuantifierConfig
        from domain.context import NireonExecutionContext
        
        # Create agent with test configuration
        config = {
            'llm_approach': 'single_call',
            'max_visualizations': 2,
            'enable_mermaid_output': True
        }
        agent = QuantifierAgent(config)
        
        print_info(f"Created QuantifierAgent with config: {config}")
        
        # Mock context
        context = Mock(spec=NireonExecutionContext)
        context.component_registry = Mock()
        context.logger = Mock()
        context.run_id = "test_run_001"
        
        # Mock all required dependencies
        mock_proto_generator = Mock()
        mock_idea_service = Mock()
        mock_gateway = Mock()
        mock_event_bus = Mock()
        mock_frame_factory = AsyncMock()
        
        # Setup registry to return mocked dependencies
        def mock_get(component_id):
            if component_id == 'proto_generator_main':
                return mock_proto_generator
            return None
            
        def mock_get_service_instance(service_type):
            service_name = str(service_type)
            if 'IdeaService' in service_name:
                return mock_idea_service
            elif 'MechanismGatewayPort' in service_name:
                return mock_gateway
            elif 'EventBusPort' in service_name:
                return mock_event_bus
            elif 'FrameFactoryService' in service_name:
                return mock_frame_factory
            return None
        
        context.component_registry.get = mock_get
        context.component_registry.get_service_instance = mock_get_service_instance
        
        # Initialize the agent
        await agent._initialize_impl(context)
        
        # Verify configuration
        assert agent.cfg.llm_approach == 'single_call', "LLM approach not set correctly"
        assert agent.cfg.max_visualizations == 2, "Max visualizations not set correctly"
        assert agent.cfg.enable_mermaid_output == True, "Mermaid output not enabled"
        
        # Verify dependencies were resolved
        assert agent.proto_generator is not None, "ProtoGenerator not resolved"
        assert agent.idea_service is not None, "IdeaService not resolved"
        assert agent.gateway is not None, "MechanismGateway not resolved"
        assert agent.event_bus is not None, "EventBus not resolved"
        assert agent.frame_factory is not None, "FrameFactory not resolved"
        
        print_success("QuantifierAgent initialized successfully")
        print_success(f"Configuration loaded: approach={agent.cfg.llm_approach}, max_viz={agent.cfg.max_visualizations}")
        print_success("All dependencies resolved correctly")
        
        return True
        
    except Exception as e:
        print_failure(f"Initialization test failed: {e}")
        logger.exception("Detailed error:")
        return False

async def test_non_quantifiable_idea():
    """Test handling of non-quantifiable ideas."""
    print_test_header("Non-Quantifiable Idea Handling")
    
    try:
        from components.mechanisms.analyst_agents.quantifier_agent import QuantifierAgent
        from components.mechanisms.analyst_agents.quantifier_agent.analysis_engine import AnalysisResult
        from domain.context import NireonExecutionContext
        
        config = {'llm_approach': 'single_call'}
        agent = QuantifierAgent(config)
        
        # Mock the analysis engine to return non-viable result
        agent.analysis_engine.analyze_idea = AsyncMock(
            return_value=AnalysisResult(viable=False, confidence=0.8)
        )
        
        # Mock dependencies
        agent.event_bus = Mock()
        agent.event_bus.publish = Mock()
        agent.frame_factory = AsyncMock()
        
        # Create mock frame
        mock_frame = Mock()
        mock_frame.id = "test_frame_001"
        agent.frame_factory.create_frame = AsyncMock(return_value=mock_frame)
        
        # Mock context
        context = Mock(spec=NireonExecutionContext)
        context.logger = Mock()
        context.metadata = {}
        context.run_id = "test_run_002"
        
        # Test data
        data = {
            'idea_id': 'test_001',
            'idea_text': 'Abstract philosophical concept about the nature of consciousness',
            'assessment_details': {
                'trust_score': 8.0,
                'metadata': {'depth': 2}
            }
        }
        
        print_info(f"Testing with idea: '{data['idea_text'][:50]}...'")
        
        # Process the non-quantifiable idea
        result = await agent._process_impl(data, context)
        
        # Verify results
        assert result.success == True, "Process should succeed even for non-quantifiable ideas"
        assert 'not suitable' in result.message, "Message should indicate idea not suitable"
        
        # Verify completion signal was published
        assert agent.event_bus.publish.called, "Event bus should publish completion signal"
        call_args = agent.event_bus.publish.call_args
        signal_type = call_args[0][0]
        signal = call_args[0][1]
        
        assert signal_type == 'GenerativeLoopFinishedSignal', f"Wrong signal type: {signal_type}"
        
        print_success("Non-quantifiable idea handled gracefully")
        print_success("GenerativeLoopFinishedSignal published correctly")
        print_info(f"Signal payload: {signal.payload if hasattr(signal, 'payload') else 'N/A'}")
        
        return True
        
    except Exception as e:
        print_failure(f"Non-quantifiable idea test failed: {e}")
        logger.exception("Detailed error:")
        return False

async def test_quantifiable_idea():
    """Test successful quantification of an idea."""
    print_test_header("Quantifiable Idea Processing")
    
    try:
        from components.mechanisms.analyst_agents.quantifier_agent import QuantifierAgent
        from components.mechanisms.analyst_agents.quantifier_agent.analysis_engine import AnalysisResult
        from domain.context import NireonExecutionContext
        from core.results import ProcessResult
        
        config = {'llm_approach': 'single_call'}
        agent = QuantifierAgent(config)
        
        # Mock successful analysis result
        analysis_result = AnalysisResult(
            viable=True,
            approach="Bar chart visualization",
            implementation_request="Create a bar chart showing market share by company...",
            libraries=["matplotlib", "pandas"],
            confidence=0.9
        )
        agent.analysis_engine.analyze_idea = AsyncMock(return_value=analysis_result)
        
        # Mock dependencies
        agent.proto_generator = Mock()
        
        # Create a mock ProcessResult with all required parameters
        mock_process_result = Mock()
        mock_process_result.success = True
        mock_process_result.message = "Proto block generated"
        mock_process_result.output_data = {'proto_id': 'proto_123'}
        mock_process_result.component_id = 'proto_generator_main'
        
        agent.proto_generator.process = AsyncMock(return_value=mock_process_result)
        
        agent.frame_factory = AsyncMock()
        mock_frame = Mock()
        mock_frame.id = "test_frame_002"
        agent.frame_factory.create_frame = AsyncMock(return_value=mock_frame)
        
        # Mock context
        context = Mock(spec=NireonExecutionContext)
        context.logger = Mock()
        context.metadata = {}
        context.run_id = "test_run_003"
        
        # Test data
        data = {
            'idea_id': 'test_002',
            'idea_text': 'Analyze market share distribution among top 5 tech companies',
            'assessment_details': {
                'trust_score': 8.5,
                'metadata': {'depth': 1}
            }
        }
        
        print_info(f"Testing with quantifiable idea: '{data['idea_text']}'")
        
        # Process the quantifiable idea
        result = await agent._process_impl(data, context)
        
        # Verify results
        assert result.success == True, "Process should succeed"
        assert 'Successfully triggered' in result.message, "Message should indicate success"
        assert result.output_data is not None, "Should have output data"
        
        # Verify ProtoGenerator was called
        assert agent.proto_generator.process.called, "ProtoGenerator should be called"
        proto_call_args = agent.proto_generator.process.call_args[0][0]
        assert 'natural_language_request' in proto_call_args, "Should pass natural language request"
        
        print_success("Quantifiable idea processed successfully")
        print_success(f"Analysis approach: {analysis_result.approach}")
        print_success(f"Libraries identified: {', '.join(analysis_result.libraries)}")
        print_success("ProtoGenerator triggered with correct request")
        
        return True
        
    except Exception as e:
        print_failure(f"Quantifiable idea test failed: {e}")
        logger.exception("Detailed error:")
        return False

async def test_input_validation():
    """Test input data validation."""
    print_test_header("Input Data Validation")
    
    try:
        from components.mechanisms.analyst_agents.quantifier_agent import QuantifierAgent
        from domain.context import NireonExecutionContext
        
        config = {'llm_approach': 'single_call'}
        agent = QuantifierAgent(config)
        
        context = Mock(spec=NireonExecutionContext)
        context.logger = Mock()
        
        # Test cases with invalid input
        test_cases = [
            ({}, "Empty data"),
            ({'idea_id': 'test_003'}, "Missing idea_text"),
            ({'idea_text': 'Some text'}, "Missing idea_id"),
            ({'idea_id': 'test_003', 'idea_text': None}, "None idea_text"),
            ({'idea_id': 'test_003', 'idea_text': 123}, "Non-string idea_text"),
        ]
        
        all_passed = True
        
        for test_data, description in test_cases:
            print_info(f"Testing: {description}")
            
            result = await agent._process_impl(test_data, context)
            
            if result.success == False and "Missing required input" in result.message:
                print_success(f"  ✓ Correctly rejected: {description}")
            else:
                print_failure(f"  ✗ Failed to reject: {description}")
                all_passed = False
        
        return all_passed
        
    except Exception as e:
        print_failure(f"Input validation test failed: {e}")
        logger.exception("Detailed error:")
        return False

async def test_configuration_options():
    """Test different configuration options."""
    print_test_header("Configuration Options")
    
    try:
        from components.mechanisms.analyst_agents.quantifier_agent import QuantifierAgent, QuantifierConfig
        
        # Test different configurations
        configs = [
            {
                'name': 'Single Call Mode',
                'config': {'llm_approach': 'single_call', 'enable_mermaid_output': True}
            },
            {
                'name': 'Iterative Mode',
                'config': {'llm_approach': 'iterative', 'enable_mermaid_output': False}
            },
            {
                'name': 'Custom Libraries',
                'config': {
                    'llm_approach': 'single_call',
                    'available_libraries': {
                        'core_data': ['numpy', 'pandas'],
                        'visualization': ['matplotlib']
                    }
                }
            }
        ]
        
        all_passed = True
        
        for test_config in configs:
            print_info(f"Testing configuration: {test_config['name']}")
            
            try:
                agent = QuantifierAgent(test_config['config'])
                
                # Verify configuration was applied
                for key, value in test_config['config'].items():
                    if hasattr(agent.cfg, key):
                        actual_value = getattr(agent.cfg, key)
                        if actual_value == value:
                            print_success(f"  ✓ {key} = {value}")
                        else:
                            print_failure(f"  ✗ {key} expected {value}, got {actual_value}")
                            all_passed = False
                
            except Exception as e:
                print_failure(f"  ✗ Failed to create agent: {e}")
                all_passed = False
        
        return all_passed
        
    except Exception as e:
        print_failure(f"Configuration test failed: {e}")
        logger.exception("Detailed error:")
        return False

async def run_all_tests():
    """Run all tests and report results."""
    print(f"\n{Colors.BLUE}{'='*60}")
    print("QUANTIFIER AGENT TEST SUITE")
    print(f"{'='*60}{Colors.RESET}\n")
    
    tests = [
        test_quantifier_initialization,
        test_non_quantifiable_idea,
        test_quantifiable_idea,
        test_input_validation,
        test_configuration_options
    ]
    
    results = []
    
    for test in tests:
        try:
            result = await test()
            results.append((test.__name__, result))
        except Exception as e:
            print_failure(f"Test {test.__name__} crashed: {e}")
            results.append((test.__name__, False))
    
    # Print summary
    print(f"\n{Colors.BLUE}{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}{Colors.RESET}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = f"{Colors.GREEN}PASSED{Colors.RESET}" if result else f"{Colors.RED}FAILED{Colors.RESET}"
        print(f"{test_name:<40} {status}")
    
    print(f"\n{Colors.BLUE}Total: {passed}/{total} tests passed{Colors.RESET}")
    
    if passed == total:
        print(f"\n{Colors.GREEN}🎉 All tests passed! The QuantifierAgent is ready for deployment.{Colors.RESET}")
    else:
        print(f"\n{Colors.RED}⚠️  Some tests failed. Please review the errors above.{Colors.RESET}")
    
    return passed == total

def main():
    """Main entry point for the test script."""
    print(f"{Colors.YELLOW}Starting QuantifierAgent test suite...{Colors.RESET}")
    
    # Create event loop and run tests
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        success = loop.run_until_complete(run_all_tests())
        exit_code = 0 if success else 1
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Tests interrupted by user{Colors.RESET}")
        exit_code = 1
    except Exception as e:
        print(f"\n{Colors.RED}Unexpected error: {e}{Colors.RESET}")
        logger.exception("Test suite crashed:")
        exit_code = 1
    finally:
        loop.close()
    
    sys.exit(exit_code)

if __name__ == "__main__":
    main()