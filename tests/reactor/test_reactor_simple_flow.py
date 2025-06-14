import pytest
import logging
import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

def setup_path():
    current_file_path = Path(__file__).resolve()
    project_root = current_file_path.parent.parent.parent
    if str(project_root) not in sys.path:
        print(f'Adding project root to path: {project_root}')
        sys.path.insert(0, str(project_root))

setup_path()

from core.registry import ComponentRegistry
from signals import SeedSignal, LoopSignal
from reactor.engine.main import MainReactorEngine
from reactor.rules.core_rules import SignalTypeMatchRule
from reactor.models import RuleContext
from core.lifecycle import ComponentMetadata

@pytest.mark.asyncio
async def test_reactor_happy_path_flow():
    logging.info('--- Running Happy Path Test ---')
    
    # 1. Setup
    registry = ComponentRegistry()
    
    # Create a mock explorer component
    mock_explorer = MagicMock()
    mock_explorer.process = AsyncMock()
    
    # Register the mock component
    explorer_metadata = ComponentMetadata(
        id='explorer_primary',
        name='MockExplorer',
        category='mechanism',
        version='1.0.0'
    )
    registry.register(mock_explorer, explorer_metadata)
    
    # Create reactor engine
    engine = MainReactorEngine(registry=registry)
    
    # --- CHANGE: Use the new, more flexible rule constructor ---
    rule = SignalTypeMatchRule(
        rule_id="test_rule",
        signal_type_to_match="SeedSignal",
        component_id_to_trigger="explorer_primary",
        # We can test the new parameters here
        input_data={"source": "from_test_rule"}
    )
    engine.add_rule(rule)
    
    # 2. Execute
    test_signal = SeedSignal(source_node_id='test_source')
    await engine.process_signal(test_signal)
    
    # 3. Assert
    assert mock_explorer.process.call_count == 1
    mock_explorer.process.assert_awaited_once()
    
    # Check that the data from the rule was passed to the component
    called_with_data = mock_explorer.process.call_args.kwargs.get('data')
    assert called_with_data == {"source": "from_test_rule"}
    
    logging.info("✅ Happy path test PASSED: Explorer's process method was called with correct data.")
    return True

# The loop safety test remains the same, as it doesn't need the new params.
@pytest.mark.asyncio
async def test_reactor_loop_safety_prevents_infinite_recursion(caplog):
    logging.info('--- Running Loop Safety Test ---')
    
    # 1. Setup
    registry = ComponentRegistry()
    engine = MainReactorEngine(registry=registry, max_recursion_depth=5)
    
    # Create a mock component that triggers infinite recursion
    mock_looper_component = MagicMock()
    
    async def loop_effect(*args, **kwargs):
        # This component will trigger another LoopSignal when called
        engine_instance = mock_looper_component.engine
        current_context: RuleContext = kwargs.get('context')
        if not current_context:
            raise ValueError('Test setup error: component did not receive context.')
        
        # Trigger another signal, continuing the loop
        await engine_instance.process_signal(
            LoopSignal(source_node_id='looper'),
            current_depth=current_context.recursion_depth + 1
        )
    
    mock_looper_component.process = AsyncMock(side_effect=loop_effect)
    mock_looper_component.engine = engine
    
    # Register the looper component
    looper_metadata = ComponentMetadata(
        id='looper_comp',
        name='MockLooper',
        category='test_component',
        version='1.0.0'
    )
    registry.register(mock_looper_component, looper_metadata)
    
    # Create a rule that triggers the looper on LoopSignal
    rule = SignalTypeMatchRule(
        rule_id='looping_rule',
        signal_type_to_match='LoopSignal',
        component_id_to_trigger='looper_comp'
    )
    engine.add_rule(rule)
    
    # 2. Execute with log capture
    with caplog.at_level(logging.ERROR):
        await engine.process_signal(LoopSignal(source_node_id='initial_trigger'))
    
    # 3. Assert
    # Component should be called exactly 5 times (max_recursion_depth)
    assert mock_looper_component.process.call_count == 5
    
    # Check that the recursion limit error was logged
    assert 'Recursion depth limit (5) exceeded' in caplog.text
    
    logging.info('✅ Loop safety test PASSED: Recursion was correctly terminated.')
    return True

async def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logging.info('========================================')
    logging.info('  RUNNING REACTOR INTEGRATION SCRIPT  ')
    logging.info('========================================')
    
    # Create a dummy caplog for the standalone script
    class DummyCaplog:
        def __init__(self):
            self.text = ''
            self._original_handler = None
            self._target_logger = logging.getLogger('reactor.engine.main')
        
        def at_level(self, level):
            return self
        
        def __enter__(self):
            self.text = ''
            self._handler = logging.StreamHandler(self)
            self._handler.setLevel(logging.DEBUG)
            self._target_logger.addHandler(self._handler)
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            self._target_logger.removeHandler(self._handler)
        
        def write(self, message):
            self.text += message
        
        def flush(self):
            pass
    
    results = []
    caplog_instance = DummyCaplog()
    
    try:
        # Run happy path test
        result1 = await test_reactor_happy_path_flow()
        results.append(result1)
        
        # Run loop safety test
        result2 = await test_reactor_loop_safety_prevents_infinite_recursion(caplog_instance)
        results.append(result2)
        
    except Exception as e:
        logging.error('A test failed with an unhandled exception!', exc_info=True)
        results.append(False)
    
    logging.info('========================================')
    if all(results):
        logging.info('🎉 All tests PASSED.')
        return 0
    else:
        logging.error('❌ Some tests FAILED.')
        return 1

if __name__ == '__main__':
    exit_code = asyncio.run(main())
    sys.exit(exit_code)