import asyncio
import logging
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

# --- Path Setup ---
# This ensures the script can find the NIREON modules when run from the project root.
def setup_path():
    """Adds the project root to the Python path for module imports."""
    current_file_path = Path(__file__).resolve()
    # Assuming the script is in nireon_v4/tests/reactor/
    project_root = current_file_path.parent.parent.parent
    if str(project_root) not in sys.path:
        print(f'Adding project root to path: {project_root}')
        sys.path.insert(0, str(project_root))

setup_path()

# --- Imports from NIREON codebase ---
try:
    from core.registry import ComponentRegistry
    from core.lifecycle import ComponentMetadata
    from reactor.loader import RuleLoader
    from reactor.engine.main import MainReactorEngine
    from reactor.rules.core_rules import ConditionalRule
    # Updated import from our previous fix
    from signals import IdeaGeneratedSignal, EpistemicSignal
except ImportError as e:
    print(f"FATAL: A required NIREON module could not be imported: {e}")
    print("Please ensure the script is run from the project root directory (`nireon_v4`) and all dependencies are installed.")
    sys.exit(1)

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)-25s - %(levelname)-8s - %(message)s')
logger = logging.getLogger("StandaloneReactorTest")


# --- Test Functions (converted from pytest) ---

async def run_yaml_loading_test() -> bool:
    """Test that rules load correctly from YAML files."""
    test_name = "YAML Rule Loading"
    logger.info(f"--- Running Test: {test_name} ---")
    try:
        rules_dir = Path("configs/reactor/rules")
        if not rules_dir.exists():
            logger.error(f"Test failed: Rules directory not found at '{rules_dir.resolve()}'")
            return False
            
        loader = RuleLoader()
        rules = loader.load_rules_from_directory(rules_dir)
        
        # Converted assertions to explicit checks
        if not len(rules) > 0:
            logger.error("Test failed: Should load at least one rule, but loaded 0.")
            return False
        
        rule_ids = [r.rule_id for r in rules]
        if "core_seed_to_explorer_rule" not in rule_ids:
            logger.error("Test failed: 'core_seed_to_explorer_rule' not found in loaded rules.")
            return False
        
        priorities = [r.priority for r in rules]
        if priorities != sorted(priorities):
            logger.error(f"Test failed: Rules are not sorted by priority. Got: {priorities}")
            return False
        
        logger.info(f"✅ PASSED: {test_name} (Loaded {len(rules)} rules successfully)")
        return True
    except Exception as e:
        logger.error(f"❌ FAILED: {test_name} with an unexpected exception: {e}", exc_info=True)
        return False


async def run_rel_expression_test() -> bool:
    """Test REL expression evaluation in rules."""
    test_name = "REL Expression Evaluation"
    logger.info(f"--- Running Test: {test_name} ---")
    try:
        # Setup mock components
        registry = ComponentRegistry()
        catalyst = MagicMock()
        catalyst.process = AsyncMock()
        metadata = ComponentMetadata(id="catalyst_mechanism", name="Catalyst", version="1.0.0", category="mechanism")
        registry.register(catalyst, metadata)
        
        # Setup engine and rule
        engine = MainReactorEngine(registry=registry)
        rel_rule = ConditionalRule(
            rule_id="test_rel_rule",
            signal_type="IdeaGeneratedSignal",
            conditions=[
                {"type": "signal_type_match", "signal_type": "IdeaGeneratedSignal"},
                {"type": "payload_expression", "expression": "signal.trust_score > 0.8"}
            ],
            actions_on_match=[{"type": "trigger_component", "component_id": "catalyst_mechanism"}]
        )
        engine.add_rule(rel_rule)
        
        # Test 1: High trust signal
        logger.info("  - Testing high-trust signal (should trigger action)...")
        high_trust_signal = IdeaGeneratedSignal(source_node_id="test", idea_id="idea_1", idea_content="High trust idea", trust_score=0.9)
        await engine.process_signal(high_trust_signal)
        if not catalyst.process.call_count == 1:
            logger.error("Test failed: High-trust signal did not trigger the component.")
            return False
        logger.info("  - OK: High-trust signal correctly triggered the component.")

        # Test 2: Low trust signal
        logger.info("  - Testing low-trust signal (should NOT trigger action)...")
        catalyst.process.reset_mock()
        low_trust_signal = IdeaGeneratedSignal(source_node_id="test", idea_id="idea_2", idea_content="Low trust idea", trust_score=0.3)
        await engine.process_signal(low_trust_signal)
        if catalyst.process.called:
            logger.error("Test failed: Low-trust signal incorrectly triggered the component.")
            return False
        logger.info("  - OK: Low-trust signal was correctly ignored.")

        logger.info(f"✅ PASSED: {test_name}")
        return True
    except Exception as e:
        logger.error(f"❌ FAILED: {test_name} with an unexpected exception: {e}", exc_info=True)
        return False


async def run_action_template_test() -> bool:
    """Test that rule actions can use templates."""
    test_name = "Rule Action Templates"
    logger.info(f"--- Running Test: {test_name} ---")
    try:
        registry = ComponentRegistry()
        mechanism = MagicMock()
        mechanism.process = AsyncMock()
        metadata = ComponentMetadata(id="template_aware_mechanism", name="Template Mechanism", version="1.0.0", category="mechanism")
        registry.register(mechanism, metadata)
        
        template_rule = ConditionalRule(
            rule_id="template_test",
            signal_type="TestSignal",
            conditions=[],
            actions_on_match=[{
                "type": "trigger_component",
                "component_id": "template_aware_mechanism",
                "template_id": "SPECIAL_TEMPLATE",
                "input_data": {"mode": "enhanced"}
            }]
        )
        engine = MainReactorEngine(registry=registry)
        engine.add_rule(template_rule)
        
        test_signal = EpistemicSignal(signal_type="TestSignal", source_node_id="test")
        await engine.process_signal(test_signal)
        
        if not mechanism.process.called:
            logger.error("Test failed: The target mechanism's process method was not called.")
            return False

        call_kwargs = mechanism.process.call_args.kwargs
        template_id_passed = call_kwargs.get("template_id")
        input_data_passed = call_kwargs.get("data", {})
        
        if template_id_passed != "SPECIAL_TEMPLATE":
            logger.error(f"Test failed: Expected template_id 'SPECIAL_TEMPLATE', got '{template_id_passed}'.")
            return False
        
        if input_data_passed.get("mode") != "enhanced":
            logger.error(f"Test failed: Expected input_data mode 'enhanced', got '{input_data_passed.get('mode')}'.")
            return False
        
        logger.info(f"✅ PASSED: {test_name}")
        return True
    except Exception as e:
        logger.error(f"❌ FAILED: {test_name} with an unexpected exception: {e}", exc_info=True)
        return False


async def main():
    """Main function to run all tests and report a summary."""
    logger.info("==========================================")
    logger.info("  NIREON REACTOR TEST SUITE (Standalone)  ")
    logger.info("==========================================")
    
    tests_to_run = [
        run_yaml_loading_test,
        run_rel_expression_test,
        run_action_template_test
    ]
    
    results = []
    for test_func in tests_to_run:
        result = await test_func()
        results.append(result)
        print("-" * 40)
        await asyncio.sleep(0.1)
        
    passed_count = sum(results)
    total_count = len(results)
    
    print("\n" + "="*40)
    print("           TEST SUMMARY")
    print("="*40)
    print(f"  Total tests: {total_count}")
    print(f"  Passed:      {passed_count}")
    print(f"  Failed:      {total_count - passed_count}")
    print("="*40)
    
    if passed_count == total_count:
        print("\n🎉 All reactor tests passed successfully!")
        return 0
    else:
        print("\n❌ Some reactor tests failed. Please review the logs above.")
        return 1

if __name__ == "__main__":
    # This block allows the script to be run directly with:
    # python tests/reactor/test_yaml_rule_loading.py
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nTest execution cancelled by user.")
        sys.exit(1)