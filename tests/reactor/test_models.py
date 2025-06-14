# FILE: tests/unit/reactor/test_models.py

import pytest
import logging
import asyncio
import sys
from pathlib import Path

# --- BOILERPLATE TO MAKE THE SCRIPT RUNNABLE ---
def setup_path():
    current_file_path = Path(__file__).resolve()
    # Path is .../nireon_v4/tests/unit/reactor/test_models.py
    # Go up four parent directories to get to the TLD
    project_root = current_file_path.parent.parent.parent.parent
    if str(project_root) not in sys.path:
        print(f"Adding project root to path: {project_root}")
        sys.path.insert(0, str(project_root))

setup_path()
# --------------------------------------------------

from reactor import models

# Original test function for pytest
def test_models_can_be_instantiated():
    """Test that our core pydantic models can be created."""
    logging.info("--- Running Model Instantiation Test ---")

    # This acts as a basic check that the model definitions are valid.
    # We use dummy data here.
    context = models.RuleContext(
        signal="dummy_signal",
        run_id="test_run",
        component_registry={},
        logger=None,
        recursion_depth=1
    )
    assert context.run_id == "test_run"
    logging.info("✓ RuleContext model OK.")

    action = models.TriggerComponentAction(component_id="explorer_1")
    assert action.component_id == "explorer_1"
    logging.info("✓ TriggerComponentAction model OK.")

    emit_action = models.EmitSignalAction(signal_type="TEST_SIGNAL", payload={"data": 123})
    assert emit_action.signal_type == "TEST_SIGNAL"
    logging.info("✓ EmitSignalAction model OK.")

    logging.info("✅ All models instantiated successfully.")
    return True

# --- The main() function to make the script directly runnable ---
async def main():
    """Main function to run tests when the script is executed directly."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.info("========================================")
    logging.info("  RUNNING REACTOR MODELS TEST SCRIPT  ")
    logging.info("========================================")

    try:
        test_passed = test_models_can_be_instantiated()
        if test_passed:
            logging.info("🎉 All tests PASSED.")
            return 0  # Exit code for success
        else:
            logging.error("❌ A test FAILED.")
            return 1  # Exit code for failure
    except Exception as e:
        logging.error("A test failed with an unhandled exception!", exc_info=True)
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)