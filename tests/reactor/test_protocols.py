import pytest
import logging
import asyncio
import sys
from pathlib import Path

def setup_path():
    current_file_path = Path(__file__).resolve()
    project_root = current_file_path.parent.parent.parent.parent
    if str(project_root) not in sys.path:
        print(f'Adding project root to path: {project_root}')
        sys.path.insert(0, str(project_root))

setup_path()

from reactor import protocols
from reactor import models

def test_protocols_can_be_imported():
    logging.info('--- Running Protocol Import Test ---')
    
    # Test protocols module
    assert hasattr(protocols, 'ReactorRule')
    logging.info('✓ ReactorRule protocol found.')
    
    assert hasattr(protocols, 'Condition')
    logging.info('✓ Condition protocol found.')
    
    # Test that Action is properly defined in models module
    assert hasattr(models, 'Action')
    logging.info('✓ Action type found in models module (where it belongs).')
    
    # Test that we can import specific action types
    assert hasattr(models, 'TriggerComponentAction')
    logging.info('✓ TriggerComponentAction found in models.')
    
    assert hasattr(models, 'EmitSignalAction')
    logging.info('✓ EmitSignalAction found in models.')
    
    logging.info('✅ All protocols and types imported successfully.')
    return True

async def main():
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logging.info('========================================')
    logging.info(' RUNNING REACTOR PROTOCOLS TEST SCRIPT ')
    logging.info('========================================')
    
    try:
        test_passed = test_protocols_can_be_imported()
        if test_passed:
            logging.info('🎉 All tests PASSED.')
            return 0
        else:
            logging.error('❌ A test FAILED.')
            return 1
    except Exception as e:
        logging.error('A test failed with an unhandled exception!', exc_info=True)
        return 1

if __name__ == '__main__':
    exit_code = asyncio.run(main())
    sys.exit(exit_code)