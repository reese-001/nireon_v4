#!/usr/bin/env python
"""Test script to verify bootstrap imports are working correctly."""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test all critical imports."""
    print("Testing bootstrap imports...")
    
    try:
        # Test main bootstrap import
        import bootstrap
        print("✓ bootstrap package imported")
        
        # Test specific imports
        from bootstrap import (
            bootstrap_nireon_system,
            BootstrapConfig,
            BootstrapOrchestrator,
            BootstrapError,
            BootstrapContextBuildError
        )
        print("✓ Main exports imported successfully")
        
        # Test exception imports
        from bootstrap.exceptions import (
            BootstrapError,
            BootstrapValidationError,
            BootstrapContextBuildError
        )
        print("✓ Exception imports working")
        
        # Test internal imports
        from context.bootstrap_context_builder import BootstrapContextBuilder
        print("✓ Internal module imports working")
        
        # Test circular import resolution
        from core.main import BootstrapOrchestrator as MainOrchestrator
        print("✓ No circular import detected")
        
        print("\n✅ All imports successful!")
        return True
        
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)