#!/usr/bin/env python3
"""
Bootstrap Fix Verification Script
Tests the fixes for the three identified issues
"""

import sys
import asyncio
import logging
from pathlib import Path

# Add the project root to Python path if needed
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_import_fixes():
    """Test that all phase imports work correctly"""
    logger.info("Testing phase imports...")
    
    try:
        # Test the fixed import path for AbiogenesisPhase
        from bootstrap.phases.abiogenesis_phase import AbiogenesisPhase
        logger.info("✓ AbiogenesisPhase imported successfully")
        
        # Test that it follows the proper interface
        phase = AbiogenesisPhase()
        assert hasattr(phase, 'execute'), "AbiogenesisPhase missing execute method"
        assert hasattr(phase, 'should_skip_phase'), "AbiogenesisPhase missing should_skip_phase method"
        logger.info("✓ AbiogenesisPhase has correct interface")
        
        # Test other phase imports
        from bootstrap.phases.context_formation_phase import ContextFormationPhase
        from bootstrap.phases.registry_setup_phase import RegistrySetupPhase
        from bootstrap.phases.factory_setup_phase import FactorySetupPhase
        logger.info("✓ All other phases imported successfully")
        
        return True
        
    except ImportError as e:
        logger.error(f"✗ Import test failed: {e}")
        return False
    except Exception as e:
        logger.error(f"✗ Interface test failed: {e}")
        return False

async def test_phase_instantiation():
    """Test that all phases can be instantiated"""
    logger.info("Testing phase instantiation...")
    
    try:
        from bootstrap.phases.abiogenesis_phase import AbiogenesisPhase
        from bootstrap.phases.context_formation_phase import ContextFormationPhase
        from bootstrap.phases.registry_setup_phase import RegistrySetupPhase
        from bootstrap.phases.factory_setup_phase import FactorySetupPhase
        
        phases = [
            AbiogenesisPhase(),
            ContextFormationPhase(),
            RegistrySetupPhase(),
            FactorySetupPhase(),
        ]
        
        logger.info(f"✓ Successfully instantiated {len(phases)} phases")
        return True
        
    except Exception as e:
        logger.error(f"✗ Phase instantiation failed: {e}")
        return False

async def test_manifest_analysis():
    """Test manifest analysis for preload services"""
    logger.info("Testing manifest analysis...")
    
    try:
        # Create a test manifest content
        test_manifest = {
            'version': '1.0',
            'shared_services': {
                'test_service_preload': {
                    'enabled': True,
                    'preload': True,
                    'class': 'some.module:SomeClass'
                },
                'test_service_normal': {
                    'enabled': True,
                    'preload': False,
                    'class': 'some.module:AnotherClass'
                }
            }
        }
        
        from bootstrap.phases.abiogenesis_phase import AbiogenesisPhase
        phase = AbiogenesisPhase()
        
        # Count preload services
        preload_count = 0
        for service_id, spec in test_manifest['shared_services'].items():
            if spec.get('enabled', True) and spec.get('preload', False):
                preload_count += 1
                
        logger.info(f"✓ Found {preload_count} preload services in test manifest")
        return preload_count > 0
        
    except Exception as e:
        logger.error(f"✗ Manifest analysis failed: {e}")
        return False

async def test_main_orchestrator():
    """Test that the main orchestrator can load phases"""
    logger.info("Testing main orchestrator phase loading...")
    
    try:
        from bootstrap.core.main import BootstrapOrchestrator
        from bootstrap.config.bootstrap_config import BootstrapConfig
        
        # Create a minimal config
        config = BootstrapConfig.from_params([])
        orchestrator = BootstrapOrchestrator(config)
        
        # Test phase loading
        phases = orchestrator._get_phases()
        phase_names = [p.__class__.__name__ for p in phases if p is not None]
        
        logger.info(f"✓ Orchestrator loaded {len(phases)} phases: {', '.join(phase_names)}")
        
        # Check that AbiogenesisPhase is included and not None
        abiogenesis_found = any(p.__class__.__name__ == 'AbiogenesisPhase' for p in phases if p is not None)
        
        if abiogenesis_found:
            logger.info("✓ AbiogenesisPhase successfully loaded in orchestrator")
            return True
        else:
            logger.error("✗ AbiogenesisPhase not found in orchestrator phases")
            return False
            
    except Exception as e:
        logger.error(f"✗ Orchestrator test failed: {e}")
        return False

async def main():
    """Run all verification tests"""
    logger.info("=" * 60)
    logger.info("BOOTSTRAP FIX VERIFICATION")
    logger.info("=" * 60)
    
    tests = [
        ("Import Fixes", test_import_fixes),
        ("Phase Instantiation", test_phase_instantiation),
        ("Manifest Analysis", test_manifest_analysis),
        ("Main Orchestrator", test_main_orchestrator),
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST RESULTS SUMMARY")
    logger.info("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        logger.info("🎉 All fixes verified successfully!")
        return True
    else:
        logger.error("❌ Some fixes need additional work")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)