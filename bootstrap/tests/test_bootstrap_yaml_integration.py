#!/usr/bin/env python3
"""
Test script to verify YAML configuration integration with bootstrap system

This script tests the integration of the YAML configuration files with
the NIREON V4 bootstrap system to ensure proper loading and processing.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add the nireon package to the path if needed
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

async def test_config_loading():
    """Test basic configuration loading"""
    logger.info("=== Testing Configuration Loading ===")
    
    try:
        from configs.config_loader import ConfigLoader
        
        # Test default environment
        config = load_config(env='default')
        logger.info(f"✓ Default config loaded successfully")
        logger.info(f"  - Environment: {config.get('env')}")
        logger.info(f"  - Bootstrap strict mode: {config.get('bootstrap_strict_mode')}")
        logger.info(f"  - Feature flags: {list(config.get('feature_flags', {}).keys())}")
        logger.info(f"  - LLM default model: {config.get('llm', {}).get('default')}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Config loading failed: {e}")
        return False

async def test_bootstrap_config_loader():
    """Test V4 bootstrap config loader"""
    logger.info("=== Testing V4 Bootstrap Config Loader ===")
    
    try:
        from configs.config_loader import ConfigLoader
        
        loader = ConfigLoader()
        config = await loader.load_global_config(env='default')
        
        logger.info(f"✓ V4 config loader succeeded")
        logger.info(f"  - Config keys: {list(config.keys())}")
        logger.info(f"  - Bootstrap strict mode: {config.get('bootstrap_strict_mode')}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ V4 config loader failed: {e}")
        return False

async def test_manifest_processing():
    """Test manifest processing with standard.yaml"""
    logger.info("=== Testing Manifest Processing ===")
    
    try:
        from bootstrap.processors.manifest_processor import ManifestProcessor
        from runtime.utils import load_yaml_robust
        from pathlib import Path
        
        # Try to load the standard manifest
        manifest_path = Path("configs/manifests/standard.yaml")
        if not manifest_path.exists():
            logger.warning(f"Standard manifest not found at {manifest_path}")
            return False
        
        manifest_data = load_yaml_robust(manifest_path)
        if not manifest_data:
            logger.error("Failed to load manifest data")
            return False
        
        processor = ManifestProcessor(strict_mode=False)
        result = await processor.process_manifest(manifest_path, manifest_data)
        
        logger.info(f"✓ Manifest processing succeeded")
        logger.info(f"  - Success: {result.success}")
        logger.info(f"  - Components found: {result.component_count}")
        logger.info(f"  - Manifest type: {result.manifest_type}")
        
        if result.errors:
            logger.warning(f"  - Errors: {result.errors}")
        if result.warnings:
            logger.warning(f"  - Warnings: {result.warnings}")
        
        return result.success
        
    except Exception as e:
        logger.error(f"✗ Manifest processing failed: {e}")
        return False

async def test_explorer_mechanism():
    """Test explorer mechanism instantiation"""
    logger.info("=== Testing Explorer Mechanism ===")
    
    try:
        from mechanisms.explorer.service import ExplorerMechanism, EXPLORER_METADATA
        from mechanisms.explorer.config import ExplorerConfig
        
        # Test config model
        config_dict = {
            'max_depth': 4,
            'application_rate': 0.6,
            'exploration_strategy': 'depth_first'
        }
        
        explorer_config = ExplorerConfig(**config_dict)
        logger.info(f"✓ Explorer config validation succeeded")
        logger.info(f"  - Max depth: {explorer_config.max_depth}")
        logger.info(f"  - Strategy: {explorer_config.exploration_strategy}")
        
        # Test metadata
        logger.info(f"✓ Explorer metadata loaded")
        logger.info(f"  - Name: {EXPLORER_METADATA.name}")
        logger.info(f"  - Category: {EXPLORER_METADATA.category}")
        logger.info(f"  - Epistemic tags: {EXPLORER_METADATA.epistemic_tags}")
        
        # Test mechanism instantiation (basic)
        instance_metadata = EXPLORER_METADATA
        instance_metadata.id = "test_explorer"  # Set instance ID
        
        explorer = ExplorerMechanism(
            config=config_dict,
            metadata_definition=instance_metadata
        )
        
        logger.info(f"✓ Explorer mechanism instantiation succeeded")
        logger.info(f"  - Component ID: {explorer.component_id}")
        logger.info(f"  - Initialized: {explorer.is_initialized}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Explorer mechanism test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_minimal_bootstrap():
    """Test minimal bootstrap with YAML configs"""
    logger.info("=== Testing Minimal Bootstrap ===")
    
    try:
        from bootstrap import bootstrap_sync, create_test_bootstrap_config
        
        # Create minimal test manifest
        test_manifest = {
            'version': '1.0',
            'metadata': {
                'name': 'YAML Integration Test',
                'description': 'Test manifest for YAML integration'
            },
            'shared_services': {},
            'mechanisms': {},
            'observers': []
        }
        
        # Create test bootstrap config
        bootstrap_config = create_test_bootstrap_config(
            test_manifest_content=test_manifest,
            strict_mode=False,
            env='test'
        )
        
        logger.info(f"✓ Test bootstrap config created")
        logger.info(f"  - Strict mode: {bootstrap_config.effective_strict_mode}")
        logger.info(f"  - Environment: {bootstrap_config.global_app_config.get('env')}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Minimal bootstrap test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all integration tests"""
    logger.info("Starting NIREON V4 YAML Integration Tests")
    logger.info("=" * 60)
    
    tests = [
        ("Config Loading", test_config_loading),
        ("Bootstrap Config Loader", test_bootstrap_config_loader), 
        ("Manifest Processing", test_manifest_processing),
        ("Explorer Mechanism", test_explorer_mechanism),
        ("Minimal Bootstrap", test_minimal_bootstrap)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results.append((test_name, result))
            logger.info("")  # Add spacing between tests
        except Exception as e:
            logger.error(f"Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
            logger.info("")
    
    # Summary
    logger.info("=" * 60)
    logger.info("TEST RESULTS SUMMARY")
    logger.info("=" * 60)
    
    passed = 0
    failed = 0
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        logger.info(f"{test_name:.<50} {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    logger.info("=" * 60)
    logger.info(f"Total: {len(results)}, Passed: {passed}, Failed: {failed}")
    
    if failed == 0:
        logger.info("🎉 All YAML integration tests PASSED!")
        return 0
    else:
        logger.error(f"❌ {failed} tests FAILED!")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)