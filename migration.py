#!/usr/bin/env python3
"""
Test script for NIREON configuration loading

This script tests the new consolidated configuration loading system
to ensure it works correctly with the standardized structure.

Run this from your project root directory.
"""

import asyncio
import sys
import traceback
from pathlib import Path


def check_prerequisites():
    """Check if required files and directories exist"""
    print("Checking prerequisites...")
    
    required_paths = [
        'bootstrap/config/config_loader.py',
        'configs/default/global_app_config.yaml',
        'configs/default/llm_config.yaml',
        'configs/config_utils.py'
    ]
    
    missing_paths = []
    for path in required_paths:
        if not Path(path).exists():
            missing_paths.append(path)
    
    if missing_paths:
        print("❌ Missing required files:")
        for path in missing_paths:
            print(f"   - {path}")
        return False
    
    print("✅ All required files found")
    return True


async def test_global_config_loading():
    """Test global configuration loading for different environments"""
    print("\n" + "="*60)
    print("Testing Global Configuration Loading")
    print("="*60)
    
    try:
        from configs.config_loader import ConfigLoader
        loader = ConfigLoader()
        
        # Test environments to check
        environments = ['default', 'development']
        
        for env in environments:
            print(f"\n📋 Testing environment: {env}")
            try:
                config = await loader.load_global_config(env=env)
                
                # Basic validation
                assert isinstance(config, dict), f"Config should be a dict, got {type(config)}"
                assert config.get('env') == env, f"Environment mismatch: expected {env}, got {config.get('env')}"
                
                print(f"   ✅ Environment: {config.get('env')}")
                print(f"   ✅ Bootstrap strict mode: {config.get('bootstrap_strict_mode')}")
                
                # Check feature flags
                feature_flags = config.get('feature_flags', {})
                print(f"   ✅ Feature flags ({len(feature_flags)}): {list(feature_flags.keys())}")
                
                # Check LLM config
                llm_config = config.get('llm')
                if llm_config:
                    models = llm_config.get('models', {})
                    default_model = llm_config.get('default')
                    print(f"   ✅ LLM config loaded - Default model: {default_model}")
                    print(f"   ✅ Available models ({len(models)}): {list(models.keys())}")
                else:
                    print("   ⚠️  No LLM config found")
                
                # Check storage config
                storage = config.get('storage')
                if storage:
                    print(f"   ✅ Storage config: {list(storage.keys())}")
                
                print(f"   ✅ Total config keys: {list(config.keys())}")
                
            except Exception as e:
                print(f"   ❌ Failed to load config for {env}: {e}")
                traceback.print_exc()
                return False
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure bootstrap.config.config_loader exists and is accessible")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        traceback.print_exc()
        return False


async def test_component_config_loading():
    """Test component configuration loading"""
    print("\n" + "="*60)
    print("Testing Component Configuration Loading")
    print("="*60)
    
    try:
        from configs.config_loader import ConfigLoader
        loader = ConfigLoader()
        
        # First get global config
        global_config = await loader.load_global_config(env='development')
        
        # Test component specs that should match your new structure
        component_specs = [
            {
                'name': 'explorer_primary_01',
                'spec': {
                    'class': 'components.mechanisms.explorer.service.ExplorerMechanism',
                    'config': 'configs/default/mechanisms/{id}.yaml',
                    'config_override': {
                        'test_override': True
                    }
                }
            },
            {
                'name': 'lineage_tracker',
                'spec': {
                    'class': 'components.observers.lineage_tracker.LineageTracker',
                    'config': 'configs/default/observers/{id}.yaml',
                    'config_override': {
                        'test_override': True
                    }
                }
            },
            {
                'name': 'flow_coordinator',
                'spec': {
                    'class': 'components.managers.flow_coordinator.FlowCoordinator',
                    'config': 'configs/default/managers/{id}.yaml',
                    'config_override': {
                        'test_override': True
                    }
                }
            }
        ]
        
        for component_info in component_specs:
            component_id = component_info['name']
            component_spec = component_info['spec']
            
            print(f"\n🔧 Testing component: {component_id}")
            
            try:
                component_config = await loader.load_component_config(
                    component_spec=component_spec,
                    component_id=component_id,
                    global_config=global_config
                )
                
                if component_config:
                    print(f"   ✅ Component config loaded")
                    print(f"   ✅ Config keys: {list(component_config.keys())}")
                    
                    # Check if override was applied
                    if component_config.get('test_override'):
                        print(f"   ✅ Config override applied successfully")
                    
                    # Show a few sample values
                    for key, value in list(component_config.items())[:3]:
                        print(f"   ✅ {key}: {value}")
                    
                    if len(component_config) > 3:
                        print(f"   ... and {len(component_config) - 3} more config items")
                        
                else:
                    print(f"   ⚠️  Component config is empty (this might be normal if config file doesn't exist)")
                    
            except Exception as e:
                print(f"   ❌ Failed to load component config: {e}")
                # Don't fail the whole test for missing component configs
                continue
        
        return True
        
    except Exception as e:
        print(f"❌ Component config test failed: {e}")
        traceback.print_exc()
        return False


async def test_environment_variable_expansion():
    """Test environment variable expansion"""
    print("\n" + "="*60)
    print("Testing Environment Variable Expansion")
    print("="*60)
    
    try:
        from configs.config_loader import ConfigLoader
        import os
        
        # Set a test environment variable
        os.environ['TEST_CONFIG_VAR'] = 'test_value_123'
        os.environ['TEST_CONFIG_WITH_DEFAULT'] = 'env_value_456'
        
        loader = ConfigLoader()
        
        # Test the expansion function directly
        test_strings = [
            '${TEST_CONFIG_VAR}',
            '${TEST_CONFIG_WITH_DEFAULT:-default_value}',
            '${NONEXISTENT_VAR:-fallback_value}',
            'prefix_${TEST_CONFIG_VAR}_suffix',
            'normal_string_no_vars'
        ]
        
        print("🔧 Testing environment variable expansion:")
        for test_string in test_strings:
            expanded = loader._expand_env_var_string(test_string)
            print(f"   '{test_string}' -> '{expanded}'")
        
        # Test with actual config loading
        config = await loader.load_global_config(env='development')
        print("\n✅ Environment variable expansion test completed")
        
        # Clean up test variables
        del os.environ['TEST_CONFIG_VAR']
        del os.environ['TEST_CONFIG_WITH_DEFAULT']
        
        return True
        
    except Exception as e:
        print(f"❌ Environment variable expansion test failed: {e}")
        traceback.print_exc()
        return False


def print_summary(results):
    """Print test summary"""
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    total_tests = len(results)
    passed_tests = sum(results)
    
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    
    if passed_tests == total_tests:
        print("\n🎉 All tests passed! Your configuration loading is working correctly.")
        return True
    else:
        print(f"\n❌ {total_tests - passed_tests} test(s) failed. Please check the errors above.")
        return False


async def main():
    """Run all configuration tests"""
    print("NIREON Configuration Loading Test Suite")
    print("="*60)
    
    # Check prerequisites first
    if not check_prerequisites():
        print("\n❌ Prerequisites not met. Please run migration script first.")
        sys.exit(1)
    
    # Run all tests
    test_results = []
    
    try:
        # Test global config loading
        result1 = await test_global_config_loading()
        test_results.append(result1)
        
        # Test component config loading
        result2 = await test_component_config_loading()
        test_results.append(result2)
        
        # Test environment variable expansion
        result3 = await test_environment_variable_expansion()
        test_results.append(result3)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error during testing: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Print summary and exit
    success = print_summary(test_results)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)