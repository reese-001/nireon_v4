#!/usr/bin/env python3
"""
LLM Subsystem Testing CLI

A comprehensive command-line tool for testing and validating the Nireon LLM subsystem.
Provides connectivity tests, configuration validation, health monitoring, and diagnostics.

Usage:
    # From the tests/llm_subsystem/ directory:
    python llm_test_cli.py --help
    python llm_test_cli.py test-connectivity --all
    python llm_test_cli.py validate-config
    python llm_test_cli.py health-check
    python llm_test_cli.py interactive
    
    # Or from project root:
    python tests/llm_subsystem/llm_test_cli.py test-connectivity --model nano_default
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime

# Add the project root to path for imports
# If script is in /tests/llm_subsystem/, go up 3 levels to reach project root
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent.parent

# Verify we found the right directory by checking for key files
key_files = ['infrastructure', 'domain', 'core']
if not all((project_root / folder).exists() for folder in key_files):
    # Try alternative paths
    alternative_roots = [
        script_path.parent.parent,  # /tests/
        script_path.parent,         # /tests/llm_subsystem/
        Path.cwd(),                 # Current working directory
    ]
    
    for alt_root in alternative_roots:
        if all((alt_root / folder).exists() for folder in key_files):
            project_root = alt_root
            break
    else:
        print("Error: Could not find project root directory.")
        print(f"Script location: {script_path}")
        print(f"Looking for directories: {key_files}")
        print("Please run the script from the project root directory or fix the path.")
        sys.exit(1)

print(f"🗂️  Project root: {project_root}")
sys.path.insert(0, str(project_root))

try:
    from infrastructure.llm.router import LLMRouter
    from infrastructure.llm.factory import create_llm_instance
    from domain.context import NireonExecutionContext
    from domain.epistemic_stage import EpistemicStage
    from core.lifecycle import ComponentMetadata
    
    print(f"✅ Successfully imported Nireon modules")
    
    # Try to import enhancements
    try:
        from infrastructure.llm.config_validator import validate_and_log_config
        from infrastructure.llm.metrics import get_metrics_collector
        from infrastructure.llm.exceptions import LLMError
        ENHANCEMENTS_AVAILABLE = True
        print(f"✅ LLM enhancements available")
    except ImportError:
        ENHANCEMENTS_AVAILABLE = False
        print(f"⚠️  LLM enhancements not available (using basic functionality)")
        def validate_and_log_config(config, logger):
            return True
        def get_metrics_collector():
            return None

except ImportError as e:
    print(f"❌ Error importing Nireon modules: {e}")
    print(f"📁 Script location: {Path(__file__).resolve()}")
    print(f"📁 Project root detected: {project_root}")
    print(f"📁 Python path: {sys.path[:3]}...")  # Show first 3 entries
    print("\nTroubleshooting:")
    print("1. Make sure you're running from the project root OR the tests/llm_subsystem/ directory")
    print("2. Verify that infrastructure/, domain/, and core/ directories exist in project root")
    print("3. Check that all dependencies are installed")
    print("4. Try running: python -c \"import infrastructure; print('Import successful')\"")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class TestResult:
    """Result of a test operation."""
    name: str
    success: bool
    duration_ms: float
    message: str
    details: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class MockComponentRegistry:
    """Mock component registry for testing purposes."""
    
    def get_service_instance(self, service_type):
        """Mock method - returns None since we don't have real services in tests."""
        return None
    
    def get(self, key, default=None):
        """Mock method for dictionary-like access."""
        return default
    
    def register_component(self, component):
        """Mock method for component registration."""
        pass
    
    def register(self, key, component):
        """Mock method for registering components by key."""
        pass
    
    def get_component(self, component_id):
        """Mock method for getting components by ID."""
        return None
    
    def list_components(self):
        """Mock method for listing components."""
        return []
    
    def __contains__(self, key):
        """Mock method for 'in' operator."""
        return False
    
    def __getitem__(self, key):
        """Mock method for dictionary-like access."""
        raise KeyError(f"Component '{key}' not found in test registry")
    
    def __setitem__(self, key, value):
        """Mock method for dictionary-like assignment."""
        pass
    
    def keys(self):
        """Mock method for getting keys."""
        return []

class LLMTester:
    """Main testing class for LLM subsystem validation."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._find_config_file()
        self.config: Dict[str, Any] = {}
        self.router: Optional[LLMRouter] = None
        self.test_results: List[TestResult] = []
        
    def _find_config_file(self) -> str:
        """Find the LLM configuration file."""
        # Since script is in /tests/llm_subsystem/, adjust paths accordingly
        script_dir = Path(__file__).parent
        project_root = script_dir.parent.parent
        
        possible_paths = [
            # Relative to project root
            project_root / "configs/default/llm_config.yaml",
            project_root / "configs/llm_config.yaml", 
            project_root / "llm_config.yaml",
            # Relative to script location
            script_dir / "llm_config.yaml",
            script_dir / "../../configs/default/llm_config.yaml",
            # Current working directory
            Path.cwd() / "configs/default/llm_config.yaml",
            Path.cwd() / "llm_config.yaml",
        ]
        
        for path in possible_paths:
            if path.exists():
                return str(path.resolve())
        
        raise FileNotFoundError(
            f"Could not find llm_config.yaml. Searched in:\n" + 
            "\n".join(f"  - {path}" for path in possible_paths) +
            "\n\nPlease specify path with --config or ensure the file exists in one of these locations."
        )
    
    def load_config(self) -> bool:
        """Load and validate the LLM configuration."""
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            
            logger.info(f"Loaded configuration from {self.config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            return False
    
    def validate_config(self) -> TestResult:
        """Validate the configuration structure and content."""
        start_time = time.time()
        
        try:
            if not self.config:
                raise ValueError("No configuration loaded")
            
            # Basic structure validation
            required_sections = ['models', 'default']
            missing_sections = [section for section in required_sections if section not in self.config]
            
            if missing_sections:
                raise ValueError(f"Missing required sections: {missing_sections}")
            
            # Validate models section
            models = self.config.get('models', {})
            if not models:
                raise ValueError("No models defined in configuration")
            
            model_issues = []
            for model_key, model_config in models.items():
                if not isinstance(model_config, dict):
                    model_issues.append(f"Model '{model_key}' config is not a dictionary")
                    continue
                
                if 'backend' not in model_config:
                    model_issues.append(f"Model '{model_key}' missing 'backend' field")
                
                if 'provider' not in model_config:
                    model_issues.append(f"Model '{model_key}' missing 'provider' field")
            
            # Validate default route
            default_route = self.config.get('default')
            routes = self.config.get('routes', {})
            
            if default_route not in models and default_route not in routes:
                model_issues.append(f"Default route '{default_route}' not found in models or routes")
            
            # Use enhanced validation if available
            if ENHANCEMENTS_AVAILABLE:
                validation_success = validate_and_log_config(self.config, logger)
                if not validation_success:
                    model_issues.append("Enhanced configuration validation failed")
            
            duration_ms = (time.time() - start_time) * 1000
            
            if model_issues:
                return TestResult(
                    name="Configuration Validation",
                    success=False,
                    duration_ms=duration_ms,
                    message=f"Configuration validation failed with {len(model_issues)} issues",
                    details={"issues": model_issues}
                )
            else:
                return TestResult(
                    name="Configuration Validation",
                    success=True,
                    duration_ms=duration_ms,
                    message="Configuration validation passed",
                    details={
                        "models_count": len(models),
                        "routes_count": len(routes),
                        "default_route": default_route,
                        "enhancements_available": ENHANCEMENTS_AVAILABLE
                    }
                )
                
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return TestResult(
                name="Configuration Validation",
                success=False,
                duration_ms=duration_ms,
                message="Configuration validation error",
                error=str(e)
            )
    
    async def setup_router(self) -> bool:
        """Initialize the LLM router for testing."""
        try:
            if not self.config:
                logger.error("No configuration loaded")
                return False
            
            # Create router with test metadata
            metadata = ComponentMetadata(
                id='test_llm_router',
                name='TestLLMRouter',
                version='1.0.0',
                category='test',
                description='LLM Router for testing purposes',
                requires_initialize=True
            )
            
            self.router = LLMRouter(self.config, metadata)
            
            # Create test context with mock registry
            test_context = NireonExecutionContext(
                run_id='llm_test_cli',
                component_id='test_llm_router',
                component_registry=MockComponentRegistry()
            )
            
            # Initialize router using proper base class method
            await self.router.initialize(test_context)
            
            logger.info("LLM Router initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup router: {e}")
            if "jsonpath_ng" in str(e):
                logger.error("Missing dependency: pip install jsonpath-ng")
            return False
    
    async def test_model_connectivity(self, model_key: str) -> TestResult:
        """Test connectivity to a specific model."""
        start_time = time.time()
        
        try:
            if not self.router:
                raise RuntimeError("Router not initialized")
            
            # Create test context
            test_context = NireonExecutionContext(
                run_id=f'connectivity_test_{model_key}',
                component_id='test_llm_router'
            )
            
            # Simple test prompt
            test_prompt = "Say 'Hello, this is a connectivity test.' and nothing else."
            
            # Test the model
            response = await self.router.call_llm_async(
                prompt=test_prompt,
                stage=EpistemicStage.DEFAULT,
                role='test',
                context=test_context,
                settings={'route': model_key, 'max_tokens': 50, 'temperature': 0.1}
            )
            
            duration_ms = (time.time() - start_time) * 1000
            
            # Check response
            if response.text and 'error' not in response:
                return TestResult(
                    name=f"Connectivity Test - {model_key}",
                    success=True,
                    duration_ms=duration_ms,
                    message=f"Successfully connected to {model_key}",
                    details={
                        "response_length": len(response.text),
                        "response_preview": response.text[:100] + "..." if len(response.text) > 100 else response.text
                    }
                )
            else:
                error_msg = response.get('error', 'Unknown error')
                return TestResult(
                    name=f"Connectivity Test - {model_key}",
                    success=False,
                    duration_ms=duration_ms,
                    message=f"Connection failed for {model_key}",
                    error=error_msg,
                    details={"response": dict(response)}
                )
                
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return TestResult(
                name=f"Connectivity Test - {model_key}",
                success=False,
                duration_ms=duration_ms,
                message=f"Exception during connectivity test for {model_key}",
                error=str(e)
            )
    
    async def test_all_connectivity(self) -> List[TestResult]:
        """Test connectivity to all configured models."""
        if not self.config:
            return [TestResult(
                name="All Connectivity Tests",
                success=False,
                duration_ms=0,
                message="No configuration loaded"
            )]
        
        models = self.config.get('models', {})
        results = []
        
        for model_key in models.keys():
            logger.info(f"Testing connectivity to {model_key}...")
            result = await self.test_model_connectivity(model_key)
            results.append(result)
            
            # Brief pause between tests
            await asyncio.sleep(1)
        
        return results
    
    def get_health_status(self) -> TestResult:
        """Get health status of all backends."""
        start_time = time.time()
        
        try:
            if not self.router:
                raise RuntimeError("Router not initialized")
            
            # Get health information
            health_info = {}
            
            # Basic router info
            health_info['router_stats'] = self.router.get_comprehensive_stats()
            
            # Enhanced health info if available
            if hasattr(self.router, 'get_backend_health'):
                health_info['backend_health'] = self.router.get_backend_health()
            
            if hasattr(self.router, 'get_circuit_breaker_stats'):
                health_info['circuit_breaker_stats'] = self.router.get_circuit_breaker_stats()
            
            if hasattr(self.router, 'get_metrics_summary'):
                health_info['metrics_summary'] = self.router.get_metrics_summary()
            
            duration_ms = (time.time() - start_time) * 1000
            
            # Determine overall health
            backend_health = health_info.get('backend_health', {})
            unhealthy_backends = [
                name for name, status in backend_health.items() 
                if not status.get('is_healthy', True)
            ]
            
            overall_healthy = len(unhealthy_backends) == 0
            
            return TestResult(
                name="Health Status Check",
                success=overall_healthy,
                duration_ms=duration_ms,
                message=f"System health: {'Healthy' if overall_healthy else f'Issues with {len(unhealthy_backends)} backends'}",
                details=health_info
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return TestResult(
                name="Health Status Check",
                success=False,
                duration_ms=duration_ms,
                message="Failed to get health status",
                error=str(e)
            )
    
    async def run_performance_test(self, model_key: str, num_calls: int = 5) -> TestResult:
        """Run performance test with multiple calls to a model."""
        start_time = time.time()
        
        try:
            if not self.router:
                raise RuntimeError("Router not initialized")
            
            results = []
            test_prompts = [
                "What is 2+2?",
                "Name three colors.",
                "What day comes after Monday?",
                "Count from 1 to 3.",
                "What is the capital of France?"
            ]
            
            for i in range(num_calls):
                call_start = time.time()
                
                test_context = NireonExecutionContext(
                    run_id=f'perf_test_{model_key}_{i}',
                    component_id='test_llm_router'
                )
                
                prompt = test_prompts[i % len(test_prompts)]
                
                response = await self.router.call_llm_async(
                    prompt=prompt,
                    stage=EpistemicStage.DEFAULT,
                    role='test',
                    context=test_context,
                    settings={'route': model_key, 'max_tokens': 100, 'temperature': 0.1}
                )
                
                call_duration = (time.time() - call_start) * 1000
                
                results.append({
                    'call_index': i,
                    'duration_ms': call_duration,
                    'success': 'error' not in response,
                    'response_length': len(response.text) if response.text else 0,
                    'error': response.get('error') if 'error' in response else None
                })
                
                # Brief pause between calls
                await asyncio.sleep(0.5)
            
            total_duration = (time.time() - start_time) * 1000
            
            # Calculate statistics
            successful_calls = [r for r in results if r['success']]
            failed_calls = [r for r in results if not r['success']]
            
            if successful_calls:
                durations = [r['duration_ms'] for r in successful_calls]
                avg_duration = sum(durations) / len(durations)
                min_duration = min(durations)
                max_duration = max(durations)
            else:
                avg_duration = min_duration = max_duration = 0
            
            success_rate = len(successful_calls) / num_calls
            
            return TestResult(
                name=f"Performance Test - {model_key}",
                success=success_rate > 0.8,  # 80% success rate threshold
                duration_ms=total_duration,
                message=f"Performance test completed: {success_rate:.1%} success rate",
                details={
                    'total_calls': num_calls,
                    'successful_calls': len(successful_calls),
                    'failed_calls': len(failed_calls),
                    'success_rate': success_rate,
                    'avg_duration_ms': avg_duration,
                    'min_duration_ms': min_duration,
                    'max_duration_ms': max_duration,
                    'call_results': results
                }
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return TestResult(
                name=f"Performance Test - {model_key}",
                success=False,
                duration_ms=duration_ms,
                message=f"Performance test failed for {model_key}",
                error=str(e)
            )
    
    def print_test_result(self, result: TestResult, verbose: bool = False):
        """Print a test result in a formatted way."""
        status = "✅ PASS" if result.success else "❌ FAIL"
        duration = f"{result.duration_ms:.1f}ms"
        
        print(f"{status} {result.name} ({duration})")
        print(f"    {result.message}")
        
        if result.error:
            print(f"    Error: {result.error}")
        
        if verbose and result.details:
            print(f"    Details: {json.dumps(result.details, indent=2)}")
        
        print()
    
    def print_summary(self):
        """Print a summary of all test results."""
        if not self.test_results:
            print("No test results to summarize.")
            return
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r.success)
        failed_tests = total_tests - passed_tests
        
        total_duration = sum(r.duration_ms for r in self.test_results)
        
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests} ✅")
        print(f"Failed: {failed_tests} ❌")
        print(f"Success Rate: {passed_tests/total_tests:.1%}")
        print(f"Total Duration: {total_duration:.1f}ms")
        print("="*60)
    
    async def interactive_test(self):
        """Interactive testing mode."""
        print("\n🔧 Interactive LLM Testing Mode")
        print("Type 'help' for commands, 'exit' to quit.")
        
        if not self.router:
            print("❌ Router not initialized. Please run setup first.")
            return
        
        while True:
            try:
                command = input("\nllm-test> ").strip().lower()
                
                if command == 'exit':
                    break
                elif command == 'help':
                    print("""
Available commands:
  help                     - Show this help
  models                   - List available models
  routes                   - List available routes  
  health                   - Show health status
  test <model>             - Test specific model
  perf <model> [calls]     - Performance test (default 5 calls)
  prompt <model> <text>    - Send custom prompt
  stats                    - Show router statistics
  exit                     - Exit interactive mode
                    """)
                elif command == 'models':
                    models = self.router.get_defined_models()
                    print(f"Available models: {', '.join(models)}")
                    
                elif command == 'routes':
                    routes = self.router.get_available_routes()
                    for route, model in routes.items():
                        print(f"  {route} -> {model}")
                        
                elif command == 'health':
                    result = self.get_health_status()
                    self.print_test_result(result, verbose=True)
                    
                elif command == 'stats':
                    if hasattr(self.router, 'get_comprehensive_stats'):
                        stats = self.router.get_comprehensive_stats()
                        print(json.dumps(stats, indent=2, default=str))
                    else:
                        print("Detailed stats not available")
                        
                elif command.startswith('test '):
                    model = command.split(' ', 1)[1]
                    print(f"Testing connectivity to {model}...")
                    result = await self.test_model_connectivity(model)
                    self.print_test_result(result)
                    
                elif command.startswith('perf '):
                    parts = command.split(' ')
                    model = parts[1]
                    calls = int(parts[2]) if len(parts) > 2 else 5
                    print(f"Running performance test on {model} with {calls} calls...")
                    result = await self.run_performance_test(model, calls)
                    self.print_test_result(result, verbose=True)
                    
                elif command.startswith('prompt '):
                    parts = command.split(' ', 2)
                    if len(parts) < 3:
                        print("Usage: prompt <model> <text>")
                        continue
                        
                    model = parts[1]
                    prompt_text = parts[2]
                    
                    print(f"Sending prompt to {model}...")
                    
                    test_context = NireonExecutionContext(
                        run_id=f'interactive_prompt_{int(time.time())}',
                        component_id='test_llm_router'
                    )
                    
                    start_time = time.time()
                    response = await self.router.call_llm_async(
                        prompt=prompt_text,
                        stage=EpistemicStage.DEFAULT,
                        role='interactive',
                        context=test_context,
                        settings={'route': model}
                    )
                    duration = (time.time() - start_time) * 1000
                    
                    print(f"\nResponse ({duration:.1f}ms):")
                    print("-" * 40)
                    print(response.text)
                    print("-" * 40)
                    
                    if 'error' in response:
                        print(f"Error: {response['error']}")
                        
                else:
                    print(f"Unknown command: {command}. Type 'help' for available commands.")
                    
            except KeyboardInterrupt:
                print("\nExiting interactive mode...")
                break
            except Exception as e:
                print(f"Error: {e}")

async def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="LLM Subsystem Testing CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s validate-config
  %(prog)s test-connectivity --model nano_default
  %(prog)s test-connectivity --all
  %(prog)s performance --model nano_default --calls 10
  %(prog)s health-check
  %(prog)s interactive
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        help='Path to LLM configuration file',
        default=None
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output with detailed information'
    )
    
    parser.add_argument(
        '--debug-paths',
        action='store_true',
        help='Print debug information about file paths and imports'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Validate config command
    validate_parser = subparsers.add_parser('validate-config', help='Validate LLM configuration')
    
    # Connectivity test command
    conn_parser = subparsers.add_parser('test-connectivity', help='Test LLM connectivity')
    conn_group = conn_parser.add_mutually_exclusive_group(required=True)
    conn_group.add_argument('--model', help='Test specific model')
    conn_group.add_argument('--all', action='store_true', help='Test all models')
    
    # Performance test command
    perf_parser = subparsers.add_parser('performance', help='Run performance tests')
    perf_parser.add_argument('--model', required=True, help='Model to test')
    perf_parser.add_argument('--calls', type=int, default=5, help='Number of test calls')
    
    # Health check command
    health_parser = subparsers.add_parser('health-check', help='Check system health')
    
    # Interactive mode command
    interactive_parser = subparsers.add_parser('interactive', help='Interactive testing mode')
    
    args = parser.parse_args()
    
    # Debug path information if requested
    if getattr(args, 'debug_paths', False):
        script_path = Path(__file__).resolve()
        print(f"🐛 DEBUG: Script path: {script_path}")
        print(f"🐛 DEBUG: Project root: {project_root}")
        print(f"🐛 DEBUG: Python path entries:")
        for i, path in enumerate(sys.path[:5]):
            print(f"    [{i}] {path}")
        print(f"🐛 DEBUG: Infrastructure exists: {(project_root / 'infrastructure').exists()}")
        print(f"🐛 DEBUG: Domain exists: {(project_root / 'domain').exists()}")
        print(f"🐛 DEBUG: Core exists: {(project_root / 'core').exists()}")
        print()
    
    if not args.command:
        parser.print_help()
        return
    
    # Set up logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize tester
    try:
        tester = LLMTester(args.config)
        
        # Load configuration
        if not tester.load_config():
            print("❌ Failed to load configuration")
            return 1
        
        print(f"📁 Using configuration: {tester.config_path}")
        if ENHANCEMENTS_AVAILABLE:
            print(f"🔧 Enhanced features: ✅ Available")
        else:
            print(f"🔧 Enhanced features: ⚠️  Basic mode (enhancements not installed)")
        print()
        
        # Handle commands
        if args.command == 'validate-config':
            result = tester.validate_config()
            tester.test_results.append(result)
            tester.print_test_result(result, args.verbose)
            
        elif args.command in ['test-connectivity', 'performance', 'health-check', 'interactive']:
            # These commands need router setup
            print("🚀 Initializing LLM Router...")
            if not await tester.setup_router():
                print("❌ Failed to initialize router")
                return 1
            
            if args.command == 'test-connectivity':
                if args.all:
                    results = await tester.test_all_connectivity()
                    tester.test_results.extend(results)
                    for result in results:
                        tester.print_test_result(result, args.verbose)
                else:
                    result = await tester.test_model_connectivity(args.model)
                    tester.test_results.append(result)
                    tester.print_test_result(result, args.verbose)
                    
            elif args.command == 'performance':
                result = await tester.run_performance_test(args.model, args.calls)
                tester.test_results.append(result)
                tester.print_test_result(result, args.verbose)
                
            elif args.command == 'health-check':
                result = tester.get_health_status()
                tester.test_results.append(result)
                tester.print_test_result(result, args.verbose)
                
            elif args.command == 'interactive':
                await tester.interactive_test()
        
        # Print summary if we have test results
        if tester.test_results:
            tester.print_summary()
            
            # Return error code if any tests failed
            if any(not r.success for r in tester.test_results):
                return 1
        
        return 0
        
    except Exception as e:
        logger.error(f"CLI error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == '__main__':
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)