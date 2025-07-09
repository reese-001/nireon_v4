# nireon_v4/00_explorer_runner/run_explorer.py

import argparse
import asyncio
import sys
from pathlib import Path
from datetime import datetime
import yaml
from typing import Dict, Any, List, Optional

# Setup paths
RUNNER_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(RUNNER_DIR))

from utils import find_project_root, setup_logging, set_component
from orchestrator import ExplorerOrchestrator
from report_generator import ReportGenerator

# Determine project root
PROJECT_ROOT = find_project_root()
if PROJECT_ROOT is None:
    print('ERROR: Could not determine the NIREON V4 project root.')
    sys.exit(1)

# Add project root to Python path
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f'ERROR: Failed to load config from {config_path}: {e}')
        sys.exit(1)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='NIREON V4 Explorer - End-to-End Generative Flow Validator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Run with default config
  python run_explorer.py
  
  # Run specific seeds
  python run_explorer.py --seeds retail_survival education_future
  
  # Run with custom timeout
  python run_explorer.py --timeout 180
  
  # Run in parallel mode
  python run_explorer.py --parallel
  
  # Use custom config file
  python run_explorer.py --config custom_config.yaml
  
  # Disable DAG logging
  python run_explorer.py --no-dag
        '''
    )
    
    parser.add_argument(
        '--config', '-c',
        type=Path,
        default=RUNNER_DIR / 'config.yaml',
        help='Path to configuration file (default: ./config.yaml)'
    )
    
    parser.add_argument(
        '--seeds', '-s',
        nargs='+',
        help='Specific seed IDs to run (default: all seeds in config)'
    )
    
    parser.add_argument(
        '--timeout', '-t',
        type=float,
        help='Override timeout from config (seconds)'
    )
    
    parser.add_argument(
        '--iterations', '-n',
        type=int,
        help='Override number of iterations from config'
    )
    
    parser.add_argument(
        '--parallel', '-p',
        action='store_true',
        help='Run seeds in parallel'
    )
    
    parser.add_argument(
        '--no-report',
        action='store_true',
        help='Skip report generation'
    )
    
    parser.add_argument(
        '--no-dag',
        action='store_true',
        help='Disable DAG logging'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be run without executing'
    )
    
    return parser.parse_args()


async def main():
    """Main execution function"""
    args = parse_arguments()
    
    # Load configuration
    config = load_config(args.config)
    
    # Apply command line overrides
    if args.timeout:
        config['execution']['timeout'] = args.timeout
    if args.iterations:
        config['execution']['iterations'] = args.iterations
    if args.parallel:
        config['execution']['parallel_execution'] = True
    if args.debug:
        config['debug']['log_level'] = 'DEBUG'
    if args.no_dag:
        config['dag_logging']['enabled'] = False
    
    # Setup logging
    logger = setup_logging(config['debug'])
    set_component('ExplorerMain')
    
    # Print header
    logger.info('=' * 80)
    logger.info('NIREON V4 EXPLORER TEST RUNNER')
    logger.info('=' * 80)
    logger.info(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f'Config File: {args.config}')
    logger.info(f'Project Root: {PROJECT_ROOT}')
    
    # Select seeds to run
    selected_seeds = config['seeds']
    if args.seeds:
        selected_seeds = [s for s in config['seeds'] if s['id'] in args.seeds]
        if not selected_seeds:
            logger.error(f'No matching seeds found for: {args.seeds}')
            sys.exit(1)
    
    logger.info(f'Seeds to Process: {len(selected_seeds)}')
    for seed in selected_seeds:
        logger.info(f"  - {seed['id']}: {seed['text'][:60]}...")
    
    # Show DAG logging status
    if config.get('dag_logging', {}).get('enabled', True):
        logger.info(f"DAG Logging: ENABLED (output: {config['dag_logging']['output_dir']})")
    else:
        logger.info("DAG Logging: DISABLED")
    
    # Dry run mode
    if args.dry_run:
        logger.info('\n🔸 DRY RUN MODE - No actual execution will occur')
        logger.info('Configuration Summary:')
        logger.info(f"  - Timeout: {config['execution']['timeout']}s")
        logger.info(f"  - Iterations: {config['execution']['iterations']}")
        logger.info(f"  - Parallel: {config['execution']['parallel_execution']}")
        logger.info(f'  - Report Generation: {not args.no_report}')
        logger.info(f"  - DAG Logging: {config.get('dag_logging', {}).get('enabled', True)}")
        return
    
    # Create orchestrator
    orchestrator = ExplorerOrchestrator(config, logger)
    
    try:
        # Bootstrap system
        logger.info('\n🚀 Bootstrapping NIREON system...')
        bootstrap_result = await orchestrator.bootstrap()
        
        if not bootstrap_result.success:
            logger.error('Bootstrap failed! See errors above.')
            sys.exit(1)
        
        # Run tests
        logger.info('\n🧪 Starting test execution...')
        test_results = await orchestrator.run_seeds(selected_seeds)
        
        # Generate reports
        if not args.no_report:
            logger.info('\n📊 Generating reports...')
            report_gen = ReportGenerator(config, logger)
            report_paths = report_gen.generate_all_reports(test_results)
            
            logger.info('Reports generated:')
            for report_type, path in report_paths.items():
                logger.info(f'  - {report_type}: {path}')
        
        # Export DAG visualizations
        if config.get('dag_logging', {}).get('enabled', True):
            logger.info('\n📈 Exporting DAG visualizations...')
            await orchestrator.shutdown()
        
        # Print summary
        logger.info('\n' + '=' * 80)
        logger.info('EXECUTION SUMMARY')
        logger.info('=' * 80)
        
        total_runs = len(test_results)
        successful_runs = sum(1 for r in test_results if r.get('test_passed', False))
        failed_runs = total_runs - successful_runs
        
        logger.info(f'Total Runs: {total_runs}')
        logger.info(f'✅ Successful: {successful_runs}')
        logger.info(f'❌ Failed: {failed_runs}')
        
        if failed_runs > 0:
            logger.info('\nFailed Runs:')
            for result in test_results:
                if not result.get('test_passed', False):
                    logger.info(f"  - {result['seed_id']}: {result.get('failure_reason', 'Unknown')}")
        
        # Exit with appropriate code
        sys.exit(0 if failed_runs == 0 else 1)
        
    except KeyboardInterrupt:
        logger.warning('\n⚠️  Execution interrupted by user')
        sys.exit(130)
    except Exception as e:
        logger.error(f'\n💥 Unexpected error: {e}', exc_info=True)
        sys.exit(1)
    finally:
        logger.info(f"\nEnd Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    # Change to project root if needed
    if Path.cwd() != PROJECT_ROOT:
        import os
        os.chdir(PROJECT_ROOT)
        print(f'Changed CWD to project root: {PROJECT_ROOT}')
    
    # Run the main async function
    asyncio.run(main())