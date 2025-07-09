# nireon_v4/00_explorer_runner/orchestrator.py

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
from bootstrap import bootstrap_nireon_system
from signals import SeedSignal, EpistemicSignal
from signals.core import ProtoResultSignal, MathProtoResultSignal, ProtoErrorSignal
from core.registry.component_registry import ComponentRegistry
from domain.ports.event_bus_port import EventBusPort
from domain.ports.idea_service_port import IdeaServicePort
from domain.context import NireonExecutionContext
from result_capturer import ResultCapturer
from debug_helpers import DebugInspector
from dag_logger import DAGLogger, EnhancedResultCapturer, DAGVisualizationExporter

class ExplorerOrchestrator:
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.registry: Optional[ComponentRegistry] = None
        self.event_bus: Optional[EventBusPort] = None
        self.idea_service: Optional[IdeaServicePort] = None
        self.context: Optional[NireonExecutionContext] = None
        self.reactor = None
        self.bootstrap_result = None
        
        # Initialize DAG logger if enabled
        if config.get('dag_logging', {}).get('enabled', True):
            dag_output_dir = Path(config.get('dag_logging', {}).get('output_dir', './dag_logs'))
            dag_output_dir.mkdir(parents=True, exist_ok=True)
            dag_log_file = dag_output_dir / f'dag_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
            self.dag_logger = DAGLogger(logger, dag_log_file)
            self.logger.info(f'DAG logging enabled. Output: {dag_log_file}')
        else:
            self.dag_logger = None
            
    async def bootstrap(self):
        self.logger.info('Starting NIREON bootstrap process...')
        
        # Log bootstrap start
        if self.dag_logger:
            self.dag_logger.log_node(
                node_id="BOOTSTRAP",
                node_type="SYSTEM",
                metadata={"phase": "initialization"}
            )
        
        from utils import find_project_root
        project_root = find_project_root()
        # Ensure project_root is a Path object
        if project_root is not None:
            project_root = Path(project_root)
        else:
            # If we can't find project root, assume we're already in it
            project_root = Path.cwd()
        manifest_path = project_root / self.config['bootstrap']['manifest']
        
        if not manifest_path.exists():
            raise FileNotFoundError(f'Manifest not found: {manifest_path}')
        
        self.bootstrap_result = await bootstrap_nireon_system(
            config_paths=[manifest_path],
            strict_mode=self.config['bootstrap']['strict_mode']
        )
        
        if not self.bootstrap_result.success:
            self.logger.error('Bootstrap failed!')
            return self.bootstrap_result
        
        self.registry = self.bootstrap_result.registry
        self.event_bus = self.registry.get_service_instance(EventBusPort)
        self.idea_service = self.registry.get_service_instance(IdeaServicePort)
        
        self.context = NireonExecutionContext(
            run_id=f'explorer_run_{int(time.time())}',
            component_registry=self.registry,
            event_bus=self.event_bus,
            config=self.bootstrap_result.global_config,
            feature_flags=self.bootstrap_result.global_config.get('feature_flags', {})
        )
        
        if hasattr(self.bootstrap_result, 'reactor'):
            self.reactor = self.bootstrap_result.reactor
            
        # Log successful bootstrap
        if self.dag_logger:
            self.dag_logger.log_event(
                event_type="BOOTSTRAP_COMPLETE",
                node_id="BOOTSTRAP",
                details={
                    "component_count": self.bootstrap_result.component_count,
                    "run_id": self.context.run_id
                }
            )
            
            # Log key components as nodes
            critical_components = [
                ('sentinel_instance_01', 'SentinelMechanism'),
                ('active_planner_instance_01', 'ActivePlanner'),
                ('quantifier_agent_primary', 'QuantifierAgent'),
                ('proto_engine_default', 'ProtoEngine')
            ]
            
            for component_id, component_type in critical_components:
                try:
                    component = self.registry.get(component_id)
                    if component:
                        self.dag_logger.log_node(
                            node_id=component_id,
                            node_type="COMPONENT",
                            metadata={"component_type": component_type}
                        )
                except Exception:
                    pass
        
        self.logger.info(f'Bootstrap complete: {self.bootstrap_result.component_count} components loaded')
        
        if self.config['debug']['enable_reactor_rules_check']:
            inspector = DebugInspector(self.registry, self.event_bus, self.logger)
            inspector.check_reactor_rules()
            
        if self.config['debug']['enable_quantifier_check']:
            inspector = DebugInspector(self.registry, self.event_bus, self.logger)
            inspector.check_quantifier_setup()
        
        # After bootstrap completes successfully
        if self.bootstrap_result.success:
            # NEW: Add diagnostic command
            self.logger.info("\n💡 TIP: You can manually check top ideas and trigger catalyst:")
            self.logger.info("   await orchestrator.show_top_ideas()")
            self.logger.info("   await orchestrator.trigger_catalyst_with_best_ideas()")
            
        return self.bootstrap_result
    
    async def run_seeds(self, seeds: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run seeds with support for multiple iterations."""
        iterations = self.config['execution'].get('iterations', 1)
        
        if iterations > 1:
            self.logger.info(f'Running {len(seeds)} seeds for {iterations} iterations each')
        
        # Create expanded seed list with iterations
        expanded_seeds = []
        for seed in seeds:
            for iteration in range(1, iterations + 1):
                # Create a copy of the seed with iteration info
                seed_copy = seed.copy()
                seed_copy['_iteration'] = iteration
                seed_copy['_original_id'] = seed['id']
                # Modify the ID to include iteration number
                seed_copy['id'] = f"{seed['id']}_iter{iteration}"
                expanded_seeds.append(seed_copy)
        
        results = []
        
        # Run seeds (existing logic)
        if self.config['execution']['parallel_execution']:
            results = await self._run_seeds_parallel(expanded_seeds)
        else:
            results = await self._run_seeds_sequential(expanded_seeds)
            
        # NEW: After all seeds are run, show the top ideas
        self.logger.info("\n" + "="*80)
        self.logger.info("SHOWING TOP IDEAS AFTER ALL RUNS")
        self.logger.info("="*80)
        
        # Show top 10 ideas
        top_ideas = self.show_top_ideas(top_n=10)
        
        # Check if we should trigger catalyst
        if top_ideas:
            high_trust_count = sum(1 for idea in top_ideas if idea.trust_score and idea.trust_score > 6.0)
            if high_trust_count >= 3:
                self.logger.info(f"\n✨ Found {high_trust_count} high-trust ideas. Triggering Catalyst...")
                await self.trigger_catalyst_with_best_ideas(count=2)
        
        return results
    
    async def _run_seeds_sequential(self, seeds: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run seeds sequentially with iteration support."""
        results = []
        total_seeds = len(seeds)
        
        for idx, seed_config in enumerate(seeds, 1):
            iteration = seed_config.get('_iteration', 1)
            original_id = seed_config.get('_original_id', seed_config['id'])
            
            self.logger.info(f"\n{'=' * 60}")
            self.logger.info(f"Running seed {idx}/{total_seeds}: {original_id} (iteration {iteration})")
            self.logger.info(f"{'=' * 60}")
            
            result = await self._run_single_seed(seed_config, idx)
            results.append(result)
            
            if idx < total_seeds:
                await asyncio.sleep(2)
        
        return results
    
    async def _run_seeds_parallel(self, seeds: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        self.logger.info(f'Running {len(seeds)} seeds in parallel...')
        
        tasks = []
        for idx, seed_config in enumerate(seeds, 1):
            task = self._run_single_seed(seed_config, idx)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                final_results.append({
                    'seed_id': seeds[i]['id'],
                    'test_passed': False,
                    'error': str(result),
                    'failure_reason': f'Exception: {result}'
                })
            else:
                final_results.append(result)
        
        return final_results
    
    async def _run_single_seed(self, seed_config: Dict[str, Any], index: int) -> Dict[str, Any]:
        """Run a single seed with iteration awareness."""
        start_time = datetime.now()
        seed_id = seed_config['id']
        seed_text = seed_config['text']
        objective = seed_config['objective']
        iteration = seed_config.get('_iteration', 1)
        original_id = seed_config.get('_original_id', seed_id)
        
        # Log seed execution start
        if self.dag_logger:
            self.dag_logger.log_node(
                node_id=f"SEED_EXEC_{seed_id}",
                node_type="SEED_EXECUTION",
                metadata={
                    "seed_id": seed_id,
                    "original_id": original_id,
                    "iteration": iteration,
                    "objective": objective[:100]
                }
            )
        
        try:
            # Create seed idea
            seed_idea = self.idea_service.create_idea(
                text=seed_text,
                parent_id=None,
                context=self.context
            )
            
            self.logger.info(f'Created seed idea: {seed_idea.idea_id}')
            
            # Create capturer - use enhanced version if DAG logging is enabled
            if self.dag_logger:
                capturer = EnhancedResultCapturer(
                    seed_idea.idea_id,
                    seed_text,
                    self.event_bus,
                    self.config,
                    self.dag_logger
                )
            else:
                capturer = ResultCapturer(
                    seed_idea.idea_id,
                    seed_text,
                    self.event_bus,
                    self.config
                )
            
            capturer.subscribe_to_signals()
            
            # Prepare seed signal
            payload = {
                'seed_idea_id': seed_idea.idea_id,
                'text': seed_idea.text,
                'metadata': {
                    'iteration': iteration,  # Include actual iteration number
                    'total_iterations': self.config['execution'].get('iterations', 1),
                    'original_seed_id': original_id,
                    'objective': objective.strip(),
                    'depth': 0,
                    'seed_config_id': seed_id
                }
            }
            
            seed_signal = SeedSignal(
                source_node_id='ExplorerOrchestrator',
                payload=payload,
                run_id=self.context.run_id
            )
            
            # Log seed signal emission
            if self.dag_logger:
                self.dag_logger.log_signal(
                    signal_type="SeedSignal",
                    source="ExplorerOrchestrator",
                    target=seed_idea.idea_id,
                    payload=payload
                )
            
            # Process signal
            if self.reactor:
                await self.reactor.process_signal(seed_signal)
            else:
                self.event_bus.publish(SeedSignal.__name__, seed_signal)
            
            # Wait for completion
            timeout = self.config['execution']['timeout']
            test_passed = True
            failure_reason = None
            
            timeout = self.config['execution']['timeout']
            completion_type = self.config['execution'].get('completion_condition', {}).get('type', 'all_assessed')
            test_passed = False
            failure_reason = None

            run_start_time = time.time()

            self.logger.info("Waiting for completion event (e.g., GenerativeLoopFinishedSignal)...")
            try:
                # Wait for the capturer's event, which is set by signals
                await asyncio.wait_for(capturer.completion_event.wait(), timeout=timeout)
                self.logger.info('✅ Completion event received.')
                test_passed = True
            except asyncio.TimeoutError:
                self.logger.error(f'❌ Timeout after {timeout}s waiting for completion event.')
                test_passed = False
                failure_reason = f'Timeout after {timeout}s'
            
            # Check for proto execution
            if capturer.proto_task_detected:
                proto_result = await self._wait_for_proto_execution()
                if proto_result and proto_result.get('success'):
                    self.logger.info('✅ Proto execution successful')
                    
                    if self.dag_logger:
                        self.dag_logger.log_event(
                            event_type="PROTO_EXECUTION_SUCCESS",
                            node_id=proto_result.get('proto_id', 'unknown'),
                            details=proto_result
                        )
                else:
                    test_passed = False
                    failure_reason = 'Proto execution failed'
                    
                    if self.dag_logger:
                        self.dag_logger.log_event(
                            event_type="PROTO_EXECUTION_FAILURE",
                            node_id=proto_result.get('proto_id', 'unknown') if proto_result else 'unknown',
                            details=proto_result or {}
                        )
            
            # Finalize and prepare result
            capturer.finalize()
            
            # NEW: Optionally show top ideas after each seed
            if self.config.get('debug', {}).get('show_progress_ideas', False):
                self.logger.info(f"\n📊 Top ideas after seed {index}:")
                self.show_top_ideas(top_n=5)
            
            result = {
                'seed_id': seed_id,
                'seed_text': seed_text,
                'objective': objective,
                'test_passed': test_passed,
                'failure_reason': failure_reason,
                'start_time': start_time.isoformat(),
                'end_time': datetime.now().isoformat(),
                'duration_seconds': (datetime.now() - start_time).total_seconds(),
                'run_data': capturer.run_data,
                'summary_stats': capturer.get_summary_stats()
            }
            
            capturer.unsubscribe_from_signals()
            
            return result
            
        except Exception as e:
            self.logger.error(f'Error running seed {seed_id}: {e}', exc_info=True)
            
            if self.dag_logger:
                self.dag_logger.log_event(
                    event_type="SEED_EXECUTION_ERROR",
                    node_id=f"SEED_EXEC_{seed_id}",
                    details={"error": str(e)}
                )
            
            return {
                'seed_id': seed_id,
                'test_passed': False,
                'error': str(e),
                'failure_reason': f'Exception: {e}',
                'start_time': start_time.isoformat(),
                'end_time': datetime.now().isoformat()
            }
    
    async def _wait_for_proto_execution(self, timeout: float = 30.0) -> Optional[Dict[str, Any]]:
        self.logger.info('Waiting for Proto execution...')
        
        proto_result_event = asyncio.Event()
        proto_result = None
        
        def proto_callback(payload):
            nonlocal proto_result
            signal_data = payload.get('payload', payload) if isinstance(payload, dict) else payload
            
            if hasattr(signal_data, 'signal_type'):
                if signal_data.signal_type in ['ProtoResultSignal', 'MathProtoResultSignal']:
                    proto_result = {
                        'success': True,
                        'proto_id': getattr(signal_data, 'proto_block_id', 'unknown'),
                        'artifacts': getattr(signal_data, 'artifacts', [])
                    }
                    proto_result_event.set()
                elif signal_data.signal_type == 'ProtoErrorSignal':
                    proto_result = {
                        'success': False,
                        'proto_id': getattr(signal_data, 'proto_block_id', 'unknown'),
                        'error': getattr(signal_data, 'error_message', 'Unknown error')
                    }
                    proto_result_event.set()
        
        self.event_bus.subscribe('ProtoResultSignal', proto_callback)
        self.event_bus.subscribe('MathProtoResultSignal', proto_callback)
        self.event_bus.subscribe('ProtoErrorSignal', proto_callback)
        
        try:
            await asyncio.wait_for(proto_result_event.wait(), timeout=timeout)
            return proto_result
        except asyncio.TimeoutError:
            self.logger.warning(f'Proto execution timeout after {timeout}s')
            return None
        finally:
            self.event_bus.unsubscribe('ProtoResultSignal', proto_callback)
            self.event_bus.unsubscribe('MathProtoResultSignal', proto_callback)
            self.event_bus.unsubscribe('ProtoErrorSignal', proto_callback)

    def show_top_ideas(self, top_n=10):
        if not self.idea_service:
            self.logger.error('Idea service not available')
            return []
        repo = self.idea_service.repository
        high_trust_ideas = repo.get_high_trust_ideas(min_trust=0.0, limit=top_n)
        self.logger.info(f"\n{'=' * 80}")
        self.logger.info(f'TOP {top_n} IDEAS BY TRUST SCORE')
        self.logger.info(f"{'=' * 80}")
        for i, idea in enumerate(high_trust_ideas, 1):
            # FIX: Prepare the novelty score string beforehand
            novelty_str = f"{idea.novelty_score:.2f}" if idea.novelty_score is not None else 'N/A'
            self.logger.info(f"\n{i}. {idea.idea_id[:8]}... (Trust: {idea.trust_score:.2f}, Novelty: {novelty_str})")
            self.logger.info(f'   Text: {idea.text[:100]}...')
            self.logger.info(f'   Stable: {idea.is_stable}')
            self.logger.info(f'   Method: {idea.method}')
        return high_trust_ideas
            
    def get_catalyst_candidates(self, trust_threshold=7.0):
        """Get ideas suitable for catalyst based on scores"""
        if not self.idea_service:
            self.logger.error("Idea service not available")
            return []
            
        repo = self.idea_service.repository
        candidates = repo.get_high_trust_ideas(min_trust=trust_threshold, limit=50)
        
        # Filter for stable ideas
        stable_candidates = [
            idea for idea in candidates 
            if idea.is_stable is True
        ]
        
        self.logger.info(f"\nFound {len(stable_candidates)} stable ideas with trust > {trust_threshold}")
        
        return stable_candidates

    async def trigger_catalyst_with_best_ideas(self, count=3):
        """Manually trigger catalyst with the best available ideas"""
        candidates = self.get_catalyst_candidates(trust_threshold=5.0)
        
        if not candidates:
            self.logger.warning("No suitable catalyst candidates found")
            return
            
        catalyst = self.registry.get('catalyst_instance_01')
        if not catalyst:
            self.logger.error("Catalyst component not found")
            return
            
        # Take top N candidates
        best_ideas = candidates[:count]
        
        for idea in best_ideas:
            self.logger.info(f"🚀 Manually triggering Catalyst with idea {idea.idea_id[:8]}... (trust={idea.trust_score:.2f})")
            
            await catalyst.process({
                'target_idea_id': idea.idea_id,
                'objective': 'Amplify and blend high-quality concepts'
            }, self.context)
            
            # Small delay between catalyst triggers
            await asyncio.sleep(0.5)
    
    async def shutdown(self):
        """Clean up and export DAG data"""
        if self.dag_logger:
            output_dir = Path(self.config.get('dag_logging', {}).get('output_dir', './dag_logs'))
            
            # Export raw graph data
            graph_data_path = output_dir / f'graph_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            self.dag_logger.export_graph_data(graph_data_path)
            self.logger.info(f'DAG data exported to: {graph_data_path}')
            
            # Auto-generate visualizations if enabled
            if self.config.get('dag_logging', {}).get('visualization', {}).get('auto_generate', True):
                exporter = DAGVisualizationExporter()
                
                # Parse the log file
                log_data = exporter.parse_log_file(self.dag_logger.output_file)
                
                formats = self.config.get('dag_logging', {}).get('visualization', {}).get('formats', ['graphviz', 'mermaid'])
                
                if 'graphviz' in formats:
                    # Export to Graphviz
                    dot_path = output_dir / f'execution_dag_{datetime.now().strftime("%Y%m%d_%H%M%S")}.dot'
                    exporter.export_to_graphviz(log_data, dot_path)
                    self.logger.info(f'Graphviz DOT file exported to: {dot_path}')
                
                if 'mermaid' in formats:
                    # Export to Mermaid
                    mermaid_path = output_dir / f'execution_dag_{datetime.now().strftime("%Y%m%d_%H%M%S")}.mmd'
                    mermaid_content = exporter.export_to_mermaid(log_data)
                    with open(mermaid_path, 'w') as f:
                        f.write(mermaid_content)
                    self.logger.info(f'Mermaid diagram exported to: {mermaid_path}')