import json
import logging
from typing import Dict, Any, List, Optional, Set, Tuple
from datetime import datetime
from pathlib import Path
import re
import time
import asyncio

from signals.base import EpistemicSignal  # Import asyncio for the Event

class DAGLogger:
    def __init__(self, base_logger: logging.Logger, output_file: Optional[Path]=None):
        self.logger = base_logger
        self.output_file = output_file
        self.nodes: Set[str] = set()
        self.edges: List[Tuple[str, str, Dict[str, Any]]] = []
        self.signal_flows: List[Dict[str, Any]] = []
        self.events: List[Dict[str, Any]] = []
        if output_file:
            handler = logging.FileHandler(output_file, mode='w')
            handler.setFormatter(logging.Formatter('%(message)s'))
            self.dag_logger = logging.getLogger('dag_logger')
            self.dag_logger.addHandler(handler)
            self.dag_logger.setLevel(logging.INFO)
        else:
            self.dag_logger = self.logger

    def log_node(self, node_id: str, node_type: str, metadata: Dict[str, Any]=None):
        self.nodes.add(node_id)
        log_entry = {'type': 'NODE', 'timestamp': datetime.now().isoformat(), 'node_id': node_id, 'node_type': node_type, 'metadata': metadata or {}}
        self.dag_logger.info(f'DAG|{json.dumps(log_entry)}')

    def log_edge(self, source: str, target: str, edge_type: str, metadata: Dict[str, Any]=None):
        self.edges.append((source, target, {'type': edge_type, 'metadata': metadata or {}}))
        log_entry = {'type': 'EDGE', 'timestamp': datetime.now().isoformat(), 'source': source, 'target': target, 'edge_type': edge_type, 'metadata': metadata or {}}
        self.dag_logger.info(f'DAG|{json.dumps(log_entry)}')

    def log_signal(self, signal_type: str, source: str, target: Optional[str]=None, payload: Dict[str, Any]=None):
        signal_data = {'type': 'SIGNAL', 'timestamp': datetime.now().isoformat(), 'signal_type': signal_type, 'source': source, 'target': target, 'payload': payload or {}}
        self.signal_flows.append(signal_data)
        self.dag_logger.info(f'DAG|{json.dumps(signal_data)}')

    def log_event(self, event_type: str, node_id: str, details: Dict[str, Any]=None):
        event_data = {'type': 'EVENT', 'timestamp': datetime.now().isoformat(), 'event_type': event_type, 'node_id': node_id, 'details': details or {}}
        self.events.append(event_data)
        self.dag_logger.info(f'DAG|{json.dumps(event_data)}')

    def export_graph_data(self, output_path: Path):
        graph_data = {
            'nodes': list(self.nodes),
            'edges': [{'source': e[0], 'target': e[1], **e[2]} for e in self.edges],
            'signals': self.signal_flows,
            'events': self.events
        }
        with open(output_path, 'w') as f:
            json.dump(graph_data, f, indent=2)

class EnhancedResultCapturer:
    def __init__(self, seed_idea_id: str, seed_text: str, event_bus, config: Dict[str, Any], dag_logger: DAGLogger):
        self.event_bus = event_bus
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.dag_logger = dag_logger
        self.assessments = {}
        self.run_data: Dict[str, Any] = {
            'seed_idea': {
                'id': seed_idea_id,
                'text': seed_text,
                'trust_score': None,
                'is_stable': None,
                'variations': {}
            },
            'metadata': {
                'run_start_time': datetime.now().isoformat(),
                'run_end_time': None,
                'total_ideas': 1,
                'total_assessments': 0,
                'high_trust_ideas': [],
                'max_depth_reached': 0,
                'signals_received': {
                    'IdeaGeneratedSignal': 0,
                    'TrustAssessmentSignal': 0,
                    'ProtoTaskSignal': 0,
                    'GenerativeLoopFinishedSignal': 0
                }
            },
            'events': []
        }
        self.idea_map: Dict[str, Dict] = {seed_idea_id: self.run_data['seed_idea']['variations']}
        self._idea_to_parent_map: Dict[str, str] = {}
        self.generated_idea_ids: Set[str] = {seed_idea_id}
        self.assessed_idea_ids: Set[str] = set()
        self.completion_event = asyncio.Event()
        self.proto_task_detected = False
        self.proto_task_signal_data = None
        self.signal_timings: Dict[str, List[float]] = {}
        self.dag_logger.log_node(node_id=seed_idea_id, node_type='SEED_IDEA', metadata={'text': seed_text[:100], 'is_root': True})
        self.last_signal_timestamp = time.time()

    def subscribe_to_signals(self):
        subscriptions = [
            ('IdeaGeneratedSignal', self._handle_idea_generated),
            ('TrustAssessmentSignal', self._handle_trust_assessment),
            ('ProtoTaskSignal', self._handle_proto_task),
            ('GenerativeLoopFinishedSignal', self._handle_generative_loop_finished)
        ]
        for signal_name, handler in subscriptions:
            self.event_bus.subscribe(signal_name, handler)
        self.logger.info('[ResultCapturer] Subscribed to all signals')

    def unsubscribe_from_signals(self):
        # In a real implementation, you would unsubscribe here.
        # For simplicity in this context, we can leave it as a pass.
        pass

    def _add_event(self, event_type: str, details: Dict[str, Any]):
        self.run_data['events'].append({
            'timestamp': datetime.now().isoformat(),
            'type': event_type,
            'details': details
        })

    def _signal_to_dict(self, signal: Any) -> Dict[str, Any]:
        if hasattr(signal, 'model_dump'):
            return signal.model_dump()
        elif hasattr(signal, '__dict__'):
            result = {}
            for key, value in signal.__dict__.items():
                if not key.startswith('_'):
                    if hasattr(value, 'model_dump'):
                        result[key] = value.model_dump()
                    elif hasattr(value, '__dict__'):
                        result[key] = self._signal_to_dict(value)
                    else:
                        result[key] = value
            return result
        else:
            return {'signal_type': type(signal).__name__, 'data': str(signal)}

    def _handle_idea_generated(self, payload: 'EpistemicSignal'):
        self.last_signal_timestamp = time.time()
        try:
            # Log the signal payload for debugging
            self.logger.debug(f"[DAG] Received IdeaGeneratedSignal payload: {self._signal_to_dict(payload)}")
            
            self.run_data['metadata']['signals_received']['IdeaGeneratedSignal'] += 1
            if hasattr(payload, 'payload'):
                signal_payload = payload.payload if isinstance(payload.payload, dict) else {}
                parent_id = signal_payload.get('parent_id') or getattr(payload, 'parent_id', None)
                idea_id = signal_payload.get('id') or getattr(payload, 'idea_id', None) or getattr(payload, 'target_id', None)
                text = signal_payload.get('text') or getattr(payload, 'idea_content', None) or getattr(payload, 'text', None)
                source = signal_payload.get('source_mechanism') or getattr(payload, 'source_node_id', None)
                depth = signal_payload.get('metadata', {}).get('depth', signal_payload.get('depth', 0))
            else:
                signal_payload = payload.get('payload', payload)
                parent_id = signal_payload.get('parent_id')
                idea_id = signal_payload.get('id') or payload.get('idea_id')
                text = signal_payload.get('text') or payload.get('idea_content')
                source = signal_payload.get('source_mechanism') or payload.get('source_node_id')
                depth = signal_payload.get('metadata', {}).get('depth', signal_payload.get('depth', 0))
            
            # Skip if idea_id is the seed - seed should never have a parent
            if idea_id == self.run_data['seed_idea']['id']:
                self.logger.warning(f"[DAG] Skipping parent set for seed idea {idea_id} (signal had parent={parent_id})")
                return  # Don't add to map or tree for seed re-emissions
            
            # Check for duplicates
            if idea_id in self._idea_to_parent_map:
                self.logger.warning(f"[DAG] Duplicate signal for {idea_id}, skipping (existing parent={self._idea_to_parent_map[idea_id]})")
                return
            
            # Enhanced cycle prevention check
            temp_curr = parent_id
            seen = set()
            while temp_curr in self._idea_to_parent_map:
                if temp_curr == idea_id:
                    self.logger.error(f"[DAG] Cycle prevented: Setting {idea_id} → {parent_id} would create loop. Path: {list(seen)}")
                    return  # Skip adding
                if temp_curr in seen:
                    break  # Already checked
                seen.add(temp_curr)
                temp_curr = self._idea_to_parent_map[temp_curr] 

            if idea_id:
                self.generated_idea_ids.add(idea_id)
                self.logger.info(f'[ResultCapturer] Generated idea {idea_id} (total: {len(self.generated_idea_ids)}) at depth {depth}')
                self.dag_logger.log_node(node_id=idea_id, node_type='GENERATED_IDEA', metadata={'text_preview': text[:100] if text else 'No text', 'depth': depth, 'source_mechanism': source})

                if depth > self.run_data['metadata']['max_depth_reached']:
                    self.run_data['metadata']['max_depth_reached'] = depth
                
                if not parent_id:
                    parent_id = self.run_data['seed_idea']['id']
                    self.logger.debug(f'[ResultCapturer] No parent_id for {idea_id}, using seed')

                initial_parent_ids = [parent_id] if parent_id else []

                if parent_id:
                    self.dag_logger.log_edge(source=parent_id, target=idea_id, edge_type='GENERATES', metadata={'generation_mechanism': source, 'depth': depth})

                self.dag_logger.log_signal(signal_type='IdeaGeneratedSignal', source=source or 'unknown', target=idea_id, payload={'parent_id': parent_id, 'depth': depth})

                if parent_id not in self.idea_map:
                    self.idea_map[parent_id] = {}

                new_idea_node = {
                    'id': idea_id,
                    'text': text or 'No text provided',
                    'source_mechanism': source or 'unknown',
                    'trust_score': None,
                    'is_stable': None,
                    'depth': depth,
                    'variations': {}
                }
                if parent_id in self.idea_map:
                    self.idea_map[parent_id][idea_id] = new_idea_node
                    self.idea_map[idea_id] = new_idea_node['variations']
                    
                    # Final cycle detection before setting parent
                    self.logger.debug(f"[DAG] Setting parent for idea {idea_id} to {parent_id}")
                    if idea_id == parent_id:
                        self.logger.error(f"[DAG] Self-loop detected! Idea {idea_id} cannot be its own parent.")
                    if parent_id in self._idea_to_parent_map and self._idea_to_parent_map[parent_id] == idea_id:
                        self.logger.error(f"[DAG] Potential cycle: Setting {idea_id} → {parent_id}, but {parent_id} → {idea_id} already exists.")
                    
                    self._idea_to_parent_map[idea_id] = parent_id
                    self.run_data['metadata']['total_ideas'] += 1
                    self._add_event('idea_generated', {'idea_id': idea_id, 'parent_id': parent_id, 'source': source, 'depth': depth})
        except Exception as e:
            self.logger.error(f'[ResultCapturer] Error handling IdeaGeneratedSignal: {e}', exc_info=True)

    def _handle_trust_assessment(self, signal: Any) -> None:
        self.last_signal_timestamp = time.time()
        try:
            self.run_data['metadata']['signals_received']['TrustAssessmentSignal'] += 1
            idea_id = signal.target_id
            trust_score = signal.trust_score
            if isinstance(signal.payload, dict):
                is_stable = signal.payload.get('is_stable', False)
            else:
                is_stable = getattr(signal.payload, 'is_stable', False)

            self.logger.info(f'[ResultCapturer] Trust assessment captured: idea_id={idea_id}, trust_score={trust_score}, is_stable={is_stable}')
            self.assessments[idea_id] = {'trust_score': trust_score, 'is_stable': is_stable, 'assessment_time': time.time()}

            if idea_id:
                self.assessed_idea_ids.add(idea_id)
                self.logger.info(f'[ResultCapturer] Assessed idea {idea_id} (total assessed: {len(self.assessed_idea_ids)})')
            
            self.dag_logger.log_event(event_type='TRUST_ASSESSMENT', node_id=idea_id, details={'trust_score': trust_score, 'is_stable': is_stable})

            node_to_update = self._find_idea_node(idea_id)
            if node_to_update:
                node_to_update['trust_score'] = trust_score
                node_to_update['is_stable'] = is_stable
                self.run_data['metadata']['total_assessments'] += 1
                threshold = self.config.get('criteria', {}).get('min_trust_score_for_quantifier', 8.0)
                if trust_score and trust_score > threshold:
                    self.run_data['metadata']['high_trust_ideas'].append({'idea_id': idea_id, 'trust_score': trust_score})
                    self.logger.info(f'[ResultCapturer] HIGH TRUST: {idea_id} = {trust_score:.2f}')
                self._add_event('trust_assessment', {'idea_id': idea_id, 'trust_score': trust_score, 'is_stable': is_stable, 'high_trust': trust_score > threshold if trust_score else False})
            
            self._check_for_completion()
        except Exception as e:
            self.logger.error(f'[ResultCapturer] Error handling TrustAssessmentSignal: {e}', exc_info=True)

    def _check_for_completion(self):
        """Checks if the run has met its completion criteria and sets the event if so."""
        # Condition 1: A proto-task has been generated. This is a terminal success state.
        if self.proto_task_detected:
            self.logger.info("[ResultCapturer] Completion condition met: ProtoTask detected.")
            if not self.completion_event.is_set():
                self.completion_event.set()
            return

        # Condition 2: All generated ideas have been assessed.
        # This prevents completion before any new ideas are generated.
        if len(self.generated_idea_ids) > 1 and self.generated_idea_ids.issubset(self.assessed_idea_ids):
            self.logger.info("[ResultCapturer] Completion condition met: All generated ideas have been assessed.")
            if not self.completion_event.is_set():
                self.completion_event.set()
            return

    def _handle_proto_task(self, raw_payload: Dict[str, Any]):
        self.last_signal_timestamp = time.time()
        self.run_data['metadata']['signals_received']['ProtoTaskSignal'] += 1
        self.logger.info('[ResultCapturer] ProtoTaskSignal detected!')
        self.proto_task_detected = True
        if hasattr(raw_payload, 'model_dump'):
            signal_data = raw_payload.model_dump()
        elif isinstance(raw_payload, dict):
            signal_data = raw_payload
        else:
            signal_data = self._signal_to_dict(raw_payload)
        
        self.proto_task_signal_data = signal_data
        proto_block = signal_data.get('payload', {}).get('proto_block', signal_data.get('proto_block', {}))
        proto_id = proto_block.get('id', 'unknown_proto')
        source_idea = signal_data.get('payload', {}).get('source_idea_id', signal_data.get('source_idea_id'))

        self.dag_logger.log_node(node_id=proto_id, node_type='PROTO_TASK', metadata={
            'description': proto_block.get('description', ''),
            'eidos': proto_block.get('eidos', 'unknown'),
            'objective': proto_block.get('objective', '')
        })
        if source_idea:
            self.dag_logger.log_edge(source=source_idea, target=proto_id, edge_type='TRIGGERS_PROTO', metadata={'quantifier_triggered': True})
        
        self.dag_logger.log_signal(signal_type='ProtoTaskSignal', source='quantifier_agent', target=proto_id, payload={'source_idea': source_idea, 'eidos': proto_block.get('eidos', 'unknown')})
        self._add_event('proto_task_created', {'payload': signal_data})
        self._check_for_completion()

    def _handle_generative_loop_finished(self, raw_payload: Any):
        self.last_signal_timestamp = time.time()
        try:
            self.run_data['metadata']['signals_received']['GenerativeLoopFinishedSignal'] += 1
            payload = raw_payload.payload if hasattr(raw_payload, 'payload') else raw_payload
            status = payload.get('status', 'unknown')
            frame_id = payload.get('frame_id', 'unknown')
            self.logger.info(f"[ResultCapturer] Received GenerativeLoopFinishedSignal for frame {frame_id} with status '{status}'.")
            self._add_event('loop_finished', {'frame_id': frame_id, 'status': status})
            if not self.completion_event.is_set():
                self.completion_event.set()
        except Exception as e:
            self.logger.error(f'[ResultCapturer] Error handling GenerativeLoopFinishedSignal: {e}', exc_info=True)
            
    def _find_idea_node(self, idea_id: str) -> Optional[Dict]:
        if idea_id == self.run_data['seed_idea']['id']:
            return self.run_data['seed_idea']

        path = []
        curr_id = idea_id
        seen = set()  # Track visited to detect cycles
        count = 0
        while curr_id in self._idea_to_parent_map:
            self.logger.debug(f"[DAG] Traversing parent: {curr_id} (path so far: {path})")
            if curr_id in seen:
                self.logger.error(f"[DAG] Cycle detected in parent map! Cycle involves: {curr_id}. Full path: {path + [curr_id]}")
                return None  # Break out to prevent infinite loop
            seen.add(curr_id)
            path.insert(0, curr_id)
            curr_id = self._idea_to_parent_map[curr_id]
            count += 1
            if count > 100:  # Safety threshold (adjust if your depths are deeper)
                self.logger.error(f"[DAG] Excessive traversal (possible cycle)! Path: {path}. Aborting.")
                return None

        if curr_id != self.run_data['seed_idea']['id']:
            self.logger.warning(f"[DAG] Idea {idea_id} does not connect to seed root. Current: {curr_id}")
            return None

        node = self.run_data['seed_idea']
        for step_id in path:
            node = node['variations'].get(step_id)
            if node is None:
                self.logger.warning(f"[DAG] Missing node in path for {idea_id}: {step_id}")
                return None
        return node
    
    def finalize(self):
        self.run_data['metadata']['run_end_time'] = datetime.now().isoformat()
        start_time = datetime.fromisoformat(self.run_data['metadata']['run_start_time'])
        end_time = datetime.fromisoformat(self.run_data['metadata']['run_end_time'])
        duration = (end_time - start_time).total_seconds()
        self.run_data['metadata']['duration_seconds'] = duration
        self.run_data['metadata']['ideas_per_second'] = len(self.generated_idea_ids) / duration if duration > 0 else 0
        self.run_data['metadata']['assessment_coverage'] = len(self.assessed_idea_ids) / len(self.generated_idea_ids) * 100 if self.generated_idea_ids else 0
        self.dag_logger.log_event(
            event_type='CAPTURE_FINALIZED',
            node_id='SYSTEM',
            details={
                'duration_seconds': duration,
                'total_ideas': len(self.generated_idea_ids),
                'total_assessed': len(self.assessed_idea_ids),
                'coverage_percent': self.run_data['metadata']['assessment_coverage']
            }
        )
        
    def get_summary_stats(self) -> Dict[str, Any]:
        return {
            'total_ideas': len(self.generated_idea_ids),
            'total_assessed': len(self.assessed_idea_ids),
            'high_trust_count': len(self.run_data['metadata']['high_trust_ideas']),
            'max_depth': self.run_data['metadata']['max_depth_reached'],
            'proto_triggered': self.proto_task_detected,
            'signals_received': self.run_data['metadata']['signals_received']
        }

class DAGVisualizationExporter:
    @staticmethod
    def parse_log_file(log_path: Path) -> Dict[str, Any]:
        nodes = {}
        edges = []
        signals = []
        events = []
        with open(log_path, 'r') as f:
            for line in f:
                if 'DAG|' in line:
                    try:
                        json_start = line.index('DAG|') + 4
                        json_data = json.loads(line[json_start:].strip())
                        if json_data['type'] == 'NODE':
                            nodes[json_data['node_id']] = {'id': json_data['node_id'], 'type': json_data['node_type'], 'metadata': json_data['metadata']}
                        elif json_data['type'] == 'EDGE':
                            edges.append({'source': json_data['source'], 'target': json_data['target'], 'type': json_data['edge_type'], 'metadata': json_data['metadata']})
                        elif json_data['type'] == 'SIGNAL':
                            signals.append(json_data)
                        elif json_data['type'] == 'EVENT':
                            events.append(json_data)
                    except (json.JSONDecodeError, KeyError) as e:
                        continue
        return {'nodes': list(nodes.values()), 'edges': edges, 'signals': signals, 'events': events}

    @staticmethod
    def export_to_graphviz(data: Dict[str, Any], output_path: Path):
        dot_content = ['digraph ExecutionDAG {']
        dot_content.append('    rankdir=TB;')
        dot_content.append('    node [shape=box, style=rounded];')
        node_styles = {
            'SEED_IDEA': 'fillcolor=lightgreen, style="rounded,filled"',
            'GENERATED_IDEA': 'fillcolor=lightblue, style="rounded,filled"',
            'PROTO_TASK': 'fillcolor=orange, style="rounded,filled"',
            'COMPONENT': 'fillcolor=pink, style="rounded,filled"',
            'SYSTEM': 'fillcolor=lavender, style="rounded,filled"',
            'SEED_EXECUTION': 'fillcolor=khaki, style="rounded,filled"'
        }
        for node in data['nodes']:
            label = f"{node['id'][:8]}...\\n{node['type']}"
            if 'text_preview' in node.get('metadata', {}):
                text = node['metadata']['text_preview'][:30] + '...'
                label += f'\\n{text}'
            style = node_styles.get(node['type'], 'fillcolor=lightgray, style="rounded,filled"')
            dot_content.append(f'''    "{node['id']}" [label="{label}", {style}];''')
        for edge in data['edges']:
            label = edge['type']
            if edge.get('metadata', {}).get('depth') is not None:
                label += f"\\ndepth={edge['metadata']['depth']}"
            dot_content.append(f'''    "{edge['source']}" -> "{edge['target']}" [label="{label}"];''')
        dot_content.append('}')
        with open(output_path, 'w') as f:
            f.write('\n'.join(dot_content))

    @staticmethod
    def export_to_mermaid(data: Dict[str, Any]) -> str:
        mermaid_lines = ['graph TD']
        for node in data['nodes']:
            node_id = node['id'].replace('-', '_')
            label = f"{node['id'][:8]}...<br/>{node['type']}"
            if node['type'] == 'SEED_IDEA':
                mermaid_lines.append(f'    {node_id}[{label}]:::seed')
            elif node['type'] == 'GENERATED_IDEA':
                mermaid_lines.append(f'    {node_id}[{label}]:::idea')
            elif node['type'] == 'PROTO_TASK':
                mermaid_lines.append(f'    {node_id}[{label}]:::proto')
            elif node['type'] == 'COMPONENT':
                mermaid_lines.append(f'    {node_id}[{label}]:::component')
            else:
                mermaid_lines.append(f'    {node_id}[{label}]')
        
        for edge in data['edges']:
            source = edge['source'].replace('-', '_')
            target = edge['target'].replace('-', '_')
            label = edge['type']
            if edge['type'] == 'TRIGGERS_PROTO':
                mermaid_lines.append(f'    {source} -.->|{label}| {target}')
            else:
                mermaid_lines.append(f'    {source} -->|{label}| {target}')
        
        mermaid_lines.extend([
            '',
            '    classDef seed fill:#90EE90,stroke:#333,stroke-width:2px;',
            '    classDef idea fill:#ADD8E6,stroke:#333,stroke-width:2px;',
            '    classDef proto fill:#FFA500,stroke:#333,stroke-width:2px;',
            '    classDef component fill:#FFB6C1,stroke:#333,stroke-width:2px;'
        ])
        return '\n'.join(mermaid_lines)