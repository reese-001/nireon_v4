import asyncio
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Callable

from domain.ports.event_bus_port import EventBusPort
from signals import IdeaGeneratedSignal, TrustAssessmentSignal

# --------------------------------------------------------------------------- #
#  Types & helpers
# --------------------------------------------------------------------------- #

@dataclass(slots=True)
class IdeaNode:
    id: str
    text: str
    source: str
    depth: int
    trust_score: Optional[float] = None
    is_stable: Optional[bool] = None
    variations: Dict[str, "IdeaNode"] = field(default_factory=dict)

class Sig(Enum):
    IDEA_GEN   = IdeaGeneratedSignal.__name__
    TRUST_ASSESS = TrustAssessmentSignal.__name__
    PROTO_TASK = 'ProtoTaskSignal'
    LOOP_DONE  = 'GenerativeLoopFinishedSignal'
    CATALYZED  = 'IdeaCatalyzedSignal'

# --------------------------------------------------------------------------- #
#  Capturer
# --------------------------------------------------------------------------- #

class ResultCapturer:
    """Collects runtime telemetry for a single DAG run in O(1) per‑event time."""
    __slots__ = (
        'logger', 'event_bus', 'config', 'run_data', 'node_by_id',
        'generated_ids', 'assessed_ids', 'completion_event',
        'proto_task_detected', 'proto_task_signal_data',
        'catalyst_triggered', '_sig_handlers', '_threshold'
    )

    # ------------------------------------------------------------------ init #
    def __init__(self, seed_idea_id: str, seed_text: str,
                 event_bus: EventBusPort, config: Dict[str, Any]) -> None:

        self.logger  = logging.getLogger('nireon.result_capturer')
        self.event_bus = event_bus
        self.config  = config
        self._threshold = config['criteria']['min_trust_score_for_quantifier']

        # Core data                                                                       
        seed_node         = IdeaNode(seed_idea_id, seed_text, 'seed', 0)
        self.node_by_id   = {seed_idea_id: seed_node}
        self.generated_ids: Set[str] = {seed_idea_id}
        self.assessed_ids:  Set[str] = set()

        # Minimal run‑wide dictionary (everything else can be recomputed)                 
        self.run_data: Dict[str, Any] = {
            'seed_idea' : asdict(seed_node),
            'metadata'  : {
                'run_start_time' : datetime.utcnow().isoformat(),
                'signals_received': {s.name: 0 for s in Sig},
                'high_trust_ideas': [],
                'max_depth_reached': 0
            },
            'events'    : []
        }

        # Async completion flag
        self.completion_event = asyncio.Event()

        # Flags
        self.proto_task_detected = False
        self.proto_task_signal_data: Optional[Dict[str, Any]] = None
        self.catalyst_triggered  = False

        # Register handlers once
        self._sig_handlers: Dict[str, Callable[[Any], None]] = {
            Sig.IDEA_GEN.value:   self._on_idea_gen,
            Sig.TRUST_ASSESS.value: self._on_trust_assess,
            Sig.PROTO_TASK.value:   self._on_proto_task,
            Sig.LOOP_DONE.value:    self._on_loop_done,
            Sig.CATALYZED.value:    self._on_catalyzed,
        }

        for sig, h in self._sig_handlers.items():
            self.event_bus.subscribe(sig, h)
        self.logger.debug('ResultCapturer subscribed to signals')

    # ---------------------------------------------------------- signal utils #
    def _add_event(self, typ: str, **details: Any) -> None:
        self.run_data['events'].append({
            'ts': datetime.utcnow().isoformat(),
            'type': typ,
            'details': details
        })

    # -------------------------------------------------------------- handlers #
    def _on_idea_gen(self, s: IdeaGeneratedSignal) -> None:
        payload = getattr(s, 'payload', s)  # works for both objects & dicts
        idea_id  = payload.get('id') or getattr(s, 'idea_id', None)
        if not idea_id:  # badly formed signal – drop
            return
        parent_id = payload.get('parent_id') or self.run_data['seed_idea']['id']
        text      = payload.get('text', 'No text')
        source    = payload.get('source_mechanism', 'unknown')
        depth     = payload.get('depth', 0)

        node = IdeaNode(idea_id, text, source, depth)
        self.node_by_id[idea_id] = node
        self.generated_ids.add(idea_id)

        # cheap parent linkage
        self.node_by_id[parent_id].variations[idea_id] = node

        md = self.run_data['metadata']
        md['signals_received'][Sig.IDEA_GEN.name] += 1
        md['max_depth_reached'] = max(md['max_depth_reached'], depth)

        self._add_event('idea', idea_id=idea_id, parent=parent_id, depth=depth)

    def _on_trust_assess(self, s: TrustAssessmentSignal | Dict[str, Any]) -> None:
        payload = getattr(s, 'payload', s)
        idea_id = payload.get('target_id') or payload.get('idea_id')
        if not idea_id:
            return
        score   = payload.get('trust_score')
        stable  = payload.get('is_stable')

        node = self.node_by_id.get(idea_id)
        if node:
            node.trust_score = score
            node.is_stable   = stable

        self.assessed_ids.add(idea_id)
        md = self.run_data['metadata']
        md['signals_received'][Sig.TRUST_ASSESS.name] += 1

        if score is not None and score > self._threshold:
            md['high_trust_ideas'].append({'id': idea_id, 'score': score})

        self._add_event('trust', idea_id=idea_id, score=score, stable=stable)
        self._maybe_complete()

    def _on_proto_task(self, s: Any) -> None:
        self.proto_task_detected = True
        self.proto_task_signal_data = getattr(s, 'payload', s)
        self.run_data['metadata']['signals_received'][Sig.PROTO_TASK.name] += 1
        self._add_event('proto_task')
        self._maybe_complete()

    def _on_catalyzed(self, s: Any) -> None:
        self.catalyst_triggered = True
        self.run_data['metadata']['signals_received'][Sig.CATALYZED.name] += 1
        self._add_event('catalyst')
        self._maybe_complete()

    def _on_loop_done(self, s: Any) -> None:
        self.run_data['metadata']['signals_received'][Sig.LOOP_DONE.name] += 1
        self._add_event('loop_done')
        # No completion here – rely on other conditions

    # ----------------------------------------------------------- completion  #
    def _maybe_complete(self) -> None:
        if self.proto_task_detected or self.catalyst_triggered:
            self.completion_event.set()
            return
        if len(self.generated_ids) >= 10 and self.generated_ids <= self.assessed_ids:
            self.completion_event.set()

    # ----------------------------------------------------------- finalise    #
    def finalize(self) -> None:
        md = self.run_data['metadata']
        md['run_end_time'] = datetime.utcnow().isoformat()
        dur = (
            datetime.fromisoformat(md['run_end_time'])
            - datetime.fromisoformat(md['run_start_time'])
        ).total_seconds() or 1e-3
        md['duration_seconds'] = dur
        md['ideas_per_second'] = len(self.generated_ids) / dur
        md['assessment_coverage'] = (
            len(self.assessed_ids) / len(self.generated_ids) * 100
        )

    # --------------------------------------------------------- public stats  #
    def get_summary_stats(self) -> Dict[str, Any]:
        md = self.run_data['metadata']
        return {
            'ideas': len(self.generated_ids),
            'assessed': len(self.assessed_ids),
            'high_trust': len(md['high_trust_ideas']),
            'max_depth': md['max_depth_reached'],
            'proto_triggered': self.proto_task_detected,
            'catalyst_triggered': self.catalyst_triggered,
            'signals': md['signals_received']
        }
