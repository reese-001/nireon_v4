# In domain\ideas\idea.py (full modified file)

from __future__ import annotations
import hashlib
import json
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

@dataclass(frozen=False)
class Idea:
    idea_id: str
    text: str
    parent_ids: List[str] = field(default_factory=list)
    children: List[str] = field(default_factory=list)
    world_facts: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    step: int = -1
    method: str = 'manual'
    metadata: Dict[str, Any] = field(default_factory=dict)
    trust_score: Optional[float] = None
    novelty_score: Optional[float] = None
    is_stable: Optional[bool] = None

    @classmethod
    def create(cls, text: str, parent_id: Optional[str]=None, *, step: int=-1, method: str='manual', metadata: Optional[Dict[str, Any]]=None, trust_score: Optional[float]=None, novelty_score: Optional[float]=None, is_stable: Optional[bool]=None, idea_id: Optional[str]=None) -> 'Idea':
        if not isinstance(text, str):
            raise TypeError(f'Idea text must be a string, got {type(text)}')
        
        # Use provided idea_id if given, else generate a new one
        if idea_id is None:
            new_idea_id = str(uuid.uuid4())
        else:
            new_idea_id = idea_id
        
        initial_parent_ids = [parent_id] if parent_id and isinstance(parent_id, str) and parent_id.strip() else []
        return cls(idea_id=new_idea_id, text=text.strip(), parent_ids=initial_parent_ids, step=int(step), method=str(method), metadata=metadata.copy() if metadata is not None else {}, trust_score=trust_score, novelty_score=novelty_score, is_stable=is_stable)

    def add_child(self, child_id: str) -> 'Idea':
        if not child_id or not isinstance(child_id, str):
            return self
        if child_id in self.children:
            return self
        return replace(self, children=[*self.children, child_id.strip()])

    def add_world_fact(self, fact_id: str) -> 'Idea':
        if not fact_id or not isinstance(fact_id, str):
            return self
        if fact_id in self.world_facts:
            return self
        return replace(self, world_facts=[*self.world_facts, fact_id.strip()])

    def with_method(self, method: str) -> 'Idea':
        if not isinstance(method, str):
            current_method = self.method
        else:
            current_method = method.strip()
        return replace(self, method=current_method)

    def with_step(self, step: int) -> 'Idea':
        try:
            current_step = int(step)
        except (ValueError, TypeError):
            current_step = self.step
        return replace(self, step=current_step)

    def with_metadata_update(self, updates: Dict[str, Any]) -> 'Idea':
        if not isinstance(updates, dict):
            return self
        new_metadata = self.metadata.copy()
        new_metadata.update(updates)
        return replace(self, metadata=new_metadata)

    def with_scores(self, trust_score: float, novelty_score: Optional[float]=None, is_stable: Optional[bool]=None) -> 'Idea':
        return replace(self, trust_score=trust_score, novelty_score=novelty_score, is_stable=is_stable)

    def with_trust_score(self, trust_score: float) -> 'Idea':
        return replace(self, trust_score=trust_score)

    def with_stability(self, is_stable: bool) -> 'Idea':
        return replace(self, is_stable=is_stable)

    def compute_hash(self, algorithm: str='sha256') -> str:
        if algorithm not in hashlib.algorithms_available:
            raise ValueError(f"Hashing algorithm '{algorithm}' is not available. Choose from: {hashlib.algorithms_available}")
        hasher = hashlib.new(algorithm)
        data_to_hash = {'idea_id': self.idea_id, 'text': self.text, 'parent_ids': sorted(list(set(self.parent_ids))), 'timestamp': self.timestamp.isoformat()}
        try:
            canonical_representation = json.dumps(data_to_hash, sort_keys=True, ensure_ascii=False).encode('utf-8')
        except TypeError as e:
            raise ValueError(f'Could not serialize idea content for hashing: {e}')
        hasher.update(canonical_representation)
        return hasher.hexdigest()

    def has_scores(self) -> bool:
        return self.trust_score is not None

    def is_high_trust(self, threshold: float=6.0) -> bool:
        return self.trust_score is not None and self.trust_score > threshold

    def __repr__(self) -> str:
        text_preview = self.text[:70] + '...' if len(self.text) > 73 else self.text
        metadata_preview_items = []
        for k, v in list(self.metadata.items())[:3]:
            v_repr = repr(v)
            v_preview = v_repr[:30] + '...' if len(v_repr) > 33 else v_repr
            metadata_preview_items.append(f'{k!r}: {v_preview}')
        metadata_str = '{' + ', '.join(metadata_preview_items)
        if len(self.metadata) > 3:
            metadata_str += ', ...'
        metadata_str += '}'
        score_info = ''
        if self.trust_score is not None:
            score_info = f', trust={self.trust_score:.2f}'
        if self.novelty_score is not None:
            score_info += f', novelty={self.novelty_score:.2f}'
        if self.is_stable is not None:
            score_info += f', stable={self.is_stable}'
        return f"{self.__class__.__name__}(idea_id='{self.idea_id}', text='{text_preview!r}', parent_ids={self.parent_ids!r}, children_count={len(self.children)}, world_facts_count={len(self.world_facts)}, timestamp='{self.timestamp.isoformat()}', step={self.step}, method='{self.method}', metadata_preview={metadata_str}{score_info})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Idea):
            return self.idea_id == other.idea_id
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.idea_id)