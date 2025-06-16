# nireon/domain/ideas/idea.py
"""
Domain entity representing an Idea.

An Idea is the core knowledge unit inside NIREON.  It holds free-text,
parent/child relationships, optional references to world-facts, and
metadata such as generation step or method.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone # Added timezone
from typing import Any, Dict, List, Optional


@dataclass(frozen=False) # Changed to frozen=False to allow __post_init__ if needed, but methods maintain immutability
class Idea:
    # ───────────────────────────────
    # Core fields
    # ───────────────────────────────
    idea_id: str
    text: str
    parent_ids: List[str] = field(default_factory=list)
    children: List[str] = field(default_factory=list)
    world_facts: List[str] = field(default_factory=list)
    # Use timezone-aware UTC datetime by default
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Generation metadata
    step: int = -1
    method: str = "manual"
    metadata: Dict[str, Any] = field(default_factory=dict)

    # --------------------------------------------------------------------- #
    # Factory
    # --------------------------------------------------------------------- #
    @classmethod
    def create(
        cls,
        text: str,
        parent_id: Optional[str] = None,
        *,
        step: int = -1,
        method: str = "manual",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "Idea":
        """
        Factory method to create a new Idea with a unique ID and explicit UTC timestamp.

        Extra keyword-arguments allow callers to pre-populate `step`,
        `method`, or arbitrary `metadata`.
        """
        # Ensure text is a string, handle potential non-string input gracefully or raise error
        if not isinstance(text, str):
            # Option 1: Raise an error for strictness
            raise TypeError(f"Idea text must be a string, got {type(text)}")
            # Option 2: Attempt to convert (could be risky depending on requirements)
            # text = str(text)

        # Generate a more robust UUID (version 4)
        new_idea_id = str(uuid.uuid4())
        
        # Initialize parent_ids carefully
        initial_parent_ids = [parent_id] if parent_id and isinstance(parent_id, str) and parent_id.strip() else []

        return cls(
            idea_id=new_idea_id,
            text=text.strip(), # Normalize text by stripping whitespace
            parent_ids=initial_parent_ids,
            step=int(step), # Ensure step is an integer
            method=str(method), # Ensure method is a string
            metadata=metadata.copy() if metadata is not None else {}, # Ensure metadata is a new dict
            # timestamp will be set by default_factory
        )

    # --------------------------------------------------------------------- #
    # Relationship helpers
    # --------------------------------------------------------------------- #
    def add_child(self, child_id: str) -> "Idea":
        """Return a *new* Idea instance with `child_id` appended to children, ensuring uniqueness."""
        if not child_id or not isinstance(child_id, str): # Basic validation for child_id
            # Optionally log a warning or raise an error
            # For now, return self to maintain current behavior for invalid input
            return self 
            
        if child_id in self.children:
            return self
        
        # Use dataclasses.replace for more robust and concise immutable updates
        return replace(self, children=[*self.children, child_id.strip()])
    
    

    def add_world_fact(self, fact_id: str) -> "Idea":
        """Return a *new* Idea instance with `fact_id` appended to world_facts, ensuring uniqueness."""
        if not fact_id or not isinstance(fact_id, str): # Basic validation for fact_id
            return self

        if fact_id in self.world_facts:
            return self
        
        return replace(self, world_facts=[*self.world_facts, fact_id.strip()])

    # --------------------------------------------------------------------- #
    # Mutators (immutability style)
    # --------------------------------------------------------------------- #
    def with_method(self, method: str) -> "Idea":
        """Return a new Idea with `method` updated."""
        if not isinstance(method, str):
            # Consider raising TypeError or defaulting/logging
            current_method = self.method # Keep current method if new one is invalid
        else:
            current_method = method.strip()

        return replace(self, method=current_method)

    def with_step(self, step: int) -> "Idea":
        """Return a new Idea with `step` updated."""
        try:
            current_step = int(step)
        except (ValueError, TypeError):
            # Consider raising error or defaulting/logging
            current_step = self.step # Keep current step if new one is invalid

        return replace(self, step=current_step)
    
    def with_metadata_update(self, updates: Dict[str, Any]) -> "Idea":
        """Return a new Idea with its metadata updated by the `updates` dictionary."""
        if not isinstance(updates, dict):
            # Log warning or raise error
            return self
            
        new_metadata = self.metadata.copy()
        new_metadata.update(updates)
        return replace(self, metadata=new_metadata)

    # --------------------------------------------------------------------- #
    # Utilities
    # --------------------------------------------------------------------- #
    def compute_hash(self, algorithm: str = 'sha256') -> str:
        """
        Compute a deterministic hash of the Idea’s significant immutable fields.

        Allows specifying the hashing algorithm.
        Fields included: idea_id, text, sorted parent_ids, timestamp (ISO format).
        Sorting parent_ids makes the hash independent of their original order if
        the set of parents is what matters for identity.
        """
        if algorithm not in hashlib.algorithms_available:
            raise ValueError(f"Hashing algorithm '{algorithm}' is not available. Choose from: {hashlib.algorithms_available}")

        hasher = hashlib.new(algorithm)

        # Create a dictionary of elements to be hashed.
        # Ensure canonical representation for complex types.
        # Sorting parent_ids ensures hash consistency regardless of their order in the list,
        # if the *set* of parents is what matters for identity.
        # If the specific order of parent_ids is crucial, do not sort.
        # For this refactoring, assuming the set of parents is more significant.
        data_to_hash = {
            "idea_id": self.idea_id,
            "text": self.text, # Text is usually case-sensitive and whitespace sensitive for identity
            "parent_ids": sorted(list(set(self.parent_ids))), # Sort unique parent IDs
            "timestamp": self.timestamp.isoformat(), # Consistent ISO format
            # Consider if other fields like `method` or a subset of `metadata`
            # should contribute to the hash depending on the definition of "identity".
        }

        try:
            # Use separators for better JSON structure and to avoid ambiguities.
            # compact_json = json.dumps(data_to_hash, sort_keys=True, separators=(',', ':'))
            # For even more robustness, consider a canonical JSON library or a more structured serialization.
            # For now, standard json.dumps with sort_keys is a good improvement.
            canonical_representation = json.dumps(data_to_hash, sort_keys=True, ensure_ascii=False).encode("utf-8")
        except TypeError as e:
            # This might happen if metadata contains non-serializable types by mistake
            raise ValueError(f"Could not serialize idea content for hashing: {e}")


        hasher.update(canonical_representation)
        return hasher.hexdigest()

    def __repr__(self) -> str:
        """Provides a developer-friendly, unambiguous string representation of the Idea instance."""
        text_preview = (self.text[:70] + '...') if len(self.text) > 73 else self.text
        # Ensure metadata preview is concise and handles potential large values
        metadata_preview_items = []
        for k, v in list(self.metadata.items())[:3]: # Preview first 3 metadata items
            v_repr = repr(v)
            v_preview = (v_repr[:30] + '...') if len(v_repr) > 33 else v_repr
            metadata_preview_items.append(f"{k!r}: {v_preview}")
        metadata_str = "{" + ", ".join(metadata_preview_items)
        if len(self.metadata) > 3:
            metadata_str += ", ..."
        metadata_str += "}"


        return (
            f"{self.__class__.__name__}("
            f"idea_id='{self.idea_id}', "
            f"text='{text_preview!r}', " # Use !r for text to show quotes
            f"parent_ids={self.parent_ids!r}, "
            f"children_count={len(self.children)}, "
            f"world_facts_count={len(self.world_facts)}, "
            f"timestamp='{self.timestamp.isoformat()}', "
            f"step={self.step}, "
            f"method='{self.method}', "
            f"metadata_preview={metadata_str}"
            ")"
        )

    def __eq__(self, other: object) -> bool:
        """Checks for equality based on the idea_id."""
        if isinstance(other, Idea):
            return self.idea_id == other.idea_id
        return NotImplemented

    def __hash__(self) -> int:
        """Allows Idea instances to be used in hash-based collections (e.g., sets, dict keys)."""
        return hash(self.idea_id)