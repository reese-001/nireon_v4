#  nireon_v4/signals/base.py
#  Optimised 2025-06-22
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_serializer, field_validator


class _CompositePolicy(str, Enum):
    """How to combine multiple scores into a composite score."""
    MIN = "min"
    MAX = "max"
    MEAN = "mean"


class EpistemicSignal(BaseModel):
    """
    Root object for all intra-NIREON signalling.

    * Public surface **unchanged** - internal helpers added.
    * Accepts arbitrary extra fields so future emitters can attach
      novel metadata without breaking older code.
    """
    # ------------------------------------------------------------------ #
    # Pydantic configuration
    # ------------------------------------------------------------------ #
    model_config = ConfigDict(
        extra="allow",
        # Prevents accidental in-place mutation of nested structures
        validate_assignment=True,
        # Use rapid orjson if available; falls back to stdlib json
        ser_json_timedelta="iso8601",
    )

    # ------------------------------------------------------------------ #
    # Core attributes (public contract – do not rename/remove!)
    # ------------------------------------------------------------------ #
    signal_type: str = Field(
        ...,
        description="The type of the signal (e.g., 'IdeaGenerated', 'TrustAssessment').",
    )
    source_node_id: str = Field(
        ...,
        description="The unique ID of the component that emitted the signal.",
    )
    payload: Dict[str, Any] = Field(
        default_factory=dict,
        description="The main data payload of the signal.",
    )
    trust_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Trust/confidence score for the signal content (0–1).",
    )
    novelty_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Novelty score indicating how unique the signal content is (0–1).",
    )
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confidence level of the emitting component (0–1).",
    )
    signal_id: str = Field(
        default_factory=lambda: f"sig_{uuid.uuid4()}",
        description="Unique identifier for this signal instance.",
    )
    parent_signal_ids: List[str] = Field(
        default_factory=list,
        description="IDs of parent signals that led to this signal (for lineage tracking).",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when the signal was created.",
    )
    context_tags: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary context tags for filtering or analysis.",
    )

    # ------------------------------------------------------------------ #
    # Private / derived attributes
    # ------------------------------------------------------------------ #
    _composite_policy: _CompositePolicy = PrivateAttr(default=_CompositePolicy.MIN)

    # ------------------------------------------------------------------ #
    # Validators & serializers
    # ------------------------------------------------------------------ #
    @field_validator("timestamp")
    @classmethod
    def _ensure_utc(cls, v: datetime) -> datetime:  # noqa: D401
        """Force naïve datetimes into UTC for consistency."""
        return v.replace(tzinfo=timezone.utc) if v.tzinfo is None else v.astimezone(timezone.utc)

    @field_serializer("timestamp")
    def _ts_iso(self, v: datetime, _info):  # noqa: D401
        """Serialize datetimes as ISO-8601 with explicit Z offset."""
        return v.astimezone(timezone.utc).isoformat()

    # ------------------------------------------------------------------ #
    # Convenience helpers (do **not** change external signature)
    # ------------------------------------------------------------------ #
    def with_lineage(self, parents: List["EpistemicSignal"]) -> "EpistemicSignal":
        """Return a *new* signal whose `parent_signal_ids` reference `parents`."""
        return self.copy(
            update={"parent_signal_ids": [p.signal_id for p in parents]},
            deep=True,
        )

    # Safety helpers ---------------------------------------------------- #
    def copy_payload(self) -> Dict[str, Any]:
        """Return a shallow *copy* of `payload` to avoid external mutation."""
        return self.payload.copy()

    # Scoring utilities ------------------------------------------------- #
    def composite_score(self, *, policy: _CompositePolicy | str | None = None) -> Optional[float]:
        """
        Collapse `trust_score`, `confidence`, and `novelty_score`
        into a single number in [0, 1] using `policy`.
        Returns ``None`` if no scores are set.
        """
        policy = _CompositePolicy(policy or self._composite_policy)
        scores = [s for s in (self.trust_score, self.confidence, self.novelty_score) if s is not None]
        if not scores:
            return None
        if policy is _CompositePolicy.MIN:
            return min(scores)
        if policy is _CompositePolicy.MAX:
            return max(scores)
        return sum(scores) / len(scores)  # MEAN

    def is_high_confidence(self, *, threshold: float = 0.7) -> bool:
        """True iff *all* available scores meet or exceed `threshold`."""
        scores = [s for s in (self.trust_score, self.confidence) if s is not None]
        return bool(scores) and all(s >= threshold for s in scores)

    def is_novel(self, *, threshold: float = 0.7) -> bool:
        """True iff `novelty_score` is defined and ≥ `threshold`."""
        return (self.novelty_score or 0.0) >= threshold

    # Representation ---------------------------------------------------- #
    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"{self.__class__.__name__}"
            f"(signal_type={self.signal_type!r}, "
            f"signal_id={self.signal_id!r}, "
            f"source_node_id={self.source_node_id!r})"
        )
