"""
Calculates the novelty of an idea by comparing its embedding against a set
of reference embeddings.
"""
import asyncio
import logging
from typing import Sequence, Tuple, TYPE_CHECKING

# V4 CHANGE: Import paths are flattened, assuming nireon_v4 is the root.
from domain.ideas.idea import Idea
from .constants import DEFAULT_NOVELTY_SCORE

if TYPE_CHECKING:
    from .service import SentinelMechanism

logger = logging.getLogger(__name__)


class NoveltyCalculator:
    """A helper class to encapsulate novelty calculation logic for Sentinel."""
    def __init__(self, sentinel: 'SentinelMechanism'):
        self.sentinel = sentinel

    async def calculate_novelty(
        self,
        idea: Idea,
        refs: Sequence[Idea]
    ) -> Tuple[float, str]:
        """
        Calculates the novelty score for a single idea against a list of references.

        The novelty score is scaled from 1 (not novel) to 10 (very novel).
        """
        if not refs:
            return DEFAULT_NOVELTY_SCORE, 'no_reference_ideas'

        relevant_refs = [r for r in refs if r.idea_id != idea.idea_id and r.text]
        if not relevant_refs:
            return DEFAULT_NOVELTY_SCORE, 'no_valid_reference_ideas_after_filter'

        try:
            # V4 CHANGE: The EmbeddingPort methods are async, so we await them directly.
            # The asyncio.to_thread wrapper is no longer necessary or correct.
            if self.sentinel.embed is None:
                raise RuntimeError("EmbeddingPort dependency is not available in Sentinel.")

            idea_vector = self.sentinel.embed.encode(idea.text)
            ref_texts = [r.text for r in relevant_refs]
            ref_vectors = self.sentinel.embed.encode_batch(ref_texts)

            if not ref_vectors:
                return DEFAULT_NOVELTY_SCORE, 'embedding_failed_for_references'

            similarities = [idea_vector.similarity(v) for v in ref_vectors]
            max_similarity = max(similarities or [0.0])
            novelty_score = self._calculate_novelty_from_similarity(max_similarity)

            return novelty_score, f'max_sim={max_similarity:.3f}'

        except Exception as e:
            logger.error(
                f'[{self.sentinel.component_id}] Novelty calculation failed for idea {idea.idea_id}: {e}',
                exc_info=True
            )
            return DEFAULT_NOVELTY_SCORE, f'novelty_calculation_error: {e}'

    def _calculate_novelty_from_similarity(self, max_similarity: float) -> float:
        """
        Converts a max similarity score (0.0 to 1.0) to a novelty score (1.0 to 10.0).
        - Max similarity of 1.0 (identical) -> Novelty of 1.0
        - Max similarity of 0.0 (completely different) -> Novelty of 10.0
        """
        novelty = (1 - max_similarity) * 9 + 1
        return max(1.0, min(10.0, novelty))

    async def calculate_batch_novelty(
        self,
        ideas: Sequence[Idea],
        refs: Sequence[Idea]
    ) -> dict[str, Tuple[float, str]]:
        """Calculates novelty for a batch of ideas against the same reference set."""
        # This implementation remains valid as it correctly awaits the async calculate_novelty.
        # For higher performance, this could be optimized to use asyncio.gather.
        results = {}
        for idea in ideas:
            score, reason = await self.calculate_novelty(idea, refs)
            results[idea.idea_id] = (score, reason)
        return results