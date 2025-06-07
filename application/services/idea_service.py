# application/services/idea_service.py
# ---------------------------------------------------------------------------
#  Implements the IdeaServicePort declared in domain.  Depends only on
#  domain-layer abstractions (ports, models, context) and infrastructure
#  injected at runtime via the component registry.
# ---------------------------------------------------------------------------
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import List

from domain.context import NireonExecutionContext
from domain.ideas.idea import Idea
from domain.ports.event_bus_port import EventBusPort
from domain.ports.idea_repository_port import IdeaRepositoryPort
from domain.ports.idea_service_port import IdeaServicePort

logger = logging.getLogger(__name__)


class IdeaService(IdeaServicePort):
    """
    Concrete application-layer implementation of IdeaServicePort.
    """

    def __init__(
        self,
        repository: IdeaRepositoryPort,
        event_bus: EventBusPort | None = None,
    ):
        self.repository = repository
        self.event_bus = event_bus
        logger.info("IdeaService initialised")

    # --------------------------------------------------------------------- #
    #  Idea creation / persistence
    # --------------------------------------------------------------------- #
    def create_idea(
        self,
        *,
        text: str,
        parent_id: str | None = None,
        context: NireonExecutionContext | None = None,
    ) -> Idea:
        idea = Idea.create(text, parent_id)

        # Optional hash (for backward compat with V3)
        if hasattr(idea, "compute_hash"):
            try:
                idea.metadata["hash"] = idea.compute_hash()
            except Exception:
                logger.debug("compute_hash() failed for new idea", exc_info=True)

        self.repository.save(idea)

        # Maintain DAG linkage
        if parent_id and hasattr(self.repository, "add_child_relationship"):
            try:
                self.repository.add_child_relationship(parent_id, idea.idea_id)
            except Exception:
                logger.debug(
                    "Repository lacks add_child_relationship() or it failed",
                    exc_info=True,
                )

        # Fire event-bus notification
        if self.event_bus:
            self.event_bus.publish(
                "idea_created",
                {
                    "idea_id": idea.idea_id,
                    "text": idea.text[:50] + "..."
                    if len(idea.text) > 50
                    else idea.text,
                    "parent_ids": idea.parent_ids,
                    "run_id": context.run_id if context else None,
                },
            )

        logger.info("Created Idea %s", idea.idea_id)
        return idea

    def save_idea(
        self, idea: Idea, context: NireonExecutionContext | None = None
    ) -> None:
        if not isinstance(idea, Idea):
            logger.error("Attempted to save non-Idea object: %s", type(idea))
            raise ValueError("Can only save Idea objects.")
        self.repository.save(idea)

        if self.event_bus:
            self.event_bus.publish(
                "idea_persisted",
                {
                    "idea_id": idea.idea_id,
                    "text_snippet": idea.text[:50] + "..."
                    if len(idea.text) > 50
                    else idea.text,
                    "method": getattr(idea, "method", None),
                    "run_id": context.run_id if context else None,
                },
            )
        logger.info("Saved Idea %s", idea.idea_id)

    # --------------------------------------------------------------------- #
    #  Retrieval helpers
    # --------------------------------------------------------------------- #
    def get_idea(self, idea_id: str) -> Idea | None:  # noqa: D401
        return self.repository.get_by_id(idea_id)

    def get_all_ideas(self) -> List[Idea]:
        return self.repository.get_all()

    # --------------------------------------------------------------------- #
    #  World-fact linkage
    # --------------------------------------------------------------------- #
    def add_world_fact(
        self,
        *,
        idea_id: str,
        fact_id: str,
        context: NireonExecutionContext | None = None,
    ) -> bool:
        if not hasattr(self.repository, "add_world_fact"):
            logger.debug("Repository lacks add_world_fact(); skipping")
            return False

        success: bool = self.repository.add_world_fact(idea_id, fact_id)
        if success and self.event_bus:
            self.event_bus.publish(
                "world_fact_added",
                {
                    "idea_id": idea_id,
                    "fact_id": fact_id,
                    "run_id": context.run_id if context else None,
                },
            )
        return success

    # --------------------------------------------------------------------- #
    #  Child retrieval fallback
    # --------------------------------------------------------------------- #
    def get_children(self, idea_id: str) -> List[Idea]:
        if hasattr(self.repository, "get_by_parent_id"):
            return self.repository.get_by_parent_id(idea_id)
        # Fallback if repository lacks specialised method
        return [i for i in self.repository.get_all() if idea_id in i.parent_ids]
