# nireon_v4/components/mechanisms/catalyst/prompt_builder.py
import logging
from typing import Dict, Any, Optional, List, Union, Deque
from collections import deque
from domain.ideas.idea import Idea  # <-- Import Idea instead of Agent
from .types import AntiConstraints

logger = logging.getLogger(__name__)

def build_prompt_for_idea(
    idea: Idea, # <-- Change parameter from agent: Agent to idea: Idea
    domain: str,
    blend: float,
    objective: str,
    prompt_template: Optional[str] = None,
    active_anti_constraints: Optional[List[str]] = None,
    anti_constraints_threshold: float = 0.15,
    recent_semantic_distances: Optional[Deque[float]] = None
) -> str:
    original_text = idea.text # <-- Use idea.text directly
    agent_id = idea.metadata.get('agent_id', 'unknown_agent') # <-- Get agent_id from metadata if needed

    apply_anti_constraints = False
    constraints_text = ''

    if active_anti_constraints and recent_semantic_distances is not None and (len(active_anti_constraints) > 0):
        semantic_distances = list(recent_semantic_distances)
        if semantic_distances:
            try:
                avg_sem_dist = sum(semantic_distances) / len(semantic_distances)
                if avg_sem_dist < anti_constraints_threshold:
                    apply_anti_constraints = True
                    constraints_text = '\nIMPORTANT CONSTRAINTS:\n' + '\n'.join([f'- {constraint}' for constraint in active_anti_constraints])
                    logger.debug(f'Applying anti-constraints for idea {idea.idea_id}. Avg sem dist: {avg_sem_dist:.3f}')
                else:
                    logger.debug(f'Not applying anti-constraints for idea {idea.idea_id}. Avg sem dist: {avg_sem_dist:.3f}')
            except Exception as e:
                logger.error(f'Error calculating average semantic distance for anti-constraints for idea {idea.idea_id}: {e}', exc_info=True)

    if prompt_template:
        try:
            # Update format keys to reflect the change from Agent object to Idea object
            prompt = prompt_template.format(
                original=original_text,
                domain=domain,
                blend=blend,
                objective=objective,
                constraints=constraints_text if apply_anti_constraints else '',
                agent_id=agent_id,
                role=idea.metadata.get('role', 'generator'), # <-- Get role from metadata
                idea_id=idea.idea_id
            )
            if apply_anti_constraints and '{constraints}' not in prompt_template:
                prompt += constraints_text
                logger.debug(f"Appended anti-constraints to prompt for idea {idea.idea_id} as template didn't include {{constraints}}.")
            return prompt
        except Exception as e:
            logger.error(f'Error formatting prompt template for idea {idea.idea_id}: {e}. Falling back to default prompt.', exc_info=True)

    default_prompt = f"You are blending an existing idea with knowledge from the domain '{domain}' at strength {blend:.2f}.\nOverall objective: {objective}\n\nOriginal Idea:\n'''{original_text}'''\n\nRespond ONLY with the new, blended idea text."
    if apply_anti_constraints:
        default_prompt += constraints_text

    return default_prompt