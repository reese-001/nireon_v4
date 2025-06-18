# nireon_v4\components\mechanisms\catalyst\prompt_builder.py
import logging
import textwrap
from typing import Deque, List, Optional
from domain.ideas.idea import Idea

logger = logging.getLogger(__name__)

DEFAULT_PROMPT_TEMPLATE = """
Synthesize a novel, creative idea by blending the following 'Seed Idea' with concepts from the 'Inspiration Domain'.

**Seed Idea:**
{seed_idea}

**Inspiration Domain:**
{domain}

**Objective:**
{objective}

**Instructions:**
1.  Understand the core concept of the 'Seed Idea'.
2.  Draw inspiration from the 'Inspiration Domain'.
3.  Create a new, hybrid idea that combines elements from both.
4.  The new idea should be a direct, coherent statement, not a commentary on the process.
{anti_constraint_block}
**New Synthesized Idea:**
"""

ANTI_CONSTRAINT_BLOCK_TEMPLATE = """
**Constraint:** The new idea MUST avoid the following themes:
- {anti_constraints_list}
"""

def build_prompt_for_idea(
    idea: Idea,
    domain: str,
    blend: float,
    objective: str,
    prompt_template: Optional[str],
    active_anti_constraints: List[str],
    anti_constraints_threshold: float,
    recent_semantic_distances: Deque[float]
) -> str:
    """Builds the final prompt for the LLM call."""
    
    template_to_use = prompt_template or DEFAULT_PROMPT_TEMPLATE
    
    anti_constraint_block = ""
    avg_distance = sum(recent_semantic_distances) / len(recent_semantic_distances) if recent_semantic_distances else 1.0
    
    if active_anti_constraints and avg_distance < anti_constraints_threshold:
        logger.debug(f'Activating anti-constraints block for prompt (avg_distance={avg_distance:.3f} < threshold={anti_constraints_threshold:.3f})')
        constraints_list_str = "\n- ".join(active_anti_constraints)
        anti_constraint_block = ANTI_CONSTRAINT_BLOCK_TEMPLATE.format(anti_constraints_list=constraints_list_str)

    prompt = template_to_use.format(
        seed_idea=idea.text,
        domain=domain,
        objective=objective,
        blend_strength=f"{blend:.2f}",
        anti_constraint_block=anti_constraint_block
    ).strip()

    return textwrap.dedent(prompt)