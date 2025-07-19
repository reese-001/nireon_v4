"""quantifier_prompts.py
========================
Prompt templates used by *QuantifierAgent* to steer the LLM.

Provides three static builders:
* :py:meth:`comprehensive_analysis` - main system prompt asking the LLM to
  choose between *quantification* and *formalization* and to return a fully
  structured YAML block.
* :py:meth:`mermaid_generation`  - quick template to request a Mermaid diagram.
* :py:meth:`viability_quick_check` - lightweight YES/NO gate used for cheap
  pre-filtering before running the heavier analysis chain.

Revisions vs previous version
-----------------------------
* Added **confidence** field to the YAML spec and *viability quick-check* to
  align with :class:`ResponseParser.extract_yes_no_decision` heuristics.
* Clarified output rules (no extra prose, raw Mermaid code only, etc.).
* Enriched docstrings and tightened type hints for better IDE support.
"""

from __future__ import annotations

from typing import Any, Dict, List

__all__: List[str] = ["QuantifierPrompts"]


class QuantifierPrompts:  # pylint: disable=too-few-public-methods
    """Factory for LLM prompt strings used by Quantifier-related workflows."""

    # ------------------------------------------------------------------
    # Comprehensive analysis template
    # ------------------------------------------------------------------
    @staticmethod
    def comprehensive_analysis(idea_text: str, available_libs: Dict[str, List[str]]) -> str:  # noqa: D401,E501
        """Return the *full* analysis prompt for classification & work-order gen."""
        lib_sections: List[str] = []
        for category, libs in available_libs.items():
            label = category.replace("_", " ").title()
            lib_sections.append(f"{label}: {', '.join(libs)}")
        libraries_text = "\n".join(lib_sections)

        return (
            f"You are a technical analyst. Your task is to analyse a concept and "
            f"decide whether it requires a **quantification** task (data visualisation / "
            f"simulation) **or** a **formalisation** task (symbolic maths / proof).\n\n"
            f"**Concept:** \"{idea_text}\"\n\n"
            f"**Available Tools & Dialects:**\n"
            f"- **Quantification (`eidos: math` or `graph`):** For visualising data, "
            f"running simulations, or analysing relationships. Uses libraries like:\n"
            f"{libraries_text}\n"
            f"- **Formalisation (`eidos: math` via PrincipiaAgent):** For symbolic maths, "
            f"solving equations, checking proofs. Uses **SymPy** for exact computation. "
            f"Keywords: *prove*, *show that*, *for all n*, *factorise*, etc.\n\n"
            f"---\n"
            f"**Your Complete Analysis (YAML format):**\n\n"
            f"1. **analysis_type:** Choose **one** - `quantification` **or** `formalization`.\n"
            f"2. **viability:** `true` / `false` - is the request doable with the tools?\n"
            f"3. **reasoning:** Short explanation (max 40 words).\n"
            f"4. **confidence:** Float 0-1 representing certainty.\n"
            f"5. **implementation_request:**\n"
            f"   • If **quantification** - *natural-language* instructions for **ProtoGenerator**.\n"
            f"     Include libraries, logic, data assumptions, expected outputs (e.g. `plot.png`).\n"
            f"   • If **formalization** - JSON object for **MathQuerySignal** with:\n"
            f"       - `natural_language_query` (original).\n"
            f"       - `expression` (SymPy).\n"
            f"       - `operations` (list of steps).\n\n"
            f"**CRITICAL RULES:**\n"
            f"- Output **only** the YAML block. No extra commentary.\n"
            f"- If `viability: false`, give a brief `implementation_request` note.\n"
            f"- For *formalisation* requests, `implementation_request` **must** be valid JSON\n"
            f"  (escape as needed).\n\n"
            f"**YAML Output:**\n"
            f"```yaml\n"
            f"analysis_type: <quantification_or_formalization>\n"
            f"viability: <true_or_false>\n"
            f"reasoning: <brief_explanation>\n"
            f"confidence: <0.0_to_1.0>\n"
            f"implementation_request: |\n"
            f"  <detailed_proto_request_OR_math_query_json>\n"
            f"```"
        )

    # ------------------------------------------------------------------
    # Mermaid generation helper
    # ------------------------------------------------------------------
    @staticmethod
    def mermaid_generation(idea_text: str, viz_type: str) -> str:  # noqa: D401,E501
        """Prompt to request a Mermaid diagram for *idea_text*."""
        return (
            f"Create a Mermaid diagram for this concept: \"{idea_text}\"\n\n"
            f"Suggested diagram type: {viz_type}\n\n"
            f"Available Mermaid diagram types:\n"
            f"- flowchart: Process flows, decision trees, workflows\n"
            f"- graph: Network relationships, connections, systems\n"
            f"- classDiagram: Object relationships, hierarchies, structures\n"
            f"- sequenceDiagram: Time-based interactions, processes\n"
            f"- gantt: Project timelines, schedules, planning\n"
            f"- gitgraph: Version control, branching processes\n"
            f"- mindmap: Concept relationships, brainstorming\n"
            f"- timeline: Historical events, progression, evolution\n\n"
            f"Generate **valid Mermaid** syntax that best represents this concept.\n"
            f"Make it comprehensive and insightful.\n\n"
            f"Output **only** the raw Mermaid code:"
        )

    # ------------------------------------------------------------------
    # Cheap viability gate
    # ------------------------------------------------------------------
    @staticmethod
    def viability_quick_check(idea_text: str) -> str:  # noqa: D401,E501
        """Return a minimal YES/NO prompt for cost-effective gating."""
        return (
            "Can this concept be meaningfully visualised, analysed, or modelled quantitatively?\n\n"
            f"Concept: \"{idea_text}\"\n\n"
            "Consider:\n"
            "- Can it be represented with charts, graphs, or diagrams?\n"
            "- Are there measurable aspects or relationships?\n"
            "- Could it benefit from data analysis or modelling?\n"
            "- Would visualisation help understanding or exploration?\n\n"
            "Respond **exactly** with `YES (confidence: X.X)` **or** `NO (confidence: X.X)` "
            "on the first line, where *X.X* is a float between 0.0 and 1.0, followed by a "
            "brief explanation.\n\n"
            "Response:"
        )
