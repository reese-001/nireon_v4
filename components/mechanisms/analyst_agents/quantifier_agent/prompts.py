"""
Prompt templates for the QuantifierAgent LLM interactions.
"""

from typing import Dict, Any

class QuantifierPrompts:
    """Collection of prompt templates for QuantifierAgent."""
    
    @staticmethod
    def comprehensive_analysis(idea_text: str, available_libs: Dict[str, Any]) -> str:
        """Single comprehensive prompt for complete analysis."""
        
        lib_descriptions = []
        for category, libs in available_libs.items():
            category_name = category.replace('_', ' ').title()
            lib_list = ', '.join(libs)
            lib_descriptions.append(f"{category_name}: {lib_list}")
        
        libraries_text = '\n'.join(lib_descriptions)
        
        return f"""You are a visualization expert. Analyze this concept and create a complete implementation plan.

Concept: "{idea_text}"

Available tools:
{libraries_text}

Alternative: Mermaid diagrams (for process flows, relationships, hierarchies)

Complete this analysis:

1. VIABILITY: Can this concept be visualized or analyzed quantitatively? (YES/NO and brief reason)

2. APPROACH: If YES, what's the best approach?
   - Traditional charts (matplotlib/seaborn): For business metrics, comparisons, trends
   - Interactive visualization (plotly): For exploratory data, multi-dimensional views
   - Network analysis (networkx): For relationships, connections, systems
   - Process flow (Mermaid): For workflows, decision trees, organizational structures
   - Text analysis (wordcloud): For content analysis, sentiment, themes
   - Statistical modeling (scikit-learn + visualization): For predictions, clustering, classification

3. IMPLEMENTATION: If viable, write a complete natural language request for the ProtoGenerator that includes:
   - Specific libraries to use from the available set
   - Function name and parameters
   - Computational logic and steps
   - Expected outputs and file names
   - Realistic data assumptions or sample data to use
   - Critical plotting requirements (e.g., numeric positioning for bar charts)

CRITICAL RULES FOR PLOTTING:
- For bar charts with string labels, ALWAYS use numeric positions (np.arange) and plt.xticks
- Include plt.tight_layout() before saving
- Save with descriptive filenames

If not viable for quantitative analysis, explain why and stop here.

Your complete analysis:"""

    @staticmethod
    def mermaid_generation(idea_text: str, viz_type: str) -> str:
        """Generate Mermaid diagram syntax."""
        
        return f"""Create a Mermaid diagram for this concept: "{idea_text}"

Suggested diagram type: {viz_type}

Available Mermaid diagram types:
- flowchart: Process flows, decision trees, workflows
- graph: Network relationships, connections, systems
- classDiagram: Object relationships, hierarchies, structures
- sequenceDiagram: Time-based interactions, processes
- gantt: Project timelines, schedules, planning
- gitgraph: Version control, branching processes
- mindmap: Concept relationships, brainstorming
- timeline: Historical events, progression, evolution

Generate valid Mermaid syntax that best represents this concept.
Make it comprehensive and insightful.

Output only the raw Mermaid code:"""

    @staticmethod
    def viability_quick_check(idea_text: str) -> str:
        """Quick viability assessment for filtering."""
        
        return f"""Can this concept be meaningfully visualized, analyzed, or modeled quantitatively?

Concept: "{idea_text}"

Consider:
- Can it be represented with charts, graphs, or diagrams?
- Are there measurable aspects or relationships?
- Could it benefit from data analysis or modeling?
- Would visualization help understand or explore it?

Respond with exactly "YES" or "NO" on the first line, followed by a brief explanation.

Response:"""