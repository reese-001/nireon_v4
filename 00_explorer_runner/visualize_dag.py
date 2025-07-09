#!/usr/bin/env python3
# nireon_v4/00_explorer_runner/visualize_dag.py
"""
NIREON DAG Visualizer
Parses execution logs and creates visual representations of the execution flow.
"""

import argparse
import json
import subprocess
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

# Optional imports for matplotlib visualization
try:
    import matplotlib.pyplot as plt
    import networkx as nx
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib/networkx not available. Install with: pip install matplotlib networkx")

class NIREONDAGVisualizer:
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.graph_data = self._parse_log_file()
        
    def _parse_log_file(self) -> Dict[str, Any]:
        """Parse DAG log file and extract graph structure"""
        nodes = {}
        edges = []
        signals = []
        events = []
        
        with open(self.log_path, 'r') as f:
            for line in f:
                if 'DAG|' in line:
                    try:
                        json_start = line.index('DAG|') + 4
                        json_str = line[json_start:].strip()
                        data = json.loads(json_str)
                        
                        if data['type'] == 'NODE':
                            nodes[data['node_id']] = data
                        elif data['type'] == 'EDGE':
                            edges.append(data)
                        elif data['type'] == 'SIGNAL':
                            signals.append(data)
                        elif data['type'] == 'EVENT':
                            events.append(data)
                    except Exception as e:
                        print(f"Error parsing line: {e}")
                        continue
        
        return {
            'nodes': list(nodes.values()),
            'edges': edges,
            'signals': signals,
            'events': events
        }
    
    def create_networkx_graph(self) -> 'nx.DiGraph':
        """Create a NetworkX directed graph from the parsed data"""
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("NetworkX not available")
            
        G = nx.DiGraph()
        
        # Add nodes with attributes
        for node in self.graph_data['nodes']:
            G.add_node(
                node['node_id'],
                node_type=node['node_type'],
                **node.get('metadata', {})
            )
        
        # Add edges with attributes
        for edge in self.graph_data['edges']:
            G.add_edge(
                edge['source'],
                edge['target'],
                edge_type=edge['edge_type'],
                **edge.get('metadata', {})
            )
        
        return G
    
    def visualize_with_matplotlib(self, output_path: Path):
        """Create a visual representation using matplotlib"""
        if not MATPLOTLIB_AVAILABLE:
            print("Skipping matplotlib visualization - libraries not available")
            return
            
        G = self.create_networkx_graph()
        
        # Create figure
        plt.figure(figsize=(20, 16))
        
        # Define node colors by type
        node_colors = {
            'SEED_IDEA': '#90EE90',
            'GENERATED_IDEA': '#ADD8E6',
            'PROTO_TASK': '#FFA500',
            'COMPONENT': '#FFB6C1',
            'SYSTEM': '#D8BFD8',
            'SEED_EXECUTION': '#F0E68C'
        }
        
        # Get node colors
        colors = []
        for node in G.nodes():
            node_type = G.nodes[node].get('node_type', 'UNKNOWN')
            colors.append(node_colors.get(node_type, '#CCCCCC'))
        
        # Calculate layout
        pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
        
        # Draw the graph
        nx.draw_networkx_nodes(
            G, pos,
            node_color=colors,
            node_size=3000,
            alpha=0.9
        )
        
        # Draw edges with different styles for different types
        edge_types = set(nx.get_edge_attributes(G, 'edge_type').values())
        edge_styles = {
            'GENERATES': 'solid',
            'TRIGGERS_PROTO': 'dashed',
            'ASSESSMENT': 'dotted'
        }
        
        for edge_type in edge_types:
            edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('edge_type') == edge_type]
            nx.draw_networkx_edges(
                G, pos,
                edgelist=edges,
                style=edge_styles.get(edge_type, 'solid'),
                edge_color='gray',
                arrows=True,
                arrowsize=20,
                width=2
            )
        
        # Add labels
        labels = {}
        for node in G.nodes():
            # Shorten node IDs for display
            if len(node) > 8 and '-' in node:
                labels[node] = f"{node[:8]}..."
            else:
                labels[node] = node
        
        nx.draw_networkx_labels(
            G, pos,
            labels,
            font_size=8,
            font_weight='bold'
        )
        
        # Add edge labels
        edge_labels = nx.get_edge_attributes(G, 'edge_type')
        nx.draw_networkx_edge_labels(
            G, pos,
            edge_labels,
            font_size=6
        )
        
        # Add title and legend
        plt.title(f"NIREON Execution DAG - {self.log_path.stem}", fontsize=16, fontweight='bold')
        
        # Create legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=color, label=node_type.replace('_', ' ').title())
            for node_type, color in node_colors.items()
        ]
        plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
        
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Matplotlib visualization saved to: {output_path}")
    
    def generate_graphviz_dot(self, output_path: Path):
        """Generate a Graphviz DOT file"""
        dot_lines = ["digraph NIREONExecution {"]
        dot_lines.append("    rankdir=TB;")
        dot_lines.append("    node [shape=box, style=rounded, fontsize=10];")
        dot_lines.append("    edge [fontsize=8];")
        
        # Node styles
        node_styles = {
            'SEED_IDEA': 'fillcolor="#90EE90", style="rounded,filled"',
            'GENERATED_IDEA': 'fillcolor="#ADD8E6", style="rounded,filled"',
            'PROTO_TASK': 'fillcolor="#FFA500", style="rounded,filled"',
            'COMPONENT': 'fillcolor="#FFB6C1", style="rounded,filled"',
            'SYSTEM': 'fillcolor="#D8BFD8", style="rounded,filled"',
            'SEED_EXECUTION': 'fillcolor="#F0E68C", style="rounded,filled"'
        }
        
        # Add nodes
        for node in self.graph_data['nodes']:
            node_id = node['node_id']
            node_type = node['node_type']
            
            # Create label
            label_parts = [f"{node_id[:12]}..." if len(node_id) > 12 else node_id]
            label_parts.append(f"Type: {node_type}")
            
            # Add metadata to label
            metadata = node.get('metadata', {})
            if 'text_preview' in metadata:
                text = metadata['text_preview'][:30] + "..." if len(metadata['text_preview']) > 30 else metadata['text_preview']
                label_parts.append(f"Text: {text}")
            if 'trust_score' in metadata:
                label_parts.append(f"Trust: {metadata['trust_score']:.2f}")
            
            label = "\\n".join(label_parts)
            style = node_styles.get(node_type, 'fillcolor="#CCCCCC", style="rounded,filled"')
            
            dot_lines.append(f'    "{node_id}" [label="{label}", {style}];')
        
        # Add edges
        for edge in self.graph_data['edges']:
            source = edge['source']
            target = edge['target']
            edge_type = edge['edge_type']
            
            # Edge attributes
            edge_attrs = [f'label="{edge_type}"']
            
            if edge_type == 'TRIGGERS_PROTO':
                edge_attrs.append('style=dashed')
                edge_attrs.append('color=red')
            elif edge_type == 'GENERATES':
                edge_attrs.append('color=blue')
            
            metadata = edge.get('metadata', {})
            if 'depth' in metadata:
                edge_attrs.append(f'xlabel="depth={metadata["depth"]}"')
            
            attrs_str = ', '.join(edge_attrs)
            dot_lines.append(f'    "{source}" -> "{target}" [{attrs_str}];')
        
        # Add timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        dot_lines.append(f'    label="Generated: {timestamp}";')
        dot_lines.append(f'    labelloc="b";')
        
        dot_lines.append("}")
        
        # Write DOT file
        with open(output_path, 'w') as f:
            f.write('\n'.join(dot_lines))
        
        print(f"Graphviz DOT file saved to: {output_path}")
        
        # Try to render with Graphviz if available
        try:
            png_path = output_path.with_suffix('.png')
            subprocess.run(['dot', '-Tpng', str(output_path), '-o', str(png_path)], check=True)
            print(f"Graphviz PNG rendered to: {png_path}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Graphviz not found. Install it to automatically render the DOT file.")
            print("On Ubuntu/Debian: sudo apt-get install graphviz")
            print("On macOS: brew install graphviz")
            print("On Windows: Download from https://graphviz.org/download/")
    
    def generate_mermaid_diagram(self, output_path: Path):
        """Generate a Mermaid diagram"""
        lines = ["graph TD"]
        
        # Add nodes
        for node in self.graph_data['nodes']:
            node_id = node['node_id'].replace('-', '_')
            node_type = node['node_type']
            
            # Create label
            label = f"{node['node_id'][:12]}...<br/>{node_type}"
            
            # Style based on type
            if node_type == 'SEED_IDEA':
                lines.append(f"    {node_id}[{label}]:::seed")
            elif node_type == 'GENERATED_IDEA':
                lines.append(f"    {node_id}[{label}]:::idea")
            elif node_type == 'PROTO_TASK':
                lines.append(f"    {node_id}[{label}]:::proto")
            elif node_type == 'COMPONENT':
                lines.append(f"    {node_id}[{label}]:::component")
            else:
                lines.append(f"    {node_id}[{label}]")
        
        # Add edges
        for edge in self.graph_data['edges']:
            source = edge['source'].replace('-', '_')
            target = edge['target'].replace('-', '_')
            edge_type = edge['edge_type']
            
            if edge_type == 'TRIGGERS_PROTO':
                lines.append(f"    {source} -.->|{edge_type}| {target}")
            else:
                lines.append(f"    {source} -->|{edge_type}| {target}")
        
        # Add styles
        lines.extend([
            "",
            "    %% Styles",
            "    classDef seed fill:#90EE90,stroke:#333,stroke-width:2px;",
            "    classDef idea fill:#ADD8E6,stroke:#333,stroke-width:2px;",
            "    classDef proto fill:#FFA500,stroke:#333,stroke-width:2px;",
            "    classDef component fill:#FFB6C1,stroke:#333,stroke-width:2px;"
        ])
        
        # Write file
        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))
        
        print(f"Mermaid diagram saved to: {output_path}")
        print("To visualize, paste the content into: https://mermaid.live/")
    
    def print_summary(self):
        """Print a summary of the execution graph"""
        print("\n" + "="*60)
        print("NIREON EXECUTION DAG SUMMARY")
        print("="*60)
        
        # Node statistics
        node_types = {}
        for node in self.graph_data['nodes']:
            node_type = node['node_type']
            node_types[node_type] = node_types.get(node_type, 0) + 1
        
        print("\nNode Statistics:")
        for node_type, count in sorted(node_types.items()):
            print(f"  {node_type}: {count}")
        print(f"  TOTAL: {len(self.graph_data['nodes'])}")
        
        # Edge statistics
        edge_types = {}
        for edge in self.graph_data['edges']:
            edge_type = edge['edge_type']
            edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
        
        print("\nEdge Statistics:")
        for edge_type, count in sorted(edge_types.items()):
            print(f"  {edge_type}: {count}")
        print(f"  TOTAL: {len(self.graph_data['edges'])}")
        
        # Signal statistics
        signal_types = {}
        for signal in self.graph_data['signals']:
            signal_type = signal['signal_type']
            signal_types[signal_type] = signal_types.get(signal_type, 0) + 1
        
        print("\nSignal Statistics:")
        for signal_type, count in sorted(signal_types.items()):
            print(f"  {signal_type}: {count}")
        print(f"  TOTAL: {len(self.graph_data['signals'])}")
        
        # Event statistics
        event_types = {}
        for event in self.graph_data['events']:
            event_type = event['event_type']
            event_types[event_type] = event_types.get(event_type, 0) + 1
        
        print("\nEvent Statistics:")
        for event_type, count in sorted(event_types.items()):
            print(f"  {event_type}: {count}")
        print(f"  TOTAL: {len(self.graph_data['events'])}")
        
        # Find high trust ideas
        high_trust_ideas = []
        for event in self.graph_data['events']:
            if event['event_type'] == 'TRUST_ASSESSMENT':
                trust_score = event.get('details', {}).get('trust_score', 0)
                if trust_score > 6.0:
                    high_trust_ideas.append({
                        'node_id': event['node_id'],
                        'trust_score': trust_score
                    })
        
        if high_trust_ideas:
            print(f"\nHigh Trust Ideas (score > 6.0): {len(high_trust_ideas)}")
            for idea in sorted(high_trust_ideas, key=lambda x: x['trust_score'], reverse=True)[:5]:
                print(f"  {idea['node_id'][:12]}...: {idea['trust_score']:.2f}")
        
        print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize NIREON execution DAGs from log files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all visualization formats
  python visualize_dag.py dag_logs/dag_log_20240628_120000.log
  
  # Generate only Graphviz DOT file
  python visualize_dag.py dag_logs/dag_log_20240628_120000.log --format graphviz
  
  # Generate with summary statistics
  python visualize_dag.py dag_logs/dag_log_20240628_120000.log --summary
  
  # Custom output directory
  python visualize_dag.py dag_logs/dag_log_20240628_120000.log --output-dir ./my_visualizations
        """
    )
    
    parser.add_argument(
        'log_file',
        type=Path,
        help='Path to the DAG log file'
    )
    
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('./dag_visualizations'),
        help='Output directory for visualizations (default: ./dag_visualizations)'
    )
    
    parser.add_argument(
        '--format',
        choices=['all', 'matplotlib', 'graphviz', 'mermaid'],
        default='all',
        help='Output format(s) to generate (default: all)'
    )
    
    parser.add_argument(
        '--summary',
        action='store_true',
        help='Print execution summary'
    )
    
    args = parser.parse_args()
    
    # Validate input
    if not args.log_file.exists():
        print(f"Error: Log file not found: {args.log_file}")
        return 1
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create visualizer
    visualizer = NIREONDAGVisualizer(args.log_file)
    
    # Print summary if requested
    if args.summary:
        visualizer.print_summary()
    
    # Generate visualizations
    base_name = args.log_file.stem
    
    if args.format in ['all', 'matplotlib']:
        try:
            output_path = args.output_dir / f"{base_name}_matplotlib.png"
            visualizer.visualize_with_matplotlib(output_path)
        except ImportError:
            print("Warning: matplotlib not available. Skipping matplotlib visualization.")
    
    if args.format in ['all', 'graphviz']:
        output_path = args.output_dir / f"{base_name}.dot"
        visualizer.generate_graphviz_dot(output_path)
    
    if args.format in ['all', 'mermaid']:
        output_path = args.output_dir / f"{base_name}.mmd"
        visualizer.generate_mermaid_diagram(output_path)
    
    print("\nVisualization complete!")
    print(f"Output directory: {args.output_dir}")
    return 0


if __name__ == '__main__':
    exit(main())

    