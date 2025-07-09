#!/bin/bash
# Example usage of DAG logging system

echo "Running NIREON Explorer with DAG logging..."
python run_explorer.py --seeds retail_survival

# Find the latest log file
LATEST_LOG=$(ls -t dag_logs/dag_log_*.log 2>/dev/null | head -1)

if [ -n "$LATEST_LOG" ]; then
    echo
    echo "Generating visualizations from: $LATEST_LOG"
    python visualize_dag.py "$LATEST_LOG" --summary
    
    echo
    echo "Visualizations created in: ./dag_visualizations/"
    echo
    echo "To view:"
    echo "  - Graphviz: Open .dot file or rendered .png"
    echo "  - Mermaid: Copy .mmd content to https://mermaid.live/"
    echo "  - Matplotlib: Open .png file"
else
    echo "No DAG log files found!"
fi
