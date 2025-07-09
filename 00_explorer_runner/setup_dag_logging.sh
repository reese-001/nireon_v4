#!/bin/bash
# nireon_v4/00_explorer_runner/setup_dag_logging.sh
# Setup script for NIREON DAG Logging System

set -e

echo "==================================="
echo "NIREON DAG Logging Setup"
echo "==================================="

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Detect OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
    OS="windows"
else
    OS="unknown"
fi

echo "Detected OS: $OS"
echo

# Create required directories
echo "Creating directories..."
mkdir -p dag_logs
mkdir -p dag_visualizations
echo "✓ Directories created"
echo

# Check Python dependencies
# echo "Checking Python dependencies..."
# python -m pip show matplotlib >/dev/null 2>&1 || {
#     echo "Installing matplotlib and networkx for visualization..."
#     pip install matplotlib networkx
# }
# echo "✓ Python dependencies checked"
echo

# Check and install Graphviz
echo "Checking Graphviz installation..."
if ! command_exists dot; then
    echo "Graphviz not found. Installing..."
    
    case $OS in
        linux)
            if command_exists apt-get; then
                sudo apt-get update
                sudo apt-get install -y graphviz
            elif command_exists yum; then
                sudo yum install -y graphviz
            else
                echo "⚠️  Please install Graphviz manually: https://graphviz.org/download/"
            fi
            ;;
        macos)
            if command_exists brew; then
                brew install graphviz
            else
                echo "⚠️  Please install Homebrew first or install Graphviz manually"
            fi
            ;;
        windows)
            echo "⚠️  Please download Graphviz from: https://graphviz.org/download/"
            echo "   Make sure to add it to your PATH"
            ;;
        *)
            echo "⚠️  Unknown OS. Please install Graphviz manually"
            ;;
    esac
else
    echo "✓ Graphviz is installed"
fi
echo

# Make visualization script executable
if [ -f "visualize_dag.py" ]; then
    chmod +x visualize_dag.py
    echo "✓ Made visualize_dag.py executable"
fi

# Create example usage script
cat > example_dag_usage.sh << 'EOF'
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
EOF

chmod +x example_dag_usage.sh
echo "✓ Created example_dag_usage.sh"
echo

# Print summary
echo "==================================="
echo "Setup Complete!"
echo "==================================="
echo
echo "To get started:"
echo "  1. Run: ./example_dag_usage.sh"
echo "  2. Or manually:"
echo "     - python run_explorer.py"
echo "     - python visualize_dag.py dag_logs/dag_log_*.log"
echo
echo "Output directories:"
echo "  - Logs: ./dag_logs/"
echo "  - Visualizations: ./dag_visualizations/"
echo
if ! command_exists dot; then
    echo "⚠️  Note: Graphviz is not installed. DOT files will be created but not rendered to images."
fi