import subprocess
import sys
import os
from pathlib import Path

def run_tach():
    """Run Tach architectural boundary validation from nested folder."""
    # Go up two levels from current file (i.e., from /validation/tach/ to project root)
    script_dir = Path(__file__).resolve().parent
    root_dir = script_dir.parent.parent  # Two levels up

    config_path = root_dir / "tach.toml"
    
    if not config_path.exists():
        print(f"❌ tach.toml not found at: {config_path}")
        sys.exit(1)
    
    try:
        print(f"🔍 Running Tach from root directory: {root_dir}")
        result = subprocess.run(
            ["tach", "check"],
            check=True,
            cwd=str(root_dir),  # Run from root
            capture_output=True,
            text=True
        )
        print("✅ Tach checks passed: No layer violations.")
        if result.stdout:
            print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("❌ Tach found architectural violations:")
        if e.stdout:
            print(e.stdout)
        if e.stderr:
            print(e.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print("❌ Tach not found. Install with: pip install tach")
        sys.exit(1)

if __name__ == "__main__":
    run_tach()
