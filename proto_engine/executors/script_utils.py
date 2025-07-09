from __future__ import annotations
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


def generate_execution_script(proto_data: Dict[str, Any]) -> str:
    code_block = proto_data.get('code', "def main():\n    return 'No code provided'")
    function_name = proto_data.get('function_name', 'main')
    requirements = proto_data.get('requirements', [])

    # The json.dumps ensures that the multiline string is correctly embedded in the script.
    safe_code_block = json.dumps(code_block)
    safe_requirements = json.dumps(requirements)
    
    # This is a more robust script template.
    script_template = f"""
import sys
import os
import json
import time
import traceback
import subprocess
from pathlib import Path

# Debug information
print("=== EXECUTION SCRIPT STARTING ===", flush=True)
print(f"Python: {{sys.version}}", flush=True)
print(f"Working directory: {{os.getcwd()}}", flush=True)
print(f"Script arguments: {{sys.argv}}", flush=True)

def install_requirements(reqs):
    \"\"\"Install required packages using pip.\"\"\"
    print(f"=== INSTALLING REQUIREMENTS: {{reqs}} ===", flush=True)
    
    if not reqs:
        print("No requirements to install.", flush=True)
        return True
    
    # Check if pip is available
    try:
        pip_check = subprocess.run(
            [sys.executable, "-m", "pip", "--version"],
            capture_output=True,
            text=True
        )
        print(f"Pip version: {{pip_check.stdout.strip()}}", flush=True)
        if pip_check.returncode != 0:
            print(f"ERROR: pip not available: {{pip_check.stderr}}", file=sys.stderr, flush=True)
            return False
    except Exception as e:
        print(f"ERROR: Failed to check pip: {{e}}", file=sys.stderr, flush=True)
        return False
    
    # Write requirements file
    req_file = Path("requirements.txt")
    req_content = "\\n".join(reqs)
    print(f"Writing requirements.txt with content: {{req_content}}", flush=True)
    try:
        req_file.write_text(req_content)
    except Exception as e:
        print(f"ERROR: Failed to write requirements.txt: {{e}}", file=sys.stderr, flush=True)
        return False
    
    # Install requirements
    try:
        print("Running pip install command...", flush=True)
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(req_file)],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout for pip install
        )
        
        if result.stdout:
            print(f"PIP STDOUT:\\n{{result.stdout}}", flush=True)
        if result.stderr:
            print(f"PIP STDERR:\\n{{result.stderr}}", flush=True)
            
        if result.returncode != 0:
            print(f"ERROR: pip install failed with return code {{result.returncode}}", file=sys.stderr, flush=True)
            return False
        else:
            print("Package installation completed successfully.", flush=True)
            return True
            
    except subprocess.TimeoutExpired:
        print("ERROR: pip install timed out after 5 minutes", file=sys.stderr, flush=True)
        return False
    except Exception as e:
        print(f"ERROR: Exception during pip install: {{e}}", file=sys.stderr, flush=True)
        import traceback
        traceback.print_exc()
        return False

# Main execution
if __name__ == "__main__":
    try:
        # Install requirements FIRST, before any user code execution
        requirements = {safe_requirements}
        if requirements:
            install_success = install_requirements(requirements)
            if not install_success:
                raise RuntimeError("Failed to install required packages")
        
        # NOW load and execute user code after requirements are installed
        print("=== EXECUTING USER CODE ===", flush=True)
        user_code = {safe_code_block}
        try:
            exec(user_code, globals())
            print("User code executed successfully", flush=True)
        except Exception as e:
            print(f"Error executing user code: {{e}}", file=sys.stderr, flush=True)
            raise
        
        # Load input data
        if len(sys.argv) > 1:
            inputs_file_path = sys.argv[1]
            print(f"Loading inputs from: {{inputs_file_path}}", flush=True)
            with open(inputs_file_path, 'r', encoding='utf-8') as f:
                inputs = json.load(f)
        else:
            print("No input file provided, using empty inputs", flush=True)
            inputs = {{}}
        
        # Find and validate the entry point function
        entry_point = globals().get('{function_name}')
        if not callable(entry_point):
            available_functions = [name for name, obj in globals().items() if callable(obj) and not name.startswith('_')]
            raise NameError(
                f"Entry point function '{function_name}' not found or not callable. "
                f"Available functions: {{available_functions}}"
            )
        
        print(f"Calling {{entry_point.__name__}} with inputs: {{inputs}}", flush=True)
        
        # Execute the main function
        start_time = time.time()
        result = entry_point(**inputs)
        execution_time = time.time() - start_time
        
        print(f"Function executed in {{execution_time:.3f}} seconds", flush=True)
        
        # Collect artifacts
        ignore_list = ['execute.py', 'inputs.json', 'requirements.txt', '__pycache__']
        artifacts = []
        for p in Path(".").iterdir():
            if p.is_file() and p.name not in ignore_list and not p.name.startswith('.'):
                artifacts.append(str(p))
                print(f"Found artifact: {{p}}", flush=True)
        
        # Prepare output
        output = {{
            "success": True,
            "result": result,
            "artifacts": artifacts,
            "execution_time": execution_time
        }}
        
        # Use a unique marker to make parsing easier
        print(f"RESULT_JSON:{{json.dumps(output, default=str)}}", flush=True)
        
    except Exception as e:
        error_output = {{
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "traceback": traceback.format_exc()
        }}
        # Print to both stdout and stderr to ensure we capture it
        error_json = json.dumps(error_output)
        print(f"RESULT_JSON:{{error_json}}", flush=True)
        print(f"RESULT_JSON:{{error_json}}", file=sys.stderr, flush=True)
        sys.exit(1)
"""
    return script_template


def parse_execution_output(stdout: str) -> Optional[Dict[str, Any]]:
    """Parse the execution output to extract the result JSON."""
    if not stdout:
        logger.warning("Empty stdout received")
        return None
    
    # Look for our result marker in the output
    lines = stdout.strip().split('\n')
    for line in reversed(lines):  # Check from the end first
        if line.startswith('RESULT_JSON:'):
            try:
                json_str = line[12:]  # Skip 'RESULT_JSON:'
                result = json.loads(json_str)
                logger.debug(f"Successfully parsed result JSON: {result}")
                return result
            except json.JSONDecodeError as e:
                logger.error(f'Failed to parse result JSON: {e}')
                logger.debug(f'Offending line: {line}')
                # Continue looking for other RESULT_JSON lines
                continue
    
    # If no valid RESULT_JSON found, log some debug info
    logger.warning('No RESULT_JSON marker found in output')
    logger.debug(f'Last 10 lines of output:\n{chr(10).join(lines[-10:])}')
    
    return None


def move_artifacts(
    artifacts: List[str], 
    exec_dir: Path, 
    artifacts_storage_dir: str, 
    exec_id: str
) -> List[str]:
    """Move artifacts from execution directory to permanent storage."""
    storage_path = Path(artifacts_storage_dir)
    storage_path.mkdir(parents=True, exist_ok=True)
    
    final_artifact_paths = []
    
    for artifact_name in artifacts:
        src_path = exec_dir / artifact_name
        
        if not src_path.exists():
            logger.warning(f"Artifact '{artifact_name}' not found at {src_path}")
            continue
            
        if src_path.name in ['execute.py', 'inputs.json', 'requirements.txt']:
            logger.debug(f"Skipping system file: {artifact_name}")
            continue
        
        try:
            # Sanitize filename
            safe_name = ''.join(
                c for c in Path(artifact_name).name 
                if c.isalnum() or c in ('_', '-', '.')
            )
            
            # Create unique filename with execution ID
            final_name = f'{exec_id}_{safe_name}'
            final_path = storage_path / final_name
            
            # Copy the file
            shutil.copy2(src_path, final_path)
            final_artifact_paths.append(str(final_path.resolve()))
            
            logger.info(f"Moved artifact '{artifact_name}' to '{final_path}'")
            
        except Exception as e:
            logger.error(f"Failed to move artifact '{artifact_name}': {e}")
    
    return final_artifact_paths