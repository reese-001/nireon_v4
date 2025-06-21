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
    safe_code_block = json.dumps(code_block)

    return f"""
import sys
import os
import json
import time
import traceback
from pathlib import Path

# --- User-provided Code Block ---
# The code is loaded from a JSON-safe string to avoid escaping issues
user_code = {safe_code_block}
exec(user_code, globals())
# ------------------------------

if __name__ == "__main__":
    try:
        if len(sys.argv) > 1:
            inputs_file_path = sys.argv[1]
            with open(inputs_file_path, 'r', encoding='utf-8') as f:
                inputs = json.load(f)
        else:
            inputs = {{}}

        entry_point = locals().get('{function_name}')
        if not callable(entry_point):
            raise NameError("Entry point function '{function_name}' not found or not callable.")

        result = entry_point(**inputs)
        
        # --- FIX: Add requirements.txt to the ignore list ---
        ignore_list = ['execute.py', 'inputs.json', 'requirements.txt']
        artifacts = [
            str(p) for p in Path(".").iterdir() 
            if p.is_file() and p.name not in ignore_list and not p.name.startswith('.')
        ]
        
        output = {{
            "success": True,
            "result": result,
            "artifacts": artifacts
        }}
        # Use an f-string to embed the JSON to avoid potential escaping issues with print
        print(f"RESULT_JSON:{{json.dumps(output, default=str)}}")
        
    except Exception as e:
        error_output = {{
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "traceback": traceback.format_exc()
        }}
        print(f"RESULT_JSON:{{json.dumps(error_output)}}")
"""

def parse_execution_output(stdout: str) -> Optional[Dict[str, Any]]:
    for line in stdout.strip().split('\n'):
        if line.startswith('RESULT_JSON:'):
            try:
                return json.loads(line[12:])
            except json.JSONDecodeError as e:
                logger.error(f'Failed to parse result JSON from output line: {e}')
                logger.debug(f'Offending line: {line}')
                return {'success': False, 'error': 'Result JSON parsing failed.', 'raw_line': line}
    return None

def move_artifacts(artifacts: List[str], exec_dir: Path, artifacts_storage_dir: str, exec_id: str) -> List[str]:
    storage_path = Path(artifacts_storage_dir)
    final_artifact_paths = []

    for artifact_name in artifacts:
        src_path = exec_dir / artifact_name
        if src_path.exists() and src_path.name not in ['execute.py', 'inputs.json']:
            try:
                safe_name = ''.join(c for c in Path(artifact_name).name if c.isalnum() or c in ('_', '-', '.'))
                final_path = storage_path / f"{exec_id}_{safe_name}"
                shutil.copy2(src_path, final_path)
                final_artifact_paths.append(str(final_path.resolve()))
            except Exception as e:
                logger.warning(f"Could not move artifact '{artifact_name}': {e}")
                
    return final_artifact_paths