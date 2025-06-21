from __future__ import annotations
import asyncio
import json
import logging
import platform
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Dict, Any, TYPE_CHECKING
from concurrent.futures import ProcessPoolExecutor

from .base import ExternalExecutor
from .script_utils import generate_execution_script, parse_execution_output, move_artifacts

if TYPE_CHECKING:
    from domain.proto.base_schema import ProtoBlock
    from domain.context import NireonExecutionContext
    from proto_engine.config import ProtoEngineConfig

logger = logging.getLogger(__name__)

def _execute_in_subprocess(execution_data: Dict[str, Any]) -> Dict[str, Any]:
    if platform.system() != 'Windows':
        import resource

    try:
        proto_data = execution_data['proto_block']
        work_dir = Path(execution_data['work_directory'])
        artifacts_dir = execution_data['artifacts_directory']
        timeout_sec = execution_data['timeout_sec']
        memory_mb = execution_data['memory_mb']
        python_exec = execution_data['python_executable']
        
        exec_id = f"{proto_data['id']}-{uuid.uuid4().hex[:8]}"
        exec_dir = work_dir / exec_id
        exec_dir.mkdir(parents=True, exist_ok=True)
        
        script_content = generate_execution_script(proto_data)
        script_path = exec_dir / 'execute.py'
        script_path.write_text(script_content, encoding='utf-8')

        inputs_path = exec_dir / 'inputs.json'
        inputs_path.write_text(json.dumps(proto_data['inputs']), encoding='utf-8')

        def set_limits():
            if platform.system() != 'Windows':
                resource.setrlimit(resource.RLIMIT_AS, (memory_mb * 1024 * 1024, memory_mb * 1024 * 1024))
                resource.setrlimit(resource.RLIMIT_CPU, (timeout_sec, timeout_sec))

        # --- START OF FIX ---
        # Remove the "-I" (isolated mode) flag, as it can cause issues with the
        # Microsoft Store version of Python on Windows.
        command = [
            str(python_exec).strip(),
            str(script_path).strip(),
            str(inputs_path).strip()
        ]
        # --- END OF FIX ---
        
        logger.debug(f"Executing subprocess command: {' '.join(command)}")
        
        start_time = time.time()
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            preexec_fn=set_limits if platform.system() != 'Windows' else None,
            cwd=exec_dir,
            encoding='utf-8',
            errors='ignore'
        )

        execution_time = time.time() - start_time
        
        result_json = parse_execution_output(process.stdout)
        
        if result_json is None:
            error_message = 'No result JSON found in subprocess output.'
            if process.stderr:
                error_message += f" Stderr: {process.stderr.strip()}"
            
            return {
                'success': False,
                'error': error_message,
                'error_type': 'OutputParsingError',
                'stdout': process.stdout,
                'stderr': process.stderr
            }
        
        if result_json.get('success') and result_json.get('artifacts'):
            result_json['artifacts'] = move_artifacts(result_json['artifacts'], exec_dir, artifacts_dir, exec_id)
        
        result_json['execution_time_sec'] = execution_time
        return result_json

    except subprocess.TimeoutExpired:
        return {'success': False, 'error': 'Execution timed out.', 'error_type': 'TimeoutError'}
    except Exception as e:
        return {'success': False, 'error': str(e), 'error_type': type(e).__name__}
    finally:
        if 'exec_dir' in locals() and exec_dir.exists():
            shutil.rmtree(exec_dir, ignore_errors=True)


class SubprocessExecutor(ExternalExecutor):
    def __init__(self, config: 'ProtoEngineConfig'):
        self.config = config
        self._executor = ProcessPoolExecutor(max_workers=2)
        if platform.system() == 'Windows':
            logger.warning('SubprocessExecutor resource limits (memory, CPU) are not supported on Windows. Use DockerExecutor for enforcement.')

    async def execute(self, proto: 'ProtoBlock', context: 'NireonExecutionContext') -> Dict[str, Any]:
        execution_data = {
            'proto_block': proto.model_dump(),
            'work_directory': self.config.work_directory,
            'artifacts_directory': self.config.artifacts_directory,
            'timeout_sec': proto.limits.get('timeout_sec', self.config.default_timeout_sec),
            'memory_mb': proto.limits.get('memory_mb', self.config.default_memory_mb),
            'python_executable': self.config.python_executable
        }
        
        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(self._executor, _execute_in_subprocess, execution_data)
        except Exception as e:
            logger.exception(f"Error submitting Proto block '{proto.id}' to subprocess.")
            return {'success': False, 'error': str(e), 'error_type': type(e).__name__}

    async def cleanup(self):
        if self._executor:
            self._executor.shutdown(wait=True)