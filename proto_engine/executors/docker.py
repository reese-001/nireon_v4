# nireon_v4\proto_engine\executors\docker.py
from __future__ import annotations
import asyncio
import json
import logging
import shutil
import time
import uuid
from pathlib import Path
from typing import Dict, Any, TYPE_CHECKING, List

from .base import ExternalExecutor
from .script_utils import generate_execution_script, parse_execution_output, move_artifacts

if TYPE_CHECKING:
    from domain.proto.base_schema import ProtoBlock
    from domain.context import NireonExecutionContext
    from proto_engine.config import ProtoEngineConfig

logger = logging.getLogger(__name__)

try:
    import docker
    from docker.errors import ContainerError, ImageNotFound, APIError
    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False
    ContainerError = ImageNotFound = APIError = Exception


class DockerExecutor(ExternalExecutor):
    def __init__(self, config: 'ProtoEngineConfig'):
        self.config = config
        if not DOCKER_AVAILABLE:
            raise RuntimeError("The 'docker' package is not installed. Please install it to use DockerExecutor.")
        try:
            # Set a longer default timeout for the docker client itself
            self.docker_client = docker.from_env(timeout=120)
            self.docker_client.ping()
            logger.info('Docker client initialized and connected successfully.')
        except Exception as e:
            logger.error(f'Docker is not available or configured correctly: {e}')
            raise RuntimeError(f'Docker not available: {e}') from e

    async def execute(self, proto: 'ProtoBlock', context: 'NireonExecutionContext') -> Dict[str, Any]:
        image_name = f'{self.config.docker_image_prefix}-{proto.eidos}:latest'
        exec_id = f'{proto.id}-{uuid.uuid4().hex[:8]}'
        work_dir = Path(self.config.work_directory) / exec_id
        work_dir.mkdir(parents=True, exist_ok=True)
        
        script_content = generate_execution_script(proto.model_dump())
        script_path = work_dir / 'execute.py'
        script_path.write_text(script_content)
        
        inputs_path = work_dir / 'inputs.json'
        inputs_path.write_text(json.dumps(proto.inputs))

        # Simplify the command, as installation is now handled inside the script
        command = ['python', '-I', '/app/execute.py', '/app/inputs.json']
        
        memory_limit_mb = min(proto.limits.get('memory_mb', self.config.default_memory_mb), self.config.default_memory_mb)
        
        # This is the timeout for the code *inside* the container
        internal_timeout_sec = proto.limits.get('timeout_sec', self.config.default_timeout_sec)
        
        # FIX: This is the timeout for our Python script waiting on the Docker API call.
        # We'll make it slightly longer than the internal timeout to prevent race conditions.
        api_wait_timeout = internal_timeout_sec + 30

        container = None
        start_time = time.time()
        try:
            logger.debug(f"Running container '{image_name}' for Proto block '{proto.id}'. Command: {' '.join(command)}")
            container = self.docker_client.containers.run(
                image=image_name,
                command=command,
                volumes={str(work_dir.resolve()): {'bind': '/app', 'mode': 'rw'}},
                mem_limit=f'{memory_limit_mb}m',
                detach=True,
                remove=False,
                user='root'
            )

            # Use the new, longer timeout for the wait() call
            exit_code_info = await asyncio.to_thread(container.wait, timeout=api_wait_timeout)
            execution_time = time.time() - start_time
            logs = container.logs().decode('utf-8', errors='ignore')
            
            logger.debug(f"Container for '{proto.id}' finished with status {exit_code_info.get('StatusCode', 'N/A')}.")

            if logs:
                log_preview = logs.replace('\\n', '\\\\n')[:500]
                logger.debug(f'Container logs (preview): {log_preview}')
            else:
                logger.debug('Container produced no logs.')

            result_json = parse_execution_output(logs)

            if result_json is None:
                error_message = 'No result JSON found in container output.'
                full_logs_for_error = f'\\n--- Full Container Logs ---\\n{logs}\\n--- End Container Logs ---'
                logger.error(f"Execution failed for Proto '{proto.id}'. {error_message} {full_logs_for_error}")
                return {'success': False, 'error': error_message, 'error_type': 'OutputParsingError', 'stdout': logs}

            if result_json.get('success') and result_json.get('artifacts'):
                result_json['artifacts'] = move_artifacts(
                    result_json['artifacts'], work_dir, self.config.artifacts_directory, exec_id
                )
            
            result_json['execution_time_sec'] = execution_time
            return result_json

        except asyncio.TimeoutError:
            logger.warning(f"Execution for Proto '{proto.id}' timed out after {internal_timeout_sec}s.")
            return {'success': False, 'error': f'Execution timed out after {internal_timeout_sec}s.', 'error_type': 'TimeoutError'}
        except ContainerError as e:
            stderr_logs = e.stderr.decode() if e.stderr else 'No stderr.'
            logger.error(f"Container error for Proto '{proto.id}': {e}. Stderr: {stderr_logs}")
            return {'success': False, 'error': f'Container error: {stderr_logs}', 'error_type': 'ContainerError'}
        except ImageNotFound:
            return {'success': False, 'error': f'Docker image not found: {image_name}. Please build it first.', 'error_type': 'ImageNotFound'}
        except APIError as e:
            return {'success': False, 'error': f'Docker API error: {e}', 'error_type': 'DockerAPIError'}
        except Exception as e:
            # This is where the ReadTimeoutError was caught
            logger.exception(f"Unexpected error during Docker execution for Proto '{proto.id}'.")
            return {'success': False, 'error': str(e), 'error_type': type(e).__name__}
        finally:
            if container:
                try:
                    # Stop the container in case it's still running (e.g., due to timeout)
                    await asyncio.to_thread(container.stop, timeout=5)
                    await asyncio.to_thread(container.remove, force=True)
                except Exception as e:
                    logger.warning(f'Could not stop/remove container {container.id}: {e}')
            if self.config.cleanup_after_execution and work_dir.exists():
                shutil.rmtree(work_dir, ignore_errors=True)

    async def cleanup(self):
        try:
            self.docker_client.close()
        except Exception:
            pass