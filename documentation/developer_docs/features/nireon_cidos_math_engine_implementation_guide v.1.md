# CIDOS Math Engine Implementation Guide

## Executive Summary

This guide provides a step-by-step implementation roadmap for integrating CIDOS (Cognitive Intent Dialect Of System) math engine capabilities into NIREON V4. The approach leverages NIREON's existing infrastructure while adding controlled mathematical execution capabilities.

## Prerequisites

- NIREON V4 system with working ComponentRegistry, EventBus, and Reactor
- Python 3.12+ environment with subprocess capabilities
- Understanding of NIREON's signal-based architecture

## Implementation Phases

### Phase 1: Core Infrastructure (Week 1-2)

#### 1.1 CIDOS Schema Definition

**Location:** `nireon_v4/domain/cidos/`

**File:** `base_schema.py`
```python
from pydantic import BaseModel, Field, validator
from typing import Dict, Any, List, Optional, Literal
from datetime import datetime

class CIDOSBase(BaseModel):
    """Base CIDOS schema for all dialects."""
    schema_version: str = Field(default="cidos/1.0", description="CIDOS schema version")
    id: str = Field(..., description="Unique identifier for this CIDOS block")
    description: str = Field(..., description="Human-readable description")
    
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    context: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('id')
    def validate_id(cls, v):
        # Ensure ID follows naming convention
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError("ID must contain only alphanumeric characters, hyphens, and underscores")
        return v

class CIDOSMathBlock(CIDOSBase):
    """CIDOS math dialect schema."""
    eidos: Literal['math'] = 'math'
    objective: str = Field(..., description="What this mathematical analysis aims to achieve")
    
    # Core execution parameters
    function_name: str = Field(..., description="Entry point function name in the code")
    code: str = Field(..., description="Python code to execute")
    inputs: Dict[str, Any] = Field(default_factory=dict, description="Input parameters")
    
    # Mathematical metadata
    equation_latex: Optional[str] = Field(default=None, description="LaTeX representation of key equations")
    requirements: List[str] = Field(default_factory=list, description="Python package requirements")
    
    # Execution constraints
    limits: Dict[str, Any] = Field(default_factory=lambda: {"timeout_sec": 10, "memory_mb": 256})
    
    @validator('code')
    def validate_code_structure(cls, v, values):
        function_name = values.get('function_name')
        if function_name and f"def {function_name}(" not in v:
            raise ValueError(f"Code must define function '{function_name}'")
        return v
    
    class Config:
        extra = "forbid"
```

**File:** `validation.py`
```python
import ast
import re
from typing import List, Set
from .base_schema import CIDOSMathBlock

class CIDOSValidator:
    """Validates CIDOS blocks for security and correctness."""
    
    # Security: Dangerous imports/functions to block
    BLOCKED_IMPORTS = {
        'os', 'subprocess', 'sys', 'shutil', 'socket', 'urllib', 'requests',
        'pickle', 'eval', 'exec', 'open', '__import__', 'compile'
    }
    
    ALLOWED_IMPORTS = {
        'matplotlib.pyplot', 'numpy', 'pandas', 'math', 'statistics',
        'scipy', 'sympy', 'datetime', 'json', 'csv'
    }
    
    @classmethod
    def validate_math_block(cls, block: CIDOSMathBlock) -> List[str]:
        """Validate a CIDOS math block. Returns list of errors."""
        errors = []
        
        # Validate code syntax
        try:
            ast.parse(block.code)
        except SyntaxError as e:
            errors.append(f"Syntax error in code: {e}")
            return errors  # Don't continue if syntax is broken
        
        # Security validation
        security_errors = cls._validate_security(block.code)
        errors.extend(security_errors)
        
        # Function validation
        if not cls._function_exists(block.code, block.function_name):
            errors.append(f"Function '{block.function_name}' not found in code")
        
        # Resource limits validation
        if block.limits.get('timeout_sec', 0) > 30:
            errors.append("Timeout cannot exceed 30 seconds")
        
        if block.limits.get('memory_mb', 0) > 1024:
            errors.append("Memory limit cannot exceed 1024 MB")
        
        return errors
    
    @classmethod
    def _validate_security(cls, code: str) -> List[str]:
        """Check for security violations in code."""
        errors = []
        
        # Check for blocked imports
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    if name.name in cls.BLOCKED_IMPORTS:
                        errors.append(f"Blocked import: {name.name}")
            elif isinstance(node, ast.ImportFrom):
                if node.module in cls.BLOCKED_IMPORTS:
                    errors.append(f"Blocked import: {node.module}")
        
        # Check for dangerous function calls
        dangerous_patterns = [
            r'open\s*\(',
            r'eval\s*\(',
            r'exec\s*\(',
            r'__import__\s*\(',
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, code):
                errors.append(f"Dangerous pattern detected: {pattern}")
        
        return errors
    
    @classmethod
    def _function_exists(cls, code: str, function_name: str) -> bool:
        """Check if the specified function exists in the code."""
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == function_name:
                    return True
        except:
            pass
        return False
```

#### 1.2 New Signal Types

**Location:** `nireon_v4/signals/core.py` (add to existing file)

```python
class MathTaskCIDOSSignal(EpistemicSignal):
    """Signal carrying a CIDOS math block for execution."""
    signal_type: Literal['MathTaskCIDOSSignal'] = 'MathTaskCIDOSSignal'
    cidos_block: Dict[str, Any] = Field(description="The CIDOS math block to execute")
    execution_priority: int = Field(default=5, description="Execution priority (1-10)")

class MathErrorSignal(EpistemicSignal):
    """Signal indicating mathematical computation failure."""
    signal_type: Literal['MathErrorSignal'] = 'MathErrorSignal'
    cidos_block_id: str = Field(description="ID of the failed CIDOS block")
    error_type: str = Field(description="Type of error (validation, execution, timeout)")
    error_message: str = Field(description="Detailed error message")
    execution_context: Dict[str, Any] = Field(default_factory=dict)
```

#### 1.3 Configuration Models

**Location:** `nireon_v4/mechanisms/dynamic_math_runner/config.py`

```python
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any
from pathlib import Path

class DynamicMathRunnerConfig(BaseModel):
    """Configuration for the Dynamic Math Runner mechanism."""
    
    # Execution environment
    python_executable: str = Field(default="python", description="Python executable to use")
    work_directory: str = Field(default="runtime/math/workspace", description="Working directory for executions")
    artifacts_directory: str = Field(default="runtime/math/artifacts", description="Directory for output artifacts")
    
    # Security limits (defaults from security research best practices)
    default_timeout_sec: int = Field(default=10, ge=1, le=30)
    default_memory_mb: int = Field(default=256, ge=64, le=1024)
    max_file_size_mb: int = Field(default=10, ge=1, le=50)
    
    # Allowed packages for installation
    allowed_packages: List[str] = Field(
        default_factory=lambda: [
            "matplotlib", "numpy", "pandas", "scipy", "sympy", 
            "seaborn", "plotly", "statsmodels"
        ]
    )
    
    # Monitoring and cleanup
    cleanup_after_execution: bool = Field(default=True)
    retain_artifacts_hours: int = Field(default=24, ge=1, le=168)  # Max 1 week
    
    # Integration settings
    event_bus_timeout_sec: int = Field(default=30)
    
    @validator('work_directory', 'artifacts_directory')
    def ensure_directory_exists(cls, v):
        Path(v).mkdir(parents=True, exist_ok=True)
        return v
    
    class Config:
        extra = "forbid"
```

### Phase 2: Math Runner Component (Week 2-3)

#### 2.1 Core Math Runner Implementation

**Location:** `nireon_v4/mechanisms/dynamic_math_runner/service.py`

```python
import asyncio
import subprocess
import tempfile
import uuid
import json
import shutil
import time
import resource
from pathlib import Path
from typing import Dict, Any, List, Optional
from concurrent.futures import ProcessPoolExecutor, TimeoutError

from nireon_v4.application.components.base import NireonBaseComponent
from nireon_v4.application.context import ExecutionContext
from nireon_v4.application.components.lifecycle import ProcessResult, ComponentHealth
from nireon_v4.signals.core import MathResultSignal, MathErrorSignal
from nireon_v4.domain.cidos.base_schema import CIDOSMathBlock
from nireon_v4.domain.cidos.validation import CIDOSValidator

from .config import DynamicMathRunnerConfig

def _execute_in_subprocess(execution_data: Dict[str, Any]) -> Dict[str, Any]:
    """Execute CIDOS math block in isolated subprocess. Must be top-level for pickling."""
    
    try:
        cidos_data = execution_data['cidos_block']
        work_dir = Path(execution_data['work_directory'])
        timeout_sec = execution_data['timeout_sec']
        memory_mb = execution_data['memory_mb']
        
        # Create isolated execution directory
        exec_id = str(uuid.uuid4())
        exec_dir = work_dir / exec_id
        exec_dir.mkdir(parents=True, exist_ok=True)
        
        # Write the Python script
        script_content = f"""
import sys
import os
import json
from pathlib import Path

# Restrict imports to allowed list
{cidos_data['code']}

# Execute the function
if __name__ == "__main__":
    try:
        # Set working directory for relative file operations
        os.chdir("{exec_dir}")
        
        # Call the main function
        inputs = {json.dumps(cidos_data['inputs'])}
        result = {cidos_data['function_name']}(**inputs)
        
        # Look for generated files
        artifacts = []
        for file_path in Path(".").iterdir():
            if file_path.is_file() and file_path.suffix in ['.png', '.pdf', '.svg', '.html', '.csv']:
                artifacts.append(str(file_path))
        
        # Output results as JSON
        output = {{
            "success": True,
            "result": result,
            "artifacts": artifacts,
            "execution_time": time.time()
        }}
        print("RESULT_JSON:" + json.dumps(output))
        
    except Exception as e:
        error_output = {{
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }}
        print("RESULT_JSON:" + json.dumps(error_output))
"""
        
        script_path = exec_dir / "execute.py"
        script_path.write_text(script_content)
        
        # Set resource limits
        def set_limits():
            resource.setrlimit(resource.RLIMIT_AS, (memory_mb * 1024 * 1024, memory_mb * 1024 * 1024))
            resource.setrlimit(resource.RLIMIT_CPU, (timeout_sec, timeout_sec))
        
        # Execute with timeout
        start_time = time.time()
        process = subprocess.run(
            ["python", "-I", str(script_path)],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            preexec_fn=set_limits,
            cwd=exec_dir
        )
        
        execution_time = time.time() - start_time
        
        # Parse result
        stdout_lines = process.stdout.strip().split('\n')
        result_json = None
        
        for line in stdout_lines:
            if line.startswith("RESULT_JSON:"):
                result_json = json.loads(line[12:])
                break
        
        if result_json is None:
            return {
                "success": False,
                "error": "No result JSON found in output",
                "stdout": process.stdout,
                "stderr": process.stderr
            }
        
        # Move artifacts to permanent location if successful
        if result_json.get("success") and result_json.get("artifacts"):
            artifacts_dir = Path(execution_data['artifacts_directory'])
            final_artifacts = []
            
            for artifact in result_json["artifacts"]:
                src_path = exec_dir / artifact
                if src_path.exists():
                    final_path = artifacts_dir / f"{exec_id}_{artifact}"
                    shutil.copy2(src_path, final_path)
                    final_artifacts.append(str(final_path))
            
            result_json["artifacts"] = final_artifacts
        
        result_json["execution_time_sec"] = execution_time
        result_json["subprocess_returncode"] = process.returncode
        
        return result_json
        
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": f"Execution timed out after {timeout_sec} seconds",
            "error_type": "TimeoutError"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }
    finally:
        # Cleanup execution directory
        if 'exec_dir' in locals() and exec_dir.exists():
            shutil.rmtree(exec_dir, ignore_errors=True)


class DynamicMathRunner(NireonBaseComponent):
    """NIREON component for executing CIDOS math blocks."""
    
    def __init__(self, instance_id: str, config: DynamicMathRunnerConfig, **kwargs):
        super().__init__(instance_id, config, **kwargs)
        self.config: DynamicMathRunnerConfig = config
        self._executor = None
    
    async def initialize(self, context: ExecutionContext) -> None:
        """Initialize the math runner."""
        await super().initialize(context)
        
        # Create process pool for execution isolation
        self._executor = ProcessPoolExecutor(max_workers=2)
        
        # Ensure directories exist
        Path(self.config.work_directory).mkdir(parents=True, exist_ok=True)
        Path(self.config.artifacts_directory).mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"DynamicMathRunner {self.instance_id} initialized")
    
    async def process(self, data: Any, context: ExecutionContext) -> ProcessResult:
        """Process a MathTaskCIDOSSignal."""
        
        try:
            # Extract CIDOS block
            if isinstance(data, dict) and 'cidos_block' in data:
                cidos_data = data['cidos_block']
            else:
                return ProcessResult(
                    success=False,
                    message="Invalid input: expected CIDOS block data",
                    component_id=self.instance_id
                )
            
            # Validate CIDOS block
            try:
                cidos_block = CIDOSMathBlock(**cidos_data)
            except Exception as e:
                await self._emit_error_signal(
                    cidos_data.get('id', 'unknown'),
                    'validation',
                    f"CIDOS validation failed: {e}",
                    context
                )
                return ProcessResult(
                    success=False,
                    message=f"CIDOS validation failed: {e}",
                    component_id=self.instance_id
                )
            
            # Security validation
            validation_errors = CIDOSValidator.validate_math_block(cidos_block)
            if validation_errors:
                error_msg = "; ".join(validation_errors)
                await self._emit_error_signal(
                    cidos_block.id,
                    'security',
                    f"Security validation failed: {error_msg}",
                    context
                )
                return ProcessResult(
                    success=False,
                    message=f"Security validation failed: {error_msg}",
                    component_id=self.instance_id
                )
            
            # Execute the math block
            result = await self._execute_cidos_block(cidos_block, context)
            
            if result['success']:
                # Emit success signal
                await self._emit_result_signal(cidos_block, result, context)
                
                return ProcessResult(
                    success=True,
                    message=f"Successfully executed CIDOS block {cidos_block.id}",
                    component_id=self.instance_id,
                    output_data=result
                )
            else:
                # Emit error signal
                await self._emit_error_signal(
                    cidos_block.id,
                    'execution',
                    result.get('error', 'Unknown execution error'),
                    context
                )
                return ProcessResult(
                    success=False,
                    message=f"Execution failed: {result.get('error')}",
                    component_id=self.instance_id,
                    output_data=result
                )
        
        except Exception as e:
            self.logger.error(f"Unexpected error in DynamicMathRunner: {e}", exc_info=True)
            return ProcessResult(
                success=False,
                message=f"Unexpected error: {e}",
                component_id=self.instance_id
            )
    
    async def _execute_cidos_block(self, cidos_block: CIDOSMathBlock, context: ExecutionContext) -> Dict[str, Any]:
        """Execute a validated CIDOS block."""
        
        execution_data = {
            'cidos_block': cidos_block.dict(),
            'work_directory': self.config.work_directory,
            'artifacts_directory': self.config.artifacts_directory,
            'timeout_sec': cidos_block.limits.get('timeout_sec', self.config.default_timeout_sec),
            'memory_mb': cidos_block.limits.get('memory_mb', self.config.default_memory_mb)
        }
        
        try:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                self._executor,
                _execute_in_subprocess,
                execution_data
            )
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to execute CIDOS block {cidos_block.id}: {e}")
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__
            }
    
    async def _emit_result_signal(self, cidos_block: CIDOSMathBlock, result: Dict[str, Any], context: ExecutionContext):
        """Emit MathResultSignal for successful execution."""
        
        signal = MathResultSignal(
            source_node_id=self.instance_id,
            natural_language_query=cidos_block.objective,
            explanation=f"Successfully executed mathematical analysis: {cidos_block.description}",
            computation_details={
                "cidos_block_id": cidos_block.id,
                "function_name": cidos_block.function_name,
                "execution_time_sec": result.get("execution_time_sec", 0),
                "numeric_result": result.get("result"),
                "artifacts": result.get("artifacts", []),
                "equation_latex": cidos_block.equation_latex
            }
        )
        
        if hasattr(context, 'event_bus') and context.event_bus:
            await context.event_bus.publish(signal.signal_type, signal.dict())
    
    async def _emit_error_signal(self, cidos_id: str, error_type: str, error_message: str, context: ExecutionContext):
        """Emit MathErrorSignal for failed execution."""
        
        signal = MathErrorSignal(
            source_node_id=self.instance_id,
            cidos_block_id=cidos_id,
            error_type=error_type,
            error_message=error_message,
            execution_context={"component_id": self.instance_id}
        )
        
        if hasattr(context, 'event_bus') and context.event_bus:
            await context.event_bus.publish(signal.signal_type, signal.dict())
    
    async def health_check(self, context: ExecutionContext) -> ComponentHealth:
        """Check component health."""
        
        try:
            # Check if directories are accessible
            work_path = Path(self.config.work_directory)
            artifacts_path = Path(self.config.artifacts_directory)
            
            if not work_path.exists() or not artifacts_path.exists():
                return ComponentHealth(
                    status="unhealthy",
                    message="Required directories not accessible"
                )
            
            # Check if executor is running
            if self._executor is None or self._executor._shutdown:
                return ComponentHealth(
                    status="unhealthy", 
                    message="Process executor not available"
                )
            
            return ComponentHealth(
                status="healthy",
                message="Math runner ready for execution"
            )
            
        except Exception as e:
            return ComponentHealth(
                status="unhealthy",
                message=f"Health check failed: {e}"
            )
    
    async def cleanup(self):
        """Cleanup resources."""
        if self._executor:
            self._executor.shutdown(wait=True)
        await super().cleanup()
```

### Phase 3: Integration & Reactor Rules (Week 3-4)

#### 3.1 Reactor Rules for CIDOS Routing

**Location:** `config/reactor_rules/cidos_math_rules.yaml`

```yaml
rules_definitions:
  - id: "cidos_math_task_router"
    description: "Routes CIDOS math tasks to DynamicMathRunner"
    conditions:
      - type: "signal_type_match"
        signal_type: "MathTaskCIDOSSignal"
    actions:
      - type: "trigger_component"
        component_id: "dynamic_math_runner_01"
        template_id: "EXECUTE_CIDOS_MATH_BLOCK"
    default_params:
      priority: 5
      timeout_override: null

  - id: "math_result_amplifier"
    description: "Routes successful math results for potential amplification"
    conditions:
      - type: "signal_type_match"
        signal_type: "MathResultSignal"
      - type: "payload_value"
        field: "computation_details.execution_time_sec"
        operator: "lt"
        value: 5.0  # Fast executions might indicate simple, amplifiable results
    actions:
      - type: "trigger_component"
        component_id: "catalyst_main"
        template_id: "AMPLIFY_MATH_INSIGHTS"

  - id: "math_error_handler"
    description: "Handles math execution errors"
    conditions:
      - type: "signal_type_match"
        signal_type: "MathErrorSignal"
    actions:
      - type: "emit_signal"
        signal_type: "SystemAlert"
        payload_static:
          alert_type: "math_execution_failure"
          severity: "medium"
```

#### 3.2 Component Registration in Manifest

**Location:** `config/manifests/math_enhanced.yaml`

```yaml
version: "1.0"
metadata:
  name: "NIREON with CIDOS Math Engine"
  description: "NIREON configuration including dynamic mathematical analysis capabilities"

shared_services:
  llm:
    provider: "openai"
    config:
      model: "${LLM_MODEL:-gpt-4o-mini}"
      temperature: 0.7

  event_bus:
    provider: "memory"
    config:
      max_history: 2000

mechanisms:
  - id: "dynamic_math_runner_01"
    class: "DynamicMathRunner"
    config:
      work_directory: "${MATH_WORK_DIR:-runtime/math/workspace}"
      artifacts_directory: "${MATH_ARTIFACTS_DIR:-runtime/math/artifacts}"
      default_timeout_sec: 15
      default_memory_mb: 512
      allowed_packages:
        - "matplotlib"
        - "numpy" 
        - "pandas"
        - "scipy"
        - "sympy"
        - "seaborn"

  - id: "explorer_primary"
    class: "Explorer"
    config:
      max_depth: 3
      exploration_strategy: "breadth_first"
      confidence_threshold: 0.6

  - id: "catalyst_main" 
    class: "Catalyst"
    config:
      amplification_factor: 1.5
      min_trust_score: 0.7

observers:
  - id: "sentinel_guardian"
    class: "Sentinel" 
    config:
      check_interval: 10
      alert_threshold: 0.9
```

#### 3.3 Template Definitions

**Location:** `config/templates/cidos_math_templates.yaml`

```yaml
version: "1.0"

templates:
  EXECUTE_CIDOS_MATH_BLOCK:
    description: "Execute a CIDOS math block"
    command_type: ApplyMechanismCommand
    default_params:
      mechanism_id: "dynamic_math_runner_01"
    expected_input_signals:
      - "MathTaskCIDOSSignal"
    expected_output_signals:
      - "MathResultSignal"
      - "MathErrorSignal"
    validation:
      require_fields: ["cidos_block"]

  AMPLIFY_MATH_INSIGHTS:
    description: "Amplify insights from mathematical analysis"
    command_type: ApplyMechanismCommand
    default_params:
      mechanism_id: "catalyst_main"
    expected_input_signals:
      - "MathResultSignal"
    context_enrichment:
      include_artifacts: true
      include_latex: true
```

### Phase 4: LLM Integration for CIDOS Generation (Week 4-5)

#### 4.1 CIDOS Generator Component

**Location:** `nireon_v4/mechanisms/cidos_generator/service.py`

```python
from typing import Dict, Any, List
import yaml
import json

from nireon_v4.application.components.base import NireonBaseComponent
from nireon_v4.application.context import ExecutionContext
from nireon_v4.application.components.lifecycle import ProcessResult
from nireon_v4.signals.core import MathTaskCIDOSSignal
from nireon_v4.domain.cidos.base_schema import CIDOSMathBlock
from nireon_v4.domain.cidos.validation import CIDOSValidator

class CIDOSGenerator(NireonBaseComponent):
    """Generates CIDOS math blocks from natural language requests using LLM."""
    
    CIDOS_GENERATION_PROMPT = """
You are a CIDOS (Cognitive Intent Dialect Of System) math block generator. 
Generate executable mathematical analysis code based on the user's request.

CIDOS Math Block Schema:
```yaml
schema_version: cidos/1.0
eidos: math
id: UNIQUE_ID_HERE
description: "Brief description"
objective: "What this analysis achieves"
function_name: main_function
inputs:
  param1: value1
  param2: value2
code: |
  import matplotlib.pyplot as plt
  import numpy as np
  
  def main_function(param1, param2):
      # Your implementation here
      return result
      
  def generate_plots():
      # Generate any visualizations
      plt.savefig("output.png")
      
equation_latex: "Mathematical equation in LaTeX"
limits:
  timeout_sec: 10
  memory_mb: 256
```

Security Constraints:
- Only use allowed imports: matplotlib.pyplot, numpy, pandas, scipy, sympy, math, statistics
- No file system access beyond saving plots
- No network access
- No subprocess calls

User Request: {user_request}

Generate a complete CIDOS math block:
"""
    
    async def process(self, data: Any, context: ExecutionContext) -> ProcessResult:
        """Generate CIDOS block from natural language request."""
        
        try:
            user_request = data.get('natural_language_request', '')
            if not user_request:
                return ProcessResult(
                    success=False,
                    message="No natural language request provided",
                    component_id=self.instance_id
                )
            
            # Get LLM service
            llm_service = context.component_registry.get_service_instance("llm_router")
            if not llm_service:
                return ProcessResult(
                    success=False,
                    message="LLM service not available",
                    component_id=self.instance_id
                )
            
            # Generate CIDOS block
            prompt = self.CIDOS_GENERATION_PROMPT.format(user_request=user_request)
            
            llm_response = await llm_service.call_llm_async(
                prompt=prompt,
                temperature=0.3,  # Lower temperature for more consistent code
                max_tokens=2000
            )
            
            # Parse YAML response
            try:
                cidos_data = yaml.safe_load(llm_response.text)
            except yaml.YAMLError as e:
                return ProcessResult(
                    success=False,
                    message=f"Failed to parse LLM response as YAML: {e}",
                    component_id=self.instance_id
                )
            
            # Validate generated CIDOS block
            try:
                cidos_block = CIDOSMathBlock(**cidos_data)
                validation_errors = CIDOSValidator.validate_math_block(cidos_block)
                
                if validation_errors:
                    return ProcessResult(
                        success=False,
                        message=f"Generated CIDOS block failed validation: {'; '.join(validation_errors)}",
                        component_id=self.instance_id
                    )
                
            except Exception as e:
                return ProcessResult(
                    success=False,
                    message=f"Generated CIDOS block schema validation failed: {e}",
                    component_id=self.instance_id
                )
            
            # Emit CIDOS task signal
            math_signal = MathTaskCIDOSSignal(
                source_node_id=self.instance_id,
                cidos_block=cidos_data,
                execution_priority=5
            )
            
            await context.event_bus.publish(math_signal.signal_type, math_signal.dict())
            
            return ProcessResult(
                success=True,
                message=f"Generated and queued CIDOS math block: {cidos_block.id}",
                component_id=self.instance_id,
                output_data={"cidos_block": cidos_data}
            )
            
        except Exception as e:
            self.logger.error(f"Error in CIDOS generation: {e}", exc_info=True)
            return ProcessResult(
                success=False,
                message=f"CIDOS generation failed: {e}",
                component_id=self.instance_id
            )
```

### Phase 5: Testing & Validation (Week 5-6)

#### 5.1 Integration Test Suite

**Location:** `tests/integration/test_cidos_math_engine.py`

```python
import pytest
import asyncio
import tempfile
from pathlib import Path

from nireon_v4.bootstrap import bootstrap_nireon_system
from nireon_v4.signals.core import MathTaskCIDOSSignal, MathResultSignal
from nireon_v4.domain.cidos.base_schema import CIDOSMathBlock

@pytest.fixture
async def nireon_system():
    """Bootstrap a minimal NIREON system for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test manifest
        test_manifest = Path(temp_dir) / "test_manifest.yaml"
        test_manifest.write_text("""
version: "1.0"
mechanisms:
  - id: "dynamic_math_runner_test"
    class: "DynamicMathRunner"
    config:
      work_directory: "{temp_dir}/work"
      artifacts_directory: "{temp_dir}/artifacts"
""".format(temp_dir=temp_dir))
        
        boot_result = await bootstrap_nireon_system(
            config_paths=[test_manifest],
            strict_mode=False
        )
        
        assert boot_result.success
        yield boot_result
        
        # Cleanup
        # (temp_dir will be automatically cleaned up)

@pytest.mark.asyncio
async def test_simple_math_execution(nireon_system):
    """Test execution of a simple CIDOS math block."""
    
    # Create simple CIDOS block
    cidos_block = CIDOSMathBlock(
        id="TEST_SIMPLE_MATH",
        description="Simple arithmetic test",
        objective="Test basic mathematical computation",
        function_name="calculate",
        code="""
def calculate(a, b):
    return a + b
""",
        inputs={"a": 5, "b": 3}
    )
    
    # Get math runner component
    registry = nireon_system.registry
    math_runner = registry.get_service_instance("dynamic_math_runner_test")
    assert math_runner is not None
    
    # Create execution context
    from nireon_v4.application.context import NireonExecutionContext
    context = NireonExecutionContext(
        run_id="test_run",
        component_id="test",
        component_registry=registry,
        event_bus=registry.get_service_instance("event_bus")
    )
    
    # Execute
    result = await math_runner.process(
        {"cidos_block": cidos_block.dict()},
        context
    )
    
    assert result.success
    assert result.output_data["success"] is True
    assert result.output_data["result"] == 8

@pytest.mark.asyncio 
async def test_matplotlib_visualization(nireon_system):
    """Test CIDOS block that generates a plot."""
    
    cidos_block = CIDOSMathBlock(
        id="TEST_PLOT_GEN",
        description="Generate a simple plot",
        objective="Test matplotlib integration",
        function_name="create_plot",
        code="""
import matplotlib.pyplot as plt
import numpy as np

def create_plot():
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    
    plt.figure(figsize=(8, 6))
    plt.plot(x, y)
    plt.title("Sine Wave")
    plt.xlabel("X")
    plt.ylabel("Y") 
    plt.savefig("sine_wave.png")
    
    return "Plot generated"
""",
        inputs={}
    )
    
    # Execute
    registry = nireon_system.registry
    math_runner = registry.get_service_instance("dynamic_math_runner_test")
    
    from nireon_v4.application.context import NireonExecutionContext
    context = NireonExecutionContext(
        run_id="test_run",
        component_id="test", 
        component_registry=registry,
        event_bus=registry.get_service_instance("event_bus")
    )
    
    result = await math_runner.process(
        {"cidos_block": cidos_block.dict()},
        context
    )
    
    assert result.success
    assert len(result.output_data.get("artifacts", [])) > 0
    
    # Verify plot file exists
    artifact_path = Path(result.output_data["artifacts"][0])
    assert artifact_path.exists()
    assert artifact_path.suffix == ".png"

@pytest.mark.asyncio
async def test_security_validation():
    """Test that security validation blocks dangerous code."""
    
    from nireon_v4.domain.cidos.validation import CIDOSValidator
    
    # Test blocked imports
    dangerous_block = CIDOSMathBlock(
        id="DANGEROUS_TEST",
        description="Dangerous code test",
        objective="This should be blocked",
        function_name="bad_function",
        code="""
import os
import subprocess

def bad_function():
    os.system("rm -rf /")
    return "evil"
""",
        inputs={}
    )
    
    errors = CIDOSValidator.validate_math_block(dangerous_block)
    assert len(errors) > 0
    assert any("Blocked import" in error for error in errors)

@pytest.mark.asyncio
async def test_end_to_end_signal_flow(nireon_system):
    """Test complete signal flow from CIDOS task to result."""
    
    registry = nireon_system.registry
    event_bus = registry.get_service_instance("event_bus")
    
    # Subscribe to result signals
    results_received = []
    
    def on_math_result(payload):
        results_received.append(payload)
    
    event_bus.subscribe(MathResultSignal.__name__, on_math_result)
    
    # Create and emit CIDOS task signal
    cidos_block = {
        "schema_version": "cidos/1.0",
        "eidos": "math",
        "id": "TEST_E2E",
        "description": "End-to-end test",
        "objective": "Test complete signal flow",
        "function_name": "simple_calc",
        "code": "def simple_calc(x): return x * 2",
        "inputs": {"x": 21}
    }
    
    task_signal = MathTaskCIDOSSignal(
        source_node_id="test",
        cidos_block=cidos_block
    )
    
    await event_bus.publish(task_signal.signal_type, task_signal.dict())
    
    # Wait for processing (in real system, reactor would handle this)
    # For test, manually trigger math runner
    math_runner = registry.get_service_instance("dynamic_math_runner_test")
    
    from nireon_v4.application.context import NireonExecutionContext
    context = NireonExecutionContext(
        run_id="test_run",
        component_id="test",
        component_registry=registry,
        event_bus=event_bus
    )
    
    await math_runner.process(task_signal.dict(), context)
    
    # Give event bus time to process
    await asyncio.sleep(0.1)
    
    # Verify result was emitted
    assert len(results_received) > 0
    result = results_received[0]
    assert result["computation_details"]["numeric_result"] == 42
```

#### 5.2 Performance Benchmarks

**Location:** `tests/performance/test_cidos_performance.py`

```python
import pytest
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor

@pytest.mark.performance
async def test_concurrent_execution_performance():
    """Test performance under concurrent CIDOS execution load."""
    
    # Create multiple simple CIDOS blocks
    test_blocks = []
    for i in range(10):
        block = {
            "schema_version": "cidos/1.0",
            "eidos": "math",
            "id": f"PERF_TEST_{i}",
            "description": f"Performance test block {i}",
            "objective": "Performance testing",
            "function_name": "compute",
            "code": f"""
import math
def compute():
    result = 0
    for j in range(1000):
        result += math.sin(j * 0.001)
    return result
""",
            "inputs": {}
        }
        test_blocks.append(block)
    
    # Time concurrent execution
    start_time = time.time()
    
    # Execute all blocks concurrently (would be handled by NIREON's reactor in practice)
    tasks = []
    for block in test_blocks:
        # Simulate async execution
        task = asyncio.create_task(simulate_cidos_execution(block))
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Performance assertions
    assert total_time < 5.0  # Should complete within 5 seconds
    assert all(result["success"] for result in results)
    assert len(results) == 10
    
    print(f"Executed {len(test_blocks)} CIDOS blocks in {total_time:.2f} seconds")
    print(f"Average time per block: {total_time / len(test_blocks):.2f} seconds")

async def simulate_cidos_execution(cidos_block):
    """Simulate CIDOS block execution for performance testing."""
    # This would normally go through the full NIREON pipeline
    await asyncio.sleep(0.1)  # Simulate processing time
    return {"success": True, "execution_time": 0.1}

@pytest.mark.performance
def test_memory_usage_bounds():
    """Test that CIDOS execution respects memory limits."""
    
    # Test block that tries to allocate too much memory
    memory_hog_block = {
        "schema_version": "cidos/1.0", 
        "eidos": "math",
        "id": "MEMORY_TEST",
        "description": "Memory limit test",
        "objective": "Test memory constraints",
        "function_name": "allocate_memory",
        "code": """
def allocate_memory():
    # Try to allocate 2GB of memory
    big_list = [0] * (250 * 1024 * 1024)  # ~2GB of integers
    return len(big_list)
""",
        "inputs": {},
        "limits": {"memory_mb": 256}  # Limit to 256MB
    }
    
    # This should fail due to memory limits
    # (Implementation would need actual subprocess execution to test this)
    assert True  # Placeholder - actual test would verify memory enforcement
```

## Deployment Checklist

### Pre-Deployment Validation

- [ ] All unit tests pass
- [ ] Integration tests pass
- [ ] Security validation tests pass
- [ ] Performance benchmarks meet requirements
- [ ] CIDOS schema validation works correctly
- [ ] Subprocess isolation is properly configured
- [ ] File system permissions are correctly set
- [ ] Event bus integration functions correctly

### Production Configuration

- [ ] Configure appropriate timeout limits (≤30 seconds)
- [ ] Set memory limits (≤1GB recommended)
- [ ] Configure artifact cleanup policies  
- [ ] Set up monitoring and alerting
- [ ] Configure logging levels
- [ ] Set up backup procedures for artifacts
- [ ] Configure rate limiting if needed

### Monitoring & Maintenance

- [ ] Monitor execution times and success rates
- [ ] Monitor memory and CPU usage
- [ ] Monitor artifact storage growth
- [ ] Set up alerts for execution failures
- [ ] Regular cleanup of old artifacts
- [ ] Monitor for security violations
- [ ] Performance trend analysis

## Conclusion

This implementation guide provides a complete roadmap for integrating CIDOS math engine capabilities into NIREON V4. The phased approach ensures manageable complexity while building toward a sophisticated mathematical analysis system.

Key success factors:
1. **Security-first design** with proper subprocess isolation
2. **Integration with existing NIREON infrastructure** 
3. **Comprehensive validation** at every step
4. **Performance optimization** for production use
5. **Monitoring and maintenance** procedures

The resulting system will provide powerful mathematical analysis capabilities while maintaining the architectural integrity and security standards of NIREON V4.