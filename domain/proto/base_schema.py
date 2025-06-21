from __future__ import annotations
from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Dict, Any, List, Optional, Literal, Union
from datetime import datetime
import ast
import re

class ProtoBlock(BaseModel):
    """
    Base Proto schema for all epistemic dialects. This is the fundamental
    declarative unit that encapsulates a task for the ProtoEngine.
    """
    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(default="proto/1.0", description="Proto schema version.")
    id: str = Field(..., description="Unique identifier for this Proto block (e.g., proto_math_123).")
    eidos: str = Field(..., description="Dialect identifier (math, graph, simulate, etc.).")
    description: str = Field(..., description="Human-readable description of the Proto block's purpose.")
    objective: str = Field(..., description="What this analysis or computation aims to achieve.")

    # Core execution parameters
    function_name: str = Field(..., description="Entry point function name within the provided code.")
    code: str = Field(..., description="The source code to be executed in the sandboxed environment.")
    inputs: Dict[str, Any] = Field(default_factory=dict, description="Input parameters passed to the entry point function.")

    # Execution constraints
    limits: Dict[str, Any] = Field(
        default_factory=lambda: {"timeout_sec": 10, "memory_mb": 256},
        description="Resource limits for the execution environment."
    )

    # Metadata
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Arbitrary metadata for context.")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional NIREON context for the task.")
    requirements: List[str] = Field(default_factory=list, description="Optional package requirements for the execution environment.")

    @field_validator('id')
    @classmethod
    def validate_id_format(cls, v: str) -> str:
        """Ensure ID follows a safe, conventional naming format."""
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError("ID must contain only alphanumeric characters, hyphens, and underscores.")
        return v

    @field_validator('code')
    @classmethod
    def validate_code_structure(cls, v: str, info: Any) -> str:
        """Use Abstract Syntax Trees (AST) for robust code validation."""
        function_name = info.data.get('function_name')
        if not function_name:
            # Can't validate function existence if name is not provided yet
            return v
            
        try:
            tree = ast.parse(v)
            function_found = any(
                isinstance(node, ast.FunctionDef) and node.name == function_name
                for node in ast.walk(tree)
            )
            if not function_found:
                raise ValueError(f"Code must define the entry point function '{function_name}'.")
        except SyntaxError as e:
            raise ValueError(f"Code contains syntax errors: {e}")
        
        return v

class ProtoMathBlock(ProtoBlock):
    """Specialized Proto block for the mathematical dialect."""
    eidos: Literal['math'] = 'math'
    
    # Math-specific metadata
    equation_latex: Optional[str] = Field(default=None, description="LaTeX representation of key equations.")
    
    # Override default limits for math
    limits: Dict[str, Any] = Field(default_factory=lambda: {
        "timeout_sec": 15, 
        "memory_mb": 512,
        "allowed_imports": ["matplotlib.pyplot", "numpy", "pandas", "scipy", "sympy", "math", "statistics"]
    })

class ProtoGraphBlock(ProtoBlock):
    """Specialized Proto block for the graph analysis dialect."""
    eidos: Literal['graph'] = 'graph'
    
    # Graph-specific metadata
    graph_format: Optional[str] = Field(default="networkx", description="Graph library format used (e.g., networkx, igraph).")
    
    limits: Dict[str, Any] = Field(default_factory=lambda: {
        "timeout_sec": 20,
        "memory_mb": 512,
        "allowed_imports": ["networkx", "numpy", "matplotlib.pyplot", "pandas"]
    })

# Union type for easy type hinting and validation
AnyProtoBlock = Union[ProtoMathBlock, ProtoGraphBlock, ProtoBlock]