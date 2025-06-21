from __future__ import annotations
import ast
import re
from typing import List, Set, Dict, Type, TYPE_CHECKING
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from .base_schema import ProtoBlock, ProtoMathBlock

class ProtoValidator(ABC):
    """Abstract base class for all Proto block validators."""
    
    @abstractmethod
    def validate(self, proto: ProtoBlock) -> List[str]:
        """
        Validates a Proto block.
        Returns:
            A list of error strings. An empty list indicates success.
        """
        pass

class SecurityValidator(ProtoValidator):
    """
    Universal security validator for all Proto dialects. This is the most
    critical validator, ensuring that no malicious code is executed.
    """
    
    # Security: A strict set of universally blocked modules and functions.
    BLOCKED_MODULES: Set[str] = {
        'os', 'subprocess', 'sys', 'shutil', 'socket', 'urllib', 'requests',
        'pickle', 'ctypes', 'multiprocessing', 'threading', 'http'
    }
    BLOCKED_BUILTINS: Set[str] = {
        'eval', 'exec', 'open', '__import__', 'compile'
    }

    def validate(self, proto: ProtoBlock) -> List[str]:
        """Perform security validation applicable to any Proto block."""
        errors = []
        
        # 1. Basic resource limits validation
        if proto.limits.get('timeout_sec', 0) > 30:
            errors.append("Execution timeout cannot exceed 30 seconds.")
        if proto.limits.get('memory_mb', 0) > 1024:
            errors.append("Memory limit cannot exceed 1024 MB.")
            
        # 2. AST-based security validation
        try:
            tree = ast.parse(proto.code)
            for node in ast.walk(tree):
                # Check for blocked imports (e.g., `import os`)
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.split('.')[0] in self.BLOCKED_MODULES:
                            errors.append(f"Security violation: Disallowed import '{alias.name}'.")
                # Check for blocked from-imports (e.g., `from os import system`)
                elif isinstance(node, ast.ImportFrom):
                    if node.module and node.module.split('.')[0] in self.BLOCKED_MODULES:
                        errors.append(f"Security violation: Disallowed import from '{node.module}'.")
                # Check for calls to blocked built-in functions
                elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    if node.func.id in self.BLOCKED_BUILTINS:
                        errors.append(f"Security violation: Disallowed function call '{node.func.id}'.")
        except SyntaxError as e:
            errors.append(f"Syntax error in code: {e}")

        return errors

class MathProtoValidator(ProtoValidator):
    """Validator specific to the 'math' dialect Proto blocks."""
    
    def __init__(self):
        self.security_validator = SecurityValidator()
    
    def validate(self, proto: ProtoMathBlock) -> List[str]:
        """Validate a math-dialect Proto block."""
        # Run universal security checks first.
        errors = self.security_validator.validate(proto)
        
        # Then, run math-specific checks.
        tree = ast.parse(proto.code)
        
        # Check if the declared entry point function exists.
        if not any(isinstance(n, ast.FunctionDef) and n.name == proto.function_name for n in ast.walk(tree)):
            errors.append(f"Entry point function '{proto.function_name}' not found in code.")
        
        # Check if imports are within the dialect's allowlist.
        allowed_imports = set(proto.limits.get('allowed_imports', []))
        used_imports = self._get_imports(tree)
        
        for imp in used_imports:
            if imp not in allowed_imports and not any(imp.startswith(f"{allowed}.") for allowed in allowed_imports):
                 errors.append(f"Import '{imp}' is not in the allowed list for the 'math' dialect.")

        return errors
    
    def _get_imports(self, tree: ast.AST) -> Set[str]:
        """Extract all imported module names from the code's AST."""
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module)
        return imports

# --- Validator Registry ---
DIALECT_VALIDATORS: Dict[str, Type[ProtoValidator]] = {
    'math': MathProtoValidator,
    # 'graph': GraphProtoValidator, # Future extension
}

def get_validator_for_dialect(eidos: str) -> ProtoValidator:
    """Factory function to get the appropriate validator for a Proto dialect."""
    validator_class = DIALECT_VALIDATORS.get(eidos)
    if not validator_class:
        # Secure by default: if the dialect is unknown, use the most restrictive
        # security validator and nothing else.
        return SecurityValidator()
    return validator_class()