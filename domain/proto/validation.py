# nireon_v4\domain\proto\validation.py
from __future__ import annotations
import ast
import re
from typing import List, Set, Dict, Type, TYPE_CHECKING
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from .base_schema import ProtoBlock, ProtoMathBlock


class ProtoValidator(ABC):
    @abstractmethod
    def validate(self, proto: 'ProtoBlock') -> List[str]:
        pass


class SecurityValidator(ProtoValidator):
    BLOCKED_MODULES: Set[str] = {
        'os', 'subprocess', 'sys', 'shutil', 'socket', 'urllib', 'requests',
        'pickle', 'ctypes', 'multiprocessing', 'threading', 'http'
    }
    BLOCKED_BUILTINS: Set[str] = {'eval', 'exec', 'open', '__import__', 'compile'}

    def validate(self, proto: 'ProtoBlock') -> List[str]:
        errors = []
        if proto.limits.get('timeout_sec', 0) > 30:
            errors.append('Execution timeout cannot exceed 30 seconds.')
        if proto.limits.get('memory_mb', 0) > 1024:
            errors.append('Memory limit cannot exceed 1024 MB.')
        try:
            tree = ast.parse(proto.code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.split('.')[0] in self.BLOCKED_MODULES:
                            errors.append(f"Security violation: Disallowed import '{alias.name}'.")
                elif isinstance(node, ast.ImportFrom):
                    if node.module and node.module.split('.')[0] in self.BLOCKED_MODULES:
                        errors.append(f"Security violation: Disallowed import from '{node.module}'.")
                elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    if node.func.id in self.BLOCKED_BUILTINS:
                        errors.append(f"Security violation: Disallowed function call '{node.func.id}'.")
        except SyntaxError as e:
            errors.append(f'Syntax error in code: {e}')
        return errors


class MathProtoValidator(ProtoValidator):
    # FIX: Define the allowed imports as a constant within the validator itself.
    # This makes the validator the single source of truth for the 'math' dialect's security policy.
    ALLOWED_MATH_IMPORTS = {
        'matplotlib.pyplot', 'numpy', 'pandas', 'scipy', 'sympy', 'math', 'statistics'
    }

    def __init__(self):
        self.security_validator = SecurityValidator()
    
    def validate(self, proto: 'ProtoMathBlock') -> List[str]:
        errors = self.security_validator.validate(proto)
        tree = ast.parse(proto.code)
        
        if not any(isinstance(n, ast.FunctionDef) and n.name == proto.function_name for n in ast.walk(tree)):
            errors.append(f"Entry point function '{proto.function_name}' not found in code.")
        
        # FIX: Use the class-level constant as the set of allowed imports.
        # This ignores whatever the LLM might have put in proto.limits.
        allowed_imports = self.ALLOWED_MATH_IMPORTS
        used_imports = self._get_imports(tree)
        
        for imp in used_imports:
            # Check if the import is directly in the allowed list or is a sub-module of an allowed package
            # (e.g., allow `scipy.stats` if `scipy` is allowed).
            if imp not in allowed_imports and not any(imp.startswith(f'{allowed}.') for allowed in allowed_imports):
                errors.append(f"Import '{imp}' is not in the allowed list for the 'math' dialect.")
                
        return errors

    def _get_imports(self, tree: ast.AST) -> Set[str]:
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module)
        return imports


DIALECT_VALIDATORS: Dict[str, Type[ProtoValidator]] = {
    'math': MathProtoValidator,
}

def get_validator_for_dialect(eidos: str) -> ProtoValidator:
    validator_class = DIALECT_VALIDATORS.get(eidos)
    if not validator_class:
        # Default to the most basic security validator if the dialect is unknown
        return SecurityValidator()
    return validator_class()