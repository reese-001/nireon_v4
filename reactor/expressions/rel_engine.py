from __future__ import annotations
import ast
import operator
import logging
from typing import Any, Dict, Optional, Set
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

class RELEngine:
    """
    Rule Expression Language (REL) Engine for evaluating complex conditions.
    
    Supports:
    - Property access: signal.payload.trust_score
    - Comparisons: >, <, >=, <=, ==, !=
    - Logical operators: and, or, not
    - Arithmetic: +, -, *, /, %
    - Membership: in, not in
    - Functions: len(), exists(), type()
    """
    
    # Safe operators
    ALLOWED_OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Mod: operator.mod,
        ast.Eq: operator.eq,
        ast.NotEq: operator.ne,
        ast.Lt: operator.lt,
        ast.LtE: operator.le,
        ast.Gt: operator.gt,
        ast.GtE: operator.ge,
        ast.And: lambda x, y: x and y,
        ast.Or: lambda x, y: x or y,
        ast.Not: operator.not_,
        ast.In: lambda x, y: x in y,
        ast.NotIn: lambda x, y: x not in y,
    }
    
    # Safe built-in functions
    ALLOWED_FUNCTIONS = {
        'len': len,
        'exists': lambda x: x is not None,
        'type': lambda x: type(x).__name__,
        'str': str,
        'int': int,
        'float': float,
        'bool': bool,
    }
    
    def __init__(self):
        self._expression_cache: Dict[str, ast.AST] = {}
        self._cache_hits = 0
        self._cache_misses = 0
    
    def evaluate(self, expression: str, context: Dict[str, Any]) -> Any:
        """
        Evaluate a REL expression in the given context.
        
        Args:
            expression: The REL expression string
            context: Dictionary containing variables (e.g., {'signal': signal_obj, 'context': rule_context})
            
        Returns:
            The result of the expression evaluation
            
        Raises:
            ValueError: If the expression is invalid or unsafe
        """
        try:
            # Check cache
            if expression in self._expression_cache:
                self._cache_hits += 1
                parsed = self._expression_cache[expression]
            else:
                self._cache_misses += 1
                parsed = self._parse_expression(expression)
                self._expression_cache[expression] = parsed
            
            # Evaluate
            return self._evaluate_node(parsed, context)
            
        except Exception as e:
            logger.error(f"Error evaluating REL expression '{expression}': {e}")
            raise ValueError(f"Invalid REL expression: {e}") from e
    
    def _parse_expression(self, expression: str) -> ast.AST:
        """Parse and validate an expression."""
        try:
            tree = ast.parse(expression, mode='eval')
            self._validate_ast(tree.body)
            return tree.body
        except SyntaxError as e:
            raise ValueError(f"Syntax error in expression: {e}")
    
    def _validate_ast(self, node: ast.AST) -> None:
        """Validate that the AST only contains safe operations."""
        if isinstance(node, ast.BoolOp):
            for value in node.values:
                self._validate_ast(value)
        elif isinstance(node, ast.BinOp):
            if type(node.op) not in self.ALLOWED_OPERATORS:
                raise ValueError(f"Operator {type(node.op).__name__} not allowed")
            self._validate_ast(node.left)
            self._validate_ast(node.right)
        elif isinstance(node, ast.UnaryOp):
            if type(node.op) not in self.ALLOWED_OPERATORS:
                raise ValueError(f"Operator {type(node.op).__name__} not allowed")
            self._validate_ast(node.operand)
        elif isinstance(node, ast.Compare):
            self._validate_ast(node.left)
            for comp in node.comparators:
                self._validate_ast(comp)
            for op in node.ops:
                if type(op) not in self.ALLOWED_OPERATORS:
                    raise ValueError(f"Comparison operator {type(op).__name__} not allowed")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id not in self.ALLOWED_FUNCTIONS:
                raise ValueError(f"Function {node.func.id} not allowed")
            for arg in node.args:
                self._validate_ast(arg)
        elif isinstance(node, ast.Attribute):
            self._validate_ast(node.value)
        elif isinstance(node, (ast.Name, ast.Constant, ast.Num, ast.Str)):
            pass  # These are safe
        else:
            raise ValueError(f"AST node type {type(node).__name__} not allowed")
    
    def _evaluate_node(self, node: ast.AST, context: Dict[str, Any]) -> Any:
        """Recursively evaluate an AST node."""
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Name):
            if node.id not in context:
                raise ValueError(f"Variable '{node.id}' not found in context")
            return context[node.id]
        elif isinstance(node, ast.Attribute):
            obj = self._evaluate_node(node.value, context)
            return getattr(obj, node.attr, None)
        elif isinstance(node, ast.BinOp):
            left = self._evaluate_node(node.left, context)
            right = self._evaluate_node(node.right, context)
            return self.ALLOWED_OPERATORS[type(node.op)](left, right)
        elif isinstance(node, ast.UnaryOp):
            operand = self._evaluate_node(node.operand, context)
            return self.ALLOWED_OPERATORS[type(node.op)](operand)
        elif isinstance(node, ast.Compare):
            left = self._evaluate_node(node.left, context)
            for op, comp in zip(node.ops, node.comparators):
                right = self._evaluate_node(comp, context)
                if not self.ALLOWED_OPERATORS[type(op)](left, right):
                    return False
                left = right
            return True
        elif isinstance(node, ast.BoolOp):
            if isinstance(node.op, ast.And):
                for value in node.values:
                    if not self._evaluate_node(value, context):
                        return False
                return True
            else:  # ast.Or
                for value in node.values:
                    if self._evaluate_node(value, context):
                        return True
                return False
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                func = self.ALLOWED_FUNCTIONS[node.func.id]
                args = [self._evaluate_node(arg, context) for arg in node.args]
                return func(*args)
        else:
            raise ValueError(f"Cannot evaluate node type: {type(node).__name__}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._cache_hits + self._cache_misses
        return {
            'cache_size': len(self._expression_cache),
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'hit_rate': self._cache_hits / total if total > 0 else 0.0
        }