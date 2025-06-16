from __future__ import annotations
import ast
import operator
import logging
from functools import lru_cache
from typing import Any, Dict, Callable

logger = logging.getLogger(__name__)

class RELEngine:
    # FIX: Add `ast.Is` and `ast.IsNot` to the whitelist of allowed operators.
    # This allows for `is None` and `is not None` checks in the YAML rules.
    _ALLOWED_BIN_OPS: dict[type[ast.AST], Callable[[Any, Any], Any]] = {
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
        ast.In: lambda x, y: x in y,
        ast.NotIn: lambda x, y: x not in y,
        ast.Is: operator.is_,      # Added
        ast.IsNot: operator.is_not  # Added
    }

    _ALLOWED_UNARY: dict[type[ast.AST], Callable[[Any], Any]] = {
        ast.Not: operator.not_,
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }

    _ALLOWED_FUNCS: dict[str, Callable[..., Any]] = {
        'len': len,
        'exists': lambda x: x is not None,
        'type': lambda x: type(x).__name__,
        'str': str,
        'int': int,
        'float': float,
        'bool': bool,
    }

    def __init__(self) -> None:
        self._expression_cache: Dict[str, ast.AST] = {}
        self._hits = self._misses = 0

    def evaluate(self, expression: str, context: Dict[str, Any]) -> Any:
        try:
            parsed = self._expression_cache.get(expression)
            if parsed is None:
                parsed = self._compile(expression)
                self._expression_cache[expression] = parsed
                self._misses += 1
            else:
                self._hits += 1
            return self._eval_node(parsed, context)
        except Exception as exc:
            logger.error('REL evaluation error: %s', exc)
            raise ValueError(f'Invalid REL expression: {exc}') from exc

    def _compile(self, expression: str) -> ast.AST:
        try:
            tree = ast.parse(expression, mode='eval').body
            self._validate_ast(tree)
            return tree
        except SyntaxError as exc:
            raise ValueError(f'Syntax error: {exc}') from exc

    def _validate_ast(self, node: ast.AST) -> None:
        # This validation logic is recursive and checks every part of the parsed expression
        if isinstance(node, ast.BinOp):
            if type(node.op) not in self._ALLOWED_BIN_OPS:
                raise ValueError(f'Operator {type(node.op).__name__} not permitted')
            self._validate_ast(node.left)
            self._validate_ast(node.right)
        elif isinstance(node, ast.BoolOp):
            for value in node.values:
                self._validate_ast(value)
        elif isinstance(node, ast.UnaryOp):
            if type(node.op) not in self._ALLOWED_UNARY:
                raise ValueError(f'Unary operator {type(node.op).__name__} not permitted')
            self._validate_ast(node.operand)
        elif isinstance(node, ast.Compare):
            self._validate_ast(node.left)
            for op in node.ops:
                if type(op) not in self._ALLOWED_BIN_OPS:
                    raise ValueError(f'Comparator {type(op).__name__} not permitted')
            for comp in node.comparators:
                self._validate_ast(comp)
        elif isinstance(node, ast.Call):
            if not (isinstance(node.func, ast.Name) and node.func.id in self._ALLOWED_FUNCS):
                raise ValueError(f'Function {ast.unparse(node.func)} not permitted')
            for arg in node.args:
                self._validate_ast(arg)
        elif isinstance(node, (ast.Name, ast.Constant, ast.Attribute)):
            # These are safe leaf nodes
            return
        else:
            raise ValueError(f'Node {type(node).__name__} not permitted')

    def _eval_node(self, node: ast.AST, ctx: Dict[str, Any]) -> Any:
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id not in ctx:
                raise ValueError(f'Variable {node.id!r} not found')
            return ctx[node.id]
        if isinstance(node, ast.Attribute):
            base = self._eval_node(node.value, ctx)
            return getattr(base, node.attr, None)
        if isinstance(node, ast.BinOp):
            left, right = (self._eval_node(node.left, ctx), self._eval_node(node.right, ctx))
            return self._ALLOWED_BIN_OPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp):
            operand = self._eval_node(node.operand, ctx)
            return self._ALLOWED_UNARY[type(node.op)](operand)
        if isinstance(node, ast.BoolOp):
            results = (self._eval_node(v, ctx) for v in node.values)
            return all(results) if isinstance(node.op, ast.And) else any(results)
        if isinstance(node, ast.Compare):
            left = self._eval_node(node.left, ctx)
            for op, comparator in zip(node.ops, node.comparators):
                right = self._eval_node(comparator, ctx)
                if not self._ALLOWED_BIN_OPS[type(op)](left, right):
                    return False
                left = right
            return True
        if isinstance(node, ast.Call):
            func = self._ALLOWED_FUNCS[node.func.id]
            args = [self._eval_node(a, ctx) for a in node.args]
            return func(*args)

        raise ValueError(f'Unhandled node type {type(node).__name__}')

    def get_stats(self) -> Dict[str, Any]:
        total = self._hits + self._misses
        return {
            'cache_size': len(self._expression_cache),
            'hits': self._hits,
            'misses': self._misses,
            'hit_rate': self._hits / total if total else 0.0
        }