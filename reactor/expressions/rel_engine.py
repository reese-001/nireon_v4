# C:\Users\erees\Documents\development\nireon_v4\reactor\expressions\rel_engine.py
from __future__ import annotations
import ast
import logging
import operator
from functools import lru_cache
from typing import Any, Callable, Dict, Final, Set

logger: Final = logging.getLogger(__name__)

__all__: list[str] = ['RELEngine']

_ALLOWED_BIN_OPS: Final[dict[type[ast.AST], Callable[[Any, Any], Any]]] = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.Mod: operator.mod, ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt, ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge, ast.And: lambda x, y: x and y, ast.Or: lambda x, y: x or y, ast.In: lambda x, y: x in y, ast.NotIn: lambda x, y: x not in y, ast.Is: operator.is_, ast.IsNot: operator.is_not}
_ALLOWED_UNARY: Final[dict[type[ast.AST], Callable[[Any], Any]]] = {ast.Not: operator.not_, ast.UAdd: operator.pos, ast.USub: operator.neg}

# FIX: Restore the 'lower' function to the list of allowed global functions.
_ALLOWED_FUNCS: Final[dict[str, Callable[..., Any]]] = {
    'len': len,
    'exists': lambda x: x is not None,
    'type': lambda x: type(x).__name__,
    'str': str,
    'int': int,
    'float': float,
    'bool': bool,
    'lower': lambda s: s.lower() if isinstance(s, str) else s,
}
# FIX: Add a whitelist for safe method calls (e.g., 'mystring'.startswith(...))
_ALLOWED_METHODS: Final[Set[str]] = {'lower', 'upper', 'startswith', 'endswith', 'strip', 'split'}


class _Validator(ast.NodeVisitor):
    __slots__ = ()
    def generic_visit(self, node: ast.AST) -> None:
        raise ValueError(f'Node {type(node).__name__} not permitted')
    visit_Name = visit_Constant = lambda self, *_: None
    def visit_Attribute(self, node: ast.Attribute) -> None:
        self.visit(node.value)
    def visit_BinOp(self, node: ast.BinOp) -> None:
        if type(node.op) not in _ALLOWED_BIN_OPS:
            raise ValueError(f'Operator {type(node.op).__name__} not permitted')
        self.visit(node.left)
        self.visit(node.right)
    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        for v in node.values:
            self.visit(v)
    def visit_UnaryOp(self, node: ast.UnaryOp) -> None:
        if type(node.op) not in _ALLOWED_UNARY:
            raise ValueError(f'Unary operator {type(node.op).__name__} not permitted')
        self.visit(node.operand)
    def visit_Compare(self, node: ast.Compare) -> None:
        self.visit(node.left)
        for op in node.ops:
            if type(op) not in _ALLOWED_BIN_OPS:
                raise ValueError(f'Comparator {type(op).__name__} not permitted')
        for comp in node.comparators:
            self.visit(comp)
    def visit_Call(self, node: ast.Call) -> None:
        func_ok = False
        if isinstance(node.func, ast.Name) and node.func.id in _ALLOWED_FUNCS:
            func_ok = True
        # FIX: Validate that method calls are in the whitelist.
        elif isinstance(node.func, ast.Attribute) and node.func.attr in _ALLOWED_METHODS:
            func_ok = True
            self.visit(node.func.value)
        if not func_ok:
            raise ValueError(f'Function {ast.unparse(node.func)} not permitted')
        for arg in node.args:
            self.visit(arg)

_VALIDATOR = _Validator()

@lru_cache(maxsize=2048)
def _compile(expr: str) -> ast.AST:
    cleaned_expr = " ".join(expr.strip().split())
    node = ast.parse(cleaned_expr, mode='eval').body
    _VALIDATOR.visit(node)
    return node

class RELEngine:
    __slots__ = ('_hits', '_misses')
    _ALLOWED_BIN_OPS = _ALLOWED_BIN_OPS
    _ALLOWED_UNARY = _ALLOWED_UNARY
    _ALLOWED_FUNCS = _ALLOWED_FUNCS

    def __init__(self) -> None:
        self._hits = 0
        self._misses = 0

    def evaluate(self, expression: str, context: Dict[str, Any]) -> Any:
        try:
            node = _compile(expression)
            if _compile.cache_info().hits > self._hits + self._misses:
                self._hits += 1
            else:
                self._misses += 1
            # Add allowed functions to the evaluation context
            eval_context = {**self._ALLOWED_FUNCS, **context}
            return self._eval(node, eval_context)
        except Exception as exc:
            logger.error(f'REL evaluation error for expression "{expression}": {exc}')
            raise ValueError(f'Invalid REL expression: {exc}') from exc

    def get_stats(self) -> Dict[str, Any]:
        total = self._hits + self._misses
        ci = _compile.cache_info()
        return {'instance_hits': self._hits, 'instance_misses': self._misses, 'instance_hit_rate': self._hits / total if total else 0.0, 'global_cache_size': ci.currsize, 'global_hits': ci.hits, 'global_misses': ci.misses}
    
    def _eval(self, node: ast.AST, ctx: Dict[str, Any]) -> Any:
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id in ctx:
                return ctx[node.id]
            raise ValueError(f'Variable or function {node.id!r} not found in context')
        if isinstance(node, ast.Attribute):
            obj = self._eval(node.value, ctx)
            return getattr(obj, node.attr, None)
        if isinstance(node, ast.BinOp):
            return _ALLOWED_BIN_OPS[type(node.op)](self._eval(node.left, ctx), self._eval(node.right, ctx))
        if isinstance(node, ast.UnaryOp):
            return _ALLOWED_UNARY[type(node.op)](self._eval(node.operand, ctx))
        if isinstance(node, ast.BoolOp):
            values = (self._eval(v, ctx) for v in node.values)
            return all(values) if isinstance(node.op, ast.And) else any(values)
        if isinstance(node, ast.Compare):
            left = self._eval(node.left, ctx)
            for op, comp in zip(node.ops, node.comparators):
                right = self._eval(comp, ctx)
                if not _ALLOWED_BIN_OPS[type(op)](left, right):
                    return False
                left = right
            return True
        if isinstance(node, ast.Call):
            func_to_call = self._eval(node.func, ctx)
            if not callable(func_to_call):
                raise TypeError(f"'{ast.unparse(node.func)}' is not callable")
            
            args = [self._eval(a, ctx) for a in node.args]
            return func_to_call(*args)

        raise ValueError(f'Unhandled node type {type(node).__name__}')