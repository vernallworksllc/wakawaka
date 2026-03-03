"""
Wakawaka Gradual Type Checker
Walks the AST and performs type inference with warnings.
Unknown types default to T_ANY (compatible with everything).
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict

from .ast_nodes import (
    Program, Block, Node,
    IntLit, FloatLit, BoolLit, StrLit, FStrLit, NoneLit,
    ArrayLit, MapLit, TupleLit,
    Ident, MemberExpr, IndexExpr, BinOp, UnaryOp,
    CallExpr, Closure, TernaryExpr, CastExpr,
    PipeExpr, SomeExpr, OkExpr, ErrExpr,
    RangeExpr, NullCoalesceExpr,
    LetDecl, ConstDecl, Assign, ExprStmt,
    ReturnStmt, BreakStmt, ContinueStmt,
    IfStmt, WhileStmt, DoWhileStmt, ForInStmt, ForCStmt,
    MatchStmt, FnDecl, Param, ClassDecl, StructDecl,
    EnumDecl, ImportDecl, ModuleDecl, MacroDecl,
    AssertStmt, PanicStmt,
)


# ── Types ─────────────────────────────────────────────────────────────────────

@dataclass
class WkType:
    name: str
    params: List['WkType'] = field(default_factory=list)  # e.g. list<int>

    def compatible_with(self, other: 'WkType') -> bool:
        if self.name == 'any' or other.name == 'any':
            return True
        if self.name == other.name:
            if not self.params and not other.params:
                return True
            if len(self.params) == len(other.params):
                return all(a.compatible_with(b) for a, b in zip(self.params, other.params))
        return False

    def __repr__(self):
        if self.params:
            inner = ', '.join(str(p) for p in self.params)
            return f'{self.name}<{inner}>'
        return self.name

    def __eq__(self, other):
        if not isinstance(other, WkType):
            return False
        return self.name == other.name and self.params == other.params

    def __hash__(self):
        return hash((self.name, tuple(self.params)))


T_ANY    = WkType('any')
T_INT    = WkType('int')
T_FLOAT  = WkType('float')
T_STR    = WkType('str')
T_BOOL   = WkType('bool')
T_NONE   = WkType('none')
T_LIST   = WkType('list')
T_MAP    = WkType('map')
T_TENSOR = WkType('tensor')
T_FN     = WkType('fn')

_ANNOTATION_MAP = {
    'int': T_INT, 'i8': T_INT, 'i16': T_INT, 'i32': T_INT, 'i64': T_INT,
    'u8': T_INT, 'u16': T_INT, 'u32': T_INT, 'u64': T_INT,
    'float': T_FLOAT, 'f32': T_FLOAT, 'f64': T_FLOAT,
    'str': T_STR, 'string': T_STR,
    'bool': T_BOOL,
    'none': T_NONE, 'void': T_NONE,
    'list': T_LIST, 'array': T_LIST,
    'map': T_MAP,
    'tensor': T_TENSOR,
    'any': T_ANY,
}

_NUMERIC = {T_INT, T_FLOAT}


# ── Warning ───────────────────────────────────────────────────────────────────

@dataclass
class TypeWarning:
    message: str
    line: int = 0

    def __repr__(self):
        if self.line:
            return f'line {self.line}: {self.message}'
        return self.message


# ── Type Environment ──────────────────────────────────────────────────────────

class TypeEnv:
    def __init__(self, parent: Optional['TypeEnv'] = None):
        self.parent = parent
        self.vars: Dict[str, WkType] = {}

    def define(self, name: str, typ: WkType):
        self.vars[name] = typ

    def lookup(self, name: str) -> WkType:
        if name in self.vars:
            return self.vars[name]
        if self.parent:
            return self.parent.lookup(name)
        return T_ANY

    def child(self) -> 'TypeEnv':
        return TypeEnv(parent=self)


# ── Checker ───────────────────────────────────────────────────────────────────

class TypeChecker:
    def __init__(self):
        self.warnings: List[TypeWarning] = []
        self._fn_return_types: Dict[str, WkType] = {}

    def check(self, program: Program) -> List[TypeWarning]:
        env = TypeEnv()
        # Pre-define builtins
        builtins = [
            'println', 'print', 'eprintln', 'readln', 'len', 'str', 'int',
            'float', 'bool', 'typeof', 'type', 'isNone', 'isSome', 'isOk',
            'isErr', 'sum', 'min', 'max', 'map', 'filter', 'reduce',
            'sorted', 'reversed', 'any', 'all', 'zip', 'enumerate',
            'range', 'sleep', 'exit', 'assert', 'copy', 'chr', 'ord',
            'hash', 'repr', 'panic', 'math', 'tensor', 'ad', 'gpu',
            'pipeline', 'model',
        ]
        for b in builtins:
            env.define(b, T_FN if b not in ('math', 'tensor', 'ad', 'gpu', 'pipeline', 'model') else T_MAP)

        for stmt in program.stmts:
            self._check_stmt(stmt, env)
        return self.warnings

    def _warn(self, msg: str, line: int = 0):
        self.warnings.append(TypeWarning(msg, line))

    def _resolve_annotation(self, ann) -> WkType:
        if ann is None:
            return T_ANY
        if isinstance(ann, str):
            return _ANNOTATION_MAP.get(ann, T_ANY)
        return T_ANY

    # ── Statements ────────────────────────────────────────────────────────

    def _check_stmt(self, node, env: TypeEnv):
        if node is None:
            return
        t = type(node)

        if t is ExprStmt:
            self._infer(node.expr, env)

        elif t is LetDecl:
            ann_type = self._resolve_annotation(getattr(node, 'type_ann', None))
            if node.value is not None:
                val_type = self._infer(node.value, env)
                if ann_type != T_ANY and val_type != T_ANY:
                    if not ann_type.compatible_with(val_type):
                        self._warn(
                            f"Type mismatch: '{node.name}' declared as {ann_type} but assigned {val_type}",
                            getattr(node, 'line', 0))
                env.define(node.name, val_type if ann_type == T_ANY else ann_type)
            else:
                env.define(node.name, ann_type)

        elif t is ConstDecl:
            ann_type = self._resolve_annotation(getattr(node, 'type_ann', None))
            val_type = self._infer(node.value, env) if node.value else T_ANY
            if ann_type != T_ANY and val_type != T_ANY:
                if not ann_type.compatible_with(val_type):
                    self._warn(
                        f"Type mismatch: const '{node.name}' declared as {ann_type} but assigned {val_type}",
                        getattr(node, 'line', 0))
            env.define(node.name, val_type if ann_type == T_ANY else ann_type)

        elif t is Assign:
            self._infer(node.value, env)

        elif t is ReturnStmt:
            if node.value:
                self._infer(node.value, env)

        elif t is IfStmt:
            self._infer(node.cond, env)
            self._check_block(node.then, env)
            for cond, block in node.elseifs:
                self._infer(cond, env)
                self._check_block(block, env)
            if node.else_:
                self._check_block(node.else_, env)

        elif t is WhileStmt:
            self._infer(node.cond, env)
            self._check_block(node.body, env)

        elif t is DoWhileStmt:
            self._check_block(node.body, env)
            self._infer(node.cond, env)

        elif t is ForInStmt:
            iter_type = self._infer(node.iter, env)
            child = env.child()
            if iter_type == T_LIST:
                child.define(node.var, T_ANY)
            elif iter_type == T_STR:
                child.define(node.var, T_STR)
            else:
                child.define(node.var, T_ANY)
            self._check_block(node.body, child)

        elif t is ForCStmt:
            child = env.child()
            if node.init:
                self._check_stmt(node.init, child)
            if node.cond:
                self._infer(node.cond, child)
            if node.update:
                self._check_stmt(node.update, child)
            self._check_block(node.body, child)

        elif t is MatchStmt:
            self._infer(node.expr, env)
            for arm in node.arms:
                self._check_block(arm.body, env)

        elif t is FnDecl:
            child = env.child()
            for p in node.params:
                ptype = self._resolve_annotation(getattr(p, 'type_ann', None))
                child.define(p.name, ptype)
            ret_type = self._resolve_annotation(getattr(node, 'return_type', None))
            self._fn_return_types[node.name] = ret_type
            env.define(node.name, T_FN)
            self._check_block(node.body, child)

        elif t is ClassDecl:
            env.define(node.name, T_FN)  # constructor callable
            for method in getattr(node, 'methods', []):
                self._check_stmt(method, env)

        elif t is StructDecl:
            env.define(node.name, T_FN)

        elif t is EnumDecl:
            env.define(node.name, T_ANY)

        elif t is ImportDecl:
            alias = getattr(node, 'alias', None) or getattr(node, 'path', '')
            if alias:
                env.define(alias, T_MAP)

        elif t is ModuleDecl:
            env.define(node.name, T_MAP)
            self._check_block(node.body, env.child())

        elif t is MacroDecl:
            env.define(node.name, T_FN)

        elif t is Block:
            self._check_block(node, env)

        elif t in (AssertStmt, PanicStmt):
            if hasattr(node, 'cond') and node.cond:
                self._infer(node.cond, env)
            if hasattr(node, 'msg') and node.msg:
                self._infer(node.msg, env)

        elif t in (BreakStmt, ContinueStmt):
            pass  # nothing to check

    def _check_block(self, block, env: TypeEnv):
        if block is None:
            return
        child = env.child()
        stmts = block.stmts if hasattr(block, 'stmts') else []
        for stmt in stmts:
            self._check_stmt(stmt, child)

    # ── Expression Type Inference ─────────────────────────────────────────

    def _infer(self, node, env: TypeEnv) -> WkType:
        if node is None:
            return T_NONE
        t = type(node)

        if t is IntLit:    return T_INT
        if t is FloatLit:  return T_FLOAT
        if t is BoolLit:   return T_BOOL
        if t is StrLit:    return T_STR
        if t is FStrLit:   return T_STR
        if t is NoneLit:   return T_NONE
        if t is ArrayLit:  return T_LIST
        if t is MapLit:    return T_MAP
        if t is TupleLit:  return T_LIST

        if t is Ident:
            return env.lookup(node.name)

        if t is BinOp:
            left = self._infer(node.left, env)
            right = self._infer(node.right, env)
            op = node.op
            # Numeric ops
            if op in ('+', '-', '*', '/', '%', '**'):
                if left == T_STR or right == T_STR:
                    if op == '+':
                        return T_STR
                if left == T_TENSOR or right == T_TENSOR:
                    return T_TENSOR
                if left == T_FLOAT or right == T_FLOAT:
                    return T_FLOAT
                if left == T_INT and right == T_INT:
                    return T_INT
                return T_ANY
            # Comparison ops
            if op in ('==', '!=', '<', '>', '<=', '>=', '&&', '||', 'in'):
                return T_BOOL
            # Bitwise
            if op in ('&', '|', '^', '<<', '>>'):
                return T_INT
            return T_ANY

        if t is UnaryOp:
            operand = self._infer(node.operand, env)
            if node.op == '!':
                return T_BOOL
            if node.op == '-':
                return operand
            if node.op == '~':
                return T_INT
            return operand

        if t is CallExpr:
            self._infer(node.callee, env)
            for arg in node.args:
                self._infer(arg, env)
            # Check known function return types
            if isinstance(node.callee, Ident):
                name = node.callee.name
                if name in ('len', 'ord', 'hash'):
                    return T_INT
                if name in ('str', 'repr', 'chr', 'readln'):
                    return T_STR
                if name in ('int',):
                    return T_INT
                if name in ('float',):
                    return T_FLOAT
                if name in ('bool', 'isNone', 'isSome', 'isOk', 'isErr', 'any', 'all'):
                    return T_BOOL
                if name in ('typeof',):
                    return T_STR
                if name in self._fn_return_types:
                    return self._fn_return_types[name]
            return T_ANY

        if t is Closure:
            return T_FN

        if t is MemberExpr:
            obj_type = self._infer(node.obj, env)
            # Tensor method returns
            if obj_type == T_TENSOR:
                if node.member in ('sum', 'mean', 'max', 'min', 'item', 'dot'):
                    return T_FLOAT
                if node.member in ('argmax', 'argmin', 'ndim', 'size'):
                    return T_INT
                if node.member in ('reshape', 'transpose', 'flatten', 'abs', 'sqrt',
                                   'matmul', 'slice', 'add', 'sub', 'mul', 'div'):
                    return T_TENSOR
                if node.member == 'shape':
                    return T_LIST
                if node.member == 'data':
                    return T_LIST
            return T_ANY

        if t is IndexExpr:
            obj_type = self._infer(node.obj, env)
            self._infer(node.index, env)
            if obj_type == T_STR:
                return T_STR
            return T_ANY

        if t is TernaryExpr:
            self._infer(node.cond, env)
            t_type = self._infer(node.then_expr, env)
            e_type = self._infer(node.else_expr, env)
            if t_type == e_type:
                return t_type
            return T_ANY

        if t is CastExpr:
            self._infer(node.expr, env)
            return self._resolve_annotation(node.to_type)

        if t is SomeExpr:
            self._infer(node.value, env)
            return T_ANY

        if t is OkExpr:
            self._infer(node.value, env)
            return T_ANY

        if t is ErrExpr:
            self._infer(node.value, env)
            return T_ANY

        if t is RangeExpr:
            return T_LIST

        if t is NullCoalesceExpr:
            self._infer(node.left, env)
            return self._infer(node.right, env)

        if t is PipeExpr:
            self._infer(node.left, env)
            return self._infer(node.right, env)

        return T_ANY
