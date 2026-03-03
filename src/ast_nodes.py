"""
Deltoo AST Node Definitions
All nodes use dataclasses for clean, inspectable trees.
"""
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple


# ── Base ──────────────────────────────────────────────────────────────────────

@dataclass
class Node:
    line: int = 0
    col: int = 0


# ── Literals ──────────────────────────────────────────────────────────────────

@dataclass
class IntLit(Node):
    value: int = 0

@dataclass
class FloatLit(Node):
    value: float = 0.0

@dataclass
class BoolLit(Node):
    value: bool = False

@dataclass
class StrLit(Node):
    value: str = ""

@dataclass
class FStrLit(Node):
    """f"Hello, {name}!" — parts alternate str and Expr"""
    parts: List[Any] = field(default_factory=list)  # str | Expr

@dataclass
class NoneLit(Node):
    pass

@dataclass
class ArrayLit(Node):
    elements: List[Any] = field(default_factory=list)

@dataclass
class MapLit(Node):
    pairs: List[Tuple[Any, Any]] = field(default_factory=list)

@dataclass
class TupleLit(Node):
    elements: List[Any] = field(default_factory=list)

@dataclass
class ShellExpr(Node):
    """$`ls -la {path}`"""
    parts: List[Any] = field(default_factory=list)  # str | Expr

@dataclass
class SqlExpr(Node):
    """@sql`SELECT * FROM users WHERE id = {uid}`"""
    parts: List[Any] = field(default_factory=list)  # str | Expr
    result_type: Optional[str] = None  # e.g. "User"

@dataclass
class PyBlock(Node):
    """pyblock { ... python code ... }"""
    code: str = ""


# ── Identifiers & Access ──────────────────────────────────────────────────────

@dataclass
class Ident(Node):
    name: str = ""

@dataclass
class MemberExpr(Node):
    obj: Any = None
    member: str = ""

@dataclass
class IndexExpr(Node):
    obj: Any = None
    index: Any = None

@dataclass
class SliceExpr(Node):
    obj: Any = None
    start: Optional[Any] = None
    end: Optional[Any] = None
    step: Optional[Any] = None


# ── Expressions ───────────────────────────────────────────────────────────────

@dataclass
class BinOp(Node):
    op: str = ""
    left: Any = None
    right: Any = None

@dataclass
class UnaryOp(Node):
    op: str = ""
    operand: Any = None

@dataclass
class CallExpr(Node):
    callee: Any = None
    args: List[Any] = field(default_factory=list)
    kwargs: List[Tuple[str, Any]] = field(default_factory=list)  # named args

@dataclass
class Closure(Node):
    params: List[str] = field(default_factory=list)
    body: Any = None  # Expr or Block

@dataclass
class TernaryExpr(Node):
    cond: Any = None
    then: Any = None
    else_: Any = None

@dataclass
class CastExpr(Node):
    expr: Any = None
    to_type: str = ""

@dataclass
class PipeExpr(Node):
    """expr |> fn"""
    left: Any = None
    right: Any = None

@dataclass
class SomeExpr(Node):
    value: Any = None

@dataclass
class OkExpr(Node):
    value: Any = None

@dataclass
class ErrExpr(Node):
    value: Any = None

@dataclass
class PropagateExpr(Node):
    """expr? — propagate Result/Option errors"""
    expr: Any = None

@dataclass
class AwaitExpr(Node):
    expr: Any = None

@dataclass
class RefExpr(Node):
    """&x"""
    expr: Any = None

@dataclass
class DerefExpr(Node):
    """*x"""
    expr: Any = None

@dataclass
class SizeofExpr(Node):
    type_name: str = ""

@dataclass
class StructLit(Node):
    """Point { x: 1.0, y: 2.0 }"""
    name: str = ""
    fields: List[Tuple[str, Any]] = field(default_factory=list)

@dataclass
class RangeExpr(Node):
    start: Any = None
    end: Any = None
    inclusive: bool = False  # ..= vs ..

@dataclass
class OptChainExpr(Node):
    """obj?.member — returns none if obj is none, else obj.member"""
    obj: Any = None
    member: str = ""

@dataclass
class OptIndexExpr(Node):
    """obj?[idx] — returns none if obj is none, else obj[idx]"""
    obj: Any = None
    index: Any = None

@dataclass
class NullCoalesceExpr(Node):
    """left ?? right — returns left if not none, else right"""
    left: Any = None
    right: Any = None

@dataclass
class ComptimeExpr(Node):
    """comptime expr — evaluated at compile/parse time"""
    expr: Any = None


# ── Type Annotations ──────────────────────────────────────────────────────────

@dataclass
class TypeName(Node):
    name: str = ""
    params: List[Any] = field(default_factory=list)  # generics
    nullable: bool = False  # ?T
    pointer: bool = False   # *T
    ref_: bool = False      # &T
    array: bool = False     # []T
    array_size: Optional[int] = None  # [N]T


# ── Statements ────────────────────────────────────────────────────────────────

@dataclass
class Block(Node):
    stmts: List[Any] = field(default_factory=list)

@dataclass
class LetDecl(Node):
    name: str = ""
    type_ann: Optional[TypeName] = None
    value: Optional[Any] = None
    mutable: bool = False  # var vs let

@dataclass
class ConstDecl(Node):
    name: str = ""
    type_ann: Optional[TypeName] = None
    value: Any = None

@dataclass
class Assign(Node):
    target: Any = None
    op: str = "="
    value: Any = None

@dataclass
class ExprStmt(Node):
    expr: Any = None

@dataclass
class ReturnStmt(Node):
    value: Optional[Any] = None

@dataclass
class BreakStmt(Node):
    label: Optional[str] = None

@dataclass
class ContinueStmt(Node):
    label: Optional[str] = None

@dataclass
class DeferStmt(Node):
    expr: Any = None

@dataclass
class GoStmt(Node):
    expr: Any = None

@dataclass
class AssertStmt(Node):
    cond: Any = None
    msg: Optional[Any] = None

@dataclass
class PanicStmt(Node):
    msg: Any = None

@dataclass
class IfStmt(Node):
    cond: Any = None
    then: Block = field(default_factory=Block)
    elseifs: List[Tuple[Any, Block]] = field(default_factory=list)
    else_: Optional[Block] = None

@dataclass
class WhileStmt(Node):
    cond: Any = None
    body: Block = field(default_factory=Block)
    label: Optional[str] = None

@dataclass
class DoWhileStmt(Node):
    body: Block = field(default_factory=Block)
    cond: Any = None

@dataclass
class ForInStmt(Node):
    var: str = ""
    iter: Any = None
    body: Block = field(default_factory=Block)
    label: Optional[str] = None

@dataclass
class ForCStmt(Node):
    """for (let i=0; i<n; i+=1) {}"""
    init: Optional[Any] = None
    cond: Optional[Any] = None
    step: Optional[Any] = None
    body: Block = field(default_factory=Block)
    label: Optional[str] = None

@dataclass
class MatchArm(Node):
    pattern: Any = None  # Pattern
    guard: Optional[Any] = None
    body: Any = None     # Expr or Block

@dataclass
class MatchStmt(Node):
    expr: Any = None
    arms: List[MatchArm] = field(default_factory=list)

@dataclass
class UnsafeBlock(Node):
    body: Block = field(default_factory=Block)


# ── Patterns (for match) ──────────────────────────────────────────────────────

@dataclass
class WildcardPat(Node):
    pass

@dataclass
class LitPat(Node):
    value: Any = None

@dataclass
class IdentPat(Node):
    name: str = ""

@dataclass
class RangePat(Node):
    start: Any = None
    end: Any = None
    inclusive: bool = True

@dataclass
class TuplePat(Node):
    elements: List[Any] = field(default_factory=list)

@dataclass
class StructPat(Node):
    name: str = ""
    fields: List[Tuple[str, Any]] = field(default_factory=list)

@dataclass
class OkPat(Node):
    inner: Any = None

@dataclass
class ErrPat(Node):
    inner: Any = None

@dataclass
class SomePat(Node):
    inner: Any = None

@dataclass
class NonePat(Node):
    pass

@dataclass
class EnumPat(Node):
    name: str = ""
    inner: Optional[Any] = None

@dataclass
class OrPat(Node):
    left: Any = None
    right: Any = None


# ── Declarations ──────────────────────────────────────────────────────────────

@dataclass
class Param(Node):
    name: str = ""
    type_ann: Optional[TypeName] = None
    default: Optional[Any] = None
    variadic: bool = False  # ..rest

@dataclass
class FnDecl(Node):
    name: str = ""
    type_params: List[str] = field(default_factory=list)
    params: List[Param] = field(default_factory=list)
    return_type: Optional[TypeName] = None
    body: Optional[Block] = None
    is_async: bool = False
    is_abstract: bool = False
    is_override: bool = False
    decorators: List[str] = field(default_factory=list)
    is_pub: bool = True

@dataclass
class StructField(Node):
    name: str = ""
    type_ann: TypeName = field(default_factory=TypeName)
    default: Optional[Any] = None

@dataclass
class StructDecl(Node):
    name: str = ""
    fields: List[StructField] = field(default_factory=list)
    type_params: List[str] = field(default_factory=list)

@dataclass
class ImplBlock(Node):
    target: str = ""
    interface: Optional[str] = None
    methods: List[FnDecl] = field(default_factory=list)

@dataclass
class ClassDecl(Node):
    name: str = ""
    parent: Optional[str] = None
    interfaces: List[str] = field(default_factory=list)
    type_params: List[str] = field(default_factory=list)
    fields: List[StructField] = field(default_factory=list)
    methods: List[FnDecl] = field(default_factory=list)
    is_abstract: bool = False
    decorators: List[str] = field(default_factory=list)

@dataclass
class InterfaceMethod(Node):
    name: str = ""
    params: List[Param] = field(default_factory=list)
    return_type: Optional[TypeName] = None
    has_default: bool = False
    body: Optional[Block] = None

@dataclass
class InterfaceDecl(Node):
    name: str = ""
    type_params: List[str] = field(default_factory=list)
    methods: List[InterfaceMethod] = field(default_factory=list)
    extends: List[str] = field(default_factory=list)

@dataclass
class EnumVariant(Node):
    name: str = ""
    fields: List[TypeName] = field(default_factory=list)

@dataclass
class EnumDecl(Node):
    name: str = ""
    variants: List[EnumVariant] = field(default_factory=list)
    type_params: List[str] = field(default_factory=list)

@dataclass
class ImportDecl(Node):
    path: str = ""
    alias: Optional[str] = None
    lang: str = ""  # "python", "js", "c", "cpp", "java", "swift", or "" for Deltoo

@dataclass
class ModuleDecl(Node):
    name: str = ""
    body: Block = field(default_factory=Block)

@dataclass
class MacroDecl(Node):
    name: str = ""
    params: List[str] = field(default_factory=list)
    body: Block = field(default_factory=Block)

@dataclass
class ActorDecl(Node):
    """actor class MyActor { ... } — actor with message mailbox"""
    name: str = ""
    type_params: List[str] = field(default_factory=list)
    fields: List[StructField] = field(default_factory=list)
    methods: List[FnDecl] = field(default_factory=list)
    parent: Optional[str] = None

@dataclass
class SpawnExpr(Node):
    """spawn MyActor(args) — create and start an actor"""
    actor_class: Any = None
    args: List[Any] = field(default_factory=list)
    kwargs: List[Tuple[str, Any]] = field(default_factory=list)

@dataclass
class ReceiveStmt(Node):
    """receive { pattern => body, ... } — receive from actor mailbox"""
    arms: List[MatchArm] = field(default_factory=list)
    timeout: Optional[Any] = None  # optional timeout expr


# ── Top-level Program ─────────────────────────────────────────────────────────

@dataclass
class Program(Node):
    stmts: List[Any] = field(default_factory=list)
