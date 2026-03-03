"""
Wakawaka Source Formatter
Walks the AST and emits properly formatted Wakawaka source code.
Note: Comments are not preserved (they are not in the AST).
"""
from .ast_nodes import *


class Formatter:
    def __init__(self, indent_size=4):
        self._indent_size = indent_size
        self._indent = 0

    def format(self, program: Program) -> str:
        lines = []
        for stmt in program.stmts:
            lines.append(self._fmt_stmt(stmt))
        return '\n'.join(lines) + '\n'

    # ── Indentation helpers ──────────────────────────────────────────────────

    def _ind(self) -> str:
        return ' ' * (self._indent * self._indent_size)

    def _indent_block(self, block):
        self._indent += 1
        lines = []
        for s in (block.stmts if isinstance(block, Block) else [block]):
            lines.append(self._fmt_stmt(s))
        self._indent -= 1
        return '\n'.join(lines)

    # ── Statements ───────────────────────────────────────────────────────────

    def _fmt_stmt(self, node) -> str:
        t = type(node)
        if t is LetDecl:      return self._fmt_let(node)
        if t is ConstDecl:    return self._fmt_const(node)
        if t is Assign:       return self._fmt_assign(node)
        if t is ExprStmt:     return f'{self._ind()}{self._fmt_expr(node.expr)};'
        if t is ReturnStmt:   return self._fmt_return(node)
        if t is BreakStmt:
            lbl = f' {node.label}' if node.label else ''
            return f'{self._ind()}break{lbl};'
        if t is ContinueStmt:
            lbl = f' {node.label}' if node.label else ''
            return f'{self._ind()}continue{lbl};'
        if t is DeferStmt:    return f'{self._ind()}defer {self._fmt_expr(node.expr)};'
        if t is GoStmt:       return f'{self._ind()}go {self._fmt_expr(node.expr)};'
        if t is AssertStmt:   return self._fmt_assert(node)
        if t is PanicStmt:    return f'{self._ind()}panic({self._fmt_expr(node.msg)});'
        if t is IfStmt:       return self._fmt_if(node)
        if t is WhileStmt:    return self._fmt_while(node)
        if t is DoWhileStmt:  return self._fmt_do_while(node)
        if t is ForInStmt:    return self._fmt_for_in(node)
        if t is ForCStmt:     return self._fmt_for_c(node)
        if t is MatchStmt:    return self._fmt_match_stmt(node)
        if t is FnDecl:       return self._fmt_fn(node)
        if t is ClassDecl:    return self._fmt_class(node)
        if t is StructDecl:   return self._fmt_struct(node)
        if t is ImplBlock:    return self._fmt_impl(node)
        if t is EnumDecl:     return self._fmt_enum(node)
        if t is InterfaceDecl: return self._fmt_interface(node)
        if t is ImportDecl:   return self._fmt_import(node)
        if t is ModuleDecl:   return self._fmt_module(node)
        if t is MacroDecl:    return self._fmt_macro(node)
        if t is ActorDecl:    return self._fmt_actor(node)
        if t is Block:        return self._fmt_block_inline(node)
        if t is UnsafeBlock:  return self._fmt_unsafe(node)
        return f'{self._ind()}/* unknown stmt */'

    def _fmt_let(self, node: LetDecl) -> str:
        kw = 'var' if node.mutable else 'let'
        s = f'{self._ind()}{kw} {node.name}'
        if node.type_ann:
            s += f': {self._fmt_type(node.type_ann)}'
        if node.value is not None:
            s += f' = {self._fmt_expr(node.value)}'
        return s + ';'

    def _fmt_const(self, node: ConstDecl) -> str:
        s = f'{self._ind()}const {node.name}'
        if node.type_ann:
            s += f': {self._fmt_type(node.type_ann)}'
        s += f' = {self._fmt_expr(node.value)}'
        return s + ';'

    def _fmt_assign(self, node: Assign) -> str:
        target = self._fmt_expr(node.target)
        return f'{self._ind()}{target} {node.op} {self._fmt_expr(node.value)};'

    def _fmt_return(self, node: ReturnStmt) -> str:
        if node.value is not None:
            return f'{self._ind()}return {self._fmt_expr(node.value)};'
        return f'{self._ind()}return;'

    def _fmt_assert(self, node: AssertStmt) -> str:
        s = f'{self._ind()}assert({self._fmt_expr(node.cond)}'
        if node.msg:
            s += f', {self._fmt_expr(node.msg)}'
        return s + ');'

    def _fmt_if(self, node: IfStmt) -> str:
        lines = [f'{self._ind()}if {self._fmt_expr(node.cond)} {{']
        lines.append(self._indent_block(node.then))
        for cond, body in node.elseifs:
            lines.append(f'{self._ind()}}} else if {self._fmt_expr(cond)} {{')
            lines.append(self._indent_block(body))
        if node.else_:
            lines.append(f'{self._ind()}}} else {{')
            lines.append(self._indent_block(node.else_))
        lines.append(f'{self._ind()}}}')
        return '\n'.join(lines)

    def _fmt_while(self, node: WhileStmt) -> str:
        label = f'{self._ind()}{node.label}:\n' if node.label else ''
        lines = [f'{label}{self._ind()}while {self._fmt_expr(node.cond)} {{']
        lines.append(self._indent_block(node.body))
        lines.append(f'{self._ind()}}}')
        return '\n'.join(lines)

    def _fmt_do_while(self, node: DoWhileStmt) -> str:
        lines = [f'{self._ind()}do {{']
        lines.append(self._indent_block(node.body))
        lines.append(f'{self._ind()}}} while {self._fmt_expr(node.cond)};')
        return '\n'.join(lines)

    def _fmt_for_in(self, node: ForInStmt) -> str:
        label = f'{self._ind()}{node.label}:\n' if node.label else ''
        lines = [f'{label}{self._ind()}for {node.var} in {self._fmt_expr(node.iter)} {{']
        lines.append(self._indent_block(node.body))
        lines.append(f'{self._ind()}}}')
        return '\n'.join(lines)

    def _fmt_for_c(self, node: ForCStmt) -> str:
        label = f'{self._ind()}{node.label}:\n' if node.label else ''
        init = self._fmt_stmt(node.init).strip().rstrip(';') if node.init else ''
        cond = self._fmt_expr(node.cond) if node.cond else ''
        step = self._fmt_stmt(node.step).strip().rstrip(';') if node.step else ''
        lines = [f'{label}{self._ind()}for ({init}; {cond}; {step}) {{']
        lines.append(self._indent_block(node.body))
        lines.append(f'{self._ind()}}}')
        return '\n'.join(lines)

    def _fmt_match_stmt(self, node: MatchStmt) -> str:
        lines = [f'{self._ind()}match {self._fmt_expr(node.expr)} {{']
        self._indent += 1
        for arm in node.arms:
            lines.append(self._fmt_match_arm(arm))
        self._indent -= 1
        lines.append(f'{self._ind()}}}')
        return '\n'.join(lines)

    def _fmt_match_arm(self, arm: MatchArm) -> str:
        pat = self._fmt_pattern(arm.pattern)
        guard = f' if {self._fmt_expr(arm.guard)}' if arm.guard else ''
        if isinstance(arm.body, Block):
            lines = [f'{self._ind()}{pat}{guard} => {{']
            lines.append(self._indent_block(arm.body))
            lines.append(f'{self._ind()}}},')
            return '\n'.join(lines)
        # Statement bodies (break, continue, return, etc.)
        if isinstance(arm.body, (BreakStmt, ContinueStmt, ReturnStmt,
                                  LetDecl, Assign, ExprStmt, DeferStmt, GoStmt)):
            body = self._fmt_stmt(arm.body).strip()
            return f'{self._ind()}{pat}{guard} => {body}'
        # Expression body
        body = self._fmt_expr(arm.body)
        return f'{self._ind()}{pat}{guard} => {body},'

    def _fmt_fn(self, node: FnDecl) -> str:
        decorators = ''
        for d in (node.decorators or []):
            decorators += f'{self._ind()}@{d}\n'
        async_ = 'async ' if node.is_async else ''
        abstract = 'abstract ' if node.is_abstract else ''
        override = 'override ' if node.is_override else ''
        tparams = ''
        if node.type_params:
            tparams = f'<{", ".join(node.type_params)}>'
        params = ', '.join(self._fmt_param(p) for p in node.params)
        ret = ''
        if node.return_type:
            ret = f' -> {self._fmt_type(node.return_type)}'
        header = f'{decorators}{self._ind()}{abstract}{override}{async_}fn {node.name}{tparams}({params}){ret}'
        if node.is_abstract or node.body is None:
            return header + ';'
        lines = [header + ' {']
        lines.append(self._indent_block(node.body))
        lines.append(f'{self._ind()}}}')
        return '\n'.join(lines)

    def _fmt_param(self, p) -> str:
        if isinstance(p, str):
            return p
        s = ''
        if p.variadic:
            s += '..'
        s += p.name
        if p.type_ann:
            s += f': {self._fmt_type(p.type_ann)}'
        if p.default is not None:
            s += f' = {self._fmt_expr(p.default)}'
        return s

    def _fmt_class(self, node: ClassDecl) -> str:
        abstract = 'abstract ' if node.is_abstract else ''
        tparams = f'<{", ".join(node.type_params)}>' if node.type_params else ''
        header = f'{self._ind()}{abstract}class {node.name}{tparams}'
        if node.parent:
            header += f' extends {node.parent}'
        if node.interfaces:
            header += f' implements {", ".join(node.interfaces)}'
        lines = [header + ' {']
        self._indent += 1
        for f in node.fields:
            lines.append(self._fmt_field(f, use_var=True) + ';')
        if node.fields and node.methods:
            lines.append('')
        for m in node.methods:
            lines.append(self._fmt_fn(m))
        self._indent -= 1
        lines.append(f'{self._ind()}}}')
        return '\n'.join(lines)

    def _fmt_field(self, f: StructField, use_var=False) -> str:
        kw = 'var ' if use_var else ''
        s = f'{self._ind()}{kw}{f.name}: {self._fmt_type(f.type_ann)}'
        if f.default is not None:
            s += f' = {self._fmt_expr(f.default)}'
        return s

    def _fmt_struct(self, node: StructDecl) -> str:
        tparams = f'<{", ".join(node.type_params)}>' if node.type_params else ''
        lines = [f'{self._ind()}struct {node.name}{tparams} {{']
        self._indent += 1
        for f in node.fields:
            lines.append(self._fmt_field(f))
        self._indent -= 1
        lines.append(f'{self._ind()}}}')
        return '\n'.join(lines)

    def _fmt_impl(self, node: ImplBlock) -> str:
        header = f'{self._ind()}impl {node.target}'
        if node.interface:
            header += f' for {node.interface}'
        lines = [header + ' {']
        self._indent += 1
        for m in node.methods:
            lines.append(self._fmt_fn(m))
        self._indent -= 1
        lines.append(f'{self._ind()}}}')
        return '\n'.join(lines)

    def _fmt_enum(self, node: EnumDecl) -> str:
        tparams = f'<{", ".join(node.type_params)}>' if node.type_params else ''
        lines = [f'{self._ind()}enum {node.name}{tparams} {{']
        self._indent += 1
        for v in node.variants:
            if v.fields:
                fields = ', '.join(self._fmt_type(f) for f in v.fields)
                lines.append(f'{self._ind()}{v.name}({fields}),')
            else:
                lines.append(f'{self._ind()}{v.name},')
        self._indent -= 1
        lines.append(f'{self._ind()}}}')
        return '\n'.join(lines)

    def _fmt_interface(self, node: InterfaceDecl) -> str:
        tparams = f'<{", ".join(node.type_params)}>' if node.type_params else ''
        ext = f' extends {", ".join(node.extends)}' if node.extends else ''
        lines = [f'{self._ind()}interface {node.name}{tparams}{ext} {{']
        self._indent += 1
        for m in node.methods:
            params = ', '.join(self._fmt_param(p) for p in m.params)
            ret = f' -> {self._fmt_type(m.return_type)}' if m.return_type else ''
            if m.has_default and m.body:
                lines.append(f'{self._ind()}fn {m.name}({params}){ret} {{')
                lines.append(self._indent_block(m.body))
                lines.append(f'{self._ind()}}}')
            else:
                lines.append(f'{self._ind()}fn {m.name}({params}){ret};')
        self._indent -= 1
        lines.append(f'{self._ind()}}}')
        return '\n'.join(lines)

    def _fmt_import(self, node: ImportDecl) -> str:
        lang = f'{node.lang} ' if node.lang else ''
        alias = f' as {node.alias}' if node.alias else ''
        return f'{self._ind()}import {lang}"{node.path}"{alias};'

    def _fmt_module(self, node: ModuleDecl) -> str:
        lines = [f'{self._ind()}module {node.name} {{']
        lines.append(self._indent_block(node.body))
        lines.append(f'{self._ind()}}}')
        return '\n'.join(lines)

    def _fmt_macro(self, node: MacroDecl) -> str:
        params = ', '.join(node.params)
        lines = [f'{self._ind()}macro {node.name}({params}) {{']
        lines.append(self._indent_block(node.body))
        lines.append(f'{self._ind()}}}')
        return '\n'.join(lines)

    def _fmt_actor(self, node: ActorDecl) -> str:
        tparams = f'<{", ".join(node.type_params)}>' if node.type_params else ''
        header = f'{self._ind()}actor {node.name}{tparams}'
        if node.parent:
            header += f' extends {node.parent}'
        lines = [header + ' {']
        self._indent += 1
        for f in node.fields:
            lines.append(self._fmt_field(f, use_var=True) + ';')
        if node.fields and node.methods:
            lines.append('')
        for m in node.methods:
            lines.append(self._fmt_fn(m))
        self._indent -= 1
        lines.append(f'{self._ind()}}}')
        return '\n'.join(lines)

    def _fmt_unsafe(self, node: UnsafeBlock) -> str:
        lines = [f'{self._ind()}unsafe {{']
        lines.append(self._indent_block(node.body))
        lines.append(f'{self._ind()}}}')
        return '\n'.join(lines)

    def _fmt_block_inline(self, node: Block) -> str:
        lines = [f'{self._ind()}{{']
        lines.append(self._indent_block(node))
        lines.append(f'{self._ind()}}}')
        return '\n'.join(lines)

    # ── Expressions ──────────────────────────────────────────────────────────

    def _fmt_expr(self, node) -> str:
        if node is None:
            return 'none'
        t = type(node)
        if t is IntLit:      return str(node.value)
        if t is FloatLit:    return str(node.value)
        if t is BoolLit:     return 'true' if node.value else 'false'
        if t is StrLit:      return f'"{self._escape_str(node.value)}"'
        if t is FStrLit:     return self._fmt_fstr(node)
        if t is NoneLit:     return 'none'
        if t is ArrayLit:    return self._fmt_array(node)
        if t is MapLit:      return self._fmt_map(node)
        if t is TupleLit:    return self._fmt_tuple(node)
        if t is StructLit:   return self._fmt_struct_lit(node)
        if t is Ident:       return node.name
        if t is MemberExpr:  return f'{self._fmt_expr(node.obj)}.{node.member}'
        if t is IndexExpr:   return f'{self._fmt_expr(node.obj)}[{self._fmt_expr(node.index)}]'
        if t is SliceExpr:   return self._fmt_slice(node)
        if t is BinOp:       return self._fmt_binop(node)
        if t is UnaryOp:     return self._fmt_unary(node)
        if t is CallExpr:    return self._fmt_call(node)
        if t is Closure:     return self._fmt_closure(node)
        if t is TernaryExpr: return f'{self._fmt_expr(node.cond)} ? {self._fmt_expr(node.then)} : {self._fmt_expr(node.else_)}'
        if t is CastExpr:    return f'{self._fmt_type(node.to_type)}({self._fmt_expr(node.expr)})'
        if t is PipeExpr:    return f'{self._fmt_expr(node.left)} |> {self._fmt_expr(node.right)}'
        if t is SomeExpr:    return f'some({self._fmt_expr(node.value)})'
        if t is OkExpr:      return f'ok({self._fmt_expr(node.value)})'
        if t is ErrExpr:     return f'err({self._fmt_expr(node.value)})'
        if t is PropagateExpr: return f'{self._fmt_expr(node.expr)}?'
        if t is AwaitExpr:   return f'await {self._fmt_expr(node.expr)}'
        if t is RefExpr:     return f'&{self._fmt_expr(node.expr)}'
        if t is DerefExpr:   return f'*{self._fmt_expr(node.expr)}'
        if t is SizeofExpr:  return f'sizeof({node.type_name})'
        if t is RangeExpr:
            op = '..=' if node.inclusive else '..'
            return f'{self._fmt_expr(node.start)}{op}{self._fmt_expr(node.end)}'
        if t is OptChainExpr:  return f'{self._fmt_expr(node.obj)}?.{node.member}'
        if t is NullCoalesceExpr: return f'{self._fmt_expr(node.left)} ?? {self._fmt_expr(node.right)}'
        if t is ComptimeExpr: return f'comptime {self._fmt_expr(node.expr)}'
        if t is ShellExpr:   return self._fmt_shell(node)
        if t is SqlExpr:     return self._fmt_sql(node)
        if t is PyBlock:     return f'pyblock {{ {node.code} }}'
        if t is SpawnExpr:   return self._fmt_spawn(node)
        if t is MatchStmt:   return self._fmt_match_expr(node)
        return '/* unknown expr */'

    def _fmt_binop(self, node: BinOp) -> str:
        left = self._fmt_expr(node.left)
        right = self._fmt_expr(node.right)
        if node.op == '<-':
            return f'{left} <- {right}'
        return f'{left} {node.op} {right}'

    def _fmt_unary(self, node: UnaryOp) -> str:
        operand = self._fmt_expr(node.operand)
        if node.op == 'not':
            return f'not {operand}'
        return f'{node.op}{operand}'

    def _fmt_call(self, node: CallExpr) -> str:
        callee = self._fmt_expr(node.callee)
        args = [self._fmt_expr(a) for a in node.args]
        for name, val in (node.kwargs or []):
            args.append(f'{name}: {self._fmt_expr(val)}')
        return f'{callee}({", ".join(args)})'

    def _fmt_closure(self, node: Closure) -> str:
        params = ', '.join(p if isinstance(p, str) else self._fmt_param(p) for p in node.params)
        if isinstance(node.body, Block):
            lines = [f'|{params}| {{']
            self._indent += 1
            for s in node.body.stmts:
                lines.append(self._fmt_stmt(s))
            self._indent -= 1
            lines.append(f'{self._ind()}}}')
            return '\n'.join(lines)
        return f'|{params}| {self._fmt_expr(node.body)}'

    def _fmt_array(self, node: ArrayLit) -> str:
        elems = ', '.join(self._fmt_expr(e) for e in node.elements)
        return f'[{elems}]'

    def _fmt_map(self, node: MapLit) -> str:
        if not node.pairs:
            return '{}'
        pairs = ', '.join(f'{self._fmt_expr(k)}: {self._fmt_expr(v)}' for k, v in node.pairs)
        return f'{{{pairs}}}'

    def _fmt_tuple(self, node: TupleLit) -> str:
        elems = ', '.join(self._fmt_expr(e) for e in node.elements)
        return f'({elems})'

    def _fmt_struct_lit(self, node: StructLit) -> str:
        fields = ', '.join(f'{name}: {self._fmt_expr(val)}' for name, val in node.fields)
        return f'{node.name} {{ {fields} }}'

    def _fmt_slice(self, node: SliceExpr) -> str:
        obj = self._fmt_expr(node.obj)
        start = self._fmt_expr(node.start) if node.start else ''
        end = self._fmt_expr(node.end) if node.end else ''
        if node.step:
            step = self._fmt_expr(node.step)
            return f'{obj}[{start}:{end}:{step}]'
        return f'{obj}[{start}:{end}]'

    def _fmt_fstr(self, node: FStrLit) -> str:
        parts = []
        for p in node.parts:
            if isinstance(p, str):
                parts.append(self._escape_str(p))
            else:
                parts.append(f'{{{self._fmt_expr(p)}}}')
        return f'f"{"".join(parts)}"'

    def _fmt_shell(self, node: ShellExpr) -> str:
        parts = []
        for p in node.parts:
            if isinstance(p, str):
                parts.append(p)
            else:
                parts.append(f'{{{self._fmt_expr(p)}}}')
        return f'$`{"".join(parts)}`'

    def _fmt_sql(self, node: SqlExpr) -> str:
        parts = []
        for p in node.parts:
            if isinstance(p, str):
                parts.append(p)
            else:
                parts.append(f'{{{self._fmt_expr(p)}}}')
        return f'@sql`{"".join(parts)}`'

    def _fmt_spawn(self, node: SpawnExpr) -> str:
        args = [self._fmt_expr(a) for a in node.args]
        for name, val in (node.kwargs or []):
            args.append(f'{name}: {self._fmt_expr(val)}')
        return f'spawn {self._fmt_expr(node.actor_class)}({", ".join(args)})'

    def _fmt_match_expr(self, node: MatchStmt) -> str:
        lines = [f'match {self._fmt_expr(node.expr)} {{']
        self._indent += 1
        for arm in node.arms:
            lines.append(self._fmt_match_arm(arm))
        self._indent -= 1
        lines.append(f'{self._ind()}}}')
        return '\n'.join(lines)

    # ── Patterns ─────────────────────────────────────────────────────────────

    def _fmt_pattern(self, pat) -> str:
        t = type(pat)
        if t is WildcardPat: return '_'
        if t is LitPat:      return self._fmt_expr(pat.value) if isinstance(pat.value, Node) else repr(pat.value)
        if t is IdentPat:    return pat.name
        if t is RangePat:
            op = '..=' if pat.inclusive else '..'
            return f'{self._fmt_expr(pat.start)}{op}{self._fmt_expr(pat.end)}'
        if t is TuplePat:
            elems = ', '.join(self._fmt_pattern(e) for e in pat.elements)
            return f'({elems})'
        if t is StructPat:
            fields = ', '.join(f'{name}: {self._fmt_pattern(p)}' for name, p in pat.fields)
            return f'{pat.name} {{ {fields} }}'
        if t is OkPat:       return f'ok({self._fmt_pattern(pat.inner)})'
        if t is ErrPat:      return f'err({self._fmt_pattern(pat.inner)})'
        if t is SomePat:     return f'some({self._fmt_pattern(pat.inner)})'
        if t is NonePat:     return 'none'
        if t is EnumPat:
            if pat.inner:
                return f'{pat.name}({self._fmt_pattern(pat.inner)})'
            return pat.name
        if t is OrPat:
            return f'{self._fmt_pattern(pat.left)} | {self._fmt_pattern(pat.right)}'
        # Fallback: if pattern is a plain string (e.g. from match arms with string literals)
        if isinstance(pat, str):
            return f'"{self._escape_str(pat)}"'
        # If it's an expression node used as a pattern
        if isinstance(pat, Node):
            return self._fmt_expr(pat)
        return str(pat)

    # ── Types ────────────────────────────────────────────────────────────────

    def _fmt_type(self, t) -> str:
        if t is None:
            return 'any'
        if isinstance(t, str):
            return t
        if isinstance(t, TypeName):
            s = t.name
            if t.params:
                s += f'<{", ".join(self._fmt_type(p) for p in t.params)}>'
            if t.array:
                if t.array_size is not None:
                    s = f'[{t.array_size}]{s}'
                else:
                    s = f'[]{s}'
            if t.nullable:
                s = f'?{s}'
            if t.pointer:
                s = f'*{s}'
            if t.ref_:
                s = f'&{s}'
            return s
        return str(t)

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _escape_str(self, s: str) -> str:
        return (s.replace('\\', '\\\\')
                 .replace('"', '\\"')
                 .replace('\n', '\\n')
                 .replace('\r', '\\r')
                 .replace('\t', '\\t'))


def format_source(source: str, filename: str = "<stdin>", indent_size: int = 4) -> str:
    from .parser import parse
    program = parse(source, filename)
    return Formatter(indent_size=indent_size).format(program)
