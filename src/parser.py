"""
Deltoo Parser — Recursive descent parser producing an AST.
"""
from typing import List, Optional, Any
from .lexer import (
    Token, tokenize, LexError,
    TK_INT, TK_FLOAT, TK_STR, TK_FSTR, TK_BOOL, TK_NONE,
    TK_IDENT, TK_KEYWORD,
    TK_PLUS, TK_MINUS, TK_STAR, TK_SLASH, TK_PERCENT, TK_STARSTAR,
    TK_AMP, TK_PIPE, TK_CARET, TK_TILDE, TK_LSHIFT, TK_RSHIFT,
    TK_AMPAMP, TK_PIPEPIPE, TK_BANG,
    TK_EQ, TK_EQEQ, TK_BANGEQ,
    TK_LT, TK_GT, TK_LTEQ, TK_GTEQ,
    TK_PLUSEQ, TK_MINUSEQ, TK_STAREQ, TK_SLASHEQ, TK_PERCENTEQ,
    TK_STARSTAREQ, TK_AMPEQ, TK_PIPEEQ, TK_CARETEQ, TK_LSHIFTEQ, TK_RSHIFTEQ,
    TK_ARROW, TK_FATARROW, TK_PIPE_GT, TK_CHAN_RECV, TK_CHAN_SEND,
    TK_QUESTION, TK_QUESTION_DOT, TK_DOUBLEQUEST,
    TK_COLON, TK_DOUBLECOLON, TK_SEMICOLON, TK_COMMA, TK_DOT,
    TK_DOTDOT, TK_DOTDOTEQ, TK_AT,
    TK_LPAREN, TK_RPAREN, TK_LBRACE, TK_RBRACE, TK_LBRACKET, TK_RBRACKET,
    TK_SHELL, TK_SQL, TK_HASH, TK_EOF,
)
from .ast_nodes import *


class ParseError(Exception):
    def __init__(self, msg, line=0, col=0):
        super().__init__(f"[Parse Error] {msg} at line {line}, col {col}")
        self.line = line
        self.col = col


ASSIGN_OPS = {"=", "+=", "-=", "*=", "/=", "%=", "**=", "&=", "|=", "^=", "<<=", ">>="}

ASSIGN_TOKS = {
    TK_EQ, TK_PLUSEQ, TK_MINUSEQ, TK_STAREQ, TK_SLASHEQ, TK_PERCENTEQ,
    TK_STARSTAREQ, TK_AMPEQ, TK_PIPEEQ, TK_CARETEQ, TK_LSHIFTEQ, TK_RSHIFTEQ,
}


class Parser:
    def __init__(self, tokens: List[Token], source: str = ""):
        self.tokens = tokens
        self.pos = 0
        self.source = source

    # ── Token helpers ─────────────────────────────────────────────────────────

    def cur(self) -> Token:
        return self.tokens[self.pos]

    def peek(self, offset=1) -> Token:
        p = self.pos + offset
        return self.tokens[p] if p < len(self.tokens) else self.tokens[-1]

    def advance(self) -> Token:
        t = self.tokens[self.pos]
        if self.pos < len(self.tokens) - 1:
            self.pos += 1
        return t

    def check(self, kind: int, value=None) -> bool:
        t = self.cur()
        if t.kind != kind:
            return False
        return value is None or t.value == value

    def match(self, kind: int, value=None) -> Optional[Token]:
        if self.check(kind, value):
            return self.advance()
        return None

    def expect(self, kind: int, value=None) -> Token:
        t = self.match(kind, value)
        if t is None:
            cur = self.cur()
            expected = f"{value!r}" if value else f"token({kind})"
            raise ParseError(
                f"Expected {expected}, got {cur.value!r}",
                cur.line, cur.col
            )
        return t

    def kw(self, word: str) -> bool:
        return self.check(TK_KEYWORD, word)

    def expect_kw(self, word: str) -> Token:
        return self.expect(TK_KEYWORD, word)

    def match_kw(self, word: str) -> Optional[Token]:
        return self.match(TK_KEYWORD, word)

    def skip_semis(self):
        while self.match(TK_SEMICOLON):
            pass

    def loc(self) -> dict:
        t = self.cur()
        return {"line": t.line, "col": t.col}

    # ── Program ───────────────────────────────────────────────────────────────

    def parse_program(self) -> Program:
        stmts = []
        self.skip_semis()
        while not self.check(TK_EOF):
            stmts.append(self.parse_top_level())
            self.skip_semis()
        return Program(stmts=stmts)

    def parse_top_level(self) -> Any:
        decorators = self._parse_decorators()
        t = self.cur()

        if t.kind == TK_KEYWORD:
            kw = t.value
            if kw == "fn":       return self._parse_fn(decorators)
            if kw == "async":
                self.advance()
                fn = self._parse_fn(decorators)
                fn.is_async = True
                return fn
            if kw == "abstract": return self._parse_class(decorators)
            if kw == "class":    return self._parse_class(decorators)
            if kw == "actor":    return self._parse_actor(decorators)
            if kw == "struct":   return self._parse_struct()
            if kw == "impl":     return self._parse_impl()
            if kw == "interface": return self._parse_interface()
            if kw == "enum":     return self._parse_enum()
            if kw == "import":   return self._parse_import()
            if kw == "module":   return self._parse_module()
            if kw == "macro":    return self._parse_macro()
            if kw == "let":      return self._parse_let()
            if kw == "var":      return self._parse_let(mutable=True)
            if kw == "const":    return self._parse_const()
            if kw == "pub":
                self.advance()
                return self.parse_top_level()

        return self.parse_stmt()

    def _parse_decorators(self) -> List[str]:
        decorators = []
        while self.check(TK_AT):
            # Lexer combines @name into a single TK_AT token whose value is the name
            tok = self.advance()
            decorators.append(tok.value)
            self.skip_semis()
        return decorators

    # ── Statements ────────────────────────────────────────────────────────────

    def parse_stmt(self) -> Any:
        t = self.cur()

        # Decorators can appear before fn/class/async inside any block
        if t.kind == TK_AT:
            decorators = self._parse_decorators()
            t2 = self.cur()
            if t2.kind == TK_KEYWORD:
                kw2 = t2.value
                if kw2 == "fn":       return self._parse_fn(decorators)
                if kw2 == "async":
                    self.advance()
                    fn = self._parse_fn(decorators)
                    fn.is_async = True
                    return fn
                if kw2 in ("class", "abstract"): return self._parse_class(decorators)
                if kw2 == "actor":    return self._parse_actor(decorators)
            raise ParseError(f"Expected fn/class after decorator, got '{t2.value}'",
                             t2.line, t2.col)

        if t.kind == TK_KEYWORD:
            kw = t.value
            if kw == "let":      return self._parse_let()
            if kw == "var":      return self._parse_let(mutable=True)
            if kw == "const":    return self._parse_const()
            if kw == "fn":       return self._parse_fn()
            if kw == "class":    return self._parse_class()
            if kw == "if":       return self._parse_if()
            if kw == "while":    return self._parse_while()
            if kw == "do":       return self._parse_do_while()
            if kw == "for":      return self._parse_for()
            if kw == "return":   return self._parse_return()
            if kw == "break":    return self._parse_break()
            if kw == "continue": return self._parse_continue()
            if kw == "defer":    return self._parse_defer()
            if kw == "go":       return self._parse_go()
            if kw == "unsafe":   return self._parse_unsafe()
            if kw == "pyblock":  return self._parse_pyblock()
            if kw == "async":
                self.advance()
                fn = self._parse_fn()
                fn.is_async = True
                return fn
            if kw == "enum":     return self._parse_enum()
            if kw == "struct":   return self._parse_struct()
            if kw == "impl":     return self._parse_impl()
            if kw == "interface": return self._parse_interface()
            if kw == "abstract": return self._parse_class()
            if kw == "actor":    return self._parse_actor()
            if kw == "spawn":    return self._parse_spawn_stmt()
            if kw == "receive":  return self._parse_receive()
            if kw == "comptime":
                line, col = self.cur().line, self.cur().col
                self.advance()
                expr = self.parse_expr()
                return ExprStmt(expr=ComptimeExpr(expr=expr, line=line, col=col),
                                line=line, col=col)

        # Labeled loop: outer: for/while ...
        if (t.kind == TK_IDENT and
                self.pos + 1 < len(self.tokens) and
                self.tokens[self.pos + 1].kind == TK_COLON):
            label = t.value
            self.advance()  # ident
            self.advance()  # colon
            self.skip_semis()
            loop_stmt = self.parse_stmt()
            if hasattr(loop_stmt, 'label'):
                loop_stmt.label = label
            return loop_stmt

        # Expression statement (assignment, call, etc.)
        expr = self.parse_expr()
        # Check for assignment
        if self.cur().kind in ASSIGN_TOKS:
            op = self.advance().value
            value = self.parse_expr()
            self.match(TK_SEMICOLON)
            return Assign(target=expr, op=op, value=value,
                          line=expr.line, col=expr.col)
        self.match(TK_SEMICOLON)
        return ExprStmt(expr=expr, line=expr.line, col=expr.col)

    def _parse_block(self) -> Block:
        line, col = self.cur().line, self.cur().col
        self.expect(TK_LBRACE)
        stmts = []
        self.skip_semis()
        while not self.check(TK_RBRACE) and not self.check(TK_EOF):
            stmts.append(self.parse_stmt())
            self.skip_semis()
        self.expect(TK_RBRACE)
        return Block(stmts=stmts, line=line, col=col)

    def _parse_let(self, mutable=False) -> LetDecl:
        line, col = self.cur().line, self.cur().col
        self.advance()  # let/var
        name = self.expect(TK_IDENT).value
        type_ann = None
        if self.match(TK_COLON):
            type_ann = self._parse_type()
        value = None
        if self.match(TK_EQ):
            value = self.parse_expr()
        self.match(TK_SEMICOLON)
        return LetDecl(name=name, type_ann=type_ann, value=value,
                       mutable=mutable, line=line, col=col)

    def _parse_const(self) -> ConstDecl:
        line, col = self.cur().line, self.cur().col
        self.advance()  # const
        name = self.expect(TK_IDENT).value
        type_ann = None
        if self.match(TK_COLON):
            type_ann = self._parse_type()
        self.expect(TK_EQ)
        value = self.parse_expr()
        self.match(TK_SEMICOLON)
        return ConstDecl(name=name, type_ann=type_ann, value=value,
                         line=line, col=col)

    def _parse_if(self) -> IfStmt:
        line, col = self.cur().line, self.cur().col
        self.advance()  # if
        cond = self.parse_expr()
        then = self._parse_block()
        elseifs = []
        else_ = None
        while self.match_kw("else"):
            if self.kw("if"):
                self.advance()
                ec = self.parse_expr()
                eb = self._parse_block()
                elseifs.append((ec, eb))
            else:
                else_ = self._parse_block()
                break
        return IfStmt(cond=cond, then=then, elseifs=elseifs, else_=else_,
                      line=line, col=col)

    def _parse_while(self) -> WhileStmt:
        line, col = self.cur().line, self.cur().col
        self.advance()
        cond = self.parse_expr()
        body = self._parse_block()
        return WhileStmt(cond=cond, body=body, line=line, col=col)

    def _parse_do_while(self) -> DoWhileStmt:
        line, col = self.cur().line, self.cur().col
        self.advance()  # do
        body = self._parse_block()
        self.expect_kw("while")
        cond = self.parse_expr()
        self.match(TK_SEMICOLON)
        return DoWhileStmt(body=body, cond=cond, line=line, col=col)

    def _parse_for(self) -> Any:
        line, col = self.cur().line, self.cur().col
        self.advance()  # for

        # C-style: for (let i=0; i<n; i+=1) {}
        if self.match(TK_LPAREN):
            init = None
            if not self.check(TK_SEMICOLON):
                if self.kw("let") or self.kw("var"):
                    init = self._parse_let(mutable=self.kw("var"))
                else:
                    init = self.parse_stmt()
            else:
                self.advance()
            cond = None
            if not self.check(TK_SEMICOLON):
                cond = self.parse_expr()
            self.expect(TK_SEMICOLON)
            step = None
            if not self.check(TK_RPAREN):
                step_expr = self.parse_expr()
                if self.cur().kind in ASSIGN_TOKS:
                    op = self.advance().value
                    val = self.parse_expr()
                    step = Assign(target=step_expr, op=op, value=val,
                                  line=step_expr.line, col=step_expr.col)
                else:
                    step = step_expr
            self.expect(TK_RPAREN)
            body = self._parse_block()
            return ForCStmt(init=init, cond=cond, step=step, body=body,
                            line=line, col=col)

        # for x in collection {}  or  for i in 0..n {}
        var = self.expect(TK_IDENT).value
        self.expect_kw("in")
        iter_ = self.parse_expr()
        body = self._parse_block()
        return ForInStmt(var=var, iter=iter_, body=body, line=line, col=col)

    def _parse_match(self) -> MatchStmt:
        line, col = self.cur().line, self.cur().col
        self.advance()  # match
        expr = self.parse_expr()
        self.expect(TK_LBRACE)
        arms = []
        self.skip_semis()
        while not self.check(TK_RBRACE) and not self.check(TK_EOF):
            pattern = self._parse_pattern()
            guard = None
            if self.match_kw("if"):
                guard = self.parse_expr()
            self.expect(TK_FATARROW)
            if self.check(TK_LBRACE):
                body = self._parse_block()
            elif self.cur().kind == TK_KEYWORD and self.cur().value in (
                "break", "continue", "return", "panic"
            ):
                body = self.parse_stmt()
            else:
                body = self.parse_expr()
            arms.append(MatchArm(pattern=pattern, guard=guard, body=body,
                                 line=line, col=col))
            self.match(TK_COMMA)
            self.skip_semis()
        self.expect(TK_RBRACE)
        return MatchStmt(expr=expr, arms=arms, line=line, col=col)

    def _parse_pattern(self) -> Any:
        line, col = self.cur().line, self.cur().col
        pat = self._parse_pattern_atom()
        # OR pattern: pat | pat
        while self.check(TK_PIPE):
            self.advance()
            right = self._parse_pattern_atom()
            pat = OrPat(left=pat, right=right, line=line, col=col)
        return pat

    def _parse_pattern_atom(self) -> Any:
        line, col = self.cur().line, self.cur().col
        t = self.cur()

        if t.kind == TK_IDENT and t.value == "_":
            self.advance()
            return WildcardPat(line=line, col=col)

        if t.kind == TK_KEYWORD and t.value == "none":
            self.advance()
            return NonePat(line=line, col=col)

        if t.kind == TK_KEYWORD and t.value == "some":
            self.advance()
            self.expect(TK_LPAREN)
            inner = self._parse_pattern()
            self.expect(TK_RPAREN)
            return SomePat(inner=inner, line=line, col=col)

        if t.kind == TK_KEYWORD and t.value == "ok":
            self.advance()
            self.expect(TK_LPAREN)
            inner = self._parse_pattern()
            self.expect(TK_RPAREN)
            return OkPat(inner=inner, line=line, col=col)

        if t.kind == TK_KEYWORD and t.value == "err":
            self.advance()
            self.expect(TK_LPAREN)
            inner = self._parse_pattern()
            self.expect(TK_RPAREN)
            return ErrPat(inner=inner, line=line, col=col)

        if t.kind in (TK_INT, TK_FLOAT, TK_STR, TK_BOOL):
            lit = self._parse_primary()
            # Range pattern?
            if self.check(TK_DOTDOT) or self.check(TK_DOTDOTEQ):
                inclusive = self.cur().kind == TK_DOTDOTEQ
                self.advance()
                end = self._parse_primary()
                return RangePat(start=lit, end=end, inclusive=inclusive,
                                line=line, col=col)
            return LitPat(value=lit, line=line, col=col)

        if t.kind == TK_MINUS:
            self.advance()
            val = self._parse_primary()
            lit = UnaryOp(op="-", operand=val, line=line, col=col)
            return LitPat(value=lit, line=line, col=col)

        if t.kind == TK_LPAREN:
            self.advance()
            pats = [self._parse_pattern()]
            while self.match(TK_COMMA):
                pats.append(self._parse_pattern())
            self.expect(TK_RPAREN)
            return TuplePat(elements=pats, line=line, col=col)

        if t.kind == TK_IDENT:
            name = self.advance().value
            if self.check(TK_LBRACE):
                # Struct pattern: Foo { x, y }
                self.advance()
                fields = []
                while not self.check(TK_RBRACE):
                    fname = self.expect(TK_IDENT).value
                    fpat = None
                    if self.match(TK_COLON):
                        fpat = self._parse_pattern()
                    else:
                        fpat = IdentPat(name=fname, line=line, col=col)
                    fields.append((fname, fpat))
                    self.match(TK_COMMA)
                self.expect(TK_RBRACE)
                return StructPat(name=name, fields=fields, line=line, col=col)
            if self.check(TK_LPAREN):
                self.advance()
                inner = self._parse_pattern()
                self.expect(TK_RPAREN)
                return EnumPat(name=name, inner=inner, line=line, col=col)
            return IdentPat(name=name, line=line, col=col)

        return LitPat(value=self.parse_expr(), line=line, col=col)

    def _parse_return(self) -> ReturnStmt:
        line, col = self.cur().line, self.cur().col
        self.advance()
        value = None
        if not self.check(TK_SEMICOLON) and not self.check(TK_RBRACE):
            value = self.parse_expr()
            # Multiple return: return a, b
            if self.match(TK_COMMA):
                vals = [value]
                vals.append(self.parse_expr())
                while self.match(TK_COMMA):
                    vals.append(self.parse_expr())
                value = TupleLit(elements=vals, line=line, col=col)
        self.match(TK_SEMICOLON)
        return ReturnStmt(value=value, line=line, col=col)

    def _parse_break(self) -> BreakStmt:
        line, col = self.cur().line, self.cur().col
        self.advance()
        label = None
        if self.check(TK_IDENT):
            label = self.advance().value
        self.match(TK_SEMICOLON)
        return BreakStmt(label=label, line=line, col=col)

    def _parse_continue(self) -> ContinueStmt:
        line, col = self.cur().line, self.cur().col
        self.advance()
        label = None
        if self.check(TK_IDENT):
            label = self.advance().value
        self.match(TK_SEMICOLON)
        return ContinueStmt(label=label, line=line, col=col)

    def _parse_defer(self) -> DeferStmt:
        line, col = self.cur().line, self.cur().col
        self.advance()
        expr = self.parse_expr()
        self.match(TK_SEMICOLON)
        return DeferStmt(expr=expr, line=line, col=col)

    def _parse_go(self) -> GoStmt:
        line, col = self.cur().line, self.cur().col
        self.advance()
        expr = self.parse_expr()
        self.match(TK_SEMICOLON)
        return GoStmt(expr=expr, line=line, col=col)

    def _parse_unsafe(self) -> UnsafeBlock:
        line, col = self.cur().line, self.cur().col
        self.advance()
        body = self._parse_block()
        return UnsafeBlock(body=body, line=line, col=col)

    def _parse_pyblock(self) -> PyBlock:
        line, col = self.cur().line, self.cur().col
        self.advance()  # pyblock
        self.expect(TK_LBRACE)
        # Collect raw Python code until matching }
        start = self.tokens[self.pos - 1].col  # approximate
        code_parts = []
        depth = 1
        while self.pos < len(self.tokens) and depth > 0:
            t = self.cur()
            if t.kind == TK_LBRACE:
                depth += 1
                code_parts.append("{")
            elif t.kind == TK_RBRACE:
                depth -= 1
                if depth > 0:
                    code_parts.append("}")
            elif t.kind == TK_EOF:
                break
            else:
                code_parts.append(str(t.value) if t.value is not None else "")
            self.advance()
        # We use the raw source between the braces
        return PyBlock(code=" ".join(code_parts), line=line, col=col)

    # ── Declarations ──────────────────────────────────────────────────────────

    def _parse_fn(self, decorators=None) -> FnDecl:
        line, col = self.cur().line, self.cur().col
        self.advance()  # fn
        # Allow keyword names like 'new', 'drop', operator names
        if self.cur().kind == TK_KEYWORD and self.cur().value in (
            "new", "drop", "self", "super", "static"
        ):
            name = self.advance().value
        elif self.cur().kind == TK_IDENT and self.cur().value == "operator":
            # Operator overload: fn operator==(self, other) -> bool
            self.advance()  # 'operator'
            op_tok = self.advance()
            name = "operator" + str(op_tok.value)
        elif self.cur().kind == TK_IDENT:
            name = self.advance().value
        else:
            name = self.expect(TK_IDENT).value
        type_params = self._parse_type_params()
        params = self._parse_params()
        return_type = None
        if self.match(TK_ARROW):
            return_type = self._parse_type()
        body = None
        if self.check(TK_LBRACE):
            body = self._parse_block()
        else:
            self.match(TK_SEMICOLON)
        return FnDecl(name=name, type_params=type_params, params=params,
                      return_type=return_type, body=body,
                      decorators=decorators or [],
                      line=line, col=col)

    def _parse_params(self) -> List[Param]:
        self.expect(TK_LPAREN)
        params = []
        while not self.check(TK_RPAREN) and not self.check(TK_EOF):
            params.append(self._parse_param())
            if not self.match(TK_COMMA):
                break
        self.expect(TK_RPAREN)
        return params

    def _parse_param(self) -> Param:
        line, col = self.cur().line, self.cur().col
        variadic = False
        if self.check(TK_DOTDOT):
            self.advance()
            variadic = True
        # Allow var/let modifier on self (mutating self)
        if self.cur().kind == TK_KEYWORD and self.cur().value in ("var", "let"):
            self.advance()  # consume modifier
        # Allow self/super/new as parameter names
        if self.cur().kind == TK_KEYWORD and self.cur().value in ("self", "super", "new"):
            name = self.advance().value
        else:
            name = self.expect(TK_IDENT).value
        type_ann = None
        if self.match(TK_COLON):
            type_ann = self._parse_type()
        default = None
        if self.match(TK_EQ):
            default = self.parse_expr()
        return Param(name=name, type_ann=type_ann, default=default,
                     variadic=variadic, line=line, col=col)

    def _parse_type_params(self) -> List[str]:
        params = []
        if self.match(TK_LT):
            while not self.check(TK_GT) and not self.check(TK_EOF):
                params.append(self.expect(TK_IDENT).value)
                # optional constraint: T: Trait
                if self.match(TK_COLON):
                    self.expect(TK_IDENT)  # constraint name (ignored in interpreter)
                if not self.match(TK_COMMA):
                    break
            self.expect(TK_GT)
        return params

    def _parse_type(self) -> TypeName:
        line, col = self.cur().line, self.cur().col
        nullable = False
        pointer = False
        ref_ = False
        array = False
        array_size = None

        if self.match(TK_QUESTION):
            nullable = True
        elif self.match(TK_STAR):
            pointer = True
        elif self.match(TK_AMP):
            ref_ = True

        if self.check(TK_LBRACKET):
            self.advance()
            if self.check(TK_INT):
                array_size = self.advance().value
            self.expect(TK_RBRACKET)
            array = True

        # fn type: fn(A, B) -> C
        if self.kw("fn"):
            self.advance()
            self.expect(TK_LPAREN)
            while not self.check(TK_RPAREN):
                self._parse_type()
                self.match(TK_COMMA)
            self.expect(TK_RPAREN)
            ret = None
            if self.match(TK_ARROW):
                ret = self._parse_type()
            return TypeName(name="fn", nullable=nullable, array=array,
                            line=line, col=col)

        # chan<T>
        if self.kw("chan"):
            self.advance()
            params = []
            if self.match(TK_LT):
                params.append(self._parse_type())
                self.expect(TK_GT)
            return TypeName(name="chan", params=params, nullable=nullable,
                            line=line, col=col)

        name = self.expect(TK_IDENT).value
        params = []
        if self.match(TK_LT):
            while not self.check(TK_GT) and not self.check(TK_EOF):
                params.append(self._parse_type())
                if not self.match(TK_COMMA):
                    break
            self.expect(TK_GT)

        return TypeName(name=name, params=params, nullable=nullable,
                        pointer=pointer, ref_=ref_, array=array,
                        array_size=array_size, line=line, col=col)

    def _parse_struct(self) -> StructDecl:
        line, col = self.cur().line, self.cur().col
        self.advance()  # struct
        name = self.expect(TK_IDENT).value
        type_params = self._parse_type_params()
        self.expect(TK_LBRACE)
        fields = []
        self.skip_semis()
        while not self.check(TK_RBRACE) and not self.check(TK_EOF):
            fname = self.expect(TK_IDENT).value
            self.expect(TK_COLON)
            ftype = self._parse_type()
            default = None
            if self.match(TK_EQ):
                default = self.parse_expr()
            fields.append(StructField(name=fname, type_ann=ftype,
                                      default=default, line=line, col=col))
            self.match(TK_COMMA)
            self.skip_semis()
        self.expect(TK_RBRACE)
        return StructDecl(name=name, fields=fields, type_params=type_params,
                          line=line, col=col)

    def _parse_impl(self) -> ImplBlock:
        line, col = self.cur().line, self.cur().col
        self.advance()  # impl
        target = self.expect(TK_IDENT).value
        interface = None
        if self.kw("for"):
            self.advance()
            interface = target
            target = self.expect(TK_IDENT).value
        self.expect(TK_LBRACE)
        methods = []
        self.skip_semis()
        while not self.check(TK_RBRACE) and not self.check(TK_EOF):
            decs = self._parse_decorators()
            is_async = bool(self.match_kw("async"))
            fn = self._parse_fn(decs)
            fn.is_async = is_async
            methods.append(fn)
            self.skip_semis()
        self.expect(TK_RBRACE)
        return ImplBlock(target=target, interface=interface, methods=methods,
                         line=line, col=col)

    def _parse_class(self, decorators=None) -> ClassDecl:
        line, col = self.cur().line, self.cur().col
        is_abstract = False
        if self.kw("abstract"):
            is_abstract = True
            self.advance()
        self.advance()  # class
        name = self.expect(TK_IDENT).value
        type_params = self._parse_type_params()
        parent = None
        if self.match_kw("extends"):
            parent = self.expect(TK_IDENT).value
        interfaces = []
        if self.match_kw("implements"):
            interfaces.append(self.expect(TK_IDENT).value)
            while self.match(TK_COMMA):
                interfaces.append(self.expect(TK_IDENT).value)
        self.expect(TK_LBRACE)
        fields = []
        methods = []
        self.skip_semis()
        while not self.check(TK_RBRACE) and not self.check(TK_EOF):
            decs = self._parse_decorators()
            is_abs = False
            if self.kw("abstract"):
                is_abs = True
                self.advance()
            is_over = False
            if self.kw("override"):
                is_over = True
                self.advance()
            is_async = bool(self.match_kw("async"))
            if self.kw("fn") or (self.cur().kind == TK_KEYWORD and self.cur().value in ("fn",)):
                fn = self._parse_fn(decs)
                fn.is_abstract = is_abs
                fn.is_override = is_over
                fn.is_async = is_async
                methods.append(fn)
            elif self.kw("let") or self.kw("var"):
                mutable = self.kw("var")
                self.advance()
                fname = self.expect(TK_IDENT).value
                ftype = None
                if self.match(TK_COLON):
                    ftype = self._parse_type()
                default = None
                if self.match(TK_EQ):
                    default = self.parse_expr()
                fields.append(StructField(name=fname, type_ann=ftype,
                                          default=default, line=line, col=col))
                self.match(TK_SEMICOLON)
            self.skip_semis()
        self.expect(TK_RBRACE)
        return ClassDecl(name=name, parent=parent, interfaces=interfaces,
                         type_params=type_params, fields=fields, methods=methods,
                         is_abstract=is_abstract, decorators=decorators or [],
                         line=line, col=col)

    def _parse_interface(self) -> InterfaceDecl:
        line, col = self.cur().line, self.cur().col
        self.advance()  # interface
        name = self.expect(TK_IDENT).value
        type_params = self._parse_type_params()
        extends = []
        if self.match_kw("extends"):
            extends.append(self.expect(TK_IDENT).value)
            while self.match(TK_COMMA):
                extends.append(self.expect(TK_IDENT).value)
        self.expect(TK_LBRACE)
        methods = []
        self.skip_semis()
        while not self.check(TK_RBRACE) and not self.check(TK_EOF):
            self.match_kw("fn")
            mname = self.expect(TK_IDENT).value
            params = self._parse_params()
            ret = None
            if self.match(TK_ARROW):
                ret = self._parse_type()
            body = None
            if self.check(TK_LBRACE):
                body = self._parse_block()
            else:
                self.match(TK_SEMICOLON)
            methods.append(InterfaceMethod(name=mname, params=params,
                                           return_type=ret, body=body,
                                           has_default=body is not None,
                                           line=line, col=col))
            self.skip_semis()
        self.expect(TK_RBRACE)
        return InterfaceDecl(name=name, type_params=type_params,
                             methods=methods, extends=extends,
                             line=line, col=col)

    def _parse_enum(self) -> EnumDecl:
        line, col = self.cur().line, self.cur().col
        self.advance()  # enum
        name = self.expect(TK_IDENT).value
        type_params = self._parse_type_params()
        self.expect(TK_LBRACE)
        variants = []
        self.skip_semis()
        while not self.check(TK_RBRACE) and not self.check(TK_EOF):
            vname = self.expect(TK_IDENT).value
            fields = []
            if self.match(TK_LPAREN):
                while not self.check(TK_RPAREN):
                    fields.append(self._parse_type())
                    self.match(TK_COMMA)
                self.expect(TK_RPAREN)
            variants.append(EnumVariant(name=vname, fields=fields,
                                        line=line, col=col))
            self.match(TK_COMMA)
            self.skip_semis()
        self.expect(TK_RBRACE)
        return EnumDecl(name=name, variants=variants, type_params=type_params,
                        line=line, col=col)

    def _parse_import(self) -> ImportDecl:
        line, col = self.cur().line, self.cur().col
        self.advance()  # import
        _LANGS = ("python", "js", "c", "cpp", "java", "swift")
        lang = ""
        for lname in _LANGS:
            if self.check(TK_IDENT, lname):
                lang = lname
                self.advance()
                break
        path = self.expect(TK_STR).value
        alias = None
        if self.match_kw("as"):
            alias = self.expect(TK_IDENT).value
        self.match(TK_SEMICOLON)
        return ImportDecl(path=path, alias=alias, lang=lang,
                          line=line, col=col)

    def _parse_module(self) -> ModuleDecl:
        line, col = self.cur().line, self.cur().col
        self.advance()  # module
        name = self.expect(TK_IDENT).value
        body = self._parse_block()
        return ModuleDecl(name=name, body=body, line=line, col=col)

    def _parse_macro(self) -> MacroDecl:
        line, col = self.cur().line, self.cur().col
        self.advance()  # macro
        name = self.expect(TK_IDENT).value
        self.expect(TK_LPAREN)
        params = []
        while not self.check(TK_RPAREN):
            params.append(self.expect(TK_IDENT).value)
            self.match(TK_COMMA)
        self.expect(TK_RPAREN)
        body = self._parse_block()
        return MacroDecl(name=name, params=params, body=body,
                         line=line, col=col)

    def _parse_actor(self, decorators=None) -> ActorDecl:
        """actor MyActor { var state: int; fn handle(self, msg: any) { ... } }"""
        line, col = self.cur().line, self.cur().col
        self.advance()  # actor
        name = self.expect(TK_IDENT).value
        type_params = self._parse_type_params()
        parent = None
        if self.match_kw("extends"):
            parent = self.expect(TK_IDENT).value
        self.expect(TK_LBRACE)
        fields = []
        methods = []
        self.skip_semis()
        while not self.check(TK_RBRACE) and not self.check(TK_EOF):
            decs = self._parse_decorators()
            is_async = bool(self.match_kw("async"))
            if self.kw("fn"):
                fn = self._parse_fn(decs)
                fn.is_async = is_async
                methods.append(fn)
            elif self.kw("let") or self.kw("var"):
                mutable = self.kw("var")
                self.advance()
                fname = self.expect(TK_IDENT).value
                ftype = None
                if self.match(TK_COLON):
                    ftype = self._parse_type()
                default = None
                if self.match(TK_EQ):
                    default = self.parse_expr()
                fields.append(StructField(name=fname, type_ann=ftype,
                                          default=default, line=line, col=col))
                self.match(TK_SEMICOLON)
            self.skip_semis()
        self.expect(TK_RBRACE)
        return ActorDecl(name=name, type_params=type_params, fields=fields,
                         methods=methods, parent=parent, line=line, col=col)

    def _parse_spawn_expr(self) -> SpawnExpr:
        """spawn MyActor(args) as expression"""
        line, col = self.cur().line, self.cur().col
        self.advance()  # spawn
        actor_class = self.parse_expr()
        return SpawnExpr(actor_class=actor_class, args=[], kwargs=[],
                         line=line, col=col)

    def _parse_spawn_stmt(self) -> ExprStmt:
        """spawn MyActor(args) as statement"""
        line, col = self.cur().line, self.cur().col
        expr = self._parse_spawn_expr()
        self.match(TK_SEMICOLON)
        return ExprStmt(expr=expr, line=line, col=col)

    def _parse_receive(self) -> ReceiveStmt:
        """receive { pattern => body, ... }"""
        line, col = self.cur().line, self.cur().col
        self.advance()  # receive
        timeout = None
        if self.match_kw("timeout"):
            timeout = self.parse_expr()
        self.expect(TK_LBRACE)
        arms = []
        self.skip_semis()
        while not self.check(TK_RBRACE) and not self.check(TK_EOF):
            pattern = self._parse_pattern()
            guard = None
            if self.match_kw("if"):
                guard = self.parse_expr()
            self.expect(TK_FATARROW)
            if self.check(TK_LBRACE):
                body = self._parse_block()
            else:
                body = self.parse_expr()
            arms.append(MatchArm(pattern=pattern, guard=guard, body=body,
                                 line=line, col=col))
            self.match(TK_COMMA)
            self.skip_semis()
        self.expect(TK_RBRACE)
        return ReceiveStmt(arms=arms, timeout=timeout, line=line, col=col)

    # ── Expressions ───────────────────────────────────────────────────────────

    # Token kinds that can start an expression (for ternary disambiguation)
    _EXPR_START = None  # set after class definition

    def parse_expr(self) -> Any:
        left = self._parse_pipe()
        # Ternary: cond ? then : else
        # Disambiguate from propagate (expr?) by checking if ? is followed by an expression
        if self.check(TK_QUESTION):
            next_kind = self.peek().kind
            STARTERS = {
                TK_INT, TK_FLOAT, TK_STR, TK_FSTR, TK_BOOL, TK_NONE,
                TK_IDENT, TK_KEYWORD,
                TK_LPAREN, TK_LBRACKET, TK_MINUS, TK_BANG, TK_TILDE,
                TK_SHELL, TK_SQL, TK_AMP, TK_STAR, TK_PIPE,
            }
            if next_kind in STARTERS:
                line, col = self.cur().line, self.cur().col
                self.advance()  # consume ?
                then = self._parse_pipe()
                self.expect(TK_COLON)
                else_ = self._parse_pipe()
                return TernaryExpr(cond=left, then=then, else_=else_,
                                   line=line, col=col)
            else:
                # Propagate: expr? — passes errors up
                line, col = self.cur().line, self.cur().col
                self.advance()
                return PropagateExpr(expr=left, line=line, col=col)
        return left

    def _parse_pipe(self) -> Any:
        left = self._parse_nullcoalesce()
        while self.check(TK_PIPE_GT):
            line, col = self.cur().line, self.cur().col
            self.advance()
            right = self._parse_or()
            left = PipeExpr(left=left, right=right, line=line, col=col)
        return left

    def _parse_nullcoalesce(self) -> Any:
        left = self._parse_or()
        while self.check(TK_DOUBLEQUEST):
            line, col = self.cur().line, self.cur().col
            self.advance()
            right = self._parse_or()
            left = NullCoalesceExpr(left=left, right=right, line=line, col=col)
        return left

    def _parse_or(self) -> Any:
        left = self._parse_and()
        while self.check(TK_PIPEPIPE) or self.kw("or"):
            line, col = self.cur().line, self.cur().col
            self.advance()
            right = self._parse_and()
            left = BinOp(op="||", left=left, right=right, line=line, col=col)
        return left

    def _parse_and(self) -> Any:
        left = self._parse_equality()
        while self.check(TK_AMPAMP) or self.kw("and"):
            line, col = self.cur().line, self.cur().col
            self.advance()
            right = self._parse_equality()
            left = BinOp(op="&&", left=left, right=right, line=line, col=col)
        return left

    def _parse_equality(self) -> Any:
        left = self._parse_comparison()
        while self.cur().kind in (TK_EQEQ, TK_BANGEQ):
            line, col = self.cur().line, self.cur().col
            op = self.advance().value
            right = self._parse_comparison()
            left = BinOp(op=op, left=left, right=right, line=line, col=col)
        return left

    def _parse_comparison(self) -> Any:
        left = self._parse_range()
        while self.cur().kind in (TK_LT, TK_GT, TK_LTEQ, TK_GTEQ):
            line, col = self.cur().line, self.cur().col
            op = self.advance().value
            right = self._parse_range()
            left = BinOp(op=op, left=left, right=right, line=line, col=col)
        return left

    def _parse_range(self) -> Any:
        left = self._parse_bitor()
        if self.cur().kind in (TK_DOTDOT, TK_DOTDOTEQ):
            line, col = self.cur().line, self.cur().col
            inclusive = self.cur().kind == TK_DOTDOTEQ
            self.advance()
            right = self._parse_bitor()
            return RangeExpr(start=left, end=right, inclusive=inclusive,
                             line=line, col=col)
        return left

    def _parse_bitor(self) -> Any:
        left = self._parse_bitxor()
        while self.check(TK_PIPE) and not self.check(TK_PIPE_GT):
            line, col = self.cur().line, self.cur().col
            self.advance()
            right = self._parse_bitxor()
            left = BinOp(op="|", left=left, right=right, line=line, col=col)
        return left

    def _parse_bitxor(self) -> Any:
        left = self._parse_bitand()
        while self.check(TK_CARET):
            line, col = self.cur().line, self.cur().col
            self.advance()
            right = self._parse_bitand()
            left = BinOp(op="^", left=left, right=right, line=line, col=col)
        return left

    def _parse_bitand(self) -> Any:
        left = self._parse_shift()
        while self.check(TK_AMP):
            line, col = self.cur().line, self.cur().col
            self.advance()
            right = self._parse_shift()
            left = BinOp(op="&", left=left, right=right, line=line, col=col)
        return left

    def _parse_shift(self) -> Any:
        left = self._parse_add()
        while self.cur().kind in (TK_LSHIFT, TK_RSHIFT):
            line, col = self.cur().line, self.cur().col
            op = self.advance().value
            right = self._parse_add()
            left = BinOp(op=op, left=left, right=right, line=line, col=col)
        return left

    def _parse_add(self) -> Any:
        left = self._parse_mul()
        while self.cur().kind in (TK_PLUS, TK_MINUS):
            line, col = self.cur().line, self.cur().col
            op = self.advance().value
            right = self._parse_mul()
            left = BinOp(op=op, left=left, right=right, line=line, col=col)
        return left

    def _parse_mul(self) -> Any:
        left = self._parse_pow()
        while self.cur().kind in (TK_STAR, TK_SLASH, TK_PERCENT):
            line, col = self.cur().line, self.cur().col
            op = self.advance().value
            right = self._parse_pow()
            left = BinOp(op=op, left=left, right=right, line=line, col=col)
        return left

    def _parse_pow(self) -> Any:
        left = self._parse_cast()
        if self.check(TK_STARSTAR):
            line, col = self.cur().line, self.cur().col
            self.advance()
            right = self._parse_pow()  # right-associative
            return BinOp(op="**", left=left, right=right, line=line, col=col)
        return left

    def _parse_cast(self) -> Any:
        left = self._parse_unary()
        if self.kw("as"):
            line, col = self.cur().line, self.cur().col
            self.advance()
            to_type = self.expect(TK_IDENT).value
            return CastExpr(expr=left, to_type=to_type, line=line, col=col)
        return left

    def _parse_unary(self) -> Any:
        line, col = self.cur().line, self.cur().col
        if self.check(TK_MINUS):
            self.advance()
            return UnaryOp(op="-", operand=self._parse_unary(), line=line, col=col)
        if self.check(TK_BANG) or self.kw("not"):
            self.advance()
            return UnaryOp(op="!", operand=self._parse_unary(), line=line, col=col)
        if self.check(TK_TILDE):
            self.advance()
            return UnaryOp(op="~", operand=self._parse_unary(), line=line, col=col)
        if self.check(TK_AMP):
            self.advance()
            return RefExpr(expr=self._parse_unary(), line=line, col=col)
        if self.check(TK_STAR):
            self.advance()
            return DerefExpr(expr=self._parse_unary(), line=line, col=col)
        if self.match_kw("await"):
            expr = self._parse_unary()
            return AwaitExpr(expr=expr, line=line, col=col)
        return self._parse_postfix()

    def _parse_postfix(self) -> Any:
        expr = self._parse_primary()
        while True:
            line, col = self.cur().line, self.cur().col
            if self.match(TK_DOT):
                # Allow keywords as member names (e.g., .new, .type, .in)
                if self.cur().kind == TK_KEYWORD:
                    member = self.advance().value
                else:
                    member = self.expect(TK_IDENT).value
                expr = MemberExpr(obj=expr, member=member, line=line, col=col)
            elif self.match(TK_QUESTION_DOT):
                # Optional chaining: obj?.member or obj?.[idx]
                if self.check(TK_LBRACKET):
                    self.advance()
                    idx = self.parse_expr()
                    self.expect(TK_RBRACKET)
                    expr = OptIndexExpr(obj=expr, index=idx, line=line, col=col)
                else:
                    if self.cur().kind == TK_KEYWORD:
                        member = self.advance().value
                    else:
                        member = self.expect(TK_IDENT).value
                    expr = OptChainExpr(obj=expr, member=member, line=line, col=col)
            elif self.check(TK_LBRACKET):
                self.advance()
                # Slice or index: a[start:end:step] or a[idx]
                start = end = step = None
                is_slice = False
                if self.check(TK_COLON):
                    # a[:...] — start is None, consume single colon
                    is_slice = True
                    self.advance()
                    if not self.check(TK_RBRACKET) and not self.check(TK_COLON) and not self.check(TK_DOUBLECOLON):
                        end = self.parse_expr()
                    if self.match(TK_COLON):
                        if not self.check(TK_RBRACKET):
                            step = self.parse_expr()
                elif self.check(TK_DOUBLECOLON):
                    # a[::step] — start and end are None
                    is_slice = True
                    self.advance()
                    if not self.check(TK_RBRACKET):
                        step = self.parse_expr()
                else:
                    start = self.parse_expr()
                    if self.match(TK_COLON):
                        is_slice = True
                        if not self.check(TK_RBRACKET) and not self.check(TK_COLON) and not self.check(TK_DOUBLECOLON):
                            end = self.parse_expr()
                        if self.match(TK_COLON):
                            if not self.check(TK_RBRACKET):
                                step = self.parse_expr()
                    elif self.match(TK_DOUBLECOLON):
                        # a[start::step] — end is None
                        is_slice = True
                        if not self.check(TK_RBRACKET):
                            step = self.parse_expr()
                self.expect(TK_RBRACKET)
                if is_slice:
                    expr = SliceExpr(obj=expr, start=start, end=end, step=step, line=line, col=col)
                else:
                    expr = IndexExpr(obj=expr, index=start, line=line, col=col)
            elif self.check(TK_LPAREN):
                args, kwargs = self._parse_call_args()
                expr = CallExpr(callee=expr, args=args, kwargs=kwargs,
                                line=line, col=col)
            elif self.check(TK_CHAN_RECV):
                # ch <- val (send) — TK_CHAN_RECV token is also used for send
                self.advance()
                value = self.parse_expr()
                expr = BinOp(op="<-", left=expr, right=value, line=line, col=col)
            else:
                break
        return expr

    def _parse_call_args(self):
        self.expect(TK_LPAREN)
        args = []
        kwargs = []
        while not self.check(TK_RPAREN) and not self.check(TK_EOF):
            # Named arg: name: value
            if (self.check(TK_IDENT) and self.peek().kind == TK_COLON
                    and self.peek(2).kind != TK_COLON):
                key = self.advance().value
                self.advance()  # :
                val = self.parse_expr()
                kwargs.append((key, val))
            else:
                args.append(self.parse_expr())
            if not self.match(TK_COMMA):
                break
        self.expect(TK_RPAREN)
        return args, kwargs

    def _parse_primary(self) -> Any:
        line, col = self.cur().line, self.cur().col
        t = self.cur()

        # Literals
        if t.kind == TK_INT:
            self.advance()
            return IntLit(value=t.value, line=line, col=col)
        if t.kind == TK_FLOAT:
            self.advance()
            return FloatLit(value=t.value, line=line, col=col)
        if t.kind == TK_BOOL:
            self.advance()
            return BoolLit(value=t.value, line=line, col=col)
        if t.kind == TK_NONE:
            self.advance()
            return NoneLit(line=line, col=col)
        if t.kind == TK_STR:
            self.advance()
            return StrLit(value=t.value, line=line, col=col)
        if t.kind == TK_FSTR:
            self.advance()
            return self._build_fstr(t.value, line, col)
        if t.kind == TK_SHELL:
            self.advance()
            return self._build_shell(t.value, line, col)
        if t.kind == TK_SQL:
            self.advance()
            return self._build_sql(t.value, line, col)

        # Keywords as expressions
        if t.kind == TK_KEYWORD:
            kw = t.value
            if kw == "true":
                self.advance()
                return BoolLit(value=True, line=line, col=col)
            if kw == "false":
                self.advance()
                return BoolLit(value=False, line=line, col=col)
            if kw == "none":
                self.advance()
                return NoneLit(line=line, col=col)
            if kw == "some":
                self.advance()
                self.expect(TK_LPAREN)
                val = self.parse_expr()
                self.expect(TK_RPAREN)
                return SomeExpr(value=val, line=line, col=col)
            if kw == "ok":
                self.advance()
                self.expect(TK_LPAREN)
                val = self.parse_expr()
                self.expect(TK_RPAREN)
                return OkExpr(value=val, line=line, col=col)
            if kw == "err":
                self.advance()
                self.expect(TK_LPAREN)
                val = self.parse_expr()
                self.expect(TK_RPAREN)
                return ErrExpr(value=val, line=line, col=col)
            if kw == "self":
                self.advance()
                return Ident(name="self", line=line, col=col)
            if kw == "super":
                self.advance()
                return Ident(name="super", line=line, col=col)
            if kw == "new":
                self.advance()
                return Ident(name="new", line=line, col=col)
            if kw == "chan":
                self.advance()
                elem_type = None
                if self.match(TK_LT):
                    elem_type = self._parse_type()
                    self.expect(TK_GT)
                self.expect(TK_LPAREN)
                cap = None
                if not self.check(TK_RPAREN):
                    cap = self.parse_expr()
                self.expect(TK_RPAREN)
                return CallExpr(
                    callee=Ident(name="__chan__", line=line, col=col),
                    args=[cap] if cap else [],
                    kwargs=[],
                    line=line, col=col
                )
            if kw == "sizeof":
                self.advance()
                self.expect(TK_LPAREN)
                tname = self.expect(TK_IDENT).value
                self.expect(TK_RPAREN)
                return SizeofExpr(type_name=tname, line=line, col=col)
            if kw == "match":
                return self._parse_match()
            if kw == "if":
                return self._parse_if()
            if kw == "spawn":
                return self._parse_spawn_expr()
            if kw == "comptime":
                self.advance()
                return ComptimeExpr(expr=self.parse_expr(), line=line, col=col)

        # Closure: |x, y| expr  or  |x| { block }
        if self.check(TK_PIPE):
            return self._parse_closure(line, col)

        # Grouping / tuple
        if self.match(TK_LPAREN):
            if self.check(TK_RPAREN):
                self.advance()
                return TupleLit(elements=[], line=line, col=col)
            expr = self.parse_expr()
            if self.match(TK_COMMA):
                # Tuple
                elems = [expr]
                while not self.check(TK_RPAREN):
                    elems.append(self.parse_expr())
                    if not self.match(TK_COMMA):
                        break
                self.expect(TK_RPAREN)
                return TupleLit(elements=elems, line=line, col=col)
            self.expect(TK_RPAREN)
            # Ternary: (cond) ? a : b — handled after
            return expr

        # Array literal
        if self.check(TK_LBRACKET):
            self.advance()
            elements = []
            while not self.check(TK_RBRACKET) and not self.check(TK_EOF):
                elements.append(self.parse_expr())
                if not self.match(TK_COMMA):
                    break
            self.expect(TK_RBRACKET)
            return ArrayLit(elements=elements, line=line, col=col)

        # Map literal: { key: val, ... }  (only if starts with str/ident : expr)
        if self.check(TK_LBRACE):
            return self._try_parse_map(line, col)

        # Identifier (possibly StructLit: Foo { ... })
        if t.kind == TK_IDENT:
            self.advance()
            name = t.value
            # Struct literal: Name { field: val, ... }
            if (self.check(TK_LBRACE) and
                    not self._looks_like_block()):
                return self._parse_struct_lit(name, line, col)
            return Ident(name=name, line=line, col=col)

        raise ParseError(f"Unexpected token {t.value!r}", t.line, t.col)

    def _looks_like_block(self) -> bool:
        """Heuristic: { followed by statement keywords = block, not struct lit."""
        if self.pos + 1 >= len(self.tokens):
            return True
        next_tok = self.tokens[self.pos + 1]
        if next_tok.kind == TK_KEYWORD:
            return True
        if next_tok.kind == TK_RBRACE:
            return False  # empty braces
        # If next is ident followed by colon it's struct lit
        if (next_tok.kind == TK_IDENT and
                self.pos + 2 < len(self.tokens) and
                self.tokens[self.pos + 2].kind == TK_COLON):
            return False
        return True

    def _parse_struct_lit(self, name: str, line: int, col: int) -> StructLit:
        self.expect(TK_LBRACE)
        fields = []
        self.skip_semis()
        while not self.check(TK_RBRACE) and not self.check(TK_EOF):
            fname = self.expect(TK_IDENT).value
            value = None
            if self.match(TK_COLON):
                value = self.parse_expr()
            else:
                # Shorthand: { x } means { x: x }
                value = Ident(name=fname, line=line, col=col)
            fields.append((fname, value))
            self.match(TK_COMMA)
            self.skip_semis()
        self.expect(TK_RBRACE)
        return StructLit(name=name, fields=fields, line=line, col=col)

    def _try_parse_map(self, line, col) -> Any:
        """Try to parse a map literal { k: v, ... }"""
        # Peek ahead to distinguish from block
        save = self.pos
        try:
            self.advance()  # {
            if self.check(TK_RBRACE):
                self.advance()
                return MapLit(pairs=[], line=line, col=col)
            pairs = []
            while not self.check(TK_RBRACE) and not self.check(TK_EOF):
                key = self.parse_expr()
                self.expect(TK_COLON)
                val = self.parse_expr()
                pairs.append((key, val))
                if not self.match(TK_COMMA):
                    break
            self.expect(TK_RBRACE)
            return MapLit(pairs=pairs, line=line, col=col)
        except ParseError:
            self.pos = save
            return self._parse_block()

    def _parse_closure(self, line, col) -> Closure:
        self.expect(TK_PIPE)
        params = []
        while not self.check(TK_PIPE) and not self.check(TK_EOF):
            params.append(self.expect(TK_IDENT).value)
            if not self.match(TK_COMMA):
                break
        self.expect(TK_PIPE)
        if self.check(TK_LBRACE):
            body = self._parse_block()
        else:
            body = self.parse_expr()
        return Closure(params=params, body=body, line=line, col=col)

    def _build_fstr(self, parts: list, line: int, col: int) -> FStrLit:
        resolved = []
        for p in parts:
            if isinstance(p, tuple) and p[0] == "expr":
                sub_tokens = tokenize(p[1])
                sub_parser = Parser(sub_tokens, p[1])
                resolved.append(sub_parser.parse_expr())
            else:
                resolved.append(p)
        return FStrLit(parts=resolved, line=line, col=col)

    def _build_shell(self, parts: list, line: int, col: int) -> ShellExpr:
        return ShellExpr(parts=self._resolve_template_parts(parts), line=line, col=col)

    def _build_sql(self, parts: list, line: int, col: int) -> SqlExpr:
        return SqlExpr(parts=self._resolve_template_parts(parts), line=line, col=col)

    def _resolve_template_parts(self, parts: list) -> list:
        resolved = []
        for p in parts:
            if isinstance(p, tuple) and p[0] == "expr":
                sub_tokens = tokenize(p[1])
                sub_parser = Parser(sub_tokens, p[1])
                resolved.append(sub_parser.parse_expr())
            else:
                resolved.append(p)
        return resolved


def parse(source: str, filename: str = "<stdin>") -> Program:
    tokens = tokenize(source, filename)
    parser = Parser(tokens, source)
    return parser.parse_program()
