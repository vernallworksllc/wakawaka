"""
Deltoo Lexer — Tokenizes .dt source files into a flat list of tokens.
"""
from dataclasses import dataclass
from typing import List, Optional
import re


# ── Token kinds ───────────────────────────────────────────────────────────────

KEYWORDS = {
    "let", "var", "const", "fn", "class", "struct", "impl", "interface",
    "enum", "if", "else", "while", "for", "do", "match", "return", "break",
    "continue", "import", "as", "in", "go", "chan", "async", "await", "defer",
    "unsafe", "abstract", "extends", "override", "module", "macro", "true",
    "false", "none", "some", "ok", "err", "pub", "priv", "static", "new",
    "self", "super", "and", "or", "not", "is", "pyblock",
    "actor", "spawn", "receive", "comptime",
}

# Token type constants — sequential integers
def _mk():
    i = 0
    names = [
        "TK_INT", "TK_FLOAT", "TK_STR", "TK_FSTR", "TK_BOOL", "TK_NONE",
        "TK_IDENT", "TK_KEYWORD",
        "TK_PLUS", "TK_MINUS", "TK_STAR", "TK_SLASH", "TK_PERCENT", "TK_STARSTAR",
        "TK_AMP", "TK_PIPE", "TK_CARET", "TK_TILDE", "TK_LSHIFT", "TK_RSHIFT",
        "TK_AMPAMP", "TK_PIPEPIPE", "TK_BANG",
        "TK_EQ", "TK_EQEQ", "TK_BANGEQ",
        "TK_LT", "TK_GT", "TK_LTEQ", "TK_GTEQ",
        "TK_PLUSEQ", "TK_MINUSEQ", "TK_STAREQ", "TK_SLASHEQ", "TK_PERCENTEQ",
        "TK_STARSTAREQ", "TK_AMPEQ", "TK_PIPEEQ", "TK_CARETEQ", "TK_LSHIFTEQ", "TK_RSHIFTEQ",
        "TK_ARROW", "TK_FATARROW", "TK_PIPE_GT", "TK_CHAN_RECV", "TK_CHAN_SEND",
        "TK_QUESTION", "TK_QUESTION_DOT", "TK_DOUBLEQUEST",
        "TK_COLON", "TK_DOUBLECOLON", "TK_SEMICOLON", "TK_COMMA", "TK_DOT",
        "TK_DOTDOT", "TK_DOTDOTEQ", "TK_AT",
        "TK_LPAREN", "TK_RPAREN", "TK_LBRACE", "TK_RBRACE", "TK_LBRACKET", "TK_RBRACKET",
        "TK_SHELL", "TK_SQL",
        "TK_HASH",
        "TK_EOF",
    ]
    return {n: v for v, n in enumerate(names)}

_TK = _mk()
TK_INT       = _TK["TK_INT"]
TK_FLOAT     = _TK["TK_FLOAT"]
TK_STR       = _TK["TK_STR"]
TK_FSTR      = _TK["TK_FSTR"]
TK_BOOL      = _TK["TK_BOOL"]
TK_NONE      = _TK["TK_NONE"]
TK_IDENT     = _TK["TK_IDENT"]
TK_KEYWORD   = _TK["TK_KEYWORD"]
TK_PLUS      = _TK["TK_PLUS"]
TK_MINUS     = _TK["TK_MINUS"]
TK_STAR      = _TK["TK_STAR"]
TK_SLASH     = _TK["TK_SLASH"]
TK_PERCENT   = _TK["TK_PERCENT"]
TK_STARSTAR  = _TK["TK_STARSTAR"]
TK_AMP       = _TK["TK_AMP"]
TK_PIPE      = _TK["TK_PIPE"]
TK_CARET     = _TK["TK_CARET"]
TK_TILDE     = _TK["TK_TILDE"]
TK_LSHIFT    = _TK["TK_LSHIFT"]
TK_RSHIFT    = _TK["TK_RSHIFT"]
TK_AMPAMP    = _TK["TK_AMPAMP"]
TK_PIPEPIPE  = _TK["TK_PIPEPIPE"]
TK_BANG      = _TK["TK_BANG"]
TK_EQ        = _TK["TK_EQ"]
TK_EQEQ     = _TK["TK_EQEQ"]
TK_BANGEQ    = _TK["TK_BANGEQ"]
TK_LT        = _TK["TK_LT"]
TK_GT        = _TK["TK_GT"]
TK_LTEQ      = _TK["TK_LTEQ"]
TK_GTEQ      = _TK["TK_GTEQ"]
TK_PLUSEQ    = _TK["TK_PLUSEQ"]
TK_MINUSEQ   = _TK["TK_MINUSEQ"]
TK_STAREQ    = _TK["TK_STAREQ"]
TK_SLASHEQ   = _TK["TK_SLASHEQ"]
TK_PERCENTEQ = _TK["TK_PERCENTEQ"]
TK_STARSTAREQ = _TK["TK_STARSTAREQ"]
TK_AMPEQ     = _TK["TK_AMPEQ"]
TK_PIPEEQ    = _TK["TK_PIPEEQ"]
TK_CARETEQ   = _TK["TK_CARETEQ"]
TK_LSHIFTEQ  = _TK["TK_LSHIFTEQ"]
TK_RSHIFTEQ  = _TK["TK_RSHIFTEQ"]
TK_ARROW     = _TK["TK_ARROW"]
TK_FATARROW  = _TK["TK_FATARROW"]
TK_PIPE_GT   = _TK["TK_PIPE_GT"]
TK_CHAN_RECV = _TK["TK_CHAN_RECV"]
TK_CHAN_SEND = _TK["TK_CHAN_SEND"]
TK_QUESTION       = _TK["TK_QUESTION"]
TK_QUESTION_DOT   = _TK["TK_QUESTION_DOT"]
TK_DOUBLEQUEST    = _TK["TK_DOUBLEQUEST"]
TK_COLON          = _TK["TK_COLON"]
TK_DOUBLECOLON = _TK["TK_DOUBLECOLON"]
TK_SEMICOLON = _TK["TK_SEMICOLON"]
TK_COMMA     = _TK["TK_COMMA"]
TK_DOT       = _TK["TK_DOT"]
TK_DOTDOT    = _TK["TK_DOTDOT"]
TK_DOTDOTEQ  = _TK["TK_DOTDOTEQ"]
TK_AT        = _TK["TK_AT"]
TK_LPAREN    = _TK["TK_LPAREN"]
TK_RPAREN    = _TK["TK_RPAREN"]
TK_LBRACE    = _TK["TK_LBRACE"]
TK_RBRACE    = _TK["TK_RBRACE"]
TK_LBRACKET  = _TK["TK_LBRACKET"]
TK_RBRACKET  = _TK["TK_RBRACKET"]
TK_SHELL     = _TK["TK_SHELL"]
TK_SQL       = _TK["TK_SQL"]
TK_HASH      = _TK["TK_HASH"]
TK_EOF       = _TK["TK_EOF"]

TK_NAMES = {v: k for k, v in _TK.items()}


@dataclass
class Token:
    kind: int
    value: any
    line: int
    col: int

    def __repr__(self):
        name = TK_NAMES.get(self.kind, str(self.kind))
        return f"Token({name}, {self.value!r}, {self.line}:{self.col})"


class LexError(Exception):
    def __init__(self, msg, line, col):
        super().__init__(f"[Lex Error] {msg} at line {line}, col {col}")
        self.line = line
        self.col = col


class Lexer:
    def __init__(self, source: str, filename: str = "<stdin>"):
        self.src = source
        self.filename = filename
        self.pos = 0
        self.line = 1
        self.col = 1
        self.tokens: List[Token] = []

    # ── Helpers ───────────────────────────────────────────────────────────────

    def cur(self) -> str:
        return self.src[self.pos] if self.pos < len(self.src) else "\0"

    def peek(self, offset=1) -> str:
        p = self.pos + offset
        return self.src[p] if p < len(self.src) else "\0"

    def advance(self) -> str:
        ch = self.src[self.pos]
        self.pos += 1
        if ch == "\n":
            self.line += 1
            self.col = 1
        else:
            self.col += 1
        return ch

    def match(self, ch: str) -> bool:
        if self.cur() == ch:
            self.advance()
            return True
        return False

    def tok(self, kind: int, value=None, line=None, col=None) -> Token:
        return Token(kind, value, line or self.line, col or self.col)

    def error(self, msg: str):
        raise LexError(msg, self.line, self.col)

    # ── Main tokenize ─────────────────────────────────────────────────────────

    def tokenize(self) -> List[Token]:
        while self.pos < len(self.src):
            self._skip_whitespace_and_comments()
            if self.pos >= len(self.src):
                break
            tok = self._next_token()
            if tok:
                self.tokens.append(tok)
        self.tokens.append(Token(TK_EOF, None, self.line, self.col))
        return self.tokens

    def _skip_whitespace_and_comments(self):
        while self.pos < len(self.src):
            c = self.cur()
            if c in " \t\r\n":
                self.advance()
            elif c == "/" and self.peek() == "/":
                # Line comment
                while self.pos < len(self.src) and self.cur() != "\n":
                    self.advance()
            elif c == "/" and self.peek() == "*":
                # Block comment
                self.advance(); self.advance()
                while self.pos < len(self.src):
                    if self.cur() == "*" and self.peek() == "/":
                        self.advance(); self.advance()
                        break
                    self.advance()
            else:
                break

    def _next_token(self) -> Optional[Token]:
        line, col = self.line, self.col
        c = self.cur()

        # Numbers
        if c.isdigit() or (c == "." and self.peek().isdigit()):
            return self._lex_number(line, col)

        # Strings
        if c == '"':
            return self._lex_string(line, col)
        if c == "f" and self.peek() == '"':
            return self._lex_fstring(line, col)
        if c == "r" and self.peek() == '"':
            self.advance(); self.advance()
            return self._lex_raw_string(line, col)
        if c == "b" and self.peek() == '"':
            self.advance(); self.advance()
            return self._lex_byte_string(line, col)

        # Shell string: $`...`
        if c == "$" and self.peek() == "`":
            return self._lex_template(TK_SHELL, line, col)

        # SQL string: @sql`...`
        if c == "@":
            self.advance()
            word = self._read_ident_chars()
            if word == "sql" and self.cur() == "`":
                return self._lex_template(TK_SQL, line, col)
            # Otherwise it's a decorator @word
            t = Token(TK_AT, word, line, col)
            return t

        # Identifiers and keywords
        if c.isalpha() or c == "_":
            return self._lex_ident(line, col)

        # Operators and punctuation
        return self._lex_symbol(line, col)

    # ── Numbers ───────────────────────────────────────────────────────────────

    def _lex_number(self, line, col) -> Token:
        start = self.pos
        is_float = False

        # Hex
        if self.cur() == "0" and self.peek() in "xX":
            self.advance(); self.advance()
            while self.cur() in "0123456789abcdefABCDEF_":
                self.advance()
            val = int(self.src[start:self.pos].replace("_", ""), 16)
            return Token(TK_INT, val, line, col)

        # Binary
        if self.cur() == "0" and self.peek() in "bB":
            self.advance(); self.advance()
            while self.cur() in "01_":
                self.advance()
            val = int(self.src[start:self.pos].replace("_", ""), 2)
            return Token(TK_INT, val, line, col)

        # Octal
        if self.cur() == "0" and self.peek() in "oO":
            self.advance(); self.advance()
            while self.cur() in "01234567_":
                self.advance()
            val = int(self.src[start:self.pos].replace("_", ""), 8)
            return Token(TK_INT, val, line, col)

        # Decimal / float
        while self.cur().isdigit() or self.cur() == "_":
            self.advance()
        if self.cur() == "." and self.peek().isdigit():
            is_float = True
            self.advance()
            while self.cur().isdigit() or self.cur() == "_":
                self.advance()
        if self.cur() in "eE":
            is_float = True
            self.advance()
            if self.cur() in "+-":
                self.advance()
            while self.cur().isdigit():
                self.advance()

        raw = self.src[start:self.pos].replace("_", "")
        if is_float:
            return Token(TK_FLOAT, float(raw), line, col)
        return Token(TK_INT, int(raw), line, col)

    # ── Strings ───────────────────────────────────────────────────────────────

    def _lex_string(self, line, col) -> Token:
        self.advance()  # skip "
        buf = []
        while self.pos < len(self.src) and self.cur() != '"':
            buf.append(self._lex_escape())
        if self.cur() != '"':
            self.error("Unterminated string")
        self.advance()
        return Token(TK_STR, "".join(buf), line, col)

    def _lex_raw_string(self, line, col) -> Token:
        buf = []
        while self.pos < len(self.src) and self.cur() != '"':
            buf.append(self.advance())
        if self.cur() != '"':
            self.error("Unterminated raw string")
        self.advance()
        return Token(TK_STR, "".join(buf), line, col)

    def _lex_byte_string(self, line, col) -> Token:
        t = self._lex_string(line, col)
        return Token(TK_STR, t.value.encode(), line, col)

    def _lex_escape(self) -> str:
        if self.cur() != "\\":
            return self.advance()
        self.advance()
        esc = self.advance()
        return {
            "n": "\n", "t": "\t", "r": "\r", "\\": "\\",
            '"': '"', "'": "'", "0": "\0", "a": "\a", "b": "\b",
        }.get(esc, "\\" + esc)

    def _lex_fstring(self, line, col) -> Token:
        """f"Hello, {name}! You have {count} items." """
        self.advance()  # f
        self.advance()  # "
        parts = []
        buf = []
        while self.pos < len(self.src) and self.cur() != '"':
            if self.cur() == "{" and self.peek() != "{":
                if buf:
                    parts.append("".join(buf))
                    buf = []
                self.advance()  # {
                # Collect the inner expression source
                depth = 1
                expr_src = []
                while self.pos < len(self.src):
                    if self.cur() == "{":
                        depth += 1
                    elif self.cur() == "}":
                        depth -= 1
                        if depth == 0:
                            break
                    expr_src.append(self.advance())
                self.advance()  # }
                # Store as raw source to be parsed later
                parts.append(("expr", "".join(expr_src).strip()))
            elif self.cur() == "{" and self.peek() == "{":
                self.advance(); self.advance()
                buf.append("{")
            elif self.cur() == "}" and self.peek() == "}":
                self.advance(); self.advance()
                buf.append("}")
            else:
                buf.append(self._lex_escape())
        if buf:
            parts.append("".join(buf))
        if self.cur() != '"':
            self.error("Unterminated f-string")
        self.advance()
        return Token(TK_FSTR, parts, line, col)

    def _lex_template(self, kind: int, line, col) -> Token:
        """Lex $`...` or @sql`...` with {expr} interpolation."""
        if kind == TK_SHELL:
            self.advance()  # $
        self.advance()  # `
        parts = []
        buf = []
        while self.pos < len(self.src) and self.cur() != "`":
            if self.cur() == "{":
                if buf:
                    parts.append("".join(buf))
                    buf = []
                self.advance()  # {
                depth = 1
                expr_src = []
                while self.pos < len(self.src):
                    if self.cur() == "{":
                        depth += 1
                    elif self.cur() == "}":
                        depth -= 1
                        if depth == 0:
                            break
                    expr_src.append(self.advance())
                self.advance()  # }
                parts.append(("expr", "".join(expr_src).strip()))
            else:
                buf.append(self.advance())
        if buf:
            parts.append("".join(buf))
        if self.cur() != "`":
            self.error(f"Unterminated template string")
        self.advance()
        return Token(kind, parts, line, col)

    # ── Identifiers / keywords ────────────────────────────────────────────────

    def _read_ident_chars(self) -> str:
        buf = []
        while self.cur().isalnum() or self.cur() == "_":
            buf.append(self.advance())
        return "".join(buf)

    def _lex_ident(self, line, col) -> Token:
        name = self._read_ident_chars()
        if name in ("true", "false"):
            return Token(TK_BOOL, name == "true", line, col)
        if name == "none":
            return Token(TK_NONE, None, line, col)
        if name in KEYWORDS:
            return Token(TK_KEYWORD, name, line, col)
        return Token(TK_IDENT, name, line, col)

    # ── Symbols / operators ───────────────────────────────────────────────────

    def _lex_symbol(self, line, col) -> Token:
        c = self.advance()

        if c == "+":
            if self.match("="): return Token(TK_PLUSEQ, "+=", line, col)
            return Token(TK_PLUS, "+", line, col)
        if c == "-":
            if self.match(">"): return Token(TK_ARROW, "->", line, col)
            if self.match("="): return Token(TK_MINUSEQ, "-=", line, col)
            return Token(TK_MINUS, "-", line, col)
        if c == "*":
            if self.cur() == "*":
                self.advance()
                if self.match("="): return Token(TK_STARSTAREQ, "**=", line, col)
                return Token(TK_STARSTAR, "**", line, col)
            if self.match("="): return Token(TK_STAREQ, "*=", line, col)
            return Token(TK_STAR, "*", line, col)
        if c == "/":
            if self.match("="): return Token(TK_SLASHEQ, "/=", line, col)
            return Token(TK_SLASH, "/", line, col)
        if c == "%":
            if self.match("="): return Token(TK_PERCENTEQ, "%=", line, col)
            return Token(TK_PERCENT, "%", line, col)
        if c == "&":
            if self.match("&"): return Token(TK_AMPAMP, "&&", line, col)
            if self.match("="): return Token(TK_AMPEQ, "&=", line, col)
            return Token(TK_AMP, "&", line, col)
        if c == "|":
            if self.match("|"): return Token(TK_PIPEPIPE, "||", line, col)
            if self.match(">"): return Token(TK_PIPE_GT, "|>", line, col)
            if self.match("="): return Token(TK_PIPEEQ, "|=", line, col)
            return Token(TK_PIPE, "|", line, col)
        if c == "^":
            if self.match("="): return Token(TK_CARETEQ, "^=", line, col)
            return Token(TK_CARET, "^", line, col)
        if c == "~":
            return Token(TK_TILDE, "~", line, col)
        if c == "<":
            if self.cur() == "<":
                self.advance()
                if self.match("="): return Token(TK_LSHIFTEQ, "<<=", line, col)
                return Token(TK_LSHIFT, "<<", line, col)
            if self.cur() == "-":
                self.advance()
                return Token(TK_CHAN_RECV, "<-", line, col)
            if self.match("="): return Token(TK_LTEQ, "<=", line, col)
            return Token(TK_LT, "<", line, col)
        if c == ">":
            if self.cur() == ">":
                self.advance()
                if self.match("="): return Token(TK_RSHIFTEQ, ">>=", line, col)
                return Token(TK_RSHIFT, ">>", line, col)
            if self.match("="): return Token(TK_GTEQ, ">=", line, col)
            return Token(TK_GT, ">", line, col)
        if c == "=":
            if self.match("="): return Token(TK_EQEQ, "==", line, col)
            if self.match(">"): return Token(TK_FATARROW, "=>", line, col)
            return Token(TK_EQ, "=", line, col)
        if c == "!":
            if self.match("="): return Token(TK_BANGEQ, "!=", line, col)
            return Token(TK_BANG, "!", line, col)
        if c == "?":
            if self.match("."): return Token(TK_QUESTION_DOT, "?.", line, col)
            if self.match("?"): return Token(TK_DOUBLEQUEST, "??", line, col)
            return Token(TK_QUESTION, "?", line, col)
        if c == ":":
            if self.match(":"): return Token(TK_DOUBLECOLON, "::", line, col)
            return Token(TK_COLON, ":", line, col)
        if c == ";":
            return Token(TK_SEMICOLON, ";", line, col)
        if c == ",":
            return Token(TK_COMMA, ",", line, col)
        if c == ".":
            if self.cur() == ".":
                self.advance()
                if self.match("="): return Token(TK_DOTDOTEQ, "..=", line, col)
                return Token(TK_DOTDOT, "..", line, col)
            return Token(TK_DOT, ".", line, col)
        if c == "(":
            return Token(TK_LPAREN, "(", line, col)
        if c == ")":
            return Token(TK_RPAREN, ")", line, col)
        if c == "{":
            return Token(TK_LBRACE, "{", line, col)
        if c == "}":
            return Token(TK_RBRACE, "}", line, col)
        if c == "[":
            return Token(TK_LBRACKET, "[", line, col)
        if c == "]":
            return Token(TK_RBRACKET, "]", line, col)
        if c == "#":
            return Token(TK_HASH, "#", line, col)

        self.error(f"Unexpected character: {c!r}")


def tokenize(source: str, filename: str = "<stdin>") -> List[Token]:
    return Lexer(source, filename).tokenize()
