"""
Wakawaka → C99 Transpiler
Walks the AST and emits a self-contained C99 source file.
Every Wakawaka value becomes a WkVal (tagged union) at runtime.
"""
import re
from typing import List, Optional, Set

from .ast_nodes import (
    Program, Block, Node,
    IntLit, FloatLit, BoolLit, StrLit, FStrLit, NoneLit,
    ArrayLit, MapLit, TupleLit, StructLit,
    Ident, MemberExpr, IndexExpr, SliceExpr,
    BinOp, UnaryOp, CallExpr, Closure, TernaryExpr, CastExpr,
    PipeExpr, SomeExpr, OkExpr, ErrExpr, PropagateExpr,
    RangeExpr, OptChainExpr, NullCoalesceExpr,
    ShellExpr, SqlExpr, PyBlock,
    LetDecl, ConstDecl, Assign, ExprStmt,
    ReturnStmt, BreakStmt, ContinueStmt, DeferStmt, GoStmt,
    AssertStmt, PanicStmt,
    IfStmt, WhileStmt, DoWhileStmt, ForInStmt, ForCStmt,
    MatchStmt, MatchArm, UnsafeBlock,
    FnDecl, Param, ClassDecl, StructDecl, ImplBlock, EnumDecl,
    InterfaceDecl, ImportDecl, ModuleDecl, MacroDecl,
    ActorDecl, SpawnExpr, ReceiveStmt,
    WildcardPat, LitPat, IdentPat, RangePat, TuplePat,
    StructPat, OkPat, ErrPat, SomePat, NonePat, EnumPat, OrPat,
    RefExpr, DerefExpr, SizeofExpr, AwaitExpr, ComptimeExpr,
)

# ── C reserved words we must not clash with ───────────────────────────────────
_C_RESERVED = {
    'auto','break','case','char','const','continue','default','do','double',
    'else','enum','extern','float','for','goto','if','inline','int','long',
    'register','restrict','return','short','signed','sizeof','static','struct',
    'switch','typedef','union','unsigned','void','volatile','while',
    '_Bool','_Complex','_Imaginary','NULL','true','false',
}

# ── Builtin names → global slot ──────────────────────────────────────────────
_BUILTINS = {
    'println':   'wk_g_println',
    'print':     'wk_g_print',
    'eprintln':  'wk_g_eprintln',
    'readln':    'wk_g_readln',
    'readlines': 'wk_g_readln',   # alias
    'len':       'wk_g_len',
    'str':       'wk_g_str',
    'int':       'wk_g_int',
    'float':     'wk_g_float',
    'bool':      'wk_g_bool',
    'typeof':    'wk_g_typeof',
    'type':      'wk_g_typeof',
    'isNone':    'wk_g_isNone',
    'isSome':    'wk_g_isSome',
    'isOk':      'wk_g_isOk',
    'isErr':     'wk_g_isErr',
    'sum':       'wk_g_sum',
    'min':       'wk_g_min',
    'max':       'wk_g_max',
    'map':       'wk_g_map',
    'filter':    'wk_g_filter',
    'reduce':    'wk_g_reduce',
    'sorted':    'wk_g_sorted',
    'reversed':  'wk_g_reversed',
    'any':       'wk_g_any',
    'all':       'wk_g_all',
    'zip':       'wk_g_zip',
    'enumerate': 'wk_g_enumerate',
    'range':     'wk_g_range',
    'sleep':     'wk_g_sleep',
    'exit':      'wk_g_exit',
    'assert':    'wk_g_assert',
    'copy':      'wk_g_copy',
    'chr':       'wk_g_chr',
    'ord':       'wk_g_ord',
    'hash':      'wk_g_hash',
    'repr':      'wk_g_repr',
    'panic':     'wk_g_panic',
    'math':      'wk_g_math',
    'tensor':    'wk_g_tensor',
    'ad':        'wk_g_ad',
    'gpu':       'wk_g_gpu',
    'pipeline':  'wk_g_pipeline',
    'model':     'wk_g_model',
    'fs':        'wk_g_fs',
    'py':        'wk_g_py',
    'jvm':       'wk_g_jvm',
    'node':      'wk_g_node',
}

_OP_MAP = {
    '+': 'wk_add', '-': 'wk_sub', '*': 'wk_mul', '/': 'wk_div',
    '%': 'wk_mod', '**': 'wk_pow',
    '&': 'wk_band', '|': 'wk_bor', '^': 'wk_bxor',
    '<<': 'wk_lshift', '>>': 'wk_rshift',
    '==': 'wk_cmp_eq', '!=': 'wk_cmp_ne',
    '<':  'wk_cmp_lt', '<=': 'wk_cmp_le',
    '>':  'wk_cmp_gt', '>=': 'wk_cmp_ge',
    'in': 'wk_in',
}


def _c_id(name: str) -> str:
    """Sanitize a Wakawaka name (e.g. 'operator==') into a valid C identifier fragment."""
    safe = re.sub(r'[^A-Za-z0-9_]', '_', name)
    if not safe or safe[0].isdigit():
        safe = '_' + safe
    return safe


def _mangle(name: str) -> str:
    """Mangle a Wakawaka identifier to a safe C name."""
    # $ prefix for PHP-compat globals
    name = name.replace('$_', 'wk_php_').replace('$', 'wk_php_')
    # Replace non-alphanumeric (except _) with _
    safe = re.sub(r'[^A-Za-z0-9_]', '_', name)
    if not safe or safe[0].isdigit():
        safe = '_' + safe
    if safe in _C_RESERVED:
        safe = 'wk_var_' + safe
    return 'wk_var_' + safe


def _escape_c_str(s: str) -> str:
    """Escape a Python string for use in a C string literal."""
    result = []
    for c in s:
        if c == '\\': result.append('\\\\')
        elif c == '"': result.append('\\"')
        elif c == '\n': result.append('\\n')
        elif c == '\r': result.append('\\r')
        elif c == '\t': result.append('\\t')
        elif ord(c) < 32 or ord(c) > 126:
            # Encode as UTF-8 bytes using octal escapes (safe — no greedy hex)
            for b in c.encode('utf-8'):
                result.append(f'\\{b:03o}')
        else:
            result.append(c)
    return ''.join(result)


class TranspilerError(Exception):
    pass


class Transpiler:
    def __init__(self):
        self._lines: List[str] = []        # current function body buffer
        self._preamble_fns: List[str] = [] # lifted closures/inner-fns (file scope)
        self._fwd:   List[str] = []        # forward declarations
        self._cls_defs: List[str] = []     # WkClass global descriptors
        self._init_stmts: List[str] = []   # wk_init() body lines
        self._indent_level = 0
        self._tmp_id = 0
        self._closure_id = 0
        self._label_id = 0
        # scope stack: list of {waka_name: c_name}
        self._scopes: List[dict] = [{}]
        # loop label stack: [(waka_label|None, break_lbl, cont_lbl)]
        self._loop_stack: List[tuple] = []
        # set of closure C function names already emitted
        self._emitted_closures: Set[str] = set()
        # current deferred list for detecting whether we need a frame
        self._has_defer: List[bool] = [False]
        # class AST map for inheritance field resolution
        self._class_map: dict = {}  # name -> ClassDecl or StructDecl
        # feature flags — set when corresponding nodes are encountered
        self._need_threads: bool = False  # go / actor
        self._need_sql:     bool = False  # @sql`...`
        self._actor_names:  Set[str] = set()  # declared actor class names
        # module import support
        self._source_dir: str = '.'     # directory of the source file being compiled
        self._module_prefix: str = ''   # namespace prefix for sub-transpiled modules
        self._imported_modules: dict = {}  # path -> True, prevents circular imports

    def _swap_to_preamble(self):
        """Save current lines/indent and start a new preamble buffer."""
        saved = (self._lines, self._indent_level)
        self._lines = []
        self._indent_level = 0
        return saved

    def _swap_restore(self, saved):
        """Move accumulated lines to _preamble_fns, restore saved state."""
        self._preamble_fns.extend(self._lines)
        self._lines, self._indent_level = saved

    # ── Output helpers ────────────────────────────────────────────────────────

    def _emit(self, line: str):
        self._lines.append('    ' * self._indent_level + line)

    def _emit_fwd(self, line: str):
        self._fwd.append(line)

    def _emit_cls(self, line: str):
        self._cls_defs.append(line)

    def _emit_init(self, line: str):
        self._init_stmts.append('    ' + line)

    def _ind(self):
        self._indent_level += 1

    def _ded(self):
        self._indent_level -= 1

    def _fresh_tmp(self) -> str:
        n = self._tmp_id; self._tmp_id += 1
        return f'_t{n}'

    def _fresh_closure(self) -> str:
        n = self._closure_id; self._closure_id += 1
        return f'_wk_closure_{n}'

    def _fresh_label(self) -> str:
        n = self._label_id; self._label_id += 1
        return f'_lbl{n}'

    # ── Scope helpers ─────────────────────────────────────────────────────────

    def _push_scope(self):
        self._scopes.append({})

    def _pop_scope(self):
        self._scopes.pop()

    def _define(self, waka_name: str, c_name: str):
        self._scopes[-1][waka_name] = c_name

    def _lookup(self, waka_name: str) -> Optional[str]:
        for scope in reversed(self._scopes):
            if waka_name in scope:
                return scope[waka_name]
        return None

    def _is_global_scope(self) -> bool:
        return len(self._scopes) == 1

    # ── Capture analysis ──────────────────────────────────────────────────────

    def _free_vars(self, node, local_params: Set[str]) -> Set[str]:
        """Collect names referenced in node that are not in local_params."""
        free: Set[str] = set()
        self._collect_free(node, local_params, set(), free)
        return free

    def _collect_free(self, node, params, local_defs, free):
        if node is None:
            return
        t = type(node)
        if t is Ident:
            name = node.name
            if name not in params and name not in local_defs and name not in _BUILTINS:
                free.add(name)
        elif t is Closure:
            inner_params = set(p if isinstance(p, str) else p.name for p in node.params)
            inner_local: Set[str] = set()
            self._collect_free(node.body, params | inner_params, inner_local, free)
        elif t is FnDecl:
            inner_params = set(p.name for p in node.params)
            inner_local: Set[str] = set()
            self._collect_free(node.body, params | inner_params, inner_local, free)
        elif t in (LetDecl, ConstDecl):
            if node.value:
                self._collect_free(node.value, params, local_defs, free)
            local_defs.add(node.name)
        elif t is ForInStmt:
            self._collect_free(node.iter, params, local_defs, free)
            inner_local = set(local_defs) | {node.var}
            self._collect_free(node.body, params, inner_local, free)
        else:
            # Recurse into all child nodes
            for attr in vars(node).values():
                if isinstance(attr, Node):
                    self._collect_free(attr, params, local_defs, free)
                elif isinstance(attr, list):
                    for item in attr:
                        if isinstance(item, Node):
                            self._collect_free(item, params, local_defs, free)
                        elif isinstance(item, tuple):
                            for sub in item:
                                if isinstance(sub, Node):
                                    self._collect_free(sub, params, local_defs, free)

    # ── Main entry point ──────────────────────────────────────────────────────

    def transpile(self, program: Program, source_dir: str = '.') -> str:
        self._source_dir = source_dir
        # Pass 0: macro expansion
        program = self._expand_macros_pass(program)
        # Pass 1: collect top-level declarations
        self._collect_decls(program.stmts)

        # Pass 2: emit top-level statements into a _wk_program() function
        self._emit_fwd('static void _wk_program(void);')
        self._lines.append('')
        self._lines.append('static void _wk_program(void) {')
        self._indent_level = 1
        self._has_defer = [False]
        self._push_scope()
        for stmt in program.stmts:
            # Skip pure declarations — already handled
            if isinstance(stmt, (FnDecl, ClassDecl, StructDecl, ImplBlock,
                                  EnumDecl, InterfaceDecl, MacroDecl, ActorDecl,
                                  ModuleDecl)):
                continue
            # Native .wk imports are handled in pass 1; foreign imports still
            # go through _emit_stmt so they emit a panic.
            if isinstance(stmt, ImportDecl) and not stmt.lang:
                continue
            self._emit_stmt(stmt)
        self._pop_scope()
        self._indent_level = 0
        self._lines.append('}')

        # Build the full output
        parts = []
        parts.append('/* Auto-generated by Wakawaka transpiler — do not edit. */')
        if self._need_sql:
            parts.append('#define WK_HAVE_SQL')
        if self._need_threads:
            parts.append('#define WK_HAVE_THREADS')
        parts.append('#include "waka_runtime.h"')
        parts.append('')
        # forward declarations first (before class descriptors that reference fns)
        if self._fwd:
            parts.extend(self._fwd)
            parts.append('')
        # class descriptors (WkClass structs, method tables)
        if self._cls_defs:
            parts.extend(self._cls_defs)
            parts.append('')
        # init function
        parts.append('static void _wk_init_user(void) {')
        parts.extend(self._init_stmts)
        parts.append('}')
        parts.append('')
        # lifted closures/inner-fns/defer functions (file scope, before callers)
        if self._preamble_fns:
            parts.extend(self._preamble_fns)
            parts.append('')
        # top-level function bodies + _wk_program
        parts.extend(self._lines)
        parts.append('')
        # main
        parts.append('int main(int argc, char **argv) {')
        parts.append('    wk_argc = argc; wk_argv = argv;')
        parts.append('    wk_runtime_init();')
        parts.append('    _wk_init_user();')
        parts.append('    _wk_program();')
        parts.append('    return 0;')
        parts.append('}')
        return '\n'.join(parts) + '\n'

    # ── Declaration collector (pass 1) ────────────────────────────────────────

    def _all_fields(self, node) -> list:
        """Return all fields (parent-first, then own) for a ClassDecl/StructDecl."""
        parent_fields = []
        parent_name = getattr(node, 'parent', None)
        if parent_name and parent_name in self._class_map:
            parent_fields = self._all_fields(self._class_map[parent_name])
        own_names = {pf.name for pf in parent_fields}
        return parent_fields + [f for f in node.fields if f.name not in own_names]

    def _collect_decls(self, stmts):
        # First pass: register all class/struct/actor names so inheritance can be resolved
        for stmt in stmts:
            if isinstance(stmt, (ClassDecl, StructDecl)):
                self._class_map[stmt.name] = stmt
            elif isinstance(stmt, ActorDecl):
                self._class_map[stmt.name] = stmt
                self._actor_names.add(stmt.name)
        for stmt in stmts:
            if isinstance(stmt, FnDecl):
                self._emit_fn(stmt, toplevel=True)
            elif isinstance(stmt, ClassDecl):
                self._emit_class(stmt)
            elif isinstance(stmt, StructDecl):
                self._emit_struct(stmt)
            elif isinstance(stmt, ImplBlock):
                self._emit_impl(stmt)
            elif isinstance(stmt, EnumDecl):
                self._emit_enum(stmt)
            elif isinstance(stmt, ActorDecl):
                self._emit_actor(stmt)
            elif isinstance(stmt, ImportDecl):
                if not stmt.lang:
                    self._emit_wk_import(stmt)
            elif isinstance(stmt, ModuleDecl):
                self._collect_decls(stmt.body.stmts)
            elif isinstance(stmt, (LetDecl, ConstDecl)):
                # Top-level let/var/const → declare as a global so functions can reference it
                pfx = self._module_prefix
                gname = f'wk_g_{pfx}{_c_id(stmt.name)}'
                self._define(stmt.name, gname)
                self._emit_fwd(f'static WkVal {gname};')

    # ── Function emission ─────────────────────────────────────────────────────

    def _emit_fn(self, node: FnDecl, toplevel=False,
                 method_of: str = None, extra_caps: list = None):
        if not node.name:
            return
        pfx = self._module_prefix
        cname = f'_wk_fn_{pfx}{_c_id(node.name)}'
        if method_of:
            # method_of already includes module prefix when applicable
            cname = f'_wk_{method_of}_{_c_id(node.name)}'

        sig = f'static WkVal {cname}(WkVal *_args, int _argc, WkFunc *_fn)'
        self._emit_fwd(sig + ';')

        # Emit body
        self._lines.append('')
        self._lines.append(sig + ' {')
        self._indent_level = 1
        self._push_scope()
        self._has_defer.append(False)

        # If method, 'self' is in captures[0] — define before params so
        # that the 'self' param is skipped during arg unpacking
        is_method = bool(method_of or extra_caps)
        if is_method:
            self._define('self', '_fn->captures[0].val')

        # Unpack parameters (skip 'self' for methods — it comes from captures)
        arg_idx = 0
        for i, p in enumerate(node.params):
            pname = p if isinstance(p, str) else p.name
            # For methods, skip the 'self' parameter — it's already bound via captures
            if is_method and pname == 'self':
                continue
            cvar = _mangle(pname)
            self._define(pname, cvar)
            # Variadic param: collect rest into a list
            if not isinstance(p, str) and getattr(p, 'variadic', False):
                self._emit(f'WkVal {cvar} = wk_make_list();')
                self._emit(f'for (int _vi = {arg_idx}; _vi < _argc; _vi++)')
                self._emit(f'    wk_list_push_raw({cvar}.as.list, _args[_vi]);')
            else:
                default = None
                if not isinstance(p, str) and getattr(p, 'default', None) is not None:
                    default = p.default
                if default is not None:
                    dexpr = self._emit_expr(default)
                    self._emit(f'WkVal {cvar} = (_argc > {arg_idx}) ? _args[{arg_idx}] : {dexpr};')
                else:
                    self._emit(f'WkVal {cvar} = (_argc > {arg_idx}) ? _args[{arg_idx}] : wk_none();')
            arg_idx += 1

        # Emit defer frame (always; will be optimised away by compiler if unused)
        self._emit('WkDeferFrame _df; wk_defer_push_frame(&_df);')

        # Body
        if node.body:
            for stmt in node.body.stmts:
                self._emit_stmt(stmt)

        # Implicit return
        self._emit('wk_defer_flush(&_df);')
        self._emit('return wk_none();')

        self._has_defer.pop()
        self._pop_scope()
        self._indent_level = 0
        self._lines.append('}')

        # Register in init (exclude 'self' from param count for methods)
        params_no_self = [p for p in node.params
                          if not (is_method and (p if isinstance(p,str) else p.name) == 'self')]
        nparams = len(params_no_self)
        pnames_c = 'NULL'
        if params_no_self:
            names = [f'"{p if isinstance(p,str) else p.name}"' for p in params_no_self]
            _pnames_prefix = f'{method_of}_' if method_of else ''
            arr_name = f'_wk_pnames_{_pnames_prefix}{_c_id(node.name)}'
            self._emit_cls(f'static const char *{arr_name}[] = {{{", ".join(names)}}};')
            pnames_c = arr_name
        wk_var = _mangle(node.name)
        if toplevel:
            gname = f'wk_g_{pfx}{node.name}'
            self._define(node.name, gname)
            self._emit_init(f'{gname} = wk_make_func("{node.name}", {cname}, '
                            f'NULL, 0, {pnames_c}, {nparams});')
            # declare global slot
            self._emit_fwd(f'static WkVal {gname};')

    # ── Class emission ────────────────────────────────────────────────────────

    def _emit_class(self, node: ClassDecl):
        pfx = self._module_prefix
        cname = pfx + node.name
        # Collect all fields (own + inherited) — for simplicity store all as own
        # Flatten all fields (own + inherited, parent-first) for correct struct layout
        field_names = [f.name for f in self._all_fields(node)]

        # Pre-register this class in scope so methods can reference the class by name
        # (e.g. Circle.new(...) called from operator+ inside Circle)
        gvar_pre = f'wk_g_cls_{cname}'
        self._define(node.name, gvar_pre)

        # Emit methods
        for method in node.methods:
            self._emit_fn(method, toplevel=False, method_of=cname,
                          extra_caps=['self'])

        # Method table
        if node.methods:
            methods_arr = f'_wk_{cname}_methods'
            method_entries = []
            for m in node.methods:
                mname = f'_wk_{cname}_{_c_id(m.name)}'
                # Exclude 'self' from param count for method table
                nmp = len([p for p in m.params
                           if (p if isinstance(p,str) else p.name) != 'self'])
                method_entries.append(
                    f'    {{"{m.name}", &(WkFunc){{1, "{m.name}", {mname}, NULL, 0, NULL, {nmp}}}}}'
                )
                self._emit_fwd(f'static WkVal _wk_{cname}_{_c_id(m.name)}'
                               f'(WkVal *_args, int _argc, WkFunc *_fn);')
            self._emit_cls(f'static WkMethod {methods_arr}[] = {{')
            for e in method_entries:
                self._emit_cls(e + ',')
            self._emit_cls('};')
        else:
            methods_arr = 'NULL'

        # Field names
        if field_names:
            fnames_arr = f'_wk_{cname}_fnames'
            fnames_c = ', '.join(f'"{f}"' for f in field_names)
            self._emit_cls(f'static const char *{fnames_arr}[] = {{{fnames_c}}};')
        else:
            fnames_arr = 'NULL'

        # Parent
        parent_expr = 'NULL'
        if node.parent:
            parent_expr = f'&_wk_cls_{node.parent}'

        # Class descriptor
        nmethods = len(node.methods)
        nfields = len(field_names)
        self._emit_cls(f'static WkClass _wk_cls_{cname} = {{')
        self._emit_cls(f'    "{cname}", {parent_expr},')
        self._emit_cls(f'    {methods_arr if nmethods else "NULL"}, {nmethods},')
        self._emit_cls(f'    {fnames_arr}, {nfields}')
        self._emit_cls('};')

        # Global var (already pre-registered in scope above)
        gvar = f'wk_g_cls_{cname}'
        self._emit_fwd(f'static WkVal {gvar};')
        self._emit_init(f'{gvar} = wk_make_class(&_wk_cls_{cname});')

    def _emit_struct(self, node: StructDecl):
        # Treat as a class with no methods; flatten inherited fields
        field_names = [f.name for f in self._all_fields(node)]
        cname = node.name
        if field_names:
            fnames_arr = f'_wk_{cname}_fnames'
            self._emit_cls(f'static const char *{fnames_arr}[] = '
                           f'{{{", ".join(f"{chr(34)}{f}{chr(34)}" for f in field_names)}}};')
        else:
            fnames_arr = 'NULL'
        self._emit_cls(f'static WkClass _wk_cls_{cname} = {{"{cname}", NULL, NULL, 0, {fnames_arr}, {len(field_names)}}};')
        gvar = f'wk_g_cls_{cname}'
        self._emit_fwd(f'static WkVal {gvar};')
        self._emit_init(f'{gvar} = wk_make_class(&_wk_cls_{cname});')
        self._define(cname, gvar)

    def _emit_impl(self, node: ImplBlock):
        for method in node.methods:
            self._emit_fn(method, toplevel=False, method_of=node.target,
                          extra_caps=['self'])
            # patch method into class descriptor via init
            m_cname = f'_wk_{node.target}_{method.name}'
            nmp = len(method.params)
            # We need to add the method to the class at runtime
            # Simplest: emit a runtime registration call
            # For v1, methods added via impl are appended via a helper
            self._emit_init(
                f'/* impl {node.target}.{method.name} registered via class methods array */'
            )

    def _emit_enum(self, node: EnumDecl):
        for variant in node.variants:
            vname = variant.name
            if not variant.fields:
                # Fieldless: global string constant
                gvar = f'wk_g_enum_{vname}'
                self._emit_fwd(f'static WkVal {gvar};')
                self._emit_init(f'{gvar} = wk_make_strz("{vname}");')
                self._define(vname, gvar)
            else:
                # Constructor function
                cfname = f'_wk_enum_ctor_{vname}'
                sig = f'static WkVal {cfname}(WkVal *_args, int _argc, WkFunc *_fn)'
                self._emit_fwd(sig + ';')
                self._lines.append('')
                self._lines.append(sig + ' {')
                self._lines.append('    (void)_fn;')
                self._lines.append(f'    static const char *_fn[] = {{"_variant", "value"}};')
                self._lines.append(f'    static WkClass _cls = {{"{vname}", NULL, NULL, 0, _fn, 2}};')
                self._lines.append(f'    WkVal _obj = wk_make_obj(&_cls);')
                self._lines.append(f'    wk_obj_set_field(_obj, "_variant", wk_make_strz("{vname}"));')
                # If single field, store as "value"; multiple → tuple
                if len(variant.fields) == 1:
                    self._lines.append(f'    wk_obj_set_field(_obj, "value", (_argc>0)?_args[0]:wk_none());')
                else:
                    self._lines.append(f'    WkVal _targs[{len(variant.fields)}];')
                    for i in range(len(variant.fields)):
                        self._lines.append(f'    _targs[{i}] = (_argc>{i})?_args[{i}]:wk_none();')
                    self._lines.append(f'    wk_obj_set_field(_obj, "value", wk_make_tuple(_targs,{len(variant.fields)}));')
                self._lines.append('    return _obj;')
                self._lines.append('}')
                gvar = f'wk_g_enum_{vname}'
                self._emit_fwd(f'static WkVal {gvar};')
                self._emit_init(f'{gvar} = wk_make_func("{vname}", {cfname}, NULL, 0, NULL, {len(variant.fields)});')
                self._define(vname, gvar)

    # ── Closure emission ──────────────────────────────────────────────────────

    def _emit_closure(self, node: Closure) -> str:
        """Emit a lifted closure C function and return its WkVal constructor expr."""
        cname = self._fresh_closure()
        params = [p if isinstance(p, str) else p.name for p in node.params]
        param_set = set(params)

        # Capture analysis: find free vars not in params or builtins
        free = self._free_vars(node.body if isinstance(node.body, Node) else node, param_set)
        # Filter to only vars currently in scope
        caps = []
        for name in sorted(free):
            cvar = self._lookup(name)
            if cvar and name not in _BUILTINS:
                caps.append((name, cvar))

        # Forward decl
        sig = f'static WkVal {cname}(WkVal *_args, int _argc, WkFunc *_fn)'
        self._emit_fwd(sig + ';')

        # Emit closure body into preamble (not inline in the current function)
        saved = self._swap_to_preamble()
        self._lines.append('')
        self._lines.append(sig + ' {')
        self._lines.append('    WkDeferFrame _df; wk_defer_push_frame(&_df);')
        self._indent_level = 1
        self._push_scope()

        # Restore captures
        for i, (name, _) in enumerate(caps):
            cvar = _mangle(name)
            self._define(name, cvar)
            self._emit(f'WkVal {cvar} = _fn->captures[{i}].val;')

        # Unpack params
        for i, pname in enumerate(params):
            cvar = _mangle(pname)
            self._define(pname, cvar)
            self._emit(f'WkVal {cvar} = (_argc > {i}) ? _args[{i}] : wk_none();')

        # Body
        if isinstance(node.body, Block):
            for stmt in node.body.stmts:
                self._emit_stmt(stmt)
        else:
            # Expression body
            result = self._emit_expr(node.body)
            self._emit(f'wk_defer_flush(&_df);')
            self._emit(f'return {result};')

        self._emit('wk_defer_flush(&_df);')
        self._emit('return wk_none();')
        self._pop_scope()
        self._indent_level = 0
        self._lines.append('}')
        self._swap_restore(saved)

        # Build the WkVal constructor expression (emitted at call site)
        if caps:
            cap_tmp = self._fresh_tmp()
            self._emit(f'WkCapture {cap_tmp}[] = {{')
            for name, cvar in caps:
                self._emit(f'    {{"{name}", {cvar}}},')
            self._emit('};')
            nparams_str = ', '.join(f'"{p}"' for p in params)
            pnames_c = 'NULL'
            if params:
                pnames_arr = self._fresh_tmp()
                self._emit(f'static const char *{pnames_arr}[] = {{{nparams_str}}};')
                pnames_c = pnames_arr
            return (f'wk_make_func("<closure>", {cname}, '
                    f'{cap_tmp}, {len(caps)}, {pnames_c}, {len(params)})')
        else:
            nparams_str = ', '.join(f'"{p}"' for p in params)
            pnames_c = 'NULL'
            if params:
                pnames_arr = self._fresh_tmp()
                self._emit(f'static const char *{pnames_arr}[] = {{{nparams_str}}};')
                pnames_c = pnames_arr
            return (f'wk_make_func("<closure>", {cname}, '
                    f'NULL, 0, {pnames_c}, {len(params)})')

    # ── Statement emitters ────────────────────────────────────────────────────

    def _emit_stmt(self, node):
        t = type(node)
        if   t is ExprStmt:     self._emit_expr_stmt(node)
        elif t is LetDecl:      self._emit_let(node)
        elif t is ConstDecl:    self._emit_const(node)
        elif t is Assign:       self._emit_assign(node)
        elif t is IfStmt:       self._emit_if(node)
        elif t is WhileStmt:    self._emit_while(node)
        elif t is DoWhileStmt:  self._emit_do_while(node)
        elif t is ForInStmt:    self._emit_for_in(node)
        elif t is ForCStmt:     self._emit_for_c(node)
        elif t is MatchStmt:    self._emit_match(node)
        elif t is ReturnStmt:   self._emit_return(node)
        elif t is BreakStmt:    self._emit_break(node)
        elif t is ContinueStmt: self._emit_continue(node)
        elif t is DeferStmt:    self._emit_defer(node)
        elif t is Block:        self._emit_block(node)
        elif t is UnsafeBlock:  self._emit_block(node.body)
        elif t is AssertStmt:   self._emit_assert(node)
        elif t is PanicStmt:    self._emit_panic_stmt(node)
        elif t is FnDecl:
            # Inner function declaration — emit as a closure with captures
            cvar = _mangle(node.name)
            inner_cname = self._fresh_closure()
            params = [p if isinstance(p,str) else p.name for p in node.params]
            param_set = set(params)

            # Capture analysis: find free vars not in params or builtins
            free = self._free_vars(node.body if node.body else node, param_set)
            caps = []
            for name in sorted(free):
                cv = self._lookup(name)
                if cv and name not in _BUILTINS:
                    caps.append((name, cv))

            sig = f'static WkVal {inner_cname}(WkVal *_args, int _argc, WkFunc *_fn)'
            self._emit_fwd(sig + ';')
            # Emit body into preamble (file scope), not inline in the enclosing fn
            saved = self._swap_to_preamble()
            self._lines.append('')
            self._lines.append(sig + ' {')
            self._indent_level = 1
            self._push_scope()
            self._emit('WkDeferFrame _df; wk_defer_push_frame(&_df);')
            # Restore captures
            for ci, (cname_, _) in enumerate(caps):
                ccvar = _mangle(cname_)
                self._define(cname_, ccvar)
                self._emit(f'WkVal {ccvar} = _fn->captures[{ci}].val;')
            for i, pname in enumerate(params):
                cv = _mangle(pname); self._define(pname, cv)
                self._emit(f'WkVal {cv} = (_argc > {i}) ? _args[{i}] : wk_none();')
            if node.body:
                for s in node.body.stmts:
                    self._emit_stmt(s)
            self._emit('wk_defer_flush(&_df);')
            self._emit('return wk_none();')
            self._pop_scope()
            self._indent_level = 0
            self._lines.append('}')
            self._swap_restore(saved)
            # Build capture array
            if caps:
                cap_tmp = self._fresh_tmp()
                self._emit(f'WkCapture {cap_tmp}[] = {{')
                for cname_, ccvar in caps:
                    self._emit(f'    {{"{cname_}", {ccvar}}},')
                self._emit('};')
                pnames_c = 'NULL'
                if params:
                    pnames_arr = self._fresh_tmp()
                    pnames_str = ', '.join(f'"{p}"' for p in params)
                    self._emit(f'static const char *{pnames_arr}[] = {{{pnames_str}}};')
                    pnames_c = pnames_arr
                self._emit(f'WkVal {cvar} = wk_make_func("{node.name}", {inner_cname}, '
                           f'{cap_tmp}, {len(caps)}, {pnames_c}, {len(params)});')
            else:
                self._emit(f'WkVal {cvar} = wk_make_func("{node.name}", {inner_cname}, NULL, 0, NULL, {len(params)});')
            self._define(node.name, cvar)
        elif t is GoStmt:
            self._emit_go(node)
        elif t is PyBlock:
            self._emit(f'wk_panic("pyblock is not supported in compiled mode — use \'waka run\' for pyblock");')
        elif t is ImportDecl:
            if node.lang:
                self._emit(f'wk_panic("import {node.lang} not supported in compiled mode");')
            else:
                self._emit_wk_import(node)
        elif t is ReceiveStmt:
            self._emit_receive(node)
        elif t in (ActorDecl, SpawnExpr):
            pass  # handled in pass 1 / as expressions
        elif t in (ClassDecl, StructDecl, ImplBlock, EnumDecl,
                   InterfaceDecl, MacroDecl, ModuleDecl):
            pass  # handled in pass 1
        else:
            # Unknown — skip
            pass

    def _emit_expr_stmt(self, node: ExprStmt):
        # Special: if the expr is a MatchStmt used as expression, handle it
        if isinstance(node.expr, MatchStmt):
            self._emit_match(node.expr)
            return
        expr = self._emit_expr(node.expr)
        self._emit(f'(void)({expr});')

    def _emit_let(self, node: LetDecl):
        # Check if already declared as a global (top-level var/let)
        existing = self._lookup(node.name)
        if existing and existing.startswith('wk_g_'):
            cvar = existing
            if node.value is not None:
                val = self._emit_expr(node.value)
                self._emit(f'{cvar} = {val};')
            else:
                self._emit(f'{cvar} = wk_none();')
        else:
            cvar = _mangle(node.name)
            self._define(node.name, cvar)
            if node.value is not None:
                val = self._emit_expr(node.value)
                self._emit(f'WkVal {cvar} = {val};')
            else:
                self._emit(f'WkVal {cvar} = wk_none();')

    def _emit_const(self, node: ConstDecl):
        # Check if already declared as a global (top-level const)
        existing = self._lookup(node.name)
        if existing and existing.startswith('wk_g_'):
            cvar = existing
            val = self._emit_expr(node.value) if node.value is not None else 'wk_none()'
            self._emit(f'{cvar} = {val};')
        else:
            cvar = _mangle(node.name)
            self._define(node.name, cvar)
            val = self._emit_expr(node.value) if node.value is not None else 'wk_none()'
            self._emit(f'WkVal {cvar} = {val};')

    def _emit_assign(self, node: Assign):
        val_expr = self._emit_expr(node.value)
        target = node.target

        # Compound assignment operators
        if node.op != '=':
            op_core = node.op[:-1]  # e.g. '+=' → '+'
            op_fn = _OP_MAP.get(op_core)
            cur = self._emit_expr(target)
            if op_fn:
                val_expr = f'{op_fn}({cur}, {val_expr})'
            else:
                val_expr = f'{cur} /* {node.op} */ {val_expr}'

        self._emit_assign_target(target, val_expr)

    def _emit_assign_target(self, target, val_expr: str):
        if isinstance(target, Ident):
            cvar = self._lookup(target.name)
            if cvar:
                self._emit(f'{cvar} = {val_expr};')
            else:
                # Auto-define
                cvar = _mangle(target.name)
                self._define(target.name, cvar)
                self._emit(f'WkVal {cvar} = {val_expr};')
        elif isinstance(target, MemberExpr):
            obj = self._emit_expr(target.obj)
            self._emit(f'wk_member_set({obj}, "{target.member}", {val_expr});')
        elif isinstance(target, IndexExpr):
            obj = self._emit_expr(target.obj)
            idx = self._emit_expr(target.index)
            self._emit(f'wk_index_set({obj}, {idx}, {val_expr});')
        else:
            self._emit(f'wk_panic("unsupported assignment target");')

    def _emit_if(self, node: IfStmt):
        cond = self._emit_expr(node.cond)
        self._emit(f'if (wk_truthy({cond})) {{')
        self._ind(); self._push_scope()
        self._emit_block(node.then)
        self._pop_scope(); self._ded()
        for elif_cond, elif_body in node.elseifs:
            ec = self._emit_expr(elif_cond)
            self._emit(f'}} else if (wk_truthy({ec})) {{')
            self._ind(); self._push_scope()
            self._emit_block(elif_body)
            self._pop_scope(); self._ded()
        if node.else_:
            self._emit('} else {')
            self._ind(); self._push_scope()
            self._emit_block(node.else_)
            self._pop_scope(); self._ded()
        self._emit('}')

    def _emit_while(self, node: WhileStmt):
        brk = self._fresh_label()
        cnt = self._fresh_label()
        self._loop_stack.append((node.label, brk, cnt))
        # Evaluate condition inside the loop so it's re-evaluated each iteration
        # (necessary for conditions with function calls or side effects)
        self._emit(f'for (;;) {{')
        self._ind(); self._push_scope()
        self._emit(f'{cnt}:;')
        cond = self._emit_expr(node.cond)
        self._emit(f'if (!wk_truthy({cond})) break;')
        self._emit_block(node.body)
        self._pop_scope(); self._ded()
        self._emit(f'}}')
        self._emit(f'{brk}:;')
        self._loop_stack.pop()

    def _emit_do_while(self, node: DoWhileStmt):
        brk = self._fresh_label()
        cnt = self._fresh_label()
        self._loop_stack.append((None, brk, cnt))
        self._emit('do {')
        self._ind(); self._push_scope()
        self._emit(f'{cnt}:;')
        self._emit_block(node.body)
        self._pop_scope(); self._ded()
        cond = self._emit_expr(node.cond)
        self._emit(f'}} while (wk_truthy({cond}));')
        self._emit(f'{brk}:;')
        self._loop_stack.pop()

    def _emit_for_in(self, node: ForInStmt):
        brk = self._fresh_label()
        cnt = self._fresh_label()
        self._loop_stack.append((node.label, brk, cnt))

        iter_tmp = self._fresh_tmp()
        iter_expr = self._emit_expr(node.iter)
        self._emit(f'{{')
        self._ind()
        self._emit(f'WkVal {iter_tmp} = {iter_expr};')
        var_c = _mangle(node.var)
        waka_lbl = node.label

        def _branch_body(cnt_b):
            """Emit loop body with per-branch continue label to avoid duplicates."""
            self._loop_stack[-1] = (waka_lbl, brk, cnt_b)
            self._emit(f'{cnt_b}:;')
            for s in node.body.stmts:
                self._emit_stmt(s)

        # Range
        ri = self._fresh_tmp(); re = self._fresh_tmp()
        self._emit(f'if ({iter_tmp}.tag == WK_RANGE) {{')
        self._ind()
        self._emit(f'int64_t {ri} = {iter_tmp}.as.rng.start;')
        self._emit(f'int64_t {re} = {iter_tmp}.as.rng.end + ({iter_tmp}.as.rng.inclusive ? 1 : 0);')
        self._emit(f'for (; {ri} < {re}; {ri}++) {{')
        self._ind(); self._push_scope()
        self._define(node.var, var_c)
        self._emit(f'WkVal {var_c} = wk_int({ri});')
        _branch_body(self._fresh_label())
        self._pop_scope(); self._ded()
        self._emit(f'}}')
        self._ded()
        # List
        li = self._fresh_tmp()
        self._emit(f'}} else if ({iter_tmp}.tag == WK_LIST) {{')
        self._ind()
        self._emit(f'WkList *_lst_{li} = {iter_tmp}.as.list;')
        self._emit(f'for (size_t {li} = 0; {li} < _lst_{li}->len; {li}++) {{')
        self._ind(); self._push_scope()
        self._define(node.var, var_c)
        self._emit(f'WkVal {var_c} = _lst_{li}->items[{li}];')
        _branch_body(self._fresh_label())
        self._pop_scope(); self._ded()
        self._emit(f'}}')
        self._ded()
        # String (chars)
        si = self._fresh_tmp()
        self._emit(f'}} else if ({iter_tmp}.tag == WK_STR) {{')
        self._ind()
        self._emit(f'for (size_t {si} = 0; {si} < {iter_tmp}.as.str->len; {si}++) {{')
        self._ind(); self._push_scope()
        self._define(node.var, var_c)
        self._emit(f'WkVal {var_c} = wk_make_str({iter_tmp}.as.str->data+{si}, 1);')
        _branch_body(self._fresh_label())
        self._pop_scope(); self._ded()
        self._emit(f'}}')
        self._ded()
        # Map (keys)
        mi = self._fresh_tmp()
        self._emit(f'}} else if ({iter_tmp}.tag == WK_MAP) {{')
        self._ind()
        self._emit(f'WkVal _mkeys_{mi} = wk_map_keys({iter_tmp});')
        self._emit(f'WkList *_mlst_{mi} = _mkeys_{mi}.as.list;')
        self._emit(f'for (size_t {mi} = 0; {mi} < _mlst_{mi}->len; {mi}++) {{')
        self._ind(); self._push_scope()
        self._define(node.var, var_c)
        self._emit(f'WkVal {var_c} = _mlst_{mi}->items[{mi}];')
        _branch_body(self._fresh_label())
        self._pop_scope(); self._ded()
        self._emit(f'}}')
        self._ded()
        # Tuple
        ti = self._fresh_tmp()
        self._emit(f'}} else if ({iter_tmp}.tag == WK_TUPLE) {{')
        self._ind()
        self._emit(f'WkTuple *_tup_{ti} = {iter_tmp}.as.tup;')
        self._emit(f'for (size_t {ti} = 0; {ti} < _tup_{ti}->len; {ti}++) {{')
        self._ind(); self._push_scope()
        self._define(node.var, var_c)
        self._emit(f'WkVal {var_c} = _tup_{ti}->items[{ti}];')
        _branch_body(self._fresh_label())
        self._pop_scope(); self._ded()
        self._emit(f'}}')
        self._ded()
        # Fallback
        self._emit(f'}} else {{')
        self._ind()
        self._emit(f'wk_panic("for-in: value is not iterable (tag=%d)", {iter_tmp}.tag);')
        self._ded()
        self._emit(f'}}')
        self._ded()
        self._emit(f'}}')
        self._emit(f'{brk}:;')
        self._loop_stack.pop()

    def _emit_for_c(self, node: ForCStmt):
        brk = self._fresh_label()
        cnt = self._fresh_label()
        self._loop_stack.append((node.label, brk, cnt))
        self._emit('{')
        self._ind(); self._push_scope()
        # Init
        if node.init:
            if isinstance(node.init, LetDecl):
                cvar = _mangle(node.init.name)
                self._define(node.init.name, cvar)
                val = self._emit_expr(node.init.value) if node.init.value else 'wk_none()'
                self._emit(f'WkVal {cvar} = {val};')
            else:
                self._emit_stmt(node.init)
        # while loop for condition
        if node.cond:
            cond = self._emit_expr(node.cond)
            self._emit(f'while (wk_truthy({cond})) {{')
        else:
            self._emit('while (1) {')
        self._ind(); self._push_scope()
        self._emit(f'{cnt}:;')
        for s in node.body.stmts:
            self._emit_stmt(s)
        # Step
        if node.step:
            if isinstance(node.step, Assign):
                self._emit_assign(node.step)
            else:
                step_expr = self._emit_expr(node.step)
                self._emit(f'(void)({step_expr});')
        # Re-evaluate condition at end of loop body
        if node.cond:
            cond2 = self._emit_expr(node.cond)
            self._emit(f'if (!wk_truthy({cond2})) break;')
        self._pop_scope(); self._ded()
        self._emit('}')
        self._pop_scope(); self._ded()
        self._emit('}')
        self._emit(f'{brk}:;')
        self._loop_stack.pop()

    _STMT_TYPES = (
        LetDecl, ConstDecl, Assign, ExprStmt, ReturnStmt, BreakStmt,
        ContinueStmt, DeferStmt, GoStmt, AssertStmt, PanicStmt,
        IfStmt, WhileStmt, DoWhileStmt, ForInStmt, ForCStmt,
        MatchStmt, UnsafeBlock, ImportDecl, ReceiveStmt,
        FnDecl, ClassDecl, StructDecl, ImplBlock, EnumDecl,
        InterfaceDecl, MacroDecl, ActorDecl, ModuleDecl,
    )

    def _emit_match_arm_body(self, body):
        """Emit a match arm body which can be a Block, statement, or expression."""
        if body is None:
            return
        if isinstance(body, Block):
            for s in body.stmts:
                self._emit_stmt(s)
        elif isinstance(body, self._STMT_TYPES):
            self._emit_stmt(body)
        else:
            r = self._emit_expr(body)
            self._emit(f'(void)({r});')

    def _emit_match(self, node: MatchStmt):
        subject = self._emit_expr(node.expr)
        subj_tmp = self._fresh_tmp()
        matched_tmp = self._fresh_tmp()
        self._emit('{')
        self._ind()
        self._emit(f'WkVal {subj_tmp} = {subject};')
        self._emit(f'int {matched_tmp} = 0;')
        for arm in node.arms:
            binds = []
            check = self._pattern_check(arm.pattern, subj_tmp, binds)
            if arm.guard:
                guard_expr = None  # evaluated after binds
            self._emit(f'if (!{matched_tmp} && ({check})) {{')
            self._ind(); self._push_scope()
            for bind_line in binds:
                self._emit(bind_line)
            if arm.guard:
                ge = self._emit_expr(arm.guard)
                self._emit(f'if (wk_truthy({ge})) {{')
                self._ind()
                self._emit(f'{matched_tmp} = 1;')
                self._emit_match_arm_body(arm.body)
                self._ded()
                self._emit('}')
            else:
                self._emit(f'{matched_tmp} = 1;')
                self._emit_match_arm_body(arm.body)
            self._pop_scope(); self._ded()
            self._emit('}')
        self._ded()
        self._emit('}')

    def _emit_return(self, node: ReturnStmt):
        if node.value:
            val = self._emit_expr(node.value)
        else:
            val = 'wk_none()'
        self._emit(f'wk_defer_flush(&_df);')
        self._emit(f'return {val};')

    def _emit_break(self, node: BreakStmt):
        # Find matching label
        for waka_lbl, brk, cnt in reversed(self._loop_stack):
            if node.label is None or node.label == waka_lbl:
                self._emit(f'goto {brk};')
                return
        self._emit('break; /* unmatched break */')

    def _emit_continue(self, node: ContinueStmt):
        for waka_lbl, brk, cnt in reversed(self._loop_stack):
            if node.label is None or node.label == waka_lbl:
                self._emit(f'goto {cnt};')
                return
        self._emit('continue; /* unmatched continue */')

    def _emit_defer(self, node: DeferStmt):
        # Build a no-arg closure for the deferred expression
        defer_cname = self._fresh_closure()
        # Determine captured vars
        free = self._free_vars(node.expr, set())
        caps = []
        for name in sorted(free):
            cvar = self._lookup(name)
            if cvar and name not in _BUILTINS:
                caps.append((name, cvar))

        sig = f'static WkVal {defer_cname}(WkVal *_args, int _argc, WkFunc *_fn)'
        self._emit_fwd(sig + ';')
        # Emit defer body into preamble (file scope)
        saved = self._swap_to_preamble()
        self._lines.append('')
        self._lines.append(sig + ' {')
        self._lines.append('    (void)_args; (void)_argc;')
        self._indent_level = 1
        self._push_scope()
        for i, (name, _) in enumerate(caps):
            cvar = _mangle(name); self._define(name, cvar)
            self._emit(f'WkVal {cvar} = _fn->captures[{i}].val;')
        expr = self._emit_expr(node.expr)
        self._emit(f'(void)({expr});')
        self._pop_scope()
        self._indent_level = 0
        self._lines.append('    return wk_none();')
        self._lines.append('}')
        self._swap_restore(saved)

        # Register the defer
        if caps:
            cap_tmp = self._fresh_tmp()
            self._emit(f'WkCapture {cap_tmp}[] = {{')
            for name, cvar in caps:
                self._emit(f'    {{"{name}", {cvar}}},')
            self._emit('};')
            self._emit(f'wk_defer_register(&_df, wk_make_func("<defer>", {defer_cname}, '
                       f'{cap_tmp}, {len(caps)}, NULL, 0));')
        else:
            self._emit(f'wk_defer_register(&_df, wk_make_func("<defer>", {defer_cname}, '
                       f'NULL, 0, NULL, 0));')

    def _emit_block(self, node: Block):
        for s in node.stmts:
            self._emit_stmt(s)

    def _emit_assert(self, node: AssertStmt):
        cond = self._emit_expr(node.cond)
        if node.msg:
            msg = self._emit_expr(node.msg)
            self._emit(f'if (!wk_truthy({cond})) wk_panic("%s", wk_to_cstr({msg}));')
        else:
            self._emit(f'if (!wk_truthy({cond})) wk_panic("assertion failed");')

    def _emit_panic_stmt(self, node: PanicStmt):
        msg = self._emit_expr(node.msg)
        self._emit(f'{{ char *_pm = wk_to_cstr({msg}); wk_panic("%s", _pm); }}')

    # ── Pattern matching helpers ──────────────────────────────────────────────

    def _pattern_check(self, pat, val_expr: str, binds: List[str]) -> str:
        """Return a C boolean expression for pat matching val_expr.
        Side effect: appends 'WkVal cvar = ...;' lines to binds."""
        t = type(pat)
        if t is WildcardPat:
            return '1'
        elif t is NonePat:
            return f'({val_expr}.tag == WK_NONE)'
        elif t is IdentPat:
            cvar = _mangle(pat.name)
            binds.append(f'WkVal {cvar} = {val_expr};')
            self._define(pat.name, cvar)
            return '1'
        elif t is LitPat:
            lit_expr = self._emit_expr(pat.value)
            return f'wk_equal({val_expr}, {lit_expr})'
        elif t is RangePat:
            lo = self._emit_expr(pat.start)
            hi = self._emit_expr(pat.end)
            if pat.inclusive:
                return (f'(wk_truthy(wk_cmp_ge({val_expr}, {lo})) && '
                        f'wk_truthy(wk_cmp_le({val_expr}, {hi})))')
            else:
                return (f'(wk_truthy(wk_cmp_ge({val_expr}, {lo})) && '
                        f'wk_truthy(wk_cmp_lt({val_expr}, {hi})))')
        elif t is SomePat:
            inner_tmp = self._fresh_tmp()
            binds.append(f'WkVal {inner_tmp} = ({val_expr}.tag==WK_SOME) ? *{val_expr}.as.inner : wk_none();')
            inner_binds = []
            inner_check = self._pattern_check(pat.inner, inner_tmp, inner_binds)
            binds.extend(inner_binds)
            return f'({val_expr}.tag == WK_SOME && ({inner_check}))'
        elif t is OkPat:
            inner_tmp = self._fresh_tmp()
            binds.append(f'WkVal {inner_tmp} = ({val_expr}.tag==WK_OK) ? *{val_expr}.as.inner : wk_none();')
            inner_binds = []
            inner_check = self._pattern_check(pat.inner, inner_tmp, inner_binds)
            binds.extend(inner_binds)
            return f'({val_expr}.tag == WK_OK && ({inner_check}))'
        elif t is ErrPat:
            inner_tmp = self._fresh_tmp()
            binds.append(f'WkVal {inner_tmp} = ({val_expr}.tag==WK_ERR) ? *{val_expr}.as.inner : wk_none();')
            inner_binds = []
            inner_check = self._pattern_check(pat.inner, inner_tmp, inner_binds)
            binds.extend(inner_binds)
            return f'({val_expr}.tag == WK_ERR && ({inner_check}))'
        elif t is TuplePat:
            checks = []
            checks.append(f'({val_expr}.tag == WK_TUPLE && {val_expr}.as.tup->len == {len(pat.elements)})')
            for i, elem in enumerate(pat.elements):
                elem_tmp = self._fresh_tmp()
                binds.append(f'WkVal {elem_tmp} = ({val_expr}.tag==WK_TUPLE && {val_expr}.as.tup->len>{i}) ? {val_expr}.as.tup->items[{i}] : wk_none();')
                elem_binds = []
                elem_check = self._pattern_check(elem, elem_tmp, elem_binds)
                binds.extend(elem_binds)
                checks.append(f'({elem_check})')
            return ' && '.join(checks)
        elif t is StructPat:
            checks = [f'({val_expr}.tag == WK_OBJ)']
            for fname, fpat in pat.fields:
                field_tmp = self._fresh_tmp()
                binds.append(f'WkVal {field_tmp} = ({val_expr}.tag==WK_OBJ) ? wk_obj_get_field({val_expr}, "{fname}") : wk_none();')
                field_binds = []
                fcheck = self._pattern_check(fpat, field_tmp, field_binds)
                binds.extend(field_binds)
                checks.append(f'({fcheck})')
            return ' && '.join(checks)
        elif t is EnumPat:
            checks = [f'({val_expr}.tag == WK_OBJ)']
            vname_tmp = self._fresh_tmp()
            binds.append(f'WkVal {vname_tmp} = ({val_expr}.tag==WK_OBJ) ? wk_obj_get_field({val_expr}, "_variant") : wk_none();')
            checks.append(f'({vname_tmp}.tag==WK_STR && strcmp({vname_tmp}.as.str->data, "{pat.name}")==0)')
            if pat.inner:
                inner_tmp = self._fresh_tmp()
                binds.append(f'WkVal {inner_tmp} = ({val_expr}.tag==WK_OBJ) ? wk_obj_get_field({val_expr}, "value") : wk_none();')
                inner_binds = []
                icheck = self._pattern_check(pat.inner, inner_tmp, inner_binds)
                binds.extend(inner_binds)
                checks.append(f'({icheck})')
            return ' && '.join(checks)
        elif t is OrPat:
            # Both sides must produce same bindings (or none)
            left_binds: List[str] = []
            right_binds: List[str] = []
            lcheck = self._pattern_check(pat.left, val_expr, left_binds)
            rcheck = self._pattern_check(pat.right, val_expr, right_binds)
            binds.extend(left_binds)  # use left bindings
            return f'(({lcheck}) || ({rcheck}))'
        else:
            return '1'  # unknown pattern — always match

    # ── Expression emitters ───────────────────────────────────────────────────

    def _emit_expr(self, node) -> str:
        """Emit any needed setup statements and return a C expression string."""
        if node is None:
            return 'wk_none()'
        t = type(node)
        if   t is IntLit:    return f'wk_int({node.value}LL)'
        elif t is FloatLit:
            f_str = repr(node.value)
            if 'inf' in f_str.lower(): f_str = '(1.0/0.0)' if node.value > 0 else '(-1.0/0.0)'
            elif 'nan' in f_str.lower(): f_str = '(0.0/0.0)'
            return f'wk_float({f_str})'
        elif t is BoolLit:   return f'wk_bool({1 if node.value else 0})'
        elif t is NoneLit:   return 'wk_none()'
        elif t is StrLit:    return f'wk_make_strz("{_escape_c_str(node.value)}")'
        elif t is FStrLit:   return self._emit_fstr(node)
        elif t is ArrayLit:  return self._emit_array(node)
        elif t is MapLit:    return self._emit_map(node)
        elif t is TupleLit:  return self._emit_tuple(node)
        elif t is StructLit: return self._emit_struct_lit(node)
        elif t is Ident:     return self._emit_ident(node)
        elif t is MemberExpr: return self._emit_member(node)
        elif t is IndexExpr:  return self._emit_index(node)
        elif t is SliceExpr:  return self._emit_slice(node)
        elif t is BinOp:      return self._emit_binop(node)
        elif t is UnaryOp:    return self._emit_unary(node)
        elif t is CallExpr:   return self._emit_call(node)
        elif t is Closure:    return self._emit_closure(node)
        elif t is TernaryExpr: return self._emit_ternary(node)
        elif t is CastExpr:   return self._emit_cast(node)
        elif t is PipeExpr:   return self._emit_pipe(node)
        elif t is SomeExpr:   return f'wk_make_some({self._emit_expr(node.value)})'
        elif t is OkExpr:     return f'wk_make_ok({self._emit_expr(node.value)})'
        elif t is ErrExpr:    return f'wk_make_err({self._emit_expr(node.value)})'
        elif t is PropagateExpr: return self._emit_propagate(node)
        elif t is RangeExpr:  return self._emit_range(node)
        elif t is OptChainExpr: return self._emit_opt_chain(node)
        elif t is NullCoalesceExpr: return self._emit_null_coalesce(node)
        elif t is IfStmt:     return self._emit_if_expr(node)
        elif t is MatchStmt:
            tmp = self._fresh_tmp()
            self._emit(f'WkVal {tmp} = wk_none();')
            # Wrap match to capture result
            subject = self._emit_expr(node.expr)
            subj_tmp = self._fresh_tmp()
            matched_tmp = self._fresh_tmp()
            self._emit(f'{{ WkVal {subj_tmp} = {subject}; int {matched_tmp}=0;')
            self._ind()
            for arm in node.arms:
                binds = []
                check = self._pattern_check(arm.pattern, subj_tmp, binds)
                self._emit(f'if (!{matched_tmp} && ({check})) {{')
                self._ind(); self._push_scope()
                for b in binds: self._emit(b)
                if arm.guard:
                    ge = self._emit_expr(arm.guard)
                    self._emit(f'if (wk_truthy({ge})) {{')
                    self._ind()
                    self._emit(f'{matched_tmp}=1;')
                    r = self._emit_expr(arm.body) if arm.body else 'wk_none()'
                    self._emit(f'{tmp} = {r};')
                    self._ded(); self._emit('}')
                else:
                    self._emit(f'{matched_tmp}=1;')
                    r = self._emit_expr(arm.body) if arm.body else 'wk_none()'
                    self._emit(f'{tmp} = {r};')
                self._pop_scope(); self._ded()
                self._emit('}')
            self._ded()
            self._emit('}')
            return tmp
        elif t is Block:
            # Block as expression: emit stmts, last one is the value
            tmp = self._fresh_tmp()
            self._emit(f'WkVal {tmp} = wk_none();')
            self._push_scope()
            stmts = node.stmts
            for i, s in enumerate(stmts):
                if i == len(stmts) - 1 and isinstance(s, ExprStmt):
                    r = self._emit_expr(s.expr)
                    self._emit(f'{tmp} = {r};')
                else:
                    self._emit_stmt(s)
            self._pop_scope()
            return tmp
        elif t is RefExpr:   return self._emit_expr(node.expr)  # refs are transparent
        elif t is DerefExpr: return self._emit_expr(node.expr)  # derefs too
        elif t is SizeofExpr: return 'wk_int(8)'  # 64-bit
        elif t is AwaitExpr: return self._emit_expr(node.expr)  # await is sync
        elif t is ComptimeExpr: return self._try_comptime_eval(node)
        elif t is SpawnExpr:
            return self._emit_spawn(node)
        elif t is ShellExpr:
            return self._emit_shell(node)
        elif t is SqlExpr:
            return self._emit_sql(node)
        elif t is PyBlock:
            self._emit('wk_panic("pyblock is not supported in compiled mode — use \'waka run\'");')
            return 'wk_none()'
        else:
            raise TranspilerError(f'unsupported expression type: {t.__name__} (line {getattr(node, "line", "?")})')

    def _emit_fstr(self, node: FStrLit) -> str:
        # Collect pairs: (literal_str, expr_val)
        # FStrLit.parts alternates: str literal, then expr node
        # We gather all into a flat list for wk_str_fmtbuild
        pairs = []  # list of (literal: str, expr_c: str)
        i = 0
        parts = node.parts
        lit_acc = ''
        while i < len(parts):
            p = parts[i]
            if isinstance(p, str):
                lit_acc += p
                i += 1
            else:
                # it's an expr node
                expr_c = self._emit_expr(p)
                pairs.append((lit_acc, expr_c))
                lit_acc = ''
                i += 1
        # trailing literal
        if lit_acc or not pairs:
            pairs.append((lit_acc, None))

        if not pairs:
            return 'wk_make_strz("")'

        # Build call to wk_str_fmtbuild
        args_parts = []
        for lit, expr_c in pairs:
            lit_c = f'"{_escape_c_str(lit)}"' if lit else 'NULL'
            if expr_c is None:
                # trailing literal with no expr — pass wk_none()
                args_parts.append(f'{lit_c}, wk_none()')
            else:
                args_parts.append(f'{lit_c}, {expr_c}')

        n = len(args_parts)
        args_str = ', '.join(args_parts)
        tmp = self._fresh_tmp()
        self._emit(f'WkVal {tmp} = wk_str_fmtbuild({n}, {args_str});')
        return tmp

    # ── Shell expression $`cmd {expr}` ───────────────────────────────────────

    def _emit_shell(self, node: ShellExpr) -> str:
        """Build command string from parts then call wk_shell_exec()."""
        pairs = []
        lit_acc = ''
        for p in node.parts:
            if isinstance(p, str):
                lit_acc += p
            else:
                pairs.append((lit_acc, self._emit_expr(p)))
                lit_acc = ''
        if lit_acc or not pairs:
            pairs.append((lit_acc, None))

        args_parts = []
        for lit, expr_c in pairs:
            lit_c = f'"{_escape_c_str(lit)}"' if lit else 'NULL'
            args_parts.append(f'{lit_c}, {expr_c}' if expr_c else f'{lit_c}, wk_none()')
        cmd_tmp = self._fresh_tmp()
        self._emit(f'WkVal {cmd_tmp} = wk_str_fmtbuild({len(args_parts)}, {", ".join(args_parts)});')
        result_tmp = self._fresh_tmp()
        self._emit(f'WkVal {result_tmp} = wk_shell_exec({cmd_tmp});')
        return result_tmp

    # ── SQL expression @sql`query {expr}` ────────────────────────────────────

    def _emit_sql(self, node: SqlExpr) -> str:
        """Build parameterized SQL query and call wk_sql_exec()."""
        self._need_sql = True
        query_parts = []
        param_exprs = []
        for p in node.parts:
            if isinstance(p, str):
                query_parts.append(p)
            else:
                query_parts.append('?')
                param_exprs.append(self._emit_expr(p))

        query_str = ''.join(query_parts)
        query_c   = f'"{_escape_c_str(query_str)}"'

        # Determine DB path: use "waka.db" by default
        db_path_c = '"waka.db"'

        tmp = self._fresh_tmp()
        if param_exprs:
            arr_tmp = self._fresh_tmp()
            self._emit(f'WkVal {arr_tmp}[] = {{{", ".join(param_exprs)}}};')
            self._emit(f'WkVal {tmp} = wk_sql_exec({db_path_c}, {query_c}, {arr_tmp}, {len(param_exprs)});')
        else:
            self._emit(f'WkVal {tmp} = wk_sql_exec({db_path_c}, {query_c}, NULL, 0);')
        return tmp

    # ── GoStmt: go expr ───────────────────────────────────────────────────────

    # ── Module import ────────────────────────────────────────────────────────

    def _emit_wk_import(self, node: ImportDecl) -> None:
        """Handle native Wakawaka module import: import "module.wk" [as alias];"""
        import os
        path = node.path
        if not path.endswith('.wk'):
            path += '.wk'
        full_path = os.path.join(self._source_dir, path)
        full_path = os.path.abspath(full_path)

        if not os.path.exists(full_path):
            self._emit(f'wk_panic("cannot find module \\"{node.path}\\"");')
            return

        # Prevent circular imports
        if full_path in self._imported_modules:
            # Already imported — just define the alias pointing to the existing map
            alias = node.alias or os.path.basename(path).replace('.wk', '')
            existing = self._imported_modules[full_path]
            self._define(alias, existing)
            return
        self._imported_modules[full_path] = None  # placeholder

        alias = node.alias or os.path.basename(path).replace('.wk', '')
        mod_prefix = _c_id(alias) + '_'

        # Parse the imported file
        from .parser import parse as parse_wk
        with open(full_path, 'r', encoding='utf-8') as f:
            src = f.read()
        prog = parse_wk(src, full_path)

        # Create sub-transpiler with shared counters
        sub = Transpiler()
        sub._module_prefix = mod_prefix
        sub._source_dir = os.path.dirname(full_path)
        sub._imported_modules = self._imported_modules
        sub._tmp_id = self._tmp_id
        sub._closure_id = self._closure_id
        sub._label_id = self._label_id

        # Pre-register global slots for top-level let/const so they persist
        # beyond the mod_init function and can be added to the module map
        for stmt in prog.stmts:
            if isinstance(stmt, (LetDecl, ConstDecl)):
                gname = f'wk_g_{mod_prefix}{_c_id(stmt.name)}'
                sub._emit_fwd(f'static WkVal {gname};')
                sub._define(stmt.name, gname)

        # Pass 1: collect declarations (emits functions, classes, etc.)
        sub._collect_decls(prog.stmts)

        # Pass 2: emit top-level statements into a module init function
        mod_init = f'_wk_mod_{_c_id(alias)}_init'
        sub._emit_fwd(f'static void {mod_init}(void);')
        sub._lines.append('')
        sub._lines.append(f'static void {mod_init}(void) {{')
        sub._indent_level = 1
        sub._push_scope()
        # Re-define module-scope names so statements can reference them
        for name, cname in sub._scopes[0].items():
            sub._define(name, cname)
        for stmt in prog.stmts:
            if isinstance(stmt, (FnDecl, ClassDecl, StructDecl, ImplBlock,
                                  EnumDecl, InterfaceDecl, MacroDecl, ActorDecl,
                                  ModuleDecl)):
                continue
            # Top-level let/const: assign to the pre-registered global slot
            if isinstance(stmt, (LetDecl, ConstDecl)):
                gname = sub._lookup(stmt.name)
                if stmt.value is not None:
                    val = sub._emit_expr(stmt.value)
                    sub._emit(f'{gname} = {val};')
                else:
                    sub._emit(f'{gname} = wk_none();')
                continue
            sub._emit_stmt(stmt)
        sub._pop_scope()
        sub._indent_level = 0
        sub._lines.append('}')

        # Sync counters back
        self._tmp_id = sub._tmp_id
        self._closure_id = sub._closure_id
        self._label_id = sub._label_id

        # Merge feature flags
        if sub._need_threads: self._need_threads = True
        if sub._need_sql:     self._need_sql = True

        # Merge sub-transpiler output into parent
        self._fwd.extend(sub._fwd)
        self._cls_defs.extend(sub._cls_defs)
        self._init_stmts.extend(sub._init_stmts)
        self._preamble_fns.extend(sub._preamble_fns)
        self._preamble_fns.extend(sub._lines)

        # Build a WkMap for the module as a global variable
        gvar = f'wk_g_mod_{_c_id(alias)}'
        self._emit_fwd(f'static WkVal {gvar};')

        # Call mod_init first (initializes globals), then populate the map
        self._emit_init(f'{mod_init}();')
        self._emit_init(f'{gvar} = wk_make_map();')
        for name, cvar in sub._scopes[0].items():
            self._emit_init(f'wk_map_set_key({gvar}, wk_make_strz("{name}"), {cvar});')

        self._define(alias, gvar)
        self._imported_modules[full_path] = gvar

    # ── Comptime Evaluation ─────────────────────────────────────────────────

    def _try_comptime_eval(self, node: ComptimeExpr) -> str:
        """Try to evaluate a comptime expression at compile time.
        Falls back to runtime emission if not evaluable."""
        val = self._extract_comptime_val(node.expr)
        if val is not None:
            return self._comptime_to_c(val)
        # Fallback: emit as runtime expression
        return self._emit_expr(node.expr)

    def _extract_comptime_val(self, node):
        """Try to get a Python value from a literal or simple arithmetic AST node."""
        import math as _math
        t = type(node)
        if t is IntLit:   return node.value
        if t is FloatLit: return node.value
        if t is BoolLit:  return node.value
        if t is StrLit:   return node.value
        if t is UnaryOp:
            operand = self._extract_comptime_val(node.operand)
            if operand is None: return None
            if node.op == '-': return -operand
            if node.op == '!': return not operand
            return None
        if t is BinOp:
            left = self._extract_comptime_val(node.left)
            right = self._extract_comptime_val(node.right)
            if left is None or right is None: return None
            return self._compute_comptime(node.op, left, right)
        # math.pi, math.e etc
        if t is MemberExpr and isinstance(node.obj, Ident) and node.obj.name == 'math':
            constants = {'pi': _math.pi, 'e': _math.e, 'tau': _math.tau, 'inf': _math.inf}
            return constants.get(node.member)
        return None

    def _compute_comptime(self, op, left, right):
        """Evaluate a binary operation at compile time."""
        try:
            if op == '+':  return left + right
            if op == '-':  return left - right
            if op == '*':  return left * right
            if op == '/':
                if right == 0: return None
                if isinstance(left, int) and isinstance(right, int):
                    return left // right
                return left / right
            if op == '%':  return left % right
            if op == '**': return left ** right
            if op == '&':  return left & right
            if op == '|':  return left | right
            if op == '^':  return left ^ right
            if op == '<<': return left << right
            if op == '>>': return left >> right
        except Exception:
            return None
        return None

    def _comptime_to_c(self, val) -> str:
        """Convert a Python value to a C literal expression."""
        if isinstance(val, bool):
            return f'wk_bool({"true" if val else "false"})'
        if isinstance(val, int):
            return f'wk_int({val})'
        if isinstance(val, float):
            if val != val:  # NaN
                return 'wk_float(NAN)'
            if val == float('inf'):
                return 'wk_float(INFINITY)'
            if val == float('-inf'):
                return 'wk_float(-INFINITY)'
            return f'wk_float({val!r})'
        if isinstance(val, str):
            escaped = val.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n').replace('\t', '\\t')
            return f'wk_make_strz("{escaped}")'
        return f'wk_int({val})'

    # ── Macro Expansion (transpiler) ──────────────────────────────────────────

    def _expand_macros_pass(self, program: Program) -> Program:
        """Collect macro declarations and expand macro calls in the AST."""
        import copy
        macros = {}
        remaining = []
        for stmt in program.stmts:
            if isinstance(stmt, MacroDecl):
                macros[stmt.name] = stmt
            else:
                remaining.append(stmt)
        if not macros:
            return program
        expanded = [self._expand_in_node(s, macros) for s in remaining]
        return Program(stmts=expanded)

    def _expand_in_node(self, node, macros):
        """Recursively walk AST and replace macro calls with expanded bodies."""
        import copy
        if node is None:
            return None
        if isinstance(node, CallExpr) and isinstance(node.callee, Ident):
            name = node.callee.name
            if name in macros:
                macro = macros[name]
                if len(node.args) != len(macro.params):
                    return node  # skip mis-matched arity
                param_map = {}
                for pname, arg in zip(macro.params, node.args):
                    param_map[pname] = self._expand_in_node(arg, macros)
                body = copy.deepcopy(macro.body)
                body = self._substitute_idents(body, param_map)
                if len(body.stmts) == 1 and isinstance(body.stmts[0], ExprStmt):
                    return body.stmts[0].expr
                return body
        if not isinstance(node, Node):
            return node
        for field_name in node.__dataclass_fields__:
            attr = getattr(node, field_name)
            if isinstance(attr, Node):
                setattr(node, field_name, self._expand_in_node(attr, macros))
            elif isinstance(attr, list):
                new_list = []
                for item in attr:
                    if isinstance(item, Node):
                        new_list.append(self._expand_in_node(item, macros))
                    elif isinstance(item, tuple):
                        new_list.append(tuple(
                            self._expand_in_node(sub, macros) if isinstance(sub, Node) else sub
                            for sub in item
                        ))
                    else:
                        new_list.append(item)
                setattr(node, field_name, new_list)
        return node

    def _substitute_idents(self, node, param_map):
        """Replace Ident nodes matching macro param names with arg expressions."""
        import copy
        if node is None:
            return None
        if isinstance(node, Ident) and node.name in param_map:
            return copy.deepcopy(param_map[node.name])
        if not isinstance(node, Node):
            return node
        for field_name in node.__dataclass_fields__:
            attr = getattr(node, field_name)
            if isinstance(attr, Node):
                setattr(node, field_name, self._substitute_idents(attr, param_map))
            elif isinstance(attr, list):
                new_list = []
                for item in attr:
                    if isinstance(item, Node):
                        new_list.append(self._substitute_idents(item, param_map))
                    elif isinstance(item, tuple):
                        new_list.append(tuple(
                            self._substitute_idents(sub, param_map) if isinstance(sub, Node) else sub
                            for sub in item
                        ))
                    else:
                        new_list.append(item)
                setattr(node, field_name, new_list)
        return node

    def _emit_go(self, node: GoStmt) -> None:
        """Spawn a detached thread running node.expr (usually a CallExpr)."""
        self._need_threads = True
        if isinstance(node.expr, CallExpr):
            fn   = self._emit_expr(node.expr.callee)
            arg_exprs = [self._emit_expr(a) for a in node.expr.args]
            argc = len(arg_exprs)
            if argc == 0:
                self._emit(f'wk_go({fn}, NULL, 0);')
            else:
                arr = self._fresh_tmp()
                self._emit(f'WkVal {arr}[] = {{{", ".join(arg_exprs)}}};')
                self._emit(f'wk_go({fn}, {arr}, {argc});')
        else:
            val = self._emit_expr(node.expr)
            self._emit(f'wk_go({val}, NULL, 0);')

    # ── ActorDecl ─────────────────────────────────────────────────────────────

    def _emit_actor(self, node: ActorDecl) -> None:
        """Emit an actor class (like a class but with mailbox support)."""
        self._need_threads = True
        cname = node.name
        # Collect fields
        field_names = [f.name for f in node.fields]

        # Pre-register in scope
        gvar_pre = f'wk_g_cls_{cname}'
        self._define(cname, gvar_pre)

        # Emit methods (same as class methods)
        for method in node.methods:
            self._emit_fn(method, toplevel=False, method_of=cname,
                          extra_caps=['self'])

        # Emit a built-in send(msg) method for all actors
        send_cname = f'_wk_{cname}_send'
        self._emit_fwd(f'static WkVal {send_cname}(WkVal *_args, int _argc, WkFunc *_fn);')
        saved = self._swap_to_preamble()
        self._preamble_fns.append('')
        self._preamble_fns.append(f'static WkVal {send_cname}(WkVal *_args, int _argc, WkFunc *_fn) {{')
        self._preamble_fns.append(f'    WkVal _self = _fn->captures[0].val;')
        self._preamble_fns.append(f'    if (_self.tag != WK_OBJ || !_self.as.obj->mailbox)')
        self._preamble_fns.append(f'        wk_panic("send: not an actor instance");')
        self._preamble_fns.append(f'    WkVal _msg = (_argc > 0) ? _args[0] : wk_none();')
        self._preamble_fns.append(f'    wk_mailbox_send(_self.as.obj->mailbox, _msg);')
        self._preamble_fns.append(f'    return wk_none();')
        self._preamble_fns.append(f'}}')
        self._swap_restore(saved)

        # Method table (user methods + built-in send)
        methods_arr = f'_wk_{cname}_methods'
        method_entries = []
        for m in node.methods:
            mname = f'_wk_{cname}_{_c_id(m.name)}'
            nmp = len([p for p in m.params
                       if (p if isinstance(p,str) else p.name) != 'self'])
            method_entries.append(
                f'    {{"{m.name}", &(WkFunc){{1, "{m.name}", {mname}, NULL, 0, NULL, {nmp}}}}}'
            )
            self._emit_fwd(f'static WkVal _wk_{cname}_{_c_id(m.name)}'
                           f'(WkVal *_args, int _argc, WkFunc *_fn);')
        # Always add the built-in send method
        method_entries.append(f'    {{"send", &(WkFunc){{1, "send", {send_cname}, NULL, 0, NULL, 1}}}}')
        self._emit_cls(f'static WkMethod {methods_arr}[] = {{')
        for e in method_entries:
            self._emit_cls(e + ',')
        self._emit_cls('};')

        # Field names
        if field_names:
            fnames_arr = f'_wk_{cname}_fnames'
            fnames_c = ', '.join(f'"{f}"' for f in field_names)
            self._emit_cls(f'static const char *{fnames_arr}[] = {{{fnames_c}}};')
        else:
            fnames_arr = 'NULL'

        nmethods = len(node.methods) + 1  # +1 for built-in send()
        nfields  = len(field_names)
        self._emit_cls(f'static WkClass _wk_cls_{cname} = {{')
        self._emit_cls(f'    "{cname}", NULL,')
        self._emit_cls(f'    {methods_arr}, {nmethods},')
        self._emit_cls(f'    {fnames_arr}, {nfields}')
        self._emit_cls('};')

        # Register in init
        self._emit_fwd(f'static WkVal wk_g_cls_{cname};')
        self._emit_init(f'wk_g_cls_{cname} = wk_make_class(&_wk_cls_{cname});')

    # ── SpawnExpr: spawn Actor(args) ─────────────────────────────────────────

    def _emit_spawn(self, node: SpawnExpr) -> str:
        """Create actor instance, init mailbox, start run() thread."""
        self._need_threads = True

        # Parser may wrap `spawn Counter(args)` as
        # SpawnExpr(actor_class=CallExpr(callee=Ident("Counter"), args=[...]), args=[])
        # Normalise to (actor_class_node, all_args).
        actor_node = node.actor_class
        all_args   = list(node.args)
        if isinstance(actor_node, CallExpr):
            all_args = list(actor_node.args) + all_args
            actor_node = actor_node.callee

        cname = actor_node.name if isinstance(actor_node, Ident) else None

        tmp = self._fresh_tmp()
        if cname and cname in self._actor_names:
            # Known actor class — emit direct struct reference
            self._emit(f'WkVal {tmp} = wk_make_obj(&_wk_cls_{cname});')
        else:
            # Generic — look up class at runtime
            cls_val = self._emit_expr(actor_node)
            self._emit(f'WkVal {tmp} = (({cls_val}).tag == WK_CLASS) '
                       f'? wk_make_obj(({cls_val}).as.cls) : wk_none();')

        # Initialize mailbox
        self._emit(f'if ({tmp}.tag == WK_OBJ) {tmp}.as.obj->mailbox = wk_mailbox_new();')

        # Call new() if it exists (use wk_obj_find_method which returns none if absent)
        arg_exprs = [self._emit_expr(a) for a in all_args]
        new_tmp = self._fresh_tmp()
        self._emit(f'{{ WkVal {new_tmp} = ({tmp}.tag==WK_OBJ) ? wk_obj_find_method({tmp}, "new") : wk_none();')
        if arg_exprs:
            arr_tmp = self._fresh_tmp()
            self._emit(f'  WkVal {arr_tmp}[] = {{{", ".join(arg_exprs)}}};')
            self._emit(f'  if ({new_tmp}.tag == WK_FUNC) wk_call({new_tmp}, {arr_tmp}, {len(arg_exprs)});')
        else:
            self._emit(f'  if ({new_tmp}.tag == WK_FUNC) wk_call0({new_tmp});')
        self._emit('}')

        # Start run() as a background thread
        run_tmp = self._fresh_tmp()
        self._emit(f'{{ WkVal {run_tmp} = ({tmp}.tag==WK_OBJ) ? wk_obj_find_method({tmp}, "run") : wk_none();')
        self._emit(f'  if ({run_tmp}.tag == WK_FUNC) wk_go({run_tmp}, NULL, 0);')
        self._emit('}')

        return tmp

    # ── ReceiveStmt ───────────────────────────────────────────────────────────

    def _emit_receive(self, node: ReceiveStmt) -> None:
        """Receive a message from self's mailbox and pattern-match it."""
        # Get timeout in ms
        timeout_ms = '100'  # default 100 ms
        if node.timeout:
            tval = self._emit_expr(node.timeout)
            timeout_tmp = self._fresh_tmp()
            self._emit(f'int {timeout_tmp} = (int)(({tval}.tag==WK_FLOAT?{tval}.as.f:{tval}.as.i)*1000.0);')
            timeout_ms = timeout_tmp

        mb_tmp = self._fresh_tmp()
        self._emit(f'WkMailbox *{mb_tmp} = (_fn && _fn->ncaptures > 0 && '
                   f'_fn->captures[0].val.tag == WK_OBJ) '
                   f'? _fn->captures[0].val.as.obj->mailbox : NULL;')
        self._emit(f'if (!{mb_tmp}) wk_panic("receive used outside actor method");')

        msg_tmp = self._fresh_tmp()
        got_tmp = self._fresh_tmp()
        self._emit(f'WkVal {msg_tmp} = wk_none();')
        self._emit(f'int {got_tmp} = wk_mailbox_recv({mb_tmp}, {timeout_ms}, &{msg_tmp});')
        self._emit(f'if ({got_tmp}) {{')
        self._ind()

        # Emit pattern matching on msg_tmp (reuse match logic)
        matched_tmp = self._fresh_tmp()
        self._emit(f'int {matched_tmp} = 0;')
        for arm in node.arms:
            binds = []
            check = self._pattern_check(arm.pattern, msg_tmp, binds)
            self._emit(f'if (!{matched_tmp} && ({check})) {{')
            self._ind(); self._push_scope()
            for bind_line in binds:
                self._emit(bind_line)
            if arm.guard:
                ge = self._emit_expr(arm.guard)
                self._emit(f'if (wk_truthy({ge})) {{')
                self._ind()
                self._emit(f'{matched_tmp} = 1;')
                if isinstance(arm.body, Block):
                    for s in arm.body.stmts: self._emit_stmt(s)
                elif arm.body is not None:
                    r = self._emit_expr(arm.body); self._emit(f'(void)({r});')
                self._ded(); self._emit('}')
            else:
                self._emit(f'{matched_tmp} = 1;')
                if isinstance(arm.body, Block):
                    for s in arm.body.stmts: self._emit_stmt(s)
                elif arm.body is not None:
                    r = self._emit_expr(arm.body); self._emit(f'(void)({r});')
            self._pop_scope(); self._ded()
            self._emit('}')

        self._ded()
        self._emit('}')

    def _emit_array(self, node: ArrayLit) -> str:
        tmp = self._fresh_tmp()
        self._emit(f'WkVal {tmp} = wk_make_list();')
        for elem in node.elements:
            e = self._emit_expr(elem)
            self._emit(f'wk_list_push_raw({tmp}.as.list, {e});')
        return tmp

    def _emit_map(self, node: MapLit) -> str:
        tmp = self._fresh_tmp()
        self._emit(f'WkVal {tmp} = wk_make_map();')
        for k, v in node.pairs:
            ke = self._emit_expr(k)
            ve = self._emit_expr(v)
            self._emit(f'wk_map_set_key({tmp}, {ke}, {ve});')
        return tmp

    def _emit_tuple(self, node: TupleLit) -> str:
        elems = [self._emit_expr(e) for e in node.elements]
        tmp = self._fresh_tmp()
        arr = self._fresh_tmp()
        if elems:
            self._emit(f'WkVal {arr}[] = {{{", ".join(elems)}}};')
            self._emit(f'WkVal {tmp} = wk_make_tuple({arr}, {len(elems)});')
        else:
            self._emit(f'WkVal {tmp} = wk_make_tuple(NULL, 0);')
        return tmp

    def _emit_struct_lit(self, node: StructLit) -> str:
        cls_var = self._lookup(node.name) or f'wk_g_cls_{node.name}'
        tmp = self._fresh_tmp()
        self._emit(f'WkVal {tmp} = ({cls_var}.tag==WK_CLASS) ? wk_make_obj({cls_var}.as.cls) : wk_make_obj(&(WkClass){{"{node.name}", NULL, NULL, 0, NULL, 0}});')
        for fname, fval in node.fields:
            ve = self._emit_expr(fval)
            self._emit(f'wk_member_set({tmp}, "{fname}", {ve});')
        return tmp

    def _emit_ident(self, node: Ident) -> str:
        name = node.name
        # Check scope
        cvar = self._lookup(name)
        if cvar:
            return cvar
        # Builtin
        if name in _BUILTINS:
            return _BUILTINS[name]
        # Unknown — might be a class or enum variant registered in init
        # Return as a global slot name
        return f'wk_g_{name}'

    def _emit_member(self, node: MemberExpr) -> str:
        obj = self._emit_expr(node.obj)
        tmp = self._fresh_tmp()
        self._emit(f'WkVal {tmp} = wk_member_get({obj}, "{node.member}");')
        return tmp

    def _emit_index(self, node: IndexExpr) -> str:
        obj = self._emit_expr(node.obj)
        idx = self._emit_expr(node.index)
        tmp = self._fresh_tmp()
        self._emit(f'WkVal {tmp} = wk_index_get({obj}, {idx});')
        return tmp

    def _emit_slice(self, node: SliceExpr) -> str:
        obj = self._emit_expr(node.obj)
        start = self._emit_expr(node.start) if node.start else 'wk_none()'
        end = self._emit_expr(node.end) if node.end else 'wk_none()'
        step = self._emit_expr(node.step) if node.step else 'wk_none()'
        tmp = self._fresh_tmp()
        self._emit(f'WkVal {tmp} = ({obj}.tag==WK_STR)'
                   f' ? wk_str_slice({obj},{start},{end})'
                   f' : wk_list_slice({obj},{start},{end},{step});')
        return tmp

    def _emit_binop(self, node: BinOp) -> str:
        op = node.op
        # Short-circuit &&, ||
        if op == '&&':
            left = self._emit_expr(node.left)
            tmp = self._fresh_tmp()
            self._emit(f'WkVal {tmp} = wk_none();')
            self._emit(f'if (wk_truthy({left})) {{')
            self._ind()
            right = self._emit_expr(node.right)
            self._emit(f'{tmp} = {right};')
            self._ded()
            self._emit(f'}} else {{ {tmp} = {left}; }}')
            return tmp
        elif op == '||':
            left = self._emit_expr(node.left)
            tmp = self._fresh_tmp()
            self._emit(f'WkVal {tmp} = wk_none();')
            self._emit(f'if (wk_truthy({left})) {{ {tmp} = {left}; }} else {{')
            self._ind()
            right = self._emit_expr(node.right)
            self._emit(f'{tmp} = {right};')
            self._ded()
            self._emit('}')
            return tmp
        elif op == 'not in':
            left = self._emit_expr(node.left)
            right = self._emit_expr(node.right)
            return f'wk_bool(!wk_truthy(wk_in({left}, {right})))'

        # Channel send: ch <- value
        if op == '<-':
            left = self._emit_expr(node.left)
            right = self._emit_expr(node.right)
            self._emit(f'wk_chan_send({left}, {right});')
            return 'wk_none()'

        fn = _OP_MAP.get(op)
        left = self._emit_expr(node.left)
        right = self._emit_expr(node.right)
        if fn:
            return f'{fn}({left}, {right})'
        # Fallback: unsupported op
        return f'wk_none() /* unsupported op {op} */'

    def _emit_unary(self, node: UnaryOp) -> str:
        operand = self._emit_expr(node.operand)
        if node.op == '-':   return f'wk_neg({operand})'
        if node.op == '!':   return f'wk_not({operand})'
        if node.op == '~':   return f'wk_bitnot({operand})'
        if node.op == 'not': return f'wk_not({operand})'
        return operand

    def _emit_call(self, node: CallExpr) -> str:
        # Channel creation: __chan__(capacity)
        if isinstance(node.callee, Ident) and node.callee.name == '__chan__':
            self._need_threads = True
            cap = self._emit_expr(node.args[0]) if node.args else 'wk_int(0)'
            tmp = self._fresh_tmp()
            self._emit(f'WkVal {tmp} = wk_make_chan((int)({cap}).as.i);')
            return tmp

        # Channel receive: __recv__(ch)
        if isinstance(node.callee, Ident) and node.callee.name == '__recv__':
            self._need_threads = True
            ch = self._emit_expr(node.args[0]) if node.args else 'wk_none()'
            tmp = self._fresh_tmp()
            self._emit(f'WkVal {tmp} = wk_chan_recv({ch});')
            return tmp

        # Check for method calls: obj.method(args)
        if isinstance(node.callee, MemberExpr):
            obj = self._emit_expr(node.callee.obj)
            method_name = node.callee.member
            # Evaluate args (positional + kwargs as positional)
            arg_exprs = [self._emit_expr(a) for a in node.args]
            arg_exprs += [self._emit_expr(v) for _, v in (node.kwargs or [])]
            tmp = self._fresh_tmp()
            # Get bound method, then call it
            method_tmp = self._fresh_tmp()
            if arg_exprs:
                arr = self._fresh_tmp()
                self._emit(f'WkVal {arr}[] = {{{", ".join(arg_exprs)}}};')
                self._emit(f'WkVal {method_tmp} = wk_member_get({obj}, "{method_name}");')
                self._emit(f'WkVal {tmp} = wk_call({method_tmp}, {arr}, {len(arg_exprs)});')
            else:
                self._emit(f'WkVal {method_tmp} = wk_member_get({obj}, "{method_name}");')
                self._emit(f'WkVal {tmp} = wk_call({method_tmp}, NULL, 0);')
            return tmp

        # Class constructor: ClassName.new(args) or ClassName(args)
        callee_expr = self._emit_expr(node.callee)
        # Kwargs treated as positional (after positional args)
        arg_exprs = [self._emit_expr(a) for a in node.args]
        arg_exprs += [self._emit_expr(v) for _, v in (node.kwargs or [])]
        tmp = self._fresh_tmp()
        self._emit(f'WkVal {tmp};')
        if arg_exprs:
            arr = self._fresh_tmp()
            self._emit(f'WkVal {arr}[] = {{{", ".join(arg_exprs)}}};')
            # If callee is a class, call the 'new' method
            self._emit(f'if ({callee_expr}.tag == WK_CLASS) {{')
            self._ind()
            new_m = self._fresh_tmp()
            self._emit(f'WkVal {new_m} = wk_obj_find_method({callee_expr}, "new");')
            self._emit(f'{tmp} = ({new_m}.tag==WK_FUNC) ? wk_call({new_m}, {arr}, {len(arg_exprs)}) : wk_make_obj({callee_expr}.as.cls);')
            self._ded()
            self._emit(f'}} else {{')
            self._ind()
            self._emit(f'{tmp} = wk_call({callee_expr}, {arr}, {len(arg_exprs)});')
            self._ded()
            self._emit(f'}}')
        else:
            self._emit(f'if ({callee_expr}.tag == WK_CLASS) {{')
            self._ind()
            new_m = self._fresh_tmp()
            self._emit(f'WkVal {new_m} = wk_obj_find_method({callee_expr}, "new");')
            self._emit(f'{tmp} = ({new_m}.tag==WK_FUNC) ? wk_call({new_m}, NULL, 0) : wk_make_obj({callee_expr}.as.cls);')
            self._ded()
            self._emit(f'}} else {{')
            self._ind()
            self._emit(f'{tmp} = wk_call({callee_expr}, NULL, 0);')
            self._ded()
            self._emit(f'}}')
        return tmp

    def _emit_ternary(self, node: TernaryExpr) -> str:
        cond = self._emit_expr(node.cond)
        tmp = self._fresh_tmp()
        self._emit(f'WkVal {tmp};')
        self._emit(f'if (wk_truthy({cond})) {{')
        self._ind()
        then_v = self._emit_expr(node.then)
        self._emit(f'{tmp} = {then_v};')
        self._ded()
        self._emit('} else {')
        self._ind()
        else_v = self._emit_expr(node.else_)
        self._emit(f'{tmp} = {else_v};')
        self._ded()
        self._emit('}')
        return tmp

    def _emit_cast(self, node: CastExpr) -> str:
        expr = self._emit_expr(node.expr)
        t = node.to_type.lower() if isinstance(node.to_type, str) else str(node.to_type)
        if t in ('int', 'i8','i16','i32','i64','u8','u16','u32','u64'):
            return f'wk_cast_int({expr})'
        elif t in ('float', 'f32', 'f64', 'double'):
            return f'wk_cast_float({expr})'
        elif t == 'bool':
            return f'wk_cast_bool({expr})'
        elif t in ('str', 'string'):
            return f'wk_cast_str({expr})'
        elif t in ('byte', 'rune', 'char'):
            return f'wk_cast_byte({expr})'
        return expr

    def _emit_pipe(self, node: PipeExpr) -> str:
        left = self._emit_expr(node.left)
        # If right is a call, prepend left as first arg
        if isinstance(node.right, CallExpr):
            rargs = [self._emit_expr(a) for a in node.right.args]
            all_args = [left] + rargs
            fn_expr = self._emit_expr(node.right.callee)
            tmp = self._fresh_tmp()
            arr = self._fresh_tmp()
            self._emit(f'WkVal {arr}[] = {{{", ".join(all_args)}}};')
            self._emit(f'WkVal {tmp} = wk_call({fn_expr}, {arr}, {len(all_args)});')
            return tmp
        else:
            fn = self._emit_expr(node.right)
            tmp = self._fresh_tmp()
            self._emit(f'WkVal {tmp} = wk_call1({fn}, {left});')
            return tmp

    def _emit_propagate(self, node: PropagateExpr) -> str:
        inner = self._emit_expr(node.expr)
        tmp = self._fresh_tmp()
        self._emit(f'WkVal {tmp} = {inner};')
        self._emit(f'if ({tmp}.tag == WK_ERR || {tmp}.tag == WK_NONE) {{')
        self._ind()
        self._emit(f'wk_defer_flush(&_df);')
        self._emit(f'return {tmp};')
        self._ded()
        self._emit(f'}}')
        result = self._fresh_tmp()
        self._emit(f'WkVal {result} = ({tmp}.tag==WK_OK||{tmp}.tag==WK_SOME) ? *{tmp}.as.inner : {tmp};')
        return result

    def _emit_range(self, node: RangeExpr) -> str:
        start = self._emit_expr(node.start)
        end = self._emit_expr(node.end)
        inc = '1' if node.inclusive else '0'
        s_cast = f'({start}.tag==WK_INT?{start}.as.i:(int64_t){start}.as.f)'
        e_cast = f'({end}.tag==WK_INT?{end}.as.i:(int64_t){end}.as.f)'
        return f'wk_range({s_cast}, {e_cast}, {inc})'

    def _emit_opt_chain(self, node: OptChainExpr) -> str:
        obj = self._emit_expr(node.obj)
        tmp = self._fresh_tmp()
        self._emit(f'WkVal {tmp} = ({obj}.tag==WK_NONE) ? wk_none() : wk_member_get({obj}, "{node.member}");')
        return tmp

    def _emit_null_coalesce(self, node: NullCoalesceExpr) -> str:
        left = self._emit_expr(node.left)
        tmp = self._fresh_tmp()
        self._emit(f'WkVal {tmp};')
        self._emit(f'if (wk_truthy({left})) {{ {tmp} = {left}; }} else {{')
        self._ind()
        right = self._emit_expr(node.right)
        self._emit(f'{tmp} = {right};')
        self._ded()
        self._emit('}')
        return tmp

    def _emit_if_expr(self, node: IfStmt) -> str:
        """if/else used as an expression."""
        tmp = self._fresh_tmp()
        self._emit(f'WkVal {tmp} = wk_none();')
        cond = self._emit_expr(node.cond)
        self._emit(f'if (wk_truthy({cond})) {{')
        self._ind(); self._push_scope()
        stmts = node.then.stmts
        for i, s in enumerate(stmts):
            if i == len(stmts)-1 and isinstance(s, ExprStmt):
                r = self._emit_expr(s.expr); self._emit(f'{tmp} = {r};')
            else: self._emit_stmt(s)
        self._pop_scope(); self._ded()
        for elif_cond, elif_body in node.elseifs:
            ec = self._emit_expr(elif_cond)
            self._emit(f'}} else if (wk_truthy({ec})) {{')
            self._ind(); self._push_scope()
            stmts = elif_body.stmts
            for i, s in enumerate(stmts):
                if i == len(stmts)-1 and isinstance(s, ExprStmt):
                    r = self._emit_expr(s.expr); self._emit(f'{tmp} = {r};')
                else: self._emit_stmt(s)
            self._pop_scope(); self._ded()
        if node.else_:
            self._emit('} else {')
            self._ind(); self._push_scope()
            stmts = node.else_.stmts
            for i, s in enumerate(stmts):
                if i == len(stmts)-1 and isinstance(s, ExprStmt):
                    r = self._emit_expr(s.expr); self._emit(f'{tmp} = {r};')
                else: self._emit_stmt(s)
            self._pop_scope(); self._ded()
        self._emit('}')
        return tmp
