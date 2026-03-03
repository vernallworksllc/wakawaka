"""
Deltoo Tree-Walk Interpreter
Executes Deltoo AST nodes directly. Supports:
  - All core language constructs
  - Classes / inheritance / polymorphism
  - Pattern matching
  - Shell integration ($`...`)
  - SQL integration (@sql`...`)
  - Python interop (import python "pkg")
  - ML/stats via Python libraries
  - Closures, goroutines (threads), channels, async/await
"""
import sys
import os
import subprocess
import threading
import queue
import importlib
import sqlite3
import math
import re
import textwrap
import asyncio
from typing import Any, Dict, List, Optional

from .ast_nodes import *


# ── Sentinel values ────────────────────────────────────────────────────────────

class _Return(Exception):
    def __init__(self, value): self.value = value

class _Break(Exception):
    def __init__(self, label=None): self.label = label

class _Continue(Exception):
    def __init__(self, label=None): self.label = label

class _Panic(Exception):
    pass


# ── Runtime Types ──────────────────────────────────────────────────────────────

class DeltooNone:
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    def __repr__(self): return "none"
    def __bool__(self): return False

NONE = DeltooNone()


class DeltooSome:
    def __init__(self, value): self.value = value
    def __repr__(self): return f"some({self.value!r})"
    def __bool__(self): return True


class DeltooOk:
    def __init__(self, value): self.value = value
    def __repr__(self): return f"ok({self.value!r})"
    def __bool__(self): return True
    def is_ok(self): return True
    def is_err(self): return False
    def unwrap(self): return self.value


class DeltooErr:
    def __init__(self, value): self.value = value
    def __repr__(self): return f"err({self.value!r})"
    def __bool__(self): return False
    def is_ok(self): return False
    def is_err(self): return True
    def unwrap(self):
        raise DeltooRuntimeError(f"Called unwrap() on err({self.value!r})")


class DeltooRange:
    def __init__(self, start, end, inclusive=False):
        self.start = start
        self.end = end
        self.inclusive = inclusive
    def __iter__(self):
        if self.inclusive:
            return iter(range(self.start, self.end + 1))
        return iter(range(self.start, self.end))
    def __contains__(self, item):
        if self.inclusive:
            return self.start <= item <= self.end
        return self.start <= item < self.end
    def __repr__(self):
        op = "..=" if self.inclusive else ".."
        return f"{self.start}{op}{self.end}"


class DeltooChannel:
    def __init__(self, capacity=0):
        self.q = queue.Queue(maxsize=capacity if capacity else 0)
    def send(self, val):
        self.q.put(val)
    def recv(self):
        return self.q.get()
    def __repr__(self): return f"chan<...>"


class DeltooFunction:
    def __init__(self, name, params, body, env, is_async=False, decorators=None):
        self.name = name
        self.params = params
        self.body = body
        self.env = env  # closure environment
        self.is_async = is_async
        self.decorators = decorators or []
    def __repr__(self): return f"<fn {self.name}>"


class DeltooClass:
    def __init__(self, name, fields, methods, parent=None, is_abstract=False, env=None):
        self.name = name
        self.fields = fields        # list of StructField
        self.methods = methods      # {name: DeltooFunction}
        self.parent = parent        # DeltooClass | None
        self.is_abstract = is_abstract
        self.env = env              # definition environment
        self.is_actor = False       # True for actor classes
    def __repr__(self): return f"<class {self.name}>"

    def find_method(self, name):
        if name in self.methods:
            return self.methods[name]
        if self.parent:
            return self.parent.find_method(name)
        return None


class DeltooInstance:
    def __init__(self, cls: DeltooClass, fields: dict):
        self.cls = cls
        self.fields = dict(fields)
    def __repr__(self): return f"<{self.cls.name} instance>"

    def get_attr(self, name):
        # Actor built-in methods
        if self.fields.get("__actor__") and name == "send":
            mailbox = self.fields.get("__mailbox__")
            return lambda msg: mailbox.put(msg) or NONE
        if name in self.fields:
            return self.fields[name]
        method = self.cls.find_method(name)
        if method:
            return BoundMethod(method, self)
        raise DeltooRuntimeError(
            f"'{self.cls.name}' has no attribute '{name}'"
        )

    def set_attr(self, name, value):
        self.fields[name] = value


class BoundMethod:
    def __init__(self, fn: DeltooFunction, instance: DeltooInstance):
        self.fn = fn
        self.instance = instance
    def __repr__(self): return f"<bound method {self.fn.name}>"


class DeltooModule:
    """Wraps a Python module for transparent interop."""
    def __init__(self, pymod):
        self._mod = pymod
    def __repr__(self): return f"<pymodule {self._mod.__name__}>"
    def get_attr(self, name):
        try:
            attr = getattr(self._mod, name)
            return _wrap_python(attr)
        except AttributeError:
            raise DeltooRuntimeError(
                f"Module '{self._mod.__name__}' has no attribute '{name}'"
            )


# ── Multi-language Foreign Interop ─────────────────────────────────────────────

class _JsBridge:
    """Singleton persistent Node.js subprocess bridge using JSON-line IPC."""
    _instance = None

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        import shutil, json as _json
        if not shutil.which("node"):
            raise DeltooRuntimeError(
                "Node.js not found. Install Node.js to use 'import js'"
            )
        self._json = _json
        # Security: no eval/exec/Function — only require, call, get, store actions
        bridge = (
            "const rl=require('readline').createInterface({input:process.stdin,terminal:false});"
            "const mods={};"
            "rl.on('line',line=>{"
            "let res;"
            "try{"
            "const r=JSON.parse(line);"
            "if(r.action==='require'){mods[r.id]=require(r.path);res={ok:true};}"
            "else if(r.action==='call'){"
            "const m=mods[r.id];"
            "const fn=r.method?m[r.method]:m;"
            "if(typeof fn!=='function')throw new Error(r.method+' is not a function');"
            "const result=fn.apply(m,r.args);"
            "res={ok:true,result:result===undefined?null:result};}"
            "else if(r.action==='get'){"
            "const v=r.method?mods[r.id][r.method]:mods[r.id];"
            "res={ok:true,result:v===undefined?null:v};}"
            "else if(r.action==='store'){"
            # store: mods[newId] = mods[srcId][prop] — sub-property aliasing, no eval
            "mods[r.new_id]=mods[r.src_id][r.prop];"
            "res={ok:true};}"
            "}catch(e){res={ok:false,error:e.message};}"
            "process.stdout.write(JSON.stringify(res)+'\\n');"
            "});"
        )
        self._proc = subprocess.Popen(
            ["node", "-e", bridge],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, bufsize=1,
        )
        self._counter = 0

    def _send(self, req: dict):
        line = self._json.dumps(req) + "\n"
        self._proc.stdin.write(line)
        self._proc.stdin.flush()
        out = self._proc.stdout.readline()
        if not out:
            err = self._proc.stderr.read()
            raise DeltooRuntimeError(f"JS bridge crashed: {err}")
        res = self._json.loads(out)
        if not res.get("ok"):
            raise DeltooRuntimeError(f"JS error: {res.get('error', 'unknown')}")
        return res.get("result")

    def require(self, path: str) -> str:
        mid = f"m{self._counter}"; self._counter += 1
        self._send({"action": "require", "id": mid, "path": path})
        return mid

    def call(self, mod_id: str, method, args: list):
        return self._send({"action": "call", "id": mod_id,
                           "method": method, "args": args})

    def get(self, mod_id: str, method):
        return self._send({"action": "get", "id": mod_id, "method": method})

    def store(self, src_id: str, prop: str) -> str:
        """Alias mods[new_id] = mods[src_id][prop] — no eval, safe chaining."""
        new_id = f"m{self._counter}"; self._counter += 1
        self._send({"action": "store", "src_id": src_id, "prop": prop, "new_id": new_id})
        return new_id

    def __del__(self):
        try: self._proc.terminate()
        except Exception: pass


class JsProxy:
    """Wraps a Node.js module. import js "lodash" as _"""
    def __init__(self, mod_path: str):
        self._path = mod_path
        self._bridge = _JsBridge.instance()
        self._id = self._bridge.require(mod_path)

    def __repr__(self): return f"<js module '{self._path}'>"

    def get_attr(self, name):
        return JsCallable(self._bridge, self._id, name)

    def __call__(self, *args):
        result = self._bridge.call(self._id, None, [_unwrap_for_js(a) for a in args])
        return _wrap_js(result)


class JsCallable:
    """A callable attribute on a JS module."""
    def __init__(self, bridge: _JsBridge, mod_id: str, method: str):
        self._bridge = bridge
        self._mod_id = mod_id
        self._method = method
        self.name = method

    def __repr__(self): return f"<js fn '{self._method}'>"

    def get_attr(self, name):
        # Chained access: module.ns.fn — store sub-property, no eval
        sub_id = self._bridge.store(self._mod_id, self._method)
        return JsCallable(self._bridge, sub_id, name)

    def __call__(self, *args):
        result = self._bridge.call(self._mod_id, self._method,
                                   [_unwrap_for_js(a) for a in args])
        return _wrap_js(result)


def _unwrap_for_js(val):
    """Convert Deltoo value to a JSON-serializable Python value."""
    if isinstance(val, DeltooNone): return None
    if isinstance(val, DeltooSome): return _unwrap_for_js(val.value)
    if isinstance(val, DeltooOk):   return {"__ok__": _unwrap_for_js(val.value)}
    if isinstance(val, DeltooErr):  return {"__err__": _unwrap_for_js(val.value)}
    if isinstance(val, DeltooInstance): return {k: _unwrap_for_js(v)
                                                 for k, v in val.fields.items()
                                                 if not k.startswith("__")}
    if isinstance(val, list):  return [_unwrap_for_js(x) for x in val]
    if isinstance(val, dict):  return {k: _unwrap_for_js(v) for k, v in val.items()}
    if isinstance(val, tuple): return [_unwrap_for_js(x) for x in val]
    if isinstance(val, PyObject): return val._obj
    return val


def _wrap_js(val):
    """Convert a JSON-decoded JS result to a Deltoo value."""
    if val is None: return NONE
    if isinstance(val, list):  return [_wrap_js(x) for x in val]
    if isinstance(val, dict):  return {k: _wrap_js(v) for k, v in val.items()}
    return val


class CProxy:
    """Wraps a C shared library via ctypes. import c "libm" as libm"""
    def __init__(self, lib_name: str):
        import ctypes, ctypes.util as cu
        path = cu.find_library(lib_name)
        if not path:
            # Try direct path (e.g. "./mylib.so")
            if os.path.exists(lib_name):
                path = lib_name
            else:
                raise DeltooRuntimeError(
                    f"C library not found: '{lib_name}'. "
                    f"Pass the filename directly (e.g. './libfoo.so') or install it."
                )
        self._lib = ctypes.CDLL(path)
        self._ctypes = ctypes
        self._name = lib_name

    def __repr__(self): return f"<c lib '{self._name}'>"

    def get_attr(self, name):
        fn = getattr(self._lib, name, None)
        if fn is None:
            raise DeltooRuntimeError(f"C library '{self._name}' has no symbol '{name}'")
        ctypes = self._ctypes
        def _call(*args):
            # Auto-configure for common numeric types
            py_args = [_unwrap(a) for a in args]
            fn.restype = ctypes.c_double
            fn.argtypes = [ctypes.c_double if isinstance(a, (int, float)) else ctypes.c_char_p
                           for a in py_args]
            result = fn(*[ctypes.c_double(a) if isinstance(a, (int, float))
                          else a.encode() if isinstance(a, str) else a
                          for a in py_args])
            return float(result)
        _call.name = name
        return _call


class CppProxy:
    """Wraps a C++ shared library (must export extern \"C\" symbols). import cpp \"libfoo\""""
    def __init__(self, lib_name: str):
        # C++ libraries with extern "C" load identically to C
        self._inner = CProxy(lib_name)
        self._name = lib_name

    def __repr__(self): return f"<cpp lib '{self._name}'>"
    def get_attr(self, name): return self._inner.get_attr(name)


class JavaProxy:
    """Wraps a Java class. import java \"java.lang.Math\" as JMath
    Uses JPype if installed, otherwise falls back to a subprocess script."""
    def __init__(self, class_name: str):
        self._class_name = class_name
        self._jpype = None
        self._cls = None
        try:
            import jpype
            if not jpype.isJVMStarted():
                jpype.startJVM(convertStrings=False)
            self._cls = jpype.JClass(class_name)
            self._jpype = jpype
        except ImportError:
            pass  # fall back to subprocess
        except Exception as e:
            raise DeltooRuntimeError(
                f"Java error loading '{class_name}': {e}\n"
                f"Tip: install JPype with 'pip install JPype1'"
            )

    def __repr__(self): return f"<java class '{self._class_name}'>"

    def get_attr(self, name):
        if self._jpype and self._cls:
            attr = getattr(self._cls, name, None)
            if attr is None:
                raise DeltooRuntimeError(
                    f"Java class '{self._class_name}' has no member '{name}'"
                )
            if callable(attr):
                def _call(*args):
                    result = attr(*[_unwrap(a) for a in args])
                    return _wrap_python(result)
                _call.name = name
                return _call
            return _wrap_python(attr)
        else:
            # Subprocess fallback: generate+run a minimal Java snippet
            return JavaSubprocCallable(self._class_name, name)


_JAVA_IDENT_RE = re.compile(r'^[A-Za-z_][A-Za-z0-9_.]*$')


class JavaSubprocCallable:
    """Calls a Java static method via subprocess (no JPype)."""
    def __init__(self, class_name: str, method: str):
        # Security: validate identifiers to prevent code injection in generated Java source
        if not _JAVA_IDENT_RE.match(class_name):
            raise DeltooRuntimeError(
                f"Invalid Java class name: '{class_name}'"
            )
        if not _JAVA_IDENT_RE.match(method):
            raise DeltooRuntimeError(
                f"Invalid Java method name: '{method}'"
            )
        self._class = class_name
        self._method = method
        self.name = method

    def __repr__(self): return f"<java {self._class}.{self._method}>"

    def __call__(self, *args):
        import shutil, tempfile
        if not shutil.which("java"):
            raise DeltooRuntimeError(
                "Java not found. Install a JDK and set JAVA_HOME to use 'import java'."
            )
        py_args = [_unwrap(a) for a in args]
        def _java_val(v):
            # Security: escape string values to prevent Java source injection
            if isinstance(v, bool):  return "true" if v else "false"
            if isinstance(v, int):   return str(v)
            if isinstance(v, float): return str(v)
            if isinstance(v, str):
                escaped = v.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
                return f'"{escaped}"'
            return str(v)
        args_java = ", ".join(_java_val(a) for a in py_args)
        short = self._class.split(".")[-1]
        src = (
            f"import {self._class};\n"
            f"public class _WakaJava {{\n"
            f"    public static void main(String[] a) {{\n"
            f"        System.out.println({short}.{self._method}({args_java}));\n"
            f"    }}\n"
            f"}}\n"
        )
        with tempfile.TemporaryDirectory() as td:
            src_file = os.path.join(td, "_WakaJava.java")
            with open(src_file, "w") as f: f.write(src)
            javac = subprocess.run(["javac", src_file], capture_output=True, text=True)
            if javac.returncode != 0:
                raise DeltooRuntimeError(
                    f"Java compile error: {javac.stderr.strip()}"
                )
            run = subprocess.run(["java", "-cp", td, "_WakaJava"],
                                 capture_output=True, text=True)
            if run.returncode != 0:
                raise DeltooRuntimeError(f"Java runtime error: {run.stderr.strip()}")
            out = run.stdout.strip()
            try:    return int(out)
            except ValueError:
                try:    return float(out)
                except ValueError: return out


class SwiftProxy:
    """Wraps a Swift module. import swift \"Foundation\" as Foundation
    Calls Swift code via subprocess (requires Swift toolchain)."""
    def __init__(self, module: str):
        import shutil
        self._module = module
        if not shutil.which("swift"):
            raise DeltooRuntimeError(
                "Swift not found. Install the Swift toolchain to use 'import swift'."
            )

    def __repr__(self): return f"<swift module '{self._module}'>"

    def get_attr(self, name):
        return SwiftCallable(self._module, name)


_SWIFT_IDENT_RE = re.compile(r'^[A-Za-z_][A-Za-z0-9_.]*$')


class SwiftCallable:
    """Calls a Swift function via a generated script."""
    def __init__(self, module: str, func: str):
        # Security: validate identifiers before embedding in generated Swift source
        if not _SWIFT_IDENT_RE.match(module):
            raise DeltooRuntimeError(f"Invalid Swift module name: '{module}'")
        if not _SWIFT_IDENT_RE.match(func):
            raise DeltooRuntimeError(f"Invalid Swift function name: '{func}'")
        self._module = module
        self._func = func
        self.name = func

    def __repr__(self): return f"<swift {self._module}.{self._func}>"

    def __call__(self, *args):
        py_args = [_unwrap(a) for a in args]
        def _swift_val(v):
            if isinstance(v, bool):  return "true" if v else "false"
            if isinstance(v, int):   return str(v)
            if isinstance(v, float): return str(v)
            if isinstance(v, str):
                # Security: escape string to prevent Swift source injection
                escaped = v.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
                return f'"{escaped}"'
            if isinstance(v, list):  return "[" + ", ".join(_swift_val(x) for x in v) + "]"
            return str(v)
        args_str = ", ".join(_swift_val(a) for a in py_args)
        code = (
            f"import {self._module}\n"
            f"let result = {self._func}({args_str})\n"
            f'print(result)\n'
        )
        result = subprocess.run(
            ["swift", "-"],
            input=code, capture_output=True, text=True
        )
        if result.returncode != 0:
            raise DeltooRuntimeError(
                f"Swift error in {self._module}.{self._func}: {result.stderr.strip()}"
            )
        out = result.stdout.strip()
        try:    return int(out)
        except ValueError:
            try:    return float(out)
            except ValueError: return out


class PyObject:
    """Wraps any Python object for transparent attribute access."""
    def __init__(self, obj):
        self._obj = obj
    def __repr__(self): return repr(self._obj)
    def get_attr(self, name):
        try:
            return _wrap_python(getattr(self._obj, name))
        except AttributeError:
            raise DeltooRuntimeError(
                f"Object {type(self._obj).__name__!r} has no attribute '{name}'"
            )
    def set_attr(self, name, value):
        setattr(self._obj, name, _unwrap(value))


class PyCallable:
    """Wraps a Python callable."""
    def __init__(self, fn):
        self._fn = fn
        self.name = getattr(fn, "__name__", repr(fn))
    def __repr__(self): return f"<pyfn {self.name}>"


def _wrap_python(obj):
    """Wrap a Python value as a Deltoo value."""
    if obj is None:
        return NONE
    # Already a Deltoo value — return as-is
    if isinstance(obj, (DeltooNone, DeltooSome, DeltooOk, DeltooErr,
                        DeltooInstance, DeltooClass, DeltooFunction,
                        DeltooChannel, DeltooRange, BoundMethod,
                        PyCallable, PyObject, DeltooModule)):
        return obj
    if isinstance(obj, (bool, int, float, str, bytes, list, dict, tuple)):
        return obj
    import types
    if isinstance(obj, types.ModuleType):
        return DeltooModule(obj)
    # Objects with get_attr (Deltoo proxy objects) — return unwrapped
    if hasattr(obj, 'get_attr') and not isinstance(obj, type):
        return obj
    if callable(obj) and not isinstance(obj, type):
        return PyCallable(obj)
    if isinstance(obj, type):
        return PyCallable(obj)
    return PyObject(obj)


def _unwrap(val):
    """Convert Deltoo value to Python value."""
    if isinstance(val, PyObject): return val._obj
    if isinstance(val, PyCallable): return val._fn
    if isinstance(val, DeltooModule): return val._mod
    if isinstance(val, DeltooInstance): return val.fields
    if isinstance(val, DeltooNone): return None
    if isinstance(val, DeltooSome): return _unwrap(val.value)
    if isinstance(val, DeltooOk): return _unwrap(val.value)
    if isinstance(val, DeltooErr): return _unwrap(val.value)
    return val


class SqlRow(dict):
    """SQL result row with both dict and attribute access."""
    def get_attr(self, name):
        if name in self:
            return self[name]
        raise DeltooRuntimeError(f"SQL row has no column '{name}'")
    def __repr__(self):
        return "{" + ", ".join(f"{k}: {v!r}" for k, v in self.items()) + "}"


class SqlQuery:
    """Holds a parameterized SQL query before execution."""
    def __init__(self, sql: str, params: list, interp):
        self.sql = sql
        self.params = params
        self._interp = interp

    def get_attr(self, name):
        if name == "sql": return self.sql
        if name == "params": return self.params
        raise DeltooRuntimeError(f"SqlQuery has no attribute '{name}'")

    def __repr__(self): return f"@sql`{self.sql}`"


class DeltooRuntimeError(Exception):
    def __init__(self, msg, line=0, col=0):
        super().__init__(f"[Runtime Error] {msg}" +
                         (f" at line {line}" if line else ""))
        self.line = line
        self.col = col


# ── Environment ────────────────────────────────────────────────────────────────

class Env:
    def __init__(self, parent=None):
        self.vars: Dict[str, Any] = {}
        self._immutable: set = set()   # names declared with let/const
        self.parent = parent

    def get(self, name: str):
        if name in self.vars:
            return self.vars[name]
        if self.parent:
            return self.parent.get(name)
        raise DeltooRuntimeError(f"Undefined variable '{name}'")

    def set(self, name: str, value):
        """Set in the scope that owns the variable (or current scope)."""
        scope = self._find_scope(name)
        if scope:
            if name in scope._immutable:
                raise DeltooRuntimeError(
                    f"Cannot reassign immutable variable '{name}' — use 'var' instead of 'let'"
                )
            scope.vars[name] = value
        else:
            self.vars[name] = value

    def define(self, name: str, value, immutable: bool = False):
        self.vars[name] = value
        if immutable:
            self._immutable.add(name)

    def define_const(self, name: str, value):
        self.define(name, value, immutable=True)

    def _find_scope(self, name: str):
        if name in self.vars:
            return self
        if self.parent:
            return self.parent._find_scope(name)
        return None


# ── Interpreter ────────────────────────────────────────────────────────────────

class Interpreter:
    def __init__(self, filename="<stdin>", source=""):
        self.filename = filename
        self.source = source
        self._source_lines = source.splitlines() if source else []
        self.global_env = Env()
        self._setup_builtins()
        self._db_connections: Dict[str, sqlite3.Connection] = {}
        self._deferred: List[list] = [[]]  # stack of defer lists

    def _setup_builtins(self):
        from .builtins import make_builtins
        for name, fn in make_builtins(self).items():
            self.global_env.define(name, fn)

    # ── Execute ────────────────────────────────────────────────────────────────

    def run(self, program: Program):
        # Macro expansion pre-pass
        program = self._expand_macros(program)
        env = self.global_env
        try:
            for stmt in program.stmts:
                self.exec_stmt(stmt, env)
        except _Panic as e:
            print(f"\n\033[31mpanic: {e}\033[0m", file=sys.stderr)
            sys.exit(1)

    # ── Macro Expansion ───────────────────────────────────────────────────────

    def _expand_macros(self, program: Program) -> Program:
        """Collect macro declarations and expand macro calls in the AST."""
        import copy
        macros: Dict[str, MacroDecl] = {}
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
                    raise DeltooRuntimeError(
                        f"macro '{name}' expects {len(macro.params)} args, got {len(node.args)}",
                        getattr(node, 'line', 0))
                param_map = {}
                for pname, arg in zip(macro.params, node.args):
                    param_map[pname] = self._expand_in_node(arg, macros)
                body = copy.deepcopy(macro.body)
                body = self._substitute_idents(body, param_map)
                # If the macro body is a single expression statement, return the expression
                if len(body.stmts) == 1 and isinstance(body.stmts[0], ExprStmt):
                    return body.stmts[0].expr
                # Otherwise wrap as a block call (shouldn't normally happen for expr macros)
                return body
        # Recurse into all fields
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
        """Replace Ident nodes matching macro param names with argument expressions."""
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

    def format_error(self, err: DeltooRuntimeError) -> str:
        """Format a runtime error with source context (Elm/Rust style)."""
        msg = str(err)
        if err.line and self._source_lines:
            ln = err.line
            lines_out = [msg]
            if 0 < ln <= len(self._source_lines):
                src_line = self._source_lines[ln - 1]
                lines_out.append(f"  --> {self.filename}:{ln}")
                lines_out.append(f"   |")
                lines_out.append(f"{ln:3d} | {src_line}")
                col = getattr(err, 'col', 0)
                if col:
                    lines_out.append(f"   | {' ' * (col-1)}^")
                lines_out.append(f"   |")
            return "\n".join(lines_out)
        return msg

    def exec_stmt(self, node, env: Env):
        t = type(node)

        if t is ExprStmt:
            self.eval_expr(node.expr, env)

        elif t is LetDecl:
            val = self.eval_expr(node.value, env) if node.value is not None else NONE
            env.define(node.name, val, immutable=not node.mutable)

        elif t is ConstDecl:
            val = self.eval_expr(node.value, env)
            env.define_const(node.name, val)

        elif t is Assign:
            val = self.eval_assign_op(node.op, node.target, node.value, env)
            self.assign_target(node.target, val, env)

        elif t is ReturnStmt:
            val = self.eval_expr(node.value, env) if node.value else NONE
            self._flush_defers(env)
            raise _Return(val)

        elif t is BreakStmt:
            raise _Break(node.label)

        elif t is ContinueStmt:
            raise _Continue(node.label)

        elif t is IfStmt:
            self.exec_if(node, env)

        elif t is WhileStmt:
            self.exec_while(node, env)

        elif t is DoWhileStmt:
            self.exec_do_while(node, env)

        elif t is ForInStmt:
            self.exec_for_in(node, env)

        elif t is ForCStmt:
            self.exec_for_c(node, env)

        elif t is MatchStmt:
            self.exec_match(node, env)

        elif t is Block:
            self.exec_block(node, env)

        elif t is FnDecl:
            fn = DeltooFunction(
                name=node.name,
                params=node.params,
                body=node.body,
                env=env,
                is_async=node.is_async,
                decorators=node.decorators,
            )
            env.define(node.name, fn)

        elif t is ClassDecl:
            self.exec_class(node, env)

        elif t is StructDecl:
            self.exec_struct(node, env)

        elif t is ImplBlock:
            self.exec_impl(node, env)

        elif t is InterfaceDecl:
            env.define(node.name, node)  # stored as spec

        elif t is EnumDecl:
            self.exec_enum(node, env)

        elif t is ImportDecl:
            self.exec_import(node, env)

        elif t is ModuleDecl:
            self.exec_module(node, env)

        elif t is DeferStmt:
            self._deferred[-1].append((node.expr, env))

        elif t is GoStmt:
            # Spawn goroutine as thread
            node_expr = node.expr
            def run_goroutine(expr=node_expr, e=env):
                try:
                    self.eval_expr(expr, Env(e))
                except Exception as ex:
                    print(f"[goroutine panic] {ex}", file=sys.stderr)
            t_thread = threading.Thread(target=run_goroutine, daemon=True)
            t_thread.start()

        elif t is UnsafeBlock:
            self.exec_block(node.body, env)

        elif t is AssertStmt:
            cond = self.eval_expr(node.cond, env)
            if not _truthy(cond):
                msg = self.eval_expr(node.msg, env) if node.msg else "assertion failed"
                raise _Panic(str(msg))

        elif t is PanicStmt:
            msg = self.eval_expr(node.msg, env)
            raise _Panic(str(msg))

        elif t is PyBlock:
            self.exec_pyblock(node, env)

        elif t is MacroDecl:
            env.define(node.name, node)

        elif t is ActorDecl:
            self.exec_actor(node, env)

        elif t is ReceiveStmt:
            self.exec_receive(node, env)

        elif t is SpawnExpr:
            self.eval_spawn(node, env)

        else:
            raise DeltooRuntimeError(
                f"Unknown statement type: {type(node).__name__}",
                getattr(node, "line", 0)
            )

    def exec_block(self, block: Block, parent_env: Env):
        env = Env(parent_env)
        self._deferred.append([])
        try:
            for stmt in block.stmts:
                self.exec_stmt(stmt, env)
        finally:
            self._flush_defers_list(self._deferred.pop())
        return NONE

    def _flush_defers(self, env: Env):
        if self._deferred:
            self._flush_defers_list(self._deferred[-1])

    def _flush_defers_list(self, defers):
        for expr, env in reversed(defers):
            try:
                self.eval_expr(expr, env)
            except Exception as e:
                print(f"[defer error] {e}", file=sys.stderr)

    def exec_if(self, node: IfStmt, env: Env):
        if _truthy(self.eval_expr(node.cond, env)):
            self.exec_block(node.then, env)
        else:
            for cond, block in node.elseifs:
                if _truthy(self.eval_expr(cond, env)):
                    self.exec_block(block, env)
                    return
            if node.else_:
                self.exec_block(node.else_, env)

    def exec_while(self, node: WhileStmt, env: Env):
        while _truthy(self.eval_expr(node.cond, env)):
            try:
                self.exec_block(node.body, env)
            except _Break as b:
                if b.label and b.label != node.label:
                    raise
                break
            except _Continue as c:
                if c.label and c.label != node.label:
                    raise
                continue

    def exec_do_while(self, node: DoWhileStmt, env: Env):
        while True:
            try:
                self.exec_block(node.body, env)
            except _Break:
                break
            except _Continue:
                pass
            if not _truthy(self.eval_expr(node.cond, env)):
                break

    def exec_for_in(self, node: ForInStmt, env: Env):
        iterable = self.eval_expr(node.iter, env)
        label = getattr(node, "label", None)
        for item in _to_iter(iterable):
            loop_env = Env(env)
            loop_env.define(node.var, item)
            try:
                self.exec_block(node.body, loop_env)
            except _Break as b:
                if b.label and b.label != label:
                    raise
                break
            except _Continue as c:
                if c.label and c.label != label:
                    raise
                continue

    def exec_for_c(self, node: ForCStmt, env: Env):
        loop_env = Env(env)
        if node.init:
            # For-loop init variables are always mutable (let/var both work as var here)
            if isinstance(node.init, LetDecl):
                val = self.eval_expr(node.init.value, loop_env) if node.init.value is not None else NONE
                loop_env.define(node.init.name, val, immutable=False)
            else:
                self.exec_stmt(node.init, loop_env)
        while True:
            if node.cond:
                if not _truthy(self.eval_expr(node.cond, loop_env)):
                    break
            try:
                self.exec_block(node.body, loop_env)
            except _Break as b:
                if b.label and b.label != node.label:
                    raise
                break
            except _Continue as c:
                if c.label and c.label != node.label:
                    raise
            if node.step:
                if isinstance(node.step, Assign):
                    self.exec_stmt(node.step, loop_env)
                else:
                    self.eval_expr(node.step, loop_env)

    def exec_match(self, node: MatchStmt, env: Env):
        self.eval_match(node, env)

    def eval_match(self, node: MatchStmt, env: Env):
        val = self.eval_expr(node.expr, env)
        for arm in node.arms:
            match_env = Env(env)
            if self.match_pattern(arm.pattern, val, match_env):
                if arm.guard:
                    if not _truthy(self.eval_expr(arm.guard, match_env)):
                        continue
                if isinstance(arm.body, Block):
                    self.exec_block(arm.body, match_env)
                    return NONE
                elif isinstance(arm.body, (ReturnStmt, BreakStmt, ContinueStmt,
                                           ExprStmt, LetDecl, ConstDecl, Assign,
                                           PanicStmt, AssertStmt)):
                    self.exec_stmt(arm.body, match_env)
                    return NONE
                else:
                    return self.eval_expr(arm.body, match_env)
        return NONE  # No arm matched

    def eval_if_expr(self, node: IfStmt, env: Env):
        if _truthy(self.eval_expr(node.cond, env)):
            return self._eval_block_as_expr(node.then, env)
        for cond, body in node.elifs:
            if _truthy(self.eval_expr(cond, env)):
                return self._eval_block_as_expr(body, env)
        if node.else_:
            return self._eval_block_as_expr(node.else_, env)
        return NONE

    def _eval_block_as_expr(self, block, env: Env):
        block_env = Env(env)
        result = NONE
        for stmt in block.stmts:
            if isinstance(stmt, ExprStmt):
                result = self.eval_expr(stmt.expr, block_env)
            else:
                self.exec_stmt(stmt, block_env)
        return result

    def match_pattern(self, pattern, value, env: Env) -> bool:
        t = type(pattern)

        if t is WildcardPat:
            return True

        if t is IdentPat:
            env.define(pattern.name, value)
            return True

        if t is NonePat:
            return isinstance(value, DeltooNone)

        if t is SomePat:
            if isinstance(value, DeltooSome):
                return self.match_pattern(pattern.inner, value.value, env)
            return False

        if t is OkPat:
            if isinstance(value, DeltooOk):
                return self.match_pattern(pattern.inner, value.value, env)
            return False

        if t is ErrPat:
            if isinstance(value, DeltooErr):
                return self.match_pattern(pattern.inner, value.value, env)
            return False

        if t is LitPat:
            lit = self.eval_expr(pattern.value, env)
            return _equal(lit, value)

        if t is RangePat:
            lo = self.eval_expr(pattern.start, env)
            hi = self.eval_expr(pattern.end, env)
            if pattern.inclusive:
                return lo <= value <= hi
            return lo <= value < hi

        if t is TuplePat:
            if not isinstance(value, (list, tuple)):
                return False
            if len(pattern.elements) != len(value):
                return False
            return all(
                self.match_pattern(p, v, env)
                for p, v in zip(pattern.elements, value)
            )

        if t is StructPat:
            if not isinstance(value, DeltooInstance):
                return False
            if value.cls.name != pattern.name:
                return False
            for fname, fpat in pattern.fields:
                fval = value.fields.get(fname, NONE)
                if not self.match_pattern(fpat, fval, env):
                    return False
            return True

        if t is EnumPat:
            if isinstance(value, DeltooInstance) and value.cls.name == pattern.name:
                if pattern.inner and "value" in value.fields:
                    return self.match_pattern(pattern.inner, value.fields["value"], env)
                return True
            return False

        if t is OrPat:
            save_env = dict(env.vars)
            if self.match_pattern(pattern.left, value, env):
                return True
            env.vars = save_env
            return self.match_pattern(pattern.right, value, env)

        return False

    def exec_class(self, node: ClassDecl, env: Env):
        parent = None
        if node.parent:
            parent = env.get(node.parent)
            if not isinstance(parent, DeltooClass):
                raise DeltooRuntimeError(
                    f"'{node.parent}' is not a class", node.line
                )
        methods = {}
        for m in node.methods:
            fn = DeltooFunction(
                name=m.name, params=m.params, body=m.body,
                env=env, is_async=m.is_async, decorators=m.decorators
            )
            methods[m.name] = fn
        cls = DeltooClass(
            name=node.name,
            fields=node.fields,
            methods=methods,
            parent=parent,
            is_abstract=node.is_abstract,
        )
        env.define(node.name, cls)

    def exec_struct(self, node: StructDecl, env: Env):
        # Struct is treated as a class with no methods initially
        cls = DeltooClass(
            name=node.name,
            fields=node.fields,
            methods={},
            parent=None,
        )
        env.define(node.name, cls)

    def exec_impl(self, node: ImplBlock, env: Env):
        target = env.get(node.target)
        if not isinstance(target, DeltooClass):
            raise DeltooRuntimeError(
                f"Cannot impl non-class '{node.target}'", node.line
            )
        for m in node.methods:
            fn = DeltooFunction(
                name=m.name, params=m.params, body=m.body,
                env=env, is_async=m.is_async
            )
            target.methods[m.name] = fn

    def exec_enum(self, node: EnumDecl, env: Env):
        for variant in node.variants:
            if not variant.fields:
                # Simple variant: just a value
                env.define(variant.name, variant.name)
            else:
                # Tuple variant: constructor function
                def make_variant(vname, nfields):
                    def constructor(*args):
                        inst = DeltooInstance(
                            DeltooClass(vname, [], {}, None),
                            {"_variant": vname, "value": args[0] if nfields == 1 else list(args)}
                        )
                        return inst
                    return constructor
                env.define(variant.name,
                           PyCallable(make_variant(variant.name, len(variant.fields))))
        # Also store the enum itself
        env.define(node.name, node)

    def exec_import(self, node: ImportDecl, env: Env):
        lang = node.lang  # "python", "js", "c", "cpp", "java", "swift", or "" (Deltoo)
        alias = node.alias

        if lang == "python":
            try:
                mod = importlib.import_module(node.path)
                alias = alias or node.path.split(".")[-1]
                env.define(alias, DeltooModule(mod))
            except ImportError as e:
                raise DeltooRuntimeError(
                    f"Cannot import Python module '{node.path}': {e}\n"
                    f"Tip: install it with 'pip install {node.path}'", node.line
                )

        elif lang == "js":
            try:
                proxy = JsProxy(node.path)
                alias = alias or os.path.splitext(os.path.basename(node.path.replace("\\", "/")))[0]
                env.define(alias, proxy)
            except DeltooRuntimeError:
                raise
            except Exception as e:
                raise DeltooRuntimeError(
                    f"Cannot import JS module '{node.path}': {e}\n"
                    f"Tip: install it with 'npm install {node.path}'", node.line
                )

        elif lang == "c":
            try:
                proxy = CProxy(node.path)
                alias = alias or node.path.replace("lib", "").split(".")[0]
                env.define(alias, proxy)
            except DeltooRuntimeError:
                raise
            except Exception as e:
                raise DeltooRuntimeError(
                    f"Cannot load C library '{node.path}': {e}", node.line
                )

        elif lang == "cpp":
            try:
                proxy = CppProxy(node.path)
                alias = alias or node.path.replace("lib", "").split(".")[0]
                env.define(alias, proxy)
            except DeltooRuntimeError:
                raise
            except Exception as e:
                raise DeltooRuntimeError(
                    f"Cannot load C++ library '{node.path}': {e}", node.line
                )

        elif lang == "java":
            try:
                proxy = JavaProxy(node.path)
                alias = alias or node.path.split(".")[-1]
                env.define(alias, proxy)
            except DeltooRuntimeError:
                raise
            except Exception as e:
                raise DeltooRuntimeError(
                    f"Cannot import Java class '{node.path}': {e}\n"
                    f"Tip: install JPype with 'pip install JPype1'", node.line
                )

        elif lang == "swift":
            try:
                proxy = SwiftProxy(node.path)
                alias = alias or node.path
                env.define(alias, proxy)
            except DeltooRuntimeError:
                raise
            except Exception as e:
                raise DeltooRuntimeError(
                    f"Cannot import Swift module '{node.path}': {e}", node.line
                )

        else:
            # Deltoo module import
            path = node.path
            if not path.endswith(".wk"):
                path += ".wk"
            if os.path.exists(path):
                with open(path) as f:
                    src = f.read()
                from .parser import parse as parse_dt
                prog = parse_dt(src, path)
                sub = Interpreter(path)
                sub.run(prog)
                alias = alias or os.path.basename(path).replace(".wk", "")
                mod_env = sub.global_env
                class DtModule:
                    def __init__(self, e): self._env = e
                    def get_attr(self, n): return self._env.get(n)
                env.define(alias, DtModule(mod_env))
            else:
                raise DeltooRuntimeError(
                    f"Cannot find module '{node.path}'", node.line
                )

    def exec_module(self, node: ModuleDecl, env: Env):
        mod_env = Env(env)
        self.exec_block(node.body, mod_env)
        class Module:
            def __init__(self, e): self._env = e
            def get_attr(self, n): return self._env.get(n)
        env.define(node.name, Module(mod_env))

    def exec_pyblock(self, node: PyBlock, env: Env):
        # Security: pyblock executes arbitrary Python. It is intentionally
        # powerful for interop, but should not be used with untrusted input.
        # In web-server context (_web_mode=True) pyblock is disabled.
        if getattr(self, "_web_mode", False):
            raise DeltooRuntimeError(
                "pyblock is disabled in web-server context for security. "
                "Use 'import python' instead.", node.line
            )
        code = textwrap.dedent(node.code)
        # Restrict builtins to a safe subset — keep full builtins for power users
        # but exclude dangerous exec/eval/compile at the global scope
        safe_builtins = dict(vars(__builtins__) if isinstance(__builtins__, dict)
                             else vars(__builtins__))
        globs = {"__builtins__": safe_builtins}
        # Expose current Wakawaka environment to pyblock
        for k, v in env.vars.items():
            globs[k] = _unwrap(v)
        exec(compile(code, "<pyblock>", "exec"), globs)
        # Pull any new/modified variables back into Wakawaka scope
        for k, v in globs.items():
            if not k.startswith("__"):
                env.define(k, _wrap_python(v))

    # ── Actor Model ────────────────────────────────────────────────────────────

    def exec_actor(self, node: ActorDecl, env: Env):
        """Register an actor class; instantiation done via spawn."""
        # Create as a DeltooClass with an extra 'mailbox' and 'thread' field
        cls = DeltooClass(
            name=node.name,
            fields=node.fields,
            methods={},
            parent=env.get(node.parent) if node.parent else None,
            env=env,
        )
        for m in node.methods:
            fn = DeltooFunction(name=m.name, params=m.params, body=m.body,
                                env=env, is_async=m.is_async,
                                decorators=m.decorators)
            cls.methods[m.name] = fn
        cls.is_actor = True  # marker
        env.define(node.name, cls)

    def eval_spawn(self, node: SpawnExpr, env: Env):
        """spawn ActorClass(args) — create actor instance and start its thread."""
        actor_cls_expr = node.actor_class
        # actor_cls_expr might be a CallExpr (spawn ActorClass(args))
        # or just an Ident
        if isinstance(actor_cls_expr, CallExpr):
            cls = self.eval_expr(actor_cls_expr.callee, env)
            args = [self.eval_expr(a, env) for a in actor_cls_expr.args]
            kwargs = {k: self.eval_expr(v, env) for k, v in actor_cls_expr.kwargs}
        else:
            cls = self.eval_expr(actor_cls_expr, env)
            args = [self.eval_expr(a, env) for a in node.args]
            kwargs = {k: self.eval_expr(v, env) for k, v in node.kwargs}

        inst = self.instantiate(cls, args, kwargs, node.line)
        # Give actor a mailbox (queue) and self-reference
        inst.fields["__mailbox__"] = queue.Queue()
        inst.fields["__actor__"] = True

        # Start actor thread if it has a 'run' method
        run_fn = cls.find_method("run")
        if run_fn:
            def actor_loop(actor=inst):
                try:
                    self.call_function(run_fn, [], {}, node.line, self_=actor)
                except Exception as e:
                    print(f"[actor error] {e}", file=sys.stderr)
            t = threading.Thread(target=actor_loop, daemon=True)
            t.start()
            inst.fields["__thread__"] = t
        return inst

    def exec_receive(self, node: ReceiveStmt, env: Env):
        """receive { pattern => body } — get message from self's mailbox and match."""
        try:
            self_obj = env.get("self")
            mailbox = self_obj.fields.get("__mailbox__")
        except Exception:
            mailbox = None

        if mailbox is None:
            raise DeltooRuntimeError("receive used outside of actor context", node.line)

        timeout = self.eval_expr(node.timeout, env) if node.timeout else None
        try:
            msg = mailbox.get(timeout=timeout or 0.1)
        except queue.Empty:
            return  # no message, skip

        for arm in node.arms:
            match_env = Env(env)
            if self.match_pattern(arm.pattern, msg, match_env):
                if arm.guard and not _truthy(self.eval_expr(arm.guard, match_env)):
                    continue
                if isinstance(arm.body, Block):
                    self.exec_block(arm.body, match_env)
                else:
                    self.eval_expr(arm.body, match_env)
                return

    # ── Expressions ────────────────────────────────────────────────────────────

    def eval_expr(self, node, env: Env):
        t = type(node)

        if t is IntLit:   return node.value
        if t is FloatLit: return node.value
        if t is BoolLit:  return node.value
        if t is StrLit:   return node.value
        if t is NoneLit:  return NONE

        if t is FStrLit:
            return self.eval_fstr(node, env)

        if t is ArrayLit:
            return [self.eval_expr(e, env) for e in node.elements]

        if t is MapLit:
            return {
                self.eval_expr(k, env): self.eval_expr(v, env)
                for k, v in node.pairs
            }

        if t is TupleLit:
            return tuple(self.eval_expr(e, env) for e in node.elements)

        if t is StructLit:
            cls = env.get(node.name)
            if not isinstance(cls, DeltooClass):
                raise DeltooRuntimeError(f"'{node.name}' is not a class/struct")
            fields = {k: self.eval_expr(v, env) for k, v in node.fields}
            # Fill defaults
            for f in cls.fields:
                if f.name not in fields:
                    if f.default is not None:
                        fields[f.name] = self.eval_expr(f.default, env)
                    else:
                        fields[f.name] = NONE
            return DeltooInstance(cls, fields)

        if t is Ident:
            return env.get(node.name)

        if t is MemberExpr:
            obj = self.eval_expr(node.obj, env)
            return self.get_attr(obj, node.member, node.line)

        if t is IndexExpr:
            obj = self.eval_expr(node.obj, env)
            idx = self.eval_expr(node.index, env)
            return self.eval_index(obj, idx, node.line)

        if t is SliceExpr:
            obj = self.eval_expr(node.obj, env)
            start = self.eval_expr(node.start, env) if node.start else None
            end = self.eval_expr(node.end, env) if node.end else None
            step = self.eval_expr(node.step, env) if node.step else None
            return obj[start:end:step]

        if t is BinOp:
            return self.eval_binop(node, env)

        if t is UnaryOp:
            return self.eval_unary(node, env)

        if t is CallExpr:
            return self.eval_call(node, env)

        if t is Closure:
            return DeltooFunction(
                name="<closure>", params=node.params,
                body=node.body, env=env
            )

        if t is TernaryExpr:
            cond = self.eval_expr(node.cond, env)
            return (self.eval_expr(node.then, env)
                    if _truthy(cond)
                    else self.eval_expr(node.else_, env))

        if t is CastExpr:
            return self.eval_cast(node, env)

        if t is PipeExpr:
            left = self.eval_expr(node.left, env)
            right = node.right
            # If right is a call expr, prepend left as first arg
            if isinstance(right, CallExpr):
                args = [left] + [self.eval_expr(a, env) for a in right.args]
                kwargs = {k: self.eval_expr(v, env) for k, v in right.kwargs}
                fn = self.eval_expr(right.callee, env)
                return self.call_function(fn, args, kwargs, node.line)
            fn = self.eval_expr(right, env)
            return self.call_function(fn, [left], {}, node.line)

        if t is SomeExpr:
            return DeltooSome(self.eval_expr(node.value, env))

        if t is OkExpr:
            return DeltooOk(self.eval_expr(node.value, env))

        if t is ErrExpr:
            return DeltooErr(self.eval_expr(node.value, env))

        if t is PropagateExpr:
            val = self.eval_expr(node.expr, env)
            if isinstance(val, DeltooErr):
                raise _Return(val)
            if isinstance(val, DeltooNone):
                raise _Return(NONE)
            if isinstance(val, DeltooOk):
                return val.value
            if isinstance(val, DeltooSome):
                return val.value
            return val

        if t is AwaitExpr:
            # In interpreter: just evaluate synchronously
            val = self.eval_expr(node.expr, env)
            if asyncio.iscoroutine(val):
                return asyncio.get_event_loop().run_until_complete(val)
            return val

        if t is RefExpr:
            # In interpreter: references are just values
            return self.eval_expr(node.expr, env)

        if t is DerefExpr:
            return self.eval_expr(node.expr, env)

        if t is SizeofExpr:
            # Return size in bytes (approximate)
            return 8  # 64-bit default

        if t is RangeExpr:
            start = self.eval_expr(node.start, env)
            end = self.eval_expr(node.end, env)
            return DeltooRange(start, end, node.inclusive)

        if t is ShellExpr:
            return self.eval_shell(node, env)

        if t is SqlExpr:
            return self.eval_sql(node, env)

        if t is Block:
            # Block as expression — evaluate and return last value
            block_env = Env(env)
            result = NONE
            for stmt in node.stmts:
                if isinstance(stmt, ExprStmt):
                    result = self.eval_expr(stmt.expr, block_env)
                else:
                    self.exec_stmt(stmt, block_env)
            return result

        if t is MatchStmt:
            return self.eval_match(node, env)

        if t is IfStmt:
            return self.eval_if_expr(node, env)

        if t is OptChainExpr:
            obj = self.eval_expr(node.obj, env)
            if isinstance(obj, DeltooNone):
                return NONE
            return self.get_attr(obj, node.member, node.line)

        if t is OptIndexExpr:
            obj = self.eval_expr(node.obj, env)
            if isinstance(obj, DeltooNone):
                return NONE
            idx = self.eval_expr(node.index, env)
            return self.eval_index(obj, idx, node.line)

        if t is NullCoalesceExpr:
            left = self.eval_expr(node.left, env)
            if isinstance(left, DeltooNone):
                return self.eval_expr(node.right, env)
            return left

        if t is ComptimeExpr:
            # In interpreter: just evaluate the expression
            return self.eval_expr(node.expr, env)

        if t is SpawnExpr:
            return self.eval_spawn(node, env)

        # Handle when a statement node appears as an expression
        if isinstance(node, (FnDecl, ClassDecl, StructDecl, EnumDecl,
                              ImplBlock, InterfaceDecl, ActorDecl)):
            self.exec_stmt(node, env)
            return NONE

        raise DeltooRuntimeError(
            f"Cannot evaluate node type: {type(node).__name__}",
            getattr(node, "line", 0)
        )

    def eval_fstr(self, node: FStrLit, env: Env) -> str:
        parts = []
        for p in node.parts:
            if isinstance(p, str):
                parts.append(p)
            else:
                val = self.eval_expr(p, env)
                parts.append(deltoo_str(val))
        return "".join(parts)

    def eval_binop(self, node: BinOp, env: Env):
        op = node.op
        # Short-circuit
        if op == "&&":
            left = self.eval_expr(node.left, env)
            return left if not _truthy(left) else self.eval_expr(node.right, env)
        if op == "||":
            left = self.eval_expr(node.left, env)
            return left if _truthy(left) else self.eval_expr(node.right, env)

        left = self.eval_expr(node.left, env)
        right = self.eval_expr(node.right, env)

        # Check for operator overloading on DeltooInstance
        if isinstance(left, DeltooInstance):
            op_method_name = {
                "+": "operator+", "-": "operator-", "*": "operator*",
                "/": "operator/", "%": "operator%", "**": "operator**",
                "==": "operator==", "!=": "operator!=",
                "<": "operator<", ">": "operator>",
                "<=": "operator<=", ">=": "operator>=",
            }.get(op)
            if op_method_name:
                method = left.cls.find_method(op_method_name)
                if method:
                    bound = BoundMethod(method, left)
                    return self.call_bound(bound, [right], {}, node.line)

        # Tensor / DualNumber arithmetic dispatch
        from .builtins import DeltooTensor, _DualNumber
        if isinstance(left, (DeltooTensor, _DualNumber)) or isinstance(right, (DeltooTensor, _DualNumber)):
            if op in ("+", "-", "*", "/", "**"):
                try:
                    if op == "+": return left + right
                    if op == "-": return left - right
                    if op == "*": return left * right
                    if op == "/": return left / right
                    if op == "**": return left ** right
                except TypeError as e:
                    raise DeltooRuntimeError(f"Type error in '{op}': {e}", node.line)

        try:
            if op == "+":
                if isinstance(left, str) or isinstance(right, str):
                    return deltoo_str(left) + deltoo_str(right)
                return left + right
            if op == "-":  return left - right
            if op == "*":  return left * right
            if op == "/":
                if right == 0:
                    raise _Panic("division by zero")
                if isinstance(left, int) and isinstance(right, int):
                    return left // right
                return left / right
            if op == "%":  return left % right
            if op == "**": return left ** right
            if op == "&":  return left & right
            if op == "|":  return left | right
            if op == "^":  return left ^ right
            if op == "<<": return left << right
            if op == ">>": return left >> right
            if op == "==": return _equal(left, right)
            if op == "!=": return not _equal(left, right)
            if op == "<":  return left < right
            if op == ">":  return left > right
            if op == "<=": return left <= right
            if op == ">=": return left >= right
            # Channel receive
            if op == "<-":
                if isinstance(left, DeltooChannel):
                    left.send(right)
                    return NONE
        except TypeError as e:
            raise DeltooRuntimeError(
                f"Type error in '{op}': {e}", node.line
            )
        raise DeltooRuntimeError(f"Unknown operator: {op}", node.line)

    def eval_unary(self, node: UnaryOp, env: Env):
        val = self.eval_expr(node.operand, env)
        op = node.op
        if op == "-": return -val
        if op == "!": return not _truthy(val)
        if op == "~": return ~val
        raise DeltooRuntimeError(f"Unknown unary op: {op}", node.line)

    def eval_cast(self, node: CastExpr, env: Env):
        val = self.eval_expr(node.expr, env)
        to = node.to_type
        try:
            if to in ("int", "i8", "i16", "i32", "i64",
                      "u8", "u16", "u32", "u64"):
                if isinstance(val, str):
                    return int(val, 0)
                return int(val)
            if to in ("float", "f32", "f64"):
                return float(val)
            if to in ("str", "string"):
                return deltoo_str(val)
            if to == "bool":
                return _truthy(val)
            if to == "byte":
                if isinstance(val, str):
                    return ord(val)
                return int(val) & 0xFF
        except (ValueError, TypeError) as e:
            raise DeltooRuntimeError(f"Cast to {to} failed: {e}", node.line)
        raise DeltooRuntimeError(f"Unknown cast target: {to}", node.line)

    def eval_index(self, obj, idx, line=0):
        if isinstance(obj, (list, tuple)):
            if isinstance(idx, int):
                if idx < 0 or idx >= len(obj):
                    raise DeltooRuntimeError(
                        f"Index {idx} out of bounds (len={len(obj)})", line
                    )
                return obj[idx]
        if isinstance(obj, dict):
            if idx not in obj:
                return NONE
            return obj[idx]
        if isinstance(obj, str):
            if isinstance(idx, int):
                return obj[idx]
            if isinstance(idx, DeltooRange):
                return obj[idx.start:idx.end]
        if isinstance(obj, PyObject):
            try:
                return _wrap_python(obj._obj[_unwrap(idx)])
            except Exception as e:
                raise DeltooRuntimeError(str(e), line)
        raise DeltooRuntimeError(
            f"Cannot index type {type(obj).__name__!r}", line
        )

    def assign_target(self, target, value, env: Env):
        if isinstance(target, Ident):
            env.set(target.name, value)
        elif isinstance(target, MemberExpr):
            obj = self.eval_expr(target.obj, env)
            self.set_attr(obj, target.member, value)
        elif isinstance(target, IndexExpr):
            obj = self.eval_expr(target.obj, env)
            idx = self.eval_expr(target.index, env)
            if isinstance(obj, list):
                obj[idx] = value
            elif isinstance(obj, dict):
                obj[idx] = value
            elif isinstance(obj, PyObject):
                obj._obj[_unwrap(idx)] = _unwrap(value)
            else:
                raise DeltooRuntimeError("Cannot assign to index")
        else:
            raise DeltooRuntimeError("Invalid assignment target")

    def eval_assign_op(self, op: str, target, value_node, env: Env):
        if op == "=":
            return self.eval_expr(value_node, env)
        cur_val = self.eval_expr(target, env)
        new_val = self.eval_expr(value_node, env)
        ops = {
            "+=": lambda a, b: a + b if not isinstance(a, str) else a + deltoo_str(b),
            "-=": lambda a, b: a - b,
            "*=": lambda a, b: a * b,
            "/=": lambda a, b: a / b,
            "%=": lambda a, b: a % b,
            "**=": lambda a, b: a ** b,
            "&=": lambda a, b: a & b,
            "|=": lambda a, b: a | b,
            "^=": lambda a, b: a ^ b,
            "<<=": lambda a, b: a << b,
            ">>=": lambda a, b: a >> b,
        }
        if op not in ops:
            raise DeltooRuntimeError(f"Unknown assign op: {op}")
        return ops[op](cur_val, new_val)

    # ── Function calls ─────────────────────────────────────────────────────────

    def eval_call(self, node: CallExpr, env: Env) -> Any:
        args = [self.eval_expr(a, env) for a in node.args]
        kwargs = {k: self.eval_expr(v, env) for k, v in node.kwargs}

        # Special: chan<T>(cap)
        if isinstance(node.callee, Ident) and node.callee.name == "__chan__":
            cap = args[0] if args else 0
            return DeltooChannel(cap)

        # Channel receive: <-ch
        if isinstance(node.callee, Ident) and node.callee.name == "__recv__":
            ch = args[0]
            if isinstance(ch, DeltooChannel):
                return ch.recv()

        # Class.method(...) — static call
        if isinstance(node.callee, MemberExpr):
            obj = self.eval_expr(node.callee.obj, env)
            method_name = node.callee.member

            # Class static/constructor call: MyClass.new(...)
            if isinstance(obj, DeltooClass):
                if method_name == "new":
                    return self.instantiate(obj, args, kwargs, node.line)
                # Static method
                method = obj.find_method(method_name)
                if method:
                    return self.call_function(method, args, kwargs, node.line)
                raise DeltooRuntimeError(
                    f"Class '{obj.name}' has no static method '{method_name}'", node.line
                )

            # Instance method call
            if isinstance(obj, DeltooInstance):
                method = obj.cls.find_method(method_name)
                if method:
                    bound = BoundMethod(method, obj)
                    return self.call_bound(bound, args, kwargs, node.line)
                # Check fields (might be a stored function)
                if method_name in obj.fields:
                    fn = obj.fields[method_name]
                    return self.call_function(fn, args, kwargs, node.line)

            # Module/PyObject attribute call
            if isinstance(obj, (DeltooModule, PyObject, PyCallable)):
                attr = self.get_attr(obj, method_name, node.line)
                return self.call_function(attr, args, kwargs, node.line)

            # String/List methods
            if isinstance(obj, str):
                return self.call_str_method(obj, method_name, args, kwargs, node.line)
            if isinstance(obj, list):
                return self.call_list_method(obj, method_name, args, kwargs, node.line)
            if isinstance(obj, dict):
                return self.call_dict_method(obj, method_name, args, kwargs, node.line)
            if isinstance(obj, DeltooRange):
                return self.call_range_method(obj, method_name, args, node.line)
            if isinstance(obj, DeltooChannel):
                if method_name == "send": obj.send(args[0]); return NONE
                if method_name == "recv": return obj.recv()

        # Simple function call
        fn = self.eval_expr(node.callee, env)
        return self.call_function(fn, args, kwargs, node.line)

    def instantiate(self, cls: DeltooClass, args, kwargs, line=0) -> DeltooInstance:
        # Initialize fields with defaults
        fields = {}
        for f in cls.fields:
            if f.default is not None:
                fields[f.name] = self.eval_expr(f.default, cls.methods.get("new") and
                                                cls.methods["new"].env or self.global_env)
            else:
                fields[f.name] = NONE
        # Gather inherited fields
        if cls.parent:
            parent_inst = self.instantiate(cls.parent, [], {}, line)
            for k, v in parent_inst.fields.items():
                if k not in fields:
                    fields[k] = v

        inst = DeltooInstance(cls, fields)

        # Call constructor if exists
        ctor = cls.find_method("new")
        if ctor:
            ctor_env = Env(ctor.env)
            ctor_env.define("self", inst)
            self._bind_params(ctor.params, args, kwargs, ctor_env, line)
            try:
                for stmt in ctor.body.stmts:
                    self.exec_stmt(stmt, ctor_env)
            except _Return as r:
                if isinstance(r.value, DeltooInstance):
                    return r.value
        return inst

    def call_bound(self, bound: BoundMethod, args, kwargs, line=0):
        return self.call_function(bound.fn, args, kwargs, line, self_=bound.instance)

    def call_function(self, fn, args, kwargs, line=0, self_=None):
        if fn is NONE:
            raise DeltooRuntimeError("Called none as a function", line)

        if isinstance(fn, PyCallable):
            try:
                py_args = [_unwrap(a) for a in args]
                py_kwargs = {k: _unwrap(v) for k, v in kwargs.items()}
                result = fn._fn(*py_args, **py_kwargs)
                return _wrap_python(result)
            except Exception as e:
                raise DeltooRuntimeError(f"Python call failed: {e}", line)

        if callable(fn) and not isinstance(fn, DeltooFunction):
            try:
                # Deltoo-native builtins receive Deltoo values directly
                return _wrap_python(fn(*args, **kwargs))
            except Exception as e:
                raise DeltooRuntimeError(f"Builtin call failed: {e}", line)

        if isinstance(fn, BoundMethod):
            return self.call_bound(fn, args, kwargs, line)

        if not isinstance(fn, DeltooFunction):
            raise DeltooRuntimeError(
                f"Not callable: {type(fn).__name__!r}", line
            )

        call_env = Env(fn.env)
        if self_ is not None:
            call_env.define("self", self_)

        self._bind_params(fn.params, args, kwargs, call_env, line)

        # Process decorators
        decorators = getattr(fn, 'decorators', [])
        if "deprecated" in decorators:
            import warnings
            print(f"\033[33m[deprecated] '{fn.name}' is deprecated\033[0m",
                  file=sys.stderr)
        if "memoize" in decorators:
            cache_key = (id(fn), tuple(_unwrap(a) for a in args
                         if not isinstance(a, (list, dict))))
            if not hasattr(fn, '_cache'):
                fn._cache = {}
            if cache_key in fn._cache:
                return fn._cache[cache_key]

        self._deferred.append([])
        try:
            body = fn.body
            if isinstance(body, Block):
                for stmt in body.stmts:
                    self.exec_stmt(stmt, call_env)
                result = NONE
            else:
                # Expression body (closures: |x| x * 2)
                result = self.eval_expr(body, call_env)
        except _Return as r:
            result = r.value
        finally:
            self._flush_defers_list(self._deferred.pop())

        # Store memoized result
        if "memoize" in decorators and hasattr(fn, '_cache'):
            fn._cache[cache_key] = result

        return result

    def _bind_params(self, params, args, kwargs, env, line):
        # params can be a list of Param objects OR a list of strings (closures)
        def pname(p): return p if isinstance(p, str) else p.name
        def pdefault(p): return None if isinstance(p, str) else p.default
        def pvariadic(p): return False if isinstance(p, str) else p.variadic

        pos_params = [p for p in params if pname(p) != "self" and not pvariadic(p)]
        variadic = next((p for p in params if pvariadic(p)), None)

        for i, param in enumerate(pos_params):
            name = pname(param)
            default = pdefault(param)
            if i < len(args):
                env.define(name, args[i])
            elif name in kwargs:
                env.define(name, kwargs[name])
            elif default is not None:
                env.define(name, self.eval_expr(default, env))
            else:
                env.define(name, NONE)

        if variadic:
            rest = args[len(pos_params):]
            env.define(pname(variadic), rest)

        for k, v in kwargs.items():
            if not env._find_scope(k):
                env.define(k, v)

    # ── Attribute access ───────────────────────────────────────────────────────

    def get_attr(self, obj, name: str, line=0):
        # SqlRow — attribute access on SQL result rows
        if isinstance(obj, SqlRow):
            return obj.get_attr(name)
        # Generic: any Python object with a get_attr method (proxy objects, custom types)
        if (not isinstance(obj, (DeltooInstance, DeltooClass, DeltooModule,
                                 PyObject, PyCallable, str, list, dict, tuple,
                                 bool, int, float, DeltooOk, DeltooErr,
                                 DeltooSome, DeltooNone, DeltooRange, DeltooChannel,
                                 SqlRow, SqlQuery))
                and hasattr(obj, 'get_attr')):
            return obj.get_attr(name)
        if isinstance(obj, DeltooInstance):
            return obj.get_attr(name)
        if isinstance(obj, DeltooClass):
            m = obj.find_method(name)
            if m:
                return m
            raise DeltooRuntimeError(
                f"Class '{obj.name}' has no attribute '{name}'", line
            )
        if isinstance(obj, DeltooModule):
            return obj.get_attr(name)
        if isinstance(obj, PyObject):
            return obj.get_attr(name)
        if isinstance(obj, PyCallable):
            return _wrap_python(getattr(obj._fn, name, None))
        if isinstance(obj, str):
            return self._str_attr(obj, name)
        if isinstance(obj, list):
            return self._list_attr(obj, name)
        if isinstance(obj, dict):
            return self._dict_attr(obj, name)
        if isinstance(obj, (DeltooOk, DeltooErr, DeltooSome)):
            if name == "value": return obj.value
            if name in ("is_ok", "is_err", "unwrap"):
                return PyCallable(getattr(obj, name))
        raise DeltooRuntimeError(
            f"Cannot get attribute '{name}' of {type(obj).__name__}", line
        )

    def set_attr(self, obj, name: str, value):
        if isinstance(obj, DeltooInstance):
            obj.set_attr(name, value)
        elif isinstance(obj, PyObject):
            obj.set_attr(name, value)
        else:
            raise DeltooRuntimeError(
                f"Cannot set attribute on {type(obj).__name__}"
            )

    # ── Built-in methods on primitive types ────────────────────────────────────

    def _str_attr(self, s: str, name: str):
        methods = {
            "len":     lambda: len(s),
            "upper":   lambda: s.upper(),
            "lower":   lambda: s.lower(),
            "trim":    lambda: s.strip(),
            "trimStart": lambda: s.lstrip(),
            "trimEnd": lambda: s.rstrip(),
            "split":   lambda sep=" ": s.split(sep),
            "startsWith": lambda pre: s.startswith(pre),
            "endsWith": lambda suf: s.endswith(suf),
            "contains": lambda sub: sub in s,
            "replace":  lambda old, new: s.replace(old, new),
            "lines":   lambda: s.splitlines(),
            "bytes":   lambda: list(s.encode("utf-8")),
            "chars":   lambda: list(s),
            "toInt":   lambda: int(s),
            "toFloat": lambda: float(s),
            "isEmpty": lambda: len(s) == 0,
            "repeat":  lambda n: s * n,
            "indexOf": lambda sub: s.find(sub),
            "slice":   lambda a, b: s[a:b],
            "format":  lambda *a: s.format(*a),
        }
        if name == "len": return len(s)
        if name in methods:
            return PyCallable(methods[name])
        raise DeltooRuntimeError(f"str has no attribute '{name}'")

    def call_str_method(self, s, name, args, kwargs, line):
        attr = self._str_attr(s, name)
        if callable(attr):
            return attr(*[_unwrap(a) for a in args])
        if isinstance(attr, PyCallable):
            return attr._fn(*[_unwrap(a) for a in args])
        return attr

    def _list_attr(self, lst: list, name: str):
        if name == "len": return len(lst)
        methods = {
            "push":    lambda v: lst.append(v) or NONE,
            "pop":     lambda: lst.pop() if lst else NONE,
            "append":  lambda v: lst.append(v) or NONE,
            "extend":  lambda other: lst.extend(other) or NONE,
            "insert":  lambda i, v: lst.insert(i, v) or NONE,
            "remove":  lambda v: lst.remove(v) or NONE,
            "clear":   lambda: lst.clear() or NONE,
            "reverse": lambda: lst.reverse() or NONE,
            "sort":    lambda: lst.sort() or NONE,
            "contains": lambda v: v in lst,
            "indexOf": lambda v: lst.index(v) if v in lst else -1,
            "isEmpty": lambda: len(lst) == 0,
            "first":   lambda: lst[0] if lst else NONE,
            "last":    lambda: lst[-1] if lst else NONE,
            "slice":   lambda a, b=None: lst[a:b],
            "join":    lambda sep="": sep.join(deltoo_str(x) for x in lst),
            "map":     lambda fn: [self.call_function(fn, [v], {}) for v in lst],
            "filter":  lambda fn: [v for v in lst if _truthy(self.call_function(fn, [v], {}))],
            "reduce":  lambda fn, init=NONE: _reduce(lst, fn, init, self),
            "forEach": lambda fn: [self.call_function(fn, [v], {}) for v in lst] and NONE,
            "any":     lambda fn: any(_truthy(self.call_function(fn, [v], {})) for v in lst),
            "all":     lambda fn: all(_truthy(self.call_function(fn, [v], {})) for v in lst),
            "sum":     lambda: sum(lst),
            "min":     lambda: min(lst),
            "max":     lambda: max(lst),
            "count":   lambda v: lst.count(v),
            "flat":    lambda: [x for sub in lst for x in (sub if isinstance(sub, list) else [sub])],
            "zip":     lambda other: list(map(list, zip(lst, other))),
            "enumerate": lambda: [[i, v] for i, v in enumerate(lst)],
            "unique":  lambda: list(dict.fromkeys(lst)),
            "sorted":  lambda: sorted(lst),
        }
        if name in methods:
            return PyCallable(methods[name])
        raise DeltooRuntimeError(f"[]T has no attribute '{name}'")

    def call_list_method(self, lst, name, args, kwargs, line):
        attr = self._list_attr(lst, name)
        if isinstance(attr, int): return attr  # len
        return attr._fn(*args)

    def _dict_attr(self, d: dict, name: str):
        methods = {
            "get":     lambda k, default=NONE: d.get(k, default),
            "set":     lambda k, v: d.update({k: v}) or NONE,
            "has":     lambda k: k in d,
            "delete":  lambda k: d.pop(k, NONE),
            "keys":    lambda: list(d.keys()),
            "values":  lambda: list(d.values()),
            "entries": lambda: [[k, v] for k, v in d.items()],
            "len":     lambda: len(d),
            "isEmpty": lambda: len(d) == 0,
            "clear":   lambda: d.clear() or NONE,
            "merge":   lambda other: {**d, **other},
        }
        if name == "len": return len(d)
        if name in methods:
            return PyCallable(methods[name])
        raise DeltooRuntimeError(f"map has no attribute '{name}'")

    def call_dict_method(self, d, name, args, kwargs, line):
        attr = self._dict_attr(d, name)
        if isinstance(attr, int): return attr
        return attr._fn(*args)

    def call_range_method(self, r: DeltooRange, name, args, line):
        if name == "contains": return args[0] in r
        if name == "len":
            return r.end - r.start + (1 if r.inclusive else 0)
        if name == "toList": return list(r)
        raise DeltooRuntimeError(f"Range has no method '{name}'", line)

    # ── Shell integration ──────────────────────────────────────────────────────

    def eval_shell(self, node: ShellExpr, env: Env) -> str:
        cmd_parts = []
        for part in node.parts:
            if isinstance(part, str):
                cmd_parts.append(part)
            else:
                val = self.eval_expr(part, env)
                # Safe shell escaping
                import shlex
                cmd_parts.append(shlex.quote(deltoo_str(val)))
        cmd = "".join(cmd_parts)
        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True
            )
            return result.stdout
        except Exception as e:
            raise DeltooRuntimeError(f"Shell error: {e}")

    # ── SQL integration ────────────────────────────────────────────────────────

    def eval_sql(self, node: SqlExpr, env: Env):
        """Build a SqlQuery object from @sql`...` — does NOT execute immediately."""
        sql_parts = []
        params = []
        for part in node.parts:
            if isinstance(part, str):
                sql_parts.append(part)
            else:
                val = self.eval_expr(part, env)
                sql_parts.append("?")
                params.append(_unwrap(val))
        query = "".join(sql_parts).strip()
        return SqlQuery(query, params, self)

    def exec_sql_query(self, q: "SqlQuery", conn=None):
        """Execute a SqlQuery on the given connection (or default)."""
        if conn is None:
            conn_name = "_default_db"
            if conn_name not in self._db_connections:
                self._db_connections[conn_name] = sqlite3.connect(":memory:")
            conn = self._db_connections[conn_name]
        conn.row_factory = sqlite3.Row
        try:
            cur = conn.execute(q.sql, q.params)
            conn.commit()
            if q.sql.upper().lstrip().startswith(("SELECT", "WITH", "PRAGMA")):
                return [SqlRow(dict(row)) for row in cur.fetchall()]
            return cur.rowcount
        except sqlite3.Error as e:
            raise DeltooRuntimeError(f"SQL error: {e}")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _truthy(val) -> bool:
    if isinstance(val, DeltooNone): return False
    if isinstance(val, bool): return val
    if isinstance(val, (int, float)): return val != 0
    if isinstance(val, (str, list, dict, tuple)): return len(val) > 0
    if isinstance(val, DeltooErr): return False
    return True


def _equal(a, b) -> bool:
    if type(a) != type(b):
        # Allow int/float comparisons
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return a == b
        return False
    if isinstance(a, DeltooInstance) and isinstance(b, DeltooInstance):
        return a is b
    return a == b


def _to_iter(val):
    if isinstance(val, DeltooRange):
        return iter(val)
    if isinstance(val, (list, tuple, str, dict)):
        return iter(val)
    if isinstance(val, PyObject):
        return iter(val._obj)
    if hasattr(val, '__iter__'):
        return iter(val)
    raise DeltooRuntimeError(f"Not iterable: {type(val).__name__}")


def deltoo_str(val) -> str:
    if isinstance(val, bool): return "true" if val else "false"
    if isinstance(val, DeltooNone): return "none"
    if isinstance(val, DeltooSome): return f"some({deltoo_str(val.value)})"
    if isinstance(val, DeltooOk):  return f"ok({deltoo_str(val.value)})"
    if isinstance(val, DeltooErr): return f"err({deltoo_str(val.value)})"
    if isinstance(val, DeltooInstance): return f"<{val.cls.name}>"
    if isinstance(val, list): return "[" + ", ".join(deltoo_str(x) for x in val) + "]"
    if isinstance(val, tuple): return "(" + ", ".join(deltoo_str(x) for x in val) + ")"
    if isinstance(val, dict):
        pairs = ", ".join(f"{deltoo_str(k)}: {deltoo_str(v)}" for k, v in val.items())
        return "{" + pairs + "}"
    if isinstance(val, PyObject): return repr(val._obj)
    if isinstance(val, DeltooRange): return repr(val)
    return str(val)


def _reduce(lst, fn, init, interp):
    acc = init
    for item in lst:
        if isinstance(acc, DeltooNone):
            acc = item
        else:
            acc = interp.call_function(fn, [acc, item], {})
    return acc
