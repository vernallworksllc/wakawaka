"""
Deltoo Built-in Functions
All built-ins available in every Deltoo program.
"""
import sys
import os
import math
import time
import random
import json
import re
import hashlib
import statistics
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .interpreter import Interpreter


# ── Module-level classes (importable from interpreter) ────────────────────────

class _RuntimeError(Exception):
    """Lightweight error for tensor/dual ops before DeltooRuntimeError is available."""
    pass


class DeltooTensor:
    """Pure-Python tensor that works with or without numpy."""
    def __init__(self, data, shape=None):
        if isinstance(data, list):
            self._data = [float(x) for x in self._flatten_list(data)]
            self._shape = shape if shape else [len(self._data)]
        else:
            try:
                import numpy as np
                if isinstance(data, np.ndarray):
                    self._data = data.astype(float).flatten().tolist()
                    self._shape = list(data.shape)
                    return
            except ImportError:
                pass
            self._data = [float(data)]
            self._shape = shape if shape else [1]

    @staticmethod
    def _flatten_list(lst):
        result = []
        for item in lst:
            if isinstance(item, (list, tuple)):
                result.extend(DeltooTensor._flatten_list(item))
            else:
                result.append(item)
        return result

    def __repr__(self):
        if len(self._data) <= 10:
            return f"tensor({self._data})"
        return f"tensor([{', '.join(str(x) for x in self._data[:5])}, ..., {', '.join(str(x) for x in self._data[-2:])}])"

    def _binop(self, other, op):
        if isinstance(other, DeltooTensor):
            if len(self._data) != len(other._data):
                raise _RuntimeError("tensor shapes don't match")
            return DeltooTensor([op(a, b) for a, b in zip(self._data, other._data)], list(self._shape))
        s = float(other)
        return DeltooTensor([op(a, s) for a in self._data], list(self._shape))

    def __add__(self, other): return self._binop(other, lambda a, b: a + b)
    def __radd__(self, other): return self._binop(other, lambda a, b: b + a)
    def __sub__(self, other): return self._binop(other, lambda a, b: a - b)
    def __rsub__(self, other): return self._binop(other, lambda a, b: b - a)
    def __mul__(self, other): return self._binop(other, lambda a, b: a * b)
    def __rmul__(self, other): return self._binop(other, lambda a, b: b * a)
    def __truediv__(self, other): return self._binop(other, lambda a, b: a / b)
    def __rtruediv__(self, other): return self._binop(other, lambda a, b: b / a)

    def _transpose(self):
        if len(self._shape) != 2:
            return DeltooTensor(list(self._data), list(reversed(self._shape)))
        rows, cols = self._shape
        result = []
        for c in range(cols):
            for r in range(rows):
                result.append(self._data[r * cols + c])
        return DeltooTensor(result, [cols, rows])

    def _matmul(self, other):
        if not isinstance(other, DeltooTensor):
            raise _RuntimeError("matmul requires two tensors")
        if len(self._shape) != 2 or len(other._shape) != 2:
            raise _RuntimeError("matmul requires 2D tensors")
        m, k1 = self._shape
        k2, n = other._shape
        if k1 != k2:
            raise _RuntimeError(f"matmul shape mismatch: {self._shape} vs {other._shape}")
        result = []
        for i in range(m):
            for j in range(n):
                s = 0.0
                for p in range(k1):
                    s += self._data[i * k1 + p] * other._data[p * n + j]
                result.append(s)
        return DeltooTensor(result, [m, n])

    def _dot(self, other):
        if not isinstance(other, DeltooTensor):
            raise _RuntimeError("dot requires two tensors")
        if len(self._data) != len(other._data):
            raise _RuntimeError("dot product requires same-length tensors")
        return sum(a * b for a, b in zip(self._data, other._data))


class _DualNumber:
    def __init__(self, val, deriv=0.0):
        self.val = float(val)
        self.deriv = float(deriv)
    def __repr__(self):
        return f"dual({self.val}, {self.deriv})"
    def __add__(self, other):
        if isinstance(other, _DualNumber):
            return _DualNumber(self.val + other.val, self.deriv + other.deriv)
        return _DualNumber(self.val + float(other), self.deriv)
    def __radd__(self, other):
        return _DualNumber(float(other) + self.val, self.deriv)
    def __sub__(self, other):
        if isinstance(other, _DualNumber):
            return _DualNumber(self.val - other.val, self.deriv - other.deriv)
        return _DualNumber(self.val - float(other), self.deriv)
    def __rsub__(self, other):
        return _DualNumber(float(other) - self.val, -self.deriv)
    def __mul__(self, other):
        if isinstance(other, _DualNumber):
            return _DualNumber(self.val * other.val, self.val * other.deriv + self.deriv * other.val)
        o = float(other)
        return _DualNumber(self.val * o, self.deriv * o)
    def __rmul__(self, other):
        o = float(other)
        return _DualNumber(o * self.val, o * self.deriv)
    def __truediv__(self, other):
        if isinstance(other, _DualNumber):
            return _DualNumber(self.val / other.val, (self.deriv * other.val - self.val * other.deriv) / (other.val ** 2))
        o = float(other)
        return _DualNumber(self.val / o, self.deriv / o)
    def __rtruediv__(self, other):
        o = float(other)
        return _DualNumber(o / self.val, -o * self.deriv / (self.val ** 2))
    def __pow__(self, other):
        n = float(other) if not isinstance(other, _DualNumber) else other.val
        return _DualNumber(self.val ** n, n * self.val ** (n - 1) * self.deriv)
    def __neg__(self):
        return _DualNumber(-self.val, -self.deriv)


def make_builtins(interp: "Interpreter") -> dict:
    from .interpreter import (
        NONE, DeltooNone, DeltooSome, DeltooOk, DeltooErr,
        DeltooRange, DeltooChannel, DeltooInstance, DeltooClass,
        DeltooRuntimeError, deltoo_str, _truthy, _unwrap, _wrap_python,
        PyCallable
    )

    def _dt_str(val):
        return deltoo_str(val)

    # ── I/O ───────────────────────────────────────────────────────────────────

    def println(*args, sep=" ", end="\n"):
        print(*[_dt_str(a) for a in args], sep=sep, end=end)
        return NONE

    def print_(*args, sep=" ", end=""):
        print(*[_dt_str(a) for a in args], sep=sep, end=end, flush=True)
        return NONE

    def eprintln(*args):
        print(*[_dt_str(a) for a in args], file=sys.stderr)
        return NONE

    def readln(prompt=""):
        try:
            return input(prompt)
        except EOFError:
            return NONE

    def readlines():
        return sys.stdin.read().splitlines()

    # ── Type conversion ───────────────────────────────────────────────────────

    def to_int(val, base=10):
        try:
            if isinstance(val, str):
                return int(val, base)
            return int(val)
        except (ValueError, TypeError):
            return NONE

    def to_float(val):
        try:
            return float(val)
        except (ValueError, TypeError):
            return NONE

    def to_str(val):
        return _dt_str(val)

    def to_bool(val):
        return _truthy(val)

    def to_list(val):
        if isinstance(val, list): return val
        if isinstance(val, (tuple, str, dict)):
            return list(val)
        if isinstance(val, DeltooRange):
            return list(val)
        return [val]

    def to_bytes(val):
        if isinstance(val, str):
            return list(val.encode("utf-8"))
        return list(val)

    def chr_(code):
        return chr(int(code))

    def ord_(ch):
        return ord(str(ch)[0])

    # ── Type checks ───────────────────────────────────────────────────────────

    def is_int(v):    return isinstance(v, int) and not isinstance(v, bool)
    def is_float(v):  return isinstance(v, float)
    def is_str(v):    return isinstance(v, str)
    def is_bool(v):   return isinstance(v, bool)
    def is_list(v):   return isinstance(v, list)
    def is_map(v):    return isinstance(v, dict)
    def is_none(v):   return isinstance(v, DeltooNone)
    def is_some(v):   return isinstance(v, DeltooSome)
    def is_ok(v):     return isinstance(v, DeltooOk)
    def is_err(v):    return isinstance(v, DeltooErr)
    def type_of(v):
        if isinstance(v, bool): return "bool"
        if isinstance(v, int): return "int"
        if isinstance(v, float): return "float"
        if isinstance(v, str): return "str"
        if isinstance(v, list): return "[]"
        if isinstance(v, dict): return "map"
        if isinstance(v, tuple): return "tuple"
        if isinstance(v, DeltooNone): return "none"
        if isinstance(v, DeltooSome): return "some"
        if isinstance(v, DeltooOk): return "ok"
        if isinstance(v, DeltooErr): return "err"
        if isinstance(v, DeltooInstance): return v.cls.name
        if isinstance(v, DeltooClass): return "class"
        if isinstance(v, DeltooRange): return "range"
        if isinstance(v, DeltooChannel): return "chan"
        return type(v).__name__

    # ── Collections ───────────────────────────────────────────────────────────

    def len_(val):
        if isinstance(val, DeltooNone): return 0
        if isinstance(val, DeltooRange):
            return val.end - val.start + (1 if val.inclusive else 0)
        try:
            return len(val)
        except TypeError:
            raise DeltooRuntimeError(f"len() not supported for {type_of(val)}")

    def range_(start, end=None, step=1):
        if end is None:
            end = start
            start = 0
        if step == 1:
            return DeltooRange(start, end, False)
        return list(range(int(start), int(end), int(step)))

    def zip_(*iters):
        return [list(t) for t in zip(*iters)]

    def enumerate_(lst, start=0):
        return [[i, v] for i, v in enumerate(lst, start)]

    def map_(lst, fn):
        return [interp.call_function(fn, [v], {}) for v in lst]

    def filter_(lst, fn):
        return [v for v in lst if _truthy(interp.call_function(fn, [v], {}))]

    def reduce_(lst, fn, init=NONE):
        from .interpreter import _reduce
        return _reduce(lst, fn, init, interp)

    def sorted_(lst, key=None, reverse=False):
        if key and key is not NONE:
            return sorted(lst, key=lambda x: interp.call_function(key, [x], {}),
                          reverse=bool(reverse))
        return sorted(lst, reverse=bool(reverse))

    def reversed_(lst):
        return list(reversed(lst))

    def any_(lst, fn=None):
        if fn and fn is not NONE:
            return any(_truthy(interp.call_function(fn, [x], {})) for x in lst)
        return any(_truthy(x) for x in lst)

    def all_(lst, fn=None):
        if fn and fn is not NONE:
            return all(_truthy(interp.call_function(fn, [x], {})) for x in lst)
        return all(_truthy(x) for x in lst)

    def sum_(lst):
        return sum(lst)

    def min_(lst, *args):
        if args: return min(lst, *args)
        return min(lst)

    def max_(lst, *args):
        if args: return max(lst, *args)
        return max(lst)

    # ── Math ──────────────────────────────────────────────────────────────────

    class MathModule:
        PI = math.pi
        E = math.e
        TAU = math.tau
        INF = math.inf
        NAN = math.nan

        @staticmethod
        def sqrt(x): return math.sqrt(x)
        @staticmethod
        def cbrt(x): return x ** (1/3)
        @staticmethod
        def pow(x, y): return math.pow(x, y)
        @staticmethod
        def abs(x): return abs(x)
        @staticmethod
        def floor(x): return math.floor(x)
        @staticmethod
        def ceil(x): return math.ceil(x)
        @staticmethod
        def round(x, n=0): return round(x, n)
        @staticmethod
        def log(x, base=math.e): return math.log(x, base)
        @staticmethod
        def log2(x): return math.log2(x)
        @staticmethod
        def log10(x): return math.log10(x)
        @staticmethod
        def sin(x): return math.sin(x)
        @staticmethod
        def cos(x): return math.cos(x)
        @staticmethod
        def tan(x): return math.tan(x)
        @staticmethod
        def asin(x): return math.asin(x)
        @staticmethod
        def acos(x): return math.acos(x)
        @staticmethod
        def atan(x): return math.atan(x)
        @staticmethod
        def atan2(y, x): return math.atan2(y, x)
        @staticmethod
        def sinh(x): return math.sinh(x)
        @staticmethod
        def cosh(x): return math.cosh(x)
        @staticmethod
        def tanh(x): return math.tanh(x)
        @staticmethod
        def exp(x): return math.exp(x)
        @staticmethod
        def gcd(a, b): return math.gcd(int(a), int(b))
        @staticmethod
        def lcm(a, b): return abs(a*b) // math.gcd(int(a), int(b))
        @staticmethod
        def factorial(n): return math.factorial(int(n))
        @staticmethod
        def isnan(x): return math.isnan(x)
        @staticmethod
        def isinf(x): return math.isinf(x)
        @staticmethod
        def clamp(x, lo, hi): return max(lo, min(hi, x))
        @staticmethod
        def lerp(a, b, t): return a + (b - a) * t
        @staticmethod
        def sign(x): return (1 if x > 0 else -1) if x != 0 else 0

    math_mod = MathModule()

    class _MathProxy:
        _consts = {
            "PI": math.pi, "E": math.e, "TAU": math.tau,
            "INF": math.inf, "NAN": math.nan,
        }
        def __init__(self): pass
        def get_attr(self, name):
            if name in self._consts:
                return self._consts[name]
            obj = getattr(MathModule, name, None)
            if obj is None:
                raise DeltooRuntimeError(f"math has no attribute '{name}'")
            if callable(obj):
                return PyCallable(obj)
            return obj
    math_proxy = _MathProxy()

    # ── Stats ─────────────────────────────────────────────────────────────────

    class _StatsProxy:
        def get_attr(self, name):
            fns = {
                "mean":    lambda lst: statistics.mean(lst),
                "median":  lambda lst: statistics.median(lst),
                "mode":    lambda lst: statistics.mode(lst),
                "stdev":   lambda lst: statistics.stdev(lst),
                "variance":lambda lst: statistics.variance(lst),
                "pstdev":  lambda lst: statistics.pstdev(lst),
                "pvariance": lambda lst: statistics.pvariance(lst),
                "percentile": lambda lst, p: sorted(lst)[int(len(lst) * p / 100)],
                "sum":     lambda lst: sum(lst),
                "min":     lambda lst: min(lst),
                "max":     lambda lst: max(lst),
                "range":   lambda lst: max(lst) - min(lst),
                "normalize": lambda lst: (
                    [x / sum(lst) for x in lst] if sum(lst) else lst
                ),
                "zscore":  lambda lst: (
                    [(x - statistics.mean(lst)) / statistics.stdev(lst) for x in lst]
                    if len(lst) > 1 else [0.0]
                ),
                "correlation": _pearson_corr,
            }
            if name in fns:
                return PyCallable(fns[name])
            raise DeltooRuntimeError(f"stats has no attribute '{name}'")
    stats_proxy = _StatsProxy()

    def _pearson_corr(xs, ys):
        n = len(xs)
        if n < 2:
            return 0.0
        mx, my = sum(xs)/n, sum(ys)/n
        num = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
        dx = sum((x-mx)**2 for x in xs)
        dy = sum((y-my)**2 for y in ys)
        denom = (dx * dy) ** 0.5
        return num / denom if denom else 0.0

    # ── Random ───────────────────────────────────────────────────────────────

    class _RandProxy:
        def get_attr(self, name):
            fns = {
                "int":    lambda lo=0, hi=100: random.randint(int(lo), int(hi)),
                "float":  lambda lo=0.0, hi=1.0: random.uniform(lo, hi),
                "bool":   lambda: random.random() < 0.5,
                "choice": lambda lst: random.choice(lst),
                "shuffle": lambda lst: random.shuffle(lst) or lst,
                "sample": lambda lst, k: random.sample(lst, k),
                "seed":   lambda s: random.seed(s) or NONE,
                "uuid":   lambda: __import__("uuid").uuid4().hex,
            }
            if name in fns:
                return PyCallable(fns[name])
            raise DeltooRuntimeError(f"rand has no attribute '{name}'")
    rand_proxy = _RandProxy()

    # ── Time ─────────────────────────────────────────────────────────────────

    class _TimeProxy:
        def get_attr(self, name):
            fns = {
                "now":     lambda: time.time(),
                "sleep":   lambda s: time.sleep(s) or NONE,
                "format":  lambda ts, fmt="%Y-%m-%d %H:%M:%S": (
                    __import__("datetime").datetime
                    .fromtimestamp(ts).strftime(fmt)
                ),
                "parse":   lambda s, fmt="%Y-%m-%d": (
                    __import__("datetime").datetime
                    .strptime(s, fmt).timestamp()
                ),
                "today":   lambda: __import__("datetime").date.today().isoformat(),
                "year":    lambda: __import__("datetime").datetime.now().year,
                "month":   lambda: __import__("datetime").datetime.now().month,
                "day":     lambda: __import__("datetime").datetime.now().day,
                "hour":    lambda: __import__("datetime").datetime.now().hour,
                "minute":  lambda: __import__("datetime").datetime.now().minute,
                "second":  lambda: __import__("datetime").datetime.now().second,
            }
            if name in fns:
                return PyCallable(fns[name])
            raise DeltooRuntimeError(f"time has no attribute '{name}'")
    time_proxy = _TimeProxy()

    # ── File I/O ──────────────────────────────────────────────────────────────

    class _FileProxy:
        def get_attr(self, name):
            fns = {
                "read":   lambda p: open(p, encoding="utf-8").read(),
                "write":  lambda p, s: open(p, "w", encoding="utf-8").write(s) or NONE,
                "append": lambda p, s: open(p, "a", encoding="utf-8").write(s) or NONE,
                "exists": lambda p: os.path.exists(p),
                "lines":  lambda p: open(p, encoding="utf-8").read().splitlines(),
                "eachLine": lambda p, fn: _file_each_line(p, fn),
                "bytes":  lambda p: list(open(p, "rb").read()),
                "remove": lambda p: os.remove(p) or NONE,
                "size":   lambda p: os.path.getsize(p),
                "mkdir":  lambda p: os.makedirs(p, exist_ok=True) or NONE,
                "ls":     lambda p=".": os.listdir(p),
                "isdir":  lambda p: os.path.isdir(p),
                "isfile": lambda p: os.path.isfile(p),
                "join":   lambda *p: os.path.join(*p),
                "basename": lambda p: os.path.basename(p),
                "dirname": lambda p: os.path.dirname(p),
                "abspath": lambda p: os.path.abspath(p),
                "open":   lambda p, mode="r": DeltooFile(open(p, mode, encoding="utf-8" if "b" not in mode else None)),
            }
            if name in fns:
                return PyCallable(fns[name])
            raise DeltooRuntimeError(f"file has no attribute '{name}'")

    def _file_each_line(path, fn):
        with open(path, encoding="utf-8") as f:
            for raw in f:
                interp.call_function(fn, [raw.rstrip("\n")], {})
        return NONE

    class DeltooFile:
        def __init__(self, f): self._f = f
        def get_attr(self, name):
            fns = {
                "read":     lambda: self._f.read(),
                "readLine": lambda: (lambda s: s.rstrip("\n") if s != "" else NONE)(self._f.readline()),
                "write":    lambda s: self._f.write(s) or NONE,
                "close":    lambda: self._f.close() or NONE,
                "lines":    lambda: self._f.read().splitlines(),
                "flush":    lambda: self._f.flush() or NONE,
            }
            if name in fns:
                return PyCallable(fns[name])
            raise DeltooRuntimeError(f"File has no attribute '{name}'")
        def __iter__(self):
            for raw in self._f:
                yield raw.rstrip("\n")
        def __enter__(self): return self
        def __exit__(self, *_): self._f.close()

    file_proxy = _FileProxy()

    # ── OS / Env ──────────────────────────────────────────────────────────────

    class _OsProxy:
        def get_attr(self, name):
            fns = {
                "env":    lambda k, default="": os.environ.get(k, default),
                "setenv": lambda k, v: os.environ.update({k: v}) or NONE,
                "cwd":    lambda: os.getcwd(),
                "chdir":  lambda p: os.chdir(p) or NONE,
                "args":   lambda: sys.argv[1:],
                "exit":   lambda code=0: sys.exit(int(code)),
                "pid":    lambda: os.getpid(),
                "platform": lambda: sys.platform,
                "hostname": lambda: __import__("socket").gethostname(),
                "username": lambda: os.environ.get("USER", os.environ.get("USERNAME", "")),
            }
            if name in fns:
                return PyCallable(fns[name])
            raise DeltooRuntimeError(f"os has no attribute '{name}'")
    os_proxy = _OsProxy()

    # ── JSON ──────────────────────────────────────────────────────────────────

    class _JsonProxy:
        def get_attr(self, name):
            fns = {
                "parse":   lambda s: json.loads(s),
                "stringify": lambda v, indent=None: json.dumps(
                    _unwrap(v),
                    indent=indent if indent is not NONE else None,
                    ensure_ascii=False
                ),
                "load":    lambda p: json.load(open(p)),
                "dump":    lambda p, v: json.dump(_unwrap(v), open(p, "w"), indent=2) or NONE,
            }
            if name in fns:
                return PyCallable(fns[name])
            raise DeltooRuntimeError(f"json has no attribute '{name}'")
    json_proxy = _JsonProxy()

    # ── CSV ───────────────────────────────────────────────────────────────────

    import csv as _csv_mod
    import io as _io

    class _CsvProxy:
        def get_attr(self, name):
            fns = {
                # csv.load(path) -> list of maps (dicts with header keys)
                "load": lambda p: _csv_load(p),
                # csv.loadRows(path) -> list of lists (no header assumed)
                "loadRows": lambda p: _csv_load_rows(p),
                # csv.parse(text) -> list of maps
                "parse": lambda s: _csv_parse(s),
                # csv.parseRows(text) -> list of lists
                "parseRows": lambda s: _csv_parse_rows(s),
                # csv.stringify(rows) -> CSV string (rows = list of maps or list of lists)
                "stringify": lambda rows, headers=NONE: _csv_stringify(rows, headers),
                # csv.dump(path, rows) -> writes CSV file
                "dump": lambda p, rows, headers=NONE: _csv_dump(p, rows, headers),
            }
            if name in fns:
                return PyCallable(fns[name])
            raise DeltooRuntimeError(f"csv has no attribute '{name}'")

    def _csv_load(path):
        with open(path, newline="", encoding="utf-8-sig") as f:
            reader = _csv_mod.DictReader(f)
            return [dict(row) for row in reader]

    def _csv_load_rows(path):
        with open(path, newline="", encoding="utf-8-sig") as f:
            reader = _csv_mod.reader(f)
            return [list(row) for row in reader]

    def _csv_parse(text):
        reader = _csv_mod.DictReader(_io.StringIO(text))
        return [dict(row) for row in reader]

    def _csv_parse_rows(text):
        reader = _csv_mod.reader(_io.StringIO(text))
        return [list(row) for row in reader]

    def _csv_stringify(rows, headers=NONE):
        rows = _unwrap(rows)
        buf = _io.StringIO()
        if not rows:
            return ""
        if isinstance(rows[0], dict):
            keys = list(rows[0].keys()) if isinstance(headers, DeltooNone) else list(_unwrap(headers))
            w = _csv_mod.DictWriter(buf, fieldnames=keys, lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
        else:
            w = _csv_mod.writer(buf, lineterminator="\n")
            w.writerows(rows)
        return buf.getvalue()

    def _csv_dump(path, rows, headers=NONE):
        text = _csv_stringify(rows, headers)
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(text)
        return NONE

    csv_proxy = _CsvProxy()

    # ── YAML ──────────────────────────────────────────────────────────────────

    class _YamlProxy:
        def get_attr(self, name):
            fns = {
                # yaml.load(path) -> parsed object
                "load":      lambda p: _yaml_load(p),
                # yaml.parse(text) -> parsed object
                "parse":     lambda s: _yaml_parse(s),
                # yaml.parseAll(text) -> list of all documents
                "parseAll":  lambda s: _yaml_parse_all(s),
                # yaml.stringify(val) -> YAML string
                "stringify": lambda v: _yaml_stringify(v),
                # yaml.dump(path, val) -> writes YAML file
                "dump":      lambda p, v: _yaml_dump(p, v),
            }
            if name in fns:
                return PyCallable(fns[name])
            raise DeltooRuntimeError(f"yaml has no attribute '{name}'")

    def _yaml_load(path):
        try:
            import yaml
            with open(path, encoding="utf-8") as f:
                return yaml.safe_load(f) or NONE
        except ImportError:
            raise DeltooRuntimeError(
                "YAML support requires PyYAML. Install with: pip install pyyaml"
            )

    def _yaml_parse(text):
        try:
            import yaml
            result = yaml.safe_load(_unwrap(text))
            return result if result is not None else NONE
        except ImportError:
            raise DeltooRuntimeError(
                "YAML support requires PyYAML. Install with: pip install pyyaml"
            )

    def _yaml_parse_all(text):
        try:
            import yaml
            return list(yaml.safe_load_all(_unwrap(text)))
        except ImportError:
            raise DeltooRuntimeError(
                "YAML support requires PyYAML. Install with: pip install pyyaml"
            )

    def _yaml_stringify(val):
        try:
            import yaml
            return yaml.dump(_unwrap(val), default_flow_style=False, allow_unicode=True).rstrip()
        except ImportError:
            raise DeltooRuntimeError(
                "YAML support requires PyYAML. Install with: pip install pyyaml"
            )

    def _yaml_dump(path, val):
        try:
            import yaml
            with open(_unwrap(path), "w", encoding="utf-8") as f:
                yaml.dump(_unwrap(val), f, default_flow_style=False, allow_unicode=True)
            return NONE
        except ImportError:
            raise DeltooRuntimeError(
                "YAML support requires PyYAML. Install with: pip install pyyaml"
            )

    yaml_proxy = _YamlProxy()

    # ── SQL module ────────────────────────────────────────────────────────────

    class _SqlModule:
        def get_attr(self, name):
            fns = {
                "connect": lambda url: _sql_connect(url, interp),
            }
            if name in fns:
                return PyCallable(fns[name])
            raise DeltooRuntimeError(f"sql has no attribute '{name}'")

    class DeltooDb:
        def __init__(self, conn, name="_"):
            self.conn = conn
            self.name = name
        def get_attr(self, name):
            fns = {
                "query":  lambda q, *p: self._query(q, p),
                "exec":   lambda q, *p: self._exec(q, p),
                "transaction": lambda fn: self._transaction(fn),
                "close":  lambda: self.conn.close() or NONE,
            }
            if name in fns:
                return PyCallable(fns[name])
            raise DeltooRuntimeError(f"DB has no attribute '{name}'")

        def _resolve(self, q, params=()):
            """Accept SqlQuery or raw (sql_str, params)."""
            from .interpreter import SqlQuery
            if isinstance(q, SqlQuery):
                return q.sql, q.params
            # Raw string
            return str(q), [_unwrap(p) for p in params]

        def _query(self, q, params=()):
            import sqlite3 as _sq
            from .interpreter import SqlRow
            sql_, prms = self._resolve(q, params)
            self.conn.row_factory = _sq.Row
            cur = self.conn.execute(sql_, prms)
            return [SqlRow(dict(r)) for r in cur.fetchall()]

        def _exec(self, q, params=()):
            sql_, prms = self._resolve(q, params)
            cur = self.conn.execute(sql_, prms)
            self.conn.commit()
            return cur.rowcount

        def _transaction(self, fn):
            try:
                interp.call_function(fn, [self], {})
                self.conn.commit()
            except Exception as e:
                self.conn.rollback()
                raise

    def _sql_connect(url: str, interp_ref):
        import sqlite3 as _sq
        if url.startswith("sqlite:"):
            path = url[7:]
        else:
            path = url
        conn = _sq.connect(path)
        db = DeltooDb(conn, path)
        interp_ref._db_connections[path] = conn
        return db

    sql_proxy = _SqlModule()

    # ── Crypto / Hash ─────────────────────────────────────────────────────────

    class _CryptoProxy:
        def get_attr(self, name):
            fns = {
                "md5":    lambda s: hashlib.md5(s.encode()).hexdigest(),
                "sha1":   lambda s: hashlib.sha1(s.encode()).hexdigest(),
                "sha256": lambda s: hashlib.sha256(s.encode()).hexdigest(),
                "sha512": lambda s: hashlib.sha512(s.encode()).hexdigest(),
                "random_bytes": lambda n: os.urandom(int(n)),
                "uuid":   lambda: __import__("uuid").uuid4().hex,
            }
            if name in fns:
                return PyCallable(fns[name])
            raise DeltooRuntimeError(f"crypto has no attribute '{name}'")
    crypto_proxy = _CryptoProxy()

    # ── HTTP ──────────────────────────────────────────────────────────────────

    class _HttpProxy:
        def get_attr(self, name):
            fns = {
                "get":  lambda url, headers=None: _http_get(url, headers),
                "post": lambda url, body=None, headers=None: _http_post(url, body, headers),
            }
            if name in fns:
                return PyCallable(fns[name])
            raise DeltooRuntimeError(f"http has no attribute '{name}'")

    class HttpResponse:
        def __init__(self, status, body, headers):
            self.fields = {"status": status, "body": body, "headers": headers}
        def get_attr(self, name):
            if name == "text": return PyCallable(lambda: self.fields["body"])
            if name == "json": return PyCallable(lambda: json.loads(self.fields["body"]))
            if name in self.fields: return self.fields[name]
            raise DeltooRuntimeError(f"Response has no attribute '{name}'")

    def _http_get(url, headers=None):
        try:
            import urllib.request
            req = urllib.request.Request(url, headers=headers or {})
            with urllib.request.urlopen(req) as r:
                body = r.read().decode()
                hdrs = dict(r.headers)
                return HttpResponse(r.status, body, hdrs)
        except Exception as e:
            raise DeltooRuntimeError(f"HTTP GET failed: {e}")

    def _http_post(url, body=None, headers=None):
        try:
            import urllib.request, urllib.parse
            data = None
            if body:
                if isinstance(body, dict):
                    data = json.dumps(body).encode()
                elif isinstance(body, str):
                    data = body.encode()
            req = urllib.request.Request(url, data=data, headers=headers or {})
            with urllib.request.urlopen(req) as r:
                return HttpResponse(r.status, r.read().decode(), dict(r.headers))
        except Exception as e:
            raise DeltooRuntimeError(f"HTTP POST failed: {e}")

    http_proxy = _HttpProxy()

    # ── Regex ─────────────────────────────────────────────────────────────────

    class _RegexProxy:
        def get_attr(self, name):
            fns = {
                "match":   lambda pat, s: DeltooSome(re.match(pat, s).group()) if re.match(pat, s) else NONE,
                "search":  lambda pat, s: DeltooSome(re.search(pat, s).group()) if re.search(pat, s) else NONE,
                "findAll": lambda pat, s: re.findall(pat, s),
                "replace": lambda pat, repl, s: re.sub(pat, repl, s),
                "split":   lambda pat, s: re.split(pat, s),
                "test":    lambda pat, s: bool(re.search(pat, s)),
                "compile": lambda pat: PyCallable(re.compile(pat).search),
            }
            if name in fns:
                return PyCallable(fns[name])
            raise DeltooRuntimeError(f"regex has no attribute '{name}'")
    regex_proxy = _RegexProxy()

    # ── ML proxy (uses scikit-learn / numpy / matplotlib if available) ─────────

    class _MlProxy:
        def get_attr(self, name):
            fns = {
                "LinearRegression": _make_linear_regression,
                "LogisticRegression": _make_logistic_regression,
                "KMeans": _make_kmeans,
                "trainTestSplit": _train_test_split,
            }
            if name in fns:
                return PyCallable(fns[name])
            raise DeltooRuntimeError(f"ml has no attribute '{name}'")

    def _make_linear_regression():
        try:
            from sklearn.linear_model import LinearRegression
            class LR:
                def __init__(self): self._model = LinearRegression()
                def get_attr(self, n):
                    fns = {
                        "fit":     lambda X, y: self._fit(X, y),
                        "predict": lambda X: self._model.predict(_unwrap(X)).tolist(),
                        "r2Score": lambda X, y: self._model.score(_unwrap(X), _unwrap(y)),
                        "coef":    lambda: self._model.coef_.tolist(),
                        "intercept": lambda: float(self._model.intercept_),
                    }
                    if n in fns: return PyCallable(fns[n])
                    raise DeltooRuntimeError(f"LinearRegression has no attribute '{n}'")
                def _fit(self, X, y):
                    self._model.fit(_unwrap(X), _unwrap(y))
                    return self
            return LR()
        except ImportError:
            raise DeltooRuntimeError(
                "sklearn not installed. Run: pip install scikit-learn"
            )

    def _make_logistic_regression():
        try:
            from sklearn.linear_model import LogisticRegression
            class LgR:
                def __init__(self): self._model = LogisticRegression()
                def get_attr(self, n):
                    fns = {
                        "fit":     lambda X, y: self._model.fit(_unwrap(X), _unwrap(y)) or self,
                        "predict": lambda X: self._model.predict(_unwrap(X)).tolist(),
                        "score":   lambda X, y: self._model.score(_unwrap(X), _unwrap(y)),
                        "proba":   lambda X: self._model.predict_proba(_unwrap(X)).tolist(),
                    }
                    if n in fns: return PyCallable(fns[n])
                    raise DeltooRuntimeError(f"LogisticRegression has no attribute '{n}'")
            return LgR()
        except ImportError:
            raise DeltooRuntimeError("sklearn not installed")

    def _make_kmeans(k=3):
        try:
            from sklearn.cluster import KMeans
            class KM:
                def __init__(self): self._model = KMeans(n_clusters=int(k))
                def get_attr(self, n):
                    fns = {
                        "fit":     lambda X: self._model.fit(_unwrap(X)) or self,
                        "predict": lambda X: self._model.predict(_unwrap(X)).tolist(),
                        "centers": lambda: self._model.cluster_centers_.tolist(),
                        "labels":  lambda: self._model.labels_.tolist(),
                    }
                    if n in fns: return PyCallable(fns[n])
                    raise DeltooRuntimeError(f"KMeans has no attribute '{n}'")
            return KM()
        except ImportError:
            raise DeltooRuntimeError("sklearn not installed")

    def _train_test_split(X, y, test_size=0.2, seed=42):
        try:
            from sklearn.model_selection import train_test_split as tts
            Xtr, Xte, ytr, yte = tts(_unwrap(X), _unwrap(y),
                                      test_size=test_size, random_state=int(seed))
            return [Xtr.tolist(), Xte.tolist(), ytr.tolist(), yte.tolist()]
        except ImportError:
            raise DeltooRuntimeError("sklearn not installed")

    ml_proxy = _MlProxy()

    # ── Plot proxy ────────────────────────────────────────────────────────────

    class _PlotProxy:
        def get_attr(self, name):
            fns = {
                "line":    lambda **kw: _plot_line(**kw),
                "scatter": lambda **kw: _plot_scatter(**kw),
                "bar":     lambda **kw: _plot_bar(**kw),
                "hist":    lambda data, bins=10: _plot_hist(data, bins),
                "show":    lambda: _plt().show() or NONE,
                "save":    lambda path="plot.png": _plt().savefig(path) or NONE,
                "clear":   lambda: _plt().clf() or NONE,
                "title":   lambda t: _plt().title(t) or NONE,
                "xlabel":  lambda l: _plt().xlabel(l) or NONE,
                "ylabel":  lambda l: _plt().ylabel(l) or NONE,
            }
            if name in fns:
                return PyCallable(fns[name])
            raise DeltooRuntimeError(f"plot has no attribute '{name}'")

    def _plt():
        try:
            import matplotlib.pyplot as plt
            return plt
        except ImportError:
            raise DeltooRuntimeError("matplotlib not installed. Run: pip install matplotlib")

    def _plot_line(x=None, y=None, title=None, label=None, **kw):
        plt = _plt()
        if x is not None and y is not None:
            plt.plot(_unwrap(x), _unwrap(y), label=label)
        elif y is not None:
            plt.plot(_unwrap(y), label=label)
        if title: plt.title(title)
        if label: plt.legend()
        return NONE

    def _plot_scatter(x, y, title=None, **kw):
        plt = _plt()
        plt.scatter(_unwrap(x), _unwrap(y))
        if title: plt.title(title)
        return NONE

    def _plot_bar(x, y, title=None, **kw):
        plt = _plt()
        plt.bar(_unwrap(x), _unwrap(y))
        if title: plt.title(title)
        return NONE

    def _plot_hist(data, bins=10):
        plt = _plt()
        plt.hist(_unwrap(data), bins=int(bins))
        return NONE

    plot_proxy = _PlotProxy()

    # ── DataFrame proxy ───────────────────────────────────────────────────────

    class DeltooDataFrame:
        def __init__(self, df): self._df = df
        def get_attr(self, name):
            fns = {
                "head":      lambda n=5: DeltooDataFrame(self._df.head(int(n))),
                "tail":      lambda n=5: DeltooDataFrame(self._df.tail(int(n))),
                "describe":  lambda: str(self._df.describe()),
                "corr":      lambda: str(self._df.corr()),
                "shape":     lambda: list(self._df.shape),
                "columns":   lambda: list(self._df.columns),
                "len":       lambda: len(self._df),
                "toList":    lambda: self._df.to_dict("records"),
                "toCSV":     lambda p: self._df.to_csv(p, index=False) or NONE,
                "dropNA":    lambda: DeltooDataFrame(self._df.dropna()),
                "fillNA":    lambda v: DeltooDataFrame(self._df.fillna(v)),
                "groupBy":   lambda col: _GroupBy(self._df.groupby(col)),
                "sortBy":    lambda col, asc=True: DeltooDataFrame(
                    self._df.sort_values(col, ascending=bool(asc))
                ),
                "filter":    lambda col, op, val: _df_filter(self._df, col, op, val),
                "select":    lambda *cols: DeltooDataFrame(self._df[list(cols)]),
                "print":     lambda: println(str(self._df)) or NONE,
            }
            if name in fns:
                return PyCallable(fns[name])
            # Column access
            if name in self._df.columns:
                return list(self._df[name])
            raise DeltooRuntimeError(f"DataFrame has no attribute '{name}'")

    def _df_filter(df, col, op, val):
        ops = {
            ">": df[col] > val,
            "<": df[col] < val,
            ">=": df[col] >= val,
            "<=": df[col] <= val,
            "==": df[col] == val,
            "!=": df[col] != val,
        }
        return DeltooDataFrame(df[ops[op]])

    class _GroupBy:
        def __init__(self, gb): self._gb = gb
        def get_attr(self, name):
            fns = {
                "sum":  lambda: DeltooDataFrame(self._gb.sum().reset_index()),
                "mean": lambda: DeltooDataFrame(self._gb.mean().reset_index()),
                "count": lambda: DeltooDataFrame(self._gb.count().reset_index()),
                "max":  lambda: DeltooDataFrame(self._gb.max().reset_index()),
                "min":  lambda: DeltooDataFrame(self._gb.min().reset_index()),
            }
            if name in fns: return PyCallable(fns[name])
            raise DeltooRuntimeError(f"GroupBy has no attribute '{name}'")

    class _DataFrameProxy:
        def get_attr(self, name):
            fns = {
                "readCSV":  lambda path, **kw: _read_csv(path, **kw),
                "readJSON": lambda path: _read_json(path),
                "from":     lambda data: _from_data(data),
                "fromList": lambda lst: _from_list(lst),
            }
            if name in fns:
                return PyCallable(fns[name])
            raise DeltooRuntimeError(f"DataFrame has no attribute '{name}'")

    def _read_csv(path, **kw):
        try:
            import pandas as pd
            return DeltooDataFrame(pd.read_csv(path, **{k: _unwrap(v) for k, v in kw.items()}))
        except ImportError:
            raise DeltooRuntimeError("pandas not installed. Run: pip install pandas")

    def _read_json(path):
        try:
            import pandas as pd
            return DeltooDataFrame(pd.read_json(path))
        except ImportError:
            raise DeltooRuntimeError("pandas not installed")

    def _from_data(data):
        try:
            import pandas as pd
            return DeltooDataFrame(pd.DataFrame(_unwrap(data)))
        except ImportError:
            raise DeltooRuntimeError("pandas not installed")

    def _from_list(lst):
        try:
            import pandas as pd
            return DeltooDataFrame(pd.DataFrame(lst))
        except ImportError:
            raise DeltooRuntimeError("pandas not installed")

    df_proxy = _DataFrameProxy()

    # ── Tensor proxy ──────────────────────────────────────────────────────────

    class _TensorProxy:
        def get_attr(self, name):
            fns = {
                "zeros":    lambda *shape: _tensor_zeros(shape),
                "ones":     lambda *shape: _tensor_ones(shape),
                "from":     lambda data: _tensor_from(data),
                "linspace": lambda a, b, n: _tensor_linspace(a, b, n),
                "arange":   lambda *args: _tensor_arange(*args),
                "eye":      lambda n: _tensor_eye(n),
                "rand":     lambda *shape: _tensor_rand(shape),
                "fill":     lambda shape, val: _tensor_fill(shape, val),
            }
            if name in fns:
                return PyCallable(fns[name])
            raise DeltooRuntimeError(f"tensor has no attribute '{name}'")

    def _norm_shape(shape):
        """Normalize shape from varargs: (3,) or ([3],) or ([2,3],) → [3] or [2,3]."""
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            return [int(s) for s in shape[0]]
        return [int(s) for s in shape]

    def _tensor_zeros(shape):
        shape = _norm_shape(shape)
        try:
            import numpy as np
            return DeltooTensor(np.zeros(shape))
        except ImportError:
            total = 1
            for s in shape:
                total *= s
            return DeltooTensor([0.0] * total, shape)

    def _tensor_ones(shape):
        shape = _norm_shape(shape)
        try:
            import numpy as np
            return DeltooTensor(np.ones(shape))
        except ImportError:
            total = 1
            for s in shape:
                total *= s
            return DeltooTensor([1.0] * total, shape)

    def _tensor_from(data):
        d = _unwrap(data)
        return DeltooTensor(d if isinstance(d, list) else [d])

    def _tensor_linspace(a, b, n):
        n = int(n)
        if n <= 1:
            return DeltooTensor([float(a)])
        step = (float(b) - float(a)) / (n - 1)
        return DeltooTensor([float(a) + step * i for i in range(n)])

    def _tensor_arange(*args):
        fargs = [float(_unwrap(a)) for a in args]
        if len(fargs) == 1:
            vals = [float(i) for i in range(int(fargs[0]))]
        elif len(fargs) == 2:
            vals = []
            v = fargs[0]
            while v < fargs[1]:
                vals.append(v)
                v += 1.0
        else:
            vals = []
            v = fargs[0]
            step = fargs[2]
            while (step > 0 and v < fargs[1]) or (step < 0 and v > fargs[1]):
                vals.append(v)
                v += step
        return DeltooTensor(vals)

    def _tensor_eye(n):
        n = int(n)
        data = []
        for i in range(n):
            for j in range(n):
                data.append(1.0 if i == j else 0.0)
        return DeltooTensor(data, [n, n])

    def _tensor_rand(shape):
        shape = _norm_shape(shape)
        total = 1
        for s in shape:
            total *= s
        return DeltooTensor([random.random() for _ in range(total)], shape)

    def _tensor_fill(shape, val):
        if isinstance(shape, (list, tuple)):
            sh = [int(s) for s in shape]
        else:
            sh = [int(shape)]
        total = 1
        for s in sh:
            total *= s
        return DeltooTensor([float(val)] * total, sh)

    # Patch DeltooTensor with get_attr that uses interpreter types
    def _tensor_get_attr(self, name):
        data = self._data
        shape = self._shape
        fns = {
            "sum":       lambda: sum(data),
            "mean":      lambda: sum(data) / len(data) if data else 0.0,
            "min":       lambda: min(data) if data else 0.0,
            "max":       lambda: max(data) if data else 0.0,
            "argmax":    lambda: data.index(max(data)) if data else 0,
            "argmin":    lambda: data.index(min(data)) if data else 0,
            "shape":     lambda: list(shape),
            "ndim":      len(shape),
            "size":      len(data),
            "toList":    lambda: list(data),
            "reshape":   lambda *s: DeltooTensor(list(data), _norm_shape(s)),
            "transpose": lambda: self._transpose(),
            "flatten":   lambda: DeltooTensor(list(data), [len(data)]),
            "matmul":    lambda other: self._matmul(other),
            "dot":       lambda other: self._dot(other),
            "abs":       lambda: DeltooTensor([abs(x) for x in data], list(shape)),
            "sqrt":      lambda: DeltooTensor([math.sqrt(x) for x in data], list(shape)),
            "item":      lambda idx=None: data[int(idx)] if idx is not None else data[0],
            "slice":     lambda start=0, end=None: DeltooTensor(data[int(start):int(end) if end is not None else len(data)]),
            "data":      list(data),
        }
        if name in fns:
            v = fns[name]
            if callable(v): return PyCallable(v)
            return v
        raise DeltooRuntimeError(f"tensor has no attribute '{name}'")
    DeltooTensor.get_attr = _tensor_get_attr

    tensor_proxy = _TensorProxy()

    # ── AI proxy ──────────────────────────────────────────────────────────────

    class _AiProxy:
        def get_attr(self, name):
            fns = {
                "complete": lambda prompt, model="claude-sonnet-4-6", **kw: _ai_complete(prompt, model, **kw),
            }
            if name in fns:
                return PyCallable(fns[name])
            raise DeltooRuntimeError(f"ai has no attribute '{name}'")

    def _ai_complete(prompt, model="claude-sonnet-4-6", **kw):
        try:
            import anthropic
            client = anthropic.Anthropic()
            msg = client.messages.create(
                model=model,
                max_tokens=1024,
                messages=[{"role": "user", "content": str(prompt)}]
            )
            class Resp:
                def __init__(self, text): self.fields = {"text": text}
                def get_attr(self, n):
                    if n in self.fields: return self.fields[n]
                    raise DeltooRuntimeError(f"Response has no attribute '{n}'")
            return Resp(msg.content[0].text)
        except ImportError:
            raise DeltooRuntimeError("anthropic not installed. Run: pip install anthropic")
        except Exception as e:
            raise DeltooRuntimeError(f"AI completion failed: {e}")

    ai_proxy = _AiProxy()

    # ── Autodiff proxy (forward-mode dual numbers) ───────────────────────────

    # Patch _DualNumber with get_attr that uses interpreter types
    def _dual_get_attr(self, name):
        if name == "value": return self.val
        if name == "deriv": return self.deriv
        raise DeltooRuntimeError(f"Dual has no attribute '{name}'")
    _DualNumber.get_attr = _dual_get_attr

    def _ad_grad(fn, x):
        from .interpreter import DeltooFunction, BoundMethod
        d = _DualNumber(float(x), 1.0)
        if isinstance(fn, (DeltooFunction, BoundMethod)):
            result = interp.call_function(fn, [d], {}, 0)
        elif callable(fn):
            result = fn(d)
        else:
            # PyCallable wrapper
            result = fn(d)
        if isinstance(result, _DualNumber):
            return result.deriv
        return 0.0

    def _ad_sin(x):
        if isinstance(x, _DualNumber):
            return _DualNumber(math.sin(x.val), math.cos(x.val) * x.deriv)
        return math.sin(float(x))

    def _ad_cos(x):
        if isinstance(x, _DualNumber):
            return _DualNumber(math.cos(x.val), -math.sin(x.val) * x.deriv)
        return math.cos(float(x))

    def _ad_exp(x):
        if isinstance(x, _DualNumber):
            ev = math.exp(x.val)
            return _DualNumber(ev, ev * x.deriv)
        return math.exp(float(x))

    def _ad_log(x):
        if isinstance(x, _DualNumber):
            return _DualNumber(math.log(x.val), x.deriv / x.val)
        return math.log(float(x))

    def _ad_sqrt(x):
        if isinstance(x, _DualNumber):
            sv = math.sqrt(x.val)
            return _DualNumber(sv, x.deriv / (2.0 * sv))
        return math.sqrt(float(x))

    def _ad_pow(x, n):
        n_val = float(n)
        if isinstance(x, _DualNumber):
            return _DualNumber(x.val ** n_val, n_val * x.val ** (n_val - 1) * x.deriv)
        return float(x) ** n_val

    class _AdProxy:
        def get_attr(self, name):
            fns = {
                "dual": lambda val, deriv=0.0: _DualNumber(float(val), float(deriv)),
                "grad": lambda fn, x: _ad_grad(fn, x),
                "value": lambda d: d.val if isinstance(d, _DualNumber) else float(d),
                "deriv": lambda d: d.deriv if isinstance(d, _DualNumber) else 0.0,
                "sin": _ad_sin,
                "cos": _ad_cos,
                "exp": _ad_exp,
                "log": _ad_log,
                "sqrt": _ad_sqrt,
                "pow": _ad_pow,
            }
            if name in fns:
                return PyCallable(fns[name])
            raise DeltooRuntimeError(f"ad has no attribute '{name}'")

    ad_proxy = _AdProxy()

    # ── GPU proxy (OpenCL) ───────────────────────────────────────────────────

    class _GpuProxy:
        def get_attr(self, name):
            fns = {
                "available": lambda: _gpu_available(),
                "devices": lambda: _gpu_devices(),
                "run": lambda kernel, *args: _gpu_run(kernel, *args),
            }
            if name in fns:
                return PyCallable(fns[name])
            raise DeltooRuntimeError(f"gpu has no attribute '{name}'")

    def _gpu_available():
        try:
            import pyopencl
            return True
        except ImportError:
            return False

    def _gpu_devices():
        try:
            import pyopencl as cl
            platforms = cl.get_platforms()
            devices = []
            for p in platforms:
                for d in p.get_devices():
                    devices.append(d.name)
            return devices
        except ImportError:
            return []

    def _gpu_run(kernel_src, *args):
        try:
            import pyopencl as cl
            import numpy as np
            ctx = cl.create_some_context(interactive=False)
            queue = cl.CommandQueue(ctx)
            prg = cl.Program(ctx, str(kernel_src)).build()
            # Convert tensor args to buffers
            buffers = []
            for a in args:
                if isinstance(a, DeltooTensor):
                    arr = np.array(a._data, dtype=np.float32)
                    buf = cl.Buffer(ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=arr)
                    buffers.append((buf, arr))
            return DeltooOk(True)
        except ImportError:
            raise DeltooRuntimeError("pyopencl not installed. Run: pip install pyopencl")
        except Exception as e:
            raise DeltooRuntimeError(f"GPU error: {e}")

    gpu_proxy = _GpuProxy()

    # ── Pipeline proxy ───────────────────────────────────────────────────────

    def _call_wk(fn, args):
        """Call a Wakawaka function or Python callable with given args."""
        from .interpreter import DeltooFunction, BoundMethod
        if isinstance(fn, (DeltooFunction, BoundMethod)):
            return interp.call_function(fn, args, {}, 0)
        return fn(*args)

    class _Pipeline:
        def __init__(self, source, steps=None):
            self._source = source if isinstance(source, list) else list(source) if hasattr(source, '__iter__') else [source]
            self._steps = steps or []

        def __repr__(self):
            return f"<Pipeline {len(self._steps)} steps>"

        def get_attr(self, name):
            fns = {
                "map":     lambda fn: _Pipeline(self._source, self._steps + [("map", fn)]),
                "filter":  lambda fn: _Pipeline(self._source, self._steps + [("filter", fn)]),
                "batch":   lambda n:  _Pipeline(self._source, self._steps + [("batch", int(n))]),
                "flatten": lambda:    _Pipeline(self._source, self._steps + [("flatten",)]),
                "take":    lambda n:  _Pipeline(self._source, self._steps + [("take", int(n))]),
                "skip":    lambda n:  _Pipeline(self._source, self._steps + [("skip", int(n))]),
                "shuffle": lambda:    _Pipeline(self._source, self._steps + [("shuffle",)]),
                "zip":     lambda o:  _Pipeline(self._source, self._steps + [("zip", o)]),
                "collect": lambda:    self._execute(),
                "reduce":  lambda fn, init=None: self._reduce(fn, init),
                "forEach": lambda fn: self._foreach(fn),
                "count":   lambda:    len(self._execute()),
            }
            if name in fns:
                return PyCallable(fns[name])
            raise DeltooRuntimeError(f"Pipeline has no attribute '{name}'")

        def _execute(self):
            data = list(self._source)
            for step in self._steps:
                kind = step[0]
                if kind == "map":
                    fn = step[1]
                    data = [_call_wk(fn, [x]) for x in data]
                elif kind == "filter":
                    fn = step[1]
                    data = [x for x in data if _truthy(_call_wk(fn, [x]))]
                elif kind == "batch":
                    n = step[1]
                    data = [data[i:i+n] for i in range(0, len(data), n)]
                elif kind == "flatten":
                    flat = []
                    for x in data:
                        if isinstance(x, list):
                            flat.extend(x)
                        else:
                            flat.append(x)
                    data = flat
                elif kind == "take":
                    data = data[:step[1]]
                elif kind == "skip":
                    data = data[step[1]:]
                elif kind == "shuffle":
                    random.shuffle(data)
                elif kind == "zip":
                    other = step[1]
                    if isinstance(other, _Pipeline):
                        other = other._execute()
                    elif not isinstance(other, list):
                        other = list(other)
                    data = list(zip(data, other))
            return data

        def _reduce(self, fn, init):
            data = self._execute()
            if init is None:
                if not data:
                    return NONE
                acc = data[0]
                for x in data[1:]:
                    acc = _call_wk(fn, [acc, x])
                return acc
            acc = init
            for x in data:
                acc = _call_wk(fn, [acc, x])
            return acc

        def _foreach(self, fn):
            for x in self._execute():
                _call_wk(fn, [x])
            return NONE

    class _PipelineProxy:
        def get_attr(self, name):
            if name == "from":
                return PyCallable(lambda data: _Pipeline(data))
            raise DeltooRuntimeError(f"pipeline has no attribute '{name}'")

    pipeline_proxy = _PipelineProxy()

    # ── Model proxy (serialization + provenance + ONNX/GGUF) ────────────────

    import struct as _struct

    def _model_save(model_map, path):
        with open(str(path), 'wb') as f:
            f.write(b'WKTM')
            f.write(_struct.pack('<B', 1))  # version
            tensors = {}
            meta = {}
            for k, v in (model_map.items() if hasattr(model_map, 'items') else []):
                if k == "__meta__":
                    meta = v
                elif isinstance(v, DeltooTensor):
                    tensors[k] = v
            f.write(_struct.pack('<I', len(tensors)))
            for name, t in tensors.items():
                data = t._data
                shape = t._shape
                name_bytes = str(name).encode('utf-8')
                f.write(_struct.pack('<I', len(name_bytes)))
                f.write(name_bytes)
                f.write(_struct.pack('<B', len(shape)))
                for s in shape:
                    f.write(_struct.pack('<q', int(s)))
                for val in data:
                    f.write(_struct.pack('<d', float(val)))
            # metadata
            meta_json = json.dumps(meta if isinstance(meta, dict) else {}).encode('utf-8')
            f.write(_struct.pack('<I', len(meta_json)))
            f.write(meta_json)
        return NONE

    def _model_load(path):
        with open(str(path), 'rb') as f:
            magic = f.read(4)
            if magic != b'WKTM':
                raise DeltooRuntimeError("Not a WKTM model file")
            ver = _struct.unpack('<B', f.read(1))[0]
            n_tensors = _struct.unpack('<I', f.read(4))[0]
            result = {}
            for _ in range(n_tensors):
                nlen = _struct.unpack('<I', f.read(4))[0]
                name = f.read(nlen).decode('utf-8')
                ndim = _struct.unpack('<B', f.read(1))[0]
                shape = [_struct.unpack('<q', f.read(8))[0] for _ in range(ndim)]
                total = 1
                for s in shape:
                    total *= s
                data_bytes = f.read(total * 8)
                values = _struct.unpack(f'<{total}d', data_bytes)
                result[name] = DeltooTensor(list(values), shape)
            # metadata
            meta_len_data = f.read(4)
            if meta_len_data and len(meta_len_data) == 4:
                meta_len = _struct.unpack('<I', meta_len_data)[0]
                if meta_len > 0:
                    meta_json = f.read(meta_len).decode('utf-8')
                    result["__meta__"] = json.loads(meta_json)
        return result

    def _model_loadONNX(path):
        try:
            import onnxruntime as ort
            session = ort.InferenceSession(str(path))
            class OnnxModel:
                def __init__(self, sess):
                    self._sess = sess
                def __repr__(self):
                    return f"<OnnxModel inputs={[i.name for i in self._sess.get_inputs()]}>"
                def get_attr(self, name):
                    if name == "infer":
                        return PyCallable(lambda inp: self._infer(inp))
                    if name == "inputs":
                        return [i.name for i in self._sess.get_inputs()]
                    if name == "outputs":
                        return [o.name for o in self._sess.get_outputs()]
                    raise DeltooRuntimeError(f"OnnxModel has no attribute '{name}'")
                def _infer(self, input_tensor):
                    import numpy as np
                    arr = input_tensor._data if isinstance(input_tensor, DeltooTensor) else _unwrap(input_tensor)
                    input_name = self._sess.get_inputs()[0].name
                    result = self._sess.run(None, {input_name: np.array(arr, dtype=np.float32)})
                    return DeltooTensor(np.array(result[0]))
            return OnnxModel(session)
        except ImportError:
            raise DeltooRuntimeError("onnxruntime not installed. Run: pip install onnxruntime")

    def _model_loadGGUF(path):
        with open(str(path), 'rb') as f:
            magic = f.read(4)
            if magic != b'GGUF':
                raise DeltooRuntimeError("Not a GGUF file")
            version = _struct.unpack('<I', f.read(4))[0]
            n_tensors = _struct.unpack('<Q', f.read(8))[0]
            n_kv = _struct.unpack('<Q', f.read(8))[0]
            # Skip metadata KV pairs
            for _ in range(n_kv):
                klen = _struct.unpack('<Q', f.read(8))[0]
                f.read(klen)  # key
                vtype = _struct.unpack('<I', f.read(4))[0]
                _gguf_skip_value(f, vtype)
            # Read tensor infos
            tinfos = []
            for _ in range(n_tensors):
                nlen = _struct.unpack('<Q', f.read(8))[0]
                name = f.read(nlen).decode('utf-8')
                ndim = _struct.unpack('<I', f.read(4))[0]
                shape = [_struct.unpack('<Q', f.read(8))[0] for _ in range(ndim)]
                dtype = _struct.unpack('<I', f.read(4))[0]
                offset = _struct.unpack('<Q', f.read(8))[0]
                tinfos.append((name, ndim, shape, dtype, offset))
            # Align to 32 bytes
            pos = f.tell()
            aligned = ((pos + 31) // 32) * 32
            f.seek(aligned)
            data_start = f.tell()
            result = {}
            for name, ndim, shape, dtype, offset in tinfos:
                total = 1
                for s in shape:
                    total *= s
                f.seek(data_start + offset)
                try:
                    import numpy as np
                    if dtype == 0:  # F32
                        raw = np.frombuffer(f.read(total * 4), dtype=np.float32)
                        result[name] = DeltooTensor(raw.astype(np.float64).reshape(shape))
                    else:
                        result[name] = DeltooTensor(np.zeros(shape, dtype=np.float64))
                except ImportError:
                    result[name] = [0.0] * total
        return result

    def _gguf_skip_value(f, vtype):
        sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
        if vtype in sizes:
            f.read(sizes[vtype])
        elif vtype == 8:
            slen = _struct.unpack('<Q', f.read(8))[0]
            f.read(slen)
        elif vtype == 9:
            atype = _struct.unpack('<I', f.read(4))[0]
            alen = _struct.unpack('<Q', f.read(8))[0]
            for _ in range(alen):
                _gguf_skip_value(f, atype)

    def _model_set_meta(model_map, key, value):
        if "__meta__" not in model_map:
            model_map["__meta__"] = {}
        model_map["__meta__"][str(key)] = value
        return NONE

    class _ModelProxy:
        def get_attr(self, name):
            fns = {
                "save": lambda m, path: _model_save(m, path),
                "load": lambda path: _model_load(path),
                "meta": lambda m, key=None: (m.get("__meta__", {}).get(str(key), NONE) if key is not None else m.get("__meta__", {})) if isinstance(m, dict) else NONE,
                "setMeta": lambda m, k, v: _model_set_meta(m, k, v),
                "loadONNX": lambda path: _model_loadONNX(path),
                "loadGGUF": lambda path: _model_loadGGUF(path),
            }
            if name in fns:
                return PyCallable(fns[name])
            raise DeltooRuntimeError(f"model has no attribute '{name}'")

    model_proxy = _ModelProxy()

    # ── fs module ─────────────────────────────────────────────────────────────

    class _FsProxy:
        def get_attr(self, name):
            fns = {
                "read":   lambda path: open(str(path), "r", encoding="utf-8").read(),
                "write":  lambda path, content: (open(str(path), "w", encoding="utf-8").write(str(content)), NONE)[1],
                "exists": lambda path: os.path.exists(str(path)),
                "args":   lambda: list(sys.argv),
            }
            if name in fns:
                return PyCallable(fns[name])
            raise DeltooRuntimeError(f"fs has no attribute '{name}'")

    fs_proxy = _FsProxy()

    # ── Foreign module proxies (py/jvm/node) ──────────────────────────────────

    class _PyProxy:
        def get_attr(self, name):
            fns = {
                "available": lambda: True,  # We're running in Python!
                "import":    lambda mod_name: __import__(str(mod_name)),
                "call":      lambda mod, method, *a: getattr(mod, str(method))(*a),
                "eval":      lambda expr: eval(str(expr)),
            }
            if name in fns:
                return PyCallable(fns[name])
            raise DeltooRuntimeError(f"py has no attribute '{name}'")

    py_proxy = _PyProxy()

    class _JvmProxy:
        def get_attr(self, name):
            if name == "available":
                return PyCallable(lambda: False)
            raise DeltooRuntimeError(f"jvm has no attribute '{name}' (JVM not available in interpreter mode)")

    jvm_proxy = _JvmProxy()

    class _NodeProxy:
        def get_attr(self, name):
            import shutil
            fns = {
                "available": lambda: shutil.which("node") is not None,
                "require":   lambda mod_name: {"__type__": "node_module", "module": str(mod_name)},
                "eval":      lambda expr: subprocess.check_output(["node", "-e", f"process.stdout.write(String({expr}))"]).decode(),
            }
            if name in fns:
                return PyCallable(fns[name])
            raise DeltooRuntimeError(f"node has no attribute '{name}'")

    node_proxy = _NodeProxy()

    # ── Format helpers ────────────────────────────────────────────────────────

    def format_(val, spec=""):
        if not spec:
            return _dt_str(val)
        return format(val, spec)

    # ── Misc ──────────────────────────────────────────────────────────────────

    def exit_(code=0):
        sys.exit(int(code))

    def panic(msg):
        from .interpreter import _Panic
        raise _Panic(str(msg))

    def assert_(cond, msg="assertion failed"):
        from .interpreter import _Panic
        if not _truthy(cond):
            raise _Panic(str(msg))
        return NONE

    def copy_(val):
        import copy
        return copy.deepcopy(val)

    def id_(val):
        return id(val)

    def hash_(val):
        try:
            return hash(val)
        except TypeError:
            return id(val)

    def repr_(val):
        return repr(val)

    def sleep_(seconds):
        time.sleep(float(seconds))
        return NONE

    # ── Channel helpers ───────────────────────────────────────────────────────

    def chan_(cap=0):
        return DeltooChannel(int(cap))

    def recv_(ch):
        if isinstance(ch, DeltooChannel):
            return ch.recv()
        raise DeltooRuntimeError("recv() expects a channel")

    def send_(ch, val):
        if isinstance(ch, DeltooChannel):
            ch.send(val)
            return NONE
        raise DeltooRuntimeError("send() expects a channel")

    # ── Return all builtins ───────────────────────────────────────────────────

    return {
        # I/O
        "println": println,
        "print":   print_,
        "eprintln": eprintln,
        "readln":  readln,
        "readlines": readlines,

        # Type conversion
        "int":   to_int,
        "float": to_float,
        "str":   to_str,
        "bool":  to_bool,
        "list":  to_list,
        "bytes": to_bytes,
        "chr":   chr_,
        "ord":   ord_,

        # Type checks
        "isInt":   is_int,
        "isFloat": is_float,
        "isStr":   is_str,
        "isBool":  is_bool,
        "isList":  is_list,
        "isMap":   is_map,
        "isNone":  is_none,
        "isSome":  is_some,
        "isOk":    is_ok,
        "isErr":   is_err,
        "typeof":  type_of,

        # Collections
        "len":      len_,
        "range":    range_,
        "zip":      zip_,
        "enumerate": enumerate_,
        "map":      map_,
        "filter":   filter_,
        "reduce":   reduce_,
        "sorted":   sorted_,
        "reversed": reversed_,
        "any":      any_,
        "all":      all_,
        "sum":      sum_,
        "min":      min_,
        "max":      max_,

        # Misc
        "exit":   exit_,
        "panic":  panic,
        "assert": assert_,
        "copy":   copy_,
        "id":     id_,
        "hash":   hash_,
        "repr":   repr_,
        "sleep":  sleep_,
        "format": format_,

        # Channels
        "chan": chan_,
        "recv": recv_,
        "send": send_,

        # Modules
        "math":      math_proxy,
        "stats":     stats_proxy,
        "rand":      rand_proxy,
        "time":      time_proxy,
        "file":      file_proxy,
        "os":        os_proxy,
        "json":      json_proxy,
        "sql":       sql_proxy,
        "http":      http_proxy,
        "regex":     regex_proxy,
        "ml":        ml_proxy,
        "plot":      plot_proxy,
        "DataFrame": df_proxy,
        "tensor":    tensor_proxy,
        "ai":        ai_proxy,
        "csv":       csv_proxy,
        "yaml":      yaml_proxy,
        "ad":        ad_proxy,
        "gpu":       gpu_proxy,
        "pipeline":  pipeline_proxy,
        "model":     model_proxy,
        "fs":        fs_proxy,
        "py":        py_proxy,
        "jvm":       jvm_proxy,
        "node":      node_proxy,
    }
