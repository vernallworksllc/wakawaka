"""
Wakawaka Built-in Web Server
============================
Serves static files and executes .wk files as dynamic pages.

Template syntax (mixed HTML + Wakawaka):
  <?wk  code  ?>    — execute Wakawaka code block
  <?=   expr  ?>    — evaluate and print expression
  Plain HTML passes through unchanged.

PHP-style web globals available in .wk pages:
  request.method   GET / POST
  request.path     URL path
  request.GET      map of query params
  request.POST     map of form/JSON body params
  request.headers  map of request headers
  request.body     raw body string
  $_GET / $_POST / $_SERVER / $_COOKIE   (PHP-compat aliases)

Output functions:
  echo(str)        write to response body
  header(name, value)  add response header
  status(code)     set HTTP status code (default 200)
  die(msg?)        stop execution (optionally write msg)

Security:
  - Path traversal prevention (no ../ escapes)
  - Directory listing disabled
  - No pyblock in web mode
  - Request data is never eval'd, only passed as Wakawaka values
"""

import os
import re
import sys
import mimetypes
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import StringIO

# Add parent dir to path so src imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.parser import parse
from src.interpreter import Interpreter, DeltooRuntimeError, _Panic, NONE, DeltooNone
from src.interpreter import deltoo_str

# Compiled template splitter: splits on <?wk ... ?> and <?= ... ?>
_TMPL_RE = re.compile(r'(<\?=\s*)(.*?)(\s*\?>)|(<\?wk\s*)(.*?)(\s*\?>)', re.DOTALL)

# Allowed static file extensions (whitelist)
_STATIC_EXTS = {
    '.html', '.htm', '.css', '.js', '.json', '.txt', '.xml',
    '.png', '.jpg', '.jpeg', '.gif', '.svg', '.ico', '.webp',
    '.woff', '.woff2', '.ttf', '.eot', '.otf',
    '.pdf', '.zip',  # read-only downloads
}

# Extensions that should never be served as raw source
_DENY_EXTS = {'.py', '.wk', '.pyc', '.env', '.cfg', '.ini', '.key', '.pem'}


def _safe_path(root: str, url_path: str) -> str | None:
    """
    Resolve URL path to filesystem path safely.
    Returns None if path traversal is detected.
    """
    # Strip leading slash, decode %xx
    rel = urllib.parse.unquote(url_path.lstrip('/'))
    # Normalise both root and target — handles Windows backslash differences
    root_norm  = os.path.normcase(os.path.abspath(root))
    joined     = os.path.normcase(os.path.abspath(os.path.join(root, rel)))
    # Must stay inside root
    if not (joined == root_norm or joined.startswith(root_norm + os.sep)):
        return None  # traversal attempt
    # Return non-normcase version for actual filesystem use
    return os.path.abspath(os.path.join(root, rel))


def _parse_query_string(qs: str) -> dict:
    result = {}
    for k, v in urllib.parse.parse_qsl(qs, keep_blank_values=True):
        result[k] = v
    return result


def _parse_template(html: str):
    """
    Parse an HTML template into a list of segments:
      ('literal', text)
      ('expr',    code)   — <?= expr ?>
      ('code',    code)   — <?wk code ?>
    """
    segments = []
    last = 0
    for m in _TMPL_RE.finditer(html):
        if m.start() > last:
            segments.append(('literal', html[last:m.start()]))
        if m.group(1):  # <?= expr ?>
            segments.append(('expr', m.group(2).strip()))
        else:           # <?wk code ?>
            segments.append(('code', m.group(5).strip()))
        last = m.end()
    if last < len(html):
        segments.append(('literal', html[last:]))
    return segments


def _run_wk_page(filepath: str, request_info: dict) -> tuple[int, dict, str]:
    """
    Execute a .wk file as a web page.
    Returns (status_code, headers_dict, body_str).
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        source = f.read()

    output_buf = StringIO()
    response_headers = {}
    response_status = [200]

    # Determine if this is a template (contains <?wk or <?=) or a pure script
    is_template = bool(_TMPL_RE.search(source))

    # Build the interpreter with web context
    interp = Interpreter(filepath, source if not is_template else "")
    interp._web_mode = True  # disables pyblock

    # Inject web globals into the interpreter environment
    env = interp.global_env

    # echo() — writes to output buffer
    def _echo(*args):
        for a in args:
            output_buf.write(deltoo_str(a))
        return NONE

    # header() — set response header
    def _header(name, value=NONE):
        response_headers[deltoo_str(name)] = deltoo_str(value)
        return NONE

    # status() — set HTTP status code
    def _status(code):
        response_status[0] = int(code) if not isinstance(code, DeltooNone) else 200
        return NONE

    # die() — stop execution
    class _Die(Exception):
        def __init__(self, msg=""): self.msg = msg

    def _die(msg=NONE):
        raise _Die(deltoo_str(msg) if not isinstance(msg, DeltooNone) else "")

    env.define("echo", _echo)
    env.define("header", _header)
    env.define("status", _status)
    env.define("die", _die)

    # Build request object
    GET  = request_info.get("GET", {})
    POST = request_info.get("POST", {})
    headers = request_info.get("headers", {})
    cookie_str = headers.get("Cookie", "")
    cookies = {}
    for part in cookie_str.split(";"):
        if "=" in part:
            k, _, v = part.strip().partition("=")
            cookies[k.strip()] = v.strip()

    class RequestProxy:
        def get_attr(self, name):
            data = {
                "method":  request_info.get("method", "GET"),
                "path":    request_info.get("path", "/"),
                "GET":     GET,
                "POST":    POST,
                "headers": headers,
                "body":    request_info.get("body", ""),
                "cookies": cookies,
            }
            if name in data:
                return data[name]
            raise DeltooRuntimeError(f"request has no attribute '{name}'")

    env.define("request", RequestProxy())
    env.define("$_GET",    GET)
    env.define("$_POST",   POST)
    env.define("$_COOKIE", cookies)
    env.define("$_SERVER", {
        "REQUEST_METHOD":  request_info.get("method", "GET"),
        "REQUEST_URI":     request_info.get("path", "/"),
        "SERVER_SOFTWARE": "Wakawaka/1.0.0",
        "REMOTE_ADDR":     request_info.get("remote_addr", ""),
    })

    try:
        if is_template:
            # Template mode: compile entire template to a single Wakawaka script.
            # This lets control-flow (for/if) span across HTML literal segments,
            # exactly like PHP — <?wk for i in 1..=n { ?> ... <?wk } ?> works.
            segments = _parse_template(source)
            script_lines = []
            for kind, content in segments:
                if kind == 'literal':
                    if content:
                        # Escape for string literal: backslash, double-quote, newline
                        esc = (content
                               .replace('\\', '\\\\')
                               .replace('"', '\\"')
                               .replace('\n', '\\n')
                               .replace('\r', ''))
                        script_lines.append(f'echo("{esc}");')
                elif kind == 'expr':
                    script_lines.append(f'echo({content.strip()});')
                elif kind == 'code':
                    script_lines.append(content.strip())
            combined = '\n'.join(script_lines)
            prog = parse(combined, filepath)
            interp.run(prog)
        else:
            # Pure script mode: redirect println/print to output buffer
            import builtins as _builtins_mod
            orig_print = _builtins_mod.print

            def _web_print(*args, sep=" ", end="\n", file=None, flush=False):
                if file is None or file is sys.stdout:
                    output_buf.write(sep.join(str(a) for a in args) + end)
                else:
                    orig_print(*args, sep=sep, end=end, file=file)

            # Patch the builtins println/print used by the interpreter
            old_println = env.vars.get("println")
            old_print   = env.vars.get("print")

            def _wk_println(*args):
                output_buf.write(" ".join(deltoo_str(a) for a in args) + "\n")
                return NONE

            def _wk_print(*args):
                output_buf.write(" ".join(deltoo_str(a) for a in args))
                return NONE

            env.define("println", _wk_println)
            env.define("print",   _wk_print)

            prog = parse(source, filepath)
            interp.run(prog)

    except _Die as d:
        output_buf.write(d.msg)
    except SystemExit:
        pass
    except (_Panic, DeltooRuntimeError) as e:
        response_status[0] = 500
        output_buf.write(f"<pre>Wakawaka Error: {e}</pre>")

    body = output_buf.getvalue()
    if "Content-Type" not in response_headers:
        response_headers["Content-Type"] = "text/html; charset=utf-8"

    return response_status[0], response_headers, body


class WakaRequestHandler(BaseHTTPRequestHandler):
    root: str = "."
    log_requests: bool = True

    def log_message(self, fmt, *args):
        if self.log_requests:
            method = self.command
            path = self.path.split("?")[0]
            code = args[1] if len(args) > 1 else "-"
            print(f"  {method} {path} -> {code}")

    def _send_response(self, code: int, headers: dict, body):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        for k, v in headers.items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _handle_request(self, method: str):
        parsed = urllib.parse.urlparse(self.path)
        url_path = parsed.path
        query_string = parsed.query

        # Security: resolve path safely
        fspath = _safe_path(self.root, url_path)
        if fspath is None:
            self._send_response(403, {"Content-Type": "text/plain"}, "403 Forbidden")
            return

        # Directory → look for index.wk or index.html
        if os.path.isdir(fspath):
            for idx in ("index.wk", "index.html", "index.htm"):
                candidate = os.path.join(fspath, idx)
                if os.path.exists(candidate):
                    fspath = candidate
                    break
            else:
                # No index — directory listing disabled for security
                self._send_response(403, {"Content-Type": "text/plain"},
                                    "403 Directory listing disabled")
                return

        if not os.path.exists(fspath):
            self._send_response(404, {"Content-Type": "text/plain"}, "404 Not Found")
            return

        ext = os.path.splitext(fspath)[1].lower()

        # .wk file — execute as dynamic page (before deny-list check)
        if ext == ".wk":
            # Parse request body for POST
            GET_params  = _parse_query_string(query_string)
            POST_params = {}
            body_raw = ""
            content_length = int(self.headers.get("Content-Length", 0))
            if content_length > 0:
                body_raw = self.rfile.read(content_length).decode("utf-8", errors="replace")
                ct = self.headers.get("Content-Type", "")
                if "application/x-www-form-urlencoded" in ct:
                    POST_params = _parse_query_string(body_raw)
                elif "application/json" in ct:
                    import json
                    try:
                        POST_params = json.loads(body_raw)
                    except Exception:
                        POST_params = {}

            request_info = {
                "method":      method,
                "path":        url_path,
                "GET":         GET_params,
                "POST":        POST_params,
                "headers":     dict(self.headers),
                "body":        body_raw,
                "remote_addr": self.client_address[0],
            }

            try:
                status, headers, body = _run_wk_page(fspath, request_info)
            except Exception as e:
                self._send_response(500, {"Content-Type": "text/plain"},
                                    f"500 Internal Server Error\n{e}")
                return

            self._send_response(status, headers, body)
            return

        # Static file — only serve whitelisted extensions
        if ext not in _STATIC_EXTS:
            self._send_response(403, {"Content-Type": "text/plain"}, "403 Forbidden")
            return

        mime = mimetypes.guess_type(fspath)[0] or "application/octet-stream"
        try:
            with open(fspath, "rb") as f:
                data = f.read()
            self._send_response(200, {"Content-Type": mime}, data)
        except OSError:
            self._send_response(500, {"Content-Type": "text/plain"}, "500 Read Error")

    def do_GET(self):
        self._handle_request("GET")

    def do_POST(self):
        self._handle_request("POST")

    def do_HEAD(self):
        # HEAD: same as GET but no body
        parsed = urllib.parse.urlparse(self.path)
        fspath = _safe_path(self.root, parsed.path)
        if fspath and os.path.exists(fspath):
            self.send_response(200)
            self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()


class WakaWebServer:
    def __init__(self, root: str, host: str = "127.0.0.1", port: int = 8080):
        self.root = os.path.abspath(root)
        self.host = host
        self.port = port

        # Inject root into handler class
        handler = type("_Handler", (WakaRequestHandler,), {"root": self.root})
        self._server = HTTPServer((host, port), handler)

    def serve_forever(self):
        self._server.serve_forever()

    def shutdown(self):
        self._server.shutdown()
