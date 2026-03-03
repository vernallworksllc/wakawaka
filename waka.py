#!/usr/bin/env python3
"""
Wakawaka Language — Main CLI
Usage:
  waka run   <file.wk> [args...]    Run a Wakawaka program
  waka build <file.wk> [-o output]  Transpile and compile to native binary
  waka serve [dir] [--port PORT]    Start built-in web server
  waka repl                          Start interactive REPL
  waka fmt   <file.wk>              Format source
  waka check <file.wk>              Type-check without running
  waka version                       Print version
"""
import sys
import os
import argparse

if sys.version_info < (3, 8):
    sys.exit("Wakawaka requires Python 3.8 or newer.")
import traceback

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.parser import parse
from src.interpreter import Interpreter, DeltooRuntimeError, _Panic

VERSION = "1.0.0"
BANNER = f"""
  ██╗    ██╗ █████╗ ██╗  ██╗ █████╗ ██╗    ██╗ █████╗ ██╗  ██╗ █████╗
  ██║    ██║██╔══██╗██║ ██╔╝██╔══██╗██║    ██║██╔══██╗██║ ██╔╝██╔══██╗
  ██║ █╗ ██║███████║█████╔╝ ███████║██║ █╗ ██║███████║█████╔╝ ███████║
  ██║███╗██║██╔══██║██╔═██╗ ██╔══██║██║███╗██║██╔══██║██╔═██╗ ██╔══██║
  ╚███╔███╔╝██║  ██║██║  ██╗██║  ██║╚███╔███╔╝██║  ██║██║  ██╗██║  ██║
   ╚══╝╚══╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚══╝╚══╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝
  Wakawaka v{VERSION} — Fast · Safe · Expressive
"""


def cmd_run(args):
    filepath = args.file
    if filepath == "-":
        # Read source from stdin (pipe support)
        source = sys.stdin.read()
        filepath = "<stdin>"
    else:
        if not os.path.exists(filepath):
            print(f"error: file not found: {filepath}", file=sys.stderr)
            sys.exit(1)
        with open(filepath, "r", encoding="utf-8") as f:
            source = f.read()

    sys.argv = [filepath] + (args.args or [])

    try:
        program = parse(source, filepath)
    except Exception as e:
        print(f"\033[31m{e}\033[0m", file=sys.stderr)
        sys.exit(1)

    interp = Interpreter(filepath, source)
    try:
        interp.run(program)
    except SystemExit as e:
        sys.exit(e.code)
    except _Panic as e:
        print(f"\n\033[31mpanic: {e}\033[0m", file=sys.stderr)
        sys.exit(1)
    except DeltooRuntimeError as e:
        print(f"\n\033[31m{interp.format_error(e)}\033[0m", file=sys.stderr)
        if args.traceback:
            traceback.print_exc()
        sys.exit(1)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        sys.exit(130)


def cmd_build(args):
    """Transpile to C and compile with gcc/clang."""
    filepath = args.file
    if not os.path.exists(filepath):
        print(f"error: file not found: {filepath}", file=sys.stderr)
        sys.exit(1)

    out = args.output or os.path.splitext(filepath)[0]
    print(f"Building {filepath} -> {out}")

    with open(filepath, "r", encoding="utf-8") as f:
        source = f.read()

    try:
        program = parse(source, filepath)
    except Exception as e:
        print(f"\033[31m{e}\033[0m", file=sys.stderr)
        sys.exit(1)

    try:
        from src.transpiler import Transpiler
        transpiler = Transpiler()
        source_dir = os.path.dirname(os.path.abspath(filepath))
        c_code = transpiler.transpile(program, source_dir=source_dir)
        c_file = out + ".c"
        with open(c_file, "w", encoding="utf-8") as f:
            f.write(c_code)
        print(f"Generated C: {c_file}")

        import subprocess
        runtime_dir = os.path.join(os.path.dirname(__file__), "runtime")
        runtime_src = os.path.join(runtime_dir, "waka_runtime.c")
        compiler = "gcc" if _which("gcc") else "clang" if _which("clang") else None
        if compiler:
            # -lm and -lpthread are Linux/macOS only; not needed on Windows
            extra_libs = [] if sys.platform == "win32" else ["-lm", "-lpthread", "-ldl"]
            result = subprocess.run(
                [compiler, c_file, runtime_src, "-o", out,
                 *extra_libs, f"-I{runtime_dir}",
                 "-std=c99", "-O2"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                print(f"\033[32mBuild successful: {out}\033[0m")
            else:
                print(f"\033[31mCompile error:\033[0m\n{result.stderr}", file=sys.stderr)
                sys.exit(1)
        else:
            print(f"No C compiler found. C source saved to {c_file}")
    except ImportError:
        print("Transpiler not yet implemented for this version.")
        print("Use 'waka run' to execute interpreted.")


def cmd_serve(args):
    """Start built-in web server."""
    from src.web_server import WakaWebServer
    root = args.directory or "."
    port = args.port
    host = args.host
    if not os.path.exists(root):
        print(f"error: directory not found: {root}", file=sys.stderr)
        sys.exit(1)
    server = WakaWebServer(root=os.path.abspath(root), host=host, port=port)
    print(f"\033[32mWakawaka web server running at http://{host}:{port}/\033[0m")
    print(f"Serving: {os.path.abspath(root)}")
    print("Press Ctrl+C to stop.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")


def cmd_repl(args):
    print(BANNER)
    print("Type Wakawaka code. Blank line executes. 'exit' or Ctrl+D to quit.\n")

    interp = Interpreter("<repl>")
    buf = []

    while True:
        try:
            prompt = "... " if buf else "wk> "
            line = input(prompt)
        except EOFError:
            print()
            break
        except KeyboardInterrupt:
            buf = []
            print()
            continue

        if line.strip() == "exit":
            break

        if line.strip() == "":
            if not buf:
                continue
            source = "\n".join(buf)
            buf = []
            _repl_eval(interp, source)
        else:
            buf.append(line)


def _repl_eval(interp: Interpreter, source: str):
    try:
        program = parse(source, "<repl>")
    except Exception as e:
        print(f"\033[31m{e}\033[0m")
        return

    try:
        from src.ast_nodes import Program, ExprStmt
        stmts = program.stmts
        if len(stmts) == 1 and isinstance(stmts[0], ExprStmt):
            from src.interpreter import deltoo_str, NONE
            result = interp.eval_expr(stmts[0].expr, interp.global_env)
            if result is not NONE:
                print(f"\033[36m=> {deltoo_str(result)}\033[0m")
        else:
            interp.run(program)
    except SystemExit:
        pass
    except _Panic as e:
        print(f"\033[31mpanic: {e}\033[0m")
    except DeltooRuntimeError as e:
        print(f"\033[31m{e}\033[0m")
    except Exception as e:
        print(f"\033[31mError: {e}\033[0m")


def cmd_check(args):
    filepath = args.file
    if not os.path.exists(filepath):
        print(f"error: file not found: {filepath}", file=sys.stderr)
        sys.exit(1)
    with open(filepath, "r", encoding="utf-8") as f:
        source = f.read()
    try:
        program = parse(source, filepath)
    except Exception as e:
        print(f"\033[31m{filepath}: {e}\033[0m", file=sys.stderr)
        sys.exit(1)

    # Run gradual type checker
    from src.type_checker import TypeChecker
    checker = TypeChecker()
    warnings = checker.check(program)
    if warnings:
        for w in warnings:
            print(f"\033[33m{filepath}:{w}\033[0m")
        print(f"\033[33m{filepath}: {len(warnings)} warning(s)\033[0m")
    else:
        print(f"\033[32m{filepath}: OK\033[0m")


def cmd_fmt(args):
    filepath = args.file
    if not os.path.exists(filepath):
        print(f"error: file not found: {filepath}", file=sys.stderr)
        sys.exit(1)
    with open(filepath, "r", encoding="utf-8") as f:
        source = f.read()
    try:
        from src.formatter import format_source
        formatted = format_source(source, filepath)
    except Exception as e:
        print(f"\033[31m{filepath}: {e}\033[0m", file=sys.stderr)
        sys.exit(1)
    if getattr(args, 'check', False):
        if source != formatted:
            print(f"{filepath}: needs formatting")
            sys.exit(1)
        else:
            print(f"{filepath}: OK")
    else:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(formatted)
        print(f"\033[32mFormatted {filepath}\033[0m")


def cmd_version(args):
    print(f"Wakawaka {VERSION}")
    print(f"Python {sys.version}")


def _which(name):
    import shutil
    return shutil.which(name) is not None


def main():
    parser = argparse.ArgumentParser(
        prog="waka",
        description="Wakawaka language toolchain",
    )
    parser.add_argument("--traceback", action="store_true",
                        help="Show full Python traceback on errors")
    sub = parser.add_subparsers(dest="command")

    # run
    p_run = sub.add_parser("run", help="Run a .wk file")
    p_run.add_argument("file", help="Path to .wk file")
    p_run.add_argument("args", nargs=argparse.REMAINDER, help="Program arguments")
    p_run.set_defaults(func=cmd_run)

    # build
    p_build = sub.add_parser("build", help="Compile a .wk file to native binary")
    p_build.add_argument("file", help="Path to .wk file")
    p_build.add_argument("-o", "--output", help="Output binary name")
    p_build.set_defaults(func=cmd_build)

    # serve
    p_serve = sub.add_parser("serve", help="Start built-in web server")
    p_serve.add_argument("directory", nargs="?", default=".",
                         help="Directory to serve (default: .)")
    p_serve.add_argument("--port", "-p", type=int, default=8080,
                         help="Port to listen on (default: 8080)")
    p_serve.add_argument("--host", default="127.0.0.1",
                         help="Host to bind to (default: 127.0.0.1)")
    p_serve.set_defaults(func=cmd_serve)

    # repl
    p_repl = sub.add_parser("repl", help="Start interactive REPL")
    p_repl.set_defaults(func=cmd_repl)

    # check
    p_check = sub.add_parser("check", help="Type-check a .wk file")
    p_check.add_argument("file")
    p_check.set_defaults(func=cmd_check)

    # fmt
    p_fmt = sub.add_parser("fmt", help="Format a .wk file")
    p_fmt.add_argument("file")
    p_fmt.add_argument("--check", action="store_true",
                        help="Check if file is formatted (exit 1 if not)")
    p_fmt.set_defaults(func=cmd_fmt)

    # version
    p_ver = sub.add_parser("version", help="Print version")
    p_ver.set_defaults(func=cmd_version)

    args = parser.parse_args()

    if not args.command:
        print(BANNER)
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
