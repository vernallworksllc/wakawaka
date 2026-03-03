#!/usr/bin/env python3
"""
Bootstrap builder for Wakawaka.
Produces a single standalone 'waka' binary using PyInstaller.

Usage:
    python build_waka.py

Output:
    dist/waka        (Linux/macOS)
    dist/waka.exe    (Windows)

After building, copy dist/waka (or dist/waka.exe) to a directory on
your PATH and use it directly:

    waka run hello.wk
    waka repl
    waka serve www/
"""
import os
import sys
import shutil
import subprocess

DIST = os.path.join(os.path.dirname(__file__), "dist")


def check_pyinstaller():
    if shutil.which("pyinstaller") is None:
        print("PyInstaller not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])


def build():
    check_pyinstaller()

    root = os.path.dirname(os.path.abspath(__file__))
    entry = os.path.join(root, "waka.py")

    cmd = [
        "pyinstaller",
        "--onefile",
        "--name", "waka",
        "--distpath", DIST,
        "--workpath", os.path.join(root, "build", "_pyinstaller"),
        "--specpath", os.path.join(root, "build"),
        # Include the src package
        "--add-data", f"{os.path.join(root, 'src')}{os.pathsep}src",
        # Include example files
        "--add-data", f"{os.path.join(root, 'examples')}{os.pathsep}examples",
        "--hidden-import", "src.lexer",
        "--hidden-import", "src.parser",
        "--hidden-import", "src.interpreter",
        "--hidden-import", "src.builtins",
        "--hidden-import", "src.ast_nodes",
        "--hidden-import", "src.web_server",
        entry,
    ]

    print("Building waka binary...")
    print(" ".join(cmd))
    result = subprocess.run(cmd, cwd=root)

    if result.returncode != 0:
        print("\n\033[31mBuild failed.\033[0m")
        sys.exit(1)

    binary = os.path.join(DIST, "waka.exe" if sys.platform == "win32" else "waka")
    print(f"\n\033[32mBuild successful: {binary}\033[0m")
    print("\nTo install system-wide:")
    if sys.platform == "win32":
        print(f"  copy {binary} to a directory in your PATH")
    else:
        print(f"  sudo cp {binary} /usr/local/bin/waka")
        print(f"  sudo chmod +x /usr/local/bin/waka")
    print("\nTest with:")
    print("  waka version")
    print("  waka run examples/hello.wk")


if __name__ == "__main__":
    build()
