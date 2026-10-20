"""
run_demo.py
───────────
One-command launcher for the IndexBuddy Live Demo Dashboard.

Starts:
  1. FastAPI Live MySQL + XGBoost Backend on http://127.0.0.1:8000
  2. React + Vite Dev Server on http://localhost:5173

Usage:
    python run_demo.py
"""
from __future__ import annotations

import subprocess
import sys
import time
import webbrowser
from pathlib import Path


ROOT = Path(__file__).resolve().parent
FRONTEND_DIR = ROOT / "frontend"


def main() -> None:
    print("Starting IndexBuddy FastAPI Backend on http://127.0.0.1:8000 ...")
    api_proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "app.api_server:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
        ],
        cwd=str(ROOT),
    )

    npm_cmd = "npm.cmd" if sys.platform.startswith("win") else "npm"
    print("Starting React + Vite Frontend on http://localhost:5173 ...")
    vite_proc = subprocess.Popen(
        [npm_cmd, "run", "dev"],
        cwd=str(FRONTEND_DIR),
    )

    time.sleep(2.5)
    print("\n============================================================")
    print(" IndexBuddy React + Vite Dashboard : http://localhost:5173")
    print(" FastAPI Backend + Built UI        : http://127.0.0.1:8000")
    print(" Press Ctrl+C to stop both servers.")
    print("============================================================\n")
    try:
        webbrowser.open("http://localhost:5173")
    except Exception:
        pass

    try:
        api_proc.wait()
    except KeyboardInterrupt:
        print("\nShutting down servers...")
    finally:
        for p in (vite_proc, api_proc):
            if p.poll() is None:
                p.terminate()


if __name__ == "__main__":
    main()
