"""
start.py — One-click launcher
Run: python start.py
Starts Flask backend (port 5000) + Next.js frontend (port 3000) together.
Ctrl+C kills both.
"""

import subprocess
import sys
import os
import time
import signal
import threading
import shutil

ROOT = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(ROOT, "backend")
FRONTEND_DIR = os.path.join(ROOT, "frontend")

# ── colour helpers ──────────────────────────────────────────────
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

def log(tag, colour, msg):
    print(f"{colour}{BOLD}[{tag}]{RESET} {msg}")

# ── stream subprocess output with a coloured prefix ─────────────
def stream(proc, tag, colour):
    for line in iter(proc.stdout.readline, b""):
        text = line.decode("utf-8", errors="replace").rstrip()
        if text:
            print(f"{colour}[{tag}]{RESET} {text}")

# ── check required tools ────────────────────────────────────────
def check_tools():
    ok = True

    # Python packages
    try:
        import flask, deepface, mtcnn, pymongo
    except ImportError as e:
        log("SETUP", RED, f"Missing Python package: {e.name}")
        log("SETUP", YELLOW, "Run:  pip install -r backend/requirements.txt")
        ok = False

    # Node / npm
    if not shutil.which("node"):
        log("SETUP", RED, "Node.js not found. Install from https://nodejs.org")
        ok = False

    if not shutil.which("npm"):
        log("SETUP", RED, "npm not found. Install Node.js from https://nodejs.org")
        ok = False

    return ok

# ── install frontend deps if node_modules missing ───────────────
def ensure_frontend_deps():
    nm = os.path.join(FRONTEND_DIR, "node_modules")
    if not os.path.exists(nm):
        log("SETUP", YELLOW, "node_modules missing — running npm install...")
        result = subprocess.run(
            ["npm", "install"],
            cwd=FRONTEND_DIR,
            shell=(sys.platform == "win32")
        )
        if result.returncode != 0:
            log("SETUP", RED, "npm install failed. Fix errors above then retry.")
            sys.exit(1)
        log("SETUP", GREEN, "npm install complete")

# ── main ────────────────────────────────────────────────────────
def main():
    print(f"\n{BOLD}{CYAN}{'='*50}")
    print("   Attendance System — One Click Launcher")
    print(f"{'='*50}{RESET}\n")

    # preflight
    if not check_tools():
        sys.exit(1)

    ensure_frontend_deps()

    processes = []

    # ── start backend ──────────────────────────────────────────
    log("BACKEND", GREEN, "Starting Flask on http://localhost:5000 ...")

    python_exe = "python3.12"          # same Python that ran this script
    backend_env = os.environ.copy()
    backend_env["PYTHONUNBUFFERED"] = "1"

    backend_proc = subprocess.Popen(
        [python_exe, "app.py"],
        cwd=BACKEND_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=backend_env
    )
    processes.append(backend_proc)
    threading.Thread(target=stream, args=(backend_proc, "BACKEND", GREEN), daemon=True).start()

    # give backend a moment to boot before frontend hits it
    time.sleep(3)

    # ── start frontend ─────────────────────────────────────────
    log("FRONTEND", CYAN, "Starting Next.js on http://localhost:3000 ...")

    npm_cmd = ["npm", "run", "dev"]
    frontend_proc = subprocess.Popen(
        npm_cmd,
        cwd=FRONTEND_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        shell=(sys.platform == "win32")
    )
    processes.append(frontend_proc)
    threading.Thread(target=stream, args=(frontend_proc, "FRONTEND", CYAN), daemon=True).start()

    print(f"\n{BOLD}{GREEN}Both servers running!{RESET}")
    print(f"  Frontend  →  {CYAN}http://localhost:3000{RESET}")
    print(f"  Backend   →  {GREEN}http://localhost:5000{RESET}")
    print(f"\n  {YELLOW}Press Ctrl+C to stop both{RESET}\n")

    # ── wait / handle Ctrl+C ───────────────────────────────────
    def shutdown(sig=None, frame=None):
        print(f"\n{YELLOW}Shutting down...{RESET}")
        for p in processes:
            try:
                if sys.platform == "win32":
                    p.terminate()
                else:
                    os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            except Exception:
                p.terminate()
        log("DONE", YELLOW, "All processes stopped.")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # keep main thread alive; exit if either process dies unexpectedly
    while True:
        time.sleep(1)
        for p in processes:
            if p.poll() is not None:
                log("ERROR", RED, f"A process exited unexpectedly (code {p.returncode}). Shutting down.")
                shutdown()

if __name__ == "__main__":
    main()
