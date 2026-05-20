"""
dashboard_server.py – Lightweight Flask API that powers the PM2‑style web dashboard.

Endpoints:
  GET  /                    → Serve the dashboard HTML
  GET  /api/processes       → List managed processes (training, simulator, bot)
  GET  /api/system          → System metrics (CPU, MEM, DISK, GPU)
  GET  /api/metrics         → Training metrics history
  GET  /api/logs/<name>     → Last N log lines for a process
  POST /api/process/<name>  → Start / stop / restart a process
"""

import os
import sys
import json
import time
import signal
import threading
import subprocess
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

# Reuse the monitor module we already built
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from monitor import collect_snapshot

app = Flask(__name__, static_folder="static")
CORS(app)

# ──────────────────────────────────────────────
# Process Manager
# ──────────────────────────────────────────────
@dataclass
class ManagedProcess:
    name: str
    script: str
    args: List[str]
    pid: Optional[int] = None
    status: str = "stopped"          # online | stopped | errored
    cpu: float = 0.0
    memory_mb: float = 0.0
    uptime: float = 0.0
    restarts: int = 0
    created_at: float = 0.0
    _proc: Optional[subprocess.Popen] = None
    _log_lines: Optional[list] = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "script": self.script,
            "pid": self.pid,
            "status": self.status,
            "cpu": self.cpu,
            "memory_mb": self.memory_mb,
            "uptime": round(time.time() - self.created_at, 1) if self.status == "online" else 0,
            "restarts": self.restarts,
        }


class ProcessManager:
    """Manage subprocesses like PM2."""

    def __init__(self):
        self._processes: Dict[str, ManagedProcess] = {}
        self._logs: Dict[str, list] = {}
        self._lock = threading.Lock()

        # Register default processes
        base = str(Path(__file__).resolve().parent.parent)
        self._register("bot", os.path.join(base, "bot.py"), [])
        self._register("training", os.path.join(base, "train_fsd.py"), [])
        self._register("inference", os.path.join(base, "inference_fsd.py"), ["--samples", "5"])
        self._register("simulator", os.path.join(base, "run_controller.py"), [])

    def _register(self, name: str, script: str, args: list):
        self._processes[name] = ManagedProcess(
            name=name, script=script, args=args, _log_lines=[]
        )
        self._logs[name] = []

    def _stream_logs(self, name: str, proc: subprocess.Popen):
        """Background thread: read stdout and store log lines."""
        try:
            for line in iter(proc.stdout.readline, ""):
                line = line.rstrip("\n")
                if line:
                    with self._lock:
                        self._logs[name].append(
                            {"ts": time.time(), "line": line}
                        )
                        # Cap at 500 lines
                        if len(self._logs[name]) > 500:
                            self._logs[name] = self._logs[name][-500:]
        except Exception:
            pass

    def start(self, name: str) -> dict:
        with self._lock:
            mp = self._processes.get(name)
            if not mp:
                return {"error": f"Unknown process: {name}"}
            if mp.status == "online" and mp._proc and mp._proc.poll() is None:
                return {"error": f"{name} is already running"}

            try:
                proc = subprocess.Popen(
                    [sys.executable, mp.script] + mp.args,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                mp._proc = proc
                mp.pid = proc.pid
                mp.status = "online"
                mp.created_at = time.time()
                self._logs[name] = []

                t = threading.Thread(target=self._stream_logs, args=(name, proc), daemon=True)
                t.start()

                # Monitor thread to detect process exit
                def _monitor():
                    proc.wait()
                    with self._lock:
                        mp.status = "stopped" if proc.returncode == 0 else "errored"

                threading.Thread(target=_monitor, daemon=True).start()

                return {"ok": True, "pid": proc.pid}
            except Exception as e:
                mp.status = "errored"
                return {"error": str(e)}

    def stop(self, name: str) -> dict:
        with self._lock:
            mp = self._processes.get(name)
            if not mp:
                return {"error": f"Unknown process: {name}"}
            if mp.status != "online" or not mp._proc:
                return {"error": f"{name} is not running"}
            try:
                mp._proc.terminate()
                mp._proc.wait(timeout=10)
            except Exception:
                mp._proc.kill()
            mp.status = "stopped"
            return {"ok": True}

    def restart(self, name: str) -> dict:
        self.stop(name)
        time.sleep(0.5)
        mp = self._processes.get(name)
        if mp:
            mp.restarts += 1
        return self.start(name)

    def list_all(self) -> list:
        with self._lock:
            # Update CPU/memory from /proc or ps
            self._refresh_stats()
            return [mp.to_dict() for mp in self._processes.values()]

    def get_logs(self, name: str, n: int = 50) -> list:
        with self._lock:
            return self._logs.get(name, [])[-n:]

    def _refresh_stats(self):
        """Lightweight stat refresh using ps."""
        try:
            out = subprocess.check_output(
                ["ps", "-eo", "pid,pcpu,rss"], text=True
            )
            pid_map = {}
            for line in out.strip().split("\n")[1:]:
                parts = line.split()
                if len(parts) >= 3:
                    pid_map[int(parts[0])] = (float(parts[1]), int(parts[2]))

            for mp in self._processes.values():
                if mp.pid and mp.pid in pid_map:
                    mp.cpu, rss = pid_map[mp.pid]
                    mp.memory_mb = round(rss / 1024, 1)
                else:
                    mp.cpu = 0.0
                    mp.memory_mb = 0.0
        except Exception:
            pass


pm = ProcessManager()


# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/processes")
def api_processes():
    return jsonify(pm.list_all())


@app.route("/api/system")
def api_system():
    snap = collect_snapshot()
    return jsonify(snap.to_dict())


@app.route("/api/metrics")
def api_metrics():
    log_path = os.path.join(
        str(Path(__file__).resolve().parent.parent), "models", "training_log.json"
    )
    if os.path.exists(log_path):
        with open(log_path) as f:
            return jsonify(json.load(f))
    return jsonify([])


@app.route("/api/logs/<name>")
def api_logs(name):
    n = request.args.get("n", 100, type=int)
    return jsonify(pm.get_logs(name, n))


@app.route("/api/process/<name>", methods=["POST"])
def api_process_action(name):
    action = request.json.get("action", "start") if request.is_json else "start"
    if action == "start":
        return jsonify(pm.start(name))
    elif action == "stop":
        return jsonify(pm.stop(name))
    elif action == "restart":
        return jsonify(pm.restart(name))
    return jsonify({"error": f"Unknown action: {action}"}), 400


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.getenv("DASHBOARD_PORT", 9615))
    print(f"🖥️  PM2‑style Dashboard running at http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
