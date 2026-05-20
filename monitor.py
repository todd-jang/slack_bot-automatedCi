"""
monitor.py – Backend monitoring & system metrics for the ChatOps Slack Bot.

Collects CPU, memory, disk, GPU (MPS/CUDA) usage and running-process info
so the bot can report real-time health to a Slack channel.
"""

import os
import json
import time
import platform
import subprocess
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List


# ──────────────────────────────────────────────
# Data classes for structured metrics
# ──────────────────────────────────────────────
@dataclass
class CPUMetrics:
    usage_percent: float       # overall CPU usage %
    core_count: int            # logical cores
    load_avg_1m: float         # 1-min load average


@dataclass
class MemoryMetrics:
    total_gb: float
    used_gb: float
    available_gb: float
    usage_percent: float


@dataclass
class DiskMetrics:
    total_gb: float
    used_gb: float
    free_gb: float
    usage_percent: float


@dataclass
class GPUMetrics:
    backend: str               # "mps" | "cuda" | "none"
    device_name: str
    available: bool
    memory_allocated_mb: Optional[float] = None
    memory_reserved_mb: Optional[float] = None


@dataclass
class ProcessInfo:
    name: str
    pid: int
    status: str                # "running" | "stopped" | "zombie" | …
    cpu_percent: float
    memory_mb: float


@dataclass
class SystemSnapshot:
    timestamp: float
    hostname: str
    os_version: str
    python_version: str
    cpu: CPUMetrics
    memory: MemoryMetrics
    disk: DiskMetrics
    gpu: GPUMetrics
    active_processes: List[Dict]

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


# ──────────────────────────────────────────────
# Collectors
# ──────────────────────────────────────────────
def _collect_cpu() -> CPUMetrics:
    """Collect CPU metrics using sysctl / os on macOS."""
    core_count = os.cpu_count() or 1
    load_avg = os.getloadavg()  # (1m, 5m, 15m)
    # Approximate usage from 1-min load average relative to core count
    usage_pct = min(100.0, (load_avg[0] / core_count) * 100)
    return CPUMetrics(
        usage_percent=round(usage_pct, 1),
        core_count=core_count,
        load_avg_1m=round(load_avg[0], 2),
    )


def _collect_memory() -> MemoryMetrics:
    """Parse vm_stat (macOS) for memory info."""
    try:
        out = subprocess.check_output(["vm_stat"], text=True)
        lines = out.strip().split("\n")
        page_size = 16384  # default on Apple Silicon
        stats: Dict[str, int] = {}
        for line in lines[1:]:
            if ":" in line:
                key, val = line.split(":", 1)
                val = val.strip().rstrip(".")
                try:
                    stats[key.strip()] = int(val)
                except ValueError:
                    pass

        pages_free = stats.get("Pages free", 0)
        pages_active = stats.get("Pages active", 0)
        pages_inactive = stats.get("Pages inactive", 0)
        pages_speculative = stats.get("Pages speculative", 0)
        pages_wired = stats.get("Pages wired down", 0)
        pages_compressed = stats.get("Pages occupied by compressor", 0)

        total_pages = pages_free + pages_active + pages_inactive + pages_speculative + pages_wired + pages_compressed
        used_pages = pages_active + pages_wired + pages_compressed
        avail_pages = pages_free + pages_inactive

        total_gb = round((total_pages * page_size) / (1024 ** 3), 2)
        used_gb = round((used_pages * page_size) / (1024 ** 3), 2)
        avail_gb = round((avail_pages * page_size) / (1024 ** 3), 2)
        usage_pct = round((used_gb / total_gb) * 100, 1) if total_gb > 0 else 0
    except Exception:
        total_gb = used_gb = avail_gb = usage_pct = 0.0

    return MemoryMetrics(
        total_gb=total_gb,
        used_gb=used_gb,
        available_gb=avail_gb,
        usage_percent=usage_pct,
    )


def _collect_disk() -> DiskMetrics:
    """Disk usage for the root volume."""
    try:
        st = os.statvfs("/")
        total = st.f_frsize * st.f_blocks
        free = st.f_frsize * st.f_bavail
        used = total - free
        total_gb = round(total / (1024 ** 3), 2)
        used_gb = round(used / (1024 ** 3), 2)
        free_gb = round(free / (1024 ** 3), 2)
        usage_pct = round((used / total) * 100, 1) if total > 0 else 0
    except Exception:
        total_gb = used_gb = free_gb = usage_pct = 0.0

    return DiskMetrics(
        total_gb=total_gb,
        used_gb=used_gb,
        free_gb=free_gb,
        usage_percent=usage_pct,
    )


def _collect_gpu() -> GPUMetrics:
    """Detect GPU backend (MPS / CUDA)."""
    try:
        import torch
        if torch.cuda.is_available():
            return GPUMetrics(
                backend="cuda",
                device_name=torch.cuda.get_device_name(0),
                available=True,
                memory_allocated_mb=round(torch.cuda.memory_allocated(0) / (1024 ** 2), 1),
                memory_reserved_mb=round(torch.cuda.memory_reserved(0) / (1024 ** 2), 1),
            )
        elif torch.backends.mps.is_available():
            return GPUMetrics(
                backend="mps",
                device_name="Apple Silicon GPU (MPS)",
                available=True,
            )
    except ImportError:
        pass

    return GPUMetrics(backend="none", device_name="N/A", available=False)


def _collect_processes(names: Optional[List[str]] = None) -> List[Dict]:
    """Return info about active Python processes (or a filtered list)."""
    procs: List[Dict] = []
    try:
        out = subprocess.check_output(
            ["ps", "-eo", "pid,pcpu,rss,comm"],
            text=True,
        )
        for line in out.strip().split("\n")[1:]:
            parts = line.split()
            if len(parts) < 4:
                continue
            pid, cpu, rss, comm = parts[0], parts[1], parts[2], " ".join(parts[3:])
            # Filter for Python processes or user-specified names
            if "python" in comm.lower() or (names and any(n in comm for n in names)):
                procs.append(
                    asdict(ProcessInfo(
                        name=comm,
                        pid=int(pid),
                        status="running",
                        cpu_percent=float(cpu),
                        memory_mb=round(int(rss) / 1024, 1),
                    ))
                )
    except Exception:
        pass
    return procs


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────
def collect_snapshot() -> SystemSnapshot:
    """Collect a full system snapshot."""
    return SystemSnapshot(
        timestamp=time.time(),
        hostname=platform.node(),
        os_version=platform.platform(),
        python_version=platform.python_version(),
        cpu=_collect_cpu(),
        memory=_collect_memory(),
        disk=_collect_disk(),
        gpu=_collect_gpu(),
        active_processes=_collect_processes(),
    )


def format_slack_message(snap: SystemSnapshot) -> str:
    """Format the snapshot as a rich Slack message (mrkdwn)."""
    c = snap.cpu
    m = snap.memory
    d = snap.disk
    g = snap.gpu

    # Progress bar helper (10 blocks)
    def bar(pct: float) -> str:
        filled = int(pct / 10)
        return "█" * filled + "░" * (10 - filled)

    lines = [
        f"*🖥️ System Monitor — {snap.hostname}*",
        f"_OS: {snap.os_version} | Python {snap.python_version}_",
        "",
        f"*CPU*  `{bar(c.usage_percent)}` {c.usage_percent}%  ({c.core_count} cores, load {c.load_avg_1m})",
        f"*MEM*  `{bar(m.usage_percent)}` {m.usage_percent}%  ({m.used_gb}/{m.total_gb} GB)",
        f"*DISK* `{bar(d.usage_percent)}` {d.usage_percent}%  ({d.used_gb}/{d.total_gb} GB)",
        "",
        f"*GPU*  {g.device_name} ({'✅ available' if g.available else '❌ not available'})",
    ]

    if g.memory_allocated_mb is not None:
        lines.append(f"       Allocated: {g.memory_allocated_mb} MB | Reserved: {g.memory_reserved_mb} MB")

    # Active Python processes
    if snap.active_processes:
        lines.append("")
        lines.append(f"*Active Python Processes ({len(snap.active_processes)})*")
        for p in snap.active_processes[:10]:  # cap at 10
            lines.append(f"  • PID {p['pid']}  CPU {p['cpu_percent']}%  MEM {p['memory_mb']} MB  `{p['name']}`")
    else:
        lines.append("\n_No active Python processes._")

    return "\n".join(lines)


# ──────────────────────────────────────────────
# CLI quick-test
# ──────────────────────────────────────────────
if __name__ == "__main__":
    snapshot = collect_snapshot()
    print(format_slack_message(snapshot))
