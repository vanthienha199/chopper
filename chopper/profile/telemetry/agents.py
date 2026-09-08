"""Per-agent CPU attribution.

When several agents (or sub-agents) run at once, per-core utilization alone
cannot say WHICH agent is stressing the system. This collector attributes CPU
usage to named agents by process tree: an agent is every process descended
from a registered root pid, or any process that self-tags with the
CHOPPER_AGENT_ID environment variable. Each sample records per-process CPU
usage under its agent's name, stamped with the shared clock, so the per-agent
load lands on the same timeline as the GPU kernel traces.

Register agents either way:
  - explicitly: agent_pids={"agent1": 1234, "agent2": 5678}
  - self-tagged: launch the agent with CHOPPER_AGENT_ID=agent1 in its env

Standalone:
    python -m chopper.profile.telemetry.agents --pid main=1234 --duration 10
"""

import os
from time import monotonic_ns, sleep
from typing import Any

import pandas as pd
import psutil
from loguru import logger

_ENV_TAG = "CHOPPER_AGENT_ID"


def _discover_tagged_roots(env_cache: dict[int, str | None]) -> dict[int, str]:
    """Find processes that self-tag with CHOPPER_AGENT_ID. Caches per-pid
    environ reads (they are expensive and often AccessDenied for other users).
    Returns {pid: agent_name} for tagged processes."""
    roots: dict[int, str] = {}
    for p in psutil.process_iter(["pid"]):
        pid = p.info["pid"]
        if pid not in env_cache:
            try:
                env_cache[pid] = p.environ().get(_ENV_TAG)
            except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
                env_cache[pid] = None
        tag = env_cache[pid]
        if tag:
            roots[pid] = tag
    return roots


def _attribute(pid: int, roots: dict[int, str], parent_of: dict[int, int]) -> str | None:
    """Walk the ppid chain until a registered agent root (or give up).
    The NEAREST tagged ancestor wins, so a sub-agent tagged e.g.
    CHOPPER_AGENT_ID=agentA.sub inside agentA's tree is attributed to the
    sub-agent, not the parent."""
    seen: set[int] = set()
    while pid and pid not in seen:
        if pid in roots:
            return roots[pid]
        seen.add(pid)
        pid = parent_of.get(pid, 0)
    return None


def _cpu_topology() -> dict[int, dict[str, int]]:
    """Map logical cpu -> {core: physical core id, package: socket id} from
    sysfs. SMT siblings share (package, core). Empty on non-Linux."""
    topo: dict[int, dict[str, int]] = {}
    base = "/sys/devices/system/cpu"
    try:
        for entry in os.listdir(base):
            if not (entry.startswith("cpu") and entry[3:].isdigit()):
                continue
            cpu = int(entry[3:])
            try:
                with open(f"{base}/{entry}/topology/core_id") as f:
                    core = int(f.read())
                with open(f"{base}/{entry}/topology/physical_package_id") as f:
                    pkg = int(f.read())
            except OSError:
                continue
            topo[cpu] = {"core": core, "package": pkg}
    except OSError:
        pass
    return topo


def main(
    stop: Any,
    filename: str = "agents.pkl",
    outdir: str = ".",
    agent_pids: dict[str, int] | None = None,
    cpu_clock: str = "monotonic",
    off: float = 0.2,
    **kwargs: Any,
) -> None:
    """Background per-agent CPU sampler (collector contract).

    agent_pids maps agent name -> root pid. Processes tagged with
    CHOPPER_AGENT_ID are picked up automatically as additional roots.
    """
    from chopper.profile.telemetry.cpu import _resolve_clock
    clock, clock_domain = _resolve_clock(cpu_clock)

    explicit_roots = {int(pid): name for name, pid in (agent_pids or {}).items()}
    env_cache: dict[int, str | None] = {}
    primed: set[int] = set()
    results: list[dict[str, Any]] = []

    logger.info(f"agent attribution: explicit roots {explicit_roots} (clock={clock_domain})")

    while not stop.value:
        roots = dict(explicit_roots)
        roots.update(_discover_tagged_roots(env_cache))
        if not roots:
            sleep(off)
            continue

        sweep_t0 = monotonic_ns()
        parent_of: dict[int, int] = {}
        procs: dict[int, Any] = {}
        for p in psutil.process_iter(["pid", "ppid", "name"]):
            parent_of[p.info["pid"]] = p.info["ppid"]
            procs[p.info["pid"]] = p

        ts = clock()
        sweep_rows: list[dict[str, Any]] = []
        for pid, p in procs.items():
            agent = _attribute(pid, roots, parent_of)
            if agent is None:
                continue
            try:
                pct = p.cpu_percent(interval=None)  # since last call
                cpu_num = p.cpu_num() if hasattr(p, "cpu_num") else None
            except (psutil.NoSuchProcess, psutil.ZombieProcess, psutil.AccessDenied):
                continue
            if pid not in primed:
                primed.add(pid)  # first cpu_percent call is a meaningless 0
                continue
            sweep_rows.append({
                "ts": ts, "agent": agent, "pid": pid,
                "name": p.info["name"], "cpu": cpu_num, "percent": pct,
            })
        # sweep cost = the sampling-rate floor; ms on a quiet compute node,
        # can approach 1s on a busy login node with thousands of processes
        sweep_s = (monotonic_ns() - sweep_t0) / 1e9
        for r in sweep_rows:
            r["sweep_s"] = sweep_s
        results.extend(sweep_rows)
        sleep(off)

    os.makedirs(outdir, exist_ok=True)
    df = pd.DataFrame(results)
    df["node"] = __import__("socket").gethostname()  # multi-node: rows carry their host
    df.attrs["clock_domain"] = clock_domain
    df.attrs["cpu_topology"] = _cpu_topology()
    df.to_pickle(f"{outdir}/{filename}")
    agents_seen = sorted(df["agent"].unique()) if len(df) else []
    logger.info(f"wrote {outdir}/{filename}: {len(df)} samples, agents {agents_seen}")


if __name__ == "__main__":
    import argparse
    from multiprocessing import Value

    parser = argparse.ArgumentParser(description="Per-agent CPU attribution sampler.")
    parser.add_argument("--pid", action="append", default=[], metavar="NAME=PID",
                        help="register an agent root, e.g. --pid main=1234 (repeatable)")
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--filename", default="agents.pkl")
    parser.add_argument("--cpu-clock", choices=["monotonic", "rocprofiler"], default="monotonic")
    parser.add_argument("--interval", type=float, default=0.2, help="sample interval seconds")
    parser.add_argument("--duration", type=float, default=10.0, help="seconds to sample")
    args = parser.parse_args()

    pids = {}
    for spec in args.pid:
        name, _, pid = spec.partition("=")
        pids[name] = int(pid)

    stop = Value("b", False)
    from threading import Thread
    t = Thread(target=main, args=(stop,), kwargs=dict(
        filename=args.filename, outdir=args.output_dir, agent_pids=pids,
        cpu_clock=args.cpu_clock, off=args.interval))
    t.start()
    sleep(args.duration)
    stop.value = True
    t.join()
