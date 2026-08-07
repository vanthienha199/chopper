"""Per-agent storage I/O telemetry.

Fills the storage column of the metric collection layer: per-process read and
write bytes (procfs io counters via psutil), attributed to agents by the same
process-tree rules as agents.py, plus system-wide disk throughput. Samples are
cumulative; consumers diff consecutive samples for rates. Stamped with the
same clock as the other collectors so I/O lands on the shared timeline.

Standalone:
    python -m chopper.profile.telemetry.io --pid main=1234 --duration 10
"""

import os
from time import sleep
from typing import Any

import pandas as pd
import psutil
from loguru import logger

from chopper.profile.telemetry.agents import _attribute, _discover_tagged_roots


def main(
    stop: Any,
    filename: str = "io.pkl",
    outdir: str = ".",
    agent_pids: dict[str, int] | None = None,
    cpu_clock: str = "monotonic",
    off: float = 0.5,
    **kwargs: Any,
) -> None:
    """Background per-agent I/O sampler (collector contract)."""
    from chopper.profile.telemetry.cpu import _resolve_clock
    clock, clock_domain = _resolve_clock(cpu_clock)

    explicit_roots = {int(pid): name for name, pid in (agent_pids or {}).items()}
    env_cache: dict[int, str | None] = {}
    results: list[dict[str, Any]] = []
    have_proc_io = hasattr(psutil.Process, "io_counters")

    logger.info(f"io telemetry: explicit roots {explicit_roots} "
                f"(clock={clock_domain}, per-process={'yes' if have_proc_io else 'NO'})")

    while not stop.value:
        roots = dict(explicit_roots)
        roots.update(_discover_tagged_roots(env_cache))

        parent_of: dict[int, int] = {}
        procs: dict[int, Any] = {}
        for p in psutil.process_iter(["pid", "ppid", "name"]):
            parent_of[p.info["pid"]] = p.info["ppid"]
            procs[p.info["pid"]] = p

        ts = clock()
        # system-wide disk throughput (always available)
        d = psutil.disk_io_counters()
        if d is not None:
            results.append({"ts": ts, "agent": "_system", "pid": 0, "name": "disk",
                            "read_bytes": d.read_bytes, "write_bytes": d.write_bytes,
                            "read_count": d.read_count, "write_count": d.write_count})
        # per-agent, per-process io counters
        if have_proc_io and roots:
            for pid, p in procs.items():
                agent = _attribute(pid, roots, parent_of)
                if agent is None:
                    continue
                try:
                    io = p.io_counters()
                except (psutil.NoSuchProcess, psutil.ZombieProcess,
                        psutil.AccessDenied, AttributeError):
                    continue
                results.append({"ts": ts, "agent": agent, "pid": pid,
                                "name": p.info["name"],
                                "read_bytes": io.read_bytes,
                                "write_bytes": io.write_bytes,
                                "read_count": io.read_count,
                                "write_count": io.write_count})
        sleep(off)

    os.makedirs(outdir, exist_ok=True)
    df = pd.DataFrame(results)
    df.attrs["clock_domain"] = clock_domain
    df.to_pickle(f"{outdir}/{filename}")
    agents_seen = sorted(a for a in df["agent"].unique() if a != "_system") if len(df) else []
    logger.info(f"wrote {outdir}/{filename}: {len(df)} samples, agents {agents_seen}")


if __name__ == "__main__":
    import argparse
    from multiprocessing import Value
    from threading import Thread

    parser = argparse.ArgumentParser(description="Per-agent storage I/O sampler.")
    parser.add_argument("--pid", action="append", default=[], metavar="NAME=PID")
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--filename", default="io.pkl")
    parser.add_argument("--cpu-clock", choices=["monotonic", "rocprofiler"], default="monotonic")
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--duration", type=float, default=10.0)
    args = parser.parse_args()

    pids = {}
    for spec in args.pid:
        name, _, pid = spec.partition("=")
        pids[name] = int(pid)

    stop = Value("b", False)
    t = Thread(target=main, args=(stop,), kwargs=dict(
        filename=args.filename, outdir=args.output_dir, agent_pids=pids,
        cpu_clock=args.cpu_clock, off=args.interval))
    t.start()
    sleep(args.duration)
    stop.value = True
    t.join()
