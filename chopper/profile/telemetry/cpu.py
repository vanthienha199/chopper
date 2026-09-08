import os
from time import monotonic_ns
from time import sleep
import psutil
import pandas as pd
from loguru import logger
from multiprocessing.sharedctypes import Synchronized
from typing import Any, Callable


def _resolve_clock(cpu_clock: str) -> tuple[Callable[[], int], str]:
    """Pick the timestamp source for CPU samples.

    'rocprofiler' puts CPU samples in the same clock domain as GPU kernel
    traces and device counters (Marco's pointer), enabling a direct join. Falls
    back to monotonic_ns if rocprofiler is unavailable so nothing breaks on a
    non-ROCm box. Returns (clock_fn, domain_name) where domain_name is recorded
    in the pickle so merge.py knows how to align it.
    """
    if cpu_clock == "rocprofiler":
        from chopper.profile.telemetry import rocprofiler_clock
        if rocprofiler_clock.is_available():
            logger.info("cpu telemetry using rocprofiler clock domain")

            def _rp() -> int:
                t = rocprofiler_clock.get_timestamp()
                return t if t is not None else monotonic_ns()

            return _rp, "rocprofiler"
        logger.warning("rocprofiler clock requested but unavailable; using monotonic_ns")
    return monotonic_ns, "monotonic_ns"


def main(
    stop: Synchronized,
    filename: str = 'cpu.pkl',
    outdir: str = '.',
    on: float = 0.0,
    off: float = 0.1,
    cpu_clock: str = "monotonic",
    **kwargs,
):
    assert hasattr(psutil.Process, "cpu_num"), "not supported"

    num_cpus = psutil.cpu_count()
    assert num_cpus is not None, "cpu_count returned None"

    clock, clock_domain = _resolve_clock(cpu_clock)

    results = []
    pause_ts = clock() + int(on * 1e9)

    while not stop.value:
        ts = clock()
        cpu_entries: dict[int, dict[str, Any]] = {}
        cpus_percent = psutil.cpu_percent(percpu=True)

        for cpu_num in range(num_cpus):
            cpu_entries[cpu_num] = {'percent': cpus_percent[cpu_num]}

        for p in psutil.process_iter(['name', 'cmdline', 'cpu_num']):
            cpu_num_val = p.info.get('cpu_num')
            if cpu_num_val is not None and isinstance(cpu_num_val, int):
                if 'name' not in cpu_entries[cpu_num_val]:
                    cpu_entries[cpu_num_val]['name'] = []
                if 'cmdline' not in cpu_entries[cpu_num_val]:
                    cpu_entries[cpu_num_val]['cmdline'] = []
                cpu_entries[cpu_num_val]['name'].append(p.info['name'])
                cpu_entries[cpu_num_val]['cmdline'].append(p.info['cmdline'])

        for cpu, entry in cpu_entries.items():
            results.append({'cpu': cpu, 'ts': ts, **entry})

        if ts >= pause_ts:
            sleep(off)
            pause_ts = clock() + int(on * 1e9)

    os.makedirs(outdir, exist_ok=True)
    df = pd.DataFrame(results)
    df["node"] = __import__("socket").gethostname()  # multi-node: rows carry their host
    # Record the clock domain so merge.py can align cpu.pkl to the GPU timeline.
    df.attrs["clock_domain"] = clock_domain
    df.to_pickle(f"{outdir}/{filename}")
