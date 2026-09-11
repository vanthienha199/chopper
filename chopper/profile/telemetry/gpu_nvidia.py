"""NVIDIA GPU telemetry via NVML (pynvml).

The NVIDIA counterpart of gpu.py (which uses amdsmi). Samples per-GPU power,
temperature, SM/memory clocks, and utilization at ~10 Hz into a pickle, matching
gpu.py's collector interface so the vendor dispatch can pick this module on
NVIDIA hardware.

Telemetry only. The CPU/GPU trace-clock alignment on NVIDIA uses CUPTI
(cuptiGetTimestamp), the analog of rocprofiler_get_timestamp, and is handled
separately; NVML telemetry is fine on monotonic_ns like the AMD side.

pynvml is nvidia-ml-py (a ctypes wrapper over libnvidia-ml.so). is_available()
is False without it or the driver, so callers can skip cleanly off NVIDIA.
"""

import os
import socket
from time import monotonic_ns, sleep

import pandas as pd
from loguru import logger


def is_available() -> bool:
    try:
        import pynvml
        pynvml.nvmlInit()
        pynvml.nvmlShutdown()
        return True
    except Exception:
        return False


def _read_device(pynvml, handle) -> dict:
    """Read one GPU's telemetry. Missing fields are recorded as None."""
    out: dict = {}

    def try_set(key, fn):
        try:
            out[key] = fn()
        except Exception:
            out[key] = None

    try_set("power_w", lambda: pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0)
    try_set("power_limit_w", lambda: pynvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000.0)
    try_set("temp_c", lambda: pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))
    try_set("sm_clock_mhz", lambda: pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM))
    try_set("mem_clock_mhz", lambda: pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM))
    try_set("gfx_clock_mhz", lambda: pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS))

    def util():
        u = pynvml.nvmlDeviceGetUtilizationRates(handle)
        return u.gpu, u.memory
    try:
        g, m = util()
        out["gpu_util_pct"] = g
        out["mem_util_pct"] = m
    except Exception:
        out["gpu_util_pct"] = None
        out["mem_util_pct"] = None

    try_set("mem_used_mb", lambda: pynvml.nvmlDeviceGetMemoryInfo(handle).used / (1024 * 1024))
    return out


def main(
    stop,
    filename: str = "gpu.pkl",
    nvidia: bool = True,
    outdir: str = ".",
    on: float = 0.0,
    off: float = 0.1,
    **kwargs,
):
    import pynvml

    if not is_available():
        logger.warning("NVML unavailable (no pynvml or driver); skipping NVIDIA GPU telemetry")
        return

    pynvml.nvmlInit()
    try:
        n = pynvml.nvmlDeviceGetCount()
        handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(n)]
        logger.info(f"NVIDIA GPU telemetry: {n} device(s)")

        results = []
        pause_ts = monotonic_ns() + int(on * 1e9)
        while not stop.value:
            for gpu, handle in enumerate(handles):
                ts = monotonic_ns()
                row = _read_device(pynvml, handle)
                row["ts"] = ts
                row["gpu"] = gpu
                results.append(row)
            if monotonic_ns() >= pause_ts:
                sleep(off)
                pause_ts = monotonic_ns() + int(on * 1e9)

        os.makedirs(outdir, exist_ok=True)
        df = pd.DataFrame(results)
        df["node"] = socket.gethostname()  # multi-node: rows carry their host
        df.attrs["vendor"] = "nvidia"
        df.to_pickle(f"{outdir}/{filename}")
    finally:
        pynvml.nvmlShutdown()
    return 0
