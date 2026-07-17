"""CPU hardware performance counters via perf_event_open (ctypes).

The CPU side of Dr. Wu's runtime CPU+GPU counter collection. psutil only gives
utilization; this reads real CPU hardware counters (instructions, cache misses,
cycles, IPC) with the Linux perf_event_open syscall, sampled in the background
and stamped with the rocprofiler clock so the samples land on the same timeline
as the GPU kernel traces and device counters.

Unprivileged and low-overhead by design: counts user-space events only
(exclude_kernel), which is what perf_event_paranoid>=2 allows without root. This
mirrors the GPU side, where full PMC accuracy also needs elevated privilege.

Linux/x86-64 only. On other platforms is_available() returns False and the
collector is skipped. Reading is cumulative; merge diffs consecutive samples.
"""

import ctypes
import os
import platform
import struct
from time import monotonic_ns, sleep

import pandas as pd
from loguru import logger

# x86-64 syscall number for perf_event_open.
_NR_perf_event_open = 298

# perf_event_attr.type
PERF_TYPE_HARDWARE = 0
# perf_event_attr.config (PERF_TYPE_HARDWARE)
HW_COUNTERS = {
    "cpu_cycles": 0,
    "instructions": 1,
    "cache_references": 2,
    "cache_misses": 3,
    "branch_instructions": 4,
    "branch_misses": 5,
}
# flags bitfield offsets within perf_event_attr
_F_DISABLED = 1 << 0
_F_INHERIT = 1 << 1
_F_EXCLUDE_KERNEL = 1 << 5
_F_EXCLUDE_HV = 1 << 6

_PERF_ATTR_SIZE = 128


class _PerfEventAttr(ctypes.Structure):
    # Only the leading fields we set; padded to the full struct size.
    _fields_ = [
        ("type", ctypes.c_uint32),
        ("size", ctypes.c_uint32),
        ("config", ctypes.c_uint64),
        ("sample_period", ctypes.c_uint64),
        ("sample_type", ctypes.c_uint64),
        ("read_format", ctypes.c_uint64),
        ("flags", ctypes.c_uint64),
        ("_pad", ctypes.c_uint8 * (_PERF_ATTR_SIZE - 48)),
    ]


def is_available() -> bool:
    """True only on Linux/x86-64 where perf_event_open is present."""
    return platform.system() == "Linux" and platform.machine() in ("x86_64", "AMD64")


def _perf_event_open(config: int, pid: int, inherit: bool) -> int:
    """Open one user-space HW counter. Returns an fd, or -1 on failure."""
    attr = _PerfEventAttr()
    attr.type = PERF_TYPE_HARDWARE
    attr.size = _PERF_ATTR_SIZE
    attr.config = config
    flags = _F_EXCLUDE_KERNEL | _F_EXCLUDE_HV
    if inherit:
        flags |= _F_INHERIT
    attr.flags = flags

    libc = ctypes.CDLL(None, use_errno=True)
    # perf_event_open(attr, pid, cpu, group_fd, flags); cpu=-1 (any), group=-1
    fd = libc.syscall(
        _NR_perf_event_open, ctypes.byref(attr),
        ctypes.c_int(pid), ctypes.c_int(-1), ctypes.c_int(-1), ctypes.c_ulong(0),
    )
    return int(fd)


def _read_counter(fd: int) -> int:
    return struct.unpack("<Q", os.read(fd, 8))[0]


def main(
    stop,
    filename: str = "cpu_counters.pkl",
    outdir: str = ".",
    pid: int = 0,
    counters: tuple = ("instructions", "cpu_cycles", "cache_references", "cache_misses"),
    cpu_clock: str = "monotonic",
    on: float = 0.0,
    off: float = 0.1,
    **kwargs,
):
    """Background CPU hardware-counter sampler.

    pid=0 measures the calling process (with inherit, its future children too).
    Pass a workload pid to profile that process tree. Writes cumulative counter
    values per sample; merge diffs them into per-interval deltas.
    """
    if not is_available():
        logger.warning("perf_event_open unavailable (not Linux/x86-64); skipping CPU counters")
        return

    from chopper.profile.telemetry.cpu import _resolve_clock
    clock, clock_domain = _resolve_clock(cpu_clock)

    fds = {}
    for name in counters:
        if name not in HW_COUNTERS:
            logger.warning(f"unknown CPU counter {name!r}, skipping")
            continue
        fd = _perf_event_open(HW_COUNTERS[name], pid, inherit=(pid == 0))
        if fd < 0:
            logger.warning(f"perf_event_open failed for {name} (errno={ctypes.get_errno()})")
            continue
        fds[name] = fd

    if not fds:
        logger.warning("no CPU counters opened; check perf_event_paranoid")
        return
    logger.info(f"CPU counters: {list(fds)} (clock={clock_domain})")

    results = []
    pause_ts = monotonic_ns() + int(on * 1e9)
    while not stop.value:
        ts = clock()
        row = {"ts": ts}
        for name, fd in fds.items():
            try:
                row[name] = _read_counter(fd)
            except OSError:
                row[name] = None
        results.append(row)
        if monotonic_ns() >= pause_ts:
            sleep(off)
            pause_ts = monotonic_ns() + int(on * 1e9)

    for fd in fds.values():
        os.close(fd)

    os.makedirs(outdir, exist_ok=True)
    df = pd.DataFrame(results)
    df.attrs["clock_domain"] = clock_domain
    df.to_pickle(f"{outdir}/{filename}")


def measure_command(
    program,
    filename: str = "cpu_counters.pkl",
    outdir: str = ".",
    counters: tuple = ("instructions", "cpu_cycles", "cache_references", "cache_misses"),
    cpu_clock: str = "monotonic",
    off: float = 0.02,
):
    """Launch a workload and sample the CPU hardware counters of IT (not the
    sampler). Counters are opened with inherit before the child is spawned, so
    they accumulate the whole workload process tree. Produces a per-interval
    time series stamped with the chosen clock, so CPU counters line up with the
    GPU timeline just like the utilization side does.
    """
    import subprocess

    if not is_available():
        logger.warning("perf_event_open unavailable; running program without CPU counters")
        subprocess.run(list(program))
        return

    from chopper.profile.telemetry.cpu import _resolve_clock
    clock, clock_domain = _resolve_clock(cpu_clock)

    fds = {}
    for name in counters:
        if name not in HW_COUNTERS:
            continue
        fd = _perf_event_open(HW_COUNTERS[name], 0, inherit=True)
        if fd >= 0:
            fds[name] = fd
    if not fds:
        logger.warning("no CPU counters opened; check perf_event_paranoid")
        subprocess.run(list(program))
        return
    logger.info(f"measuring '{' '.join(program)}' with CPU counters {list(fds)} (clock={clock_domain})")

    results = []
    proc = subprocess.Popen(list(program))
    while proc.poll() is None:
        ts = clock()
        row = {"ts": ts}
        for name, fd in fds.items():
            try:
                row[name] = _read_counter(fd)
            except OSError:
                row[name] = None
        results.append(row)
        sleep(off)
    proc.wait()

    for fd in fds.values():
        os.close(fd)
    os.makedirs(outdir, exist_ok=True)
    df = pd.DataFrame(results)
    df.attrs["clock_domain"] = clock_domain
    df.to_pickle(f"{outdir}/{filename}")
    logger.info(f"wrote {outdir}/{filename}: {len(df)} samples")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Measure a workload's CPU hardware counters over time.")
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--filename", default="cpu_counters.pkl")
    parser.add_argument("--cpu-clock", choices=["monotonic", "rocprofiler"], default="monotonic")
    parser.add_argument("--off", type=float, default=0.02, help="sample interval seconds")
    parser.add_argument("program", nargs="+", help="workload to run and measure")
    args = parser.parse_args()
    measure_command(args.program, args.filename, args.output_dir,
                    cpu_clock=args.cpu_clock, off=args.off)
