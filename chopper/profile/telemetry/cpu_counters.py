"""CPU hardware performance counters via perf_event_open (ctypes).

The CPU side of Dr. Wu's runtime CPU+GPU counter collection. psutil only gives
utilization; this reads real CPU hardware counters (instructions, cache misses,
cycles, IPC) with the Linux perf_event_open syscall, sampled in the background
and stamped with the rocprofiler clock so the samples land on the same timeline
as the GPU kernel traces and device counters.

Unprivileged and low-overhead by design: hardware events count user-space only
(exclude_kernel), which is what perf_event_paranoid>=2 allows without root. This
mirrors the GPU side, where full PMC accuracy also needs elevated privilege.
Software events (task_clock, cpu_clock) are scheduler accounting rather than PMU
counters. They are tried first without the exclude bits, which counts kernel
time spent on behalf of the task; at perf_event_paranoid=2 the kernel refuses
that for an unprivileged user with EACCES, so they fall back to the user-only
scope that matches the hardware counters beside them. SW_COUNTER_SCOPE records
which scope was obtained. task_clock is the one counter that gives host busy
time directly in nanoseconds, which is why it is here: without it, busy time can
only be inferred from cycles over an assumed core frequency. Measured on an
EPYC 7763 login node, task_clock recovered a known duty cycle to within 0.1
percent at 25, 50, 75 and 100 percent, while the cycles-over-frequency estimate
spanned a bracket 38 percent wide over the 2.45 to 3.50 GHz range.

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
PERF_TYPE_SOFTWARE = 1
PERF_TYPE_HW_CACHE = 3
# perf_event_attr.config (PERF_TYPE_HARDWARE)
HW_COUNTERS = {
    "cpu_cycles": 0,
    "instructions": 1,
    "cache_references": 2,
    "cache_misses": 3,
    "branch_instructions": 4,
    "branch_misses": 5,
}
# per-level cache events (PERF_TYPE_HW_CACHE):
# config = cache_id | (op_id << 8) | (result_id << 16). A miss at one level
# is traffic INTO the next level, so per-level rooflines read: L1D accesses
# = core-side traffic, L1D misses = L2 traffic, LLC misses = DRAM traffic.
# Availability varies by CPU model; unsupported events are skipped with a
# warning, never fatal.
_C_L1D, _C_LL = 0, 2
_OP_READ, _RES_ACCESS, _RES_MISS = 0, 0, 1
HW_CACHE_COUNTERS = {
    "l1d_read_access": _C_L1D | (_OP_READ << 8) | (_RES_ACCESS << 16),
    "l1d_read_miss": _C_L1D | (_OP_READ << 8) | (_RES_MISS << 16),
    "llc_read_access": _C_LL | (_OP_READ << 8) | (_RES_ACCESS << 16),
    "llc_read_miss": _C_LL | (_OP_READ << 8) | (_RES_MISS << 16),
}
# software events (PERF_TYPE_SOFTWARE). task_clock counts nanoseconds the task
# was actually on a CPU, so it measures host busy time directly. Without it the
# only route to busy time is cycles divided by an assumed core frequency, which
# turns every busy-versus-idle split into an interval bracketed over the CPU's
# frequency range instead of a measurement. cpu_clock is the wall-clock
# companion, and the pair gives the occupancy of the sampled task.
SW_COUNTERS = {
    "cpu_clock_ns": 0,
    "task_clock_ns": 1,
}
# Filled by _open_named: which scope each software counter was actually
# opened with, since that depends on perf_event_paranoid and changes what the
# number means. Written into the output DataFrame attrs so the reader sees it.
SW_COUNTER_SCOPE: dict = {}
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


def _perf_event_open(config: int, pid: int, inherit: bool,
                     type_: int = PERF_TYPE_HARDWARE,
                     exclude_kernel: bool = True) -> int:
    """Open one counter. Returns an fd, or -1 on failure.

    exclude_kernel controls the scope. Hardware counters always exclude kernel
    mode, which is what perf_event_paranoid>=2 permits unprivileged. Software
    events prefer the full-task scope, and _open_named retries them with the
    exclude bits when the kernel refuses.
    """
    attr = _PerfEventAttr()
    attr.type = type_
    attr.size = _PERF_ATTR_SIZE
    attr.config = config
    flags = (_F_EXCLUDE_KERNEL | _F_EXCLUDE_HV) if exclude_kernel else 0
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


def _open_named(name: str, pid: int, inherit: bool) -> int:
    """Open a counter by name from either the generic HW table or the
    per-level cache table. Returns fd or -1."""
    if name in HW_COUNTERS:
        return _perf_event_open(HW_COUNTERS[name], pid, inherit)
    if name in HW_CACHE_COUNTERS:
        return _perf_event_open(HW_CACHE_COUNTERS[name], pid, inherit,
                                type_=PERF_TYPE_HW_CACHE)
    if name in SW_COUNTERS:
        # Prefer the full-task version, which includes kernel time spent on
        # behalf of the task. perf_event_paranoid=2, the default on the
        # clusters we use, refuses that with EACCES for an unprivileged user,
        # so fall back to the user-only version rather than losing the counter.
        # The fallback then has the same scope as the hardware counters beside
        # it, which is the consistent reading; SW_COUNTER_SCOPE records which
        # one was obtained so a reader is never left guessing.
        fd = _perf_event_open(SW_COUNTERS[name], pid, inherit,
                              type_=PERF_TYPE_SOFTWARE,
                              exclude_kernel=False)
        if fd >= 0:
            SW_COUNTER_SCOPE[name] = "user+kernel"
            return fd
        fd = _perf_event_open(SW_COUNTERS[name], pid, inherit,
                              type_=PERF_TYPE_SOFTWARE,
                              exclude_kernel=True)
        if fd >= 0:
            SW_COUNTER_SCOPE[name] = "user only (perf_event_paranoid blocked "
            SW_COUNTER_SCOPE[name] += "kernel-mode accounting)"
        return fd
    return -1


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
        if (name not in HW_COUNTERS and name not in HW_CACHE_COUNTERS
                and name not in SW_COUNTERS):
            logger.warning(f"unknown CPU counter {name!r}, skipping")
            continue
        fd = _open_named(name, pid, inherit=(pid == 0))
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
    df["node"] = __import__("socket").gethostname()  # multi-node: rows carry their host
    df.attrs["clock_domain"] = clock_domain
    if SW_COUNTER_SCOPE:
        df.attrs["sw_counter_scope"] = dict(SW_COUNTER_SCOPE)
    df.to_pickle(f"{outdir}/{filename}")


def measure_command(
    program,
    filename: str = "cpu_counters.pkl",
    outdir: str = ".",
    counters: tuple = ("instructions", "cpu_cycles", "cache_references",
                       "cache_misses", "l1d_read_access", "l1d_read_miss",
                       "llc_read_access", "llc_read_miss"),
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
        fd = _open_named(name, 0, inherit=True)
        if fd >= 0:
            fds[name] = fd
        elif name in HW_CACHE_COUNTERS:
            logger.warning(f"cache event {name} unsupported on this CPU, skipped")
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
    df["node"] = __import__("socket").gethostname()  # multi-node: rows carry their host
    df.attrs["clock_domain"] = clock_domain
    if SW_COUNTER_SCOPE:
        df.attrs["sw_counter_scope"] = dict(SW_COUNTER_SCOPE)
    df.to_pickle(f"{outdir}/{filename}")
    logger.info(f"wrote {outdir}/{filename}: {len(df)} samples")


_PERF_EVENTS = ("instructions", "cpu-cycles", "cache-references",
                "cache-misses", "branches", "branch-misses")


def attach_pid(
    pid: int,
    filename: str = "cpu_counters.pkl",
    outdir: str = ".",
    cpu_clock: str = "monotonic",
    interval_ms: int = 200,
    duration_s: float = 0.0,
):
    """Attach to an ALREADY-RUNNING process (e.g. a vLLM server) and record its
    CPU hardware counters over time, without launching it.

    Uses `perf stat -p <pid> -I` under the hood because it correctly attaches to
    all threads of a running process (perf_event_open with inherit only catches
    future children, not existing sibling threads). Each interval is stamped with
    the chosen clock so it lines up with the GPU timeline. Linux + perf only.

    This is the hook for profiling Suhas's vLLM run: start his server, get its
    pid, attach.
    """
    import shutil
    import subprocess

    if shutil.which("perf") is None:
        logger.warning("perf tool not found; cannot attach to pid")
        return
    from chopper.profile.telemetry.cpu import _resolve_clock
    clock, clock_domain = _resolve_clock(cpu_clock)

    events = ",".join(e + ":u" for e in _PERF_EVENTS)  # user-space, unprivileged
    cmd = ["perf", "stat", "-p", str(pid), "-I", str(interval_ms), "-x", ",", "-e", events]
    logger.info(f"attaching perf to pid {pid}, events {_PERF_EVENTS} (clock={clock_domain})")
    proc = subprocess.Popen(cmd, stderr=subprocess.PIPE, text=True, bufsize=1)

    rows: dict = {}
    end_at = None
    assert proc.stderr is not None
    for line in proc.stderr:
        parts = line.strip().split(",")
        # interval CSV: <time>,<value>,<unit>,<event>,<run>,<pct>,...
        if len(parts) < 4:
            continue
        try:
            itime = float(parts[0])
        except ValueError:
            continue
        val = parts[1]
        event = parts[3].split(":")[0]
        row = rows.setdefault(itime, {"ts": clock(), "perf_interval_s": itime})
        try:
            row[event] = int(val)
        except ValueError:
            row[event] = None
        if end_at is None and duration_s > 0:
            end_at = time_monotonic() + duration_s
        if end_at is not None and time_monotonic() >= end_at:
            proc.terminate()
            break
    proc.wait()

    os.makedirs(outdir, exist_ok=True)
    df = pd.DataFrame(list(rows.values()))
    df["node"] = __import__("socket").gethostname()  # multi-node: rows carry their host
    df.attrs["clock_domain"] = clock_domain
    if SW_COUNTER_SCOPE:
        df.attrs["sw_counter_scope"] = dict(SW_COUNTER_SCOPE)
    df.to_pickle(f"{outdir}/{filename}")
    logger.info(f"wrote {outdir}/{filename}: {len(df)} intervals from pid {pid}")


def time_monotonic() -> float:
    from time import monotonic
    return monotonic()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Measure a workload's CPU hardware counters over time.")
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--filename", default="cpu_counters.pkl")
    parser.add_argument("--cpu-clock", choices=["monotonic", "rocprofiler"], default="monotonic")
    parser.add_argument("--off", type=float, default=0.02, help="sample interval seconds")
    parser.add_argument("--attach", type=int, metavar="PID",
                        help="attach to an already-running process instead of launching one")
    parser.add_argument("--interval-ms", type=int, default=200, help="perf interval for --attach")
    parser.add_argument("--duration", type=float, default=0.0, help="seconds to profile with --attach (0=until it exits)")
    parser.add_argument("program", nargs="*", help="workload to run and measure")
    args = parser.parse_args()
    if args.attach:
        attach_pid(args.attach, args.filename, args.output_dir,
                   cpu_clock=args.cpu_clock, interval_ms=args.interval_ms,
                   duration_s=args.duration)
    else:
        measure_command(args.program, args.filename, args.output_dir,
                        cpu_clock=args.cpu_clock, off=args.off)
