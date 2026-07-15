"""Read CUPTI's timestamp clock from Python via ctypes (NVIDIA side).

The NVIDIA analog of rocprofiler_clock.py. On AMD, cpu.py stamps samples with
rocprofiler_get_timestamp so they share the GPU kernel-trace clock; on NVIDIA the
equivalent is CUPTI's cuptiGetTimestamp, which returns a normalized nanosecond
timestamp in the same domain as CUPTI Activity API kernel records. Calling it
from Python puts CPU telemetry in the NVIDIA GPU clock domain, so the CPU+GPU
timeline join (merge_cpu_gpu_timeline) works on NVIDIA the same way it does on
AMD.

Loading is lazy and failure-safe: without CUPTI, is_available() is False and
get_timestamp() returns None, so callers fall back to time.monotonic_ns(). Set
CHOPPER_CUPTI_LIB to point at a specific libcupti.so if auto-discovery fails.

UNVALIDATED ON HARDWARE: written to the CUPTI API (cuptiGetTimestamp) and the
rocprofiler_clock pattern; confirm on an NVIDIA node before relying on it.
"""

import ctypes
import os
from typing import Callable, Optional

from loguru import logger

# CUPTI_SUCCESS == 0 in cupti_result.h
_CUPTI_SUCCESS = 0

_LIB_NAMES = (
    "libcupti.so",
    "libcupti.so.12",
    "libcupti.so.11",
)


def _candidate_paths() -> list[str]:
    env = os.environ.get("CHOPPER_CUPTI_LIB")
    paths = [env] if env else []
    # CUPTI lives under the CUDA toolkit's extras dir; try common roots.
    roots = []
    for var in ("CUDA_HOME", "CUDA_PATH", "CUDA_ROOT"):
        if os.environ.get(var):
            roots.append(os.environ[var])
    roots += ["/usr/local/cuda", "/opt/cuda"]
    dirs = []
    for r in roots:
        dirs.append(os.path.join(r, "extras", "CUPTI", "lib64"))
        dirs.append(os.path.join(r, "lib64"))
    # Bare names let the dynamic linker search LD_LIBRARY_PATH first (module env).
    for name in _LIB_NAMES:
        paths.append(name)
        for d in dirs:
            paths.append(os.path.join(d, name))
    return paths


_fn: Optional[Callable[..., int]] = None
_load_attempted = False


def _load() -> None:
    """Lazily locate and bind cuptiGetTimestamp. Never raises."""
    global _fn, _load_attempted
    if _load_attempted:
        return
    _load_attempted = True

    for path in _candidate_paths():
        try:
            lib = ctypes.CDLL(path)
        except OSError:
            continue
        try:
            fn = lib.cuptiGetTimestamp
        except AttributeError:
            continue
        # CUptiResult cuptiGetTimestamp(uint64_t* timestamp)
        fn.argtypes = [ctypes.POINTER(ctypes.c_uint64)]
        fn.restype = ctypes.c_int
        _fn = fn
        logger.info(f"CUPTI clock bound from {path}")
        return

    logger.warning(
        "CUPTI not found; NVIDIA CPU telemetry will use monotonic_ns. "
        "Set CHOPPER_CUPTI_LIB to override."
    )


def is_available() -> bool:
    """True if cuptiGetTimestamp could be bound (CUPTI present)."""
    _load()
    return _fn is not None


def get_timestamp() -> Optional[int]:
    """Return the CUPTI timestamp in nanoseconds, or None if unavailable.

    Same clock as CUPTI Activity kernel records, so a value stamped here lines up
    with the NVIDIA GPU-side timeline.
    """
    _load()
    if _fn is None:
        return None
    ts = ctypes.c_uint64(0)
    status = _fn(ctypes.byref(ts))
    if status != _CUPTI_SUCCESS:
        logger.warning(f"cuptiGetTimestamp returned status {status}")
        return None
    return int(ts.value)
