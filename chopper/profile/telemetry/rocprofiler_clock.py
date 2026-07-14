"""Read rocprofiler-sdk's timestamp clock from Python via ctypes.

Marco's pointer (Slack, 2026-07-10): the device sampler stamps every GPU counter
sample and kernel-dispatch record with ``rocprofiler_get_timestamp``
(``telemetry/src/device_counters_tool.cpp:536-537``). Calling the SAME function
from Python puts CPU telemetry samples in the SAME clock domain as the GPU
kernel traces and device counters, so the two can be joined on one timeline
without guessing a clock offset.

This supplies only the shared clock. It does NOT collect CPU hardware counters
(that waits on Marco's counter pointers). It is the foundation for the
CPU-and-GPU-on-one-timeline view Dr. Wu asked for in the 2026-07-10 meeting.

Loading is lazy and failure-safe: on a machine without ROCm (e.g. a Mac),
``is_available()`` returns False and ``get_timestamp()`` returns None, so callers
fall back to ``time.monotonic_ns()``. Set ``CHOPPER_ROCPROFILER_LIB`` to point at
a specific ``librocprofiler-sdk.so`` if auto-discovery fails.

UNVALIDATED ON HARDWARE: written to Marco's spec and the C++ usage; confirm on
the AMD cluster with ``examples/streaming/check_rocprofiler_clock.py`` before
relying on it (Marco: try it before collecting CPU data).
"""

import ctypes
import os
from typing import Callable, Optional

from loguru import logger

# ROCPROFILER_STATUS_SUCCESS == 0 in rocprofiler-sdk/fwd.h
_STATUS_SUCCESS = 0

_LIB_NAMES = (
    "librocprofiler-sdk.so",
    "librocprofiler-sdk.so.0",
    "librocprofiler-sdk.so.1",
)
_LIB_DIRS = (
    os.environ.get("ROCM_PATH", "/opt/rocm") + "/lib",
    os.environ.get("ROCM_PATH", "/opt/rocm") + "/lib64",
    "/opt/rocm/lib",
    "/opt/rocm/lib64",
)

_fn: Optional[Callable[..., int]] = None
_load_attempted = False


def _candidate_paths() -> list[str]:
    env = os.environ.get("CHOPPER_ROCPROFILER_LIB")
    paths = [env] if env else []
    # Bare names let the dynamic linker search LD_LIBRARY_PATH first.
    paths.extend(_LIB_NAMES)
    for d in _LIB_DIRS:
        for name in _LIB_NAMES:
            paths.append(os.path.join(d, name))
    return paths


def _load() -> None:
    """Lazily locate and bind rocprofiler_get_timestamp. Never raises."""
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
            fn = lib.rocprofiler_get_timestamp
        except AttributeError:
            continue
        # rocprofiler_status_t rocprofiler_get_timestamp(rocprofiler_timestamp_t*)
        fn.argtypes = [ctypes.POINTER(ctypes.c_uint64)]
        fn.restype = ctypes.c_int
        _fn = fn
        logger.info(f"rocprofiler clock bound from {path}")
        return

    logger.warning(
        "rocprofiler-sdk not found; CPU telemetry will use monotonic_ns. "
        "Set CHOPPER_ROCPROFILER_LIB to override."
    )


def is_available() -> bool:
    """True if rocprofiler_get_timestamp could be bound (ROCm present)."""
    _load()
    return _fn is not None


def get_timestamp() -> Optional[int]:
    """Return the rocprofiler timestamp in nanoseconds, or None if unavailable.

    This is the same clock the device sampler and GPU kernel traces use, so a
    value stamped here lines up directly with the GPU-side timeline.
    """
    _load()
    if _fn is None:
        return None
    ts = ctypes.c_uint64(0)
    status = _fn(ctypes.byref(ts))
    if status != _STATUS_SUCCESS:
        logger.warning(f"rocprofiler_get_timestamp returned status {status}")
        return None
    return int(ts.value)
