"""Auto-detect the GPU vendor so collect.py can pick the right subroutine.

Dr. Wu's ask (2026-07-10 area): Chopper should detect the underlying hardware
and automatically choose the correct backend instead of relying on a manual
--nvidia flag. This returns a Vendor; the telemetry/counter collectors dispatch
on it (amdsmi/rocprofv3 for AMD, pynvml/CUPTI for NVIDIA).

Detection prefers a cheap library-spec check (does amdsmi / pynvml import) and
falls back to the vendor CLIs (rocminfo / nvidia-smi). Returns UNKNOWN if
neither is present. Keep --vendor {auto,amd,nvidia} as a manual override.
"""

import importlib.util
import shutil
from enum import Enum


class Vendor(str, Enum):
    AMD = "amd"
    NVIDIA = "nvidia"
    UNKNOWN = "unknown"


def _has_module(name: str) -> bool:
    # find_spec avoids importing amdsmi/pynvml, which can init drivers as a side effect.
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _has_exe(name: str) -> bool:
    return shutil.which(name) is not None


def detect_vendor() -> Vendor:
    if _has_module("amdsmi") or _has_exe("rocminfo") or _has_exe("amd-smi"):
        return Vendor.AMD
    if _has_module("pynvml") or _has_exe("nvidia-smi"):
        return Vendor.NVIDIA
    return Vendor.UNKNOWN


def resolve_vendor(override: str = "auto") -> Vendor:
    """Resolve the vendor, honoring a manual override ('auto'|'amd'|'nvidia')."""
    if override == "auto":
        return detect_vendor()
    return Vendor(override)
