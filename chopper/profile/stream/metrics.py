"""Instantaneous metric derivation from per-interval counter deltas.

These are the same formulas Chopper uses offline in
``common/rocm_metrics.py`` and ``plots/device_timeline.py``, reimplemented on
plain floats (no pandas) so they can run inside the live stream and be ported
to the dashboard's language verbatim.

Inputs are per-interval DELTAS of the cumulative device counters plus the
sample interval ``dt_ns``. The device sampler reports counters as running
totals, so callers must diff consecutive samples before calling ``derive``
(``producer.StreamProducer`` does this).

Metric -> required raw counters (matches device_timeline.METRIC_COUNTERS):
  tensor_util_pct : SQ_VALU_MFMA_BUSY_CYCLES, GRBM_GUI_ACTIVE
  gpu_freq_mhz    : GRBM_GUI_ACTIVE
  read_bw_gbs     : TCC_BUBBLE, TCC_EA0_RDREQ, TCC_EA0_RDREQ_32B
  write_bw_gbs    : TCC_EA0_WRREQ_64B, TCC_EA0_WRREQ
"""

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class ArchConstants:
    """Architecture constants used by the ROCm counter formulas.

    Defaults match the MI300/MI325 series values hard-coded in
    ``rocm_metrics.derive_tensor_util_rocm`` (n_xcd=8, n_cu=304, 2100 MHz
    nominal). Kept as data so an NVIDIA or future backend can supply its own.
    """
    n_xcd: int = 8
    n_cu: int = 304
    nominal_freq_mhz: int = 2100


MI300_SERIES = ArchConstants()

# Metric name -> the raw counter names it needs. A metric is derived only when
# all of its inputs are present in the delta (a real run collects <=4 counters
# per group, so any single group supports only a subset of metrics).
METRIC_COUNTERS: dict[str, tuple[str, ...]] = {
    "tensor_util_pct": ("SQ_VALU_MFMA_BUSY_CYCLES", "GRBM_GUI_ACTIVE"),
    "gpu_freq_mhz": ("GRBM_GUI_ACTIVE",),
    "read_bw_gbs": ("TCC_BUBBLE", "TCC_EA0_RDREQ", "TCC_EA0_RDREQ_32B"),
    "write_bw_gbs": ("TCC_EA0_WRREQ_64B", "TCC_EA0_WRREQ"),
}


def available_metrics(counter_names: Iterable[str]) -> list[str]:
    """Return the metrics derivable from the given set of counter names."""
    have = set(counter_names)
    return [m for m, need in METRIC_COUNTERS.items() if have.issuperset(need)]


def _tensor_util_pct(d: dict[str, float], dt_ns: float, arch: ArchConstants) -> float:
    active = d["GRBM_GUI_ACTIVE"]
    if active <= 0:
        return 0.0
    # rocm_metrics.derive_tensor_util_rocm
    return 100.0 * d["SQ_VALU_MFMA_BUSY_CYCLES"] / (arch.n_cu * active / arch.n_xcd * 4)


def _gpu_freq_mhz(d: dict[str, float], dt_ns: float, arch: ArchConstants) -> float:
    if dt_ns <= 0:
        return 0.0
    # device_timeline: (GRBM_GUI_ACTIVE / n_xcd) cycles over the interval.
    # cycles / microseconds = MHz, and dt_ns / 1e3 = microseconds.
    return (d["GRBM_GUI_ACTIVE"] / arch.n_xcd) / (dt_ns / 1e3)


def _read_bytes(d: dict[str, float]) -> float:
    # rocm_metrics.derive_l2_fabric_read_bw byte model.
    return (
        128 * d["TCC_BUBBLE"]
        + 64 * (d["TCC_EA0_RDREQ"] - d["TCC_BUBBLE"] - d["TCC_EA0_RDREQ_32B"])
        + 32 * d["TCC_EA0_RDREQ_32B"]
    )


def _write_bytes(d: dict[str, float]) -> float:
    # rocm_metrics.derive_l2_fabric_write_bw byte model.
    return 64 * d["TCC_EA0_WRREQ_64B"] + 32 * (d["TCC_EA0_WRREQ"] - d["TCC_EA0_WRREQ_64B"])


def derive(deltas: dict[str, float], dt_ns: float, arch: ArchConstants = MI300_SERIES) -> dict[str, float]:
    """Derive instantaneous metrics from one interval of counter deltas.

    Args:
        deltas: per-interval counter deltas (cumulative counters already diffed)
        dt_ns: length of the interval in nanoseconds
        arch: architecture constants for the formulas

    Returns:
        Dict of {metric_name: value} for every metric whose inputs are present.
        Bandwidth is returned in GB/s (bytes per nanosecond), plus the raw byte
        totals for the interval, which the dashboard can re-window as needed.
    """
    out: dict[str, float] = {}
    have = set(deltas)

    if have.issuperset(METRIC_COUNTERS["tensor_util_pct"]):
        out["tensor_util_pct"] = _tensor_util_pct(deltas, dt_ns, arch)
    if have.issuperset(METRIC_COUNTERS["gpu_freq_mhz"]):
        out["gpu_freq_mhz"] = _gpu_freq_mhz(deltas, dt_ns, arch)
    if have.issuperset(METRIC_COUNTERS["read_bw_gbs"]):
        rb = _read_bytes(deltas)
        out["read_bytes"] = rb
        out["read_bw_gbs"] = rb / dt_ns if dt_ns > 0 else 0.0
    if have.issuperset(METRIC_COUNTERS["write_bw_gbs"]):
        wb = _write_bytes(deltas)
        out["write_bytes"] = wb
        out["write_bw_gbs"] = wb / dt_ns if dt_ns > 0 else 0.0

    return out
