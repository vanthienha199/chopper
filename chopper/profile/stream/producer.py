"""Turn raw device counter samples into a stream of derived-metric records.

The device sampler reports each counter as a running total at a wall-clock
timestamp. ``StreamProducer`` keeps the previous sample per GPU, diffs the new
one against it, derives instantaneous metrics (metrics.derive), wraps the
result in a record, and emits it to a sink.

Record schemas (JSON):
  gpu_counters:
    {"type":"gpu_counters","source":str,"gpu":int,"t_ns":int,"dt_ns":int,
     "metrics":{...},"counters":{raw deltas...}}
  kernel_dispatch:
    {"type":"kernel_dispatch","source":str,"gpu":int,"name":str,
     "start_ns":int,"end_ns":int,"dur_ns":int}

The producer is backend-agnostic: it only assumes counters are cumulative and
named as in metrics.METRIC_COUNTERS. An NVIDIA backend that reports the same
logical quantities can feed the same producer.
"""

from typing import Any

from chopper.profile.stream import metrics
from chopper.profile.stream.sink import StreamSink, NullSink


class StreamProducer:
    """Stateful per-GPU differ + deriver that emits records to a sink."""

    def __init__(
        self,
        sink: StreamSink | None = None,
        arch: metrics.ArchConstants = metrics.MI300_SERIES,
        source: str = "chopper",
    ):
        self._sink = sink if sink is not None else NullSink()
        self._arch = arch
        self._source = source
        self._prev: dict[int, tuple[int, dict[str, float]]] = {}

    def push_counter_sample(
        self, gpu: int, t_ns: int, counters: dict[str, float]
    ) -> dict[str, Any] | None:
        """Ingest one cumulative counter sample for a GPU.

        Returns the emitted record, or None for the first sample of a GPU
        (no previous sample to diff against yet).
        """
        prev = self._prev.get(gpu)
        self._prev[gpu] = (t_ns, dict(counters))
        if prev is None:
            return None

        prev_t, prev_c = prev
        dt_ns = t_ns - prev_t
        if dt_ns <= 0:
            return None

        deltas: dict[str, float] = {}
        for name, val in counters.items():
            if name in prev_c:
                # Counters are monotonic; clamp tiny negatives from wraps/resets.
                deltas[name] = max(0.0, float(val) - float(prev_c[name]))

        derived = metrics.derive(deltas, dt_ns, self._arch)
        record = {
            "type": "gpu_counters",
            "source": self._source,
            "gpu": gpu,
            "t_ns": t_ns,
            "dt_ns": dt_ns,
            "metrics": derived,
            "counters": deltas,
        }
        self._sink.emit(record)
        return record

    def push_kernel(
        self, gpu: int, name: str, start_ns: int, end_ns: int
    ) -> dict[str, Any]:
        """Emit a kernel-dispatch record (for a live kernel timeline)."""
        record = {
            "type": "kernel_dispatch",
            "source": self._source,
            "gpu": gpu,
            "name": name,
            "start_ns": start_ns,
            "end_ns": end_ns,
            "dur_ns": end_ns - start_ns,
        }
        self._sink.emit(record)
        return record

    def push_activity(
        self, gpu: int, t_ns: int, window_ns: int, busy_pct: float, n_kernels: int
    ) -> dict[str, Any]:
        """Emit a GPU busy/idle record for one time window.

        Used when the source is a kernel timeline (no HW counters), e.g.
        replaying a real ts.pkl. ``busy_pct`` is the fraction of the window
        covered by kernel execution on this GPU.
        """
        record = {
            "type": "gpu_activity",
            "source": self._source,
            "gpu": gpu,
            "t_ns": t_ns,
            "window_ns": window_ns,
            "busy_pct": busy_pct,
            "n_kernels": n_kernels,
        }
        self._sink.emit(record)
        return record

    def close(self) -> None:
        self._sink.close()
