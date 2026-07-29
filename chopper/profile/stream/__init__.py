"""Real-time streaming of Chopper device telemetry to an external consumer.

This package is a thin, dependency-light layer that turns Chopper's device
counter samples into a live stream of derived metrics (MFMA util, GPU clock,
HBM bandwidth) pushed over a socket as newline-delimited JSON. The sink is
consumer-agnostic: it emits plain JSON to any live dashboard, so a run can be
watched in real time instead of waiting for the offline ``merge.py`` step.

Design constraints:
  - Stdlib only (no pandas/numpy/loguru) so the exact same derivation can be
    re-implemented on the dashboard side and so the sink never drags heavy
    imports into a profiled workload.
  - Emitting must never block or crash the workload being profiled. The socket
    sink is fully non-blocking and drops records if the consumer is slow or
    absent (telemetry is best-effort; the training run is not).
  - Additive only. Nothing here touches the existing AMD collection path.

Entry points:
  - ``StreamProducer`` (producer.py): feed it raw counter samples, it diffs,
    derives metrics, and emits records to a sink.
  - ``JsonLinesSocketSink`` (sink.py): non-blocking newline-delimited JSON over
    TCP.
  - ``FakeCounterSource`` (source.py): synthetic GEMM-plus-idle samples so the
    whole pipeline runs on a laptop with no GPU.
  - ``python -m chopper.profile.stream`` (__main__.py): runnable demo /
    receiver / self-test. The receiver is a minimal stand-in for whatever real
    dashboard consumes the stream.
"""

from chopper.profile.stream.metrics import ArchConstants, MI300_SERIES, derive, available_metrics
from chopper.profile.stream.sink import (
    StreamSink,
    NullSink,
    CallbackSink,
    JsonLinesSocketSink,
)
from chopper.profile.stream.producer import StreamProducer

__all__ = [
    "ArchConstants",
    "MI300_SERIES",
    "derive",
    "available_metrics",
    "StreamSink",
    "NullSink",
    "CallbackSink",
    "JsonLinesSocketSink",
    "StreamProducer",
]
