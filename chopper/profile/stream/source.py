"""Sources of counter samples for the stream.

``FakeCounterSource`` synthesizes cumulative counter samples with alternating
active (kernel running) and idle windows, so the whole streaming pipeline runs
on a laptop with no GPU. The active/idle shape is deliberate: it mirrors the
agentic-AI pattern where the GPU sits idle during tool calls, which is the
thing the live dashboard is meant to make visible.

``CsvTailSource`` follows a real ``counter_samples.csv`` as the device sampler
appends to it, so the same producer can drive a real AMD run later.

Both yield ``events`` in time order:
  ("counter", gpu:int, t_ns:int, cumulative_counters:dict)
  ("kernel",  gpu:int, name:str, start_ns:int, end_ns:int)
"""

import csv
import random
import time
from pathlib import Path
from typing import Any, Callable, Iterator

from chopper.profile.stream.metrics import ArchConstants, MI300_SERIES
from chopper.profile.stream.producer import StreamProducer

Event = tuple[Any, ...]

# The four counter groups' worth of raw counters, all emitted together so the
# producer can derive every metric in the demo. A real run collects <=4 per
# group; the producer already handles partial sets.
ALL_COUNTERS = (
    "GRBM_GUI_ACTIVE",
    "SQ_VALU_MFMA_BUSY_CYCLES",
    "TCC_BUBBLE",
    "TCC_EA0_RDREQ",
    "TCC_EA0_RDREQ_32B",
    "TCC_EA0_WRREQ_64B",
    "TCC_EA0_WRREQ",
)


class FakeCounterSource:
    """Synthetic active/idle counter samples for GPU-free testing."""

    def __init__(
        self,
        n_gpus: int = 2,
        duration_s: float = 6.0,
        sample_ms: int = 1,
        active_freq_mhz: float = 2100.0,
        idle_freq_mhz: float = 800.0,
        active_util_pct: float = 78.0,
        active_read_gbs: float = 3200.0,
        active_write_gbs: float = 1100.0,
        phase_s: float = 0.6,
        seed: int = 0,
        arch: ArchConstants = MI300_SERIES,
    ):
        self.n_gpus = n_gpus
        self.duration_s = duration_s
        self.sample_ms = sample_ms
        self.active_freq_mhz = active_freq_mhz
        self.idle_freq_mhz = idle_freq_mhz
        self.active_util_pct = active_util_pct
        self.active_read_gbs = active_read_gbs
        self.active_write_gbs = active_write_gbs
        self.phase_s = phase_s
        self.arch = arch
        self._rng = random.Random(seed)

    def _is_active(self, gpu: int, t_s: float) -> bool:
        # Square-ish active/idle wave, phase-shifted per GPU so they stagger
        # (a crude stand-in for one straggler GPU running long).
        shift = 0.12 * gpu
        return ((t_s + shift) % (2 * self.phase_s)) < self.phase_s

    def _jitter(self) -> float:
        return 1.0 + self._rng.uniform(-0.04, 0.04)

    def _interval_deltas(self, gpu: int, active: bool, dt_ns: float) -> dict[str, float]:
        freq = self.active_freq_mhz if active else self.idle_freq_mhz
        grbm = freq * self.arch.n_xcd * (dt_ns / 1e3) * self._jitter()

        util = self.active_util_pct if active else 0.0
        mfma = util / 100.0 * (self.arch.n_cu * grbm / self.arch.n_xcd * 4)

        read_bytes = (self.active_read_gbs if active else 0.0) * dt_ns * self._jitter()
        write_bytes = (self.active_write_gbs if active else 0.0) * dt_ns * self._jitter()

        return {
            "GRBM_GUI_ACTIVE": grbm,
            "SQ_VALU_MFMA_BUSY_CYCLES": mfma,
            "TCC_BUBBLE": 0.0,
            "TCC_EA0_RDREQ": read_bytes / 64.0,
            "TCC_EA0_RDREQ_32B": 0.0,
            "TCC_EA0_WRREQ_64B": write_bytes / 64.0,
            "TCC_EA0_WRREQ": write_bytes / 64.0,
        }

    def events(self) -> Iterator[Event]:
        """Yield counter and kernel events in time order (no sleeping)."""
        dt_ns = self.sample_ms * 1_000_000
        n_steps = int(self.duration_s * 1000 / self.sample_ms)
        cum = {g: {c: 0.0 for c in ALL_COUNTERS} for g in range(self.n_gpus)}
        active_prev = {g: False for g in range(self.n_gpus)}
        kern_start = {g: 0 for g in range(self.n_gpus)}

        for step in range(n_steps):
            t_ns = step * dt_ns
            t_s = t_ns / 1e9
            for g in range(self.n_gpus):
                active = self._is_active(g, t_s)
                deltas = self._interval_deltas(g, active, dt_ns)
                for c, dv in deltas.items():
                    cum[g][c] += dv
                yield ("counter", g, t_ns, dict(cum[g]))

                # Emit a kernel record on each active->idle edge (kernel ended).
                if active and not active_prev[g]:
                    kern_start[g] = t_ns
                if not active and active_prev[g]:
                    yield ("kernel", g, "Cijk_gemm_fp16", kern_start[g], t_ns)
                active_prev[g] = active

    def stream(self, producer: StreamProducer, realtime: bool = False, speed: float = 1.0) -> None:
        """Drive a StreamProducer from these events.

        realtime=True paces to wall clock (divided by ``speed``); False runs as
        fast as possible (for tests).
        """
        t0 = time.monotonic()
        for ev in self.events():
            if ev[0] == "counter":
                _, gpu, t_ns, counters = ev
                if realtime:
                    target = t0 + (t_ns / 1e9) / speed
                    delay = target - time.monotonic()
                    if delay > 0:
                        time.sleep(delay)
                producer.push_counter_sample(gpu, t_ns, counters)
            else:
                _, gpu, name, start_ns, end_ns = ev
                producer.push_kernel(gpu, name, start_ns, end_ns)


class CsvTailSource:
    """Follow a real device-sampler counter_samples.csv as rows are appended.

    Columns match device_counters_tool.cpp output:
      timestamp_ns,counter_name,counter_value,agent_id,<dims...>
    Rows for the same timestamp_ns are grouped into one cumulative sample.
    Best-effort: if the file does not exist yet it waits for it.
    """

    def __init__(self, path: str, gpu: int = 0, poll_s: float = 0.05):
        self.path = Path(path)
        self.gpu = gpu
        self.poll_s = poll_s

    def events(self, stop: Callable[[], bool] = lambda: False) -> Iterator[Event]:
        while not self.path.exists() and not stop():
            time.sleep(self.poll_s)
        if stop():
            return

        # Tail by reading newly-appended bytes and splitting on newlines, so a
        # partially written last line is buffered until it completes. Avoids
        # csv.DictReader, whose internal next() disables tell()/seek().
        with open(self.path, newline="") as f:
            buf = ""
            header: list[str] | None = None
            col: dict[str, int] = {}
            pending_ts: int | None = None
            pending: dict[str, float] = {}
            while not stop():
                chunk = f.read()
                if not chunk:
                    time.sleep(self.poll_s)
                    continue
                buf += chunk
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    if not line.strip():
                        continue
                    fields = next(csv.reader([line]))
                    if header is None:
                        header = fields
                        col = {name: i for i, name in enumerate(header)}
                        continue
                    ts = int(fields[col["timestamp_ns"]])
                    if pending_ts is not None and ts != pending_ts:
                        yield ("counter", self.gpu, pending_ts, dict(pending))
                        pending = {}
                    pending_ts = ts
                    pending[fields[col["counter_name"]]] = float(fields[col["counter_value"]])
