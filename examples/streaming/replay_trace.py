"""Replay a real Chopper ts.pkl through the streaming layer.

Reads a merged trace pickle (real kernel timeline from an actual run), picks
one training iteration, computes per-GPU GPU-busy fraction in fixed time bins,
and streams both the busy timeline and the real kernel dispatches over a socket
to a live consumer. This is real GPU data (real kernel names + timings), just
replayed, so the stream can be demonstrated without a live GPU.

Note: a torchtitan training trace keeps the GPU largely busy; the busy/idle
curve here shows real compute density and per-GPU straggler, not an agentic
idle pattern (that needs an agent run's trace).

Usage:
    python examples/streaming/replay_trace.py --pkl ~/projects/research/ts.pkl \
        --host 127.0.0.1 --port 8900 [--iteration N] [--bin-ms 5] [--speed 4]
"""

import argparse

import pandas as pd

from chopper.profile.stream.producer import StreamProducer
from chopper.profile.stream.sink import JsonLinesSocketSink


def build_events(pkl: str, iteration=None, bin_ms: int = 5):
    """Return (events, meta). events is a time-sorted list of tuples:
      ("activity", gpu, t_ns, bin_ns, busy_pct, n_kernels)
      ("kernel",   gpu, name, start_ns, end_ns)
    Timestamps are normalized so the iteration starts at 0.
    """
    df = pd.read_pickle(pkl)
    # end_ts in ts.pkl is NOT the kernel end; kernel end is ts + dur.
    df = df[["name", "ts", "dur", "gpu", "iteration"]].copy()

    iters = sorted(df["iteration"].dropna().unique())
    if iteration is None:
        iteration = iters[len(iters) // 2]
    sub = df[df["iteration"] == iteration].copy()
    assert len(sub) > 0, f"no kernels in iteration {iteration}"

    t0 = int(sub["ts"].min())
    sub["s"] = sub["ts"].astype("int64") - t0
    sub["e"] = sub["s"] + sub["dur"].astype("int64")
    span_ns = int(sub["e"].max())
    bin_ns = bin_ms * 1_000_000
    n_bins = span_ns // bin_ns + 1
    gpus = sorted(int(g) for g in sub["gpu"].unique())

    events = []
    for gpu in gpus:
        g = sub[sub["gpu"] == gpu].sort_values("s")
        starts = g["s"].to_numpy()
        ends = g["e"].to_numpy()

        # Merge overlapping kernel intervals (concurrent compute+comm streams)
        # into a non-overlapping busy timeline, so busy% is a true union.
        merged = []
        cs, ce = int(starts[0]), int(ends[0])
        for i in range(1, len(starts)):
            si, ei = int(starts[i]), int(ends[i])
            if si <= ce:
                ce = max(ce, ei)
            else:
                merged.append((cs, ce))
                cs, ce = si, ei
        merged.append((cs, ce))

        mi = 0
        for b in range(n_bins):
            lo = b * bin_ns
            hi = lo + bin_ns
            covered = 0
            j = mi
            while j < len(merged) and merged[j][0] < hi:
                ov = min(merged[j][1], hi) - max(merged[j][0], lo)
                if ov > 0:
                    covered += ov
                if merged[j][1] <= hi:
                    j += 1
                else:
                    break
            # advance mi past intervals fully behind this bin
            while mi < len(merged) and merged[mi][1] <= lo:
                mi += 1
            n_k = int(((starts >= lo) & (starts < hi)).sum())
            events.append(("activity", gpu, lo, bin_ns, 100.0 * covered / bin_ns, n_k))

        # stream real kernel dispatches too (sampled to keep it readable)
        step = max(1, len(g) // 400)
        gg = g.iloc[::step]
        for _, r in gg.iterrows():
            events.append(("kernel", gpu, str(r["name"])[:60], int(r["s"]), int(r["s"] + r["dur"])))

    events.sort(key=lambda ev: (ev[2] if ev[0] == "activity" else ev[3]))
    meta = {"iteration": float(iteration), "gpus": gpus, "span_s": span_ns / 1e9,
            "n_bins": n_bins, "bin_ms": bin_ms, "n_kernels": int(len(sub))}
    return events, meta


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pkl", required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8900)
    p.add_argument("--iteration", type=float, default=None)
    p.add_argument("--bin-ms", type=int, default=5)
    p.add_argument("--speed", type=float, default=4.0, help="realtime speed multiplier")
    p.add_argument("--no-realtime", action="store_true")
    a = p.parse_args()

    print(f"[replay] loading {a.pkl} ...")
    events, meta = build_events(a.pkl, a.iteration, a.bin_ms)
    print(f"[replay] iteration {meta['iteration']:.0f}: {meta['n_kernels']} real kernels, "
          f"{len(meta['gpus'])} GPUs, {meta['span_s']:.2f}s, {len(events)} events")

    sink = JsonLinesSocketSink(host=a.host, port=a.port)
    producer = StreamProducer(sink=sink, source="ts.pkl-replay")

    import time
    t_start = time.monotonic()
    for ev in events:
        if not a.no_realtime:
            t_ns = ev[2] if ev[0] == "activity" else ev[3]
            target = t_start + (t_ns / 1e9) / a.speed
            delay = target - time.monotonic()
            if delay > 0:
                time.sleep(delay)
        if ev[0] == "activity":
            _, gpu, t_ns, bin_ns, busy, n_k = ev
            producer.push_activity(gpu, t_ns, bin_ns, busy, n_k)
        else:
            _, gpu, name, s, e = ev
            producer.push_kernel(gpu, name, s, e)

    time.sleep(0.3)
    print(f"[replay] done, sink stats: {sink.stats}")
    producer.close()


if __name__ == "__main__":
    main()
