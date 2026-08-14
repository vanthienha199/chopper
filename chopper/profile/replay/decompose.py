"""Per-turn time decomposition (Samsung metric set, first bullet).

Splits each agent turn's wall time into:
  model_wait_s   time waiting on the model endpoint (ttft + response stream)
  tool_exec_s    recorded tool/command execution inside the turn window
  gpu_busy_s     union of GPU kernel/graph intervals inside the window
                 (needs a kernel trace whose timestamps are epoch ns, which
                 is what the CUPTI tracer emits; column is NaN when no trace
                 covers the window, e.g. a tracer outage)
  other_s        the residual: harness bookkeeping, scheduling, untracked
                 waits. Non-negative by construction.

A "turn" is the interval from one model call's arrival to the next call's
arrival (last turn ends at its model call end + trailing tool time). Inputs
are the proxy's model_calls.jsonl and the adapter's actions.jsonl; both are
epoch-stamped, so no clock conversion is needed. Optional kernel_traces.csv
adds the GPU column.

    python -m chopper.profile.replay.decompose \
        --model-calls model_calls.jsonl --actions actions.jsonl \
        [--kernel-csv kernel_traces.csv] [--task TASK_ID] \
        [--out-csv turns.csv] [--out-json report.json]
"""

import argparse
import json
from typing import Any


def _union_seconds(intervals: list[tuple[float, float]],
                   lo: float, hi: float) -> float:
    """Total covered time of intervals clipped to [lo, hi]."""
    clipped = sorted((max(s, lo), min(e, hi))
                     for s, e in intervals if e > lo and s < hi)
    total = 0.0
    cur_s, cur_e = None, None
    for s, e in clipped:
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_s
    return total


def _load_jsonl(path: str) -> list[dict[str, Any]]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _load_kernel_intervals(path: str) -> list[tuple[float, float]]:
    """Kernel CSV rows -> [(start_s, end_s)] in epoch seconds."""
    out: list[tuple[float, float]] = []
    with open(path) as f:
        header = f.readline()
        if "start_ns" not in header:
            return out
        for line in f:
            parts = line.rsplit(",", 3)
            if len(parts) != 4:
                continue
            try:
                s = int(parts[1]) / 1e9
                e = int(parts[2]) / 1e9
            except ValueError:
                continue
            if e >= s:
                out.append((s, e))
    return out


def decompose(model_calls: list[dict[str, Any]],
              actions: list[dict[str, Any]],
              kernel_intervals: list[tuple[float, float]] | None = None,
              task: str | None = None) -> list[dict[str, Any]]:
    calls = [c for c in model_calls
             if not task or c.get("task_id") == task]
    calls = sorted(calls, key=lambda c: c.get("ts_epoch_s", 0.0))
    acts = [a for a in actions if not task or a.get("task_id") == task]

    turns: list[dict[str, Any]] = []
    for i, c in enumerate(calls):
        t0 = float(c.get("ts_epoch_s", 0.0))
        model_s = float(c.get("ttft_s", 0.0)) + float(c.get("duration_s", 0.0))
        if i + 1 < len(calls):
            t1 = float(calls[i + 1].get("ts_epoch_s", t0 + model_s))
        else:
            tail = [float(a.get("ts_epoch_s", 0)) + float(a.get("duration_s", 0))
                    for a in acts if float(a.get("ts_epoch_s", 0)) >= t0]
            t1 = max([t0 + model_s] + tail)
        total = max(t1 - t0, 0.0)
        model_s = min(model_s, total)

        tool_iv = [(float(a["ts_epoch_s"]),
                    float(a["ts_epoch_s"]) + float(a.get("duration_s", 0.0)))
                   for a in acts if "ts_epoch_s" in a]
        # non-overlapping split: model window is [t0, t0+model_s]; tool time
        # OUTSIDE it is tool_exec; tool time INSIDE it (harness runs a tool
        # while the response still streams) is reported separately and does
        # not double-count. model + tool_exec + other == total exactly.
        model_iv = [(t0, t0 + model_s)]
        covered = _union_seconds(model_iv + tool_iv, t0, t1)
        tool_union = _union_seconds(tool_iv, t0, t1)
        tool_excl = max(covered - model_s, 0.0)
        tool_during_model = max(tool_union - tool_excl, 0.0)

        gpu_s: float | None = None
        if kernel_intervals:
            trace_lo = kernel_intervals[0][0]
            trace_hi = kernel_intervals[-1][1]
            if trace_lo <= t1 and trace_hi >= t0:
                gpu_s = _union_seconds(kernel_intervals, t0, t1)
            # else: trace does not cover this window at all -> None

        other = max(total - covered, 0.0)
        turns.append({
            "turn": c.get("turn", i), "t_start_epoch_s": t0,
            "total_s": round(total, 4),
            "model_wait_s": round(model_s, 4),
            "tool_exec_s": round(tool_excl, 4),
            "tool_during_model_s": round(tool_during_model, 4),
            "gpu_busy_s": None if gpu_s is None else round(gpu_s, 4),
            "other_s": round(other, 4),
        })
    return turns


def summarize(turns: list[dict[str, Any]]) -> dict[str, Any]:
    tot = sum(t["total_s"] for t in turns)
    agg = {
        "n_turns": len(turns),
        "total_s": round(tot, 2),
        "model_wait_s": round(sum(t["model_wait_s"] for t in turns), 2),
        "tool_exec_s": round(sum(t["tool_exec_s"] for t in turns), 2),
        "tool_during_model_s": round(sum(t["tool_during_model_s"] for t in turns), 2),
        "gpu_busy_s": round(sum(t["gpu_busy_s"] or 0.0 for t in turns), 2),
        "gpu_covered_turns": sum(1 for t in turns if t["gpu_busy_s"] is not None),
        "other_s": round(sum(t["other_s"] for t in turns), 2),
    }
    if tot > 0:
        for k in ("model_wait_s", "tool_exec_s", "other_s"):
            agg[k.replace("_s", "_pct")] = round(100.0 * agg[k] / tot, 1)
    return agg


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-calls", required=True)
    p.add_argument("--actions", required=True)
    p.add_argument("--kernel-csv", default=None)
    p.add_argument("--task", default=None)
    p.add_argument("--out-csv", default=None)
    p.add_argument("--out-json", default=None)
    a = p.parse_args()

    kernels = _load_kernel_intervals(a.kernel_csv) if a.kernel_csv else None
    turns = decompose(_load_jsonl(a.model_calls), _load_jsonl(a.actions),
                      kernels, a.task)
    report = {"summary": summarize(turns), "turns": turns}

    if a.out_csv:
        cols = ["turn", "t_start_epoch_s", "total_s", "model_wait_s",
                "tool_exec_s", "tool_during_model_s", "gpu_busy_s", "other_s"]
        with open(a.out_csv, "w") as f:
            f.write(",".join(cols) + "\n")
            for t in turns:
                f.write(",".join("" if t[c] is None else str(t[c])
                                 for c in cols) + "\n")
    if a.out_json:
        with open(a.out_json, "w") as f:
            json.dump(report, f, indent=2)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
