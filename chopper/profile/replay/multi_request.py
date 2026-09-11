"""Multi-request metrics: N concurrent agent sessions on one serving backend.

The single-session metrics (decompose.py) assume one conversation owns the
backend. Under multi-request that assumption breaks in one specific way:
time to first token stops being prefill alone and becomes queue wait plus
prefill, and the interesting quantity becomes how much the sessions slow
each other down. This module reads one or more model_calls.jsonl files
written by record_proxy.py (schema 9, with session_id, turn_in_session and
inflight_at_arrival) and emits three CSVs.

Per call: session, turn in session, arrival, in-flight count at arrival,
observed ttft, the queue-wait and prefill split described below, service
duration, and the turn duration (this call's arrival to the next arrival in
the same session; the last turn of a session ends at its own response end).
Note on fields: record_proxy measures duration_s from request send to
response end, so duration_s already contains ttft_s, unlike the mock
endpoint's own journal where the two are disjoint. model_wait_s here is the
wall time of the call under either convention.

Per session: number of turns, total wall time, model wait, tool time. Tool
time is the part of each turn that is not model wait, which for a replay
driver is the emulated tool execution between calls.

Per concurrency level: p50, p95 and p99 of turn duration and of ttft, mean
in-flight count at arrival, interference factor, and a fairness ratio.
  - The interference factor is the mean turn duration at concurrency N
    divided by the mean turn duration at concurrency 1 for the same task,
    combined across tasks as a call-count weighted mean of the per-task
    ratios. A task present at only one of the two levels is skipped.
  - The fairness measure is the total wall time of the slowest session
    divided by that of the fastest session at the same level. This is a
    crude fairness proxy. It is sensitive to a single outlier session and
    it says nothing about the shape of the distribution between the two
    extremes. It is reported because it is cheap and it moves in the right
    direction, not because it is a principled fairness index.

The concurrency level of an input file is the concurrency_level field when
the records carry one, otherwise the number of distinct session_ids in that
file. One file per level is the expected layout.

WHAT THIS CANNOT DO. With client-side observation only, queue wait and
prefill cannot be separated inside a single call. The client sees one
number, the time to first token, and the backend does not tell it how that
number was spent. What this module reports as queue_wait_s is

    queue_wait_s = max(ttft at concurrency N
                       - median ttft at concurrency 1 for the same task and
                         prompt-length bucket, 0)
    prefill_s    = ttft - queue_wait_s

so the excess over the solo baseline is attributed to queueing. That
attribution assumes prefill cost itself does not change with batching,
which is only approximately true: a real engine batches prefill chunks
together, so per-request prefill time does move with load, and the
attribution then charges some genuine prefill slowdown to the queue. The
split is an estimate, not a measurement. A backend that reports its own
queue time, or a mock run with --max-concurrency (which journals the wait
it imposed), is what turns this into a checkable number.

    python -m chopper.profile.replay.multi_request \
        --model-calls n1.jsonl n2.jsonl n4.jsonl n8.jsonl \
        [--baseline n1.jsonl] [--out-calls calls.csv] \
        [--out-sessions sessions.csv] [--out-aggregate aggregate.csv]
"""

import argparse
import csv
import json
import statistics
from typing import Any

_BUCKET_EDGES = (512, 1024, 2048, 4096, 8192, 16384, 32768)


def _load_jsonl(path: str) -> list[dict[str, Any]]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def prompt_bucket(prompt_tokens: Any) -> str:
    """Coarse prompt-length bucket, used to match a call to a solo baseline."""
    try:
        n = int(prompt_tokens or 0)
    except (TypeError, ValueError):
        n = 0
    lo = 0
    for edge in _BUCKET_EDGES:
        if n < edge:
            return f"{lo}-{edge}"
        lo = edge
    return f"{lo}+"


def percentile(values: list[float], q: float) -> float:
    """Linear-interpolated percentile, q in [0, 100]. Empty input gives 0.0."""
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * (q / 100.0)
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return xs[lo] + (xs[hi] - xs[lo]) * frac


def level_of(records: list[dict[str, Any]]) -> int:
    """Concurrency level of one input file."""
    for r in records:
        if r.get("concurrency_level"):
            return int(r["concurrency_level"])
    return len({r.get("session_id") for r in records if r.get("session_id")})


def per_call(records: list[dict[str, Any]], level: int) -> list[dict[str, Any]]:
    """One row per model call, with turn duration inside its own session."""
    by_session: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        by_session.setdefault(str(r.get("session_id", "unknown")), []).append(r)

    rows: list[dict[str, Any]] = []
    for session, calls in by_session.items():
        calls.sort(key=lambda c: (float(c.get("ts_epoch_s", 0.0)),
                                  int(c.get("turn_in_session", 0))))
        for i, c in enumerate(calls):
            t0 = float(c.get("ts_epoch_s", 0.0))
            ttft = float(c.get("ttft_s", 0.0))
            dur = float(c.get("duration_s", 0.0))
            # record_proxy times duration_s from request send to response
            # end, so it already contains ttft_s. The mock endpoint's own
            # journal uses disjoint fields (sleep ttft, then pace duration).
            # Whichever produced the record, model wait is the wall time
            # from the request going out to the response completing.
            model_wait = dur if dur >= ttft else ttft + dur
            if i + 1 < len(calls):
                turn = max(float(calls[i + 1].get("ts_epoch_s", t0)) - t0, 0.0)
            else:
                turn = model_wait
            rows.append({
                "concurrency_level": level,
                "session_id": session,
                "turn_in_session": int(c.get("turn_in_session", i + 1)),
                "seq": int(c.get("seq", c.get("turn", i + 1))),
                "task_id": c.get("task_id"),
                "ts_epoch_s": round(t0, 6),
                "inflight_at_arrival": c.get("inflight_at_arrival"),
                "prompt_tokens": c.get("prompt_tokens", 0),
                "prompt_bucket": prompt_bucket(c.get("prompt_tokens")),
                "ttft_s": round(ttft, 6),
                "duration_s": round(dur, 6),
                "model_wait_s": round(model_wait, 6),
                "turn_duration_s": round(turn, 6),
                "tool_time_s": round(max(turn - model_wait, 0.0), 6),
                "queue_wait_s": None,
                "prefill_s": None,
                "baseline_source": "none",
            })
    rows.sort(key=lambda r: (r["session_id"], r["turn_in_session"]))
    return rows


def baseline_ttft(call_rows: list[dict[str, Any]]) -> dict[Any, float]:
    """Median solo ttft keyed by (task_id, prompt bucket), plus an overall key.

    Input rows must all come from a concurrency level of 1.
    """
    groups: dict[Any, list[float]] = {}
    for r in call_rows:
        groups.setdefault((r["task_id"], r["prompt_bucket"]), []).append(r["ttft_s"])
        groups.setdefault("__overall__", []).append(r["ttft_s"])
    return {k: statistics.median(v) for k, v in groups.items() if v}


def apply_baseline(call_rows: list[dict[str, Any]],
                   base: dict[Any, float]) -> None:
    """Fill queue_wait_s and prefill_s in place. See the module docstring for
    what the split assumes."""
    if not base:
        return
    for r in call_rows:
        key = (r["task_id"], r["prompt_bucket"])
        if key in base:
            ref, src = base[key], "bucket"
        elif "__overall__" in base:
            ref, src = base["__overall__"], "overall"
        else:
            continue
        queue = max(r["ttft_s"] - ref, 0.0)
        r["queue_wait_s"] = round(queue, 6)
        r["prefill_s"] = round(r["ttft_s"] - queue, 6)
        r["baseline_source"] = src


def per_session(call_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per session: turns, wall time, model wait, tool time."""
    by_session: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for r in call_rows:
        by_session.setdefault((r["concurrency_level"], r["session_id"]), []).append(r)

    out = []
    for (level, session), rows in sorted(by_session.items()):
        rows.sort(key=lambda r: r["ts_epoch_s"])
        first = rows[0]
        last = rows[-1]
        wall = (last["ts_epoch_s"] + last["model_wait_s"]) - first["ts_epoch_s"]
        out.append({
            "concurrency_level": level,
            "session_id": session,
            "task_id": first["task_id"],
            "n_turns": len(rows),
            "wall_s": round(max(wall, 0.0), 6),
            "model_wait_s": round(sum(r["model_wait_s"] for r in rows), 6),
            "tool_time_s": round(sum(r["tool_time_s"] for r in rows), 6),
            "mean_ttft_s": round(statistics.fmean(r["ttft_s"] for r in rows), 6),
        })
    return out


def _mean_turn_by_task(rows: list[dict[str, Any]]) -> dict[Any, float]:
    groups: dict[Any, list[float]] = {}
    for r in rows:
        groups.setdefault(r["task_id"], []).append(r["turn_duration_s"])
    return {k: statistics.fmean(v) for k, v in groups.items()}


def interference_factor(rows_at_level: list[dict[str, Any]],
                        rows_at_one: list[dict[str, Any]]) -> float | None:
    """Mean turn duration at this level over mean turn duration at level 1,
    per task, combined as a call-count weighted mean of the per-task ratios."""
    if not rows_at_level or not rows_at_one:
        return None
    solo = _mean_turn_by_task(rows_at_one)
    loaded = _mean_turn_by_task(rows_at_level)
    counts: dict[Any, int] = {}
    for r in rows_at_level:
        counts[r["task_id"]] = counts.get(r["task_id"], 0) + 1
    num, den = 0.0, 0
    for task, mean_loaded in loaded.items():
        if task not in solo or solo[task] <= 0:
            continue
        num += (mean_loaded / solo[task]) * counts[task]
        den += counts[task]
    return num / den if den else None


def aggregate(call_rows: list[dict[str, Any]],
              session_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per concurrency level."""
    levels = sorted({r["concurrency_level"] for r in call_rows})
    at_one = [r for r in call_rows if r["concurrency_level"] == 1]
    out = []
    for level in levels:
        rows = [r for r in call_rows if r["concurrency_level"] == level]
        sessions = [s for s in session_rows if s["concurrency_level"] == level]
        turns = [r["turn_duration_s"] for r in rows]
        ttfts = [r["ttft_s"] for r in rows]
        inflight = [r["inflight_at_arrival"] for r in rows
                    if r["inflight_at_arrival"] is not None]
        queues = [r["queue_wait_s"] for r in rows if r["queue_wait_s"] is not None]
        walls = [s["wall_s"] for s in sessions if s["wall_s"] > 0]
        fairness = max(walls) / min(walls) if len(walls) >= 2 else None
        out.append({
            "concurrency_level": level,
            "n_sessions": len(sessions),
            "n_calls": len(rows),
            "turn_p50_s": round(percentile(turns, 50), 6),
            "turn_p95_s": round(percentile(turns, 95), 6),
            "turn_p99_s": round(percentile(turns, 99), 6),
            "turn_mean_s": round(statistics.fmean(turns), 6) if turns else 0.0,
            "ttft_p50_s": round(percentile(ttfts, 50), 6),
            "ttft_p95_s": round(percentile(ttfts, 95), 6),
            "ttft_p99_s": round(percentile(ttfts, 99), 6),
            "mean_inflight_at_arrival":
                round(statistics.fmean(inflight), 4) if inflight else None,
            "mean_queue_wait_s":
                round(statistics.fmean(queues), 6) if queues else None,
            "interference_factor":
                _round_opt(interference_factor(rows, at_one), 4),
            "fairness_slowest_over_fastest": _round_opt(fairness, 4),
        })
    return out


def _round_opt(v: float | None, nd: int) -> float | None:
    return None if v is None else round(v, nd)


def _write_csv(path: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if v is None else v) for k, v in r.items()})


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-calls", nargs="+", required=True,
                   help="one model_calls.jsonl per concurrency level")
    p.add_argument("--baseline", default=None,
                   help="solo (concurrency 1) recording for the ttft "
                        "baseline; defaults to the level-1 input if present")
    p.add_argument("--out-calls", default=None)
    p.add_argument("--out-sessions", default=None)
    p.add_argument("--out-aggregate", default=None)
    a = p.parse_args()

    call_rows: list[dict[str, Any]] = []
    for path in a.model_calls:
        records = _load_jsonl(path)
        if not records:
            print(f"[multi-request] {path}: no records, skipped")
            continue
        call_rows.extend(per_call(records, level_of(records)))

    if a.baseline:
        base_rows = per_call(_load_jsonl(a.baseline), 1)
    else:
        base_rows = [r for r in call_rows if r["concurrency_level"] == 1]
    base = baseline_ttft(base_rows)
    if not base:
        print("[multi-request] no concurrency-1 baseline given or found: "
              "queue_wait_s and prefill_s are left empty")
    apply_baseline(call_rows, base)

    session_rows = per_session(call_rows)
    agg = aggregate(call_rows, session_rows)

    if a.out_calls:
        _write_csv(a.out_calls, call_rows)
    if a.out_sessions:
        _write_csv(a.out_sessions, session_rows)
    if a.out_aggregate:
        _write_csv(a.out_aggregate, agg)

    print(json.dumps(agg, indent=2))


if __name__ == "__main__":
    main()
