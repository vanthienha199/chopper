"""Instructions / cache traffic per agent turn (Samsung resource-efficiency
metrics, same pattern as energy per turn).

Inputs:
  --counters   cpu_counters.pkl from telemetry.cpu_counters.measure_command
               (cumulative user-space HW counters of the harness process
               tree, monotonic-clock stamps)
  --journal    the mock endpoint's replay journal (replay-time epoch stamps
               per model call = turn boundaries on THIS machine)
  --anchor     json with {"epoch_s": ..., "monotonic_ns": ...} captured
               back-to-back on the same machine, to convert journal epochs
               onto the counter clock

Per turn (model call k arrival -> call k+1 arrival) the cumulative counters
are linearly interpolated at both edges and differenced:
  instructions, cpu_cycles, cache_references, cache_misses, ipc,
  dram_bytes_proxy = cache_misses * 64
The DRAM column is an LLC-miss-traffic PROXY (64B lines): exact DRAM bytes
need uncore IMC counters, which require elevated privilege (same class of
wall as GPU PMC access). Documented fallback per the metric layer doc.

    python -m chopper.profile.replay.counters_per_turn \
        --counters cpu_counters.pkl --journal replay_calls.jsonl \
        --anchor anchor.json --out-csv turns_counters.csv
"""

import argparse
import json

import pandas as pd


def _interp(df: pd.DataFrame, col: str, t_ns: float) -> float:
    s = df["ts"].to_numpy()
    v = df[col].to_numpy(dtype="float64")
    if t_ns <= s[0]:
        return float(v[0])
    if t_ns >= s[-1]:
        return float(v[-1])
    import numpy as np
    i = int(np.searchsorted(s, t_ns))
    f = (t_ns - s[i - 1]) / (s[i] - s[i - 1])
    return float(v[i - 1] + f * (v[i] - v[i - 1]))


def per_turn(counters: pd.DataFrame, journal: list[dict],
             anchor: dict) -> list[dict]:
    cdf = counters.dropna().reset_index(drop=True)
    cols = [c for c in cdf.columns if c != "ts"]

    def epoch_to_mono_ns(epoch_s: float) -> float:
        return (epoch_s - anchor["epoch_s"]) * 1e9 + anchor["monotonic_ns"]

    calls = sorted(journal, key=lambda c: c["ts_epoch_s"])
    rows = []
    for i, c in enumerate(calls):
        t0 = epoch_to_mono_ns(c["ts_epoch_s"])
        if i + 1 < len(calls):
            t1 = epoch_to_mono_ns(calls[i + 1]["ts_epoch_s"])
        else:
            t1 = float(cdf["ts"].iloc[-1])
        row = {"turn": c.get("turn", i),
               "window_s": round((t1 - t0) / 1e9, 4)}
        for col in cols:
            row[col] = int(_interp(cdf, col, t1) - _interp(cdf, col, t0))
        if row.get("cpu_cycles"):
            row["ipc"] = round(row.get("instructions", 0)
                               / row["cpu_cycles"], 3)
        if "cache_misses" in row:
            row["dram_bytes_proxy"] = row["cache_misses"] * 64
        rows.append(row)
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--counters", required=True)
    p.add_argument("--journal", required=True)
    p.add_argument("--anchor", required=True)
    p.add_argument("--out-csv", default=None)
    a = p.parse_args()

    counters = pd.read_pickle(a.counters)
    journal = [json.loads(l) for l in open(a.journal) if l.strip()]
    anchor = json.load(open(a.anchor))
    rows = per_turn(counters, journal, anchor)

    df = pd.DataFrame(rows)
    if a.out_csv:
        df.to_csv(a.out_csv, index=False)
    tot_i = df.get("instructions")
    print(f"{len(df)} turns; total instructions "
          f"{int(tot_i.sum()) if tot_i is not None else 'n/a'}; "
          f"median/turn {int(tot_i.median()) if tot_i is not None else 'n/a'}")
    if "dram_bytes_proxy" in df:
        print(f"DRAM-proxy total {df['dram_bytes_proxy'].sum() / 1e9:.2f} GB "
              f"(LLC-miss x 64B, see module docstring)")


if __name__ == "__main__":
    main()
