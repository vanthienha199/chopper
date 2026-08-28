"""Auto-label agent turns as compute / memory / io bound (or idle), and
emit the per-harness fingerprint report Dr. Wu asked for: on this
benchmark, what fraction of the harness's turns (and of its time) falls in
each class.

Inputs per replay run:
  --counters-csv   per-turn instructions/ipc/cache (counters_per_turn)
  --io-csv         cumulative per-process io samples (telemetry.io)
  --journal        replay journal (turn windows, epoch)
  --anchor         epoch/monotonic anchor json

I/O per turn: cumulative read+write bytes of the harness's processes are
interpolated at each turn edge and differenced, same pattern as the HW
counters.

v1 labeling rule, applied in order (documented heuristic, thresholds are
CLI-tunable):
  idle           instructions < idle-insn AND io < idle-io
  io_bound       io_bytes >= io-abs OR io rate >= io-rate
  memory_bound   misses-per-kilo-instruction >= mpki AND ipc <= ipc-max
  compute_bound  everything else
"""

import argparse
import json

import numpy as np
import pandas as pd


def io_per_turn(io_df: pd.DataFrame, journal: list[dict],
                anchor: dict) -> list[int]:
    d = io_df[io_df["agent"] != "_system"].copy()
    d["bytes"] = d["read_bytes"] + d["write_bytes"]
    # cumulative per pid; sum across pids per sample time
    tot = d.groupby("ts")["bytes"].sum().reset_index()
    s = tot["ts"].to_numpy(dtype="float64")
    v = tot["bytes"].to_numpy(dtype="float64")

    def at(t_ns: float) -> float:
        if t_ns <= s[0]:
            return float(v[0])
        if t_ns >= s[-1]:
            return float(v[-1])
        i = int(np.searchsorted(s, t_ns))
        f = (t_ns - s[i - 1]) / (s[i] - s[i - 1])
        return float(v[i - 1] + f * (v[i] - v[i - 1]))

    def to_mono(epoch_s: float) -> float:
        return (epoch_s - anchor["epoch_s"]) * 1e9 + anchor["monotonic_ns"]

    calls = sorted(journal, key=lambda c: c["ts_epoch_s"])
    out = []
    for i, c in enumerate(calls):
        t0 = to_mono(c["ts_epoch_s"])
        t1 = to_mono(calls[i + 1]["ts_epoch_s"]) if i + 1 < len(calls) else s[-1]
        out.append(max(int(at(t1) - at(t0)), 0))
    return out


def label(row: pd.Series, idle_insn: float, idle_io: float, io_abs: float,
          io_rate: float, mpki_thr: float, ipc_max: float) -> str:
    if row["instructions"] < idle_insn and row["io_bytes"] < idle_io:
        return "idle"
    rate = row["io_bytes"] / row["window_s"] if row["window_s"] > 0 else 0
    if row["io_bytes"] >= io_abs or rate >= io_rate:
        return "io_bound"
    mpki = (1000.0 * row["cache_misses"] / row["instructions"]
            if row["instructions"] else 0.0)
    if mpki >= mpki_thr and row.get("ipc", 99) <= ipc_max:
        return "memory_bound"
    return "compute_bound"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--counters-csv", required=True)
    p.add_argument("--io-csv", required=True)
    p.add_argument("--journal", required=True)
    p.add_argument("--anchor", required=True)
    p.add_argument("--harness", default="harness")
    p.add_argument("--out-csv", default=None)
    p.add_argument("--idle-insn", type=float, default=100e6)
    p.add_argument("--idle-io", type=float, default=5e6)
    p.add_argument("--io-abs", type=float, default=50e6)
    p.add_argument("--io-rate", type=float, default=25e6)
    p.add_argument("--mpki", type=float, default=5.0)
    p.add_argument("--ipc-max", type=float, default=1.2)
    a = p.parse_args()

    df = pd.read_csv(a.counters_csv)
    io_df = pd.read_csv(a.io_csv)
    journal = [json.loads(l) for l in open(a.journal) if l.strip()]
    anchor = json.load(open(a.anchor))

    io_b = io_per_turn(io_df, journal, anchor)
    df["io_bytes"] = io_b[:len(df)]
    df["mpki"] = 1000.0 * df["cache_misses"] / df["instructions"].clip(lower=1)
    df["label"] = df.apply(label, axis=1, args=(
        a.idle_insn, a.idle_io, a.io_abs, a.io_rate, a.mpki, a.ipc_max))

    if a.out_csv:
        df.to_csv(a.out_csv, index=False)

    print(f"fingerprint for {a.harness} ({len(df)} turns):")
    tot_t = df["window_s"].sum()
    for lab in ("compute_bound", "memory_bound", "io_bound", "idle"):
        sub = df[df["label"] == lab]
        print(f"  {lab:>13}: {len(sub):>3} turns ({100*len(sub)/len(df):.0f}%), "
              f"{100*sub['window_s'].sum()/tot_t:.0f}% of time, "
              f"{sub['io_bytes'].sum()/1e6:.0f} MB io, "
              f"{sub['instructions'].sum()/1e9:.1f}B insn")


if __name__ == "__main__":
    main()
