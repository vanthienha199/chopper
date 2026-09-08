"""Energy per agent turn from GPU power telemetry.

Integrates the power samples chopper already collects (gpu.pkl, ~10 Hz) over
turn boundaries, giving joules per turn: the "energy per turn" resource
efficiency metric. No new collection is needed, only turn boundary timestamps
joined onto the telemetry clock (the harness/proxy records turn start and end;
the clock anchors in the profile manifest convert them to the telemetry
domain).

Turn boundaries CSV: turn,start_ns,end_ns   (telemetry clock domain)

    python -m chopper.profile.replay.energy --gpu-pkl gpu.pkl \
        --turns turns.csv --out energy_per_turn.csv
"""

import argparse
from typing import Any

import pandas as pd


def _single_gpu_guard(gpu_df):
    # integrating an interleaved multi-GPU (or multi-node) series gives
    # meaningless joules; the caller must filter to one device first
    for col in ("gpu", "node"):
        if col in gpu_df.columns and gpu_df[col].nunique() > 1:
            raise ValueError(
                f"energy_per_turn needs a single-device series but got "
                f"{gpu_df[col].nunique()} distinct {col!r} values; "
                f"filter or groupby first")


def energy_per_turn(gpu: pd.DataFrame, turns: pd.DataFrame,
                    power_col: str, ts_col: str = "ts") -> pd.DataFrame:
    g = gpu.sort_values(ts_col)
    ts = g[ts_col].to_numpy().astype("int64")
    watts = g[power_col].to_numpy().astype("float64")
    rows: list[dict[str, Any]] = []
    for t in turns.itertuples(index=False):
        lo, hi = int(t.start_ns), int(t.end_ns)
        mask = (ts >= lo) & (ts <= hi)
        n = int(mask.sum())
        if n < 2:
            # fewer than two samples inside the turn: fall back to the mean of
            # the nearest samples times the turn duration, and say so
            near = watts[max(0, ts.searchsorted(lo) - 1):ts.searchsorted(hi) + 1]
            mean_w = float(near.mean()) if len(near) else 0.0
            joules = mean_w * (hi - lo) / 1e9
            method = "nearest-mean"
        else:
            seg_ts = ts[mask]
            seg_w = watts[mask]
            # trapezoidal integration over the samples inside the turn
            dt = (seg_ts[1:] - seg_ts[:-1]) / 1e9
            joules = float((0.5 * (seg_w[1:] + seg_w[:-1]) * dt).sum())
            mean_w = float(seg_w.mean())
            method = "trapezoid"
        rows.append({"turn": t.turn, "duration_s": (hi - lo) / 1e9,
                     "energy_j": round(joules, 3), "mean_power_w": round(mean_w, 2),
                     "samples": n, "method": method})
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpu-pkl", required=True)
    p.add_argument("--turns", required=True, help="csv: turn,start_ns,end_ns")
    p.add_argument("--power-col", default=None,
                   help="power column name (default: first column containing 'power')")
    p.add_argument("--out", default="energy_per_turn.csv")
    a = p.parse_args()

    gpu = pd.read_pickle(a.gpu_pkl)
    turns = pd.read_csv(a.turns)
    col = a.power_col
    if col is None:
        cands = [c for c in gpu.columns if "power" in str(c).lower()]
        assert cands, f"no power column found in {list(gpu.columns)}"
        col = cands[0]
    out = energy_per_turn(gpu, turns, col)
    out.to_csv(a.out, index=False)
    total = out["energy_j"].sum()
    print(f"[energy] {len(out)} turns, total {total:.1f} J, "
          f"mean {out['energy_j'].mean():.1f} J/turn -> {a.out}")


if __name__ == "__main__":
    main()
