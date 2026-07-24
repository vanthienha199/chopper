"""Per-core CPU + GPU timeline with labeled phases (addresses two meeting asks).

Fixes the "faint CPU line" problem by showing every CPU core instead of the mean
(Dr. Wu wanted per-core), and labels the GPU-compute vs CPU-tool phases on the
timeline (Marco's annotation idea) so a gap is readable instead of ambiguous.

Top panel: GPU-busy % over time, with each phase shaded and labeled.
Bottom panel: per-core CPU utilization heatmap on the same shared clock, so you
can literally see how many cores lit up during each phase.

    python examples/cpu_gpu_align/plot_percore_annotated.py \
        --cpu-pkl cpu.pkl --kernel-csv kernel_traces.csv --out percore.png
"""

import argparse

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def gpu_busy_per_bin(starts, ends, t0, bin_ns, n_bins):
    order = starts.argsort()
    s = (starts[order] - t0).astype("int64")
    e = (ends[order] - t0).astype("int64")
    merged = []
    cs, ce = int(s[0]), int(e[0])
    for i in range(1, len(s)):
        si, ei = int(s[i]), int(e[i])
        if si <= ce:
            ce = max(ce, ei)
        else:
            merged.append((cs, ce))
            cs, ce = si, ei
    merged.append((cs, ce))
    busy = np.zeros(n_bins)
    for cs, ce in merged:
        for b in range(max(0, cs // bin_ns), min(n_bins - 1, ce // bin_ns) + 1):
            lo, hi = b * bin_ns, b * bin_ns + bin_ns
            busy[b] += max(0, min(ce, hi) - max(cs, lo))
    return 100.0 * busy / bin_ns


def label_phases(gpu, bin_ms):
    """Return (start_s, end_s, kind) runs where kind is 'gpu' or 'cpu'."""
    active = gpu > 25
    runs = []
    i = 0
    n = len(active)
    while i < n:
        j = i
        while j < n and active[j] == active[i]:
            j += 1
        if (j - i) * bin_ms >= 120:   # ignore tiny blips
            runs.append((i * bin_ms / 1000.0, j * bin_ms / 1000.0,
                         "gpu" if active[i] else "cpu"))
        i = j
    return runs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cpu-pkl", required=True)
    p.add_argument("--kernel-csv", required=True)
    p.add_argument("--out", default="percore_annotated.png")
    p.add_argument("--bin-ms", type=int, default=20)
    a = p.parse_args()

    cpu = pd.read_pickle(a.cpu_pkl)
    k = pd.read_csv(a.kernel_csv)
    bin_ns = a.bin_ms * 1_000_000
    cpu_ts = cpu["ts"].to_numpy().astype("int64")
    starts = k["start_ns"].to_numpy().astype("int64")
    ends = k["end_ns"].to_numpy().astype("int64")
    t0 = int(min(cpu_ts.min(), starts.min()))
    t_end = int(max(cpu_ts.max(), ends.max()))
    n_bins = (t_end - t0) // bin_ns + 1

    gpu = gpu_busy_per_bin(starts, ends, t0, bin_ns, n_bins)

    # per-core CPU heatmap: cores x time-bin
    cpu = cpu.copy()
    cpu["bin"] = (cpu_ts - t0) // bin_ns
    piv = cpu.pivot_table(index="cpu", columns="bin", values="percent", aggfunc="mean")
    piv = piv.reindex(columns=range(n_bins))
    ncores = piv.shape[0]
    active_cores = (piv.fillna(0) > 20).sum(axis=0)  # cores busy per bin

    t = np.arange(n_bins) * a.bin_ms / 1000.0
    runs = label_phases(gpu, a.bin_ms)

    fig, (ax0, ax1) = plt.subplots(
        2, 1, figsize=(12, 6.5), sharex=True,
        gridspec_kw={"height_ratios": [1, 1.4], "hspace": 0.12})

    # top: GPU busy + phase labels
    ax0.plot(t, gpu, color="#c1432b", lw=1.6, label="GPU busy %")
    ax0.plot(t, active_cores / max(1, ncores) * 100, color="#2b6ec1", lw=1.2,
             label="CPU cores active (% of cores)")
    ax0.set_ylabel("utilization (%)")
    ax0.set_ylim(0, 105)
    ax0.legend(loc="upper right", fontsize=9)
    ax0.grid(True, alpha=0.3)
    for s, e, kind in runs:
        ax0.axvspan(s, e, color="#c1432b" if kind == "gpu" else "#2b6ec1", alpha=0.06)
        ax0.text((s + e) / 2, 96,
                 "GPU: model compute" if kind == "gpu" else "CPU: tool phase",
                 ha="center", va="top", fontsize=8,
                 color="#8a2f20" if kind == "gpu" else "#1f4e8a")
    ax0.set_title(f"CPU+GPU timeline on the shared clock  "
                  f"({ncores} logical cores, clock={cpu.attrs.get('clock_domain')})")

    # bottom: per-core heatmap
    im = ax1.imshow(piv.fillna(0).to_numpy(), aspect="auto", origin="lower",
                    extent=[0, t[-1], 0, ncores], cmap="viridis", vmin=0, vmax=100,
                    interpolation="nearest")
    ax1.set_ylabel("CPU core")
    ax1.set_xlabel("time (s), shared clock")
    cbar = fig.colorbar(im, ax=ax1, pad=0.01)
    cbar.set_label("per-core utilization (%)", fontsize=9)

    fig.tight_layout()
    fig.savefig(a.out, dpi=140)
    peak = int(active_cores.max())
    print(f"[percore] {ncores} cores, peak {peak} cores active at once, "
          f"{len(runs)} labeled phases -> {a.out}")


if __name__ == "__main__":
    main()
