"""Build + plot the NVIDIA CPU+GPU timeline from cpu.pkl + kernel_traces.csv.

Both are on the CUPTI clock, so this is the NVIDIA equivalent of the AMD
merge_cpu_gpu_timeline demo: per time bin, mean CPU utilization and GPU busy%
(union of kernel intervals). Shows the CPU/GPU trade-off on one shared clock.
"""
import os
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

cpu_pkl, kernel_csv, out_png = sys.argv[1], sys.argv[2], sys.argv[3]
bin_ms = int(sys.argv[4]) if len(sys.argv) > 4 else 50
bin_ns = bin_ms * 1_000_000

cpu = pd.read_pickle(cpu_pkl)
k = pd.read_csv(kernel_csv)
cpu_ts = cpu["ts"].to_numpy().astype("int64")
starts = k["start_ns"].to_numpy().astype("int64")
ends = k["end_ns"].to_numpy().astype("int64")

t0 = int(min(cpu_ts.min(), starts.min()))
t_end = int(max(cpu_ts.max(), ends.max()))
n_bins = (t_end - t0) // bin_ns + 1

cpu = cpu.copy()
cpu["bin"] = (cpu_ts - t0) // bin_ns
cpu_busy = cpu.groupby("bin")["percent"].mean()

# GPU busy% per bin via union of kernel intervals
order = starts.argsort()
s = starts[order] - t0
e = ends[order] - t0
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
gpu = np.zeros(n_bins)
for cs, ce in merged:
    for b in range(max(0, cs // bin_ns), min(n_bins - 1, ce // bin_ns) + 1):
        lo, hi = b * bin_ns, b * bin_ns + bin_ns
        gpu[b] += max(0, min(ce, hi) - max(cs, lo))
gpu = 100.0 * gpu / bin_ns

t = np.arange(n_bins) * bin_ms / 1000.0
cpu_series = np.array([cpu_busy.get(b, 0.0) for b in range(n_bins)])
corr = np.corrcoef(cpu_series, gpu)[0, 1] if n_bins > 2 else 0.0
dom = cpu.attrs.get("clock_domain")
print(f"[nv-timeline] {n_bins} bins, clock={dom}, "
      f"mean CPU={cpu_series.mean():.1f}% GPU={gpu.mean():.1f}% corr={corr:.2f}, "
      f"{len(k)} kernels")

fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(t, gpu, lw=1.4, color="#76b900", label="GPU busy %")   # nvidia green
ax.plot(t, cpu_series, lw=1.4, color="#2b6ec1", label="CPU busy % (mean core)")
ax.set_xlabel("time (s), shared CUPTI clock")
ax.set_ylabel("utilization (%)")
device = os.environ.get("CHOPPER_NV_DEVICE", "NVIDIA GPU")
ax.set_title(f"Chopper CPU+GPU on one timeline, {device} "
             f"(clock={dom}, CPU-GPU corr={corr:.2f})")
ax.grid(True, alpha=0.3)
ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig(out_png, dpi=140)
print(f"[nv-timeline] saved {out_png}")
