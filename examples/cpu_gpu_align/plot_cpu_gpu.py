"""Plot the CPU+GPU timeline produced by merge.py --cpu-pkl/--kernel-csv.

Shows CPU utilization and GPU busy% on one shared-clock time axis (the view
Dr. Wu asked for: when GPU is busy the CPU is typically idle, and vice versa).

    python examples/cpu_gpu_align/plot_cpu_gpu.py --pkl timeline.pkl --out cpu_gpu.png
"""

import argparse
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pkl", required=True)
    p.add_argument("--out", default="cpu_gpu_timeline.png")
    a = p.parse_args()

    with open(a.pkl, "rb") as f:
        data = pickle.load(f)
    df = data["timeline"]
    corr = df["cpu_busy_pct"].corr(df["gpu_busy_pct"])

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(df["t_ms"] / 1000.0, df["gpu_busy_pct"], lw=1.4, color="#c1432b", label="GPU busy %")
    ax.plot(df["t_ms"] / 1000.0, df["cpu_busy_pct"], lw=1.4, color="#2b6ec1", label="CPU busy % (mean core)")
    ax.set_xlabel("time (s), shared rocprofiler clock")
    ax.set_ylabel("utilization (%)")
    ax.set_title(f"Chopper CPU+GPU on one timeline "
                 f"(clock={data.get('clock_domain')}, CPU-GPU corr={corr:.2f})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(a.out, dpi=140)
    print(f"saved {a.out} (corr={corr:.2f})")


if __name__ == "__main__":
    main()
