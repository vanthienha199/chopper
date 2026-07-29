"""Live consumer for the Chopper stream that renders a per-GPU busy timeline.

Binds a socket, receives newline-delimited JSON records from the streaming
layer, and (when the stream ends) saves a PNG of per-GPU GPU-busy% over time.
Stands in for a real dashboard for demo/screenshot purposes.

Usage:
    python examples/streaming/plot_consumer.py --port 8900 --out stream_live.png
"""

import argparse
import json
import socket
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8900)
    p.add_argument("--out", default="stream_live.png")
    p.add_argument("--sample-out", default="stream_sample.jsonl")
    a = p.parse_args()

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((a.host, a.port))
    srv.listen(1)
    print(f"[consumer] listening on {a.host}:{a.port}")

    busy = defaultdict(lambda: ([], []))  # gpu -> (t_s list, busy list)
    n_kernels = 0
    n_activity = 0
    source = "?"
    sample_lines = []

    conn, addr = srv.accept()
    print(f"[consumer] producer connected from {addr}")
    with conn, conn.makefile("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if len(sample_lines) < 12:
                sample_lines.append(line)
            rec = json.loads(line)
            source = rec.get("source", source)
            if rec["type"] == "gpu_activity":
                t_s = rec["t_ns"] / 1e9
                busy[rec["gpu"]][0].append(t_s)
                busy[rec["gpu"]][1].append(rec["busy_pct"])
                n_activity += 1
                if n_activity <= 12:
                    print(f"gpu{rec['gpu']} t={t_s:5.3f}s busy={rec['busy_pct']:5.1f}% "
                          f"kernels_in_bin={rec['n_kernels']}")
            elif rec["type"] == "kernel_dispatch":
                n_kernels += 1

    print(f"[consumer] received {n_activity} activity records, {n_kernels} kernels, "
          f"across {len(busy)} GPUs")

    with open(a.sample_out, "w") as sf:
        sf.write("\n".join(sample_lines) + "\n")

    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(11, 5.2))
    cmap = plt.get_cmap("tab10")
    for gpu in sorted(busy):
        ts, bs = busy[gpu]
        mean_b = sum(bs) / len(bs) if bs else 0
        ax.plot(ts, bs, lw=1.3, color=cmap(gpu % 10),
                label=f"GPU {gpu} (avg {mean_b:.0f}%)")
    ax.set_xlabel("time within iteration (s)")
    ax.set_ylabel("GPU busy (%)")
    ax.set_ylim(0, 105)
    ax.set_title(f"Chopper live stream: per-GPU busy%  (source: {source}, "
                 f"real MI300X trace replayed over socket)")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=4, fontsize=8, loc="lower center")
    fig.tight_layout()
    fig.savefig(a.out, dpi=140)
    print(f"[consumer] saved {a.out} and sample {a.sample_out}")


if __name__ == "__main__":
    main()
