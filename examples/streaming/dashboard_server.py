"""Live web dashboard for Chopper: watch CPU and GPU on one timeline in a browser.

Serves a self-contained HTML page plus a Server-Sent-Events stream, so a browser
shows CPU-busy and GPU-busy updating live as a run goes, instead of only a static
plot after the fact. Stdlib only (http.server), no external packages for the web
side.

Modes:
  --replay cpu.pkl kernel_traces.csv   stream a real run's CPU/GPU timeline live
  --fake                               synthetic active/idle demo (no data files)
  --listen PORT                        accept the streaming producer's JSON lines
                                       over TCP and rebroadcast to the browser

    python examples/streaming/dashboard_server.py --replay cpu.pkl kt.csv
    open http://127.0.0.1:8080
"""

import argparse
import json
import queue
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn

HERE = Path(__file__).parent
_subs: "set[queue.Queue]" = set()
_lock = threading.Lock()
_history: "list[str]" = []   # recent records, so a new page instantly shows history
_HIST_MAX = 260


def publish(rec: dict) -> None:
    data = json.dumps(rec)
    with _lock:
        _history.append(data)
        if len(_history) > _HIST_MAX:
            del _history[0]
        subs = list(_subs)
    for q in subs:
        try:
            q.put_nowait(data)
        except queue.Full:
            pass


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            body = (HERE / "dashboard.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/events":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            q: queue.Queue = queue.Queue(maxsize=2000)
            with _lock:
                for old in _history:      # backfill so the chart is populated at once
                    try:
                        q.put_nowait(old)
                    except queue.Full:
                        break
                _subs.add(q)
            try:
                while True:
                    try:
                        data = q.get(timeout=15)
                    except queue.Empty:
                        self.wfile.write(b": keepalive\n\n")
                        self.wfile.flush()
                        continue
                    self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            finally:
                with _lock:
                    _subs.discard(q)
        else:
            self.send_response(404)
            self.end_headers()


class Server(ThreadingMixIn, HTTPServer):
    daemon_threads = True


def _gpu_busy_series(starts, ends, t0, bin_ns, n_bins):
    import numpy as np
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


def replay_source(cpu_pkl, kernel_csv, bin_ms=50, tick_ms=60):
    """Build the CPU/GPU timeline from a real run, then publish it live, looping."""
    import numpy as np
    import pandas as pd

    cpu = pd.read_pickle(cpu_pkl)
    k = pd.read_csv(kernel_csv)
    bin_ns = bin_ms * 1_000_000
    cpu_ts = cpu["ts"].to_numpy().astype("int64")
    starts = k["start_ns"].to_numpy().astype("int64")
    ends = k["end_ns"].to_numpy().astype("int64")
    ngpu = int(cpu["cpu"].nunique() and 8)  # display hint only
    t0 = int(min(cpu_ts.min(), starts.min()))
    t_end = int(max(cpu_ts.max(), ends.max()))
    n_bins = (t_end - t0) // bin_ns + 1
    cpu = cpu.copy()
    cpu["bin"] = (cpu_ts - t0) // bin_ns
    cpu_busy = cpu.groupby("bin")["percent"].mean()
    gpu = _gpu_busy_series(starts, ends, t0, bin_ns, n_bins)
    cpu_series = np.array([float(cpu_busy.get(b, 0.0)) for b in range(n_bins)])
    print(f"[dashboard] replaying {n_bins} bins from a real run "
          f"(clock={cpu.attrs.get('clock_domain')})")

    def run():
        while True:
            for b in range(n_bins):
                publish({"type": "tl", "t": round(b * bin_ms / 1000.0, 3),
                         "cpu": round(cpu_series[b], 1), "gpu": round(float(gpu[b]), 1),
                         "src": "real MI210 run"})
                time.sleep(tick_ms / 1000.0)
            time.sleep(0.8)
    threading.Thread(target=run, daemon=True).start()


def fake_source(tick_ms=60):
    """Synthetic anti-correlated CPU/GPU demo, no data files needed."""
    import math

    def run():
        t = 0.0
        while True:
            phase = (t % 2.4)
            gpu = 78 if phase < 1.2 else 2
            cpu = 4 if phase < 1.2 else 82
            gpu += 6 * math.sin(t * 7)
            cpu += 8 * math.sin(t * 9)
            publish({"type": "tl", "t": round(t, 2),
                     "cpu": round(max(0, cpu), 1), "gpu": round(max(0, gpu), 1),
                     "src": "synthetic demo"})
            t += tick_ms / 1000.0
            time.sleep(tick_ms / 1000.0)
    threading.Thread(target=run, daemon=True).start()


def listen_source(port):
    """Accept the streaming producer's newline-delimited JSON over TCP."""
    def run():
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("127.0.0.1", port))
        srv.listen(4)
        print(f"[dashboard] listening for producer JSON on tcp/{port}")
        while True:
            conn, _ = srv.accept()
            with conn, conn.makefile("r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            publish(json.loads(line))
                        except json.JSONDecodeError:
                            pass
    threading.Thread(target=run, daemon=True).start()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replay", nargs=2, metavar=("CPU_PKL", "KERNEL_CSV"))
    p.add_argument("--fake", action="store_true")
    p.add_argument("--listen", type=int, metavar="TCP_PORT")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    a = p.parse_args()

    if a.replay:
        replay_source(a.replay[0], a.replay[1])
    elif a.listen:
        listen_source(a.listen)
    else:
        fake_source()

    httpd = Server((a.host, a.port), Handler)
    print(f"[dashboard] open http://{a.host}:{a.port}")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
