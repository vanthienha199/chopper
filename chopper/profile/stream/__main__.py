"""Runnable demo, receiver, and self-test for the streaming pipeline.

  # Terminal A: a stand-in dashboard consumer that prints what it receives
  python -m chopper.profile.stream --serve --port 8900

  # Terminal B: stream synthetic GPU telemetry to it (no GPU needed)
  python -m chopper.profile.stream --fake --port 8900

  # One-shot end-to-end check (spins up both in-process, asserts data flows)
  python -m chopper.profile.stream --self-test
"""

import argparse
import json
import logging
import socket
import threading
import time
from typing import Any

from chopper.profile.stream.producer import StreamProducer
from chopper.profile.stream.sink import JsonLinesSocketSink
from chopper.profile.stream.source import FakeCounterSource, CsvTailSource


def _serve(host: str, port: int) -> None:
    """Minimal newline-delimited-JSON receiver (stands in for the dashboard)."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(1)
    print(f"[receiver] listening on {host}:{port} (ctrl-c to stop)")
    while True:
        conn, addr = srv.accept()
        print(f"[receiver] connection from {addr}")
        with conn, conn.makefile("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec["type"] == "gpu_counters":
                    m = rec["metrics"]
                    print(
                        f"gpu{rec['gpu']} t={rec['t_ns']/1e9:6.3f}s "
                        f"freq={m.get('gpu_freq_mhz', 0):7.1f}MHz "
                        f"util={m.get('tensor_util_pct', 0):5.1f}% "
                        f"rd={m.get('read_bw_gbs', 0):7.1f} "
                        f"wr={m.get('write_bw_gbs', 0):7.1f} GB/s"
                    )
                else:
                    print(f"gpu{rec['gpu']} kernel {rec['name']} "
                          f"dur={rec['dur_ns']/1e6:.2f}ms")


def _fake(host: str, port: int, duration_s: float, n_gpus: int, speed: float) -> None:
    sink = JsonLinesSocketSink(host=host, port=port)
    producer = StreamProducer(sink=sink, source="fake")
    src = FakeCounterSource(n_gpus=n_gpus, duration_s=duration_s)
    print(f"[producer] streaming {duration_s}s of fake telemetry to {host}:{port}")
    src.stream(producer, realtime=True, speed=speed)
    time.sleep(0.3)
    print(f"[producer] done, sink stats: {sink.stats}")
    producer.close()


def _follow(csv_path: str, gpu: int, host: str, port: int) -> None:
    """Follow a real device-sampler counter_samples.csv and stream it live."""
    sink = JsonLinesSocketSink(host=host, port=port)
    producer = StreamProducer(sink=sink, source="chopper")
    src = CsvTailSource(csv_path, gpu=gpu)
    print(f"[producer] following {csv_path} -> {host}:{port} (ctrl-c to stop)")
    try:
        for ev in src.events():
            _, g, t_ns, counters = ev
            producer.push_counter_sample(g, t_ns, counters)
    except KeyboardInterrupt:
        pass
    finally:
        print(f"[producer] stopping, sink stats: {sink.stats}")
        producer.close()


def _self_test() -> int:
    """In-process end-to-end: receiver thread + socket sink + fake source."""
    received: list[dict[str, Any]] = []
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]
    done = threading.Event()

    def accept_loop() -> None:
        srv.settimeout(5.0)
        try:
            conn, _ = srv.accept()
        except OSError:
            return
        with conn, conn.makefile("r") as f:
            for line in f:
                line = line.strip()
                if line:
                    received.append(json.loads(line))
                if done.is_set() and not line:
                    break

    t = threading.Thread(target=accept_loop, daemon=True)
    t.start()
    time.sleep(0.1)

    sink = JsonLinesSocketSink(host="127.0.0.1", port=port)
    producer = StreamProducer(sink=sink, source="selftest")
    src = FakeCounterSource(n_gpus=2, duration_s=2.0, sample_ms=2)
    src.stream(producer, realtime=False)
    time.sleep(0.5)
    done.set()
    producer.close()
    srv.close()
    t.join(timeout=2.0)

    counters = [r for r in received if r["type"] == "gpu_counters"]
    kernels = [r for r in received if r["type"] == "kernel_dispatch"]
    active = [r for r in counters if r["metrics"].get("tensor_util_pct", 0) > 1]

    print(f"[self-test] received {len(received)} records "
          f"({len(counters)} counters, {len(kernels)} kernels)")
    print(f"[self-test] sink stats: {sink.stats}")

    ok = True
    checks = [
        (len(counters) > 100, "got a substantial counter stream"),
        (len(kernels) >= 1, "got at least one kernel dispatch"),
        (len(active) > 10, "some samples show active-window GPU utilization"),
    ]
    for r in active[:1] + counters[:1]:
        m = r["metrics"]
        checks.append((0 < m.get("gpu_freq_mhz", 0) < 5000, "freq in a sane range"))
        break
    for passed, desc in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {desc}")
        ok = ok and passed

    if active:
        m = active[len(active) // 2]["metrics"]
        print(f"[self-test] sample active metrics: "
              f"freq={m.get('gpu_freq_mhz'):.0f}MHz util={m.get('tensor_util_pct'):.1f}% "
              f"read={m.get('read_bw_gbs'):.0f} write={m.get('write_bw_gbs'):.0f} GB/s")

    print(f"[self-test] {'OK' if ok else 'FAILED'}")
    return 0 if ok else 1


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--serve", action="store_true", help="run the receiver")
    p.add_argument("--fake", action="store_true", help="stream synthetic telemetry")
    p.add_argument("--follow", metavar="CSV",
                   help="follow a real device-sampler counter_samples.csv and stream it")
    p.add_argument("--self-test", action="store_true", help="in-process end-to-end check")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8900)
    p.add_argument("--duration", type=float, default=6.0)
    p.add_argument("--gpus", type=int, default=2)
    p.add_argument("--gpu", type=int, default=0, help="GPU/rank id for --follow")
    p.add_argument("--speed", type=float, default=1.0, help="realtime speed multiplier")
    args = p.parse_args()

    if args.self_test:
        return _self_test()
    if args.serve:
        _serve(args.host, args.port)
        return 0
    if args.fake:
        _fake(args.host, args.port, args.duration, args.gpus, args.speed)
        return 0
    if args.follow:
        _follow(args.follow, args.gpu, args.host, args.port)
        return 0
    p.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
