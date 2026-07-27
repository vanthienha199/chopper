"""Profiling overhead: how much does Chopper slow the workload down?

Runs the same workload N times bare and N times under Chopper's live
collectors (telemetry + kernel timeline, the "cheap" bucket), and reports
wall-time distributions and the slowdown. This quantifies the claim that the
live path is low overhead, and gives the on/off comparison Marco asked for.

    python examples/overhead/compare_onoff.py -n 5 -- python bench.py
    python examples/overhead/compare_onoff.py -n 5 \
        --collect-args "--gpu-telemetry --cpu-telemetry --device --cpu-clock rocprofiler" \
        -- python bench.py
"""

import argparse
import statistics
import subprocess
import sys
import time


_FAIL_MARKERS = ("Traceback (most recent call last)", "CalledProcessError",
                 "RuntimeError", "Segmentation fault")


def timed_runs(cmd: list, n: int, label: str) -> list:
    times = []
    for i in range(n):
        t0 = time.monotonic()
        r = subprocess.run(cmd, capture_output=True, text=True)
        dt = time.monotonic() - t0
        combined = r.stdout + r.stderr
        # collect.py's collector processes can swallow a workload crash and
        # still exit 0, which would count a broken run as a fast "success".
        # Treat any error marker in the output as a failed run.
        marker = next((m for m in _FAIL_MARKERS if m in combined), None)
        if r.returncode != 0 or marker is not None:
            why = f"exited {r.returncode}" if r.returncode != 0 else f"output contains {marker!r}"
            print(f"[overhead] {label} run {i} FAILED ({why}), excluding")
            sys.stdout.write(combined[-2000:])
            continue
        times.append(dt)
        print(f"[overhead] {label} run {i}: {dt:.3f}s")
    return times


def summarize(times: list) -> str:
    return (f"n={len(times)} median={statistics.median(times):.3f}s "
            f"mean={statistics.mean(times):.3f}s min={min(times):.3f}s max={max(times):.3f}s")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-n", type=int, default=5, help="runs per configuration")
    p.add_argument("--collect-args", default="--gpu-telemetry --cpu-telemetry",
                   help="args for chopper.profile.collect in the ON configuration")
    p.add_argument("--warmup", type=int, default=1,
                   help="untimed warmup runs before each configuration")
    p.add_argument("program", nargs="+", help="workload to run (after --)")
    a = p.parse_args()

    bare = list(a.program)
    profiled = [sys.executable, "-m", "chopper.profile.collect",
                *a.collect_args.split(), "--", *a.program]

    for _ in range(a.warmup):
        subprocess.run(bare, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    off = timed_runs(bare, a.n, "off")

    for _ in range(a.warmup):
        subprocess.run(profiled, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    on = timed_runs(profiled, a.n, "on ")

    if not off or not on:
        print("[overhead] not enough successful runs to compare")
        return 1

    m_off, m_on = statistics.median(off), statistics.median(on)
    print(f"\n[overhead] profiling OFF: {summarize(off)}")
    print(f"[overhead] profiling ON:  {summarize(on)}")
    print(f"[overhead] slowdown: {100.0 * (m_on - m_off) / m_off:+.2f}% "
          f"(median {m_off:.3f}s -> {m_on:.3f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
