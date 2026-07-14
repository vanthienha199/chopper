"""Validate the rocprofiler clock before collecting CPU data (Marco's step 1).

Confirms rocprofiler_get_timestamp is callable from Python, that it advances
monotonically, and measures its offset/drift vs time.monotonic_ns so we know how
the CPU-side clock relates to the GPU kernel-trace clock.

On a machine without ROCm this reports "unavailable" and exits 0. Run it on the
AMD cluster to actually validate:
    python examples/cpu_gpu_align/check_rocprofiler_clock.py
"""

import time

from chopper.profile.telemetry import rocprofiler_clock


def main() -> int:
    if not rocprofiler_clock.is_available():
        print("[check] rocprofiler clock UNAVAILABLE on this machine (no ROCm).")
        print("[check] Run this on the AMD cluster. On a ROCm box, set")
        print("        CHOPPER_ROCPROFILER_LIB if auto-discovery fails.")
        return 0

    print("[check] rocprofiler clock available. Sampling...")
    ok = True

    # 1. monotonic advance
    vals = []
    for _ in range(10):
        t = rocprofiler_clock.get_timestamp()
        assert t is not None
        vals.append(t)
        time.sleep(0.001)
    advancing = all(b >= a for a, b in zip(vals, vals[1:]))
    print(f"[check] 10 samples advance monotonically: {advancing}")
    print(f"[check] first={vals[0]} last={vals[-1]} span_ns={vals[-1]-vals[0]}")
    ok = ok and advancing

    # 2. offset + drift vs monotonic_ns (both nanoseconds)
    def pair():
        m = time.monotonic_ns()
        r = rocprofiler_clock.get_timestamp()
        assert r is not None
        return m, r

    m0, r0 = pair()
    time.sleep(1.0)
    m1, r1 = pair()
    off0 = r0 - m0
    off1 = r1 - m1
    print(f"[check] offset(rocprofiler - monotonic) start={off0} ns end={off1} ns")
    print(f"[check] drift over ~1s: {off1 - off0} ns "
          f"({(off1 - off0) / 1e9 * 100:.4f}% of 1s)")
    print("[check] (small stable offset = same-rate clocks, just a fixed shift)")

    # 3. reminder about the GPU side
    print("[check] Next: run examples/device_profiling/go.sh to confirm the GPU "
          "kernel traces use the same clock, then wire cpu.py with "
          "--cpu-clock rocprofiler.")

    print(f"[check] {'OK' if ok else 'FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
