"""Sample CPU utilization stamped with the CUPTI clock, into a cpu.pkl.

Runs concurrently with the CUDA workload so CPU% lands in the same clock domain
as the CUPTI kernel trace. Writes rows {cpu, ts, percent} matching cpu.py's
format so the timeline join works. Stops after --seconds.
"""
import sys
import time

import pandas as pd
import psutil

from chopper.profile.telemetry import cupti_clock

seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0
outpath = sys.argv[2] if len(sys.argv) > 2 else "cpu.pkl"

clock = cupti_clock.get_timestamp if cupti_clock.is_available() else time.monotonic_ns
domain = "cupti" if cupti_clock.is_available() else "monotonic_ns"
ncpu = psutil.cpu_count()

rows = []
t_end = time.monotonic() + seconds
psutil.cpu_percent(percpu=True)  # prime
while time.monotonic() < t_end:
    ts = clock()
    per = psutil.cpu_percent(percpu=True)
    for c in range(ncpu):
        rows.append({"cpu": c, "ts": ts, "percent": per[c]})
    time.sleep(0.03)

df = pd.DataFrame(rows)
df.attrs["clock_domain"] = domain
df.to_pickle(outpath)
print(f"[cpu-sampler] wrote {outpath}: {len(df)} rows, clock={domain}")
