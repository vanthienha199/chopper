"""Single-command benchmark runner (Samsung artifact: ./run-benchmark).

One command executes a replay-based benchmark on THIS machine and emits a
machine-readable report plus a human summary:

  1. clock anchor (epoch + monotonic, captured back to back)
  2. mock endpoint serving the recorded run (FIFO, recorded timing,
     replay journal on)
  3. best-effort collectors: CPU hardware counters wrap the harness tree
     when the platform supports them (Linux perf_event_open); collectors
     that are unavailable are SKIPPED and the skip is recorded in the
     report, never silently
  4. the harness command runs against the mock in the given workspace
  5. metrics: per-turn time decomposition from the replay journal (plus
     tool splits when an actions log is provided) and per-turn hardware
     counters when step 3 ran
  6. report.json (machine-readable) + summary.txt (human-readable)

    python run_benchmark.py --recording model_calls.jsonl \
        --task astropy__astropy-12907 --workspace ./workspace \
        --harness-cmd "mini -y -l 0 -m openai/gpt-oss-120b -t \"$(cat p.txt)\"" \
        --outdir run_out [--speed 1] [--port 8123] [--timeout 600]

The harness command runs with OPENAI_BASE_URL/OPENAI_API_BASE pointed at
the mock and MSWEA_* set for mini-swe-agent; pass extra env as --env K=V.
"""

import argparse
import json
import os
import platform
import shlex
import socket
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load_sibling(name: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(_HERE, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--recording", required=True)
    p.add_argument("--task", default=None)
    p.add_argument("--workspace", required=True)
    p.add_argument("--harness-cmd", required=True)
    p.add_argument("--actions", default=None,
                   help="harness adapter actions.jsonl produced during the "
                        "run, for the tool split (optional)")
    p.add_argument("--outdir", default="benchmark_out")
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--port", type=int, default=8123)
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument("--env", action="append", default=[], metavar="K=V")
    a = p.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    outdir = os.path.abspath(a.outdir)
    skips: list[str] = []

    anchor = {"epoch_s": time.time(), "monotonic_ns": time.monotonic_ns()}
    with open(f"{outdir}/anchor.json", "w") as f:
        json.dump(anchor, f)

    journal = f"{outdir}/replay_calls.jsonl"
    mock_cmd = [sys.executable, os.path.join(_HERE, "mock_endpoint.py"),
                "--recording", os.path.abspath(a.recording),
                "--port", str(a.port), "--speed", str(a.speed),
                "--journal", journal]
    if a.task:
        mock_cmd += ["--task", a.task]
    mock_log = open(f"{outdir}/mock.log", "w")
    mock = subprocess.Popen(mock_cmd, stdout=mock_log, stderr=subprocess.STDOUT)

    for _ in range(50):
        try:
            socket.create_connection(("127.0.0.1", a.port), timeout=0.2).close()
            break
        except OSError:
            time.sleep(0.1)

    env = dict(os.environ)
    env.update({
        "OPENAI_API_KEY": "dummy",
        "OPENAI_BASE_URL": f"http://127.0.0.1:{a.port}/v1",
        "OPENAI_API_BASE": f"http://127.0.0.1:{a.port}/v1",
        "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{a.port}",
        "MSWEA_CONFIGURED": "true",
        "MSWEA_COST_TRACKING": "ignore_errors",
    })
    for kv in a.env:
        k, _, v = kv.partition("=")
        env[k] = v

    counters_pkl = f"{outdir}/cpu_counters.pkl"
    use_counters = False
    cpu_counters = None
    try:
        sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))
        from chopper.profile.telemetry import cpu_counters as _cc
        cpu_counters = _cc
        use_counters = _cc.is_available()
        if not use_counters:
            skips.append("cpu_counters: platform lacks perf_event_open "
                         f"({platform.system()}/{platform.machine()})")
    except ImportError as e:
        skips.append(f"cpu_counters: import failed ({e})")

    harness = shlex.split(a.harness_cmd)
    t0 = time.time()
    rc: int | None = None
    old_cwd = os.getcwd()
    os.chdir(a.workspace)
    os.environ.update(env)  # measure_command spawns with the current env
    try:
        if use_counters:
            cpu_counters.measure_command(
                harness, filename="cpu_counters.pkl", outdir=outdir, off=0.05)
            rc = 0  # measure_command waits for the workload itself
        else:
            proc = subprocess.Popen(harness, env=env)
            try:
                rc = proc.wait(timeout=a.timeout)
            except subprocess.TimeoutExpired:
                proc.terminate()
                rc = -1
                skips.append(f"harness: hit --timeout {a.timeout}s, terminated")
    finally:
        os.chdir(old_cwd)
    wall_s = time.time() - t0

    mock.send_signal(2)
    time.sleep(0.5)
    mock.terminate()
    mock_log.close()

    dec = _load_sibling("decompose")
    calls = dec._load_jsonl(journal) if os.path.exists(journal) else []
    acts = dec._load_jsonl(a.actions) if a.actions else []
    if not a.actions:
        skips.append("actions: no adapter log given, tool split unavailable "
                     "(tool time lands in 'other')")
    turns = dec.decompose(calls, acts, None, None)
    summary = dec.summarize(turns) if turns else {}

    counters_summary = None
    if use_counters and os.path.exists(counters_pkl) and calls:
        try:
            cpt = _load_sibling("counters_per_turn")
            import pandas as pd
            rows = cpt.per_turn(pd.read_pickle(counters_pkl), calls, anchor)
            pd.DataFrame(rows).to_csv(f"{outdir}/turns_counters.csv", index=False)
            tot = sum(r.get("instructions", 0) for r in rows)
            counters_summary = {
                "turns": len(rows), "instructions_total": tot,
                "instructions_median": sorted(
                    r.get("instructions", 0) for r in rows)[len(rows) // 2],
                "dram_bytes_proxy_total": sum(
                    r.get("dram_bytes_proxy", 0) for r in rows)}
        except Exception as e:  # report, never crash the report step
            skips.append(f"counters_per_turn: {e}")

    report = {
        "schema": "chopper-run-benchmark/0.1",
        "host": {"hostname": platform.node(), "system": platform.system(),
                 "machine": platform.machine(),
                 "cpu_count": os.cpu_count()},
        "recording": os.path.abspath(a.recording), "task": a.task,
        "harness_cmd": a.harness_cmd, "harness_rc": rc,
        "wall_s": round(wall_s, 2),
        "calls_served": len(calls),
        "turn_decomposition": summary,
        "counters": counters_summary,
        "skipped_collectors": skips,
        "files": {"journal": "replay_calls.jsonl", "anchor": "anchor.json",
                  "mock_log": "mock.log",
                  **({"counters_csv": "turns_counters.csv"}
                     if counters_summary else {})},
    }
    with open(f"{outdir}/report.json", "w") as f:
        json.dump(report, f, indent=2)

    lines = [
        "run-benchmark summary",
        f"  host: {report['host']['hostname']} "
        f"({report['host']['system']}/{report['host']['machine']})",
        f"  harness: {a.harness_cmd[:70]} (rc={rc})",
        f"  wall time: {wall_s:.1f}s, model calls served: {len(calls)}",
    ]
    if summary:
        lines.append(
            f"  per-turn split: model {summary.get('model_wait_pct', '?')}% / "
            f"tool {summary.get('tool_exec_pct', '?')}% / "
            f"other {summary.get('other_pct', '?')}% over "
            f"{summary.get('n_turns')} turns")
    if counters_summary:
        lines.append(
            f"  instructions: {counters_summary['instructions_total']/1e9:.1f}B "
            f"total, median/turn "
            f"{counters_summary['instructions_median']/1e6:.0f}M")
    for s in skips:
        lines.append(f"  skipped: {s}")
    text = "\n".join(lines)
    with open(f"{outdir}/summary.txt", "w") as f:
        f.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
