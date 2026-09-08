"""Convert a run's raw files into a chopper-agent-trace/0.1 bundle
(TRACE_FORMAT.md): manifest.json + OTel-compatible spans.jsonl, with the
telemetry files indexed in place. This is the converter the format doc
promises, so the proxy and adapters never have to change their output.

    python to_trace_bundle.py --run-id myrun \
        --model-calls model_calls.jsonl [--actions actions.jsonl] \
        [--kernel-csv kernel_traces.csv] [--anchor anchor.json] \
        [--telemetry name=path ...] --outdir bundle_dir

Spans: run -> turn -> (model_call | tool_exec). Times are epoch unix nano
(the proxy and adapters stamp epoch seconds). Attributes use the
namespaced agent.*/model.*/tool.* fields from the format doc.
"""

import argparse
import hashlib
import json
import os
import platform
import shutil


def _load_jsonl(path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _span_id(*parts) -> str:
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:16]


def _nano(epoch_s: float) -> int:
    return int(epoch_s * 1e9)


def build_spans(run_id, calls, actions):
    trace_id = hashlib.sha256(run_id.encode()).hexdigest()[:32]
    spans = []
    calls = sorted(calls, key=lambda c: c.get("ts_epoch_s", 0.0))
    acts = sorted(actions, key=lambda a: a.get("ts_epoch_s", 0.0))

    run_span = _span_id(run_id, "run")
    t_first = calls[0]["ts_epoch_s"] if calls else 0.0
    ends = [c["ts_epoch_s"] + c.get("ttft_s", 0) + c.get("duration_s", 0)
            for c in calls]
    ends += [a["ts_epoch_s"] + a.get("duration_s", 0) for a in acts]
    t_last = max(ends) if ends else t_first

    spans.append({
        "trace_id": trace_id, "span_id": run_span, "parent_span_id": None,
        "name": "run", "kind": "SPAN_KIND_INTERNAL",
        "start_time_unix_nano": _nano(t_first),
        "end_time_unix_nano": _nano(t_last),
        "status": {"code": "STATUS_CODE_OK"},
        "attributes": {"agent.run_id": run_id},
    })

    for i, c in enumerate(calls):
        t0 = c["ts_epoch_s"]
        t1 = (calls[i + 1]["ts_epoch_s"] if i + 1 < len(calls)
              else t_last)
        turn = c.get("turn", i)
        turn_span = _span_id(run_id, "turn", turn)
        spans.append({
            "trace_id": trace_id, "span_id": turn_span,
            "parent_span_id": run_span, "name": f"turn.{turn}",
            "kind": "SPAN_KIND_INTERNAL",
            "start_time_unix_nano": _nano(t0),
            "end_time_unix_nano": _nano(t1),
            "status": {"code": "STATUS_CODE_OK"},
            "attributes": {"agent.turn": turn,
                           "agent.id": c.get("task_id") or run_id},
        })
        model_end = t0 + c.get("ttft_s", 0) + c.get("duration_s", 0)
        spans.append({
            "trace_id": trace_id,
            "span_id": _span_id(run_id, "model", turn),
            "parent_span_id": turn_span, "name": "model_call",
            "kind": "SPAN_KIND_CLIENT",
            "start_time_unix_nano": _nano(t0),
            "end_time_unix_nano": _nano(min(model_end, t1) if model_end > t0 else t1),
            "status": {"code": "STATUS_CODE_OK"},
            "attributes": {
                "agent.turn": turn,
                "model.ttft_s": c.get("ttft_s", 0.0),
                "model.prompt_tokens": c.get("prompt_tokens", 0),
                "model.completion_tokens": c.get("completion_tokens", 0),
                "model.prompt_sha256": c.get("prompt_sha256", ""),
                "model.finish_reason": (c.get("finish_reasons") or [""])[0],
                "model.tool_calls": json.dumps(c.get("tool_calls") or []),
            },
        })

    for j, a in enumerate(acts):
        t0 = a["ts_epoch_s"]
        t1 = t0 + a.get("duration_s", 0.0)
        turn = a.get("after_turn", a.get("turn"))
        parent = (_span_id(run_id, "turn", turn)
                  if turn is not None else run_span)
        spans.append({
            "trace_id": trace_id,
            "span_id": _span_id(run_id, "tool", j),
            "parent_span_id": parent, "name": "tool_exec",
            "kind": "SPAN_KIND_INTERNAL",
            "start_time_unix_nano": _nano(t0),
            "end_time_unix_nano": _nano(t1),
            "status": {"code": "STATUS_CODE_OK" if a.get("exit_code", 0) == 0
                       else "STATUS_CODE_ERROR"},
            "attributes": {
                "agent.id": a.get("agent", ""),
                "tool.command": json.dumps(a.get("command", a.get("type", ""))),
                "tool.exit_code": a.get("exit_code"),
                "tool.cwd": a.get("cwd") or "",
            },
        })
    return spans


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-id", required=True)
    p.add_argument("--model-calls", required=True)
    p.add_argument("--actions", default=None)
    p.add_argument("--kernel-csv", default=None)
    p.add_argument("--anchor", default=None)
    p.add_argument("--telemetry", action="append", default=[],
                   metavar="NAME=PATH", help="e.g. gpu_telemetry=gpu.pkl")
    p.add_argument("--task", default=None)
    p.add_argument("--outdir", required=True)
    a = p.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    calls = _load_jsonl(a.model_calls)
    if a.task:
        calls = [c for c in calls if c.get("task_id") == a.task]
    actions = _load_jsonl(a.actions) if a.actions else []
    if a.task and actions:
        actions = [x for x in actions if x.get("task_id") == a.task]

    spans = build_spans(a.run_id, calls, actions)
    with open(f"{a.outdir}/spans.jsonl", "w") as f:
        for s in spans:
            f.write(json.dumps(s) + "\n")

    files = {"spans": "spans.jsonl"}
    shutil.copy(a.model_calls, f"{a.outdir}/model_calls.jsonl")
    files["replay/model_calls"] = "model_calls.jsonl"
    if a.kernel_csv:
        shutil.copy(a.kernel_csv, f"{a.outdir}/kernel_traces.csv")
        files["kernels"] = "kernel_traces.csv"
    for spec in a.telemetry:
        name, _, path = spec.partition("=")
        base = os.path.basename(path)
        shutil.copy(path, f"{a.outdir}/{base}")
        files[name] = base

    anchors = []
    if a.anchor:
        anc = json.load(open(a.anchor))
        anchors = [dict(anc, taken="start")]

    manifest = {
        "schema": "chopper-agent-trace/0.1",
        "run_id": a.run_id, "task_id": a.task,
        "host": {"hostname": platform.node(),
                 "system": platform.system(),
                 "machine": platform.machine()},
        "clocks": {"anchors": anchors, "gpu_clock": "cupti_or_rocprofiler"},
        "files": files,
        "encodings": {"spans": "jsonl", "kernels": "csv",
                      "telemetry": "pickle"},
    }
    with open(f"{a.outdir}/manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"bundle written to {a.outdir}: {len(spans)} spans "
          f"({len(calls)} model calls, {len(actions)} tool execs), "
          f"{len(files)} files indexed")


if __name__ == "__main__":
    main()
