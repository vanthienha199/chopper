# Unified Agent Trace Format (draft v0.1)

Goal: one directory per run that holds EVERYTHING needed to understand,
replay, and compare an agentic run across machines, with the CPU and GPU
sides on one timeline. Decisions from the 2026-08-19 discussion: the
semantic layer is OpenTelemetry-compatible (Dr. Wu), v1 targets single-node
runs (Dr. Wu); harness-side fields follow the proxy's schema_version 8 as
currently observed, to be re-checked against the stable field list.

## Design: two layers joined by clock anchors

Sparse EVENTS and dense SAMPLES have different shapes, so they get
different encodings:

1. **Semantic layer, `spans.jsonl`**: sparse events (model calls, tool
   executions, turns) as OTel-compatible spans. Existing OTel tooling can
   read this file as-is; our extra fields ride in namespaced attributes.
2. **Telemetry layer, columnar files**: dense periodic samples (power,
   per-core CPU, per-agent attribution, I/O, HW counters) and the kernel
   trace. Sampled data as spans would be bloat; these stay columnar.

Both layers resolve onto one timeline through the anchors in the manifest.

## Bundle layout

    run_<id>/
      manifest.json         run + host metadata, clock anchors, file index
      spans.jsonl           OTel spans: turns, model calls, tool executions
      kernel_traces.csv     per-kernel/graph-launch GPU intervals
      gpu_telemetry.parquet power/temp/clocks/util samples (10 Hz)
      cpu_samples.parquet   per-core utilization samples
      agents.parquet        per-process, per-agent attribution samples
      io.parquet            per-agent storage I/O counters
      cpu_counters.parquet  cumulative HW counters (instructions, cache, ...)
      initial_state/        workspace pin (repo, commit, container digest)
      replay/               model_calls.jsonl (+ journal when replayed)

Pickle is accepted where parquet is not available yet; the manifest names
the encoding per file. CSV stays for the kernel trace (tool compatibility).

## manifest.json

    {
      "schema": "chopper-agent-trace/0.1",
      "run_id": "...", "task_id": "...",
      "host": {"hostname": ..., "cpu": ..., "sockets": 2, "smt": false,
               "gpu": ..., "gpu_vendor": "amd|nvidia", "driver": ...},
      "clocks": {
        "anchors": [
          {"epoch_s": ..., "monotonic_ns": ..., "gpu_clock_ns": ...,
           "taken": "start"},
          {... "taken": "stop"}
        ],
        "gpu_clock": "rocprofiler|cupti"
      },
      "files": {"spans": "spans.jsonl", "kernels": "kernel_traces.csv", ...},
      "collectors": {"sampling_hz": {...}, "counters": [...]}
    }

Every file declares its clock domain; the anchors convert any domain to any
other. Single-node v1: one anchor set. Multi-node (v2) adds per-node
anchor sets and cross-node offset estimation.

## spans.jsonl (OTel-compatible)

One JSON object per line, standard OTel span fields: `trace_id`, `span_id`,
`parent_span_id`, `name`, `kind`, `start_time_unix_nano`,
`end_time_unix_nano`, `status`, `attributes`.

Span hierarchy: `run` -> `turn` -> (`model_call` | `tool_exec`).

Namespaced attributes (extension fields):

| attribute                | on span    | source (today)                  |
|--------------------------|------------|---------------------------------|
| `agent.id`               | all        | CHOPPER_AGENT_ID / harness name |
| `agent.turn`             | turn       | proxy `turn`                    |
| `model.ttft_s`           | model_call | proxy                           |
| `model.prompt_tokens`    | model_call | proxy                           |
| `model.completion_tokens`| model_call | proxy                           |
| `model.prompt_sha256`    | model_call | proxy                           |
| `model.finish_reason`    | model_call | proxy `finish_reasons[0]`       |
| `model.tool_calls`       | model_call | proxy (flat name/arguments/id)  |
| `tool.command`           | tool_exec  | adapter `actions.jsonl`         |
| `tool.exit_code`         | tool_exec  | adapter                         |
| `tool.cwd`               | tool_exec  | adapter                         |

Mapping from today's files: `model_calls.jsonl` row -> `model_call` span;
`actions.jsonl` row -> `tool_exec` span; turn windows (decompose.py) ->
`turn` spans. A converter ships with chopper, so Suhas's proxy does not
need to change its output format.

## Telemetry layer

Existing chopper collectors, unchanged semantics, one row schema each
(current pickles already match; parquet is a re-encoding):

| file          | key columns                                             |
|---------------|---------------------------------------------------------|
| gpu_telemetry | ts, gpu, power_w, temp_c, sclk, mclk, util              |
| cpu_samples   | ts, core, util_pct                                      |
| agents        | ts, agent, pid, name, cpu, percent, sweep_s             |
| io            | ts, agent, pid, read_bytes, write_bytes, counts         |
| cpu_counters  | ts, instructions, cpu_cycles, cache_references, misses  |
| kernel_traces | kernel_name, start_ns, end_ns, duration_ns              |

Kernel rows include `[cuda_graph N]` graph launches and `[nvtx_begin/end]`
operator range boundaries, so the kernel->operator mapping lives in the
same file.

## Why this shape (rationale for review)

- OTel spans make the agent-semantic layer readable by existing tools
  (Jaeger, Tempo, vendors) with zero work; the price is verbose field
  names, paid only on sparse events.
- Columnar telemetry keeps dense data compact and fast to scan; a 10 Hz
  power stream does not belong in spans.
- Clock anchors, not clock conversion at write time: every collector
  writes its native clock, conversion happens at read time. This is what
  already works in chopper on both AMD (rocprofiler clock) and NVIDIA
  (CUPTI clock) and is replay-proof.
- `initial_state/` + `replay/` in the bundle make every trace re-runnable
  by construction (the replay work shipped this month).

## Open questions (for the AMD experts + team)

1. DRAM traffic: user-space gives LLC-miss x 64B as a proxy; exact bytes
   need IMC/DataFabric counters (privileged). Is a privileged collector
   mode worth specifying in v0.2?
2. Parquet vs jsonl for telemetry: parquet halves size, jsonl greps. Ship
   both behind the manifest's encoding field, or pick one?
3. Span granularity for parallel tool fan-out (MCP servers): child spans
   per subprocess, or one span with a count attribute?
4. Multi-node (v2): per-node anchors + PTP offset estimation, or require a
   shared clock source?
