# Real-time streaming (Chopper -> dashboard)

A thin, stdlib-only layer that turns Chopper's device counter samples into a
live stream of derived metrics (MFMA/tensor util, GPU clock, HBM read/write
bandwidth) pushed as newline-delimited JSON over TCP. The sink is
consumer-agnostic: any live dashboard can subscribe. Goal: watch a run in real
time instead of waiting for the offline `merge.py` step.

This is a prototype (Slice 1). It is additive: nothing here touches the
existing AMD collection path.

## Try it with no GPU

```
# Terminal A: a stand-in consumer that prints what it receives
python -m chopper.profile.stream --serve --port 8900

# Terminal B: stream synthetic active/idle GPU telemetry to it
python -m chopper.profile.stream --fake --port 8900
```

The synthetic source alternates active (kernel running) and idle windows, so
you can see clocks and bandwidth collapse during the idle windows, which is the
agentic-AI pattern (GPU idle during tool calls) the dashboard is meant to show.

One-shot end-to-end check (spins up both in-process, asserts data flows):

```
python -m chopper.profile.stream --self-test
```

## Pieces

- `metrics.py` : counter deltas -> instantaneous metrics. Same formulas as
  `common/rocm_metrics.py` / `plots/device_timeline.py`, on plain floats so the
  dashboard can reimplement them verbatim.
- `producer.py` : `StreamProducer` keeps the previous cumulative sample per GPU,
  diffs, derives, and emits a record per interval.
- `sink.py` : `JsonLinesSocketSink` ships records over TCP. Non-blocking and
  lossy under backpressure by design, so a profiled workload is never stalled
  by a slow or absent dashboard. Dropped counts are logged, never hidden.
- `source.py` : `FakeCounterSource` (GPU-free synthetic samples) and
  `CsvTailSource` (follow a real `counter_samples.csv` as the device sampler
  appends to it).

## Record schemas

```
gpu_counters:  {"type":"gpu_counters","source","gpu","t_ns","dt_ns",
                "metrics":{tensor_util_pct,gpu_freq_mhz,read_bw_gbs,write_bw_gbs,...},
                "counters":{raw per-interval deltas}}
kernel_dispatch:{"type":"kernel_dispatch","source","gpu","name",
                "start_ns","end_ns","dur_ns"}
```

## Wiring into a real run (next step, needs an AMD GPU)

Two tee-off points into the live device sampler
(`telemetry/src/device_counters_tool.cpp`):

1. Python side (what this prototype does): point `CsvTailSource` at the
   `counter_samples.csv` the sampler is writing and drive a `StreamProducer`.
   No C++ change; the sampler already writes rows as it goes.
2. C++ side (lower latency, later): emit each sample from the sampler's buffer
   callback over a socket in addition to the `std::ofstream` CSV write.

The `--fake` path exercises the whole producer/sink/consumer chain today so the
only remaining unknown is the real counter feed.
