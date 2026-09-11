# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Chopper is a GPU characterization tool for analyzing multi-GPU performance during distributed deep learning workloads. It processes Chrome traces (from PyTorch profiler), AMD ROCm hardware counters, and telemetry data to generate performance visualizations.

## Commands

### Installation
```bash
pip install .
```

### Profiling
```bash
# Basic profiling with GPU and CPU telemetry
python -m chopper.profile.collect --gpu-telemetry --cpu-telemetry -- <workload args>

# Per-agent CPU attribution and per-agent storage I/O. Both find their subjects
# by looking for processes tagged with CHOPPER_AGENT_ID, so tag the agent or the
# output will be empty. The collector warns when the tag is missing.
CHOPPER_AGENT_ID=my-agent python -m chopper.profile.collect \
    --cpu-telemetry --agent-telemetry --io-telemetry -- <workload args>

# With hardware counters (slower)
python -m chopper.profile.collect --gpu-telemetry --cpu-telemetry --counters COUNTER1 COUNTER2 -- <workload args>

# Using container for workload isolation
python -m chopper.profile.collect --container /path/to/image.sif --gpu-telemetry -- <workload args>
```

### Visualization
```bash
# Launch PyQt6 GUI for interactive plot exploration
python -m chopper.window
```

### Testing
```bash
# Run mypy type checking with strict mode
python -m tests
```

## Architecture

### Core Modules

**profile/** - Data collection pipeline
- `runner.py`: Multiprocess orchestrator using cooperative shutdown pattern. One process is designated as "stop_when_complete" to trigger shutdown of all collectors.
- `collect.py`: CLI entry point that orchestrates parallel collection of traces, counters, and telemetry.
- `merge.py`: Complex three-stage pipeline that parses JSON Chrome traces, correlates CPU operations to GPU kernels, and merges hardware counter data by kernel name + instance index. Outputs unified pickle file (`ts.pkl`).
- `telemetry/`: Parallel data collectors (cpu.py, gpu.py, counters.py) that run concurrently and write to pickle files.

**common/** - Shared data processing utilities
- `load.py`: Primary API for loading and transforming trace data. Use `get_df()` for basic loading with filtering/aggregation. Contains specialized aggregations: `get_straggler_df()` for GPU load imbalance, `get_overlap_df()` for communication-computation overlap.
- `annotations.py`: FSDP-aware semantic analysis. Handles differences between FSDPv1 and FSDPv2. Key functions: `no_overlap_mask()` filters to compute-only ops, `assign_chunks()` categorizes as fwd/bwd/opt.
- `trace_metrics.py`: Derives timing metrics (launch overhead, prep overhead, overlap CDF) from trace timestamps.
- `rocm_metrics.py`: 1000+ lines of AMD hardware counter formulas. Derives high-level metrics (MFMA util, cache hit rates, memory bandwidth) from raw counters.

**plots/** - Plugin-based visualizations
- Each plot module follows strict contract: `get_data(**kwargs)` loads data, `draw(fig, input_data, **kwargs)` renders to matplotlib Figure.
- Type annotations on kwargs drive automatic UI generation in window.py.
- Separates expensive data loading from fast visualization tweaking.

**window.py** - Interactive PyQt6 GUI
- Uses Python introspection to discover plot modules and generate UI dynamically.
- Maps type annotations to widgets: `list[str]` → StrlistSelection, `bool` → checkbox, etc.
- Caches loaded data to enable fast re-rendering with different draw parameters.

### Data Flow

1. **Profiling**: Parallel collection of PyTorch traces (JSON), ROCm counters (CSV), and telemetry (pickle).
2. **Merging**: `merge.py` parses traces, links CPU ops to GPU kernels via correlation IDs, joins counter data by kernel name + instance.
3. **Analysis**: `load.py` loads unified DataFrame, applies transformations (iteration filtering, chunk assignment, aggregation), computes derived metrics.
4. **Visualization**: GUI introspects plot signatures, generates parameter widgets, calls `get_data()` once, then allows rapid `draw()` calls with different params.

### Key Patterns

**Cooperative Shutdown**: `runner.py` uses shared `c_bool` flag. One collector (typically `counters.py`) is marked as stop_when_complete; when it finishes, all others terminate.

**Type Annotation-Driven UI**: Window introspects function signatures to map type hints to PyQt6 widgets. Add a new plot parameter and the UI automatically updates.

**Lazy Data Loading**: `get_data()` is expensive (loads traces, aggregates). `draw()` is cheap (just matplotlib). Cache separates these concerns.

**Framework-Specific Filtering**: Code handles FSDPv1 vs FSDPv2 differences in distributed training patterns. Critical for accurate straggler analysis.

**Binary Search for Hierarchical Context**: `merge.py` uses `np.searchsorted()` to efficiently map subordinate CPU operations to parent context (O(n log n) instead of O(n²)).

### Chrome Trace Event Relationships

- `user_annotation`: Layer/iteration/operator markers with time ranges
- `cpu_op`: CPU-side PyTorch operations with external_id
- `cuda_runtime`: CUDA API calls with correlation ID
- `kernel`: GPU kernel execution linked via external_id and correlation
- `fwdbwd`: Forward/backward pass markers

Merge sequence: cuda_runtime → kernel → cpu_op → user_annotation → hardware counters.

### Hardware Counter Processing

1. `counters.py` batches counters (max 3 per rocprofv3 run), outputs CSV per GPU.
2. `merge.py` pivots by kernel name, adds instance index to handle repeated kernels, left joins by [kernel_name, GPU, instance].
3. `rocm_metrics.py` applies domain-specific formulas to derive high-level metrics normalized by architecture constants (CU count, XCD count, memory config).

### Straggler Analysis

`get_straggler_df()` groups by GPU/layer/operator, finds max timestamp across GPUs, computes s-value (how far behind each GPU is). Used to identify load imbalance.

## Extension Points

**Adding a telemetry collector**: Create `chopper/profile/telemetry/mymodule.py` with `main(stop: bool, **kwargs)`. Add to `collect.py` via `runner.add()`.

**Adding a plot**: Create `chopper/plots/myplot.py` with `get_data(**kwargs)` and `draw(fig, input_data, **kwargs)`. Type annotations define UI widgets. Module automatically discovered.

**Adding hardware counter derivations**: Add function to `rocm_metrics.py` with signature `def derive_mymetric(df: pd.DataFrame) -> None:`. Compute raw counters → high-level metric.

## Important Notes

- Traces must be generated by PyTorch profiler with Chrome trace format enabled.
- Hardware counters require ROCm environment and rocprofv3 tool.
- GUI requires PyQt6 installation.
- Type checking uses mypy in strict mode.
- Framework enum in `annotations.py` must match the distributed training framework used (FSDPv1 vs FSDPv2).
