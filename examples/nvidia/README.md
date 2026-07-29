# NVIDIA CPU+GPU timeline (H100)

The NVIDIA counterpart of the AMD CPU+GPU timeline. Same idea: stamp CPU samples
and GPU kernel timestamps with one clock, then plot them together. On NVIDIA the
shared clock is CUPTI's `cuptiGetTimestamp` (see `telemetry/cupti_clock.py`), and
the GPU kernel timeline comes from the CUPTI Activity API tracer
(`telemetry/src/cupti_kernel_trace.cpp`), the analog of the AMD device sampler's
kernel trace.

Pieces here:
- `cuda_probe.cu` — phased workload (GPU-heavy then all-core CPU-heavy).
- `cpu_sampler_cupti.py` — samples CPU utilization stamped with the CUPTI clock.
- `plot_cpu_gpu_nvidia.py` — bins both on the shared clock and plots CPU vs GPU
  busy% (union of kernel intervals).

## Build

```
module load CUDA/12.2.0 GCC/12.3.0
# kernel tracer (loaded via CUDA_INJECTION64_PATH)
g++ -std=c++17 -fPIC -shared -o libcupti_kernel_trace.so \
    ../../chopper/profile/telemetry/src/cupti_kernel_trace.cpp \
    -I$CUDA_HOME/include -I$CUDA_HOME/extras/CUPTI/include \
    -L$CUDA_HOME/extras/CUPTI/lib64 -lcupti
nvcc -O2 -std=c++17 -Xcompiler -pthread -o cuda_probe cuda_probe.cu
```

## Run

```
export LD_LIBRARY_PATH=$CUDA_HOME/extras/CUPTI/lib64:$LD_LIBRARY_PATH
python cpu_sampler_cupti.py 6 cpu.pkl &          # CPU on the CUPTI clock
CUDA_INJECTION64_PATH=$PWD/libcupti_kernel_trace.so \
CHOPPER_NV_TRACE_OUTPUT=$PWD/kernel_traces.csv ./cuda_probe
wait
python plot_cpu_gpu_nvidia.py cpu.pkl kernel_traces.csv nv_timeline.png 50
```

Validated on an ACES H100 PCIe: 44k kernels traced, kernel timestamps fall inside
the CUPTI python-clock window ("same clock domain"), CPU and GPU come out
anti-correlated on the shared clock. The kernel tracer + `merge_cpu_gpu_timeline`
are the reusable parts; this example is the harness that exercises them.
