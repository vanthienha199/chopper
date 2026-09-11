from argparse import ArgumentParser
from loguru import logger
from chopper.profile.runner import Runner
from chopper.profile.telemetry import clock_anchor


def _namespace_outdir(outdir):
    # multi-node SLURM jobs: two nodes writing gpu.pkl into one shared
    # outdir silently clobber each other, so each node gets its own subdir
    import os
    import socket
    if int(os.environ.get("SLURM_JOB_NUM_NODES", "1")) > 1:
        return os.path.join(outdir, socket.gethostname())
    return outdir


def main(program,
         counter_names,
         outdir,
         container,
         nvidia,
         cpu_telemetry,
         gpu_telemetry,
         device,
         telemetry_on=0.0,
         telemetry_off=0.1,
         sample_ms=1,
         cpu_clock="monotonic",
         vendor="auto"):
    if len(program) == 0:
        logger.error("Please pass a program to run")
        return -1
    outdir = _namespace_outdir(outdir)

    runner = Runner()

    if cpu_telemetry:
        from chopper.profile.telemetry import cpu
        runner.add(
            cpu.main,
            False,
            outdir=outdir,
            on=telemetry_on,
            off=telemetry_off,
            cpu_clock=cpu_clock,
        )
    if gpu_telemetry:
        from chopper.profile.telemetry.detect import resolve_vendor, Vendor
        vend = resolve_vendor(vendor)
        if vend is Vendor.UNKNOWN and nvidia:
            vend = Vendor.NVIDIA
        if vend is Vendor.NVIDIA:
            from chopper.profile.telemetry import gpu_nvidia as gpu_mod
        else:
            from chopper.profile.telemetry import gpu as gpu_mod
        logger.info(f"GPU telemetry backend: {vend.value}")
        runner.add(
            gpu_mod.main,
            False,
            nvidia=(vend is Vendor.NVIDIA),
            outdir=outdir,
            on=telemetry_on,
            off=telemetry_off,
        )

    if device:
        from chopper.profile.telemetry import device_counters
        runner.add(
            device_counters.main,
            True,
            program,
            counter_names,
            outdir,
            container,
            nvidia,
            sample_ms,
        )
    else:
        from chopper.profile.telemetry import counters
        # rocprofv3 dispatch counters expect a flat list
        flat_counters = None
        if counter_names is not None:
            flat_counters = [c for group in counter_names for c in group]
        runner.add(
            counters.main,
            True,
            program,
            flat_counters,
            outdir,
            container,
            nvidia,
        )
    # Bracket the collection window with a clock anchor pair. Every collector
    # stamps its own native clock, so these two readings are what lets a
    # reader put this node's samples on the shared epoch timeline later, and
    # what makes the node's clock drift over the run a measured number.
    clock_anchor.write_anchor(outdir, clock_anchor.TAKEN_START)
    runner.start()
    runner.join()
    clock_anchor.write_anchor(outdir, clock_anchor.TAKEN_STOP)


if __name__ == "__main__":
    parser = ArgumentParser(
        usage='pass program to run and what to collect (i.e., hardware counters, CPU and GPU telemetry)',
    )
    parser.add_argument(
        '--counters',
        nargs='+',
        action='append',
        required=False,
        help='Hardware counters to collect. Use once for auto-grouping by 4, '
             'or repeat for explicit groups: --counters A B C --counters D E'
    )
    parser.add_argument(
        '--nvidia',
        action='store_true',
        required=False,
        help='Force NVIDIA backend (superseded by --vendor auto-detect)'
    )
    parser.add_argument(
        '--vendor',
        choices=['auto', 'amd', 'nvidia'],
        default='auto',
        help='GPU vendor for telemetry backend (default auto-detect)'
    )
    parser.add_argument(
        '--cpu-telemetry',
        action='store_true',
        required=False,
        help='collect CPU telemetry'
    )
    parser.add_argument(
        '--gpu-telemetry',
        action='store_true',
        required=False,
        help='collect GPU telemetry'
    )
    parser.add_argument(
        '--output-dir',
        required=False,
        default=".",
        help='directory to put counters'
    )
    parser.add_argument(
        '--container',
        required=False,
        help="Container image to use"
    )
    parser.add_argument(
        '--device',
        action='store_true',
        required=False,
        help='Use device-level counter sampling (no kernel serialization) instead of rocprofv3 dispatch profiling'
    )
    parser.add_argument(
        '--sample-ms',
        type=int,
        default=1,
        help='Device counter sampling interval in ms (default: 1, only used with --device)'
    )
    parser.add_argument(
        '--cpu-clock',
        choices=['monotonic', 'rocprofiler'],
        default='monotonic',
        help='Clock domain for CPU telemetry timestamps. "rocprofiler" aligns '
             'CPU samples with GPU kernel traces (needs ROCm); falls back to '
             'monotonic if unavailable.'
    )
    parser.add_argument(
        '--telemetry-on',
        type=float,
        default=0.0,
        help='duration (seconds) to sample continuously before pausing (default: 0.0)',
    )
    parser.add_argument(
        '--telemetry-off',
        type=float,
        default=0.1,
        help='sleep duration (seconds) between samples (default: 0.1 = 10 Hz)',
    )
    parser.add_argument(
        'program',
        nargs='*',
        help='program to run',
    )
    args = parser.parse_args()
    exit(main(
        args.program,
        args.counters,
        args.output_dir,
        args.container,
        args.nvidia,
        args.cpu_telemetry,
        args.gpu_telemetry,
        args.device,
        args.telemetry_on,
        args.telemetry_off,
        args.sample_ms,
        args.cpu_clock,
        args.vendor,
    ))
