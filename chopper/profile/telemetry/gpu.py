import os
import socket
from time import monotonic_ns
import pandas as pd
from loguru import logger
from amdsmi import (
    amdsmi_get_processor_handles,
    amdsmi_get_gpu_metrics_info,
    amdsmi_get_gpu_kfd_info,
    amdsmi_shut_down,
    amdsmi_init,
    AmdSmiException,
)
from time import sleep


def _main(
    stop,
    filename,
    outdir,
    on,
    off,
):
    try:
        devices = amdsmi_get_processor_handles()
        gpu_map = {
            amdsmi_get_gpu_kfd_info(device)['node_id']: device
            for device in devices
        }

        results = []

        pause_ts = monotonic_ns() + int(on * 1e9)
        while not stop.value:
            for gpu, device in gpu_map.items():
                ts = monotonic_ns()
                result = amdsmi_get_gpu_metrics_info(device)
                result['ts'] = ts
                result['gpu'] = gpu
                results.append(result)
            if ts >= pause_ts:
                sleep(off)
                pause_ts = monotonic_ns() + int(on * 1e9)

        df = pd.DataFrame(results)
        df["node"] = socket.gethostname()  # multi-node: rows carry their host
        os.makedirs(outdir, exist_ok=True)
        df.to_pickle(f"{outdir}/{filename}")
    except AmdSmiException as e:
        logger.warning(e)


def main(
    stop: bool,
    filename: str = 'gpu.pkl',
    nvidia: bool = False,
    outdir: str = '.',
    on: float = 0.0,
    off: float = 0.1,
):
    assert nvidia is False, "NVIDIA GPU telemetry is not supported currently"
    # TODO change for nvidia
    try:
        amdsmi_init()
        _main(stop, filename, outdir, on, off)
        amdsmi_shut_down()
    except AmdSmiException as e:
        logger.warning(e)
    return 0
