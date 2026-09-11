"""
PyTorch benchmark: GEMM, AllGather, or both concurrently.

Modes:
  gemm       - 100 FP16 8192x8192x8192 GEMMs on GPU 0
  comm       - 100 256MB AllGathers across all GPUs
  both       - GEMM + AllGather running concurrently on separate streams

For 'comm' and 'both' modes, use torchrun:
  torchrun --nproc-per-node=8 bench.py --mode comm
  torchrun --nproc-per-node=8 bench.py --mode both

For 'gemm' mode:
  python bench.py --mode gemm
"""

import argparse
import os
import time
import torch
import torch.distributed as dist


def run_gemm(stream, a, b, n_iters, sleep_ms):
    """Run GEMMs on the given stream with sync + sleep between iterations."""
    with torch.cuda.stream(stream):
        for _ in range(n_iters):
            torch.mm(a, b)
    torch.cuda.synchronize()


def run_comm(stream, chunk, output, n_iters, sleep_ms):
    """Run AllGathers on the given stream with sync + sleep between iterations."""
    with torch.cuda.stream(stream):
        for _ in range(n_iters):
            dist.all_gather_into_tensor(output, chunk)
    torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["gemm", "comm", "both"], default="gemm")
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--sleep-ms", type=int, default=2,
                        help="Sleep between iterations for stroboscopic sampling")
    parser.add_argument("--m", type=int, default=8192)
    parser.add_argument("--n", type=int, default=8192)
    parser.add_argument("--k", type=int, default=8192)
    parser.add_argument("--comm-mb", type=int, default=256,
                        help="Total AllGather size in MB")
    args = parser.parse_args()

    # local_rank picks the GPU on this node and restarts at 0 on every node.
    # global_rank is unique across the job and is what "rank 0 prints" means.
    # Using the global rank to select a device breaks on the second node of a
    # multi-node job, where it exceeds the local device count.
    local_rank = int(os.environ.get("LOCAL_RANK",
                                    os.environ.get("SLURM_LOCALID", 0)))
    global_rank = int(os.environ.get("RANK",
                                     os.environ.get("SLURM_PROCID", local_rank)))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if args.mode in ("comm", "both") and world_size == 1:
        print("ERROR: comm/both modes need multiple ranks. Use:")
        print("  torchrun --nproc-per-node=8 bench.py --mode", args.mode)
        return

    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group("nccl")

    torch.cuda.set_device(local_rank)

    # Create streams
    compute_stream = torch.cuda.Stream()
    comm_stream = torch.cuda.Stream() if args.mode in ("comm", "both") else None

    # Allocate GEMM data
    a = torch.randn(args.m, args.k, dtype=torch.float16, device="cuda")
    b = torch.randn(args.k, args.n, dtype=torch.float16, device="cuda")

    # Allocate comm data
    chunk = None
    output = None
    if args.mode in ("comm", "both"):
        n_elems = (args.comm_mb * 1024 * 1024) // (2 * world_size)
        chunk = torch.randn(n_elems, dtype=torch.float16, device="cuda")
        output = torch.empty(n_elems * world_size, dtype=torch.float16, device="cuda")

    # Warmup
    for _ in range(args.warmup):
        if args.mode in ("gemm", "both"):
            with torch.cuda.stream(compute_stream):
                torch.mm(a, b)
        if args.mode in ("comm", "both"):
            with torch.cuda.stream(comm_stream):
                dist.all_gather_into_tensor(output, chunk)
    torch.cuda.synchronize()

    # Initial sleep for baseline
    time.sleep(args.sleep_ms / 1000.0)

    # Main loop
    for i in range(args.iters):
        if args.mode == "gemm":
            with torch.cuda.stream(compute_stream):
                torch.mm(a, b)
            torch.cuda.synchronize()
        elif args.mode == "comm":
            with torch.cuda.stream(comm_stream):
                dist.all_gather_into_tensor(output, chunk)
            torch.cuda.synchronize()
        elif args.mode == "both":
            with torch.cuda.stream(compute_stream):
                torch.mm(a, b)
            with torch.cuda.stream(comm_stream):
                dist.all_gather_into_tensor(output, chunk)
            torch.cuda.synchronize()

        time.sleep(args.sleep_ms / 1000.0)

    if global_rank == 0:
        print(f"Completed {args.iters} iterations in '{args.mode}' mode")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
