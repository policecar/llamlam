"""
Two-process CPU run of llamlam.train.main that prints, per rank, a hash of each
epoch's batch order and a checksum of the final weights, so an interrupted and
resumed run can be compared with an uninterrupted one.

From the repository root (DATA is a text file, one example per line):

    export HF_HUB_OFFLINE=1
    L="python -m torch.distributed.run --nproc_per_node 2 -m scripts.two_process_cpu_check"
    $L DATA --output_dir runs --run_name full                     # uninterrupted
    $L DATA --output_dir runs --run_name res --interrupt_epoch 1  # stops in epoch 1
    $L DATA --output_dir runs --run_name res --resume_from latest

Other flags go to main() (e.g. --model_type gpt); the defaults are a tiny model
trained for 3 epochs. The full run's lines must equal the interrupted run's
(epoch 0) plus the resumed run's (epochs 1, 2, weights). Sets ACCELERATE_USE_CPU=1
and WANDB_MODE=disabled.

Workaround, in this script only (--no_workaround shows the failure): accelerate
>= 1.11 calls torch.distributed.barrier(device_ids=[local_rank]) also on CPU
(gloo), and torch 2.15 then runs the barrier on the detected accelerator. On a
Mac that is MPS, so the first barrier fails with "NotImplementedError: The
distributed operator 'c10d::barrier' is not currently implemented for the MPS
device". For gloo barriers with MPS detected, the script drops device_ids, so
the barrier runs on the CPU as on a machine without an accelerator; any other
call (Linux, CUDA, nccl) passes through unchanged. It is not in llamlam because
it patches torch globally for a development-only setup (CPU processes on a Mac);
multi-process CUDA runs should be unaffected (read from the source, not run).
"""

import argparse
import functools
import hashlib
import os

import torch
import torch.distributed as dist

import llamlam.train as train
from llamlam.config import Config

os.environ["ACCELERATE_USE_CPU"] = "1"  # read when main() builds the Accelerator
os.environ["WANDB_MODE"] = "disabled"


class Interrupted(Exception):
    pass


def install_barrier_workaround():
    barrier = dist.barrier

    def gloo_barrier(*args, device_ids=None, **kwargs):
        group = kwargs.get("group", args[0] if args else None)
        mps = getattr(torch.accelerator.current_accelerator(), "type", None) == "mps"
        if device_ids is not None and mps and dist.get_backend(group) == "gloo":
            device_ids = None  # torch then places the gloo barrier on the CPU
        return barrier(*args, device_ids=device_ids, **kwargs)

    dist.barrier = gloo_barrier


def digest(chunks):
    return hashlib.sha256(b"".join(chunks)).hexdigest()[:16]


def run(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("data_file")
    parser.add_argument("--interrupt_epoch", type=int)
    parser.add_argument("--no_workaround", action="store_true")
    args, main_argv = parser.parse_known_args(argv)
    if not args.no_workaround:
        install_barrier_workaround()
    train.Config = functools.partial(  # defaults of a fresh run; a resume stores them
        Config, data_path="text", data_files=args.data_file, data_name=None,
        n_layers=2, n_heads=2, dim_head=8, max_seq_length=32, eval_steps=5,
        n_data_workers=0, gradient_accumulation_steps=2, n_warmup_steps=2,
    )  # fmt: skip

    batches, orders, seen = [], {}, {}
    count_real_tokens = train.count_real_tokens  # called once per training batch

    def counting(batch):
        batches.append(batch["input_ids"].cpu().numpy().tobytes())
        return count_real_tokens(batch)

    def log(metrics, step=None):
        if "train_loss" in metrics:
            if metrics["epoch"] == args.interrupt_epoch:
                raise Interrupted
            orders.setdefault(metrics["epoch"], []).append(batches[-1])

    run_training = train.run_training

    def recording(model, optimizer, scheduler, train_loader, val_loader, accelerator,
                  *rest, **kwargs):  # fmt: skip
        seen["rank"] = f"rank {accelerator.process_index}/{accelerator.num_processes}"
        result = run_training(model, optimizer, scheduler, train_loader, val_loader,
                              accelerator, *rest, **{**kwargs, "log": log})  # fmt: skip
        params = accelerator.unwrap_model(model).named_parameters()
        seen["weights"] = digest(p.detach().cpu().numpy().tobytes() for _, p in params)
        return result

    train.count_real_tokens, train.run_training = counting, recording
    try:
        train.main(main_argv)
    except Interrupted:
        seen["weights"] = f"none (interrupted in epoch {args.interrupt_epoch})"
    rank = seen["rank"]
    for epoch, order in orders.items():
        print(f"{rank} epoch {epoch}: {len(order)} batches, order {digest(order)}")
    print(f"{rank} weights {seen['weights']}", flush=True)


if __name__ == "__main__":
    run()
