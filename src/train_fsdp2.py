"""Experimental multi-GPU FSDP2 LoRA, BF16 backbone.

torchrun --standalone --nproc_per_node=4 -m src.train_fsdp2 --config configs/fsdp2.yaml
IMPORTANT: loading 33B independently onto CPU per rank requires very large RAM;
see README for this limitation. Not intended for Colab single GPU.
"""
import argparse
import os
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.checkpoint import save as dcp_save, load as dcp_load
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.utils.data import DataLoader, DistributedSampler
from transformers import get_cosine_schedule_with_warmup

from .collator import DocVQACollator
from .common import load_config, set_seed
from .dataset import DocVQADataset
from .model import load_processor, load_bf16_lora


def checkpoint(model, optimizer, scheduler, step, path):
    """Distributed checkpoint. ALL ranks must call."""
    model_sd, optim_sd = get_state_dict(model, optimizer)
    state = {"model": model_sd, "optimizer": optim_sd,
             "step": torch.tensor(step, device="cpu"),
             "scheduler": scheduler.state_dict()}
    dcp_save(state_dict=state, checkpoint_id=str(path))


def restore(model, optimizer, scheduler, path):
    model_sd, optim_sd = get_state_dict(model, optimizer)
    state = {"model": model_sd, "optimizer": optim_sd,
             "step": torch.tensor(0), "scheduler": scheduler.state_dict()}
    dcp_load(state_dict=state, checkpoint_id=str(path))
    set_state_dict(model, optimizer, model_state_dict=state["model"],
                   optim_state_dict=state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    return int(state["step"].item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/fsdp2.yaml")
    parser.add_argument("--resume", default=None, help="DCP checkpoint directory")
    args = parser.parse_args()
    cfg = load_config(args.config)
    dist.init_process_group(backend="nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    if world < 2:
        raise ValueError("FSDP2 benchmark requires >=2 GPUs")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    mesh = init_device_mesh("cuda", (world,))
    set_seed(cfg.get("seed", 42))
    proc = load_processor(cfg)
    # Loads BF16 backbone on host independently on all ranks before sharding.
    # Plan ~70+ GB host RAM per rank during model construction.
    model = load_bf16_lora(cfg)
    text_layers = model.base_model.model.model.language_model.layers
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    for layer in text_layers:
        fully_shard(layer, mesh=mesh, mp_policy=mp, reshard_after_forward=True)
    fully_shard(model, mesh=mesh, mp_policy=mp, reshard_after_forward=True)
    model.train()
    ds = DocVQADataset(cfg["data_root"], "train", cfg.get("max_train_samples"))
    sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True,
                                 seed=cfg.get("seed", 42), drop_last=True)
    loader = DataLoader(ds, sampler=sampler, batch_size=cfg.get("batch_size", 1),
                        collate_fn=DocVQACollator(proc, cfg["max_length"]),
                        num_workers=0, drop_last=True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=cfg["learning_rate"])
    total = cfg["max_steps"]
    sched = get_cosine_schedule_with_warmup(opt, int(total * 0.03), total)
    start_step = restore(model, opt, sched, args.resume) if args.resume else 0
    acc = cfg.get("gradient_accumulation_steps", 8)
    iterator = iter(loader)
    epoch = 0
    opt.zero_grad(set_to_none=True)
    for step in range(start_step, total):
        start = time.perf_counter()
        loss_total = 0.0
        for micro in range(acc):
            try:
                batch = next(iterator)
            except StopIteration:
                epoch += 1
                sampler.set_epoch(epoch)
                iterator = iter(loader)
                batch = next(iterator)
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            # With FSDP2, sync only on last microbatch.
            model.set_requires_gradient_sync(micro == acc - 1)
            output = model(**batch)
            loss = output.loss / acc
            loss.backward()
            loss_total += loss.detach().item()
        model.set_requires_gradient_sync(True)
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        if rank == 0 and (step + 1) % 5 == 0:
            print(f"step={step+1} loss={loss_total:.4f} sec={time.perf_counter()-start:.1f}", flush=True)
        if (step + 1) % cfg.get("save_steps", 50) == 0:
            checkpoint(model, opt, sched, step + 1, Path(cfg["output_dir"]) / f"step_{step+1}")
    checkpoint(model, opt, sched, total, Path(cfg["output_dir"]) / "final")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
