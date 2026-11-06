"""Measure memory and batch latency with an actual model and real DocVQA examples."""
import argparse
import json
import time
import torch

from .collator import DocVQACollator
from .common import load_config, save_json
from .dataset import DocVQADataset
from .model import load_processor, load_qlora


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/qlora.yaml")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--output", default="results/benchmark.json")
    args = parser.parse_args()
    cfg = load_config(args.config)
    proc = load_processor(cfg)
    model = load_qlora(cfg)
    ds = DocVQADataset(cfg["data_root"], "train", max_samples=args.steps+args.warmup)
    collate = DocVQACollator(proc, cfg["max_length"])
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg["learning_rate"])
    latency, token_counts = [], []
    for i in range(args.steps + args.warmup):
        sample = ds[i % len(ds)]
        batch = {k: v.to("cuda") for k,v in collate([sample]).items()}
        torch.cuda.synchronize()
        if i == args.warmup:
            torch.cuda.reset_peak_memory_stats()
        t = time.perf_counter()
        loss = model(**batch).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        if i >= args.warmup:
            latency.append(time.perf_counter()-t)
            token_counts.append(int(batch["attention_mask"].sum()))
    result = {"samples":len(latency), "mean_seconds_per_sample":sum(latency)/len(latency),
              "samples_per_second":len(latency)/sum(latency),
              "mean_sequence_tokens":sum(token_counts)/len(token_counts),
              "peak_allocated_gib":torch.cuda.max_memory_allocated()/2**30,
              "peak_reserved_gib":torch.cuda.max_memory_reserved()/2**30,
              "gpu":torch.cuda.get_device_name()}
    save_json(result, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
