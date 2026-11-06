"""Multi-GPU BF16 LoRA + DeepSpeed ZeRO-3 (not 4-bit QLoRA).

accelerate launch --num_processes 4 -m src.train_deepspeed
"""
import argparse
from pathlib import Path

from transformers import Trainer, TrainingArguments
from .collator import DocVQACollator
from .common import load_config, set_seed
from .dataset import DocVQADataset
from .model import load_processor, load_bf16_lora


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/fsdp2.yaml")
    ap.add_argument("--deepspeed", default="configs/deepspeed_zero3.json")
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    processor = load_processor(cfg)
    # DeepSpeed ZeRO-3 init with HF Trainer needs model construction/config awareness.
    # For 32B, full CPU load per rank may exceed host RAM; see README.
    model = load_bf16_lora(cfg)
    ds = DocVQADataset(cfg["data_root"], "train", cfg.get("max_train_samples"))
    tr_args = TrainingArguments(
        output_dir=cfg["output_dir"].replace("fsdp2", "deepspeed"),
        per_device_train_batch_size=cfg.get("batch_size", 1),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 8),
        max_steps=cfg["max_steps"], learning_rate=cfg["learning_rate"],
        bf16=True, deepspeed=args.deepspeed,
        gradient_checkpointing=True, save_strategy="steps",
        save_steps=cfg.get("save_steps", 50), logging_steps=5,
        report_to="tensorboard", remove_unused_columns=False,
        dataloader_num_workers=0, save_total_limit=2,
    )
    trainer = Trainer(model=model, args=tr_args, train_dataset=ds,
                     data_collator=DocVQACollator(processor, cfg["max_length"]))
    trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model()
    trainer.save_state()
    if trainer.is_world_process_zero():
        processor.save_pretrained(tr_args.output_dir)


if __name__ == "__main__":
    main()
