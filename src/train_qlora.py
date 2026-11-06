"""Single-GPU QLoRA: python -m src.train_qlora --config configs/qlora.yaml"""
import argparse
from pathlib import Path

from transformers import Trainer, TrainingArguments

from .collator import DocVQACollator
from .common import load_config, set_seed, visible_gpu
from .dataset import DocVQADataset
from .model import load_processor, load_qlora


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/qlora.yaml")
    ap.add_argument("--resume", default=None, help="Trainer checkpoint-* directory")
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    print(visible_gpu())
    processor = load_processor(cfg)
    model = load_qlora(cfg)
    ds = DocVQADataset(cfg["data_root"], "train", cfg.get("max_train_samples"))
    training_args = TrainingArguments(
        output_dir=cfg["output_dir"],
        logging_dir=cfg.get("logging_dir", "results/qlora"),
        per_device_train_batch_size=cfg.get("batch_size", 1),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 8),
        max_steps=cfg["max_steps"], learning_rate=cfg["learning_rate"],
        weight_decay=cfg.get("weight_decay", 0.01),
        warmup_ratio=cfg.get("warmup_ratio", 0.03), lr_scheduler_type="cosine",
        bf16=True, gradient_checkpointing=True,
        save_strategy="steps", save_steps=cfg.get("save_steps", 50),
        save_total_limit=2, logging_steps=cfg.get("logging_steps", 5),
        report_to="tensorboard", remove_unused_columns=False,
        dataloader_num_workers=0, optim="paged_adamw_8bit",
        seed=cfg.get("seed", 42),
    )
    trainer = Trainer(
        model=model, args=training_args, train_dataset=ds,
        data_collator=DocVQACollator(processor, cfg["max_length"], train=True),
    )
    trainer.train(resume_from_checkpoint=args.resume)
    final_dir = Path(cfg["output_dir"]) / "final_adapter"
    trainer.save_model(str(final_dir))
    processor.save_pretrained(str(final_dir))
    trainer.save_state()
    print(f"Saved adapter: {final_dir}")


if __name__ == "__main__":
    main()
