# Qwen3-VL Fine-Tuning for Document Visual Question Answering

> A modular research project for parameter-efficient fine-tuning and distributed training of vision-language models on document question answering tasks.

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-FSDP2-ee4c2c)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow)](https://huggingface.co/docs/transformers/)
[![Status](https://img.shields.io/badge/status-experimental-orange)](#project-status)

## Overview

This repository provides a reproducible starting point for fine-tuning **[Qwen3-VL-32B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct)** on **[DocVQA](https://site.docvqa.org/datasets/docvqa)**. It explores the intersection of multimodal adaptation and hardware-efficient model training: *How much document-question-answering performance can we gain within a fixed GPU-memory and compute budget?*

The project supports a **single-GPU QLoRA workflow** for practical experimentation and includes **experimental multi-GPU BF16 LoRA workflows** using PyTorch FSDP2 and DeepSpeed ZeRO-3. The training backends are **alternatives**, not technologies intended to be stacked together.

The current implementation targets DocVQA, but its separation of dataset, collator, model, training backend, and evaluation is designed to make other image-question-answering datasets easier to integrate. It is **not** a generic plug-and-play fine-tuner for arbitrary VLM architectures or datasets without modification.

### Features

- **Parameter-efficient fine-tuning:** 4-bit NF4 QLoRA, configurable LoRA rank, alpha, and dropout.
- **Multimodal supervision:** Qwen3-VL processor for document images and questions; loss computed on assistant answer tokens.
- **Distributed experimentation:** independent reference implementations for FSDP2 and DeepSpeed ZeRO-3 with BF16 LoRA.
- **Evaluation:** Answer Normalized Levenshtein Similarity (ANLS) and exact match (EM), plus per-example predictions.
- **Performance profiling:** step latency, sample/token throughput, and CUDA memory statistics.
- **Reproducibility:** YAML configurations, checkpoint/resume support, deterministic seed setting, and unit tests for metrics.

## Project Status

| Component | Status | Notes |
| --- | --- | --- |
| Dataset conversion and loading | Implemented | Requires locally obtained official DocVQA files |
| Single-GPU QLoRA training | Implemented; GPU validation pending | Recommended starting workflow |
| Base/adapted model evaluation | Implemented; GPU validation pending | ANLS is an approximation of official scoring |
| Training microbenchmark | Implemented; GPU validation pending | Run a short warm-up before planning costs |
| PyTorch FSDP2 BF16 LoRA | **Experimental** | Startup CPU-memory requirements are high; multi-GPU 32B end-to-end run not verified |
| DeepSpeed ZeRO-3 BF16 LoRA | **Experimental** | Needs ZeRO-aware low-memory initialization for robust 32B deployment |

**Validation:** Python syntax checks and CPU-only metric unit tests have passed. No successful end-to-end Qwen3-VL-32B GPU training run is claimed. Hardware requirements and software compatibility must be confirmed on the target instance.

## Architecture

```text
DocVQA images + questions + answers
              |
       Data preparation
              |
     Dataset + Collator
              |
    Qwen3-VL processor
              |
       Qwen3-VL-32B
              |
       LoRA adapters
              |
       Training backend
       /      |       \
    QLoRA   FSDP2   ZeRO-3
       \      |       /
        Checkpoints
              |
  Evaluation + GPU profiling
```

**Default adaptation strategy:** freeze the pretrained backbone and apply LoRA to supported attention and MLP projections. The training target is autoregressive answer prediction conditioned on the document image and the question. Visual resolution is controlled through an image pixel-area budget (`min_pixels`, `max_pixels`), not a fixed image width or height.

## Repository Structure

```text
qwen3vl-docvqa/
├── configs/
│   ├── qlora.yaml                 # Single-GPU QLoRA experiment
│   ├── fsdp2.yaml                 # Multi-GPU BF16 LoRA experiment
│   └── deepspeed_zero3.json      # DeepSpeed ZeRO-3 configuration
├── data/                          # Prepared dataset (not committed)
├── notebooks/
│   └── colab_qlora.ipynb         # Colab-oriented entry point
├── scripts/
│   └── prepare_docvqa.py         # Official annotation/image conversion
├── src/
│   ├── common.py                 # Configuration, seeds and utilities
│   ├── dataset.py                # DocVQA dataset
│   ├── collator.py               # Multimodal batching and loss masking
│   ├── model.py                  # Processor, base model and LoRA loading
│   ├── train_qlora.py            # Single-GPU QLoRA training
│   ├── train_fsdp2.py            # Experimental FSDP2 training
│   ├── train_deepspeed.py        # Experimental DeepSpeed training
│   ├── evaluate.py               # Inference and answer metrics
│   ├── metrics.py                # ANLS and normalization
│   └── benchmark.py              # GPU throughput/memory benchmark
├── tests/
│   └── test_metrics.py
├── checkpoints/                   # Generated checkpoints (not committed)
├── results/                       # Generated predictions and logs
├── requirements.txt
└── README.md
```

## Requirements

- **Python:** 3.11 recommended.
- **Hardware:** NVIDIA CUDA-capable GPU; **A100 80 GB or H100 80 GB** recommended as an initial target for 32B QLoRA. Availability of memory depends heavily on document resolution and sequence length. A100 40 GB may run out of memory.
- **Distributed training:** multiple GPUs on the same node are strongly recommended, with sufficient host RAM and fast GPU interconnects. The current BF16 distributed model-loading path may require approximately the full pretrained model in CPU memory **per rank** before sharding.
- **Storage:** plan sufficient space for model weights, dataset, Hugging Face cache, checkpoints, and backups (potentially hundreds of GB).
- **Access:** obtain the model from Hugging Face and download DocVQA according to their respective terms.

### Installation

Clone or extract the repository, then from its root directory:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For the optional DeepSpeed backend:

```bash
pip install 'deepspeed>=0.17'
```

Check the environment:

```bash
python -c "import torch, transformers, peft; print('CUDA:', torch.cuda.is_available()); print('Torch:', torch.__version__); print('Transformers:', transformers.__version__)"
nvidia-smi
pytest -q tests/
```

**Environment note:** `requirements.txt` gives minimum package versions, not an immutable lock file. Pin exact versions after a verified GPU run. Install a CUDA-compatible PyTorch build for your driver and platform. For cloud VMs, use an appropriate CUDA/PyTorch image when available. For notebooks, consult `notebooks/colab_qlora.ipynb`.

## Quick Start

The shortest useful path is to prepare the dataset, run a small QLoRA training job, and compare it with the base model on the same validation examples.

### 1. Prepare DocVQA

Download the official **DocVQA Task 1** annotations and corresponding images. Files are **not included** in this repository.

```bash
python scripts/prepare_docvqa.py \
  --annotations /path/to/docvqa_train_v1.0.json \
  --images /path/to/train_images \
  --split train

python scripts/prepare_docvqa.py \
  --annotations /path/to/docvqa_val_v1.0.json \
  --images /path/to/validation_images \
  --split validation
```

By default, image files are **symlinked** rather than duplicated. Add `--copy-images` when moving the prepared dataset to another machine or container.

Expected layout:

```text
data/
├── train/
│   ├── annotations.json
│   └── images/
│       └── <document-image-files>
└── validation/
    ├── annotations.json
    └── images/
        └── <document-image-files>
```

A simplified annotation entry is:

```json
{
  "data": [
    {
      "questionId": 1,
      "image": "document_001.png",
      "question": "What is the invoice number?",
      "answers": ["INV-12345", "INV 12345"]
    }
  ]
}
```

The training loader uses the **first** reference answer as its target, while evaluation considers all available references. Keep the official train/validation separation; avoid leaking pages or document IDs between splits when creating custom subsets.

### 2. Configure the experiment

Edit `configs/qlora.yaml`. The default file runs a **small 1,000-example experiment** with a maximum of 100 optimizer steps; it is not configured for a full-epoch benchmark.

```yaml
model_id: Qwen/Qwen3-VL-32B-Instruct
max_train_samples: 1000
max_eval_samples: 200
max_pixels: 786432
min_pixels: 65536
max_length: 4096
rank: 16
alpha: 32
learning_rate: 0.0002
batch_size: 1
gradient_accumulation_steps: 8
max_steps: 100
```

The values above are starting points, **not validated optimal hyperparameters**. Decrease `max_pixels` or `max_length` if memory is insufficient; confirm the model still receives enough visual detail to answer the questions.

### 3. Fine-tune with QLoRA (single GPU)

```bash
python -m src.train_qlora --config configs/qlora.yaml
```

The output directory is configured through `output_dir` (default `checkpoints/qlora`). The final adapter and processor are saved in `checkpoints/qlora/final_adapter`.

Resume an interrupted run from a **Trainer checkpoint** rather than the final adapter directory:

```bash
python -m src.train_qlora \
  --config configs/qlora.yaml \
  --resume checkpoints/qlora/checkpoint-50
```

On interruptible GPU instances, regularly copy checkpoints to persistent storage. The model weights themselves are not embedded in the LoRA adapter.

### 4. Evaluate baseline and adapted model

Use the **same validation set, generation settings, and sample limit** for a fair comparison.

```bash
# Base model
python -m src.evaluate \
  --config configs/qlora.yaml \
  --max-samples 200 \
  --output results/base_eval.json

# Fine-tuned adapter
python -m src.evaluate \
  --config configs/qlora.yaml \
  --adapter checkpoints/qlora/final_adapter \
  --max-samples 200 \
  --output results/adapter_eval.json
```

This produces aggregate ANLS/EM scores and per-question predictions. **ANLS in this project is an approximate implementation**; verify official normalization and submission rules before reporting benchmark-comparable scores.

### 5. Measure training performance

```bash
python -m src.benchmark \
  --config configs/qlora.yaml \
  --warmup 2 \
  --steps 10
```

Measure actual **seconds/sample**, peak CUDA allocated/reserved memory, and throughput before estimating cloud training costs. Include model loading and checkpoint time in your overall budget separately.

## Distributed Training (Experimental)

Distributed entry points are provided for researching training efficiency, not as production-ready 32B launch recipes. **Do not enable FSDP2 and DeepSpeed simultaneously.** Neither path uses 4-bit QLoRA; both are designed for BF16 LoRA with sharded training state.

### PyTorch FSDP2

```bash
torchrun --standalone --nproc_per_node=4 \
  -m src.train_fsdp2 \
  --config configs/fsdp2.yaml
```

Resume from a distributed checkpoint directory:

```bash
torchrun --standalone --nproc_per_node=4 \
  -m src.train_fsdp2 \
  --config configs/fsdp2.yaml \
  --resume checkpoints/fsdp2/step_50
```

The reference implementation applies `fully_shard()` to language decoder blocks and then to the root model. It uses PyTorch Distributed Checkpoint. **Known limitations:** large per-rank CPU loading overhead, unverified 32B-scale PEFT/FSDP2 interoperability, and incomplete exact sampler/RNG restoration on resume. Validate on a smaller Qwen3-VL variant and two GPUs before paying for a full multi-GPU job.

### DeepSpeed ZeRO-3

```bash
accelerate launch --multi_gpu --num_processes 4 \
  -m src.train_deepspeed \
  --config configs/fsdp2.yaml \
  --deepspeed configs/deepspeed_zero3.json
```

This is an independent BF16 LoRA route. The present model loader does **not yet implement a fully ZeRO-aware low-memory model initialization path**, so loading a 32B model can exceed host RAM before sharding. Treat this entry point as a reference for adaptation and integration testing, not a turnkey 32B training recipe.

## Experiment Design

The suggested experiment matrix emphasizes consistent data and precision controls rather than mixing incompatible training approaches.

| ID | Setup | Question |
| --- | --- | --- |
| E0 | Base Qwen3-VL-32B, no tuning | What is the initial DocVQA performance? |
| E1 | QLoRA, rank 16, small subset | Is the complete fine-tuning pipeline functional? |
| E2 | QLoRA, rank 32, matched samples | Does additional adapter capacity improve ANLS? |
| E3 | QLoRA, varied `max_pixels` | What is the quality–memory trade-off of document resolution? |
| E4 | FSDP2 BF16 LoRA, multiple GPUs | What are memory, throughput, and communication costs? |
| E5 | DeepSpeed ZeRO-3 BF16 LoRA | How does ZeRO-3 compare under matched conditions? |

**Recommended reporting:** ANLS, EM, training loss, samples/sec, tokens/sec, step time, peak allocated/reserved VRAM, effective batch size, GPU model/count, precision, image pixel budget, package versions, total GPU-hours, and approximate compute cost.

For a fair FSDP2-vs-ZeRO-3 comparison, match the model, trainable modules, input data/order, pixel budget, global batch size, optimizer, precision, and number of measured steps. Compare end-to-end duration as well as steady-state step time.

## Reproducibility and Checkpoints

- Store the exact YAML/JSON configuration with every run.
- Record the GPU type, GPU count, CUDA driver, PyTorch/Transformers/PEFT versions, dataset revision, and seed.
- Keep a fixed validation subset when comparing the base model with an adapter.
- Persist final adapters, Trainer checkpoints, evaluation JSON files, and raw performance logs outside ephemeral GPU volumes.
- For interrupted FSDP2 training, the current checkpoint path restores model/optimizer-related state but does **not** guarantee exact continuation of the sample stream or all RNG state.
- The repository does not ship training results, pretrained weights, or the DocVQA dataset.

## Troubleshooting

| Symptom | Suggestions |
| --- | --- |
| CUDA out of memory | Reduce `max_pixels`, `max_length`, or batch size; check that 4-bit loading was enabled for QLoRA |
| Model/processor API error | Confirm installed `transformers` version and cross-check official Qwen3-VL model examples |
| Missing images | Re-check `--images` path and symlink targets; use `--copy-images` when transferring datasets |
| Unexpectedly slow training | Measure image token counts, I/O, CUDA memory, checkpointing overhead, and GPU utilization |
| FSDP2/ZeRO-3 startup runs out of host RAM | Use a smaller model for validation; implement sharded/meta-device initialization before 32B runs |
| Evaluation differs from official leaderboard | Confirm the official DocVQA ANLS implementation, reference normalization, answer handling, and test protocol |

## Roadmap

Potential future improvements (not currently implemented):

- [ ] Verify end-to-end 32B QLoRA training on A100/H100 and publish exact tested dependencies.
- [ ] Add robust low-memory model initialization for multi-GPU FSDP2 and ZeRO-3.
- [ ] Add an official DocVQA ANLS parity test and fixed benchmark manifest.
- [ ] Add fully resumable distributed samplers and RNG/checkpoint restoration.
- [ ] Support alternative datasets via dataset adapters and schema validation.
- [ ] Produce reproducible performance dashboards and GPU-hour cost reports.

## References and Acknowledgments

**Models and fine-tuning**

- [Qwen3-VL-32B-Instruct — model card](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct)
- [Qwen3-VL — official repository and fine-tuning framework](https://github.com/QwenLM/Qwen3-VL/tree/main/qwen-vl-finetune)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [Hugging Face PEFT](https://github.com/huggingface/peft)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)

**Datasets and evaluation**

- [DocVQA official website and dataset](https://site.docvqa.org/datasets/docvqa)
- [DocVQA: A Dataset for VQA on Document Images](https://arxiv.org/abs/2007.00398)

**Distributed and efficient training**

- [PyTorch FSDP2 / `fully_shard` documentation](https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html)
- [PyTorch Distributed Checkpoint](https://docs.pytorch.org/docs/stable/distributed.checkpoint.html)
- [DeepSpeed](https://github.com/deepspeedai/DeepSpeed)
- [DeepSpeed ZeRO documentation](https://www.deepspeed.ai/tutorials/zero/)
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) — reference for future tensor/pipeline parallel work; **not integrated** in this repository.

## Contributing

Issues, reproducibility reports, and pull requests are welcome after publishing the repository. When reporting a training issue, include your GPU, memory size, package versions, exact command, relevant configuration, and a minimal error log. Do not attach private document images or sensitive training data.

## License and Data Use

**No repository license has been selected yet.** Before public distribution, choose an appropriate license and add a `LICENSE` file. The licenses and terms of Qwen3-VL model weights, DocVQA data, and third-party dependencies apply separately; using this repository does not grant additional rights to redistribute them.

## Disclaimer

This is a research-oriented engineering implementation. Performance figures, 32B memory feasibility, and distributed scalability should not be treated as verified results until reproduced on actual hardware.
