import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration


def load_processor(cfg):
    processor = AutoProcessor.from_pretrained(cfg["model_id"])
    # Qwen VL image_processor uses pixel area limits rather than edge length.
    processor.image_processor.min_pixels = cfg.get("min_pixels", 65536)
    processor.image_processor.max_pixels = cfg.get("max_pixels", 786432)
    return processor


def lora_model(model, cfg):
    # Restrict to language_model names to prevent unintended LoRA on ViT.
    target_suffixes = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    selected = [name for name, module in model.named_modules()
                if "language_model" in name and name.endswith(target_suffixes)
                and isinstance(module, torch.nn.Linear)]
    if not selected:
        raise RuntimeError("No text-language projection modules found; inspect model.named_modules()")
    lcfg = LoraConfig(
        r=cfg["rank"], lora_alpha=cfg["alpha"],
        lora_dropout=cfg.get("lora_dropout", 0.05),
        target_modules=selected, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()
    return model


def load_qlora(cfg):
    if not torch.cuda.is_available():
        raise RuntimeError("QLoRA requires NVIDIA CUDA GPU")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This baseline requires BF16 capable GPU (A100/H100 etc.)")
    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        cfg["model_id"], quantization_config=quant, dtype=torch.bfloat16,
        device_map={"": torch.cuda.current_device()},
        attn_implementation="sdpa", low_cpu_mem_usage=True,
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.config.use_cache = False
    return lora_model(model, cfg)


def load_bf16_lora(cfg, device=None):
    """BF16 backbone; each rank initially loads the whole model on CPU.

    Requires sufficient CPU RAM *on each rank*; FSDP2 shards after loading.
    For production-scale sharded loading use meta-init + DCP checkpoint conversion.
    """
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        cfg["model_id"], dtype=torch.bfloat16, attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model = lora_model(model, cfg)
    if device is not None:
        model.to(device)
    return model
