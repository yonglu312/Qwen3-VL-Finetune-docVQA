"""DocVQA ANLS inference and scoring; requires validated reference answers."""
import argparse
import json
import re
import unicodedata
from pathlib import Path

import torch
from peft import PeftModel
from transformers import Qwen3VLForConditionalGeneration

from .collator import DocVQACollator
from .common import load_config, save_json, visible_gpu
from .dataset import DocVQADataset
from .model import load_processor


def normalize(s):
    s = unicodedata.normalize("NFKC", str(s)).lower().strip()
    return re.sub(r"\s+", " ", s)


def edit_distance(a, b):
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def anls(prediction, references, threshold=0.5):
    """ANLS: max(1-NED) for available references if similarity >= 0.5."""
    p = normalize(prediction)
    best = 0.0
    for ref in references:
        r = normalize(ref)
        denom = max(len(p), len(r))
        similarity = 1.0 if denom == 0 else 1.0 - edit_distance(p, r) / denom
        best = max(best, similarity)
    return best if best >= threshold else 0.0


def evaluate(cfg, adapter=None, max_samples=None, max_new_tokens=64):
    print(visible_gpu())
    proc = load_processor(cfg)
    # Quantized inference supports 32B on a single GPU with enough memory.
    from transformers import BitsAndBytesConfig
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_use_double_quant=True,
                              bnb_4bit_compute_dtype=torch.bfloat16)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        cfg["model_id"], dtype=torch.bfloat16, quantization_config=quant,
        device_map={"": 0}, attn_implementation="sdpa")
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    ds = DocVQADataset(cfg["data_root"], "validation", max_samples)
    collate = DocVQACollator(proc, max_length=cfg["max_length"], train=False)
    records = []
    for i, sample in enumerate(ds):
        encoded = collate([sample])
        encoded = {k: v.to("cuda") for k, v in encoded.items()}
        prompt_len = encoded["input_ids"].shape[1]
        with torch.inference_mode():
            out = model.generate(**encoded, max_new_tokens=max_new_tokens,
                                 do_sample=False, use_cache=True)
        pred = proc.batch_decode(out[:, prompt_len:], skip_special_tokens=True,
                                 clean_up_tokenization_spaces=False)[0].strip()
        refs = sample["answers"]
        records.append({"question_id": sample["question_id"], "prediction": pred,
                        "references": refs, "anls": anls(pred, refs),
                        "exact_match": int(normalize(pred) in [normalize(r) for r in refs])})
        if (i + 1) % 20 == 0:
            print(f"Evaluated {i+1}/{len(ds)}", flush=True)
    result = {"samples": len(records), "anls": sum(x["anls"] for x in records) / len(records),
              "exact_match": sum(x["exact_match"] for x in records) / len(records),
              "predictions": records}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/qlora.yaml")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--output", default="results/evaluation.json")
    args = parser.parse_args()
    result = evaluate(load_config(args.config), args.adapter, args.max_samples, args.max_new_tokens)
    save_json(result, args.output)
    print(json.dumps({k:v for k,v in result.items() if k != "predictions"}, indent=2))


if __name__ == "__main__":
    main()
