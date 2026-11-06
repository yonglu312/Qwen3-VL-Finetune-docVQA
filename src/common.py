import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def visible_gpu():
    if not torch.cuda.is_available():
        raise RuntimeError("NVIDIA CUDA GPU required")
    return f"{torch.cuda.get_device_name()} ({torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB)"
