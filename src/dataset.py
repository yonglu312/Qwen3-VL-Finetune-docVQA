"""Read official DocVQA Task 1 JSON and locally downloaded images.

Expected: data/train/annotations.json and data/train/images/<image_name>.
The official format uses data['data'] with image, question, answers, questionId.
No official test labels are needed/assumed.
"""
import json
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


class DocVQADataset(Dataset):
    def __init__(self, data_root, split="train", max_samples=None):
        self.folder = Path(data_root) / split
        annotation = self.folder / "annotations.json"
        if not annotation.exists():
            raise FileNotFoundError(f"Missing {annotation}; see README dataset setup")
        raw = json.loads(annotation.read_text(encoding="utf-8"))
        entries = raw["data"] if isinstance(raw, dict) and "data" in raw else raw
        self.entries = entries[:max_samples] if max_samples is not None else entries
        self.images = self.folder / "images"
        if not self.entries:
            raise ValueError(f"No examples in {annotation}")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        item = self.entries[index]
        name = item.get("image") or item.get("image_path")
        if not name:
            raise ValueError(f"No image name for entry {index}")
        path = Path(name)
        if not path.is_absolute():
            path = self.images / path.name
        with Image.open(path) as im:
            image = im.convert("RGB")
        answers = item.get("answers") or ([item["answer"]] if item.get("answer") else [])
        return {
            "image": image,
            "question": item["question"],
            "answer": str(answers[0]) if answers else "",
            "answers": [str(a) for a in answers],
            "question_id": item.get("questionId", item.get("question_id", index)),
        }
