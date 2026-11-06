"""Convert locally downloaded official DocVQA splits to repository format.

python scripts/prepare_docvqa.py --annotations path/to/docvqa_train_v1.0.json \
  --images path/to/train_images --split train
"""
import argparse
import json
import shutil
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--annotations", required=True)
    p.add_argument("--images", required=True)
    p.add_argument("--split", required=True, choices=["train", "validation", "test"])
    p.add_argument("--data-root", default="data")
    p.add_argument("--copy-images", action="store_true", help="Copy images rather than symlink")
    args = p.parse_args()
    src = Path(args.annotations)
    raw = json.loads(src.read_text(encoding="utf-8"))
    items = raw["data"] if isinstance(raw, dict) else raw
    dst = Path(args.data_root) / args.split
    target_images = dst / "images"
    target_images.mkdir(parents=True, exist_ok=True)
    missing = []
    for item in items:
        name = Path(item["image"]).name
        origin = Path(args.images) / name
        target = target_images / name
        if not origin.exists():
            missing.append(str(origin))
            continue
        if not target.exists():
            if args.copy_images:
                shutil.copy2(origin, target)
            else:
                target.symlink_to(origin.resolve())
        item["image"] = name
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} images; first: {missing[:3]}")
    (dst / "annotations.json").write_text(json.dumps({"data": items}, indent=2), encoding="utf-8")
    print(f"Prepared {len(items)} items in {dst}")


if __name__ == "__main__":
    main()
