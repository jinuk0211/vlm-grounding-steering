"""Local dataset conversion and image-group isolation; no dataset downloads."""

from __future__ import annotations

import json
import os
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlsplit

Record = dict[str, Any]
SPLITS = {"discovery", "fit", "tune", "test"}
TASKS = {"pope", "caption", "causal", "mme", "mmhal"}
CAPTION_PROMPT = "Please describe this image in detail."
YES_NO_SUFFIX = " Answer yes or no."


def _read_json_records(path: str | Path) -> list[Record]:
    text = Path(path).read_text(encoding="utf-8-sig")
    if not text.strip():
        return []
    if text.lstrip().startswith("["):
        records = json.loads(text)
    else:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(records, list) or any(not isinstance(r, dict) for r in records):
        raise ValueError(f"{path}: expected a JSON array or JSONL objects")
    return records


def read_manifest(path: str | Path) -> list[Record]:
    """Read JSON/JSONL; resolve relative image paths against the manifest folder."""
    path = Path(path).resolve()
    records = _read_json_records(path)
    for record in records:
        for key in ("image", "object_masked_image", "control_image"):
            if key in record:
                image = Path(record[key])
                record[key] = str(
                    (path.parent / image).resolve() if not image.is_absolute() else image
                )
    validate_disjoint(records)
    return records


def write_manifest(records: Sequence[Mapping[str, Any]], path: str | Path) -> None:
    """Write a validated JSONL manifest, including absolute local image paths."""
    validate_disjoint(records)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(dict(record), ensure_ascii=False) + "\n")


def validate_disjoint(
    manifests: Iterable[Mapping[str, Any] | str | Path | Sequence[Mapping[str, Any]]],
    *,
    check_files: bool = True,
) -> dict[str, int]:
    """Reject duplicate IDs and image/original-image leakage across named splits.

    Reusing an image for different questions or tasks in the SAME split is valid.
    Original and edited images MUST use the same group_id.
    """
    records: list[Mapping[str, Any]] = []
    for item in manifests:
        if isinstance(item, Mapping):
            records.append(item)
        elif isinstance(item, (str, Path)):
            records.extend(read_manifest(item))
        else:
            records.extend(item)
    seen_ids: set[str] = set()
    groups: dict[str, str] = {}
    files: dict[str, tuple[str, str]] = {}
    split_counts: defaultdict[str, int] = defaultdict(int)
    for record in records:
        missing = {"id", "group_id", "image", "prompt", "task", "split"} - record.keys()
        if missing:
            raise ValueError(f"Record is missing fields: {sorted(missing)}")
        identifier, group, split = str(record["id"]), str(record["group_id"]), record["split"]
        if not identifier or not group:
            raise ValueError("id and group_id must be nonempty")
        if identifier in seen_ids:
            raise ValueError(f"Duplicate record id: {identifier}")
        seen_ids.add(identifier)
        if split not in SPLITS or record["task"] not in TASKS:
            raise ValueError(f"{identifier}: unsupported split/task")
        if not isinstance(record["prompt"], str) or not record["prompt"].strip():
            raise ValueError(f"{identifier}: prompt must be a nonempty string")
        if record.get("label") not in (None, "yes", "no"):
            raise ValueError(f"{identifier}: label must be yes/no or absent")
        if group in groups and groups[group] != split:
            raise ValueError(f"Image-group leakage: {group} in {groups[group]} and {split}")
        groups[group] = split
        for field in ("image", "object_masked_image", "control_image"):
            if field not in record:
                continue
            path = Path(record[field]).resolve()
            if check_files and not path.is_file():
                raise FileNotFoundError(f"{identifier}: missing {field}: {path}")
            key = str(path).casefold() if os.name == "nt" else str(path)
            if key in files and files[key] != (group, split):
                raise ValueError(f"Image file reused under different groups/splits: {path}")
            files[key] = (group, split)
        split_counts[split] += 1
    return dict(split_counts)


def image_group_id(image_name: str) -> str:
    """Map COCO originals and Causal-HalBench _NNN edits to the same image ID."""
    stem = Path(image_name).stem
    match = re.search(r"(?:^|_)0*(\d{1,12})(?:_\d{3})?$", stem)
    if stem.startswith("COCO_") and match:
        return f"coco:{int(match.group(1))}"
    if stem.isdigit():
        return f"coco:{int(stem)}"
    base = re.sub(r"_\d{3}$", "", stem)
    return f"image:{base}"


def _local_image(root: str | Path, name: str) -> str:
    root = Path(root).resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Image path escapes image_root: {name}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return str(path)


def _yes_no_label(value: Any) -> str:
    label = str(value).strip().lower()
    if label not in {"yes", "no"}:
        raise ValueError(f"Ground-truth label must be yes/no, got {value!r}")
    return label


def prepare_pope(
    input_path: str | Path,
    image_root: str | Path,
    split: str = "test",
    *,
    subset: str | None = None,
) -> list[Record]:
    """Convert official POPE question_id/image/text/label JSONL or JSON."""
    subset = subset or Path(input_path).stem
    records = [
        {
            "id": f"pope:{subset}:{row['question_id']}",
            "group_id": image_group_id(row["image"]),
            "image": _local_image(image_root, row["image"]),
            "prompt": row["text"],
            "label": _yes_no_label(row["label"]),
            "task": "pope",
            "split": split,
            "metadata": {"subset": subset, "question_id": row["question_id"]},
        }
        for row in _read_json_records(input_path)
    ]
    validate_disjoint(records)
    return records


def prepare_causal(
    input_path: str | Path, image_root: str | Path, split: str = "test"
) -> list[Record]:
    """Convert official Causal-HalBench qa.json (question, answer, image_name)."""
    records = []
    for row in _read_json_records(input_path):
        for key in ("image_name", "question", "answer", "id", "tag", "type"):
            if key not in row:
                raise ValueError(f"Causal-HalBench entry missing {key}")
        records.append(
            {
                "id": f"causal:{row['id']}",
                "group_id": image_group_id(row["image_name"]),
                "image": _local_image(image_root, row["image_name"]),
                "prompt": row["question"],
                "label": _yes_no_label(row["answer"]),
                "task": "causal",
                "split": split,
                "metadata": {"causal_qa": dict(row)},
            }
        )
    validate_disjoint(records)
    return records


def prepare_mme(root: str | Path, split: str = "test") -> list[Record]:
    """Read official category/{images,questions_answers_YN} layout or flat categories."""
    root = Path(root)
    records = []
    for category in sorted(p for p in root.iterdir() if p.is_dir()):
        question_dir = category / "questions_answers_YN"
        image_dir = category / "images"
        if not question_dir.is_dir():
            question_dir, image_dir = category, category
        for question_file in sorted(question_dir.glob("*.txt")):
            candidates = [
                image_dir / f"{question_file.stem}{extension}"
                for extension in (".jpg", ".png", ".jpeg", ".JPEG", ".JPG")
            ]
            candidates = [p for p in candidates if p.is_file()]
            if len(candidates) != 1:
                raise ValueError(f"{question_file}: expected exactly one matching image")
            lines = question_file.read_text(encoding="utf-8-sig").splitlines()
            for index, line in enumerate(line for line in lines if line.strip()):
                fields = line.rsplit("\t", 1)
                if len(fields) != 2:
                    raise ValueError(f"{question_file}: expected question TAB yes/no")
                records.append(
                    {
                        "id": f"mme:{category.name}:{question_file.stem}:{index}",
                        "group_id": f"mme:{category.name}:{question_file.stem}",
                        "image": str(candidates[0].resolve()),
                        "prompt": fields[0],
                        "label": _yes_no_label(fields[1]),
                        "task": "mme",
                        "split": split,
                        "metadata": {"category": category.name, "question_index": index},
                    }
                )
    if not records:
        raise ValueError(f"No MME question files found below {root}")
    validate_disjoint(records)
    return records


def prepare_mmhal(
    input_path: str | Path, image_root: str | Path, split: str = "test"
) -> list[Record]:
    """Convert the ordered official MMHal-Bench response_template.json and local images."""
    records = []
    for index, row in enumerate(_read_json_records(input_path)):
        required = {"image_id", "image_src", "image_content", "question",
                    "gt_answer", "question_type", "question_topic"}
        if required - row.keys():
            raise ValueError(f"MMHal entry {index} missing {sorted(required - row.keys())}")
        image_name = urlsplit(row["image_src"]).path.rsplit("/", 1)[-1]
        if not image_name:
            raise ValueError(f"MMHal entry {index} has no image filename")
        records.append({
            "id": f"mmhal:{index:03d}",
            "group_id": f"mmhal:{row['image_id']}",
            "image": _local_image(image_root, image_name),
            "prompt": row["question"],
            "task": "mmhal", "split": split,
            "metadata": {
                "mmhal_index": index,
                "image_id": row["image_id"], "image_src": row["image_src"],
                "image_content": row["image_content"], "gt_answer": row["gt_answer"],
                "question_type": row["question_type"],
                "question_topic": row["question_topic"],
            },
        })
    if not records:
        raise ValueError("MMHal template is empty")
    validate_disjoint(records)
    return records


def _segmentation_mask(segmentation: Any, width: int, height: int):
    """Use actual polygon masks; RLE requires pycocotools, never substitutes boxes."""
    import numpy as np
    from PIL import Image, ImageDraw

    if isinstance(segmentation, list):
        canvas = Image.new("1", (width, height))
        draw = ImageDraw.Draw(canvas)
        for polygon in segmentation:
            if len(polygon) < 6 or len(polygon) % 2:
                raise ValueError("Invalid COCO segmentation polygon")
            draw.polygon(list(zip(polygon[::2], polygon[1::2])), fill=1)
        return np.asarray(canvas, dtype=bool)
    if isinstance(segmentation, dict):
        try:
            from pycocotools import mask as mask_utils
        except ImportError as exc:
            raise ImportError("COCO RLE masks require pip install pycocotools") from exc
        rle = segmentation
        if isinstance(rle["counts"], list):
            rle = mask_utils.frPyObjects(rle, height, width)
        elif isinstance(rle["counts"], str):
            rle = {**rle, "counts": rle["counts"].encode("ascii")}
        mask = mask_utils.decode(rle).astype(bool)
        if mask.shape != (height, width):
            raise ValueError("COCO RLE mask size does not match its image")
        return mask
    raise ValueError("Missing/unsupported segmentation: bounding boxes are not masks")


def _translated_control(target, all_objects, rng: random.Random, attempts: int = 512):
    """Find an equal-shape/area translated mask that touches no annotated object."""
    import numpy as np

    ys, xs = np.nonzero(target)
    if not len(xs):
        return None
    crop = target[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    height, width = target.shape
    crop_height, crop_width = crop.shape
    for _ in range(attempts):
        x = rng.randrange(width - crop_width + 1)
        y = rng.randrange(height - crop_height + 1)
        if not (all_objects[y : y + crop_height, x : x + crop_width] & crop).any():
            control = np.zeros_like(target)
            control[y : y + crop_height, x : x + crop_width] = crop
            return control
    return None


def prepare_coco(
    annotation_path: str | Path,
    image_root: str | Path,
    output_dir: str | Path,
    *,
    seed: int = 42,
    discovery_count: int = 200,
    fit_count: int = 200,
    tune_count: int = 200,
    test_count: int = 0,
    exclude_group_ids: Iterable[str] = (),
) -> list[Record]:
    """Build disjoint COCO groups, diagnostic occlusions, and balanced yes/no pairs.

    Counts refer to images, not questions. Discovery creates one positive question
    per image with two diagnostic occlusions; fit/tune create one positive and one
    annotation-absent question plus a caption prompt; test creates captions only.
    """
    import numpy as np
    from PIL import Image

    counts = {
        "discovery": discovery_count, "fit": fit_count, "tune": tune_count, "test": test_count
    }
    if any(value < 0 for value in counts.values()):
        raise ValueError("Image counts must be nonnegative")
    annotations = json.loads(Path(annotation_path).read_text(encoding="utf-8"))
    categories = {row["id"]: row["name"] for row in annotations["categories"]}
    by_image: defaultdict[int, list[Record]] = defaultdict(list)
    for row in annotations["annotations"]:
        by_image[row["image_id"]].append(row)
    excluded = set(exclude_group_ids)
    images = [
        row for row in sorted(annotations["images"], key=lambda r: r["id"])
        if by_image[row["id"]] and f"coco:{row['id']}" not in excluded
    ]
    rng = random.Random(seed)
    rng.shuffle(images)
    candidates = iter(images)
    output_dir = Path(output_dir).resolve()
    records: list[Record] = []
    for split, count in counts.items():
        accepted = 0
        while accepted < count:
            try:
                image_info = next(candidates)
            except StopIteration as exc:
                raise ValueError(
                    f"Insufficient eligible images for {split}: {accepted}/{count}. "
                    "Discovery requires an object mask and same-shape background control."
                ) from exc
            image_id = image_info["id"]
            group = f"coco:{image_id}"
            image_path = _local_image(image_root, image_info["file_name"])
            objects = by_image[image_id]
            present = sorted({row["category_id"] for row in objects})
            absent = sorted(set(categories) - set(present))
            common = {
                "group_id": group, "image": image_path, "split": split,
                "metadata": {
                    "coco_image_id": image_id,
                    "present_objects": [categories[key] for key in present],
                    "annotation_source": str(Path(annotation_path).resolve()),
                    "seed": seed,
                },
            }
            if split == "discovery":
                with Image.open(image_path) as original:
                    image = original.convert("RGB")
                width, height = image.size
                masks: dict[int, Any] = {}
                for obj in objects:
                    mask = _segmentation_mask(obj.get("segmentation"), width, height)
                    category = obj["category_id"]
                    masks[category] = masks.get(category, np.zeros_like(mask)) | mask
                all_objects = np.logical_or.reduce(list(masks.values()))
                selected = None
                for category in sorted(masks, key=lambda key: int(masks[key].sum())):
                    target = masks[category]
                    area = int(target.sum())
                    if not 0.005 <= area / (width * height) <= 0.3:
                        continue
                    control = _translated_control(target, all_objects, rng)
                    if control is not None:
                        selected = category, target, control
                        break
                if selected is None:
                    continue
                category, target, control = selected
                mask_dir = output_dir / "occlusions"
                mask_dir.mkdir(parents=True, exist_ok=True)
                variants = {}
                original_array = np.asarray(image)
                for name, mask in (("object_masked_image", target), ("control_image", control)):
                    pixels = original_array.copy()
                    pixels[mask] = (128, 128, 128)
                    path = mask_dir / f"{image_id}_{category}_{name}.png"
                    Image.fromarray(pixels).save(path)
                    variants[name] = str(path)
                records.append({
                    **common,
                    "id": f"coco:discovery:{image_id}:{category}",
                    "task": "pope",
                    "prompt": f"Is there a {categories[category]} in the image?{YES_NO_SUFFIX}",
                    "label": "yes",
                    **variants,
                    "metadata": {
                        **common["metadata"], "object_category_id": category,
                        "object_name": categories[category],
                        "mask_pixels": int(target.sum()),
                        "control_pixels": int(control.sum()),
                        "occlusion_fill_rgb": [128, 128, 128],
                        "mask_source": "COCO segmentation; PIL polygons / pycocotools RLE",
                        "diagnostic_only": True,
                    },
                })
            else:
                if split in {"fit", "tune"}:
                    if not absent:
                        continue
                    for category, label in ((rng.choice(present), "yes"), (rng.choice(absent), "no")):
                        records.append({
                            **common, "id": f"coco:{split}:{image_id}:{label}",
                            "task": "pope",
                            "prompt": f"Is there a {categories[category]} in the image?{YES_NO_SUFFIX}",
                            "label": label,
                            "chosen": "Yes." if label == "yes" else "No.",
                            "rejected": "No." if label == "yes" else "Yes.",
                            "metadata": {
                                **common["metadata"], "object_name": categories[category],
                                "subset": "coco_annotation_balanced",
                                "preference_source": "annotation yes/no; task-adapted, not original paper pairs",
                            },
                        })
                records.append({
                    **common, "id": f"coco:{split}:{image_id}:caption",
                    "task": "caption", "prompt": CAPTION_PROMPT,
                })
            accepted += 1
    validate_disjoint(records)
    return records
