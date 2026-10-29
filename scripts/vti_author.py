"""Prepare pinned official VTI sources; load models only for extract/run."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
URL = "https://github.com/shengliu66/VTI.git"
COMMIT = "bb5dfee33a9e555f22af9faa7b3495c0ac3df69e"
DEFAULT_REPO = ROOT / "upstream" / "vti-author"
REQUIRED = (
    "vti_utils/pca.py",
    "vti_utils/utils.py",
    "vti_utils/llm_layers.py",
    "experiments/data/hallucination_vti_demos.jsonl",
    "experiments/eval/run_mmhal_vti.py",
    "requirement.txt",
)


def git(repo: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *arguments], check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_source(repo: Path) -> dict:
    repo = repo.resolve()
    if not repo.is_dir():
        raise FileNotFoundError(f"Clone VTI first: python scripts/vti_author.py prepare --repo {repo}")
    if Path(git(repo, "rev-parse", "--show-toplevel")).resolve() != repo:
        raise ValueError("--repo must point to the VTI Git checkout itself")
    if git(repo, "rev-parse", "HEAD") != COMMIT:
        raise ValueError(f"VTI must be checked out at {COMMIT}; existing checkouts are never reset")
    if git(repo, "remote", "get-url", "origin").removesuffix(".git").rstrip("/") != URL[:-4]:
        raise ValueError("VTI origin must be the official author repository")
    if git(repo, "status", "--porcelain", "--untracked-files=no"):
        raise ValueError("VTI tracked sources are modified; prepare a separate clean checkout")
    missing = [name for name in REQUIRED if not (repo / name).is_file() or not (repo / name).stat().st_size]
    if missing:
        raise ValueError(f"VTI checkout is incomplete: {missing}")
    return {"url": URL, "commit": COMMIT, "files": {name: sha(repo / name) for name in REQUIRED}}


def prepare(repo: Path) -> dict:
    repo = repo.resolve()
    if not repo.exists():
        repo.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "-c", "core.autocrlf=false", "clone", "--no-checkout", URL, str(repo)],
            check=True,
        )
        git(repo, "-c", "core.autocrlf=false", "checkout", "--detach", COMMIT)
    return verify_source(repo)


def demos(repo: Path, image_root: Path, count: int) -> list[dict]:
    """Use official random.sample with the fixed seed 42."""
    source = repo / "experiments/data/hallucination_vti_demos.jsonl"
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    if count < 2 or count > len(rows):
        raise ValueError(f"--num-demos must be between 2 and {len(rows)}")
    selected = []
    for row in random.Random(42).sample(rows, count):
        for key in ("id", "image", "question", "value", "h_value"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"Official demo has invalid {key}")
        path = (image_root / row["image"]).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Missing COCO training image: {path}")
        selected.append({**row, "image_path": str(path), "image_sha256": sha(path),
                         "group_id": f"coco:{int(row['id'])}"})
    return selected


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    for name in ("prepare", "check", "extract", "run"):
        command = commands.add_parser(name)
        command.add_argument("--repo", type=Path, default=DEFAULT_REPO)
        if name == "check":
            command.add_argument("--image-root", type=Path, help="COCO train2014 directory")
            command.add_argument("--num-demos", type=int, default=70)
        if name in ("extract", "run"):
            command.add_argument("--config", type=Path, required=True)
            command.add_argument("--output", type=Path, required=True)
            command.add_argument("--overwrite", action="store_true")
        if name == "extract":
            command.add_argument("--image-root", type=Path, required=True)
            command.add_argument("--num-demos", type=int, default=70)
            command.add_argument("--num-trials", type=int, default=50)
            command.add_argument("--mask-ratio", type=float, default=0.99)
            command.add_argument("--alpha-text", type=float, default=0.9)
            command.add_argument("--alpha-image", type=float, default=0.9)
            command.add_argument("--seed", type=int, default=42)
        if name == "run":
            command.add_argument("--artifact", type=Path, required=True)
            command.add_argument("--manifest", type=Path, required=True)
            command.add_argument("--split", choices=("tune", "test"), default="test")
            command.add_argument("--task", choices=("caption", "pope", "mme", "mmhal", "causal"))
            command.add_argument("--limit", type=int)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    args.repo = args.repo.resolve()
    if args.command == "prepare":
        result = {"status": "prepared", "repo": str(args.repo), "source": prepare(args.repo)}
    else:
        source = verify_source(args.repo)
        if args.command == "check":
            result = {"status": "source_verified", "repo": str(args.repo), "source": source,
                      "model_loaded": False, "gpu_used": False}
            if args.image_root:
                rows = demos(args.repo, args.image_root, args.num_demos)
                result["demo_images_verified"] = len(rows)
        else:
            # Preparation/check/help do not import torch, load a model, or touch CUDA.
            sys.path.insert(0, str(ROOT / "src"))
            from vti_author_gemma import extract, run

            result = extract(args, source) if args.command == "extract" else run(args, source)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
