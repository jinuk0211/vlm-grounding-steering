"""Opt-in fetching of pinned source checkouts, never executed automatically."""

from __future__ import annotations

import json
import subprocess
from importlib.resources import files
from pathlib import Path


def source_catalog() -> dict:
    text = files("vlm_grounding").joinpath("resources/upstream.json").read_text(encoding="utf-8")
    return json.loads(text)


def _git(arguments: list[str], path: Path) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=path,
        capture_output=True,
        text=True,
        check=True,
        timeout=300,
    )
    return result.stdout.strip()


def fetch_sources(names: list[str], destination: str | Path) -> dict:
    """Fetch only catalogued HTTPS repositories and verify the exact commit."""
    catalog = source_catalog()["sources"]
    unknown = set(names) - catalog.keys()
    if unknown:
        raise ValueError(f"Unknown sources: {sorted(unknown)}")
    root = Path(destination).resolve()
    root.mkdir(parents=True, exist_ok=True)
    result = {}
    for name in dict.fromkeys(names):
        source = catalog[name]
        target = root / name
        if target.exists():
            current = _git(["rev-parse", "HEAD"], target)
            if current != source["commit"] or _git(["status", "--porcelain"], target):
                raise ValueError(f"{target} differs from the pinned clean checkout; not modifying it")
        else:
            target.mkdir()
            _git(["init"], target)
            _git(["remote", "add", "origin", source["url"]], target)
            _git(["fetch", "--depth", "1", "origin", source["commit"]], target)
            _git(["checkout", "--detach", "FETCH_HEAD"], target)
            current = _git(["rev-parse", "HEAD"], target)
            if current != source["commit"]:
                raise RuntimeError(f"Fetched commit mismatch for {name}")
        result[name] = {"path": str(target), "commit": current, "url": source["url"]}
    return result
