"""Download a HuggingFace repo revision with parallel range requests and check every LFS file's SHA-256.

The server's single connections are slow and the xet transfer stalled, so large files are fetched in parallel chunks.
Usage: python pilot/fetch_hf.py REPO REVISION DEST [--only FILE ...]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CHUNK = 32 * 2**20
WORKERS = 16


def tree(repo: str, revision: str) -> list[dict]:
    with urllib.request.urlopen(f"https://huggingface.co/api/models/{repo}/tree/{revision}?recursive=true", timeout=60) as response:
        return [entry for entry in json.load(response) if entry["type"] == "file"]


def fetch_range(url: str, path: Path, start: int, end: int) -> None:
    for attempt in range(8):
        try:
            request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(request, timeout=120) as response:
                data = response.read()
            if len(data) != end - start + 1:
                raise IOError(f"short read {len(data)} of {end - start + 1}")
            with open(path, "r+b") as handle:
                handle.seek(start)
                handle.write(data)
            return
        except Exception as error:  # retried; the last failure is raised
            if attempt == 7:
                raise
            print(f"  retry {attempt + 1} bytes {start}-{end}: {error}", flush=True)
            time.sleep(2 ** min(attempt, 5))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(2**24), b""):
            digest.update(block)
    return digest.hexdigest()


def download(repo: str, revision: str, entry: dict, dest: Path) -> None:
    path = dest / entry["path"]
    expected = entry.get("lfs", {}).get("oid")
    if path.exists() and path.stat().st_size == entry["size"] and (expected is None or sha256(path) == expected):
        print(f"have {entry['path']}", flush=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/{repo}/resolve/{revision}/{entry['path']}"
    partial = path.with_suffix(path.suffix + ".part")
    with open(partial, "wb") as handle:
        handle.truncate(entry["size"])
    began = time.perf_counter()
    ranges = [(start, min(start + CHUNK, entry["size"]) - 1) for start in range(0, entry["size"], CHUNK)] or [(0, -1)]
    if entry["size"] == 0:
        partial.replace(path)
        return
    with ThreadPoolExecutor(WORKERS) as pool:
        list(pool.map(lambda r: fetch_range(url, partial, *r), ranges))
    seconds = time.perf_counter() - began
    if expected is not None and sha256(partial) != expected:
        raise IOError(f"SHA-256 mismatch for {entry['path']}")
    partial.replace(path)
    print(f"got {entry['path']} {entry['size'] / 1e6:.0f} MB in {seconds:.0f}s ({entry['size'] / 1e6 / max(seconds, 1e-3):.1f} MB/s)"
          + (" sha256 ok" if expected else ""), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo")
    parser.add_argument("revision")
    parser.add_argument("dest", type=Path)
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args()
    entries = [e for e in tree(args.repo, args.revision) if not args.only or e["path"] in args.only]
    for entry in sorted(entries, key=lambda e: e["size"]):
        download(args.repo, args.revision, entry, args.dest)
    (args.dest / "REVISION").write_text(args.revision)
    print("FETCH_OK", flush=True)


if __name__ == "__main__":
    main()
