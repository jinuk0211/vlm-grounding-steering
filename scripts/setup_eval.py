"""Download genuine NLTK resources required by the unmodified CHAIR scorer."""

from __future__ import annotations

import argparse
import shlex
import stat
import zipfile
from pathlib import Path, PurePosixPath


def extract_wordnet(archive: Path, destination: Path) -> None:
    """Extract only the WordNet tree after validating every archive member."""
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            normalized = member.filename.replace("\\", "/")
            parts = PurePosixPath(normalized).parts
            target = (destination / normalized).resolve()
            if (
                not parts
                or parts[0] != "wordnet"
                or ".." in parts
                or not target.is_relative_to(destination)
                or stat.S_ISLNK(member.external_attr >> 16)
            ):
                raise ValueError(f"Unsafe or unexpected WordNet archive member: {member.filename}")
        source.extractall(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--download-dir", type=Path, default=Path(".cache/nltk_data"),
        help="Store official NLTK resources here (default: .cache/nltk_data)",
    )
    args = parser.parse_args()
    import nltk

    directory = args.download_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    nltk.data.path.insert(0, str(directory))
    resources = ("wordnet", "wordnet_ic", "sentiwordnet", "punkt", "punkt_tab")
    for resource in resources:
        nltk.download(resource, download_dir=str(directory), raise_on_error=True)
    if not (directory / "corpora" / "wordnet").is_dir():
        extract_wordnet(directory / "corpora" / "wordnet.zip", directory / "corpora")
    # Exercise the same dependencies as CHAIR. Do not substitute a tokenizer.
    from pattern.en import singularize

    if singularize("birds") != "bird" or not nltk.word_tokenize("Two cats."):
        raise RuntimeError("CHAIR dependency smoke check failed")
    print(f"\nCHAIR dependencies checked; resources saved in {directory}")
    print("Set this in each shell before evaluation:")
    print(f"  Linux/macOS: export NLTK_DATA={shlex.quote(str(directory))}")
    powershell_path = str(directory).replace("'", "''")
    print(f"  PowerShell:  $env:NLTK_DATA = '{powershell_path}'")


if __name__ == "__main__":
    main()
