"""Utility to rewrite `prismatic.*` imports under an OpenVLA-aware namespace."""

from __future__ import annotations

import argparse
import pathlib
import textwrap
from typing import Iterable


OLD_PREFIX = "prismatic."
NEW_PREFIX = "vla_arena.models.univla.prismatic."


def find_files(base_dir: pathlib.Path) -> Iterable[pathlib.Path]:
    for path in base_dir.rglob("*"):
        if path.is_file():
            yield path


def rewrite_file(path: pathlib.Path, dry_run: bool) -> bool:
    try:
        data = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return False

    updated = data.replace(OLD_PREFIX, NEW_PREFIX)
    if updated == data:
        return False

    if dry_run:
        print(f"[dry-run] would rewrite {path}")
        return True

    path.write_text(updated, encoding="utf-8")
    print(f"rewrote {path}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=textwrap.dedent("""
        Walks a directory tree and rewrites occurrences of `prismatic.` to
        `vla_arena.models.openvla.prismatic.` so import statements stay correct.
    """))
    parser.add_argument("path", type=pathlib.Path, help="Folder to process")
    parser.add_argument("--dry-run", action="store_true", help="Only print files that would be changed")
    args = parser.parse_args()

    processed = 0
    for file_path in find_files(args.path):
        if rewrite_file(file_path, dry_run=args.dry_run):
            processed += 1

    print(f"{processed} files updated" if not args.dry_run else f"{processed} files would be updated")


if __name__ == "__main__":
    main()