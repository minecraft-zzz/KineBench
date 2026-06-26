#!/usr/bin/env python
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


DEFAULT_SOURCES = {
    "ManiSkill-main": "/zzz_new/projects/gemini/VideoX_Fun_GRPO/ManiSkill-main",
    "FoundationPose-main": "/zzz_new/projects/gemini/FoundationPose-main",
    "pyroki-main": "/zzz_new/projects/gemini/pyroki-main",
    "MoGe-main": "/zzz_new/projects/gemini/VideoX_Fun_GRPO/MoGe-main",
}

EXCLUDE_NAMES = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "build",
    "dist",
    "outputs",
    "debug",
    "logs",
    "weights",
    "wandb",
}
EXCLUDE_SUFFIXES = {".pt", ".pth", ".ckpt", ".safetensors", ".onnx", ".engine", ".mp4", ".avi", ".mov", ".npy", ".npz"}


def ignore(dirpath: str, names: list[str]) -> set[str]:
    return {name for name in names if name in EXCLUDE_NAMES or Path(name).suffix in EXCLUDE_SUFFIXES}


def copy_vendor(name: str, source: str | Path, target_root: Path, overwrite: bool = False) -> None:
    src = Path(source)
    dst = target_root / name
    if not src.exists():
        raise FileNotFoundError(f"Missing source for {name}: {src}")
    if dst.exists():
        if not overwrite:
            print(f"[skip] {dst} already exists")
            return
        shutil.rmtree(dst)
    print(f"[copy] {src} -> {dst}")
    shutil.copytree(src, dst, ignore=ignore, symlinks=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Vendor third-party repositories into third_party/.")
    parser.add_argument("--target", default="third_party")
    parser.add_argument("--overwrite", action="store_true")
    for name, src in DEFAULT_SOURCES.items():
        parser.add_argument(f"--{name.replace('-main', '').lower()}-src", default=src)
    args = parser.parse_args()
    target = Path(args.target)
    target.mkdir(parents=True, exist_ok=True)
    sources = {
        "ManiSkill-main": args.maniskill_src,
        "FoundationPose-main": args.foundationpose_src,
        "pyroki-main": args.pyroki_src,
        "MoGe-main": args.moge_src,
    }
    for name, src in sources.items():
        copy_vendor(name, src, target, overwrite=args.overwrite)


if __name__ == "__main__":
    main()

