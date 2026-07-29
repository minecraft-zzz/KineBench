#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kinebench import KineBenchEvaluator, load_config
from scripts.run_all_fake_video_eval import (
    DEFAULT_DATASET_ROOT,
    TASK_PROMPTS,
    VIDEO_NAME,
    parse_background_type_id,
)

ENV_ID = "StoreCube-v1"
TASK_NAME = "store_cube_v1"


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def discover_store_cube_v1_videos(dataset_root: Path) -> list[tuple[int, Path]]:
    videos = []
    for child in dataset_root.glob(f"{TASK_NAME}_*"):
        if not child.is_dir():
            continue
        suffix = child.name.removeprefix(f"{TASK_NAME}_")
        try:
            dataset_index = int(suffix)
        except ValueError:
            continue
        video_path = child / VIDEO_NAME
        if video_path.exists():
            videos.append((dataset_index, video_path))
    return sorted(videos, key=lambda item: item[0])


def select_video(videos: list[tuple[int, Path]], number: int, selection_mode: str) -> tuple[int, int, Path]:
    if selection_mode == "suffix":
        by_suffix = {dataset_index: video_path for dataset_index, video_path in videos}
        if number not in by_suffix:
            raise FileNotFoundError(f"No {TASK_NAME}_{number} video found.")
        return number, number, by_suffix[number]

    offset = number - 1
    if offset < 0 or offset >= len(videos):
        raise IndexError(f"Ordinal {number} is out of range; only {len(videos)} videos found.")
    dataset_index, video_path = videos[offset]
    return number, dataset_index, video_path


def build_config(
    base_config: Path,
    output_dir: Path,
    run_name: str,
    background: str,
    background_type_id: list[int],
    video_path: Path,
    disable_gripper: bool,
) -> dict:
    cfg = load_config(base_config)
    cfg["env_id"] = ENV_ID
    cfg["num_episodes"] = 1
    cfg["timestamp_outputs"] = False
    cfg["run_name"] = run_name
    cfg["output_dir"] = str(output_dir)
    cfg.setdefault("generation", {}).update({"kind": "local", "path": str(video_path)})
    cfg.setdefault("runtime", {}).update(
        {
            "background": background,
            "background_type_id": background_type_id,
        }
    )
    cfg.setdefault("prompts", {}).update({"StoreCube-v1": TASK_PROMPTS["StoreCube-v1"]})
    if disable_gripper:
        cfg.setdefault("gripper", {})["enabled"] = False
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small KineBench fake-video test on StoreCube-v1.")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/eval/fake_close_faucet_video.yaml")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs")
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--backgrounds",
        default="Table,RoboCasa",
        help="Comma-separated backgrounds. Use Table, RoboCasa, or both.",
    )
    parser.add_argument("--table-video", type=int, default=1, help="Table video number.")
    parser.add_argument("--robocasa-video", type=int, default=51, help="RoboCasa video number.")
    parser.add_argument(
        "--selection-mode",
        choices=["ordinal", "suffix"],
        default="ordinal",
        help="ordinal means 1-based sorted position; suffix means exact directory suffix.",
    )
    parser.add_argument("--background-type-id", type=parse_background_type_id, default=[-1, -1])
    parser.add_argument("--disable-gripper", action="store_true", help="Use constant gripper=-1 instead of DINOv3.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    videos = discover_store_cube_v1_videos(args.dataset_root)
    if not videos:
        raise FileNotFoundError(f"No {TASK_NAME} videos found under {args.dataset_root}")

    backgrounds = [item.strip() for item in args.backgrounds.split(",") if item.strip()]
    batch_name = args.run_name or f"store_cube_v1_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    batch_dir = args.output_dir / batch_name

    selected = []
    for background in backgrounds:
        normalized = background.lower()
        if normalized == "table":
            number = args.table_video
            label = "Table"
        elif normalized in {"robocasa", "robo_casa"}:
            number = args.robocasa_video
            label = "RoboCasa"
        else:
            raise ValueError(f"Unknown background: {background}")
        ordinal, dataset_index, video_path = select_video(videos, number, args.selection_mode)
        selected.append((label, ordinal, dataset_index, video_path))

    for background, ordinal, dataset_index, video_path in selected:
        print(
            f"{ENV_ID} | {background} | ordinal={ordinal} | "
            f"suffix={dataset_index} | video={video_path}"
        )
    if args.dry_run:
        return

    for background, ordinal, dataset_index, video_path in selected:
        run_name = f"{TASK_NAME}_{background.lower()}_{dataset_index:04d}"
        cfg = build_config(
            base_config=args.config,
            output_dir=batch_dir,
            run_name=run_name,
            background=background,
            background_type_id=args.background_type_id,
            video_path=video_path,
            disable_gripper=args.disable_gripper,
        )
        out = KineBenchEvaluator(cfg).run(env_id=ENV_ID, num_episodes=1)
        print(f"Written: {Path(out).resolve()}")


if __name__ == "__main__":
    main()
