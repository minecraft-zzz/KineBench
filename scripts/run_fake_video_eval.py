#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kinebench import KineBenchEvaluator, load_config

DEFAULT_FAKE_VIDEO = Path(
    "/data/zzz/kinebench/datasets/hf_datasets/hf_datasets_new/suite0/videos/videos/close_faucet_31/observation.images.base_camera.mp4"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run KineBench while pretending the video generator returned a fixed local video."
    )
    parser.add_argument(
        "--config",
        default="configs/eval/fake_close_box_video.yaml",
        help="YAML config to load before overriding generation with the fake video.",
    )
    parser.add_argument("--video-path", default=str(DEFAULT_FAKE_VIDEO), help="Fixed mp4/npy/npz returned by the fake generator.")
    parser.add_argument("--env-id", default=None, help="Override env id.")
    parser.add_argument("--num-episodes", type=int, default=None, help="Override episode count.")
    args = parser.parse_args()

    video_path = Path(args.video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Fake video does not exist: {video_path}")

    cfg = deepcopy(load_config(args.config))
    cfg.setdefault("generation", {})
    cfg["generation"].update({"kind": "local", "path": str(video_path)})
    if args.env_id:
        cfg["env_id"] = args.env_id
    if args.num_episodes is not None:
        cfg["num_episodes"] = args.num_episodes

    out = KineBenchEvaluator(cfg).run(env_id=cfg.get("env_id"), num_episodes=cfg.get("num_episodes"))
    print(f"Fake-video KineBench results written to: {Path(out).resolve()}")
    print(f"Fake generator source video: {video_path}")


if __name__ == "__main__":
    main()
