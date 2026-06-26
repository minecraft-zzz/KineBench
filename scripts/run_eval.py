#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from kinebench import KineBenchEvaluator, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run KineBench evaluation.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--env-id", default=None, help="Override environment id.")
    parser.add_argument("--num-episodes", type=int, default=None, help="Override number of episodes.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.env_id:
        cfg["env_id"] = args.env_id
    if args.num_episodes is not None:
        cfg["num_episodes"] = args.num_episodes
    out = KineBenchEvaluator(cfg).run(env_id=cfg.get("env_id"), num_episodes=cfg.get("num_episodes"))
    print(f"KineBench results written to: {Path(out).resolve()}")


if __name__ == "__main__":
    main()

