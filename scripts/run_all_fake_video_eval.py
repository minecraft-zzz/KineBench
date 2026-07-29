#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import sys
import traceback
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kinebench import KineBenchEvaluator, load_config


DEFAULT_DATASET_ROOT = Path(
    "/data/zzz/kinebench/datasets/hf_datasets/hf_datasets_new/suite0/videos/videos"
)
VIDEO_NAME = "observation.images.base_camera.mp4"

TASK_ENV_IDS = {
    "pickfruits": "PickFruits-v1",
    "storefruitbox": "StoreFruitsBox-v1",
    # "close_box": "CloseBox-v1",
    # "close_drawer": "CloseDrawer-v1",
    # "close_faucet": "CloseFaucet-v1",
    # "close_laptop_easy": "CloseLaptopEasy-v1",
    # "close_laptop_hard": "CloseLaptopHard-v1",
    "lift_peg": "LiftPegUpright-v1",
    # "open_box_easy_v1": "OpenBoxEasy-v1",
    # "open_box_easy_v2": "OpenBoxEasy-v2",
    # "open_box_hard_v1": "OpenBoxHard-v1",
    # "open_box_hard_v2": "OpenBoxHard-v2",
    # "open_drawer": "OpenDrawer-v1",
    # "open_faucet": "OpenFaucet-v1",
    # "open_laptop_easy": "OpenLaptopEasy-v1",
    # "open_laptop_hard": "OpenLaptopHard-v1",
    # "pull_cube": "PullCube-v1",
    # "stack_cube_v1": "StackCube-v1",
    # "store_cube_v1": "StoreCube-v1",
    # "store_cube_v2": "StoreCube-v2",
}

TASK_PROMPTS = {
    "OpenFaucet-v1": "In the following video, based on the current visual state, determine the task progress and continue to complete the remaining steps. Task: OpenFaucet-v1. A faucet is in front of the robot arm. The faucet may appear in multiple variants (shape/size/style can change), so focus on the handle orientation rather than the exact geometry. The faucet starts closed, with the handle pointing inward (toward the inside of the scene). Move the gripper down to the right side of the handle, make controlled contact, then sweep the gripper left and slightly forward to rotate the handle counter-clockwise until the handle points to the left, indicating the faucet is open. Stop once the handle reaches the fully open orientation.do not rotate clockwise (that would close it more); Do not push the handle in the wrong direction when the faucet model changes; do not ignore the handle orientation.",
    "CloseFaucet-v1": "In the following video, based on the current visual state, determine the task progress and continue to complete the remaining steps. Task: CloseFaucet-v1. A faucet is in front of the robot arm. The faucet model may vary across multiple variants (shape/size may change), so focus on the handle orientation. The faucet starts in an open state with the handle pointing to the left. Move the gripper down and slightly forward to approach the front/right side of the handle, then sweep the gripper to the right and backward to rotate the handle clockwise until the handle points backward (closed position). Maintain controlled contact and stop when the handle reaches the fully closed orientation. Do not rotate counter-clockwise (that would open it more); do not push the handle in the wrong direction; do not ignore the handle orientation when faucet geometry changes.",
    "OpenLaptopEasy-v1": "In the following video, based on the current visual state, determine the task progress and continue to complete the remaining steps. Task: OpenLaptopEasy-v1. A laptop is in front of the robot arm. The laptop lid is almost closed but already slightly ajar (the opening angle is not tiny). Move the gripper down to the front edge of the lid, then push the lid forward and upward in a smooth motion until the laptop is fully open. Because the lid is already partially open, this task can typically be completed by a single continuous forward-up push. Stop when the lid reaches the fully open position, then retract the gripper. Do not press downward to close the lid; do not try to pry or lift aggressively; do not push in a direction that makes the lid close further.",
    "OpenLaptopHard-v1": "In the following video, based on the current visual state, determine the task progress and continue to complete the remaining steps. Task: OpenLaptopHard-v1. A laptop is in front of the robot arm. The laptop lid is tightly closed (very small or near-zero opening angle). Move the gripper down to the front edge of the lid. First, rotate the gripper upward to lift/pry the lid up slightly and create a usable opening angle. After the lid is lifted to a small angle, push forward and upward until the laptop is fully open. This task usually requires two phases: pry up first, then push open. Stop when the lid is fully open, then retract the gripper. Do not only push forward when the lid is fully shut (it won't open); do not press downward; do not attempt to close the laptop.",
    "CloseLaptopEasy-v1": "In the following video, based on the current visual state, determine the task progress and continue to complete the remaining steps. Task: CloseLaptopEasy-v1. A laptop is in front of the robot arm. The laptop lid is open only slightly (small opening angle). Move the gripper forward toward the front/top edge of the lid, then pull the lid backward toward the robot while pressing downward, closing the lid fully in one smooth continuous motion. Because the lid is only slightly open, the forward reach should be short and the closing can be done in a single combined pull-and-press action. Stop once the laptop is completely closed, then retract the gripper upward. Do not try to open the laptop; do not push the lid further backward to increase the opening; do not grasp and lift the laptop.",
    "CloseLaptopHard-v1": "In the following video, based on the current visual state, determine the task progress and continue to complete the remaining steps. Task: CloseLaptopHard-v1. A laptop is in front of the robot arm. The laptop lid is open widely (large opening angle). Reach the gripper forward to the front/top region of the lid. First pull the lid backward toward the robot to reduce the opening angle. After the lid is partially closed, press downward to close it completely. Because the lid starts widely open, the reach should be larger and the closing typically requires two phases: pull back, then press down. Stop once the laptop is fully closed, then retract the gripper upward. Do not try to open the laptop further; do not press straight down too early when the lid is still widely open; do not push from behind the lid in a way that increases the opening.",
    "OpenDrawer-v1": "In the following video, based on the current visual state, determine the task progress and continue to complete the remaining steps. Task: OpenDrawer-v1. A drawer unit is placed sideways in front of the robot arm, and the drawer is slightly open. Rotate the gripper 90 degrees around the vertical (z) axis to align the gripper tip with the drawer gap. Move the gripper down so the tip inserts into the opened gap. Then pull outward along the drawer opening direction to open the drawer fully. Finish when the drawer is clearly open, then retract the gripper. Do not push inward to close the drawer.",
    "CloseDrawer-v1": "In the following video, based on the current visual state, determine the task progress and continue to complete the remaining steps. Task: CloseDrawer-v1. A cabinet/drawer unit is in front of the robot arm, and one drawer is open. Move the gripper forward and slightly downward to reach the inside/back side of the open drawer front. Then push the gripper forward to slide the drawer inward until it is fully closed and flush. Stop once the drawer is completely closed, then retract the gripper upward.",
    "StoreCube-v1": "In the following video, based on the current visual state, determine the task progress and continue to complete the remaining steps. Task: StoreCube-v1. A cabinet/drawer unit is positioned in front of the robot arm, with one drawer open, and a small red cube located above the drawer. Move the gripper toward the small cube and grasp it, then move the gripper above the open drawer and release the gripper so that the cube drops into the drawer. Move the gripper backward and slightly downward to reach the inside/back side of the open drawer front. Then push the gripper forward to slide the drawer inward until it is fully closed and flush with the cabinet.",
    "StoreCube-v2": "In the following video, based on the current visual state, determine the task progress and continue to complete the remaining steps. Task: StoreCube-v2. A cabinet/drawer unit is positioned in front of the robot arm, with one drawer open, and a small red cube located above the drawer. First, lift the gripper to avoid contacting the cube, then close the gripper and move it downward in front of the red cube, and then move the gripper backward to drag the cube along the tabletop until it falls into the drawer. Next, push the drawer inward until it is fully closed.",
    "OpenBoxEasy-v1": "In the following video, based on the current visual state, determine the task progress and continue to complete the remaining steps. Task: OpenBoxEasy-v1. A box is in front of the robot arm. The lid is closed but already slightly open. Move the gripper to the front of the lid, then push the lid forward in one smooth motion until the box is fully open. Stop when the lid is fully open, then retract the gripper.",
    "OpenBoxEasy-v2": "In the following video, based on the current visual state, determine the task progress and continue to complete the remaining steps. Task: OpenBoxEasy-v2. A box with a partially open lid is placed on a tabletop, and the box is oriented toward the right. Move the gripper to the edge of the box lid. Rotate the gripper to face the box, then push the lid upward and forward until the box is fully open.",
    "OpenBoxHard-v1": "In the following video, based on the current visual state, determine the task progress and continue to complete the remaining steps. Task: OpenBoxHard-v1. A box is in front of the robot arm. The lid is fully closed and hard to open. Reach the gripper forward while rotating the gripper so that the gripper tip can approach the side edge of the lid. Use the gripper tip to pry the lid up. Once the lid is lifted to a workable angle, push forward to open the lid fully.",
    "OpenBoxHard-v2": "In the following video, based on the current visual state, determine the task progress and continue to complete the remaining steps. Task: OpenBoxHard-v2. A box is positioned in front of the robot arm, and the box lid is tightly closed. Move the gripper downward to the front edge of the lid. First, rotate the gripper upward to slightly lift/pry the lid and create a usable opening angle. After the lid has been lifted to a small angle, continue pushing forward and upward until the box is fully open.",
    "CloseBox-v1": "In the following video, based on the current visual state, determine the task progress and continue to complete the remaining steps. Task: CloseBox-v1. A box is placed on its side in front of the robot arm. The box lid is open. Move the gripper to hover above the lid, then apply a gentle downward press with a slight rightward component to fold the lid shut until it is fully closed. Finish when the lid is completely closed, then retract the gripper.",
    "LiftPegUpright-v1": "In the following video, based on the current visual state, determine the task progress and continue to complete the remaining steps. Task: LiftPegUpright-v1. A tabletop single-arm robot with a parallel gripper faces a two-colored peg. Lower the gripper to the tip/end of the red section, align the gripper with the peg, close the gripper to grasp the red part, and lift/rotate the peg into an upright standing pose so that the red section points upward. Then move the gripper down and open the gripper to release the peg so it stands on the table by itself.",
    "PullCube-v1": "In the following video, based on the current visual state, determine the task progress and continue to complete the remaining steps. Task: PullCube-v1. A small blue cube is on the tabletop, along with a red-and-white target marker. Close the gripper and move it downward to an appropriate position in front of the blue cube. Then move the gripper straight backward to drag the cube along the tabletop to the center of the target marker. Do not lift the cube.",
    "StackCube-v1": "In the following video, based on the current visual state, determine the task progress and continue to complete the remaining steps. Task: StackCube-v1. There are two small cubes on the tabletop: one red cube and one green cube. At the start, keep the gripper open. Move the gripper to directly above the green cube, lower it slightly, then close the gripper to grasp the green cube. Lift the green cube upward, move it to directly above the red cube, then open the gripper to place the green cube on top of the red cube.",
}


@dataclass(frozen=True)
class Case:
    task_name: str
    env_id: str
    background: str
    ordinal: int
    dataset_index: int
    video_path: Path


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_background_type_id(value: str) -> list[int]:
    ids = parse_int_list(value)
    if len(ids) != 2:
        raise argparse.ArgumentTypeError("--background-type-id must contain exactly two integers, e.g. -1,-1")
    return ids


def split_task_dir_name(path: Path) -> tuple[str, int] | None:
    name = path.name
    prefix, sep, suffix = name.rpartition("_")
    if not sep:
        return None
    try:
        return prefix, int(suffix)
    except ValueError:
        return None


def discover_videos(dataset_root: Path) -> dict[str, list[tuple[int, Path]]]:
    discovered: dict[str, list[tuple[int, Path]]] = {}
    for child in dataset_root.iterdir():
        if not child.is_dir():
            continue
        parsed = split_task_dir_name(child)
        if parsed is None:
            continue
        task_name, dataset_index = parsed
        video_path = child / VIDEO_NAME
        if video_path.exists():
            discovered.setdefault(task_name, []).append((dataset_index, video_path))
    return {task: sorted(items, key=lambda item: item[0]) for task, items in discovered.items()}


def select_by_ordinals(items: list[tuple[int, Path]], ordinals: list[int]) -> list[tuple[int, int, Path]]:
    selected = []
    for ordinal in ordinals:
        offset = ordinal - 1
        if offset < 0 or offset >= len(items):
            continue
        dataset_index, video_path = items[offset]
        selected.append((ordinal, dataset_index, video_path))
    return selected


def select_by_suffixes(items: list[tuple[int, Path]], suffixes: list[int]) -> list[tuple[int, int, Path]]:
    by_index = {dataset_index: video_path for dataset_index, video_path in items}
    return [(suffix, suffix, by_index[suffix]) for suffix in suffixes if suffix in by_index]


def build_cases(
    discovered: dict[str, list[tuple[int, Path]]],
    task_names: list[str] | None,
    table_numbers: list[int],
    robocasa_numbers: list[int],
    selection_mode: str,
) -> list[Case]:
    selected_tasks = task_names or sorted(task for task in discovered if task in TASK_ENV_IDS)
    cases: list[Case] = []
    selector = select_by_suffixes if selection_mode == "suffix" else select_by_ordinals
    for task_name in selected_tasks:
        if task_name not in TASK_ENV_IDS:
            print(f"[skip] no env mapping for dataset task: {task_name}")
            continue
        items = discovered.get(task_name, [])
        if not items:
            print(f"[skip] no videos found for dataset task: {task_name}")
            continue
        for background, numbers in (("Table", table_numbers), ("RoboCasa", robocasa_numbers)):
            for ordinal, dataset_index, video_path in selector(items, numbers):
                cases.append(
                    Case(
                        task_name=task_name,
                        env_id=TASK_ENV_IDS[task_name],
                        background=background,
                        ordinal=ordinal,
                        dataset_index=dataset_index,
                        video_path=video_path,
                    )
                )
    return cases


def apply_case_config(base_cfg: dict, case: Case, batch_dir: Path, background_type_id: list[int]) -> dict:
    cfg = deepcopy(base_cfg)
    cfg["env_id"] = case.env_id
    cfg["num_episodes"] = 1
    cfg["timestamp_outputs"] = False
    cfg["run_name"] = f"{case.task_name}_{case.background.lower()}_{case.dataset_index:04d}"
    cfg["output_dir"] = str(batch_dir)

    runtime_cfg = cfg.setdefault("runtime", {})
    runtime_cfg["background"] = case.background
    runtime_cfg["background_type_id"] = background_type_id

    generation_cfg = cfg.setdefault("generation", {})
    generation_cfg.update({"kind": "local", "path": str(case.video_path)})

    prompts = dict(TASK_PROMPTS)
    prompts.update(cfg.get("prompts", {}))
    cfg["prompts"] = prompts
    return cfg


def write_summary(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "task_name",
        "env_id",
        "background",
        "ordinal",
        "dataset_index",
        "video_path",
        "status",
        "output_path",
        "error",
    ]
    with path.open("w", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one fake-video KineBench evaluation for every dataset task and both Table/RoboCasa backgrounds."
    )
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/eval/fake_close_faucet_video.yaml")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--tasks", default=None, help="Comma-separated dataset task names, e.g. close_box,open_faucet.")
    parser.add_argument("--table-videos", default="1", help="1-based sorted positions for Table videos.")
    parser.add_argument("--robocasa-videos", default="51", help="1-based sorted positions for RoboCasa videos.")
    parser.add_argument(
        "--selection-mode",
        choices=["ordinal", "suffix"],
        default="ordinal",
        help="ordinal uses 1-based sorted positions; suffix uses exact directory suffixes.",
    )
    parser.add_argument("--background-type-id", type=parse_background_type_id, default=[-1, -1])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    if not args.dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {args.dataset_root}")

    task_names = None
    if args.tasks:
        task_names = [item.strip() for item in args.tasks.split(",") if item.strip()]

    discovered = discover_videos(args.dataset_root)
    cases = build_cases(
        discovered=discovered,
        task_names=task_names,
        table_numbers=parse_int_list(args.table_videos),
        robocasa_numbers=parse_int_list(args.robocasa_videos),
        selection_mode=args.selection_mode,
    )
    if not cases:
        raise RuntimeError("No runnable cases were selected.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"all_fake_video_eval_{timestamp}"
    batch_dir = args.output_dir / run_name
    summary_path = batch_dir / "batch_summary.csv"

    print(f"Selected {len(cases)} cases.")
    for case in cases:
        print(
            f"  {case.task_name:18s} {case.background:8s} "
            f"ordinal={case.ordinal:03d} suffix={case.dataset_index:03d} env={case.env_id}"
        )
    if args.dry_run:
        print("Dry run only; no evaluations executed.")
        return

    base_cfg = load_config(args.config)
    rows: list[dict] = []
    for case_idx, case in enumerate(cases, start=1):
        print(
            f"[{case_idx}/{len(cases)}] {case.task_name} | {case.background} | "
            f"ordinal={case.ordinal} suffix={case.dataset_index}"
        )
        row = {
            "task_name": case.task_name,
            "env_id": case.env_id,
            "background": case.background,
            "ordinal": case.ordinal,
            "dataset_index": case.dataset_index,
            "video_path": str(case.video_path),
            "status": "pending",
            "output_path": "",
            "error": "",
        }
        try:
            cfg = apply_case_config(base_cfg, case, batch_dir, args.background_type_id)
            output_path = KineBenchEvaluator(cfg).run(env_id=case.env_id, num_episodes=1)
            row["status"] = "ok"
            row["output_path"] = str(Path(output_path).resolve())
        except Exception as exc:
            row["status"] = "error"
            row["error"] = repr(exc)
            traceback.print_exc()
            if args.fail_fast:
                rows.append(row)
                write_summary(summary_path, rows)
                raise
        rows.append(row)
        write_summary(summary_path, rows)

    ok_count = sum(row["status"] == "ok" for row in rows)
    print(f"Finished {ok_count}/{len(rows)} cases. Summary: {summary_path.resolve()}")


if __name__ == "__main__":
    main()
