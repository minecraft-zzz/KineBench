from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from kinebench.perception.foundationpose import run_foundationpose
from kinebench.perception.moge import moge_infer_for_basepath
from kinebench.perception.workspaces import fp_workspaces_to_pose7, prepare_all_workspaces, run_yolo_masks


@dataclass
class PosePipelineResult:
    pose7: np.ndarray
    yolo_status: str


class VideoToPosePipeline:
    def __init__(self, config: dict):
        self.config = config

    def run(self, frames: np.ndarray, workspace: str | Path, moge_model=None, pose_transforms: dict | None = None) -> PosePipelineResult:
        cfg = self.config
        prepare_all_workspaces(workspace, frames, cfg["cad_path"], max_workers=int(cfg.get("workspace_workers", 8)), link_mesh=bool(cfg.get("link_mesh", False)))
        yolo_status = run_yolo_masks(workspace, cfg["yolo_weights"], imgsz=cfg.get("yolo_imgsz"), conf=float(cfg.get("yolo_conf", 0.15)), device=cfg.get("device"))
        if yolo_status == "yolo_failed":
            raise RuntimeError("Failed to find gripper in videos")
        if moge_model is None:
            from moge.model.v2 import MoGeModel

            moge_model = MoGeModel.from_pretrained(cfg["moge_checkpoint"])
        moge_infer_for_basepath(moge_model, workspace, fov_x=float(cfg.get("fov_x", 60)), device=cfg.get("device"))
        run_foundationpose(workspace, cfg["foundationpose_repo"], conda_bin=cfg.get("conda_bin", "conda"), env_name=cfg.get("foundationpose_env", "foundationpose"), max_workers=int(cfg.get("fp_workers", 8)))
        pose_transforms = pose_transforms or {}
        return PosePipelineResult(
            pose7=fp_workspaces_to_pose7(
                workspace,
                world_t_camera=pose_transforms.get("world_T_camera"),
                robot_t_world=pose_transforms.get("robot_T_world"),
                scene_transform=pose_transforms.get("scene_transform"),
            ),
            yolo_status=yolo_status,
        )

