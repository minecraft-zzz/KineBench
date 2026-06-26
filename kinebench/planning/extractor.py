from __future__ import annotations

from pathlib import Path

import numpy as np

from kinebench.perception.pipeline import VideoToPosePipeline
from kinebench.planning.gripper import predict_gripper_actions
from kinebench.planning.pyroki import pose_to_ik_actions
from kinebench.planning.transforms import pose7_batch_to_eepose_actions


class TrajectoryExtractor:
    def __init__(self, config: dict):
        self.config = config
        self.perception = VideoToPosePipeline(config.get("perception", {}))

    def video_to_actions(self, frames: np.ndarray, workspace: str | Path, control_mode: str = "pd_ee_pose", moge_model=None) -> tuple[np.ndarray, dict]:
        workspace = Path(workspace)
        if self.config.get("mock_actions"):
            actions = np.asarray(self.config["mock_actions"], dtype=np.float32)
            if actions.ndim == 2:
                actions = actions[None, ...]
            return actions, {"mode": "mock"}
        pose_result = self.perception.run(frames, workspace / "fp_workspaces", moge_model=moge_model)
        pose7 = pose_result.pose7
        b, t, _ = pose7.shape
        gripper = predict_gripper_actions(workspace / "fp_workspaces", self.config.get("gripper", {}), length=t, batch_size=b)
        if control_mode == "pd_joint_pos":
            qpos, costs, costs_per_frame = pose_to_ik_actions(
                pose7,
                workspace / "pyroki",
                self.config["pyroki"]["repo"],
                conda_bin=self.config["pyroki"].get("conda_bin", "conda"),
                env_name=self.config["pyroki"].get("env_name", "pyroki_new"),
                manipulability_weight=float(self.config["pyroki"].get("manipulability_weight", 0.2)),
            )
            actions = np.concatenate([qpos[:, :, :-1], gripper], axis=-1)
            meta = {"mode": "pd_joint_pos", "costs_per_sample": costs, "costs_per_frame": costs_per_frame, "yolo_status": pose_result.yolo_status}
        else:
            actions = pose7_batch_to_eepose_actions(pose7, gripper.reshape(b, t), z_offset=float(self.config.get("eef_z_offset", 0.12)))
            meta = {"mode": "pd_ee_pose", "pose7": pose7, "yolo_status": pose_result.yolo_status}
        return actions, meta

