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

    def _resolve_fp_workspace(self, episode_workspace: Path) -> Path:
        perception_cfg = self.config.get("perception", {})
        configured = perception_cfg.get("fp_workspace")
        if not configured:
            return episode_workspace / "fp_workspaces"
        configured = str(configured).format(episode_dir=str(episode_workspace), episode_name=episode_workspace.name)
        path = Path(configured)
        if not path.is_absolute():
            path = episode_workspace / path
        return path

    def video_to_actions(
        self,
        frames: np.ndarray,
        workspace: str | Path,
        control_mode: str = "pd_ee_pose",
        moge_model=None,
        pose_transforms: dict | None = None,
        env_id: str | None = None,
    ) -> tuple[np.ndarray, dict]:
        workspace = Path(workspace)
        if self.config.get("mock_actions"):
            actions = np.asarray(self.config["mock_actions"], dtype=np.float32)
            if actions.ndim == 2:
                actions = actions[None, ...]
            return actions, {"mode": "mock"}
        fp_workspace = self._resolve_fp_workspace(workspace)
        pose_result = self.perception.run(frames, fp_workspace, moge_model=moge_model, pose_transforms=pose_transforms)
        pose7 = pose_result.pose7
        b, t, _ = pose7.shape
        gripper = predict_gripper_actions(fp_workspace, self.config.get("gripper", {}), length=t, batch_size=b)
        pyroki_meta = self._maybe_compute_manipulation_cost(pose7, workspace)
        planning_cfg = self.config.get("planning", {})
        task_cfg = planning_cfg.get("task_overrides", {}).get(env_id or "", {})
        eef_z_offset = float(task_cfg.get("eef_z_offset", self.config.get("eef_z_offset", planning_cfg.get("eef_z_offset", 0.12))))
        eef_offset_frame = str(task_cfg.get("eef_offset_frame", planning_cfg.get("eef_offset_frame", "local")))

        if control_mode == "pd_joint_pos" and self.config.get("pyroki", {}).get("use_ik_actions", False):
            qpos, costs, costs_per_frame = pose_to_ik_actions(
                pose7,
                workspace / "pyroki",
                self.config["pyroki"]["repo"],
                conda_bin=self.config["pyroki"].get("conda_bin", "conda"),
                env_name=self.config["pyroki"].get("env_name", "pyroki_new"),
                manipulability_weight=float(self.config["pyroki"].get("manipulability_weight", 0.2)),
                robot_urdf_file=self.config["pyroki"].get("robot_urdf_file"),
                robot_mesh_dir=self.config["pyroki"].get("robot_mesh_dir"),
            )
            actions = np.concatenate([qpos[:, :, :-1], gripper], axis=-1)
            meta = {"mode": "pd_joint_pos_from_pyroki_ik", "costs_per_sample": costs, "costs_per_frame": costs_per_frame, "yolo_status": pose_result.yolo_status}
        else:
            actions = pose7_batch_to_eepose_actions(pose7, gripper.reshape(b, t), z_offset=eef_z_offset, offset_frame=eef_offset_frame)
            meta = {
                "mode": "pd_ee_pose_from_foundationpose",
                "pose7": pose7,
                "yolo_status": pose_result.yolo_status,
                "eef_z_offset": eef_z_offset,
                "eef_offset_frame": eef_offset_frame,
            }
            meta.update(pyroki_meta)
        return actions, meta

    def _maybe_compute_manipulation_cost(self, pose7: np.ndarray, workspace: Path) -> dict:
        pyroki_cfg = self.config.get("pyroki", {})
        if not pyroki_cfg.get("compute_cost", False):
            return {}
        _, costs, costs_per_frame = pose_to_ik_actions(
            pose7,
            workspace / "pyroki_cost",
            pyroki_cfg["repo"],
            conda_bin=pyroki_cfg.get("conda_bin", "conda"),
            env_name=pyroki_cfg.get("env_name", "pyroki_new"),
            manipulability_weight=float(pyroki_cfg.get("manipulability_weight", 0.2)),
            robot_urdf_file=pyroki_cfg.get("robot_urdf_file"),
            robot_mesh_dir=pyroki_cfg.get("robot_mesh_dir"),
        )
        return {"manipulation_costs_per_sample": costs, "manipulation_costs_per_frame": costs_per_frame}

