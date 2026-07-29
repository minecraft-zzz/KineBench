from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np


def run_pyroki(
    poses_path: str | Path,
    result_path: str | Path,
    pyroki_repo: str | Path,
    conda_bin: str = "conda",
    env_name: str = "pyroki_new",
    manipulability_weight: float = 0.2,
    robot_urdf_file: str | Path | None = None,
    robot_mesh_dir: str | Path | None = None,
) -> None:
    cmd = [
        conda_bin,
        "run",
        "-n",
        env_name,
        "python",
        "examples/ik_for_grpo_05.py",
        "--batch_frames_eef_pose_path",
        str(Path(poses_path).resolve()),
        "--manipulability_weight",
        str(manipulability_weight),
        "--result_path",
        str(Path(result_path).resolve()),
    ]
    if robot_urdf_file is not None:
        cmd.extend(["--robot_urdf_file", str(Path(robot_urdf_file).resolve())])
    if robot_mesh_dir is not None:
        cmd.extend(["--robot_mesh_dir", str(Path(robot_mesh_dir).resolve())])
    subprocess.run(cmd, cwd=str(pyroki_repo), check=True)


def pose_to_ik_actions(
    pose7: np.ndarray,
    workspace: str | Path,
    pyroki_repo: str | Path,
    conda_bin: str = "conda",
    env_name: str = "pyroki_new",
    manipulability_weight: float = 0.2,
    robot_urdf_file: str | Path | None = None,
    robot_mesh_dir: str | Path | None = None,
) -> tuple[np.ndarray, list[float], np.ndarray]:
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    poses_path = workspace / "poses.npy"
    result_path = workspace / "results_per_frame.npy"
    np.save(poses_path, np.asarray(pose7))
    run_pyroki(
        poses_path,
        result_path,
        pyroki_repo,
        conda_bin=conda_bin,
        env_name=env_name,
        manipulability_weight=manipulability_weight,
        robot_urdf_file=robot_urdf_file,
        robot_mesh_dir=robot_mesh_dir,
    )
    results = np.load(result_path, allow_pickle=False)
    poses = np.asarray(pose7)
    if poses.ndim != 3:
        raise ValueError(f"Expected pose7 [B,T,7], got {poses.shape}")
    b, t, _ = poses.shape
    qpos, costs = [], []
    for i in range(b):
        block = results[t * i : t * (i + 1)]
        costs.append(float(block[:, -1].mean()))
        qpos.append(np.concatenate([block[:, :-2], -np.ones((t, 1))], axis=1))
    return np.asarray(qpos), costs, results

