from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R


def quat_wxyz_to_rotmat(q: np.ndarray) -> np.ndarray:
    return R.from_quat(np.asarray(q), scalar_first=True).as_matrix()


def poses7_to_T(
    pose7: np.ndarray,
    layout: tuple[str, ...] = ("px", "py", "pz", "qw", "qx", "qy", "qz"),
    normalize_quat: bool = True,
) -> np.ndarray:
    pose7 = np.asarray(pose7, dtype=np.float32).squeeze()
    if pose7.ndim == 1:
        pose7 = pose7[None, :]
    if pose7.ndim != 2 or pose7.shape[1] != 7:
        raise ValueError(f"Expected pose7 shape [N,7], got {pose7.shape}")
    name2idx = {name: idx for idx, name in enumerate(layout)}
    p = pose7[:, [name2idx[k] for k in ("px", "py", "pz")]]
    q = pose7[:, [name2idx[k] for k in ("qw", "qx", "qy", "qz")]]
    if normalize_quat:
        q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
    out = np.zeros((pose7.shape[0], 4, 4), dtype=np.float32)
    out[:, 3, 3] = 1.0
    out[:, :3, :3] = quat_wxyz_to_rotmat(q)
    out[:, :3, 3] = p
    return out


def poses_wxyz_to_xyz_euler(poses: np.ndarray) -> np.ndarray:
    poses = np.asarray(poses, dtype=np.float32)
    if poses.ndim == 1:
        poses = poses[None, :]
    if poses.ndim != 2 or poses.shape[1] != 7:
        raise ValueError(f"Expected poses shape [T,7], got {poses.shape}")
    p = poses[:, :3]
    q = poses[:, 3:7]
    q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
    euler = R.from_quat(q, scalar_first=True).as_euler("XYZ", degrees=False)
    return np.concatenate([p, euler], axis=1)


def pose7_batch_to_eepose_actions(
    poses: np.ndarray,
    gripper_actions: np.ndarray,
    z_offset: float = 0.12,
    offset_frame: str = "local",
) -> np.ndarray:
    poses = np.asarray(poses)
    if poses.ndim == 3:
        poses = poses[0]
    mats = poses7_to_T(poses)
    if offset_frame == "local":
        offset = np.eye(4, dtype=mats.dtype)
        offset[2, 3] = z_offset
        mats = mats @ offset
    elif offset_frame == "world":
        mats[:, 2, 3] += z_offset
    else:
        raise ValueError(f"offset_frame must be 'local' or 'world', got {offset_frame!r}")
    pose7 = np.concatenate([mats[:, :3, 3], R.from_matrix(mats[:, :3, :3]).as_quat(scalar_first=True)], axis=1)
    euler = poses_wxyz_to_xyz_euler(pose7)
    grip = np.asarray(gripper_actions).reshape(euler.shape[0], -1)
    return np.concatenate([euler, grip[:, :1]], axis=1)[None, ...]

