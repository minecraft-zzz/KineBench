from __future__ import annotations

import os
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation as R
from torchvision.utils import save_image


MANISKILL_CAMERA_TO_EEF = np.array(
    [[0.0, 0.529999, -0.847998, 0.4], [1.0, 0.0, 0.0, 0.0], [0.0, -0.847998, -0.529999, 0.45], [0.0, 0.0, 0.0, 1.0]]
)
ROBOT_T_WORLD = np.linalg.inv(np.array([[1.0, 0.0, 0.0, -0.615], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]))


def _prepare_one_workspace(i: int, base_path: Path, frames_i: torch.Tensor, cad_path: Path, link_mesh: bool) -> str:
    ws = base_path / f"workspace{i:03d}"
    rgb_dir = ws / "rgb"
    mesh_dir = ws / "mesh"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    if mesh_dir.exists() or mesh_dir.is_symlink():
        if mesh_dir.is_dir() and not mesh_dir.is_symlink():
            shutil.rmtree(mesh_dir)
        else:
            mesh_dir.unlink()
    if link_mesh:
        os.symlink(cad_path, mesh_dir, target_is_directory=True)
    else:
        mesh_dir.mkdir(parents=True, exist_ok=True)
        if cad_path.exists():
            for p in cad_path.iterdir():
                if p.is_file():
                    shutil.copy2(p, mesh_dir / p.name)
                elif p.is_dir():
                    shutil.copytree(p, mesh_dir / p.name, dirs_exist_ok=True)
    for t in range(frames_i.shape[1]):
        save_image(frames_i[:, t], str(rgb_dir / f"rgb{t:03d}.png"))
    return str(ws)


def prepare_all_workspaces(base_path: str | Path, batched_frames: np.ndarray | torch.Tensor, cad_path: str | Path, max_workers: int = 8, link_mesh: bool = False) -> list[str]:
    base = Path(base_path)
    cad = Path(cad_path)
    base.mkdir(parents=True, exist_ok=True)
    frames = torch.as_tensor(batched_frames, dtype=torch.float32)
    if frames.ndim != 5:
        raise ValueError(f"Expected frames [B,3,T,H,W] or [B,T,3,H,W], got {tuple(frames.shape)}")
    if frames.shape[1] != 3 and frames.shape[2] == 3:
        frames = frames.permute(0, 2, 1, 3, 4)
    if frames.shape[1] != 3:
        raise ValueError(f"Expected RGB channels, got {tuple(frames.shape)}")
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        return list(ex.map(lambda item: _prepare_one_workspace(item[0], base, item[1], cad, link_mesh), enumerate(frames)))


def run_yolo_masks(base_path: str | Path, model_path: str | Path, imgsz: int | None = None, conf: float = 0.15, device: str | int | None = None, scale: float = 0.5) -> str:
    from ultralytics import YOLO

    base = Path(base_path)
    items: list[tuple[Path, np.ndarray]] = []
    for ws in sorted(base.glob("workspace*")):
        img_path = ws / "rgb" / "rgb000.png"
        if img_path.exists():
            items.append((ws, np.asarray(Image.open(img_path).convert("RGB"))))
    if not items:
        raise FileNotFoundError(f"No workspace rgb000.png under {base}")
    model = YOLO(str(model_path))
    results = model([cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for _, img in items], imgsz=imgsz, conf=conf, device=device)
    success_mask = None
    failed, success = [], 0
    for (ws, _), r in zip(items, results):
        h, w = r.orig_shape
        mask = np.zeros((h, w), dtype=np.uint8)
        if r.boxes is None or len(r.boxes) == 0:
            failed.append(ws)
            continue
        k = int(r.boxes.conf.detach().cpu().numpy().argmax())
        cx_n, cy_n, w_n, h_n = r.boxes.xywhn[k].tolist()
        cv2.circle(mask, (int(cx_n * w), int(cy_n * h)), int(scale * min(w_n * w, h_n * h)), 255, thickness=-1)
        out = ws / "masks"
        out.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out / "masks000.png"), mask)
        success_mask = mask if success_mask is None else success_mask
        success += 1
    if success == 0:
        return "yolo_failed"
    for ws in failed:
        out = ws / "masks"
        out.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out / "masks000.png"), success_mask)
    return "yolo_partial_success" if failed else "yolo_success"


def load_poses_from_folder(folder: str | Path, prefix: str = "rgb", suffix: str = ".txt") -> np.ndarray:
    folder = Path(folder)
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+){re.escape(suffix)}$")
    indexed = []
    for p in folder.iterdir():
        match = pattern.match(p.name)
        if match:
            indexed.append((int(match.group(1)), p))
    if not indexed:
        raise FileNotFoundError(f"No pose txt files found in {folder}")
    mats = [np.loadtxt(path, dtype=float).reshape(4, 4) for _, path in sorted(indexed)]
    return np.stack(mats, axis=0)


def left_mult(mats: np.ndarray, transform: np.ndarray) -> np.ndarray:
    mats = np.asarray(mats, dtype=float)
    return np.asarray(transform, dtype=float).reshape(4, 4)[None] @ mats


def fp_workspaces_to_pose7(base_path: str | Path) -> np.ndarray:
    poses = []
    for ws in sorted(Path(base_path).glob("workspace*")):
        ob_in_cam = load_poses_from_folder(ws / "FP_result" / "ob_in_cam")
        eef_world = left_mult(ob_in_cam, MANISKILL_CAMERA_TO_EEF)
        eef_robot = left_mult(eef_world, ROBOT_T_WORLD)
        q = R.from_matrix(eef_robot[:, :3, :3]).as_quat(scalar_first=True)
        poses.append(np.concatenate([eef_robot[:, :3, 3], q], axis=1))
    return np.asarray(poses)

