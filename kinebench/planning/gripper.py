from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def default_gripper_actions(length: int, batch_size: int = 1, value: float = -1.0) -> np.ndarray:
    return np.full((batch_size, length, 1), value, dtype=np.float32)


def predict_gripper_actions(base_path: str | Path, config: dict, length: int, batch_size: int = 1) -> np.ndarray:
    if not config or not config.get("enabled", False):
        return default_gripper_actions(length, batch_size, float(config.get("default", -1.0) if config else -1.0))
    utils_path = config.get("videox_fun_utils")
    if utils_path and str(utils_path) not in sys.path:
        sys.path.insert(0, str(utils_path))
    from gripper_classifier import infer_base_path

    res = infer_base_path(
        base_path=str(base_path),
        yolo_pose_weights=config["yolo_pose_weights"],
        mlp_ckpt=config["mlp_ckpt"],
        dino_repo=config["dino_repo"],
        device=config.get("device", "cuda:0"),
        yolo_imgsz=int(config.get("yolo_imgsz", 512)),
        yolo_batch=int(config.get("yolo_batch", 256)),
        yolo_conf=float(config.get("yolo_conf", 0.5)),
        low_conf_warn=float(config.get("low_conf_warn", 0.5)),
        use_depth=bool(config.get("use_depth", False)),
    )
    actions = []
    for workspace_name, item in sorted(res.items()):
        pred_raw = np.asarray(item["pred"], dtype=np.float32)
        prob_raw = np.asarray(item.get("prob", []), dtype=np.float32)
        workspace_dir = Path(base_path) / workspace_name
        np.save(workspace_dir / "gripper_action.npy", pred_raw)
        if prob_raw.size:
            np.save(workspace_dir / "gripper_action_prob.npy", prob_raw)

        pred = pred_raw[:length]
        if pred.shape[0] < length:
            pad_value = pred[-1] if pred.shape[0] else float(config.get("default", -1.0))
            pred = np.pad(pred, (0, length - pred.shape[0]), mode="constant", constant_values=pad_value)

        if bool(config.get("map_labels_to_actions", True)):
            close_label = float(config.get("close_label", 1.0))
            open_action = float(config.get("open_action", 1.0))
            close_action = float(config.get("close_action", -1.0))
            pred = np.where(pred == close_label, close_action, open_action).astype(np.float32)
        actions.append(pred[:, None])
    out = np.asarray(actions, dtype=np.float32)
    if out.shape[0] != batch_size:
        out = out[:batch_size]
    return out

