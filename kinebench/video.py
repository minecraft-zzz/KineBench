from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import imageio.v3 as iio
import numpy as np


def ensure_bcthw(
    video: np.ndarray,
    target_frames: int | None = None,
    size: tuple[int, int] | None = None,
    sampling: str = "uniform",
) -> np.ndarray:
    arr = np.asarray(video)
    if arr.ndim == 4 and arr.shape[-1] == 3:
        arr = arr[None, ...]  # B,T,H,W,C
    if arr.ndim == 4 and arr.shape[0] == 3:
        arr = arr[None, ...]  # B,C,T,H,W
    if arr.ndim != 5:
        raise ValueError(f"Expected 4D/5D video array, got {arr.shape}")
    if arr.shape[-1] == 3:
        arr = np.transpose(arr, (0, 4, 1, 2, 3))
    if arr.shape[1] != 3:
        raise ValueError(f"Expected RGB channels, got {arr.shape}")
    arr = arr.astype(np.float32)
    if arr.max(initial=0) > 1.0:
        arr = arr / 255.0
    arr = np.clip(arr, 0.0, 1.0)
    if size is not None:
        w, h = size
        arr = np.stack([resize_bcthw(v[None], (w, h))[0] for v in arr], axis=0)
    if target_frames is not None:
        arr = resample_bcthw(arr, target_frames, sampling=sampling)
    return arr


def resample_bcthw(video: np.ndarray, target_frames: int, sampling: str = "uniform") -> np.ndarray:
    b, c, t, h, w = video.shape
    if t == target_frames:
        return video
    if t < target_frames:
        pad = np.repeat(video[:, :, -1:, :, :], target_frames - t, axis=2)
        return np.concatenate([video, pad], axis=2)
    if sampling == "head":
        return video[:, :, :target_frames, :, :]
    if sampling != "uniform":
        raise ValueError(f"sampling must be 'uniform' or 'head', got {sampling!r}")
    idx = np.linspace(0, t - 1, target_frames).round().astype(int)
    return video[:, :, idx, :, :]


def resize_bcthw(video: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    w, h = size
    b, c, t, _, _ = video.shape
    out = np.empty((b, c, t, h, w), dtype=np.float32)
    for bi in range(b):
        for ti in range(t):
            frame = np.transpose(video[bi, :, ti], (1, 2, 0))
            out[bi, :, ti] = np.transpose(cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA), (2, 0, 1))
    return out


def read_video(
    path: str | Path,
    target_frames: int | None = None,
    size: tuple[int, int] | None = None,
    sampling: str = "uniform",
) -> np.ndarray:
    path = Path(path)
    if path.suffix == ".npy":
        return ensure_bcthw(np.load(path), target_frames=target_frames, size=size, sampling=sampling)
    if path.suffix == ".npz":
        data = np.load(path)
        key = "video" if "video" in data else data.files[0]
        return ensure_bcthw(data[key], target_frames=target_frames, size=size, sampling=sampling)
    frames = iio.imread(path)
    return ensure_bcthw(frames, target_frames=target_frames, size=size, sampling=sampling)


def write_video(path: str | Path, frames: Iterable[np.ndarray] | np.ndarray, fps: int = 17) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(list(frames) if not isinstance(frames, np.ndarray) else frames)
    if arr.ndim == 5:
        arr = np.transpose(arr[0], (1, 2, 3, 0))
    if arr.ndim == 4 and arr.shape[0] == 3:
        arr = np.transpose(arr, (1, 2, 3, 0))
    arr = np.clip(arr * 255.0 if arr.dtype.kind == "f" else arr, 0, 255).astype(np.uint8)
    iio.imwrite(path, arr, fps=fps)


def image_batch_to_first_frame(rgb_batch: np.ndarray) -> np.ndarray:
    arr = np.asarray(rgb_batch)
    arr = arr[None] if arr.ndim == 3 else arr
    if arr.shape[-1] != 3:
        raise ValueError(f"Expected BHWC RGB image batch, got {arr.shape}")
    return np.transpose(2.0 * arr.astype(np.float32) / 255.0 - 1.0, (0, 3, 1, 2))

