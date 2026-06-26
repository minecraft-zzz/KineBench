from __future__ import annotations

from pathlib import Path

import numpy as np

from kinebench.generation.base import GeneratedVideo
from kinebench.video import read_video


class LocalVideoGenerator:
    """Generator adapter for already generated mp4/npy/npz videos."""

    def __init__(self, path: str | Path, target_frames: int = 49, size: tuple[int, int] = (384, 384)):
        self.path = Path(path)
        self.target_frames = target_frames
        self.size = size

    def generate(
        self,
        prompt: str,
        first_frame: np.ndarray,
        ref_video: str | Path | None = None,
        negative_prompt: str | None = None,
        seed: int | None = None,
    ) -> GeneratedVideo:
        path = Path(ref_video) if ref_video else self.path
        frames = read_video(path, target_frames=self.target_frames, size=self.size).astype(np.float32)
        return GeneratedVideo(frames=frames, source_path=path, metadata={"kind": "local", "prompt": prompt, "seed": seed})

