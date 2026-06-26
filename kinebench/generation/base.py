from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


@dataclass
class GeneratedVideo:
    frames: np.ndarray  # [B,3,T,H,W], float32, [0,1]
    source_path: Path | None = None
    metadata: dict | None = None


class VideoGenerator(Protocol):
    def generate(
        self,
        prompt: str,
        first_frame: np.ndarray,
        ref_video: str | Path | None = None,
        negative_prompt: str | None = None,
        seed: int | None = None,
    ) -> GeneratedVideo:
        ...

