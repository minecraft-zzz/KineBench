from __future__ import annotations

import base64
import uuid
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
import requests

from kinebench.generation.base import GeneratedVideo
from kinebench.video import ensure_bcthw, read_video


def _is_loopback_url(url: str) -> bool:
    host = urlparse(url).hostname
    return host in {"127.0.0.1", "localhost", "::1"}


class HttpVideoGenerator:
    """Generator adapter for a local/remote HTTP video generation service."""

    def __init__(
        self,
        endpoint: str,
        target_frames: int = 49,
        size: tuple[int, int] = (384, 384),
        timeout_sec: int = 1800,
        save_dir: str | Path | None = None,
        headers: dict[str, str] | None = None,
        extra_payload: dict | None = None,
        sampling: str = "uniform",
    ):
        self.endpoint = endpoint
        self.target_frames = int(target_frames)
        self.size = size
        self.timeout_sec = int(timeout_sec)
        self.save_dir = Path(save_dir) if save_dir else None
        self.headers = dict(headers or {})
        self.extra_payload = dict(extra_payload or {})
        self.sampling = sampling
        self.session = requests.Session()
        if _is_loopback_url(endpoint):
            self.session.trust_env = False

    @staticmethod
    def _first_frame_uri(first_frame: np.ndarray) -> str:
        arr = np.asarray(first_frame)
        if arr.ndim == 4:
            arr = arr[0]
        if arr.shape[0] == 3:
            arr = np.transpose(arr, (1, 2, 0))
        if arr.dtype != np.uint8:
            if arr.min(initial=0) < 0:
                arr = (arr + 1.0) * 0.5
            arr = np.clip(arr * 255.0 if arr.max(initial=0) <= 1.0 else arr, 0, 255).astype(np.uint8)
        ok, buf = cv2.imencode(".png", cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
        if not ok:
            raise RuntimeError("Failed to encode first frame as PNG.")
        return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("utf8")

    def generate(
        self,
        prompt: str,
        first_frame: np.ndarray,
        ref_video: str | Path | None = None,
        negative_prompt: str | None = None,
        seed: int | None = None,
    ) -> GeneratedVideo:
        payload = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "seed": seed,
            "first_frame": self._first_frame_uri(first_frame),
            "target_frames": self.target_frames,
            "num_frames": self.target_frames,
            "size": list(self.size),
            "output_size": list(self.size),
        }
        if ref_video is not None:
            payload["ref_video"] = str(ref_video)
        payload.update(self.extra_payload)

        headers = {"Content-Type": "application/json", **self.headers}
        response = self.session.post(self.endpoint, json=payload, headers=headers, timeout=self.timeout_sec)
        response.raise_for_status()
        data = response.json()
        video_path = self._resolve_response_video(data, seed)
        frames = read_video(video_path, target_frames=self.target_frames, size=self.size, sampling=self.sampling).astype(np.float32)
        return GeneratedVideo(frames=ensure_bcthw(frames), source_path=video_path, metadata={"kind": "http", "endpoint": self.endpoint, "seed": seed, "response": data})

    def _resolve_response_video(self, data: dict, seed: int | None) -> Path:
        for key in ("video_path", "path", "output_path"):
            value = data.get(key)
            if value:
                path = Path(value)
                if not path.exists():
                    raise FileNotFoundError(f"HTTP generator returned {key}, but file does not exist: {path}")
                return path

        for key in ("video_url", "url"):
            value = data.get(key)
            if value:
                return self._download_video(str(value), seed)

        for key in ("video_base64", "video"):
            value = data.get(key)
            if value:
                return self._write_base64_video(str(value), seed)

        keys = ", ".join(sorted(data.keys()))
        raise RuntimeError(f"HTTP generator response must include video_path, video_url, or video_base64. Got keys: {keys}")

    def _output_path(self, seed: int | None) -> Path:
        out_dir = self.save_dir or Path.cwd() / "tmp" / "http_video_generator"
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir / f"http_seed{seed if seed is not None else 'none'}_{uuid.uuid4().hex[:8]}.mp4"

    def _download_video(self, url: str, seed: int | None) -> Path:
        out = self._output_path(seed)
        response = self.session.get(url, stream=True, timeout=self.timeout_sec)
        response.raise_for_status()
        with out.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
        return out

    def _write_base64_video(self, value: str, seed: int | None) -> Path:
        if value.startswith("data:"):
            value = value.split(",", 1)[1]
        out = self._output_path(seed)
        out.write_bytes(base64.b64decode(value))
        return out
