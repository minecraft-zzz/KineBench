from __future__ import annotations

import base64
import os
import time
import uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import requests

from kinebench.generation.base import GeneratedVideo
from kinebench.video import ensure_bcthw, read_video


class MiniMaxHailuoGenerator:
    """MiniMax Hailuo image-to-video generator adapter."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "MiniMax-Hailuo-02",
        duration: int = 6,
        resolution: str = "512P",
        target_frames: int = 49,
        size: tuple[int, int] = (384, 384),
        save_dir: str | Path | None = None,
        timeout_sec: int = 1800,
        poll_interval_sec: int = 10,
        prompt_optimizer: bool | None = None,
        fast_pretreatment: bool | None = None,
        aigc_watermark: bool | None = False,
        base_url: str = "https://api.minimaxi.com",
        sampling: str = "uniform",
        extra_payload: dict[str, Any] | None = None,
    ):
        self.api_key = api_key or os.environ.get("MINIMAX_API_KEY")
        if not self.api_key:
            raise ValueError("MiniMax API key is required. Set MINIMAX_API_KEY or generation.api_key.")
        self.model = model
        self.duration = int(duration)
        self.resolution = resolution
        self.target_frames = int(target_frames)
        self.size = size
        self.save_dir = Path(save_dir) if save_dir else Path.cwd() / "tmp" / "minimax_video_generator"
        self.timeout_sec = int(timeout_sec)
        self.poll_interval_sec = int(poll_interval_sec)
        self.prompt_optimizer = prompt_optimizer
        self.fast_pretreatment = fast_pretreatment
        self.aigc_watermark = aigc_watermark
        self.base_url = base_url.rstrip("/")
        self.sampling = sampling
        self.extra_payload = dict(extra_payload or {})
        self.session = requests.Session()
        self.headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

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
        task_id = self._create_task(prompt, first_frame)
        result = self._wait_for_success(task_id)
        file_id = result["file_id"]
        out = self._download_file(file_id, seed)
        frames = read_video(out, target_frames=self.target_frames, size=self.size, sampling=self.sampling).astype(np.float32)
        return GeneratedVideo(
            frames=ensure_bcthw(frames),
            source_path=out,
            metadata={
                "kind": "minimax",
                "model": self.model,
                "duration": self.duration,
                "resolution": self.resolution,
                "task_id": task_id,
                "file_id": file_id,
                "response": result,
            },
        )

    def _create_task(self, prompt: str, first_frame: np.ndarray) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "first_frame_image": self._first_frame_uri(first_frame),
            "duration": self.duration,
            "resolution": self.resolution,
        }
        if self.prompt_optimizer is not None:
            payload["prompt_optimizer"] = bool(self.prompt_optimizer)
        if self.fast_pretreatment is not None:
            payload["fast_pretreatment"] = bool(self.fast_pretreatment)
        if self.aigc_watermark is not None:
            payload["aigc_watermark"] = bool(self.aigc_watermark)
        payload.update(self.extra_payload)

        response = self.session.post(f"{self.base_url}/v1/video_generation", headers=self.headers, json=payload, timeout=self.timeout_sec)
        response.raise_for_status()
        data = response.json()
        self._raise_for_base_resp(data, "MiniMax video_generation")
        task_id = data.get("task_id")
        if not task_id:
            raise RuntimeError(f"MiniMax video_generation response missing task_id: {data}")
        return str(task_id)

    def _wait_for_success(self, task_id: str) -> dict[str, Any]:
        deadline = time.monotonic() + self.timeout_sec
        last: dict[str, Any] | None = None
        while time.monotonic() < deadline:
            time.sleep(self.poll_interval_sec)
            response = self.session.get(
                f"{self.base_url}/v1/query/video_generation",
                headers=self.headers,
                params={"task_id": task_id},
                timeout=min(60, self.timeout_sec),
            )
            response.raise_for_status()
            data = response.json()
            self._raise_for_base_resp(data, "MiniMax query/video_generation")
            last = data
            status = data.get("status")
            if status == "Success":
                if not data.get("file_id"):
                    raise RuntimeError(f"MiniMax task succeeded without file_id: {data}")
                return data
            if status == "Fail":
                raise RuntimeError(f"MiniMax task failed: {data}")
        raise TimeoutError(f"MiniMax task timed out after {self.timeout_sec}s: task_id={task_id}, last={last}")

    def _download_file(self, file_id: str, seed: int | None) -> Path:
        response = self.session.get(
            f"{self.base_url}/v1/files/retrieve",
            headers=self.headers,
            params={"file_id": file_id},
            timeout=min(60, self.timeout_sec),
        )
        response.raise_for_status()
        data = response.json()
        self._raise_for_base_resp(data, "MiniMax files/retrieve")
        download_url = data.get("file", {}).get("download_url")
        if not download_url:
            raise RuntimeError(f"MiniMax files/retrieve response missing download_url: {data}")

        self.save_dir.mkdir(parents=True, exist_ok=True)
        out = self.save_dir / f"minimax_seed{seed if seed is not None else 'none'}_{uuid.uuid4().hex[:8]}.mp4"
        video_response = self.session.get(download_url, stream=True, timeout=self.timeout_sec)
        video_response.raise_for_status()
        with out.open("wb") as f:
            for chunk in video_response.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
        return out

    @staticmethod
    def _raise_for_base_resp(data: dict[str, Any], context: str) -> None:
        base_resp = data.get("base_resp") or {}
        status_code = base_resp.get("status_code")
        if status_code not in (None, 0):
            raise RuntimeError(f"{context} returned base_resp error: {base_resp}")
