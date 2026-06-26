from __future__ import annotations

import base64
import os
import tempfile
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
import requests

from kinebench.generation.base import GeneratedVideo
from kinebench.video import ensure_bcthw, read_video


class DashScopeWanGenerator:
    """Optional DashScope Wan2.6 I2V/R2V generator behind the VideoGenerator API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "wan2.6-i2v",
        size: str = "960*960",
        duration: int = 5,
        shot_type: str = "single",
        watermark: bool = False,
        out_frames: int = 49,
        out_hw: tuple[int, int] = (384, 384),
        poll_interval: float = 3.0,
        timeout_sec: int = 600,
        base_url: str = "https://dashscope.aliyuncs.com",
        save_dir: str | Path | None = None,
    ):
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY", "")
        if not self.api_key:
            raise ValueError("Missing DASHSCOPE_API_KEY for DashScopeWanGenerator.")
        self.model = model
        self.size = size
        self.duration = int(duration)
        self.shot_type = shot_type
        self.watermark = watermark
        self.out_frames = int(out_frames)
        self.out_hw = out_hw
        self.poll_interval = poll_interval
        self.timeout_sec = timeout_sec
        self.base_url = base_url.rstrip("/")
        self.save_dir = Path(save_dir) if save_dir else None

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
        task_id = self._create_task(prompt, first_frame, ref_video, negative_prompt, seed)
        url = self._wait_and_get_video_url(task_id)
        saved = self._download_video(url, seed)
        frames = read_video(saved, target_frames=self.out_frames, size=self.out_hw)
        return GeneratedVideo(frames=ensure_bcthw(frames), source_path=saved, metadata={"task_id": task_id, "seed": seed})

    def _headers(self, async_task: bool = False) -> dict[str, str]:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        if async_task:
            headers["X-DashScope-Async"] = "enable"
            headers["X-DashScope-OssResourceResolve"] = "enable"
        return headers

    def _upload_local_file_get_oss_url(self, file_path: Path) -> str:
        policy_url = f"{self.base_url}/api/v1/uploads?action=getPolicy&model={self.model}"
        r = requests.get(policy_url, headers=self._headers(), timeout=60)
        r.raise_for_status()
        data = r.json()["data"]
        key = f"{data['upload_dir']}/{int(time.time() * 1000)}_{file_path.name}"
        form = {
            "OSSAccessKeyId": data["oss_access_key_id"],
            "policy": data["policy"],
            "Signature": data["signature"],
            "key": key,
            "x-oss-object-acl": data["x_oss_object_acl"],
            "x-oss-forbid-overwrite": data["x_oss_forbid_overwrite"],
            "success_action_status": "200",
        }
        with file_path.open("rb") as f:
            up = requests.post(data["upload_host"], data=form, files={"file": (file_path.name, f, "video/mp4")}, timeout=300)
        up.raise_for_status()
        return f"oss://{key}"

    def _create_task(
        self,
        prompt: str,
        first_frame: np.ndarray,
        ref_video: str | Path | None,
        negative_prompt: str | None,
        seed: int | None,
    ) -> str:
        input_payload: dict[str, object] = {"prompt": prompt}
        if self.model.startswith("wan2.6-r2v"):
            if ref_video is None:
                raise ValueError("R2V generation requires ref_video.")
            input_payload["reference_urls"] = [self._upload_local_file_get_oss_url(Path(ref_video))]
        else:
            input_payload["img_url"] = self._first_frame_uri(first_frame)
        if negative_prompt:
            input_payload["negative_prompt"] = negative_prompt
        params: dict[str, object] = {
            "size": self.size,
            "duration": self.duration,
            "shot_type": self.shot_type,
            "watermark": self.watermark,
        }
        if seed is not None:
            params["seed"] = int(seed)
        payload = {"model": self.model, "input": input_payload, "parameters": params}
        r = requests.post(
            f"{self.base_url}/api/v1/services/aigc/video-generation/video-synthesis",
            headers=self._headers(async_task=True),
            json=payload,
            timeout=120,
        )
        r.raise_for_status()
        task_id = r.json().get("output", {}).get("task_id")
        if not task_id:
            raise RuntimeError(f"DashScope task creation failed: {r.text}")
        return task_id

    def _wait_and_get_video_url(self, task_id: str) -> str:
        t0 = time.time()
        while True:
            r = requests.get(f"{self.base_url}/api/v1/tasks/{task_id}", headers={"Authorization": f"Bearer {self.api_key}"}, timeout=60)
            r.raise_for_status()
            out = r.json().get("output", {})
            status = out.get("task_status")
            if status == "SUCCEEDED":
                if not out.get("video_url"):
                    raise RuntimeError(f"Task succeeded without video_url: {out}")
                return out["video_url"]
            if status in {"FAILED", "CANCELED", "UNKNOWN"}:
                raise RuntimeError(f"DashScope task {status}: {out}")
            if time.time() - t0 > self.timeout_sec:
                raise TimeoutError(f"DashScope task timeout after {self.timeout_sec}s: {task_id}")
            time.sleep(self.poll_interval)

    def _download_video(self, video_url: str, seed: int | None) -> Path:
        out_dir = self.save_dir or Path(tempfile.mkdtemp(prefix="kinebench_dashscope_"))
        out_dir.mkdir(parents=True, exist_ok=True)
        name = f"wan26_seed{seed if seed is not None else 'none'}_{uuid.uuid4().hex[:8]}.mp4"
        out = out_dir / name
        r = requests.get(video_url, stream=True, timeout=300)
        r.raise_for_status()
        with out.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
        return out

