from __future__ import annotations

import csv
import shutil
from pathlib import Path

import cv2
import numpy as np

from kinebench.envs.runtime import ManiSkillRuntime
from kinebench.generation.dashscope import DashScopeWanGenerator
from kinebench.generation.local import LocalVideoGenerator
from kinebench.io import write_json
from kinebench.planning.extractor import TrajectoryExtractor
from kinebench.tasks import prompt_for_task
from kinebench.video import image_batch_to_first_frame, write_video


NEGATIVE_PROMPT = (
    "camera movement, camera motion, pan, tilt, zoom, dolly, rotation, perspective change, viewpoint change, "
    "reframing, cropping, dynamic camera, handheld, deformation, shape change, geometry change, morphing, warping"
)


class KineBenchEvaluator:
    def __init__(self, config: dict):
        self.config = config
        root = Path(config.get("repo_root", Path.cwd()))
        third_party = config.get("third_party_root", root / "third_party")
        runtime_cfg = config.get("runtime", {})
        self.runtime = ManiSkillRuntime(third_party_root=third_party, control_mode=runtime_cfg.get("control_mode", "pd_ee_pose"), sim_backend=runtime_cfg.get("sim_backend", "cpu"))
        self.extractor = TrajectoryExtractor(config)

    def _make_generator(self, episode_dir: Path):
        cfg = self.config.get("generation", {})
        kind = cfg.get("kind", "local")
        if kind == "local":
            return LocalVideoGenerator(cfg["path"], target_frames=int(cfg.get("target_frames", 49)), size=tuple(cfg.get("size", [384, 384])))
        if kind == "dashscope":
            return DashScopeWanGenerator(
                model=cfg.get("model", "wan2.6-i2v"),
                size=cfg.get("size_name", "960*960"),
                duration=int(cfg.get("duration", 5)),
                out_frames=int(cfg.get("target_frames", 49)),
                out_hw=tuple(cfg.get("size", [384, 384])),
                save_dir=episode_dir,
            )
        raise ValueError(f"Unknown generator kind: {kind}")

    def run(self, env_id: str | None = None, num_episodes: int | None = None) -> Path:
        env_id = env_id or self.config.get("env_id", "StackCube-v1")
        num_episodes = int(num_episodes or self.config.get("num_episodes", 1))
        run_name = self.config.get("run_name", "debug")
        out_root = Path(self.config.get("output_dir", "outputs")) / run_name / env_id
        out_root.mkdir(parents=True, exist_ok=True)
        rows = []
        for episode in range(num_episodes):
            rows.append(self._run_episode(env_id, episode, out_root))
        self._write_summary(out_root / "summary.csv", rows)
        return out_root

    def _run_episode(self, env_id: str, episode: int, out_root: Path) -> dict:
        episode_dir = out_root / f"{episode:04d}"
        if episode_dir.exists() and bool(self.config.get("overwrite_episode", True)):
            shutil.rmtree(episode_dir)
        episode_dir.mkdir(parents=True, exist_ok=True)
        env = None
        info = {}
        try:
            runtime_cfg = self.config.get("runtime", {})
            bg_split = float(runtime_cfg.get("bg_separate", 1.0))
            background = "Table" if episode <= int(bg_split * max(1, int(self.config.get("num_episodes", 1)))) else "RoboCasa"
            env = self.runtime.make_env(env_id, background=background)
            rgb = self.runtime.reset_rgb(env)
            first_frame = image_batch_to_first_frame(rgb)
            prompt = prompt_for_task(env_id, self.config.get("prompts"), add_visual_lock=bool(self.config.get("add_visual_lock_prompt", True)))
            generated = self._make_generator(episode_dir).generate(
                prompt=prompt,
                first_frame=first_frame,
                ref_video=self.config.get("generation", {}).get("ref_video"),
                negative_prompt=self.config.get("generation", {}).get("negative_prompt", NEGATIVE_PROMPT),
                seed=int(self.config.get("seed", 43)) + episode,
            )
            write_video(episode_dir / "generated.mp4", generated.frames, fps=int(self.config.get("fps", 17)))
            actions, action_meta = self.extractor.video_to_actions(generated.frames, episode_dir, control_mode=runtime_cfg.get("control_mode", "pd_ee_pose"))
            action_length = int(runtime_cfg.get("action_length", actions.shape[1]))
            rollout = self.runtime.rollout(env, actions[:, :action_length], repeats=int(runtime_cfg.get("action_repeat", 4)))
            self._write_side_by_side(episode_dir / "rollout.mp4", rollout.rgb, generated.frames, fps=int(self.config.get("fps", 17)))
            info = {
                "env_id": env_id,
                "episode": episode,
                "background": background,
                "done": rollout.done,
                "env_info": rollout.info,
                "action_meta": action_meta,
                "generated_source": str(generated.source_path) if generated.source_path else None,
            }
            write_json(episode_dir / "info.json", info)
            return {"episode": episode, "done": rollout.done, "success": self._success_from_info(rollout.info), "error": ""}
        except Exception as exc:
            info = {"env_id": env_id, "episode": episode, "error": f"{type(exc).__name__}: {exc}"}
            write_json(episode_dir / "info.json", info)
            return {"episode": episode, "done": False, "success": False, "error": info["error"]}
        finally:
            if env is not None:
                env.close()

    @staticmethod
    def _success_from_info(info) -> bool:
        if isinstance(info, dict):
            for key in ("success", "is_success"):
                if key in info:
                    return bool(np.asarray(info[key]).any())
            if "final_info" in info and isinstance(info["final_info"], dict):
                return KineBenchEvaluator._success_from_info(info["final_info"])
        return False

    @staticmethod
    def _write_summary(path: Path, rows: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf8") as f:
            writer = csv.DictWriter(f, fieldnames=["episode", "done", "success", "error"])
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _write_side_by_side(path: Path, rollout_rgb: np.ndarray, generated: np.ndarray, fps: int) -> None:
        frames = []
        gen = np.transpose(generated[0], (1, 2, 3, 0))
        n = min(len(rollout_rgb), len(gen))
        for t in range(n):
            env_frame = np.asarray(rollout_rgb[t])
            if env_frame.ndim == 4:
                env_frame = env_frame[0]
            env_frame = np.clip(env_frame, 0, 255).astype(np.uint8)
            sample = np.clip(gen[t] * 255.0, 0, 255).astype(np.uint8)
            env_frame = cv2.resize(env_frame, (sample.shape[1], sample.shape[0]), interpolation=cv2.INTER_AREA)
            frames.append(np.concatenate([env_frame, sample], axis=1))
        if frames:
            write_video(path, np.asarray(frames), fps=fps)

