from __future__ import annotations

import csv
import shutil
import traceback
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from kinebench.envs.runtime import ManiSkillRuntime
from kinebench.generation.dashscope import DashScopeWanGenerator
from kinebench.generation.http import HttpVideoGenerator
from kinebench.generation.local import LocalVideoGenerator
from kinebench.generation.minimax import MiniMaxHailuoGenerator
from kinebench.io import write_json
from kinebench.planning.extractor import TrajectoryExtractor
from kinebench.tasks import prompt_for_env
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
        self.run_dir_name = self._build_run_dir_name()

    def _build_run_dir_name(self) -> str:
        run_name = str(self.config.get("run_name", "debug"))
        if not bool(self.config.get("timestamp_outputs", True)):
            return run_name
        timestamp = self.config.get("run_timestamp") or datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{run_name}_{timestamp}"

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
        if kind == "http":
            return HttpVideoGenerator(
                endpoint=cfg["endpoint"],
                target_frames=int(cfg.get("target_frames", 49)),
                size=tuple(cfg.get("size", [384, 384])),
                timeout_sec=int(cfg.get("timeout_sec", 1800)),
                save_dir=episode_dir,
                headers=cfg.get("headers"),
                extra_payload=cfg.get("extra_payload"),
                sampling=cfg.get("sampling", "uniform"),
            )
        if kind == "minimax":
            return MiniMaxHailuoGenerator(
                api_key=cfg.get("api_key"),
                model=cfg.get("model", "MiniMax-Hailuo-02"),
                duration=int(cfg.get("duration", 6)),
                resolution=cfg.get("resolution", "512P"),
                target_frames=int(cfg.get("target_frames", 49)),
                size=tuple(cfg.get("size", [384, 384])),
                timeout_sec=int(cfg.get("timeout_sec", 1800)),
                poll_interval_sec=int(cfg.get("poll_interval_sec", 10)),
                prompt_optimizer=cfg.get("prompt_optimizer"),
                fast_pretreatment=cfg.get("fast_pretreatment"),
                aigc_watermark=cfg.get("aigc_watermark", False),
                save_dir=episode_dir,
                base_url=cfg.get("base_url", "https://api.minimaxi.com"),
                sampling=cfg.get("sampling", "uniform"),
                extra_payload=cfg.get("extra_payload"),
            )
        raise ValueError(f"Unknown generator kind: {kind}")

    def _select_background(self, episode: int, num_episodes: int, runtime_cfg: dict) -> str:
        configured = runtime_cfg.get("background")
        if configured is not None:
            normalized = str(configured).strip().lower()
            if normalized in {"table"}:
                return "Table"
            if normalized in {"robocasa", "robo_casa"}:
                return "RoboCasa"
            if normalized not in {"mixed", "auto", "bg_separate"}:
                raise ValueError("runtime.background must be one of: Table, RoboCasa, mixed")
        bg_split = float(runtime_cfg.get("bg_separate", 1.0))
        table_count = int(bg_split * max(1, num_episodes))
        return "Table" if episode < table_count else "RoboCasa"

    def run(self, env_id: str | None = None, num_episodes: int | None = None) -> Path:
        env_id = env_id or self.config.get("env_id", "StackCube-v1")
        num_episodes = int(num_episodes or self.config.get("num_episodes", 1))
        out_root = Path(self.config.get("output_dir", "outputs")) / self.run_dir_name / env_id
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
            num_episodes = int(self.config.get("num_episodes", 1))
            background = self._select_background(episode, num_episodes, runtime_cfg)
            background_type_id = runtime_cfg.get("background_type_id")
            if background_type_id is None:
                background_type_id = [-1, -1]
            obj_type_id = runtime_cfg.get("obj_type_id")
            if obj_type_id is None and env_id in {"PickFruits-v1", "StoreFruitsBox-v1"}:
                obj_type_id = [-3]
            env = self.runtime.make_env(env_id, background=background, background_type_id=background_type_id, obj_type_id=obj_type_id)
            rgb = self.runtime.reset_rgb(env)
            pose_transforms = self.runtime.pose_transforms(env)
            first_frame = image_batch_to_first_frame(rgb)
            prompt, task_context = prompt_for_env(env_id, env, self.config.get("prompts"), add_visual_lock=bool(self.config.get("add_visual_lock_prompt", True)))
            max_rollouts = int(runtime_cfg.get("max_rollouts", self.config.get("max_videos_per_episode", 1)))
            if max_rollouts < 1:
                raise ValueError(f"max_rollouts must be >= 1, got {max_rollouts}")
            fps = int(self.config.get("fps", 17))
            output_cfg = self.config.get("output", {})
            save_attempts = bool(output_cfg.get("save_attempts", self.config.get("save_attempts", True)))
            save_generated = bool(output_cfg.get("save_generated", self.config.get("save_generated", True)))
            save_h5 = bool(output_cfg.get("save_h5", self.config.get("save_h5", False)))
            combined_generated = []
            combined_rollout = []
            attempt_records = []
            rollouts = []
            success = False
            rollout = None
            for rollout_idx in range(max_rollouts):
                attempt_dir = episode_dir / f"attempt_{rollout_idx:03d}"
                attempt_dir.mkdir(parents=True, exist_ok=True)
                generated = self._make_generator(attempt_dir).generate(
                    prompt=prompt,
                    first_frame=first_frame,
                    ref_video=self.config.get("generation", {}).get("ref_video"),
                    negative_prompt=self.config.get("generation", {}).get("negative_prompt", NEGATIVE_PROMPT),
                    seed=int(self.config.get("seed", 43)) + episode * max_rollouts + rollout_idx,
                )
                if save_attempts and save_generated:
                    write_video(attempt_dir / "generated.mp4", generated.frames, fps=fps)
                actions, action_meta = self.extractor.video_to_actions(
                    generated.frames,
                    attempt_dir,
                    control_mode=runtime_cfg.get("control_mode", "pd_ee_pose"),
                    pose_transforms=pose_transforms,
                    env_id=env_id,
                )
                action_length = int(runtime_cfg.get("action_length", actions.shape[1]))
                rollout = self.runtime.rollout(env, actions[:, :action_length], repeats=int(runtime_cfg.get("action_repeat", 4)))
                side_by_side = self._side_by_side_frames(rollout.rgb, generated.frames)
                if side_by_side:
                    if save_attempts:
                        write_video(attempt_dir / "rollout.mp4", np.asarray(side_by_side), fps=fps)
                    combined_rollout.extend(side_by_side)
                combined_generated.append(generated.frames)
                attempt_records.append(
                    {
                        "rollout_id": rollout_idx + 1,
                        "attempt_dir": attempt_dir.name,
                        "actions": actions[:, :action_length],
                        "generated_frames": generated.frames,
                        "rollout_rgb": rollout.rgb,
                        "eef_states": rollout.eef_states,
                        "side_by_side": np.asarray(side_by_side) if side_by_side else np.empty((0,), dtype=np.uint8),
                        "done": rollout.done,
                        "env_info": rollout.info,
                        "action_meta": action_meta,
                        "generated_source": str(generated.source_path) if generated.source_path else None,
                        "task_context": task_context,
                    }
                )
                success = self._success_from_info(rollout.info)
                rollouts.append(
                    {
                        "rollout_id": rollout_idx + 1,
                        "attempt_dir": attempt_dir.name,
                        "done": rollout.done,
                        "success": success,
                        "env_info": rollout.info,
                        "action_meta": action_meta,
                        "generated_source": str(generated.source_path) if generated.source_path else None,
                        "task_context": task_context,
                    }
                )
                if not save_attempts:
                    shutil.rmtree(attempt_dir, ignore_errors=True)
                if success:
                    break
                if len(rollout.rgb) == 0:
                    break
                first_frame = image_batch_to_first_frame(rollout.rgb[-1])
            h5_path = None
            if save_h5:
                h5_path = episode_dir / "trajectory.h5"
                self._write_rollout_h5(
                    h5_path,
                    attempt_records,
                    env_id=env_id,
                    episode=episode,
                    background=background,
                    background_type_id=background_type_id,
                    obj_type_id=obj_type_id,
                    task_context=task_context,
                    prompt=prompt,
                    pose_transforms=pose_transforms,
                )
            if save_generated and combined_generated:
                write_video(episode_dir / "generated.mp4", np.concatenate(combined_generated, axis=2), fps=fps)
            if combined_rollout:
                write_video(episode_dir / "rollout.mp4", np.asarray(combined_rollout), fps=fps)
            reached_max_rollouts = len(rollouts) >= max_rollouts and not success
            info = {
                "env_id": env_id,
                "episode": episode,
                "background": background,
                "background_type_id": background_type_id,
                "obj_type_id": obj_type_id,
                "task_context": task_context,
                "prompt": prompt,
                "done": bool(rollout.done) if rollout is not None else False,
                "success": success,
                "reached_max_rollouts": reached_max_rollouts,
                "max_rollouts": max_rollouts,
                "num_rollouts": len(rollouts),
                "env_info": rollout.info if rollout is not None else {},
                "pose_transforms": pose_transforms,
                "save_attempts": save_attempts,
                "save_generated": save_generated,
                "h5_path": str(h5_path) if h5_path is not None else None,
                "rollouts": rollouts,
            }
            write_json(episode_dir / "info.json", info)
            return {"episode": episode, "done": bool(rollout.done) if rollout is not None else False, "success": success, "error": ""}
        except Exception:
            print(f"[KineBench ERROR] episode failed | env_id={env_id} | episode={episode} | dir={episode_dir}")
            traceback.print_exc()
            raise
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
    def _write_rollout_h5(path: Path, attempts: list[dict], **attrs) -> None:
        try:
            import h5py
        except ImportError as exc:
            raise RuntimeError("output.save_h5 requires h5py to be installed") from exc

        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as f:
            f.attrs["format"] = "kinebench_rollout_v1"
            for key, value in attrs.items():
                if key == "pose_transforms" and isinstance(value, dict):
                    grp = f.create_group("pose_transforms")
                    for name, mat in value.items():
                        KineBenchEvaluator._h5_write_dataset(grp, name, mat)
                else:
                    f.attrs[key] = KineBenchEvaluator._h5_attr_value(value)
            for item in attempts:
                grp = f.create_group(f"traj_{int(item['rollout_id']) - 1}")
                grp.attrs["rollout_id"] = int(item["rollout_id"])
                grp.attrs["attempt_dir"] = str(item["attempt_dir"])
                grp.attrs["done"] = bool(item["done"])
                grp.attrs["generated_source"] = str(item.get("generated_source") or "")
                KineBenchEvaluator._h5_write_dataset(grp, "actions", item["actions"])
                KineBenchEvaluator._h5_write_dataset(grp, "generated_frames", item["generated_frames"], compression="gzip")
                KineBenchEvaluator._h5_write_dataset(grp, "rollout_rgb", item["rollout_rgb"], compression="gzip")
                KineBenchEvaluator._h5_write_dataset(grp, "side_by_side", item["side_by_side"], compression="gzip")
                eef_states = KineBenchEvaluator._stack_eef_states(item.get("eef_states", []))
                if eef_states is not None:
                    KineBenchEvaluator._h5_write_dataset(grp, "eef_states", eef_states)
                grp.attrs["env_info_json"] = KineBenchEvaluator._json_dumps(item.get("env_info", {}))
                grp.attrs["action_meta_json"] = KineBenchEvaluator._json_dumps(item.get("action_meta", {}))

    @staticmethod
    def _h5_write_dataset(group, name: str, value, compression: str | None = None) -> None:
        arr = np.asarray(value)
        if arr.dtype == object:
            arr = arr.astype(str)
        kwargs = {"compression": compression} if compression and arr.size > 0 else {}
        group.create_dataset(name, data=arr, **kwargs)

    @staticmethod
    def _stack_eef_states(states) -> np.ndarray | None:
        if not states:
            return None
        arrays = [np.asarray(x) for x in states if x is not None]
        if not arrays:
            return None
        try:
            return np.stack(arrays, axis=0)
        except ValueError:
            return np.asarray(arrays, dtype=object)

    @staticmethod
    def _json_dumps(value) -> str:
        import json

        from kinebench.io import to_jsonable

        return json.dumps(to_jsonable(value), ensure_ascii=False)

    @staticmethod
    def _h5_attr_value(value):
        if isinstance(value, (str, int, float, bool, np.integer, np.floating, np.bool_)) or value is None:
            return "" if value is None else value
        return KineBenchEvaluator._json_dumps(value)

    @staticmethod
    def _write_side_by_side(path: Path, rollout_rgb: np.ndarray, generated: np.ndarray, fps: int) -> None:
        frames = KineBenchEvaluator._side_by_side_frames(rollout_rgb, generated)
        if frames:
            write_video(path, np.asarray(frames), fps=fps)

    @staticmethod
    def _side_by_side_frames(rollout_rgb: np.ndarray, generated: np.ndarray) -> list[np.ndarray]:
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
        return frames

