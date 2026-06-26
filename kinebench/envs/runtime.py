from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class RolloutResult:
    rgb: np.ndarray
    eef_states: list[Any]
    done: bool
    info: Any


def add_vendor_paths(third_party_root: str | Path) -> None:
    root = Path(third_party_root)
    for child in ("ManiSkill-main", "MoGe-main"):
        path = root / child
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))


def make_single_env(
    env_id: str = "StackCube-v1",
    obj_type_id: list[int] | None = None,
    background: str = "Table",
    background_type_id: list[int] | None = None,
    control_mode: str = "pd_ee_pose",
    sim_backend: str = "cpu",
):
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401

    kwargs = {
        "id": env_id,
        "obs_mode": "rgb",
        "control_mode": control_mode,
        "render_mode": None,
        "sim_backend": sim_backend,
        "background": background,
    }
    if obj_type_id is not None:
        kwargs["obj_type_id"] = obj_type_id
    if background_type_id is not None:
        kwargs["background_type_id"] = background_type_id
    return gym.make(**kwargs)


def _cpu_numpy(x):
    try:
        import torch

        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return x


def rollout_actions(actions: np.ndarray, envs, repeats: int = 4) -> RolloutResult:
    actions = np.asarray(actions)
    if actions.ndim != 3:
        raise ValueError(f"Expected actions [B,T,A], got {actions.shape}")
    b, t, _ = actions.shape
    all_rgb, eef_states = [], []
    done = False
    info = {}
    for ti in range(t):
        obs = None
        terminated = truncated = False
        for _ in range(repeats):
            obs, _, terminated, truncated, info = envs.step(actions[:, ti, :] if b > 1 else actions[0, ti, :])
        if obs is None:
            continue
        rgb = _cpu_numpy(obs["sensor_data"]["base_camera"]["rgb"])
        tcp = _cpu_numpy(obs.get("extra", {}).get("tcp_pose"))
        all_rgb.append(rgb)
        eef_states.append(tcp)
        if b > 1:
            done = bool(np.all(np.asarray(_cpu_numpy(terminated)) | np.asarray(_cpu_numpy(truncated))))
        else:
            done = bool(np.asarray(_cpu_numpy(terminated)).item() or np.asarray(_cpu_numpy(truncated)).item())
        if done:
            break
    return RolloutResult(rgb=np.asarray(all_rgb), eef_states=eef_states, done=done, info=info)


class ManiSkillRuntime:
    def __init__(self, third_party_root: str | Path | None = None, control_mode: str = "pd_ee_pose", sim_backend: str = "cpu"):
        if third_party_root:
            add_vendor_paths(third_party_root)
        self.control_mode = control_mode
        self.sim_backend = sim_backend

    def make_env(self, env_id: str, background: str = "Table", obj_type_id: list[int] | None = None):
        return make_single_env(env_id=env_id, obj_type_id=obj_type_id, background=background, control_mode=self.control_mode, sim_backend=self.sim_backend)

    def reset_rgb(self, env) -> np.ndarray:
        obs, _ = env.reset()
        return _cpu_numpy(obs["sensor_data"]["base_camera"]["rgb"])

    def rollout(self, env, actions: np.ndarray, repeats: int = 4) -> RolloutResult:
        return rollout_actions(actions, env, repeats=repeats)

