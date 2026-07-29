# KineBench

KineBench is an evaluation harness for video-to-action robotic manipulation on ManiSkill. It evaluates a generated manipulation video by extracting object pose and gripper actions, rolling those actions back into ManiSkill, and writing videos, trajectories, and JSON summaries for inspection.

## Pipeline

```text
ManiSkill reset
  -> render first frame and build task prompt
  -> video generator: local / HTTP VideoX / MiniMax / DashScope
  -> read generated video as [B, 3, T, H, W]
  -> MoGe depth/intrinsics
  -> YOLO mask + FoundationPose tracking
  -> gripper prediction + PyRoki IK / end-effector actions
  -> ManiSkill rollout
  -> generated.mp4, rollout.mp4, trajectory.h5, info.json, summary.csv
```

The first-frame prompt includes a visual lock instruction by default, so video generators should preserve the camera, robot identity, geometry, and scene layout.

## Repository Layout

- `kinebench/`: Python package for generation, perception, planning, tasks, and evaluation.
- `configs/eval/`: ready-to-run YAML configs.
- `scripts/`: CLI entrypoints and helper services.
- `docs/`: interface notes, including the VideoX HTTP contract.
- `third_party/`: vendored ManiSkill, FoundationPose, PyRoki, and MoGe trees.
- `assets/`, `checkpoints/`: local assets and model weights.
- `outputs/`, `tmp/`: runtime outputs and scratch directories.

## Install

Use the `kinebench` environment for KineBench evaluation:

```bash
cd /zzz_new/projects/KineBench
conda activate kinebench
pip install -e ".[dev]"
```

The full pipeline also expects separate environments/configs for FoundationPose and PyRoki, as referenced in each YAML file:

```yaml
perception.foundationpose_env: foundationpose
pyroki.env_name: pyroki
```

VideoX local generation uses the separate `videox` conda environment and the source tree at:

```text
/zzz_new/projects/VideoX-Fun
```

Large weights, generated videos, logs, and runtime outputs should stay outside git.

## Evaluation Configs

| Config | Purpose |
| --- | --- |
| `configs/eval/local_smoke.yaml` | Minimal local-video plumbing test. |
| `configs/eval/fake_close_box_video.yaml` | Single fake-video evaluation example. |
| `configs/eval/fake_close_faucet_video.yaml` | Single fake-video evaluation example. |
| `configs/eval/maniskill_videox_http.yaml` | KineBench calls a local VideoX HTTP service. |
| `configs/eval/maniskill_minimax_i2v.yaml` | KineBench calls MiniMax Hailuo image-to-video API. |
| `configs/eval/maniskill_wan26.yaml` | DashScope Wan2.6 image-to-video API path. |

## Supported Task Prompts

Default prompts live in `kinebench/tasks.py`. They include static prompts for manipulation tasks and dynamic templates for fruit tasks:

- `PickFruits-v1`: fills the target `{fruit_name}` from the environment.
- `StoreFruitsBox-v1`: fills `{fruit_sequence_text}` from the environment.
- Box, drawer, faucet, laptop, cube, and peg tasks are configured in `TASK_PROMPTS`.

You can override prompts in a YAML file:

```yaml
prompts:
  CloseBox-v1: "Custom prompt text..."
  PickFruits-v1: "Pick the {fruit_name} and place it into the container."
  StoreFruitsBox-v1: "Store fruits in this order: {fruit_sequence_text}."
```

## Local Smoke Test

Create a tiny video and run the simplest plumbing check:

```bash
cd /zzz_new/projects/KineBench

python - <<'PY'
import numpy as np
v = np.zeros((4, 128, 128, 3), dtype=np.uint8)
v[..., 0] = 180
np.save("examples/local_video.npy", v)
PY

conda run -n kinebench python scripts/run_eval.py \
  --config configs/eval/local_smoke.yaml
```

This path avoids external APIs and most heavy perception components.

## Fake-Video Evaluation

Run one fake video:

```bash
cd /zzz_new/projects/KineBench

conda run -n kinebench python scripts/run_fake_video_eval.py \
  --config configs/eval/fake_close_faucet_video.yaml
```

Override the video path:

```bash
conda run -n kinebench python scripts/run_fake_video_eval.py \
  --config configs/eval/fake_close_faucet_video.yaml \
  --video-path /path/to/observation.images.base_camera.mp4
```

Run a batch over discovered dataset videos:

```bash
conda run -n kinebench python scripts/run_all_fake_video_eval.py \
  --config configs/eval/fake_close_faucet_video.yaml \
  --tasks close_box,close_faucet,lift_peg \
  --table-videos 1 \
  --robocasa-videos 51
```

Use `--dry-run` first to print selected cases without executing.

## VideoX HTTP Service

Use this when a local VideoX service should generate videos for KineBench. The HTTP contract is documented in `docs/videox_http_service.md`.

Start the service in the `videox` environment:

```bash
cd /zzz_new/projects/KineBench

conda run -n videox python scripts/videox_http_service.py \
  --host 127.0.0.1 \
  --port 8008
```

The service defaults to:

```text
VIDEOX_ROOT=/zzz_new/projects/VideoX-Fun
VIDEOX_MODEL_PATH=/zzz_new/projects/VideoX-Fun/models/Diffusion_Transformer/Wan2.1-Fun-V1.1-1.3B-InP
VIDEOX_OUT_DIR=/zzz_new/projects/KineBench/tmp/videox_http_outputs
```

Optional tuning:

```bash
VIDEOX_STEPS=30 \
VIDEOX_GPU_MEMORY_MODE=model_cpu_offload \
VIDEOX_OUT_DIR=/zzz_new/projects/KineBench/tmp/videox_http_outputs \
conda run -n videox python scripts/videox_http_service.py --port 8008
```

Run KineBench against the service in another terminal:

```bash
cd /zzz_new/projects/KineBench

conda run -n kinebench python scripts/run_eval.py \
  --config configs/eval/maniskill_videox_http.yaml \
  --env-id CloseBox-v1 \
  --num-episodes 1
```

If the machine has `http_proxy` set, KineBench bypasses proxies automatically for `127.0.0.1`, `localhost`, and `::1` endpoints.

## MiniMax Hailuo I2V Evaluation

Use this when MiniMax should generate the video. The current config targets `MiniMax-Hailuo-02` image-to-video, `512P`, `6s`.

Set your API key:

```bash
export MINIMAX_API_KEY="your MiniMax API key"
```

Run one episode:

```bash
cd /zzz_new/projects/KineBench

conda run -n kinebench python scripts/run_eval.py \
  --config configs/eval/maniskill_minimax_i2v.yaml \
  --env-id CloseBox-v1 \
  --num-episodes 1
```

Try fruit tasks with dynamic prompts:

```bash
conda run -n kinebench python scripts/run_eval.py \
  --config configs/eval/maniskill_minimax_i2v.yaml \
  --env-id PickFruits-v1 \
  --num-episodes 1

conda run -n kinebench python scripts/run_eval.py \
  --config configs/eval/maniskill_minimax_i2v.yaml \
  --env-id StoreFruitsBox-v1 \
  --num-episodes 1
```

Important generation fields:

```yaml
generation:
  kind: minimax
  model: MiniMax-Hailuo-02
  duration: 6
  resolution: 512P
  target_frames: 49
  size: [384, 384]
  poll_interval_sec: 10
  timeout_sec: 1800
```

The MiniMax adapter submits a task, polls until success, downloads the resulting mp4, then reads it into KineBench.

## DashScope Wan2.6 Evaluation

The DashScope path is still available through `configs/eval/maniskill_wan26.yaml`.

```bash
export DASHSCOPE_API_KEY="your DashScope API key"

conda run -n kinebench python scripts/run_eval.py \
  --config configs/eval/maniskill_wan26.yaml \
  --env-id StackCube-v1 \
  --num-episodes 1
```

## Output Files

Evaluation outputs are written to:

```text
outputs/{run_name}_{YYYYMMDD_HHMMSS}/{env_id}/{episode_id}/
```

Useful files:

```text
info.json                 # task, prompt, rollout metadata, success flag
generated.mp4             # generated source video, if enabled
rollout.mp4               # side-by-side generated/rollout video
trajectory.h5             # attempts, actions, poses, frames, if enabled
attempt_000/generated.mp4
attempt_000/rollout.mp4
summary.csv
```

Aggregate summaries:

```bash
conda run -n kinebench python scripts/analyze_results.py outputs
```

## Common Issues

`KeyError: No prompt configured for env_id=...`
: Add or override the task prompt in `kinebench/tasks.py` or the YAML `prompts:` section.

`503 Server Error` for `http://127.0.0.1:8008/generate`
: Usually caused by environment HTTP proxies. The HTTP generator now disables proxy usage for loopback endpoints. Also verify the VideoX service is running:

```bash
curl --noproxy 127.0.0.1 http://127.0.0.1:8008/health
```

`MiniMax API key is required`
: Export `MINIMAX_API_KEY`, or set `generation.api_key` in the YAML.

Returned/generated video is too short
: KineBench pads by repeating the last frame when needed. It samples or truncates extra frames according to `generation.sampling`.

## Python API

```python
from kinebench import KineBenchEvaluator, load_config

cfg = load_config("configs/eval/maniskill_minimax_i2v.yaml")
out_dir = KineBenchEvaluator(cfg).run(env_id="CloseBox-v1", num_episodes=1)
print(out_dir)
```

## License And Notice

KineBench has its own license in `LICENSE`. Vendored third-party projects retain their original licenses and notices; see `NOTICE.md` and each `third_party/*/LICENSE`.
