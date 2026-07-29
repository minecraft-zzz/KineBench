from __future__ import annotations

import argparse
import base64
import os
import sys
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from fastapi import FastAPI, HTTPException
from omegaconf import OmegaConf
from pydantic import BaseModel, Field


DEFAULT_VIDEOX_ROOT = Path("/zzz_new/projects/VideoX-Fun")
DEFAULT_MODEL_PATH = DEFAULT_VIDEOX_ROOT / "models/Diffusion_Transformer/Wan2.1-Fun-V1.1-1.3B-InP"
DEFAULT_OUT_DIR = Path("/zzz_new/projects/KineBench/tmp/videox_http_outputs")
DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走"
)


def _insert_videox_paths(videox_root: Path) -> None:
    root = str(videox_root.resolve())
    examples = str((videox_root / "examples" / "wan2.1_fun").resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    if examples not in sys.path:
        sys.path.insert(0, examples)


class GenerateRequest(BaseModel):
    prompt: str
    negative_prompt: str | None = None
    seed: int | None = None
    first_frame: str
    target_frames: int = 49
    num_frames: int = 49
    size: list[int] = Field(default_factory=lambda: [384, 384])
    output_size: list[int] = Field(default_factory=lambda: [384, 384])
    guidance_scale: float | None = None
    num_inference_steps: int | None = None
    fps: int | None = None
    shift: float | None = None


@dataclass(frozen=True)
class ServiceConfig:
    videox_root: Path
    model_path: Path
    config_path: Path
    out_dir: Path
    gpu_memory_mode: str
    sampler_name: str
    default_steps: int
    default_guidance_scale: float
    default_fps: int
    default_shift: float
    weight_dtype: torch.dtype
    enable_teacache: bool
    teacache_threshold: float
    num_skip_start_steps: int
    teacache_offload: bool
    cfg_skip_ratio: float | None


class VideoXGenerator:
    def __init__(self, cfg: ServiceConfig):
        self.cfg = cfg
        self._lock = threading.Lock()
        self._loaded = False
        self._runtime: dict[str, Any] = {}

    def generate(self, req: GenerateRequest, first_frame_path: Path, output_path: Path) -> None:
        with self._lock:
            self._load_once()
            self._generate_locked(req, first_frame_path, output_path)

    def _load_once(self) -> None:
        if self._loaded:
            return

        _insert_videox_paths(self.cfg.videox_root)
        os.chdir(self.cfg.videox_root)

        from videox_fun.dist import set_multi_gpus_devices
        from videox_fun.models import AutoencoderKLWan, CLIPModel, WanT5EncoderModel, WanTransformer3DModel
        from videox_fun.models.cache_utils import get_teacache_coefficients
        from videox_fun.pipeline import WanFunInpaintPipeline
        from videox_fun.utils import register_auto_device_hook, safe_enable_group_offload
        from videox_fun.utils.fm_solvers import FlowDPMSolverMultistepScheduler
        from videox_fun.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
        from videox_fun.utils.fp8_optimization import convert_model_weight_to_float8, convert_weight_dtype_wrapper, replace_parameters_by_name
        from videox_fun.utils.utils import filter_kwargs
        from transformers import AutoTokenizer

        device = set_multi_gpus_devices(1, 1)
        config = OmegaConf.load(self.cfg.config_path)
        model_name = str(self.cfg.model_path)

        transformer = WanTransformer3DModel.from_pretrained(
            os.path.join(model_name, config["transformer_additional_kwargs"].get("transformer_subpath", "transformer")),
            transformer_additional_kwargs=OmegaConf.to_container(config["transformer_additional_kwargs"]),
            low_cpu_mem_usage=True,
            torch_dtype=self.cfg.weight_dtype,
        )
        vae = AutoencoderKLWan.from_pretrained(
            os.path.join(model_name, config["vae_kwargs"].get("vae_subpath", "vae")),
            additional_kwargs=OmegaConf.to_container(config["vae_kwargs"]),
        ).to(self.cfg.weight_dtype)
        tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(model_name, config["text_encoder_kwargs"].get("tokenizer_subpath", "tokenizer")),
        )
        text_encoder = WanT5EncoderModel.from_pretrained(
            os.path.join(model_name, config["text_encoder_kwargs"].get("text_encoder_subpath", "text_encoder")),
            additional_kwargs=OmegaConf.to_container(config["text_encoder_kwargs"]),
            low_cpu_mem_usage=True,
            torch_dtype=self.cfg.weight_dtype,
        ).eval()
        clip_image_encoder = CLIPModel.from_pretrained(
            os.path.join(model_name, config["image_encoder_kwargs"].get("image_encoder_subpath", "image_encoder")),
        ).to(self.cfg.weight_dtype).eval()

        schedulers = {
            "Flow": FlowMatchEulerDiscreteScheduler,
            "Flow_Unipc": FlowUniPCMultistepScheduler,
            "Flow_DPM++": FlowDPMSolverMultistepScheduler,
        }
        scheduler_cls = schedulers[self.cfg.sampler_name]
        if self.cfg.sampler_name in {"Flow_Unipc", "Flow_DPM++"}:
            config["scheduler_kwargs"]["shift"] = 1
        scheduler = scheduler_cls(**filter_kwargs(scheduler_cls, OmegaConf.to_container(config["scheduler_kwargs"])))

        pipeline = WanFunInpaintPipeline(
            transformer=transformer,
            vae=vae,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            scheduler=scheduler,
            clip_image_encoder=clip_image_encoder,
        )

        if self.cfg.gpu_memory_mode == "sequential_cpu_offload":
            replace_parameters_by_name(transformer, ["modulation"], device=device)
            transformer.freqs = transformer.freqs.to(device=device)
            pipeline.enable_sequential_cpu_offload(device=device)
        elif self.cfg.gpu_memory_mode == "model_group_offload":
            register_auto_device_hook(pipeline.transformer)
            safe_enable_group_offload(pipeline, onload_device=device, offload_device="cpu", offload_type="leaf_level", use_stream=True)
        elif self.cfg.gpu_memory_mode == "model_cpu_offload_and_qfloat8":
            convert_model_weight_to_float8(transformer, exclude_module_name=["modulation"], device=device)
            convert_weight_dtype_wrapper(transformer, self.cfg.weight_dtype)
            pipeline.enable_model_cpu_offload(device=device)
        elif self.cfg.gpu_memory_mode == "model_full_load_and_qfloat8":
            convert_model_weight_to_float8(transformer, exclude_module_name=["modulation"], device=device)
            convert_weight_dtype_wrapper(transformer, self.cfg.weight_dtype)
            pipeline.to(device=device)
        elif self.cfg.gpu_memory_mode == "model_cpu_offload":
            pipeline.enable_model_cpu_offload(device=device)
        else:
            pipeline.to(device=device)

        coefficients = get_teacache_coefficients(model_name) if self.cfg.enable_teacache else None
        self._runtime = {
            "config": config,
            "device": device,
            "pipeline": pipeline,
            "vae": vae,
            "coefficients": coefficients,
        }
        self._loaded = True

    def _generate_locked(self, req: GenerateRequest, first_frame_path: Path, output_path: Path) -> None:
        from videox_fun.utils.utils import get_image_to_video_latent, save_videos_grid

        pipeline = self._runtime["pipeline"]
        vae = self._runtime["vae"]
        device = self._runtime["device"]
        coefficients = self._runtime["coefficients"]

        width, height = _coerce_size(req.output_size or req.size)
        num_frames = int(req.num_frames or req.target_frames)
        num_frames = max(1, num_frames)
        video_length = int((num_frames - 1) // vae.config.temporal_compression_ratio * vae.config.temporal_compression_ratio) + 1
        steps = int(req.num_inference_steps or self.cfg.default_steps)
        guidance_scale = float(req.guidance_scale or self.cfg.default_guidance_scale)
        fps = int(req.fps or self.cfg.default_fps)
        shift = float(req.shift or self.cfg.default_shift)
        seed = int(req.seed if req.seed is not None else 43)

        if coefficients is not None:
            pipeline.transformer.enable_teacache(
                coefficients,
                steps,
                self.cfg.teacache_threshold,
                num_skip_start_steps=self.cfg.num_skip_start_steps,
                offload=self.cfg.teacache_offload,
            )
        if self.cfg.cfg_skip_ratio is not None:
            pipeline.transformer.enable_cfg_skip(self.cfg.cfg_skip_ratio, steps)

        generator = torch.Generator(device=device).manual_seed(seed)
        sample_size = [height, width]
        input_video, input_video_mask, clip_image = get_image_to_video_latent(
            str(first_frame_path),
            None,
            video_length=video_length,
            sample_size=sample_size,
        )

        with torch.no_grad():
            sample = pipeline(
                req.prompt,
                num_frames=video_length,
                negative_prompt=req.negative_prompt or DEFAULT_NEGATIVE_PROMPT,
                height=height,
                width=width,
                generator=generator,
                guidance_scale=guidance_scale,
                num_inference_steps=steps,
                video=input_video,
                mask_video=input_video_mask,
                clip_image=clip_image,
                shift=shift,
            ).videos

        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_videos_grid(sample, str(output_path), fps=fps)


def _coerce_size(size: list[int]) -> tuple[int, int]:
    if len(size) != 2:
        raise HTTPException(status_code=400, detail=f"size/output_size must contain [width, height], got {size!r}")
    width, height = int(size[0]), int(size[1])
    if width <= 0 or height <= 0:
        raise HTTPException(status_code=400, detail=f"size/output_size must be positive, got {size!r}")
    return width, height


def save_first_frame(first_frame_data_uri: str, path: Path) -> None:
    if "," in first_frame_data_uri:
        payload = first_frame_data_uri.split(",", 1)[1]
    else:
        payload = first_frame_data_uri
    try:
        raw = base64.b64decode(payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="first_frame must be a base64 PNG data URI") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)


def create_app(cfg: ServiceConfig) -> FastAPI:
    app = FastAPI(title="VideoX KineBench HTTP Service")
    generator = VideoXGenerator(cfg)

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "model_loaded": generator._loaded, "model_path": str(cfg.model_path)}

    @app.post("/generate")
    def generate(req: GenerateRequest) -> dict[str, str]:
        stem = f"seed{req.seed if req.seed is not None else 'none'}_{uuid.uuid4().hex[:8]}"
        first_frame_path = cfg.out_dir / f"{stem}_first_frame.png"
        output_path = cfg.out_dir / f"{stem}.mp4"
        save_first_frame(req.first_frame, first_frame_path)
        try:
            generator.generate(req, first_frame_path, output_path)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"VideoX generation failed: {exc}") from exc
        return {"video_path": str(output_path)}

    return app


def build_config_from_env() -> ServiceConfig:
    videox_root = Path(os.environ.get("VIDEOX_ROOT", str(DEFAULT_VIDEOX_ROOT))).resolve()
    model_path = Path(os.environ.get("VIDEOX_MODEL_PATH", str(DEFAULT_MODEL_PATH))).resolve()
    return ServiceConfig(
        videox_root=videox_root,
        model_path=model_path,
        config_path=Path(os.environ.get("VIDEOX_CONFIG_PATH", str(videox_root / "config/wan2.1/wan_civitai.yaml"))).resolve(),
        out_dir=Path(os.environ.get("VIDEOX_OUT_DIR", str(DEFAULT_OUT_DIR))).resolve(),
        gpu_memory_mode=os.environ.get("VIDEOX_GPU_MEMORY_MODE", "model_cpu_offload"),
        sampler_name=os.environ.get("VIDEOX_SAMPLER", "Flow"),
        default_steps=int(os.environ.get("VIDEOX_STEPS", "30")),
        default_guidance_scale=float(os.environ.get("VIDEOX_GUIDANCE_SCALE", "6.0")),
        default_fps=int(os.environ.get("VIDEOX_FPS", "16")),
        default_shift=float(os.environ.get("VIDEOX_SHIFT", "3")),
        weight_dtype=torch.float16 if os.environ.get("VIDEOX_WEIGHT_DTYPE", "bf16").lower() in {"fp16", "float16"} else torch.bfloat16,
        enable_teacache=os.environ.get("VIDEOX_TEACACHE", "1") not in {"0", "false", "False"},
        teacache_threshold=float(os.environ.get("VIDEOX_TEACACHE_THRESHOLD", "0.10")),
        num_skip_start_steps=int(os.environ.get("VIDEOX_TEACACHE_SKIP_START", "5")),
        teacache_offload=os.environ.get("VIDEOX_TEACACHE_OFFLOAD", "0") in {"1", "true", "True"},
        cfg_skip_ratio=float(os.environ.get("VIDEOX_CFG_SKIP_RATIO", "0")),
    )


app = create_app(build_config_from_env())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a KineBench-compatible VideoX HTTP service.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8008)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    import uvicorn

    if args.reload:
        uvicorn.run("scripts.videox_http_service:app", host=args.host, port=args.port, reload=True)
    else:
        uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
