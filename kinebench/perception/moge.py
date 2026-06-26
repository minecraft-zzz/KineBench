from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


def moge_infer_for_basepath(model, base_path: str | Path, fov_x: float = 60, device: str | None = None, micro_batch: int | None = None) -> None:
    base = Path(base_path)
    workspaces = [ws for ws in sorted(base.glob("workspace*")) if list((ws / "rgb").glob("rgb*.png"))]
    if not workspaces:
        raise FileNotFoundError(f"No workspaces with rgb frames under {base}")
    rgb_lists = [sorted((ws / "rgb").glob("rgb*.png")) for ws in workspaces]
    t_count = len(rgb_lists[0])
    if any(len(x) != t_count for x in rgb_lists):
        raise RuntimeError("All workspaces must have the same frame count for MoGe batching.")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    for ws in workspaces:
        (ws / "depth").mkdir(parents=True, exist_ok=True)
    intr_accum = [[] for _ in workspaces]
    mb = micro_batch or len(workspaces)
    with torch.no_grad():
        for t in range(t_count):
            for start in range(0, len(workspaces), mb):
                paths = [rgb_lists[i][t] for i in range(start, min(start + mb, len(workspaces)))]
                imgs = [torch.from_numpy(np.asarray(Image.open(p).convert("RGB")) / 255.0).float().permute(2, 0, 1) for p in paths]
                batch = torch.stack(imgs, dim=0).to(device)
                out = model.infer(batch, resolution_level=9, fov_x=fov_x)
                h, w = batch.shape[-2:]
                if "depth" in out:
                    depth = (out["depth"].detach().cpu().numpy() * 1000.0).clip(0, 65535).astype("uint16")
                    for j, ws in enumerate(workspaces[start : start + len(paths)]):
                        cv2.imwrite(str(ws / "depth" / f"depth{t:03d}.png"), depth[j])
                if "intrinsics" in out:
                    intr = out["intrinsics"].detach().cpu().numpy()
                    intr[:, 0, 0] *= w
                    intr[:, 1, 1] *= h
                    intr[:, 0, 2] = 0.5 * w
                    intr[:, 1, 2] = 0.5 * h
                    for j in range(intr.shape[0]):
                        intr_accum[start + j].append(intr[j])
    for ws, ks in zip(workspaces, intr_accum):
        k = np.mean(np.asarray(ks), axis=0)
        np.savetxt(ws / "cam_K.txt", k, fmt="%.6f")

