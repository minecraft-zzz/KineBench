from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def to_jsonable(x: Any) -> Any:
    try:
        import torch

        if isinstance(x, torch.Tensor):
            y = x.detach().cpu()
            return y.item() if y.numel() == 1 else y.tolist()
    except Exception:
        pass
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer, np.floating, np.bool_)):
        return x.item()
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    return x


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf8") as f:
        json.dump(to_jsonable(data), f, ensure_ascii=False, indent=2)

