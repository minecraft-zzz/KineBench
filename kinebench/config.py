from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml


def _merge_dict(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), dict):
            out[key] = _merge_dict(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(os.path.expanduser(value))
    if isinstance(value, list):
        return [_expand(v) for v in value]
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    return value


def load_config(path: str | Path, overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg.setdefault("repo_root", str(path.parents[2] if len(path.parents) >= 3 else Path.cwd()))
    if overrides:
        cfg = _merge_dict(cfg, overrides)
    return _expand(cfg)


def require_path(value: str | Path | None, name: str) -> Path:
    if value is None or str(value) == "":
        raise ValueError(f"Missing required path: {name}")
    path = Path(value)
    if not path.exists():
        raise FileNotFoundError(f"{name} does not exist: {path}")
    return path

