from __future__ import annotations

import subprocess
from pathlib import Path


def run_foundationpose(base_path: str | Path, repo: str | Path, conda_bin: str = "conda", env_name: str = "foundationpose", max_workers: int = 8) -> None:
    cmd = [conda_bin, "run", "-n", env_name, "python", "run_fp_grpo.py", "--base_path", str(base_path), "--max_workers", str(max_workers)]
    subprocess.run(cmd, cwd=str(repo), check=True)

