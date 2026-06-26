from __future__ import annotations

import csv
from pathlib import Path


def summarize_results(root: str | Path, out_csv: str | Path | None = None) -> Path:
    root = Path(root)
    rows = []
    for summary in root.glob("**/summary.csv"):
        if summary.name == "all_summary.csv":
            continue
        with summary.open("r", encoding="utf8") as f:
            items = list(csv.DictReader(f))
        total = len(items)
        success = sum(str(x.get("success", "")).lower() == "true" for x in items)
        rows.append({"path": str(summary.parent), "episodes": total, "success": success, "success_rate": success / total if total else 0.0})
    out = Path(out_csv) if out_csv else root / "all_summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "episodes", "success", "success_rate"])
        writer.writeheader()
        writer.writerows(rows)
    return out

