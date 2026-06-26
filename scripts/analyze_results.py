#!/usr/bin/env python
from __future__ import annotations

import argparse

from kinebench.eval.summary import summarize_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate KineBench summary.csv files.")
    parser.add_argument("root", help="Output root containing run summaries.")
    parser.add_argument("--out-csv", default=None, help="Optional output CSV path.")
    args = parser.parse_args()
    out = summarize_results(args.root, args.out_csv)
    print(f"Aggregate summary written to: {out}")


if __name__ == "__main__":
    main()

