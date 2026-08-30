"""Emit the §4.3 human rating sheet.  python -m rvr.make_rating_sheet [--n 50]"""

from __future__ import annotations

import argparse
from pathlib import Path

from .plausibility import stratified_sample, write_rating_sheet
from .taskloader import load_all

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--out", default=str(ROOT / "results" / "plausibility"))
    a = ap.parse_args()

    tasks = [t.task_id for t in load_all()]
    rows = stratified_sample(tasks, a.n)
    out = Path(a.out) / "rating_sheet.csv"
    write_rating_sheet(rows, ROOT / "trajectories", out)
    n = sum(1 for _ in out.open()) - 1
    print(f"wrote {n} items -> {out}")
    print(f"transcripts -> {out.parent / 'transcripts'}/")
    print(f"blind key   -> {out.parent / 'rating_key.json'}  (do not open before rating)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
