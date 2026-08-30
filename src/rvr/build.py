"""Generate every arm for every task, run the §8 audits, write artifacts.

    python -m rvr.build [--out DIR]

Writes trajectories/<task_id>/<arm_id>.json (§10 auditability) and
results/audit.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .audits import run_all
from .generate import render_all
from .taskloader import load_all

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT))
    args = ap.parse_args()
    out = Path(args.out)

    tasks = load_all()
    audits, n_ctx = [], 0
    for task in tasks:
        ctxs = render_all(task)
        d = out / "trajectories" / task.task_id
        d.mkdir(parents=True, exist_ok=True)
        for c in ctxs:
            (d / f"{c.arm_id}.json").write_text(json.dumps(c.to_dict(), indent=2, ensure_ascii=False))
        n_ctx += len(ctxs)
        audits.append(run_all(task, ctxs))

    (out / "results").mkdir(parents=True, exist_ok=True)
    summary = {
        "n_tasks": len(tasks),
        "n_contexts": n_ctx,
        "domains": sorted({t.domain for t in tasks}),
        "all_ok": all(a["all_ok"] for a in audits),
        "per_task": audits,
    }
    (out / "results" / "audit.json").write_text(json.dumps(summary, indent=2))
    print(f"{len(tasks)} tasks -> {n_ctx} contexts; audits all_ok={summary['all_ok']}")
    for a in audits:
        failed = [k for k, v in a.items() if isinstance(v, dict) and v.get("ok") is False]
        print(f"  {a['task_id']}: {'PASS' if not failed else 'FAIL ' + ','.join(failed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
