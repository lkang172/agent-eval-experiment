"""Load task specs from tasks/*.py. Each module exposes a `TASK: TaskSpec`."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from .schema import TaskSpec

TASKS_DIR = Path(__file__).resolve().parents[2] / "tasks"


def load_all(tasks_dir: Path | None = None) -> list[TaskSpec]:
    d = tasks_dir or TASKS_DIR
    for extra in (str(Path(__file__).resolve().parents[1]), str(d)):
        if extra not in sys.path:
            sys.path.insert(0, extra)
    out: list[TaskSpec] = []
    for p in sorted(d.glob("*.py")):
        if p.name.startswith("_"):
            continue
        spec = importlib.util.spec_from_file_location(f"rvr_task_{p.stem}", p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        task: TaskSpec = mod.TASK
        errs = task.validate()
        if errs:
            raise ValueError("invalid task spec:\n  " + "\n  ".join(errs))
        out.append(task)
    return out
