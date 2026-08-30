"""Probe A training data: SAD `stages_oversight`.

Probe A is the transferred, out-of-domain probe (§4.2). Nguyen et al.'s released
probe cannot serve here -- it was fit on Llama-3.3-70B (hidden dim 8192) and is
dimensionally inapplicable to Llama-3.1-8B -- so Probe A is trained on a chat
eval/deploy split instead. It stays out-of-domain relative to our agentic
trajectories, which is what the A/B contrast requires.

Source: github.com/LRudL/sad, `sad/stages/oversight`. Two batches carry the
binary split:

    test_oversight    splits.answer == "yes"   -> eval-like    (label 1)
    deploy_oversight  splits.answer == "no"    -> deploy-like  (label 0)

DATA HANDLING. SAD's README requires that question and answer text never appear
in plain text anywhere scrapable, including private repositories. Everything
here downloads into `data/sad/`, which .gitignore excludes. Do not copy SAD text
into results, notebooks, logs, or commit messages.
"""

from __future__ import annotations

import io
import os
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import yaml

RAW = "https://raw.githubusercontent.com/LRudL/sad/main/sad/stages/oversight"
# Published in the SAD repo's own unzip.sh; the zip is a scraping deterrent,
# not a secret.
ZIP_PASSWORD = b"sadtimesforthesetimes"

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "sad"

EVAL_BATCH = "test_oversight"
DEPLOY_BATCH = "deploy_oversight"


@dataclass
class ChatSplit:
    texts: list[str]
    labels: list[int]        # 1 = eval-like, 0 = deploy-like
    ids: list[str]
    source: str = "sad/stages/oversight"

    def __len__(self) -> int:
        return len(self.texts)


def download(dest: Path | None = None, force: bool = False) -> Path:
    """Fetch and decrypt structs.zip into `data/sad/structs/`."""
    d = dest or DATA_DIR
    d.mkdir(parents=True, exist_ok=True)
    structs = d / "structs"
    if structs.exists() and not force and any(structs.rglob("*.yaml")):
        return structs

    zip_path = d / "structs.zip"
    if not zip_path.exists() or force:
        with urllib.request.urlopen(f"{RAW}/structs.zip", timeout=120) as r:
            zip_path.write_bytes(r.read())

    with zipfile.ZipFile(zip_path) as z:
        z.extractall(structs, pwd=ZIP_PASSWORD)
    return structs


def _load_batch(structs: Path, name: str) -> list[dict]:
    f = structs / "batch" / f"{name}.yaml"
    if not f.exists():
        raise FileNotFoundError(f"missing SAD batch {name} at {f}")
    with f.open() as fh:
        doc = yaml.safe_load(fh)
    samples = doc.get("samples") if isinstance(doc, dict) else doc
    if not samples:
        raise ValueError(f"no samples in {f}")
    return samples


def load_oversight(structs: Path | None = None) -> ChatSplit:
    """Load the balanced eval/deploy chat split used to train Probe A."""
    s = structs or download()
    texts, labels, ids = [], [], []
    for name, label in ((EVAL_BATCH, 1), (DEPLOY_BATCH, 0)):
        for smp in _load_batch(s, name):
            body = smp.get("body")
            if not body:
                continue
            texts.append(body)
            labels.append(label)
            ids.append(str(smp.get("id", f"{name}_{len(ids)}")))
    if not texts:
        raise ValueError("SAD oversight split loaded empty")
    return ChatSplit(texts, labels, ids)


# --------------------------------------------------------------------------
# surface baselines on the Probe A split (§3.3, §4.4)
# --------------------------------------------------------------------------

SPECIAL_CHARS = set("{}[]()<>\"'`,;:/\\|=+-_*&^%$#@!~?.")


def surface_features(texts: list[str]) -> dict[str, list[float]]:
    """Length and special-character counts, the two baselines the §4.4 gate must
    be cleared against. Nguyen et al.'s 0.584 / 0.598 were measured on a
    different model and split, so we recompute ours rather than compare to theirs."""
    return {
        "chars": [float(len(t)) for t in texts],
        "special_chars": [float(sum(c in SPECIAL_CHARS for c in t)) for t in texts],
    }


def baseline_auroc(split: ChatSplit) -> dict[str, float]:
    """AUROC a length-only or special-character-only classifier achieves here."""
    from .analysis import auroc

    feats = surface_features(split.texts)
    out = {}
    for name, vals in feats.items():
        neg = [v for v, y in zip(vals, split.labels) if y == 0]
        pos = [v for v, y in zip(vals, split.labels) if y == 1]
        a = auroc(neg, pos)
        # a baseline that separates in either direction is equally damaging
        out[name] = max(a, 1.0 - a)
    return out
