"""Probe training, layer sweep, and the §4.4 gates.

Probe A  chat-trained, out-of-domain. Trained here rather than transferred:
         Nguyen et al.'s released probe is a Llama-3.3-70B direction (hidden dim
         8192) and cannot be applied to an 8B residual stream at all.
Probe B  in-domain ceiling, trained on our agentic contexts and HELD OUT BY
         TASK. Variants of one task share content, so a random split would leak
         between train and test and inflate the ceiling.

Without Probe B a weak Probe A is uninterpretable: "agents don't carry the
signal" and "the probe doesn't transfer" look identical. The A-B gap is itself
a result (§4.2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .analysis import auroc


@dataclass
class LayerResult:
    layer: int
    auroc_mean: float
    auroc_std: float
    folds: list[float] = field(default_factory=list)

    def __repr__(self) -> str:
        return f"L{self.layer:02d} auroc={self.auroc_mean:.3f}±{self.auroc_std:.3f}"


@dataclass
class FittedProbe:
    layer: int
    coef: np.ndarray
    intercept: float
    mean: np.ndarray
    scale: np.ndarray
    train_auroc: float
    cv_auroc: float
    info: dict = field(default_factory=dict)

    def score(self, X: np.ndarray) -> np.ndarray:
        """Signed distance from the decision boundary; higher = more eval-like."""
        Z = (X - self.mean) / self.scale
        return Z @ self.coef + self.intercept


def _standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    scale[scale < 1e-8] = 1.0
    return (X - mean) / scale, mean, scale


def _fit_logreg(X: np.ndarray, y: np.ndarray, C: float, seed: int):
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(C=C, max_iter=2000, random_state=seed)
    clf.fit(X, y)
    return clf


def cross_val_auroc(X: np.ndarray, y: Sequence[int], groups: Sequence | None = None,
                    n_splits: int = 5, C: float = 1.0, seed: int = 0) -> tuple[float, float, list[float]]:
    """Grouped CV when `groups` is given, stratified otherwise.

    For Probe B, `groups` MUST be task ids -- that is what "held out by task"
    means and it is the difference between a ceiling and a leak.
    """
    from sklearn.model_selection import GroupKFold, StratifiedKFold

    y = np.asarray(y)
    if groups is not None:
        g = np.asarray(groups)
        n = min(n_splits, len(np.unique(g)))
        splitter = GroupKFold(n_splits=n).split(X, y, g)
    else:
        n = min(n_splits, np.bincount(y).min())
        splitter = StratifiedKFold(n_splits=max(2, n), shuffle=True,
                                   random_state=seed).split(X, y)

    folds = []
    for tr, te in splitter:
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        Xtr, mean, scale = _standardize(X[tr])
        clf = _fit_logreg(Xtr, y[tr], C, seed)
        s = ((X[te] - mean) / scale) @ clf.coef_[0] + clf.intercept_[0]
        folds.append(auroc(s[y[te] == 0], s[y[te] == 1]))
    if not folds:
        return float("nan"), float("nan"), []
    return float(np.mean(folds)), float(np.std(folds)), folds


def layer_sweep(X_all: np.ndarray, y: Sequence[int], groups: Sequence | None = None,
                layers: Sequence[int] | None = None, C: float = 1.0,
                n_splits: int = 5, seed: int = 0, progress: bool = True) -> list[LayerResult]:
    """AUROC per layer.

    §4.1: sweep, never hardcode a layer from a paper using a different model
    size -- the layer of maximum linear recoverability shifts with scale
    (2606.29196), and the signal is often recoverable before the midpoint.
    """
    idxs = list(layers) if layers is not None else list(range(X_all.shape[1]))
    out = []
    for i, L in enumerate(idxs):
        m, s, folds = cross_val_auroc(X_all[:, L, :], y, groups, n_splits, C, seed)
        out.append(LayerResult(L, m, s, folds))
        if progress:
            print(f"\r  layer sweep {i+1}/{len(idxs)}  L{L:02d}={m:.3f}", end="", flush=True)
    if progress:
        print()
    return out


def best_layer(results: Sequence[LayerResult]) -> LayerResult:
    return max((r for r in results if not np.isnan(r.auroc_mean)),
               key=lambda r: r.auroc_mean)


def fit_at_layer(X_all: np.ndarray, y: Sequence[int], layer: int,
                 groups: Sequence | None = None, C: float = 1.0,
                 n_splits: int = 5, seed: int = 0) -> FittedProbe:
    """Fit on all data at one layer, reporting both train and CV AUROC.

    The CV number is the honest one; train AUROC is kept only to make
    overfitting visible at a glance.
    """
    X = X_all[:, layer, :]
    y = np.asarray(y)
    cv_mean, _, _ = cross_val_auroc(X, y, groups, n_splits, C, seed)
    Z, mean, scale = _standardize(X)
    clf = _fit_logreg(Z, y, C, seed)
    coef, intercept = clf.coef_[0], float(clf.intercept_[0])
    s = Z @ coef + intercept
    return FittedProbe(layer, coef, intercept, mean, scale,
                       train_auroc=auroc(s[y == 0], s[y == 1]),
                       cv_auroc=cv_mean,
                       info={"C": C, "n": len(y), "grouped": groups is not None})


# --------------------------------------------------------------------------
# gates (§4.4)
# --------------------------------------------------------------------------


def gate_probe_a(cv_auroc: float, baselines: dict[str, float],
                 threshold: float = 0.75,
                 embedding_auroc: float | None = None,
                 embedding_max: float = 0.65) -> dict:
    """Probe A must clear 0.75, beat our own surface baselines, AND not be
    separable at the embedding layer.

    The embedding check is a disqualifier, not a bar to clear. Layer 0 is the
    embedding output: nothing has been computed there, so separability at layer
    0 means the two classes differ in token identity -- a corpus or vocabulary
    confound. Length and special-character baselines cannot detect that, which
    is how a probe reading "benchmark item vs chat log" can score 0.9999 and
    still pass a naive gate.

    Nguyen et al.'s 0.517 / 0.584 / 0.598 were measured on Llama-3.3-70B and a
    different split; they are reference points, not comparators for an 8B model.
    """
    beats = {k: cv_auroc > v for k, v in baselines.items()}
    confounded = embedding_auroc is not None and embedding_auroc > embedding_max
    passed = cv_auroc >= threshold and all(beats.values()) and not confounded
    if confounded:
        why = (f"embedding-layer AUROC {embedding_auroc:.3f} > {embedding_max}: the "
               "classes are separable from raw token embeddings, so this probe "
               "reads surface form, not an evaluation-awareness representation")
    elif passed:
        why = "proceed"
    else:
        why = ("Probe A does not clear the gate; the transferred arm is "
               "uninterpretable and Probe B must carry the study")
    return {
        "gate": "probe_A", "auroc": cv_auroc, "threshold": threshold,
        "our_baselines": baselines, "beats_baselines": beats,
        "embedding_auroc": embedding_auroc, "embedding_max": embedding_max,
        "surface_confounded": confounded, "passed": passed, "consequence": why,
    }


def gate_probe_b(cv_auroc: float, threshold: float = 0.65) -> dict:
    """If Probe B is at chance, agentic trajectories carry no linearly
    recoverable signal, there is no quantity to attribute, and that null IS the
    finding -- stop and write it up (§4.4, §11)."""
    passed = cv_auroc > threshold
    return {
        "gate": "probe_B", "auroc": cv_auroc, "threshold": threshold,
        "passed": passed,
        "consequence": ("proceed to attribution" if passed else
                        "STOP: no linearly recoverable signal in agentic "
                        "trajectories; report the null with the A-B transfer gap"),
    }


# --------------------------------------------------------------------------
# out-of-fold scoring
# --------------------------------------------------------------------------


def fit_folds_and_score_all(X_train: np.ndarray, y_train: Sequence[int],
                            groups_train: Sequence[str], X_all: np.ndarray,
                            tasks_all: Sequence[str], layer: int,
                            n_splits: int = 5, C: float = 1.0,
                            seed: int = 0) -> tuple[np.ndarray, dict]:
    """Score EVERY context with a probe that never saw its task.

    Why this exists. Fitting Probe B on the base and deploy arms and then
    scoring those same contexts measures training-set fit, not an effect: an
    overfit probe separates its own training points almost perfectly, so Δ_base
    saturates at 1.0 and the F0 null test fails for reasons that have nothing to
    do with the pipeline leaking.

    Here a probe is fit per fold on the training arms of the training tasks,
    then applied to ALL arms of the held-out tasks -- including the control,
    null and stacked arms, which never appear in any training set. Every score
    used downstream is therefore out-of-sample with respect to its own task.
    """
    from sklearn.model_selection import GroupKFold

    y_train = np.asarray(y_train)
    g_train = np.asarray(groups_train)
    tasks_all = np.asarray(tasks_all)

    n_folds = min(n_splits, len(np.unique(g_train)))
    scores = np.full(len(X_all), np.nan)
    covered, fold_aurocs = set(), []

    for tr, te in GroupKFold(n_splits=n_folds).split(X_train, y_train, g_train):
        held_out = set(g_train[te])
        Xtr, mean, scale = _standardize(X_train[tr, layer, :])
        clf = _fit_logreg(Xtr, y_train[tr], C, seed)
        w, b = clf.coef_[0], float(clf.intercept_[0])

        # honest fold AUROC on the held-out training arms
        s_te = ((X_train[te, layer, :] - mean) / scale) @ w + b
        if len(np.unique(y_train[te])) == 2:
            fold_aurocs.append(auroc(s_te[y_train[te] == 0], s_te[y_train[te] == 1]))

        # apply to every arm of every held-out task
        mask = np.isin(tasks_all, list(held_out))
        scores[mask] = ((X_all[mask, layer, :] - mean) / scale) @ w + b
        covered |= held_out

    info = {
        "layer": layer, "n_folds": n_folds,
        "cv_auroc_mean": float(np.mean(fold_aurocs)) if fold_aurocs else float("nan"),
        "n_scored": int((~np.isnan(scores)).sum()),
        "n_unscored": int(np.isnan(scores).sum()),
        "tasks_covered": len(covered),
    }
    return scores, info
