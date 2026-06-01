"""Model explainability: permutation feature importance for the churn pipeline.

Permutation importance measures how much the test AUC drops when each input
feature is randomly shuffled — a model-agnostic way to see which features the
model actually relies on. (No heavy extra dependencies beyond scikit-learn.)
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.inspection import permutation_importance

from .data import FEATURES


def permutation_importances(pipe, X, y, n_repeats: int = 10, seed: int = 42):
    """Return features sorted by mean permutation importance (drop in ROC AUC)."""
    r = permutation_importance(pipe, X, y, scoring="roc_auc",
                               n_repeats=n_repeats, random_state=seed, n_jobs=-1)
    order = np.argsort(r.importances_mean)[::-1]
    return [(FEATURES[i], float(r.importances_mean[i]), float(r.importances_std[i]))
            for i in order]


def plot_importances(rows, path: str, top: int = 15, title: str = ""):
    """Horizontal bar chart of the top-k most important features."""
    rows = rows[:top][::-1]
    names = [r[0] for r in rows]
    means = [r[1] for r in rows]
    errs = [r[2] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 0.4 * len(rows) + 1.5))
    ax.barh(names, means, xerr=errs, color="#1f77b4", capsize=3)
    ax.set_xlabel("Permutation importance (mean drop in ROC AUC)")
    ax.set_title(title or "Which features drive churn predictions?")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
