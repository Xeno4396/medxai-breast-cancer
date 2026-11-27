"""Exp03.1: compare global SHAP and LIME feature rankings."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr


def _rank(values: np.ndarray) -> np.ndarray:
    """Return descending average ranks, with rank 1 assigned to the largest value."""
    return pd.Series(values).rank(method="average", ascending=False).to_numpy()


def run_global_comparison(context: object) -> pd.DataFrame:
    output_dir = Path(__file__).resolve().parent / "results" / "global"
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_names = context.feature_names

    shap_values = np.asarray(context.malignant_shap.values)
    shap_importance = np.mean(np.abs(shap_values), axis=0)

    lime_abs_weights = []
    for test_index in range(len(context.X_test)):
        explanation = context.lime_explanation(test_index)
        weights = dict(explanation.local_exp[1])
        lime_abs_weights.append(
            [abs(weights[feature_index]) for feature_index in range(len(feature_names))]
        )
    lime_importance = np.mean(np.asarray(lime_abs_weights), axis=0)

    shap_rank = _rank(shap_importance)
    lime_rank = _rank(lime_importance)
    shap_top = set(np.argsort(shap_importance)[-10:])
    lime_top = set(np.argsort(lime_importance)[-10:])
    overlap = shap_top & lime_top
    spearman = spearmanr(shap_rank, lime_rank)
    kendall = kendalltau(shap_rank, lime_rank)

    comparison = pd.DataFrame(
        {
            "feature": feature_names,
            "shap_mean_abs_value": shap_importance,
            "shap_rank": shap_rank,
            "shap_top_10": [i in shap_top for i in range(len(feature_names))],
            "lime_mean_abs_weight": lime_importance,
            "lime_rank": lime_rank,
            "lime_top_10": [i in lime_top for i in range(len(feature_names))],
            "in_both_top_10": [i in overlap for i in range(len(feature_names))],
            "top_10_overlap_count": len(overlap),
            "spearman_rank_correlation": float(spearman.statistic),
            "kendall_rank_correlation": float(kendall.statistic),
        }
    )
    comparison.to_csv(output_dir / "global_comparison.csv", index=False)

    for method, scores, filename, title, color in (
        (
            "SHAP",
            shap_importance,
            "shap_global_importance.png",
            "Global SHAP Importance (mean |class-1 SHAP value|)",
            "#2a6fbb",
        ),
        (
            "LIME",
            lime_importance,
            "lime_global_importance.png",
            "Global LIME Importance (mean |class-1 local weight|)",
            "#d97732",
        ),
    ):
        order = np.argsort(scores)[-10:]
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.barh(np.array(feature_names)[order], scores[order], color=color)
        ax.set_xlabel("Mean absolute contribution (method-specific scale)")
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=200, bbox_inches="tight")
        plt.close(fig)

    # Compare ranks, not raw contribution magnitudes.
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(shap_rank, lime_rank, color="#555555", alpha=0.75)
    label_features = shap_top | lime_top
    for i, name in enumerate(feature_names):
        if i in label_features:
            ax.annotate(name, (shap_rank[i], lime_rank[i]), fontsize=7, alpha=0.85)
    ax.set_xlabel("SHAP importance rank (1 = most important)")
    ax.set_ylabel("LIME importance rank (1 = most important)")
    ax.set_title(
        "Global Feature Rank Comparison\n"
        f"Top-10 overlap={len(overlap)}/10; "
        f"Spearman={spearman.statistic:.3f}; Kendall={kendall.statistic:.3f}"
    )
    ax.set_xlim(0, len(feature_names) + 1)
    ax.set_ylim(0, len(feature_names) + 1)
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "feature_rank_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(
        "Global comparison: overlap",
        len(overlap),
        "Spearman",
        round(float(spearman.statistic), 4),
        "Kendall",
        round(float(kendall.statistic), 4),
    )
    return comparison
