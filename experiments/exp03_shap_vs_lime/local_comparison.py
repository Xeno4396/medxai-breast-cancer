"""Exp03.2: compare local explanations for matched TP, TN, and FN samples."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr


def _rank(values: np.ndarray) -> np.ndarray:
    return pd.Series(np.abs(values)).rank(method="average", ascending=False).to_numpy()


def _lime_weights(context: object, test_index: int) -> np.ndarray:
    explanation = context.lime_explanation(test_index)
    mapped = dict(explanation.local_exp[1])
    return np.asarray([mapped[i] for i in range(len(context.feature_names))], dtype=float)


def _make_sample_rows(context: object, test_index: int, sample_type: str) -> list[dict]:
    feature_names = context.feature_names
    shap_weights = np.asarray(context.malignant_shap.values[test_index], dtype=float)
    lime_weights = _lime_weights(context, test_index)
    shap_ranks = _rank(shap_weights)
    lime_ranks = _rank(lime_weights)
    shap_top = set(np.argsort(np.abs(shap_weights))[-10:])
    lime_top = set(np.argsort(np.abs(lime_weights))[-10:])
    spearman = spearmanr(shap_ranks, lime_ranks).statistic
    kendall = kendalltau(shap_ranks, lime_ranks).statistic
    overlap = shap_top & lime_top

    return [
        {
            "test_index": test_index,
            "sample_type": sample_type,
            "actual_class": int(context.y_test.iloc[test_index]),
            "predicted_class": int(context.y_prediction[test_index]),
            "malignant_probability": float(context.y_probability[test_index]),
            "feature": feature_names[i],
            "shap_class1_contribution": shap_weights[i],
            "shap_abs_rank": shap_ranks[i],
            "shap_direction": "positive" if shap_weights[i] >= 0 else "negative",
            "lime_class1_local_weight": lime_weights[i],
            "lime_abs_rank": lime_ranks[i],
            "lime_direction": "positive" if lime_weights[i] >= 0 else "negative",
            "shap_top_10": i in shap_top,
            "lime_top_10": i in lime_top,
            "in_both_top_10": i in overlap,
            "top_10_overlap_count": len(overlap),
            "spearman_rank_correlation": float(spearman),
            "kendall_rank_correlation": float(kendall),
            "rank_difference_shap_minus_lime": shap_ranks[i] - lime_ranks[i],
            "direction_agreement": np.sign(shap_weights[i]) == np.sign(lime_weights[i]),
        }
        for i in range(len(feature_names))
    ]


def _save_rank_plot(context: object, test_index: int, sample_type: str, out_dir: Path) -> None:
    feature_names = context.feature_names
    shap_weights = np.asarray(context.malignant_shap.values[test_index], dtype=float)
    lime_weights = _lime_weights(context, test_index)
    shap_rank = _rank(shap_weights)
    lime_rank = _rank(lime_weights)
    top = set(np.argsort(np.abs(shap_weights))[-10:]) | set(
        np.argsort(np.abs(lime_weights))[-10:]
    )
    indices = sorted(top, key=lambda i: min(shap_rank[i], lime_rank[i]))
    labels = [feature_names[i] for i in indices]
    y_positions = np.arange(len(indices))

    fig, axes = plt.subplots(1, 2, figsize=(12, max(5, len(indices) * 0.28)))
    for ax, ranks, weights, method, color in (
        (axes[0], shap_rank, shap_weights, "SHAP", "#2a6fbb"),
        (axes[1], lime_rank, lime_weights, "LIME", "#d97732"),
    ):
        vals = [ranks[i] for i in indices]
        colors = ["#20874b" if weights[i] >= 0 else "#bf3f3f" for i in indices]
        ax.scatter(vals, y_positions, c=colors, s=35)
        ax.set_yticks(y_positions, labels)
        ax.invert_yaxis()
        ax.set_xlabel("Absolute contribution rank (1 = strongest)")
        ax.set_title(f"{method} rank and direction")
        ax.grid(axis="x", alpha=0.25)
    fig.suptitle(
        f"{sample_type} — test index {test_index}; malignant probability "
        f"{context.y_probability[test_index]:.3f}\n"
        "Green = positive class-1 contribution/association; red = negative. "
        "Rank only; contribution scales are not compared."
    )
    fig.tight_layout()
    fig.savefig(
        out_dir / f"local_sample_{test_index}_comparison.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def run_local_comparison(context: object) -> pd.DataFrame:
    output_dir = Path(__file__).resolve().parent / "results" / "local"
    output_dir.mkdir(parents=True, exist_ok=True)
    actual = context.y_test.to_numpy()
    predicted = context.y_prediction
    tp = np.flatnonzero((actual == 1) & (predicted == 1))
    tn = np.flatnonzero((actual == 0) & (predicted == 0))
    fn = np.flatnonzero((actual == 1) & (predicted == 0))
    if not len(tp) or not len(tn):
        raise AssertionError("Could not select both a TP and TN sample.")

    tp_index = int(tp[np.argmin(np.abs(context.y_probability[tp] - 0.50))])
    tn_index = int(tn[np.argmin(np.abs(context.y_probability[tn] - 0.50))])
    selected = [(tp_index, "TP"), (tn_index, "TN")]
    selected.extend((int(i), "FN") for i in fn)

    rows = []
    for test_index, sample_type in selected:
        context.lime_explanation(test_index)
        rows.extend(_make_sample_rows(context, test_index, sample_type))
        _save_rank_plot(context, test_index, sample_type, output_dir)

    comparison = pd.DataFrame(rows)
    comparison.to_csv(output_dir / "local_comparison.csv", index=False)
    print(
        "Local samples:",
        {"TP": tp_index, "TN": tn_index, "FN": fn.tolist()},
    )
    return comparison
