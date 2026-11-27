"""Exp03.3: compare SHAP and LIME explanations for baseline false negatives."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy.stats import kendalltau, spearmanr


def _rank(values: np.ndarray) -> np.ndarray:
    return pd.Series(np.abs(values)).rank(method="average", ascending=False).to_numpy()


def run_fn_comparison(context: object) -> pd.DataFrame:
    output_dir = Path(__file__).resolve().parent / "results" / "false_negative"
    output_dir.mkdir(parents=True, exist_ok=True)
    actual = context.y_test.to_numpy()
    fn_indices = np.flatnonzero((actual == 1) & (context.y_prediction == 0))
    if len(fn_indices) != 4:
        raise AssertionError(f"Expected 4 default-threshold FN samples; got {len(fn_indices)}")

    feature_names = context.feature_names
    rows = []
    for test_index in fn_indices:
        test_index = int(test_index)
        explanation = context.lime_explanation(test_index)
        lime_map = dict(explanation.local_exp[1])
        shap_weights = np.asarray(context.malignant_shap.values[test_index], dtype=float)
        lime_weights = np.asarray(
            [lime_map[i] for i in range(len(feature_names))], dtype=float
        )
        shap_ranks = _rank(shap_weights)
        lime_ranks = _rank(lime_weights)
        shap_top = set(np.argsort(np.abs(shap_weights))[-10:])
        lime_top = set(np.argsort(np.abs(lime_weights))[-10:])
        overlap = shap_top & lime_top
        spearman = spearmanr(shap_ranks, lime_ranks).statistic
        kendall = kendalltau(shap_ranks, lime_ranks).statistic

        for i, feature in enumerate(feature_names):
            rows.append(
                {
                    "test_index": test_index,
                    "actual_class": int(actual[test_index]),
                    "predicted_class": int(context.y_prediction[test_index]),
                    "malignant_probability": float(context.y_probability[test_index]),
                    "feature": feature,
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
                    "direction_agreement": np.sign(shap_weights[i])
                    == np.sign(lime_weights[i]),
                }
            )

        # Save the requested SHAP waterfall for malignant class 1.
        shap.plots.waterfall(context.malignant_shap[test_index], show=False)
        plt.gcf().suptitle(
            f"False Negative — test index {test_index}; "
            f"malignant probability {context.y_probability[test_index]:.3f}"
        )
        plt.savefig(
            output_dir / f"false_negative_{test_index}_shap.png",
            dpi=200,
            bbox_inches="tight",
        )
        plt.close("all")

        # Save LIME's class-1 local surrogate explanation.
        fig = explanation.as_pyplot_figure(label=1)
        fig.suptitle(
            f"LIME False Negative — test index {test_index}; "
            f"malignant probability {context.y_probability[test_index]:.3f}"
        )
        fig.tight_layout()
        fig.savefig(
            output_dir / f"false_negative_{test_index}_lime.png",
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(fig)

        # Compare direction and ranks in a paired plot without mixing value scales.
        top = shap_top | lime_top
        ordered = sorted(top, key=lambda i: min(shap_ranks[i], lime_ranks[i]))
        labels = [feature_names[i] for i in ordered]
        positions = np.arange(len(ordered))
        fig, axes = plt.subplots(1, 2, figsize=(12, max(5, len(ordered) * 0.28)))
        for ax, ranks, weights, method in (
            (axes[0], shap_ranks, shap_weights, "SHAP"),
            (axes[1], lime_ranks, lime_weights, "LIME"),
        ):
            colors = ["#20874b" if weights[i] >= 0 else "#bf3f3f" for i in ordered]
            ax.scatter([ranks[i] for i in ordered], positions, c=colors, s=35)
            ax.set_yticks(positions, labels)
            ax.invert_yaxis()
            ax.set_xlabel("Absolute contribution rank (1 = strongest)")
            ax.set_title(method)
            ax.grid(axis="x", alpha=0.25)
        fig.suptitle(
            f"FN test index {test_index}: paired rank and direction comparison\n"
            f"Top-10 overlap={len(overlap)}/10; Spearman={spearman:.3f}; "
            f"Kendall={kendall:.3f}. Green=positive, red=negative. "
            "Raw contribution scales are not compared."
        )
        fig.tight_layout()
        fig.savefig(
            output_dir / f"false_negative_{test_index}_comparison.png",
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(fig)

    comparison = pd.DataFrame(rows)
    comparison.to_csv(output_dir / "false_negative_comparison.csv", index=False)
    print("Default-threshold FN indices:", fn_indices.tolist())
    return comparison
