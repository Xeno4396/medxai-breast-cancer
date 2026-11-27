"""Run the three Exp03 comparisons with one shared Random Forest model."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from importlib.metadata import version
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import scipy
import shap
import sklearn
from lime.lime_tabular import LimeTabularExplainer
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from fn_comparison import run_fn_comparison
from global_comparison import run_global_comparison
from local_comparison import run_local_comparison


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RANDOM_STATE = 42
LIME_NUM_SAMPLES = 5000
THRESHOLD = 0.50


@dataclass
class ExperimentContext:
    """Shared data, fitted model, and explainers for every Exp03 sub-study."""

    dataset: Any
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    model: RandomForestClassifier
    y_probability: np.ndarray
    y_prediction: np.ndarray
    malignant_shap: shap.Explanation
    lime_explainer: LimeTabularExplainer
    lime_num_samples: int = LIME_NUM_SAMPLES
    lime_cache: dict[int, Any] = field(default_factory=dict)

    @property
    def feature_names(self) -> list[str]:
        return list(self.X_test.columns)

    def lime_explanation(self, test_index: int) -> Any:
        """Return and cache a class-1 LIME explanation for a test-set row."""
        if test_index not in self.lime_cache:
            row = self.X_test.iloc[test_index].to_numpy(dtype=float)
            explanation = self.lime_explainer.explain_instance(
                row,
                self.predict_proba_with_feature_names,
                labels=(1,),
                num_features=len(self.feature_names),
                num_samples=self.lime_num_samples,
            )
            weights = explanation.local_exp.get(1, [])
            if len(weights) != len(self.feature_names):
                raise AssertionError(
                    f"LIME returned {len(weights)} malignant-class weights; "
                    f"expected {len(self.feature_names)}."
                )
            self.lime_cache[test_index] = explanation
        return self.lime_cache[test_index]

    def predict_proba_with_feature_names(self, instances: np.ndarray) -> np.ndarray:
        """Call the shared model while preserving the training feature names."""
        frame = pd.DataFrame(instances, columns=self.feature_names)
        return self.model.predict_proba(frame)


def build_context() -> ExperimentContext:
    """Load the baseline data split and fit exactly one shared model."""
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    dataset = load_breast_cancer(as_frame=True)
    X = dataset.data
    # Match Exp01/Exp02: malignant is class 1 and benign is class 0.
    y = (dataset.target == 0).astype(int)
    if X.shape != (569, 30):
        raise AssertionError(f"Unexpected WDBC shape: {X.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    if len(X_train) != 455 or len(X_test) != 114:
        raise AssertionError(
            f"Unexpected split sizes: train={len(X_train)}, test={len(X_test)}"
        )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    if list(model.classes_) != [0, 1]:
        raise AssertionError(f"Unexpected model class order: {model.classes_}")

    y_probability = model.predict_proba(X_test)[:, 1]
    y_prediction = (y_probability >= THRESHOLD).astype(int)

    # Match Exp01's reported scores and confusion matrix.
    metrics = {
        "accuracy": accuracy_score(y_test, y_prediction),
        "malignant_precision": precision_score(y_test, y_prediction, zero_division=0),
        "malignant_recall": recall_score(y_test, y_prediction, zero_division=0),
        "malignant_f1": f1_score(y_test, y_prediction, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_probability),
    }
    expected = {
        "accuracy": 110 / 114,
        "malignant_precision": 1.0,
        "malignant_recall": 38 / 42,
        "malignant_f1": 0.95,
        "roc_auc": 0.9942,
    }
    for metric, target in expected.items():
        if not np.isclose(metrics[metric], target, atol=5e-5):
            raise AssertionError(
                f"Exp01 metric mismatch for {metric}: "
                f"got {metrics[metric]:.8f}, expected {target:.8f}"
            )

    matrix = confusion_matrix(y_test, y_prediction, labels=[0, 1])
    if not np.array_equal(matrix, np.array([[72, 0], [4, 38]])):
        raise AssertionError(f"Exp01 confusion matrix mismatch: {matrix.tolist()}")

    # Compute malignant-class (class 1) SHAP values once for the shared test set.
    tree_explainer = shap.TreeExplainer(model)
    all_class_shap = tree_explainer(X_test)
    malignant_shap = all_class_shap[:, :, 1]
    reconstructed_probability = (
        np.asarray(malignant_shap.base_values).reshape(-1)
        + malignant_shap.values.sum(axis=1)
    )
    np.testing.assert_allclose(
        reconstructed_probability,
        y_probability,
        atol=1e-5,
        err_msg="Class-1 SHAP outputs do not reconstruct malignant probabilities.",
    )

    lime_explainer = LimeTabularExplainer(
        training_data=X_train.to_numpy(dtype=float),
        feature_names=list(X.columns),
        class_names=["Benign", "Malignant"],
        mode="classification",
        discretize_continuous=False,
        feature_selection="none",
        random_state=RANDOM_STATE,
    )

    return ExperimentContext(
        dataset=dataset,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        model=model,
        y_probability=y_probability,
        y_prediction=y_prediction,
        malignant_shap=malignant_shap,
        lime_explainer=lime_explainer,
    )


def save_metadata(context: ExperimentContext) -> None:
    """Save configuration and library versions for reproducibility."""
    metadata = {
        "dataset": "Wisconsin Diagnostic Breast Cancer (scikit-learn)",
        "samples": int(context.dataset.data.shape[0]),
        "features": int(context.dataset.data.shape[1]),
        "label_encoding": "(dataset.target == 0).astype(int)",
        "class_names": {"0": "Benign", "1": "Malignant"},
        "split": {
            "test_size": 0.2,
            "random_state": RANDOM_STATE,
            "stratified": True,
            "train_samples": int(len(context.X_train)),
            "test_samples": int(len(context.X_test)),
        },
        "model": {
            "class": "RandomForestClassifier",
            "n_estimators": 200,
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
        },
        "threshold": THRESHOLD,
        "metrics": {
            "accuracy": accuracy_score(context.y_test, context.y_prediction),
            "malignant_precision": precision_score(
                context.y_test, context.y_prediction, zero_division=0
            ),
            "malignant_recall": recall_score(
                context.y_test, context.y_prediction, zero_division=0
            ),
            "malignant_f1": f1_score(
                context.y_test, context.y_prediction, zero_division=0
            ),
            "roc_auc": roc_auc_score(context.y_test, context.y_probability),
            "confusion_matrix_labels_0_1": confusion_matrix(
                context.y_test, context.y_prediction, labels=[0, 1]
            ).tolist(),
        },
        "shap_version": shap.__version__,
        "lime_version": version("lime"),
        "scikit_learn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "pandas_version": pd.__version__,
        "matplotlib_version": matplotlib.__version__,
        "lime": {
            "random_state": RANDOM_STATE,
            "num_samples_per_explanation": LIME_NUM_SAMPLES,
            "num_features": len(context.feature_names),
            "feature_selection": "none",
            "discretize_continuous": False,
            "explained_class": 1,
        },
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print("Baseline metrics:", json.dumps(metadata["metrics"], indent=2))


def main() -> None:
    context = build_context()
    save_metadata(context)

    # LIME explanations for the complete test set are computed once and cached.
    run_global_comparison(context)
    run_local_comparison(context)
    run_fn_comparison(context)
    print(f"Exp03 completed. Results saved under: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
