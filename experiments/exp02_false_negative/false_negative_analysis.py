from pathlib import Path
import numpy as np
import shap
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# 1. 加载数据与训练模型 (必须重新加载一次，保持实验独立)
dataset = load_breast_cancer(as_frame=True)
X = dataset.data
y = (dataset.target == 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y,
)

model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

y_prob = model.predict_proba(X_test)[:, 1]

# ====================================================
# 核心实验：调整阈值 (从 0.5 降至 0.4 或 0.35，以降低假阴性)
# ====================================================
threshold = 0.4
y_pred_adjusted = (y_prob >= threshold).astype(int)

print(f"\n--- Exp02: Threshold Adjusted to {threshold} ---")
print(classification_report(y_test, y_pred_adjusted, target_names=["Benign", "Malignant"]))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_adjusted))

# 2. 分析假阴性样本 (使用调整后的预测)
false_negative_indices = np.where((y_test.to_numpy() == 1) & (y_pred_adjusted == 0))[0]

print(f"\nFalse Negative Analysis (Remaining FNs: {len(false_negative_indices)}):")

# 3. 为假阴性样本生成 SHAP 解释
explainer = shap.TreeExplainer(model)
shap_values = explainer(X_test)
malignant_shap = shap_values[:, :, 1]

# 保存到 exp02 自己的 results 目录下
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

for idx in false_negative_indices:
    print(f"Test index: {idx}, Malignant probability: {y_prob[idx]:.4f}")

    shap.plots.waterfall(malignant_shap[idx], show=False)
    plt.savefig(RESULTS_DIR / f"false_negative_{idx}.png", dpi=200, bbox_inches="tight")
    plt.close()

print("\nExp02 Completed! FN Figures saved in exp02_false_negative/results/")
