from pathlib import Path
import numpy as np
import shap
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, ConfusionMatrixDisplay

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# 1. 加载数据
dataset = load_breast_cancer(as_frame=True)
X = dataset.data
y = (dataset.target == 0).astype(int) # 1 = 恶性，0 = 良性

print("Dataset shape:", X.shape)
print("Malignant samples:", y.sum())

# 确保 results 目录存在
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ====================================================
# 新增图 1：数据集类别分布图 (Dataset Distribution)
# ====================================================
plt.figure(figsize=(6, 4))
y.value_counts().plot(kind='bar', color=['#66b3ff', '#ff9999'])
plt.title('Dataset Distribution (0 = Benign, 1 = Malignant)')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks([0, 1], ['Benign (0)', 'Malignant (1)'], rotation=0)
plt.savefig(RESULTS_DIR / "dataset_distribution.png", dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved: {RESULTS_DIR / 'dataset_distribution.png'}")
# ====================================================

# 2. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y,
)

# 3. 训练随机森林
model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# 4. 评估模型（默认 0.5 阈值）
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\n--- Exp01: Baseline Results (Threshold=0.5) ---")
print(classification_report(y_test, y_pred, target_names=["Benign", "Malignant"]))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ====================================================
# 新增图 2：混淆矩阵可视化 (Confusion Matrix)
# ====================================================
fig, ax = plt.subplots(figsize=(6, 6))
ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    display_labels=["Benign", "Malignant"],
    cmap=plt.cm.Blues,
    ax=ax
)
plt.title("Confusion Matrix (Threshold = 0.5)")
plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved: {RESULTS_DIR / 'confusion_matrix.png'}")
# ====================================================

# 5. SHAP 解释
explainer = shap.TreeExplainer(model)
shap_values = explainer(X_test)
malignant_shap = shap_values[:, :, 1] # class 1 = 恶性

# 检验 SHAP 贡献加和是否等于模型输出
reconstructed = malignant_shap.base_values + malignant_shap.values.sum(axis=1)
np.testing.assert_allclose(reconstructed, y_prob, atol=1e-5)

# 6. 保存全局解释图
shap.plots.beeswarm(malignant_shap, show=False)
plt.savefig(RESULTS_DIR / "shap_beeswarm.png", dpi=200, bbox_inches="tight")
plt.close()

# 7. 保存单样本解释图
shap.plots.waterfall(malignant_shap[0], show=False)
plt.savefig(RESULTS_DIR / "shap_waterfall.png", dpi=200, bbox_inches="tight")
plt.close()

# 8. 分析假阴性样本 (基于默认的 0.5 阈值)
false_negative_indices = np.where(
    (y_test.to_numpy() == 1) & (y_pred == 0)
)[0]

print("\nFalse Negative Analysis:")
for idx in false_negative_indices:
    print(f"Test index: {idx}, Malignant probability: {y_prob[idx]:.4f}")

    shap.plots.waterfall(malignant_shap[idx], show=False)
    plt.savefig(
        RESULTS_DIR / f"false_negative_{idx}.png",
        dpi=200,
        bbox_inches="tight"
    )
    plt.close()

print("\nExp01 Completed! All figures saved in exp01_baseline/results/")
