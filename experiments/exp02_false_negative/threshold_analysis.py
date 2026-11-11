
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

# ====================================================
# 1. 项目路径设置
# ====================================================

# 当前文件位于：
# experiments/exp02_false_negative/threshold_analysis.py
# 因此 parents[2] 为项目根目录

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "exp02_false_negative"
    / "results"
)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ====================================================
# 2. 加载数据集
# ====================================================

dataset = load_breast_cancer(as_frame=True)

X = dataset.data

# 重新编码：
# 0 = Benign
# 1 = Malignant
y = (dataset.target == 0).astype(int)

print("Dataset shape:", X.shape)
print("Malignant samples:", y.sum())
print("Benign samples:", (y == 0).sum())

# ====================================================
# 3. 划分训练集和测试集
# ====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# ====================================================
# 4. 训练 Random Forest
# ====================================================

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train, y_train)

# 获取恶性类别（class 1）的预测概率
y_prob = model.predict_proba(X_test)[:, 1]

# ROC-AUC 不依赖分类阈值
roc_auc = roc_auc_score(y_test, y_prob)

print("\nModel ROC-AUC:", round(roc_auc, 4))

# ====================================================
# 5. 设置实验阈值
# ====================================================

thresholds = [
    0.30,
    0.35,
    0.40,
    0.45,
    0.48,
    0.50,
    0.55,
    0.60,
]

results = []

# ====================================================
# 6. 遍历不同分类阈值
# ====================================================

print("\n========== Threshold Analysis ==========")

for threshold in thresholds:

    # 根据当前阈值生成分类结果
    y_pred = (y_prob >= threshold).astype(int)

    # 混淆矩阵
    tn, fp, fn, tp = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1],
    ).ravel()

    # 计算评价指标
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

    # 保存结果
    results.append({
        "Threshold": threshold,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "Specificity": specificity,
        "F1": f1,
        "FNR": fnr,
        "ROC_AUC": roc_auc,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    })

    # 终端输出
    print(f"\nThreshold: {threshold:.2f}")
    print(f"Accuracy:    {accuracy:.4f}")
    print(f"Precision:   {precision:.4f}")
    print(f"Recall:      {recall:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"F1 Score:    {f1:.4f}")
    print(f"FNR:         {fnr:.4f}")

    print(f"TN: {tn} | FP: {fp}")
    print(f"FN: {fn} | TP: {tp}")

# ====================================================
# 7. 保存实验结果 CSV
# ====================================================

results_df = pd.DataFrame(results)

csv_path = RESULTS_DIR / "threshold_metrics.csv"

results_df.to_csv(
    csv_path,
    index=False,
    float_format="%.6f",
)

print("\n========== Results Summary ==========")

print(
    results_df[
        [
            "Threshold",
            "Recall",
            "Specificity",
            "Precision",
            "F1",
            "FN",
            "FP",
        ]
    ].to_string(index=False)
)

print(f"\nCSV saved to: {csv_path}")

# ====================================================
# 8. 绘制 Recall 和 Specificity 变化曲线
# ====================================================

plt.figure(figsize=(9, 6))

plt.plot(
    results_df["Threshold"],
    results_df["Recall"],
    marker="o",
    label="Recall (Sensitivity)",
)

plt.plot(
    results_df["Threshold"],
    results_df["Specificity"],
    marker="s",
    label="Specificity",
)

plt.plot(
    results_df["Threshold"],
    results_df["Precision"],
    marker="^",
    label="Precision",
)

plt.xlabel("Classification Threshold")
plt.ylabel("Score")
plt.title("Threshold Analysis: Classification Metrics")

plt.ylim(0, 1.05)
plt.xticks(thresholds)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "threshold_metrics.png",
    dpi=200,
    bbox_inches="tight",
)

plt.close()

# ====================================================
# 9. 绘制 FN 和 FP 数量变化
# ====================================================

plt.figure(figsize=(9, 6))

plt.plot(
    results_df["Threshold"],
    results_df["FN"],
    marker="o",
    label="False Negatives",
)

plt.plot(
    results_df["Threshold"],
    results_df["FP"],
    marker="s",
    label="False Positives",
)

plt.xlabel("Classification Threshold")
plt.ylabel("Number of Samples")
plt.title("False Negative vs False Positive Analysis")

plt.xticks(thresholds)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "fn_fp_comparison.png",
    dpi=200,
    bbox_inches="tight",
)

plt.close()

print("\nExp02.1 Threshold Analysis Completed!")

print("Generated files:")
print("- threshold_metrics.csv")
print("- threshold_metrics.png")
print("- fn_fp_comparison.png")
