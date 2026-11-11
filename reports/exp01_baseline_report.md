# Experiment 01 — Baseline Model Training and SHAP Explanation

**Project:** MedXAI-Breast  
**Research Topic:** Explainable Breast Cancer Classification Using Random Forest and SHAP  
**Dataset:** Wisconsin Diagnostic Breast Cancer (WDBC)  
**Model:** Random Forest Classifier  
**Explainability Method:** SHAP (SHapley Additive exPlanations)

---

## 1. Introduction

Machine learning models have demonstrated considerable potential in medical classification tasks. However, high predictive performance alone does not necessarily provide sufficient insight into how a model reaches its decisions. In medical applications, understanding the contribution of individual features to model predictions is particularly important for improving model transparency and supporting further investigation of model behavior.

This experiment establishes a baseline breast cancer classification model using the Random Forest algorithm and applies SHAP to interpret its predictions.

The primary objectives are:

1. Develop a baseline Random Forest classifier using the Wisconsin Diagnostic Breast Cancer dataset.
2. Evaluate the predictive performance using standard classification metrics.
3. Investigate global feature importance using SHAP summary plots.
4. Examine individual predictions through local SHAP explanations.
5. Establish a reproducible experimental baseline for subsequent false negative analysis and comparisons between explainability methods.

## 2. Dataset Description

### 2.1 Dataset Source

The experiment uses the Wisconsin Diagnostic Breast Cancer (WDBC) dataset provided by the `sklearn.datasets` module.

The dataset contains 569 samples and 30 numerical features extracted from digitized images of fine needle aspirates (FNA) of breast masses.

Each sample is assigned one of two diagnostic labels:

- Benign (B): Non-malignant tumor.
- Malignant (M): Malignant tumor.

The original dataset contains 357 benign samples and 212 malignant samples.

### 2.2 Feature Description

The 30 numerical features describe characteristics of cell nuclei, including:

- Radius
- Texture
- Perimeter
- Area
- Smoothness
- Compactness
- Concavity
- Concave points
- Symmetry
- Fractal dimension

These measurements are represented through three statistical groups:

- Mean: Average feature measurements.
- Standard Error: Measurement variability.
- Worst: The mean of the three largest measurements for each feature.

### Figure 1 — Dataset Overview

**[INSERT FIGURE 1 HERE: Dataset class distribution]**

*Suggested filename: `dataset_distribution.png`*

*Caption: Figure 1. Class distribution of the Wisconsin Diagnostic Breast Cancer dataset, containing 357 benign and 212 malignant samples.*

The dataset contains more benign than malignant samples. Stratified sampling was therefore applied during dataset splitting to preserve the original class proportions in both training and testing subsets.

## 3. Experimental Methodology

### 3.1 Data Preprocessing

The original dataset labels were encoded as follows:

- 0: Benign
- 1: Malignant

This encoding was implemented by converting the original target labels, where 0 represents malignant and 1 represents benign, into a binary malignant-class indicator.

No additional feature selection or dimensionality reduction was performed.

### 3.2 Train-Test Split

The dataset was divided into training and testing subsets using stratified random sampling.

Experimental configuration:

| Parameter | Value |
|---|---|
| Total samples | 569 |
| Number of features | 30 |
| Training proportion | 80% |
| Testing proportion | 20% |
| Random state | 42 |
| Stratification | Enabled |
| Training samples | 455 |
| Testing samples | 114 |

Stratification ensures that the proportions of benign and malignant samples are approximately maintained across both subsets.

### 3.3 Random Forest Classifier

A Random Forest classifier was selected as the baseline predictive model.

Random Forest is an ensemble learning algorithm that combines predictions from multiple decision trees. Its ensemble structure allows it to model nonlinear relationships and interactions between input features.

The model configuration was:

| Hyperparameter | Value |
|---|---|
| Algorithm | Random Forest |
| Number of estimators | 200 |
| Random state | 42 |
| Number of parallel jobs | -1 |
| Other parameters | Scikit-learn defaults |

The model was trained using the training subset and evaluated on the held-out test subset.

### 3.4 Model Evaluation Metrics

The following evaluation metrics were used:

**Accuracy**

Measures the proportion of correctly classified samples.

**Precision**

Measures the proportion of correctly predicted positive samples among all positive predictions.

**Recall (Sensitivity)**

Measures the proportion of actual malignant samples correctly identified by the model.

**F1-score**

Represents the harmonic mean of precision and recall.

**ROC-AUC**

Measures the model's ability to distinguish between benign and malignant samples across different classification thresholds.

The malignant class was treated as the positive class throughout the experiment.

### 3.5 SHAP-Based Explainability

SHAP was employed as a post-hoc explainability method to investigate the contribution of individual input features to the trained Random Forest model's predictions.

The TreeExplainer implementation was selected because it supports tree-based machine learning models.

Two main types of explanations were generated:

**Global Explanation — SHAP Beeswarm Plot**

The Beeswarm plot summarizes feature contributions across multiple samples. It provides information about:

- The overall distribution of SHAP values.
- The relative importance of individual features.
- The relationship between feature values and their contributions to model output.

**[INSERT FIGURE 2 HERE: SHAP Beeswarm Plot]**

*Suggested filename: `shap_beeswarm.png`*

*Caption: Figure 2. SHAP Beeswarm plot illustrating the global distribution of feature contributions to the Random Forest model's malignant-class predictions.*

**Local Explanation — SHAP Waterfall Plot**

The Waterfall plot explains an individual prediction by decomposing the model output into a baseline value and individual feature contributions.

Positive SHAP values increase the model output for the malignant class, whereas negative SHAP values decrease it.

**[INSERT FIGURE 3 HERE: SHAP Waterfall Plot]**

*Suggested filename: `shap_waterfall.png`*

*Caption: Figure 3. SHAP Waterfall plot showing the contribution of individual features to the malignant-class prediction for a selected test sample.*

## 4. Experimental Results

### 4.1 Classification Performance

The Random Forest classifier achieved the following results on the held-out test set.

**Table 1. Classification Report**

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Benign | 0.95 | 1.00 | 0.97 | 72 |
| Malignant | 1.00 | 0.90 | 0.95 | 42 |
| Macro Average | 0.97 | 0.95 | 0.96 | 114 |
| Weighted Average | 0.97 | 0.96 | 0.96 | 114 |

Overall classification accuracy was 96.49%, while the ROC-AUC reached 0.9942.

These results indicate that the baseline model achieved strong discrimination performance on the held-out test set.

The model correctly identified all 72 benign samples. However, four malignant samples were incorrectly classified as benign.

### 4.2 Confusion Matrix Analysis

**[INSERT FIGURE 4 HERE: Confusion Matrix]**

*Suggested filename: `confusion_matrix.png`*

*Caption: Figure 4. Confusion matrix of the Random Forest classifier on the held-out test set.*

The confusion matrix is presented below:

| Actual / Predicted | Benign | Malignant |
|---|---:|---:|
| Benign | 72 | 0 |
| Malignant | 4 | 38 |

The model correctly classified 110 out of 114 samples.

Although the overall accuracy was high, the four false negative predictions represent an important limitation.

In medical classification scenarios, false negative predictions deserve particular attention because malignant cases may be incorrectly classified as benign.

This observation motivates the subsequent false negative analysis in Experiment 02.

### 4.3 Global SHAP Explanation

The SHAP Beeswarm plot identified several influential features in the model's malignant-class predictions.

The most prominent features included:

1. Worst perimeter
2. Worst area
3. Worst concave points
4. Mean concave points
5. Worst radius
6. Mean radius
7. Mean area
8. Mean concavity
9. Mean perimeter

The Beeswarm visualization also indicated that higher values of several influential features generally contributed positively to the model's malignant-class output.

These findings suggest that measurements associated with cell nucleus size and shape played an important role in the model's predictions.

However, SHAP values describe contributions to the model's learned predictions, rather than establishing biological or clinical causality.

### 4.4 Local SHAP Explanation

The SHAP Waterfall plot was used to investigate an individual test sample.

The selected sample had the following model outputs:

| Parameter | Value |
|---|---:|
| Expected model output | 0.374 |
| Individual prediction output | 0.015 |

The baseline output was shifted by the combined contributions of the 30 input features, resulting in a final model output of 0.015 for the malignant class.

Several features, including worst perimeter, worst area and worst concave points, contributed negatively to the malignant-class output for this particular sample.

This illustrates how SHAP provides a sample-specific explanation by showing the direction and magnitude of individual feature contributions.

The explanation should be interpreted as an explanation of the model's prediction, not as a direct clinical assessment of the patient's disease status.

## 5. Discussion

The baseline experiment demonstrates that a Random Forest classifier can achieve strong predictive performance on the Wisconsin Diagnostic Breast Cancer dataset.

The ROC-AUC of 0.9942 indicates that the model has strong discrimination ability on the evaluated test set.

However, the presence of four false negative predictions highlights why accuracy alone is insufficient for evaluating medical classification systems.

The SHAP analysis provides additional insight into the model's prediction behavior.

The global Beeswarm plot identifies influential features across the evaluated samples, while the local Waterfall plot explains how individual feature contributions combine to produce a specific prediction.

An important distinction is that global feature importance and local feature contributions describe different aspects of model behavior. A feature that is highly influential overall does not necessarily have a large contribution to every individual prediction.

Furthermore, SHAP explanations do not establish causal relationships between medical measurements and malignancy.

These findings provide a basis for further investigating model errors, comparing different explainability methods and evaluating explanation reliability.

## 6. Limitations

Several limitations should be considered:

1. **Dataset limitation:** The experiment uses a single publicly available benchmark dataset rather than independent clinical datasets from multiple hospitals.

2. **Evaluation limitation:** The model was evaluated using one stratified train-test split, so the results may vary under different random partitions.

3. **Model limitation:** Only Random Forest was evaluated in this experiment. Comparisons with other machine learning architectures are reserved for subsequent experiments.

4. **Explainability limitation:** SHAP provides model-specific feature attribution but does not establish causal relationships or guarantee clinical interpretability.

5. **Clinical applicability:** The experiment is an educational and methodological investigation, not a clinically validated diagnostic system.

## 7. Conclusion

Experiment 01 established a baseline breast cancer classification model using Random Forest and investigated its predictive behavior through SHAP.

The model achieved 96.49% accuracy and an ROC-AUC of 0.9942 on the held-out test set.

SHAP analysis identified influential features associated with malignant-class predictions and demonstrated how individual feature contributions influence specific model outputs.

Nevertheless, four malignant samples were incorrectly classified as benign, highlighting the importance of investigating false negative predictions beyond conventional performance metrics.

The experimental findings establish a reproducible baseline for subsequent investigations.

Experiment 02 extends this work through false negative analysis and classification threshold evaluation, while Experiment 03 will compare SHAP and LIME as alternative post-hoc explainability methods.

---

## References

1. Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions. *Advances in Neural Information Processing Systems (NeurIPS)*.

2. Breiman, L. (2001). Random Forests. *Machine Learning*, 45, 5–32.

3. Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why Should I Trust You?": Explaining the Predictions of Any Classifier. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*.

4. Scikit-learn Developers. *Breast Cancer Wisconsin (Diagnostic) Dataset*. Scikit-learn documentation.

5. SHAP Developers. *SHAP Documentation*. https://shap.readthedocs.io/