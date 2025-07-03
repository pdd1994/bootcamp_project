# Summere-bootcamp


##  Problem Description
The goal of this project is to assess the **risk of improper police behavior** during interactions with the public. Given the **extremely imbalanced nature** of the dataset—where most interactions are appropriate—we focus not on hard classification accuracy, but rather on **calibrated predicted probabilities** to quantify risk. Well-calibrated models are crucial in high-stakes applications like this, where confidence matters more than classification.

##  Data Overview
We used a subset of the [ICPSR 38872 dataset](https://www.icpsr.umich.edu/web/NACJD/studies/38872), focusing on the following features:
- **AGE** (Categorical)
- **HISP** (Hispanic status)
- **SEX**
- **INCOME**
- **V347** (Target: Indicator of proper police behavior)

Data preprocessing included:
- Removing missing values
- Encoding categorical variables using `pd.get_dummies`
- Target: `proper_behave = V347`

Final dataset size after cleaning: **17,535 records**

##  Modeling Approach
We used **three classifiers** and focused on **probability calibration**:

1. **Logistic Regression**  
   - Baseline model  
   - Naturally well-calibrated under linear assumptions
   - SMOTE processing for imbalanced data set

2. **AdaBoostClassifier**  
   - Robust ensemble method  
   - Known for overconfident probabilities → used uncalibrated

3. **XGBoost (Calibrated)**  
   - Powerful gradient-boosting model  
   - Calibrated using **Platt Scaling** (`sigmoid`) via `CalibratedClassifierCV`

### Train/Test Split
- 80% training / 20% test using `train_test_split(random_state=42)`
- All models trained and tested on the **same split**

##  Key Results

### 1. ROC Curve


| Model              | ROC AUC |
|--------------------|---------|
| Logistic Regression| 0.56    |
| Logistic Regression with SMTOE| 0.54    |
| AdaBoost           | 0.54    |
| Calibrated XGBoost | 0.54    |

---

### 2. Precision–Recall Curve


All models achieved an **Average Precision (AP)** around **0.95**, indicating excellent performance on the dominant class, though less informative for the minority class.

---

### 3. Calibration Curve
Logistic Regression was closest to the **perfect calibration line**, followed by **Calibrated XGBoost**. AdaBoost showed poorly calibrated confidence (sharp spike at 0.5).

---

### 4. Histogram of Predicted Probabilities


- Logistic & Calibrated XGBoost produced high-confidence predictions clustered near **0.95+**
- AdaBoost was overconfident around **0.5**, highlighting calibration concerns

---

##  Summary of Implications
- **Logistic Regression** provides a strong calibrated baseline and is suitable for **interpretable risk scoring**
- **Logistic Regression with SMOTE** has a lower predicted probabilities and are more spread out. This suggests the model is less certain
- **AdaBoost** improves robustness but requires careful **post-hoc calibration** to be useful for probability interpretation
- **XGBoost**, although powerful, outputs **overconfident predictions** unless calibrated. With **Platt scaling**, its performance aligns with Logistic Regression.

In real-world use cases like policing oversight or legal accountability, **well-calibrated probabilities are critical** for fair decision-making and transparency.
