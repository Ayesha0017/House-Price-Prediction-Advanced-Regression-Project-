# House Price Prediction (Advanced Regression Project)

## Overview

This project focuses on predicting house prices using advanced regression techniques on the Ames Housing dataset (Kaggle). The goal was to build a **robust, production-ready regression pipeline** while addressing real-world challenges like missing values, multicollinearity, and high-dimensional data.

---

## Objectives

* Build a high-performance regression model
* Handle missing values effectively
* Address multicollinearity
* Apply feature engineering
* Compare regularization techniques (Ridge vs Lasso)

---

## Dataset

* Source: Kaggle (Ames Housing Dataset)
* ~1460 rows
* 80+ original features → expanded to **200+ features after encoding**

---

## Data Preprocessing

### Missing Value Handling

* **Categorical (NA = absence):** Filled with `"None"`
* **Numerical (absence-related):** Filled with `0`
* **LotFrontage:** Filled with median
* **Low-missing categorical:** Filled with mode

---

### Feature Engineering

* `HouseAge = YrSold - YearBuilt`
* `RemodelAge = YrSold - YearRemodAdd`
* `TotalBaths = FullBath + 0.5*HalfBath + BsmtFullBath + 0.5*BsmtHalfBath`

---

### Encoding

* **Ordinal Encoding:** Applied based on domain knowledge (quality, condition, etc.)
* **One-Hot Encoding:** Applied to nominal categorical variables

---

## Multicollinearity Handling

* Initially evaluated using **VIF (Variance Inflation Factor)**
* Highly correlated features were identified (e.g., basement & floor area breakdowns)
* Some redundant features were removed

 However:

* Final model uses **Ridge Regularization**, which inherently handles multicollinearity
* Therefore, aggressive feature dropping was avoided to preserve signal

---

## Target Transformation

```python
y = log(1 + SalePrice)
```

✔ Helps handle skewness
✔ Improves model performance

---

## Modeling Approach

### Linear Regression (Baseline)

* Observed overfitting
* Poor generalization

---

### Ridge Regression (Regularization)

```python
RidgeCV(alphas=[0.1, 1, 10, 50, 100], cv=5)
```

* Handles multicollinearity
* Shrinks coefficients
* Improves stability

---

### Lasso Regression (Feature Selection)

```python
LassoCV(cv=5)
```

* Performs automatic feature selection
* Reduces dimensionality

---

## Model Performance

| Model             | Train R² | Test R² | RMSE      |
| ----------------- | -------- | ------- | --------- |
| Linear Regression | ~0.92    | ~0.67   | High ❌    |
| Ridge Regression  | ~0.92    | ~0.89   | ~24,988 ✅ |
| Lasso Regression  | ~0.90    | ~0.89   | ~26,282 ✅ |

---

## Key Insights

* Regularization significantly improved generalization
* Ridge performed slightly better in terms of RMSE
* Lasso reduced feature count with minimal performance drop
* Many engineered and encoded features were redundant

---

## Ridge vs Lasso

| Aspect            | Ridge           | Lasso               |
| ----------------- | --------------- | ------------------- |
| Multicollinearity | Handles well    | Selects one feature |
| Feature Selection | ❌               | ✅                   |
| Performance       | Slightly better | Comparable          |
| Interpretability  | Lower           | Higher              |

---

## Final Conclusion

* **Ridge Regression** chosen for best predictive performance
* **Lasso Regression** useful for feature selection and interpretability

---

## Key Learnings

* Importance of handling multicollinearity
* Impact of regularization on model stability
* Feature engineering significantly boosts performance
* Trade-off between performance and interpretability

---

## Future Improvements

* Try **ElasticNet (Ridge + Lasso)**
* Apply **feature importance analysis**
* Use **tree-based models (XGBoost, LightGBM)**
* Hyperparameter tuning with GridSearchCV

---

## Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* Statsmodels

---

## Project Structure

```
├── data/
├── notebook.ipynb
├── README.md
```
## 📥 Dataset Access

Dataset is available on Kaggle:
👉 https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques
