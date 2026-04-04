#  House Price Prediction — End-to-End Machine Learning Project

## Overview

This project focuses on building an end-to-end machine learning pipeline to predict house prices using advanced regression techniques. The goal was not only to achieve high predictive accuracy but also to extract meaningful business insights from the data.

---

## Project Workflow

### 1. Exploratory Data Analysis (EDA)
- Identified missing values and skewed distributions
- Observed strong right-skew in target variable (SalePrice)
- Detected relationships between price and key features

---

### 2. Data Cleaning & Preprocessing
- Handled missing values:
  - Numerical → median / 0
  - Categorical → mode
- Feature transformations:
  - Log transformation on target variable (`log1p`)
- Converted ordinal categorical features using domain-based mapping
- Applied One-Hot Encoding for nominal features

---

### 3. Feature Engineering
- Combined related features (e.g., total area)
- Converted quality-based categorical variables into numerical scales
- Reduced redundancy in highly correlated features

---

### 4️. Modeling Approach

Multiple models were trained and compared:

#### 🔹 Linear Models
- Linear Regression
- Ridge Regression
- Lasso Regression
- Elastic Net

#### 🔹 Distance-Based Model
- KNN Regressor

#### 🔹 Tree-Based Models
- Decision Tree Regressor
- Random Forest Regressor

#### 🔹 Boosting Models
- Gradient Boosting
- XGBoost
- CatBoost

---

### 5. Model Evaluation Metrics

- R² Score
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)

---

## Final Model Comparison

| Model | Test R² |
|------|--------|
| CatBoost | **0.903** |
| XGBoost | 0.897 |
| Ridge | 0.892 |
| Elastic Net | 0.891 |
| Lasso | 0.890 |
| Gradient Boosting | 0.887 |
| Random Forest | 0.883 |
| Decision Tree | 0.79 |
| KNN | 0.78 |

---

##  Key Insights

### 1. Quality is the strongest driver
- **OverallQual** had the highest impact on price
- Buyers prioritize construction quality over size

---

### 2. Living space matters
- **GrLivArea, TotalBsmtSF**
- Larger usable area → higher price

---

### 3. Garage features are critical
- Garage capacity, area, and quality significantly influence price

---

### 4. Interior quality impacts value
- Kitchen quality, basement quality, fireplace condition
- Interior upgrades strongly affect pricing

---

### 5. Comfort features add premium
- Central Air Conditioning
- Fireplaces

---

### 6. Location plays a role
- Neighborhood and zoning features contribute to pricing

---

## Model Insights

- **CatBoost performed best** due to handling complex interactions
- **XGBoost showed similar strong performance**
- **Linear models performed surprisingly well**, indicating strong linear relationships after preprocessing
- **KNN and Decision Tree underperformed** due to:
  - High dimensionality (KNN)
  - Overfitting (Decision Tree)

---

## Key Learning

> Effective feature engineering and preprocessing can make simple models perform nearly as well as complex models.

---

## Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- CatBoost
- Matplotlib

---

## Conclusion

Boosting models like CatBoost and XGBoost provided the best performance by capturing non-linear relationships and feature interactions. However, strong results from linear models highlight the importance of high-quality data preprocessing.

---

## Future Improvements

- Hyperparameter tuning with Bayesian Optimization
- Feature selection using SHAP values
- Deployment using Flask / Streamlit
- Model explainability dashboards

---

## Author

Ayesha Firdaus Honnur

---
