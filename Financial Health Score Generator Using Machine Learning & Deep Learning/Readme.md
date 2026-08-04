# Financial Health Score Generator

An end-to-end machine learning pipeline that scores a person's financial health (0–100) from their income, spending, savings, debt, and credit data — and turns that score into personalized financial recommendations.

## Overview

The notebook (`mpl2.ipynb`) takes raw personal finance records, engineers a set of financial ratio features, constructs a custom `financial_health_score` target, explores the data visually, trains and compares several regression models, selects and refines the best one, and packages everything needed to deploy it.

## Workflow

1. **Setup & Data Loading** — Import libraries and load `personal_finance_dataset.csv`.
2. **Data Inspection & Cleaning** — Shape, types, summary stats, missing values, duplicates; convert `date` to datetime.
3. **Feature Engineering** — Derive financial ratios and build the target score.
4. **Exploratory Data Analysis (EDA)** — Distributions, category counts, correlation heatmap, target/feature distributions, outlier checks, and relationship plots.
5. **Encoding & Train/Test Preparation** — Label-encode categoricals, drop leakage-prone columns, split, scale, check multicollinearity (VIF).
6. **Model Training & Comparison** — Train and evaluate 7 regression models.
7. **Best Model Selection & Feature Selection (RFE)** — Pick the top model and reduce to 6 key features.
8. **Retraining on Selected Features** — Refit the best model on the reduced feature set.
9. **Recommendation Engine** — Map a score to tiered, plain-language financial advice.
10. **Save Model Artifacts** — Persist the model, encoders, feature list, and scaler for deployment.

## Dataset

Input: `personal_finance_dataset.csv` (not included — place it in the same directory as the notebook).

Expected raw columns include (among others): `monthly_income`, `monthly_expense_total`, `actual_savings`, `investment_amount`, `loan_payment`, `emergency_fund`, `credit_score`, `savings_rate`, `debt_to_income_ratio`, `discretionary_spending`, `cash_flow_status`, `financial_stress_level`, `savings_goal_met`, `fraud_flag`, `financial_scenario`, `income_type`, `category`, `date`, `user_id`, `financial_advice_score`.

## Feature Engineering

| Feature | Formula |
|---|---|
| `expense_ratio` | `monthly_expense_total / monthly_income` |
| `saving ratio` | `actual_savings / monthly_income` |
| `investment Ratio` | `investment_amount / monthly_income` |
| `debt burden` | `loan_payment / monthly_expense_total` |
| `spending_gap` | `monthly_income - monthly_expense_total` |
| `emergency_fund_coverage` | `emergency_fund / monthly_expense_total` |

Ratio columns have infinite values replaced with NaN, are clipped at the 99th percentile, and remaining NaNs filled with 0.

## Target Variable: `financial_health_score`

A custom 0–100 score built as a weighted blend of normalized signals:

| Component | Weight |
|---|---|
| Credit score | 0.18 |
| Savings rate | 0.16 |
| Debt-to-income (inverted) | 0.14 |
| Emergency fund coverage | 0.14 |
| Discretionary spending (inverted) | 0.08 |
| Investment ratio | 0.06 |
| Loan/debt burden (inverted) | 0.06 |
| Cash flow status | 0.08 |
| Financial stress level | 0.06 |
| Savings goal met | 0.04 |

A penalty of 10 points is subtracted if `fraud_flag` is set, and the final score is clipped to `[0, 100]`.

## Exploratory Data Analysis

This version of the notebook includes an expanded EDA section beyond the basic distributions and correlation heatmap:

- **Target distribution** — histogram + KDE of `financial_health_score`.
- **Engineered ratio distributions** — grid of histograms for all five ratio features (`expense_ratio`, `saving ratio`, `investment Ratio`, `debt burden`, `emergency_fund_coverage`).
- **Outlier checks** — boxplots of the core raw numeric columns (`monthly_income`, `monthly_expense_total`, `actual_savings`, `loan_payment`).
- **Income vs. savings scatter** — `monthly_income` vs. `actual_savings`, colored by `cash_flow_status`.
- **Score by scenario** — boxplot of `financial_health_score` across `financial_scenario` categories.
- **Pairwise relationships** — pairplot of the four core ratio features, colored by `cash_flow_status`.

## Models Trained & Compared

| Model | Notes |
|---|---|
| Linear Regression | Baseline, with 5-fold CV |
| Ridge Regression | `alpha=0.01`, with 5-fold CV |
| Decision Tree Regressor | Tuned via `GridSearchCV` (depth, split, leaf) |
| Random Forest Regressor | Tuned via `GridSearchCV`; `n_estimators=300`, `max_depth=8` |
| Gradient Boosting Regressor | `n_estimators=300`, `max_depth=3`, `learning_rate=0.05` |
| K-Nearest Neighbors | Best `k` selected by scanning RMSE across `k=1..30` |
| Artificial Neural Network (Keras) | Dense 128→64→32→1, Adam optimizer, MSE loss, 50 epochs |

Each model is evaluated with **MAE**, **RMSE**, and **R²**, and results are ranked to pick the best performer.

## Feature Selection

Recursive Feature Elimination (RFE) is applied using the best-performing model to narrow the full feature set down to the **6 most predictive features**. The model is then retrained on this reduced set (final model: **Gradient Boosting Regressor**).

## Recommendation Engine

`generate_recommendations(score)` maps the predicted score into one of four tiers, each with tailored advice:

- **70–100 — Excellent:** maintain habits, diversify investments, increase long-term investing, review insurance/retirement.
- **50–69 — Good:** increase savings, build a 6-month emergency fund, cut unnecessary spending, boost investment contributions.
- **30–49 — Moderate:** prioritize high-interest debt payoff, build/follow a budget, raise savings rate, avoid new loans.
- **0–29 — Poor:** cover essentials only, strict budget, build an emergency fund incrementally, reduce debt quickly, consider professional advice.

## Deployment Artifacts

The final cells save everything needed to run inference without retraining:

| File | Contents |
|---|---|
| `model_5.pkl` | Final trained Gradient Boosting model |
| `encoders.pkl` | Fitted `LabelEncoder`s for categorical columns |
| `features.pkl` | List of the 6 RFE-selected feature names |
| `scaler.pkl` | Fitted `StandardScaler` for the selected features |

All saved with `joblib.dump(...)`.

## Requirements

- Python 3.x
- `numpy`, `pandas`
- `matplotlib`, `seaborn`
- `scikit-learn`
- `tensorflow` (for the Keras ANN)
- `statsmodels` (for VIF)
- `joblib`

## Notes

- Identifier and leakage-prone columns (`date`, `user_id`, `financial_advice_score`, `financial_health_score`, `monthly_income`, `monthly_expense_total`, `spending_gap`) are dropped before training so the model can't "cheat" using the raw inputs used to construct the target.
- VIF is computed to flag multicollinearity among numeric features before modeling.
- Train/test split is 70/30 with `random_state=42` for reproducibility.
- Compared to a simpler version of this pipeline, this notebook adds a richer EDA pass (target distribution, ratio-feature distributions, outlier boxplots, scatter and pairplot relationship views) before moving into encoding and modeling.
