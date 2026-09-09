# RetailMindAI
A Machine Learning Framework for Retail Demand Forecasting, Customer Segmentation, and Revenue Prediction

## Overview
RetailMindAI is an academic and portfolio machine learning project developed to address core analytical challenges in retail operations. Retail organizations frequently struggle with fragmented data, uncertain SKU demand, uncharacterized customer behavioral churn, and inaccurate transaction-level revenue estimates. 

Using historical transaction data from an Amazon retail dataset spanning 2020 through 2024, this project develops a structured, modular machine learning pipeline. It transitions from raw transactional data cleansing to feature engineering, time-series demand forecasting, unsupervised customer segmentation, and supervised sales revenue prediction, concluding with reproducible business intelligence signals.

---

## Problem Statement
Modern retail data systems capture massive volumes of transaction logs, but converting raw logs into actionable operational decisions presents distinct challenges:

1. **Transaction Sales Prediction:** Accurately predicting the monetary value of customer transactions before order completion to evaluate basket size, promotional responsiveness, and product pricing tiers.
2. **SKU Demand Understanding & Forecasting:** Modeling daily product demand across varying seasonality, trend cycles, and historical demand lags to guide inventory replenishment and minimize stockout or overstock situations.
3. **Customer Behavioral Segmentation:** Grouping heterogeneous customer bases by purchasing recency, frequency, monetary spend, and friction (returns/cancellations) to identify high-value cohorts and at-risk buyers.
4. **Converting Analytics to Actionable Signals:** Translating machine learning predictions and model outputs into measurable retail indicators?such as demand opportunity scores, volatility indices, and customer churn risk.

---

## Objectives
The primary technical and analytical goals of RetailMindAI are:
- Clean and validate raw transactional telemetry across 100,000 purchase records.
- Engineer time-series lag features, customer RFM summaries, and catalog-level metrics without lookahead leakage.
- Benchmark multiple regression algorithms on holdout test data to forecast SKU-level daily velocity.
- Segment customers using unsupervised clustering ($k$-Means) validated by Elbow and Silhouette methods.
- Train, evaluate, and benchmark regression models (Linear Regression, Random Forest, GBDT, XGBoost) to predict transaction sales (`TotalAmount`).
- Select the optimal predictive model using data-driven validation criteria.
- Formulate deterministic, mathematically reproducible retail opportunity, volatility, and churn risk metrics.

---

## Current Capabilities

### Implemented
- **Data Understanding & Exploratory Analysis (`01_data_understanding.ipynb`, `02_eda.ipynb`):** Complete schema audit, duplicate checks, missing value resolution, distribution histograms, and correlation heatmaps.
- **Feature Engineering (`03_feature_engineering.ipynb`):** Construction of customer RFM attributes, product catalog demand statistics, and a 91,250-row full-grid daily demand table with 1, 7, 14, and 28-day lag and rolling features.
- **Demand Forecasting (`04_demand_forecasting.ipynb`):** Chronological train/validation/test split evaluation comparing LightGBM, XGBoost, and Ridge regression for SKU-level daily quantity prediction.
- **Customer Segmentation (`05_customer_segmentation.ipynb`):** Standardized $k$-Means clustering on customer RFM features, selecting $k=4$ cohorts supported by Elbow inertia and Silhouette analysis, visualized via 2D PCA.
- **Sales & Revenue Prediction (`prediction/01_sales_prediction.ipynb`):** Supervised regression benchmark across 4 models on 20,000 holdout test transactions; pipeline export (`.pkl` and `.joblib`) and CSV export with actuals, predictions, and errors.
- **Business Signals & KPI Engineering (`06_business_signals_and_kpis.ipynb`):** Aggregation of 60 monthly retail KPI observations, SKU opportunity scores, demand volatility indices ($CV$), and customer churn risk scores.

### In Progress
- Additional diagnostic visualizations comparing cross-category elasticity.
- Modularization of notebook helper functions into reusable Python scripts under `src/`.

### Planned / Future Work
- **SHAP-based model interpretation:** Detailed Shapley additive explanation analysis.
- **Web Application / API:** REST API endpoints (e.g., FastAPI) and interactive UI dashboards (e.g., Streamlit).
- **Automated Retraining:** Continuous integration pipelines for periodic model refreshing.

---

## Machine Learning Pipeline

```text
                        Raw Retail Data (Amazon.csv)
                                     ?
                         Data Understanding & EDA
                                     ?
                            Feature Engineering
                   ?????????????????????????????????????
                   ?                 ?                 ?
          Demand Forecasting     Customer       Sales Prediction
             (LightGBM,         Segmentation       (OLS, RF,
            XGBoost, Ridge)      (k-Means)        GBDT, XGBoost)
                   ?                 ?                 ?
                   ?                 ?          Model Evaluation
                   ?                 ?                 ?
                   ?????????????????????????????????????
                                     ?
                             Feature Importance
                                     ?
                          Business Signals & KPIs
```

---

## Machine Learning Modules

### 1. Demand Forecasting
- **Objective:** Forecast the next-day quantity demanded (`DailyQuantity`) for each of the 50 catalog products.
- **Input:** Daily product demand dataset (`data/processed/daily_product_demand.csv`) containing 29 modeling features (lagged demand, rolling means, rolling standard deviations, day-of-week, month).
- **Main Processing:** Chronological split into Train (59,324 rows; 2020-03-31 to 2023-06-30), Validation (13,747 rows; 2023-07-01 to 2024-03-31), and Test (13,643 rows; 2024-04-01 to 2024-12-29).
- **Models / Algorithms:** LightGBM Regressor, XGBoost Regressor, Ridge Regression baseline.
- **Output:** Evaluation metrics (MAE, RMSE, MAPE) across validation and holdout test horizons, plus LightGBM feature importance plots.

### 2. Customer Segmentation
- **Objective:** Categorize customers into actionable behavioral cohorts based on historical order habits and transaction friction.
- **Input:** Customer feature summary (`data/processed/customer_features.csv`, 43,233 unique buyers).
- **Main Processing:** Standard scaling followed by $k$-Means clustering evaluated from $k=2$ to $k=8$; dimensionality reduction via 2D PCA for visual separation.
- **Model / Algorithm:** $k$-Means ($k=4$).
- **Output:** 
  - Segment 0: Inactive / At-Risk (low frequency $\mu=1.20$, high recency $\mu=944$ days; 38.1% of cohort)
  - Segment 1: Steady Regulars (moderate frequency $\mu=2.48$, tenure $\mu=807$ days; 39.8% of cohort)
  - Segment 2: Loyal Champions (highest spend $\mu=\$4,347$, frequent orders $\mu=4.26$; 19.1% of cohort)
  - Segment 3: High-Friction / Cancellers (elevated cancellation rate $\mu=66\%$; 3.0% of cohort)
  - Dataset: `data/processed/customer_segments.csv` and summary table `outputs/customer_segment_profile.csv`.

### 3. Sales & Revenue Prediction
- **Objective:** Predict transaction revenue (`TotalAmount`) at order inception.
- **Input:** Cleaned transactions merged with customer segmentation labels and product demand intelligence features (`X`: 21 raw features).
- **Main Processing:** `ColumnTransformer` with `StandardScaler` on numerical features and `OneHotEncoder(drop='first')` on categorical features; fixed 80/20 train/test split (`random_state=42`).
- **Models / Algorithms:** Linear Regression (OLS), Random Forest Regressor, Gradient Boosting Regressor (GBDT), XGBoost Regressor.
- **Output:** Serialized inference pipeline (`prediction/models/best_sales_prediction_model.pkl`), holdout predictions (`prediction/outputs/predictions.csv`), and performance comparison visuals.

### 4. Business Intelligence & Measurable Signals
- **Objective:** Transform model and historical telemetry into deterministic decision metrics.
- **Input:** Merged processed datasets.
- **Formulated Signals:**
  - **Demand Opportunity Score (0 - 100):** $100 \times (0.50 \times \text{Rank}(\text{Revenue})/N + 0.50 \times \text{Rank}(\text{Growth})/N)$
  - **Demand Volatility Index ($CV$):** $\text{DemandVolatility} / (\text{HistoricalAverageDemand} + 10^{-5})$
  - **Customer Churn Risk Score (0.0 - 1.0):** $\text{clip}(\text{Recency} / 365.0, 0.0, 1.0)$
  - **Customer Loyalty Index ($CLI$, 0 - 100):** Weighted percentile rank of Monetary ($40\%$), Frequency ($35\%$), and ActiveDays ($25\%$).
- **Output:** `monthly_product_demand.csv`, `monthly_business_kpis.csv`, `product_opportunity_risk.csv`, `customer_retention_risk.csv`.

---

## Model Evaluation

### Sales & Revenue Prediction Benchmark
Evaluated on the unseen holdout test set ($N_{\text{test}} = 20,000$ transactions) using a fixed 80/20 train/test split (`random_state=42`):

| Model | MAE (\$) | RMSE (\$) | $R^2$ Score | Training Time (s) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **XGBoost Regressor** | **\$262.65** | **\$377.40** | **0.7266** | **0.67s** | **Selected Best Model** |
| Gradient Boosting Regressor (GBDT) | \$263.12 | \$377.67 | 0.7262 | 118.14s | Evaluated |
| Random Forest Regressor | \$261.30 | \$379.60 | 0.7234 | 71.65s | Evaluated |
| Linear Regression (OLS Baseline) | \$330.27 | \$424.73 | 0.6537 | 0.32s | Evaluated |

**Model Selection Rationale:**
**XGBoost Regressor** was selected as the final model based on empirical criteria:
1. **Variance & Error Control:** Achieves the highest $R^2$ ($0.7266$) and lowest RMSE ($\$377.40$) across all candidates. While Random Forest achieves a marginally lower MAE ($\$261.30$ vs $\$262.65$, a $\$1.35$ difference), XGBoost significantly reduces large residual errors (RMSE).
2. **Computational Speed:** Completes 100-tree training in **$0.67\text{s}$**, compared to **$71.65\text{s}$** for Random Forest (~$100\times$ faster) and **$118.14\text{s}$** for Scikit-Learn GBDT (~$175\times$ faster).

### Demand Forecasting Benchmark
Evaluated on chronological holdout test data ($N_{\text{test}} = 13,643$ daily records, 2024-04-01 to 2024-12-29):

| Model | Test MAE | Test RMSE | Test MAPE | Validation MAE | Validation RMSE |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Ridge Regression** | **2.819** | **3.495** | **61.87%** | 2.808 | 3.486 |
| **LightGBM Regressor** | 2.821 | 3.494 | 61.99% | 2.808 | 3.485 |
| **XGBoost Regressor** | 2.821 | 3.494 | 62.00% | 2.808 | 3.484 |

*Note: All three models yield comparable error metrics on the standardized lag feature set; Ridge achieves the lowest test MAE by a marginal margin of 0.002 units.*

---

## Explainability

SHAP-based explainability is planned as part of the model interpretation stage.

Current model explainability is provided through tree-based split gain feature importances from the trained XGBoost sales predictor and LightGBM demand forecaster:
- **Top Sales Revenue Drivers (XGBoost Gain):**
  1. `UnitPrice` ($48.2\%$ relative gain): Catalog price tier is the primary anchor of basket value.
  2. `AverageOrderValue` ($28.9\%$): Customer past spending history strongly predicts transaction magnitude.
  3. `Monetary` ($3.3\%$): Cumulative customer lifetime spend.
  4. `Discount` ($2.9\%$): Promotional markdown percentage.
- **Top Demand Velocity Drivers (LightGBM):**
  1. `rolling_mean_28` and `rolling_mean_7`: Short and medium-term historical sales velocity.
  2. `lag_1` and `lag_7`: Recent purchase momentum and weekly seasonality.

---

## Dataset

- **Source:** Amazon Retail Transaction Dataset (`data/raw/Amazon.csv`).
- **Observation Period:** January 1, 2020 through December 29, 2024 ($1,825$ days).
- **Volume:** 100,000 raw transactional orders across 50 products and 43,233 unique customers.
- **Target Variables:**
  - `TotalAmount` (Continuous numeric variable in USD $): Target for transaction-level sales prediction.
  - `DailyQuantity` (Integer count): Target for SKU-level daily demand forecasting.
- **Key Feature Fields:**
  - Identifiers: `OrderID`, `CustomerID`, `ProductID`
  - Transaction Context: `OrderDate`, `Quantity`, `UnitPrice`, `Discount`, `Tax`, `ShippingCost`
  - Categoricals: `Category`, `Brand`, `PaymentMethod`, `OrderStatus`, `Country`, `City`
- **Preprocessing Applied:**
  - Datetime parsing and temporal decomposition (`Year`, `Month`, `DayOfWeek`, `IsWeekend`, `Quarter`).
  - Creation of binary status indicators: `is_returned`, `is_cancelled`, `is_delivered`, `is_shipped`, `is_pending`.
  - Exclusion of post-fulfillment target leakage variables (`Tax`, `ShippingCost`, `DiscountPct`) from predictive feature sets.
  - Non-negativity bounding on predicted revenue to respect physical retail transaction boundaries.

---

## Repository Structure

```text
RetailMindAI/
??? data/
?   ??? raw/
?   ?   ??? Amazon.csv                               # Source raw transaction log (100,000 rows)
?   ??? processed/
?       ??? retail_cleaned.csv                       # Cleaned transaction table (100,000 x 26)
?       ??? customer_features.csv                    # Customer RFM metrics (43,233 x 15)
?       ??? customer_segments.csv                    # Customer table with KMeans labels (43,233 x 16)
?       ??? product_features.csv                     # Product catalog intelligence (50 x 20)
?       ??? daily_product_demand.csv                 # Daily demand grid with lags (91,250 x 48)
?       ??? monthly_product_demand.csv               # Monthly product demand & seasonality (17,931 x 17)
?       ??? monthly_business_kpis.csv                # 60 monthly enterprise retail KPIs (60 x 20)
?       ??? product_opportunity_risk.csv             # SKU opportunity score & stockout risk (50 x 28)
?       ??? customer_retention_risk.csv              # Customer churn scores & loyalty index (43,233 x 21)
??? notebook/
?   ??? 01_data_understanding.ipynb                  # Data schema validation and summary stats
?   ??? 02_eda.ipynb                                 # Exploratory data analysis and visualizations
?   ??? 03_feature_engineering.ipynb                 # Customer RFM, product features, and demand lags
?   ??? 04_demand_forecasting.ipynb                  # Time-split demand forecasting (LGBM, XGB, Ridge)
?   ??? 05_customer_segmentation.ipynb               # Unsupervised KMeans clustering (k=4) & PCA
?   ??? 06_business_signals_and_kpis.ipynb           # Monthly KPIs, opportunity matrix, churn risk
??? prediction/
?   ??? 01_sales_prediction.ipynb                    # Supervised sales regression benchmark
?   ??? models/
?   ?   ??? best_sales_prediction_model.pkl          # Serialized inference pipeline (Pickle)
?   ?   ??? best_sales_prediction_model.joblib       # Serialized inference pipeline (Joblib)
?   ??? outputs/
?       ??? predictions.csv                          # Holdout predictions with actuals & residuals (20,000 x 11)
??? outputs/
?   ??? customer_segment_profile.csv                 # Centroid summaries of customer segments
?   ??? figures/                                     # Diagnostic and evaluation plots (.png)
??? src/                                             # Helper modules (placeholder for packaging)
??? .gitignore                                       # Python and Jupyter ignore configuration
??? README.md                                        # Project documentation
```

---

## How to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/chiluverusrija7/RetailMindAI.git
   cd RetailMindAI
   ```

2. **Install dependencies:**
   ```bash
   pip install numpy pandas scikit-learn xgboost lightgbm matplotlib seaborn joblib nbclient nbformat
   ```

3. **Execute the pipeline notebooks in sequence:**
   ```bash
   jupyter nbconvert --to notebook --execute notebook/01_data_understanding.ipynb
   jupyter nbconvert --to notebook --execute notebook/02_eda.ipynb
   jupyter nbconvert --to notebook --execute notebook/03_feature_engineering.ipynb
   jupyter nbconvert --to notebook --execute notebook/04_demand_forecasting.ipynb
   jupyter nbconvert --to notebook --execute notebook/05_customer_segmentation.ipynb
   jupyter nbconvert --to notebook --execute notebook/06_business_signals_and_kpis.ipynb
   jupyter nbconvert --to notebook --execute prediction/01_sales_prediction.ipynb
   ```

---

## Project Limitations & Assumptions
- **Absence of Wholesale Cost Data:** Wholesale cost of goods sold (COGS) is not present in the dataset; product margin realization is estimated via net unit revenue (`UnitPrice * (1 - Discount)`).
- **Catalog Size:** The catalog is composed of 50 persistent products across the 5-year observation period.
- **No Real-Time Serving Infrastructure:** Models and pipelines are serialized for offline batch inference; no active web API or streaming ingestion service is currently deployed.

---

## Authors & Academic Context
- **Author:** Chiluveru Srija ([@chiluverusrija7](https://github.com/chiluverusrija7))
- **Project:** RetailMindAI ? Machine Learning Academic Project
