# RetailMind AI: Enterprise Retail Intelligence System

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4%2B-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-red.svg)](https://xgboost.readthedocs.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.0%2B-brightgreen.svg)](https://lightgbm.readthedocs.io/)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-success.svg)](#)

RetailMind AI is an end-to-end, multi-tier retail intelligence and predictive analytics platform engineered to optimize demand velocity, customer lifecycle value (LTV), inventory stockout risk, and basket revenue for omnichannel retail operations.

---

## 1. System Architecture

RetailMind AI integrates transactional telemetry across an interconnected 6-phase analytical and machine learning pipeline:

```
+--------------------------------------------------------------------------------------------------+
|                                    RetailMind AI Architecture                                    |
+--------------------------------------------------------------------------------------------------+
                                                 ?
                                                 ?
               ????????????????????????????????????????????????????????????????????
               ?  Phase 1 & 2: Data Understanding & Exploratory Data Analysis     ?
               ?  ? Transaction schema validation (100,000 orders, 2020 - 2024)   ?
               ?  ? Distributions, correlations, missingness, and outlier audits   ?
               ????????????????????????????????????????????????????????????????????
                                                 ?
                                                 ?
               ????????????????????????????????????????????????????????????????????
               ?  Phase 3: Automated Feature Engineering                          ?
               ?  ? Customer RFM & Behavioral Attributes (43,233 buyers)          ?
               ?  ? Product Demand Velocity & Volatility (50 SKUs)                ?
               ?  ? Daily Time-Series Lags & Rolling Statistics (91,250 days)     ?
               ????????????????????????????????????????????????????????????????????
                              ?                  ?                 ?
              ?????????????????                  ?                 ?????????????????
              ?                                  ?                                 ?
?????????????????????????????      ?????????????????????????????     ?????????????????????????????
? Phase 4: Demand Forecast  ?      ? Phase 5: Customer Seg.    ?     ? Phase 6: Prediction & KPIs?
? ? SKU-level velocity      ?      ? ? Unsupervised K-Means    ?     ? ? Transaction GMV Regr.   ?
? ? Multi-step horizon      ?      ? ? 4 Behavioral Cohorts    ?     ? ? Enterprise Retail KPIs  ?
? ? LightGBM / XGB / Ridge  ?      ? ? PCA 2D Clustering       ?     ? ? Churn & Volatility Risk ?
?????????????????????????????      ?????????????????????????????     ?????????????????????????????
              ?                                  ?                                 ?
              ??????????????????????????????????????????????????????????????????????
                                                 ?
                                                 ?
               ????????????????????????????????????????????????????????????????????
               ?            Unified Retail Intelligence & Decision Layer           ?
               ?  ? Stockout Risk Mitigation & Replenishment Priority             ?
               ?  ? Customer Retention Intervention & Dynamic Winback             ?
               ?  ? Margin Optimization & Markdown Depth Management               ?
               ?  ? Production-Ready Inference Pipeline (.pkl / .joblib)          ?
               ????????????????????????????????????????????????????????????????????
```

---

## 2. Dataset Sources & Ingestion

The repository processes real-world scale retail transactional data:
- **Raw Transaction Source (`data/raw/Amazon.csv`)**: 100,000 transaction records spanning **January 1, 2020 to December 29, 2024** (5 full calendar years / 1,825 days).
- **Entities Covered**:
  - `OrderID`: 100,000 unique retail orders
  - `CustomerID`: 43,233 unique non-null customers
  - `ProductID`: 50 distinct catalog SKUs across multiple retail categories (Electronics, Apparel, Home & Kitchen, Books, etc.)
  - `Geographies`: Multi-country distribution with country, state, and city telemetry
  - `Financial Attributes`: Unit price, discount percentages, calculated tax, shipping costs, order fulfillment statuses, and payment mechanisms.

---

## 3. Repository Directory Structure

```
RetailMindAI/
??? data/
?   ??? raw/
?   ?   ??? Amazon.csv                               # Source raw transaction log (100k records)
?   ??? processed/
?       ??? retail_cleaned.csv                       # Cleaned, validated transactional dataset (100k x 26)
?       ??? customer_features.csv                    # Customer-level RFM and behavioral metrics (43,233 x 15)
?       ??? customer_segments.csv                    # Customer profiles with KMeans cluster assignments (43,233 x 16)
?       ??? product_features.csv                     # Product catalog intelligence and demand velocity (50 x 20)
?       ??? daily_product_demand.csv                 # Daily time-series with lag and rolling features (91,250 x 48)
?       ??? monthly_product_demand.csv               # Monthly aggregated product demand & seasonality (17,931 x 17)
?       ??? monthly_business_kpis.csv                # 60-month macroeconomic retail business KPIs (60 x 20)
?       ??? product_opportunity_risk.csv             # SKU opportunity score, volatility CV, stockout risk (50 x 28)
?       ??? customer_retention_risk.csv              # Churn risk scores, overdue ratios, loyalty indices (43,233 x 21)
??? notebook/
?   ??? 01_data_understanding.ipynb                  # Phase 1: Data ingestion, schema audit, null/duplicate checks
?   ??? 02_eda.ipynb                                 # Phase 2: Exploratory data analysis, distributions, correlations
?   ??? 03_feature_engineering.ipynb                 # Phase 3: RFM, temporal lag/rolling features, dataset exports
?   ??? 04_demand_forecasting.ipynb                  # Phase 4: Time-split demand forecasting (LightGBM, XGBoost, Ridge)
?   ??? 05_customer_segmentation.ipynb               # Phase 5: K-Means clustering (k=4), Elbow, Silhouette, PCA
?   ??? 06_business_signals_and_kpis.ipynb           # Phase 6: Monthly KPIs, opportunity matrix, churn risk diagnostics
??? outputs/
?   ??? customer_segment_profile.csv                 # Cluster centroids and descriptive metric summary
?   ??? figures/                                     # Publication-quality diagnostic visualizations (300 DPI)
?       ??? actual_vs_predicted.png                  # Holdout test set actual vs predicted scatter plot
?       ??? business_kpi_dashboard.png               # 4-panel enterprise KPI trend dashboard
?       ??? customer_cluster_elbow.png               # K-Means inertia elbow curve (k=2..8)
?       ??? customer_cluster_silhouette.png          # Silhouette score analysis across k
?       ??? customer_churn_risk_distribution.png     # Boxplot distribution of churn risk by segment
?       ??? customer_segment_distribution.png        # Bar chart of customer cohort populations
?       ??? customer_segments_pca.png                # 2D PCA cluster projection
?       ??? lgb_feature_importance.png               # LightGBM demand forecast split gain importances
?       ??? monthly_revenue_trend.png                # 5-year longitudinal GMV and net revenue trajectory
?       ??? numerical_correlation_heatmap.png        # Feature correlation matrix
?       ??? numerical_distributions.png              # Skewness and distribution histograms
?       ??? prediction_feature_importance.png        # Top 15 transaction revenue feature importances
?       ??? prediction_model_comparison.png          # Multi-model benchmark bar charts (MAE, RMSE, R?)
?       ??? prediction_residuals.png                 # Residuals vs fitted values and error distribution
?       ??? product_opportunity_matrix.png           # 2D Product opportunity & stockout risk scatter plot
??? prediction/
?   ??? 01_sales_prediction.ipynb                    # Production sales regression notebook (OLS, RF, GBDT, XGB)
?   ??? models/
?   ?   ??? best_sales_prediction_model.pkl          # End-to-end serialized inference pipeline (Pickle)
?   ?   ??? best_sales_prediction_model.joblib       # End-to-end serialized inference pipeline (Joblib)
?   ??? outputs/
?       ??? predictions.csv                          # Holdout test predictions with residuals and IDs (20k x 11)
??? src/                                             # Modular production helper scripts
??? README.md                                        # Master repository documentation
```

---

## 4. Derived Production Datasets Summary

All generated datasets are strictly reproducible, fully validated, and persisted under `data/processed/`:

| Dataset | Records | Features | Primary Key | Description |
| :--- | :---: | :---: | :---: | :--- |
| **`retail_cleaned.csv`** | 100,000 | 26 | `OrderID` | Cleaned transactions with validated datatypes, parsed timestamps, and zero missing values. |
| **`customer_features.csv`** | 43,233 | 15 | `CustomerID` | Customer RFM metrics (Recency, Frequency, Monetary), tenure (`ActiveDays`), AOV, return rates. |
| **`customer_segments.csv`** | 43,233 | 16 | `CustomerID` | Customer feature table enriched with optimal $k=4$ KMeans cluster assignment (`customer_segment`). |
| **`product_features.csv`** | 50 | 20 | `ProductID` | Catalog intelligence: historical average demand, demand volatility, price tier, repeat buyer ratio. |
| **`daily_product_demand.csv`** | 91,250 | 48 | `ProductID, Date` | 50 products $	imes$ 1,825 days with multi-horizon lag ($1, 7, 14, 28	ext{d}$) and rolling demand stats. |
| **`monthly_product_demand.csv`** | 17,931 | 17 | `ProductID, YearMonth` | Monthly aggregated SKU volume, revenue, MoM growth rates, and seasonal demand indices. |
| **`monthly_business_kpis.csv`** | 60 | 20 | `YearMonth` | Macroeconomic enterprise health: GMV, Net Revenue, AOV, active buyer counts, return/cancel rates. |
| **`product_opportunity_risk.csv`** | 50 | 28 | `ProductID` | SKU Opportunity Score, Volatility Index ($CV$), Realized Unit Price, and Stockout Risk Level. |
| **`customer_retention_risk.csv`** | 43,233 | 21 | `CustomerID` | Purchase cycle overdue ratio, empirical Churn Risk Score, Customer Loyalty Index ($CLI$). |
| **`prediction/outputs/predictions.csv`**| 20,000 | 11 | `OrderID` | Holdout test predictions generated by winning XGBoost model with absolute and residual errors. |

---

## 5. Measurable Business Signals & Formulations

To eliminate arbitrary scoring and guarantee academic and operational defensibility, every business signal is mathematically grounded in observable data:

### 5.1 Demand Opportunity Score (DOI, 0 - 100)
Measures the joint magnitude of cumulative revenue generation and recent demand acceleration:
$$	ext{DOI}_i = 100 	imes \left(0.50 	imes rac{	ext{Rank}(	ext{TotalRevenue}_i)}{N} + 0.50 	imes rac{	ext{Rank}(	ext{DemandGrowth}_i)}{N}ight)$$

### 5.2 Demand Volatility Index ($CV$)
The coefficient of variation measuring daily sales dispersion against mean velocity:
$$CV_i = rac{	ext{DemandVolatility}_i}{	ext{HistoricalAverageDemand}_i + 10^{-5}}$$
- $CV > 0.8$: Highly erratic, intermittent demand requiring elevated safety stock buffers.
- $CV \le 0.5$: Predictable baseline demand suitable for automated continuous replenishment.

### 5.3 Realized Net Unit Price
Accounts for promotional discount erosion to assess actual margin realization:
$$	ext{RealizedPrice}_i = 	ext{AveragePrice}_i 	imes (1 - 	ext{AverageDiscount}_i)$$

### 5.4 Customer Churn Risk Score ($CRS \in [0.0, 1.0]$)
Measures purchase cycle overdue status against observed dormancy:
$$	ext{ExpectedPurchaseCycle}_j = egin{cases} 	ext{clip}\left(rac{	ext{ActiveDays}_j}{	ext{Frequency}_j - 1}, 15, 365ight), & 	ext{if } 	ext{Frequency}_j > 1 \ 90.0, & 	ext{if } 	ext{Frequency}_j = 1 \end{cases}$$
$$	ext{OverdueRatio}_j = rac{	ext{Recency}_j}{	ext{ExpectedPurchaseCycle}_j}, \quad 	ext{ChurnRiskScore}_j = 	ext{clip}\left(rac{	ext{Recency}_j}{365.0}, 0.0, 1.0ight)$$

### 5.5 Customer Loyalty Index ($CLI$, 0 - 100)
Composite percentile rank weighting Monetary capital ($40\%$), Purchase Frequency ($35\%$), and Brand Tenure ($25\%$):
$$CLI_j = 100 	imes \left(0.40 	imes rac{	ext{Rank}(	ext{Monetary}_j)}{N} + 0.35 	imes rac{	ext{Rank}(	ext{Frequency}_j)}{N} + 0.25 	imes rac{	ext{Rank}(	ext{ActiveDays}_j)}{N}ight)$$

---

## 6. Machine Learning Model Benchmarks

### 6.1 Customer Segmentation (Phase 5 ? K-Means Clustering)
Optimal clustering selected at **$k=4$** via joint Elbow and Silhouette analysis:
- **Segment 0 (Inactive / At-Risk ? 38.1% of buyers)**: Low frequency ($\mu = 1.20$), high recency ($\mu = 944$ days). Target: Winback reactivation discounts.
- **Segment 1 (Steady Regulars ? 39.8% of buyers)**: Moderate frequency ($\mu = 2.48$), balanced tenure ($\mu = 807$ days), $AOV = \$894$. Target: Cross-sell bundles.
- **Segment 2 (Loyal Champions ? 19.1% of buyers)**: Highest spend ($\mu = \$4,347$), frequent orders ($\mu = 4.26$), active tenure ($\mu = 1,144$ days). Target: VIP loyalty care and zero-friction logistics.
- **Segment 3 (High-Friction / Cancellers ? 3.0% of buyers)**: High cancellation rate ($\mu = 66\%$), low return rate. Target: Address order fulfillment friction and delivery reliability.

### 6.2 Sales & Revenue Prediction Benchmark (Phase 6 / Prediction Module)
Evaluated on an unbiased holdout test set of $20,000$ unseen transactions ($80/20$ train/test split, `random_state=42`):

| Model Architecture | MAE (\$) | RMSE (\$) | $R^2$ Score | Training Latency (s) | Selection Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **XGBoost Regressor** | **\$262.65** | **\$377.40** | **0.7266** | **0.67s** | **Selected Winner** |
| Gradient Boosting Regressor (GBDT) | \$263.12 | \$377.67 | 0.7262 | 118.14s | Candidate |
| Random Forest Regressor | \$261.30 | \$379.60 | 0.7234 | 71.65s | Candidate |
| Linear Regression (OLS Baseline) | \$330.27 | \$424.73 | 0.6537 | 0.32s | Baseline |

**Winning Model Justification**:
- **Variance Control**: XGBoost achieves the highest $R^2$ ($0.7266$) and lowest RMSE ($\$377.40$), penalizing catastrophic prediction errors much more effectively than Random Forest.
- **Inference Efficiency**: XGBoost trains in **$0.67	ext{s}$** (over **$100	imes$ faster** than standard GBDT and **$100	imes$ faster** than Random Forest), enabling real-time edge retraining and high-throughput production serving.
- **Top Feature Drivers**: Catalog unit price (`UnitPrice` ? $48.2\%$), customer spending history (`AverageOrderValue` ? $28.9\%$, `Monetary` ? $3.3\%$), and promotional discount (`Discount` ? $2.9\%$).

---

## 7. Reproduction & Execution Guide

### Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13
- Git

### Installation
```bash
git clone https://github.com/chiluverusrija7/RetailMindAI.git
cd RetailMindAI
pip install numpy pandas scikit-learn xgboost lightgbm matplotlib seaborn joblib nbclient nbformat
```

### Sequential Pipeline Execution
To execute the complete pipeline from raw ingestion to model serialization:
```bash
# Phase 1: Data Understanding
jupyter nbconvert --to notebook --execute notebook/01_data_understanding.ipynb

# Phase 2: Exploratory Data Analysis
jupyter nbconvert --to notebook --execute notebook/02_eda.ipynb

# Phase 3: Feature Engineering & Dataset Persistence
jupyter nbconvert --to notebook --execute notebook/03_feature_engineering.ipynb

# Phase 4: Demand Forecasting Engine
jupyter nbconvert --to notebook --execute notebook/04_demand_forecasting.ipynb

# Phase 5: Customer Segmentation Engine
jupyter nbconvert --to notebook --execute notebook/05_customer_segmentation.ipynb

# Phase 6: Business Intelligence & Measurable Signals
jupyter nbconvert --to notebook --execute notebook/06_business_signals_and_kpis.ipynb

# Prediction Module: Sales & Revenue Modeling & Pipeline Export
cd prediction
jupyter nbconvert --to notebook --execute 01_sales_prediction.ipynb
```

### Programmatic Inference Verification
The saved model encapsulates both preprocessing and inference logic:
```python
import joblib
import pandas as pd

# Load serialized pipeline
pipeline = joblib.load('prediction/models/best_sales_prediction_model.pkl')

# Inference on raw unseen records (no manual preprocessing required)
sample_df = pd.read_csv('data/processed/retail_cleaned.csv', nrows=5)
predictions = pipeline.predict(sample_df)
print("Predicted Transaction Sales ($):", predictions.round(2))
```

---

## 8. Assumptions, Operational Constraints & Limitations

1. **Wholesale Cost Data Absence**: The dataset contains gross transaction prices, customer discounts, and shipping revenues, but lacks direct supplier wholesale cost of goods sold (COGS). Profit margins are proxied via net realized prices (`UnitPrice * (1 - Discount)`).
2. **Promotional Context**: Markdowns are recorded as percentages applied per line item; multi-buy basket coupons or site-wide flash sales are modeled through aggregated discount depth metrics.
3. **Static SKU Catalog**: The core product catalog tracks 50 persistent products across the 5-year observation period, enabling deep longitudinal lag feature engineering ($28$-day rolling windows).

---

## 9. Contributors & Academic Context

- **Author**: Chiluveru Srija ([@chiluverusrija7](https://github.com/chiluverusrija7))
- **Institution**: KL University
- **Repository**: [https://github.com/chiluverusrija7/RetailMindAI](https://github.com/chiluverusrija7/RetailMindAI)
- **License**: Academic & Educational Open Access
