# 🌍 Air Quality Index (AQI) Prediction System

An end-to-end Machine Learning and Time-Series framework built to analyze, preprocess, and accurately predict the **Air Quality Index (AQI)** across various cities. This repository features a data processing pipeline engineered to preserve chronological integrity, statistical checks for multicollinearity, and regression model benchmarking.

---

## 🚀 Key Features

- 📊 **Advanced Exploratory Data Analysis (EDA)**: Automatic computation and plotting of missing value distributions, seasonal trends, pollutant concentrations, IQR-based outlier counts, and target-variable distribution (including log-scale testing).
- 🧩 **Multicollinerity & Lag Auditing**: Integrated Variance Inflation Factor (VIF) evaluation across pollutants alongside weekly lag and autocorrelation diagnostics.
- ⚙️ **Chronological Feature Engineering**:
  - Drops derived targets (`AQI_Bucket`) to completely eliminate data leakage.
  - Generates seasonal cyclical encodings (`Month_sin`, `Month_cos`) to capture cyclical context.
  - Constructs multi-variable 1-day lag states (`AQI`, `PM2.5`, `PM10`) partitioned securely by City group bounds.
- 📉 **Leakage-Free Validation Architecture**: Performs chronological 3-way splits (70% Train / 15% Validation / 15% Test) with `shuffle=False` and constraints feature scaling calculations exclusively to training distributions.
- 🤖 **Baseline Regressor Benchmarking**: Cross-evaluates performance metrics (RMSE, MAE, $R^2$, and runtimes) across ElasticNet, Random Forest, and XGBoost frameworks.

---

## 🛠️ Tech Stack

- **Core Language**: Python 3.14+
- **Data Operations**: Pandas, NumPy
- **Machine Learning**: Scikit-Learn, XGBoost
- **Statistical Operations**: Statsmodels, SciPy
- **Visualizations**: Matplotlib, Seaborn, Missingno

---
