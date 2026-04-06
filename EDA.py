import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")

OUTPUT_DIR = "eda_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PALETTE  = "viridis"
BG_COLOR = "#0d1117"
FG_COLOR = "#e6edf3"
ACCENT   = "#58a6ff"
plt.rcParams.update({
    "figure.facecolor": BG_COLOR,
    "axes.facecolor":   BG_COLOR,
    "axes.edgecolor":   "#30363d",
    "axes.labelcolor":  FG_COLOR,
    "text.color":       FG_COLOR,
    "xtick.color":      FG_COLOR,
    "ytick.color":      FG_COLOR,
    "grid.color":       "#21262d",
    "grid.linestyle":   "--",
    "grid.alpha":       0.5,
})

print("=" * 60)
print(" AIR QUALITY EDA")
print("=" * 60)

df = pd.read_csv("dataset/city_day.csv", parse_dates=["Datetime"])
df.rename(columns={"Datetime": "Date"}, inplace=True)

print(f"\n[INFO] Dataset shape : {df.shape}")
print(f"[INFO] Columns       : {df.columns.tolist()}")
print(f"[INFO] Date range    : {df['Date'].min().date()} - {df['Date'].max().date()}")
print(f"[INFO] Cities        : {sorted(df['City'].unique())}")

# 1. BASIC INFO

print("\n" + "─" * 60)
print("1. DATA TYPES & MEMORY")
print("─" * 60)
print(df.dtypes)
print(f"\nMemory usage : {df.memory_usage(deep=True).sum() / 1_048_576:.2f} MB")

# 2. MISSING VALUES

print("\n" + "─" * 60)
print("2. MISSING VALUES")
print("─" * 60)
null_pct = (df.isnull().sum() / len(df) * 100).round(2)
null_df  = pd.DataFrame({"Missing Count": df.isnull().sum(), "Missing %": null_pct})
print(null_df[null_df["Missing Count"] > 0].sort_values("Missing %", ascending=False))

fig, ax = plt.subplots(figsize=(12, 5))
null_df["Missing %"].sort_values(ascending=False).plot(kind="bar", ax=ax, color=ACCENT, edgecolor="#30363d")
ax.set_title("Missing Value Percentage per Feature", fontsize=14, weight="bold", color=FG_COLOR)
ax.set_xlabel("Feature", fontsize=11)
ax.set_ylabel("Missing %", fontsize=11)
ax.yaxis.set_major_formatter(mticker.PercentFormatter())
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/01_missing_values.png", dpi=150)
plt.close()
print(f"  Saved: {OUTPUT_DIR}/01_missing_values.png")

# 3. DESCRIPTIVE STATISTICS

print("\n" + "=" * 60)
print("3. DESCRIPTIVE STATISTICS")
print("=" * 60)
desc = df.describe().round(3)
print(desc.to_string())

# 4. AQI DISTRIBUTION

print("\n" + "=" * 60)
print("4. AQI DISTRIBUTION")
print("=" * 60)
print(df["AQI"].describe().round(2))

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Histogram + KDE
axes[0].hist(df["AQI"].dropna(), bins=60, color=ACCENT, edgecolor=BG_COLOR, alpha=0.8, density=True)
df["AQI"].dropna().plot(kind="kde", ax=axes[0], color="#f78166", linewidth=2)
axes[0].set_title("AQI Distribution (Histogram + KDE)", fontsize=12, weight="bold")
axes[0].set_xlabel("AQI")

# Boxplot
axes[1].boxplot(df["AQI"].dropna(), patch_artist=True,
                boxprops=dict(facecolor=ACCENT, color="#30363d"),
                medianprops=dict(color="#f78166", linewidth=2),
                whiskerprops=dict(color=FG_COLOR),
                capprops=dict(color=FG_COLOR),
                flierprops=dict(marker="o", color="#f78166", markersize=3, alpha=0.5))
axes[1].set_title("AQI Boxplot", fontsize=12, weight="bold")
axes[1].set_ylabel("AQI")

# Log-scale histogram
axes[2].hist(np.log1p(df["AQI"].dropna()), bins=60, color="#3fb950", edgecolor=BG_COLOR, alpha=0.8)
axes[2].set_title("AQI Distribution (log1p scale)", fontsize=12, weight="bold")
axes[2].set_xlabel("log1p(AQI)")

plt.suptitle("AQI Distribution Analysis", fontsize=15, weight="bold", color=FG_COLOR, y=1.02)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_aqi_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"   Saved: {OUTPUT_DIR}/02_aqi_distribution.png")

# Skewness & Kurtosis
skew = df["AQI"].skew()
kurt = df["AQI"].kurtosis()
print(f"  Skewness : {skew:.4f}  |  Kurtosis : {kurt:.4f}")


# 5. AQI BY CITY

print("\n" + "─" * 60)
print("5. AQI BY CITY")
print("─" * 60)
city_stats = df.groupby("City")["AQI"].agg(["mean", "median", "std", "max"]).round(2)
print(city_stats.sort_values("mean", ascending=False))

city_order = city_stats["mean"].sort_values(ascending=False).index
colors_city = plt.cm.viridis(np.linspace(0.2, 0.9, len(city_order)))

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Bar chart
axes[0].bar(city_order, city_stats.loc[city_order, "mean"], color=colors_city, edgecolor="#30363d")
axes[0].set_title("Mean AQI by City", fontsize=13, weight="bold")
axes[0].set_xlabel("City")
axes[0].set_ylabel("Mean AQI")
for i, city in enumerate(city_order):
    axes[0].text(i, city_stats.loc[city, "mean"] + 3, f'{city_stats.loc[city, "mean"]:.0f}',
                 ha="center", va="bottom", fontsize=10, color=FG_COLOR)

# Violin plot
parts = axes[1].violinplot([df[df["City"] == c]["AQI"].dropna().values for c in city_order],
                            showmedians=True)
for i, pc in enumerate(parts["bodies"]):
    pc.set_facecolor(colors_city[i])
    pc.set_alpha(0.75)
parts["cmedians"].set_color("#f78166")
parts["cbars"].set_color(FG_COLOR)
parts["cmins"].set_color(FG_COLOR)
parts["cmaxes"].set_color(FG_COLOR)
axes[1].set_xticks(range(1, len(city_order) + 1))
axes[1].set_xticklabels(city_order)
axes[1].set_title("AQI Violin Plot by City", fontsize=13, weight="bold")
axes[1].set_ylabel("AQI")

plt.suptitle("AQI by City", fontsize=15, weight="bold", color=FG_COLOR)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/03_aqi_by_city.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"   Saved: {OUTPUT_DIR}/03_aqi_by_city.png")


# 6. AQI BUCKET DISTRIBUTION

print("\n" + "─" * 60)
print("6. AQI BUCKET DISTRIBUTION")
print("─" * 60)
bucket_order = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
bucket_counts = df["AQI_Bucket"].value_counts().reindex(bucket_order).fillna(0)
print(bucket_counts)

bucket_colors = ["#3fb950", "#58a6ff", "#e3b341", "#f78166", "#da3633", "#8b949e"]
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].bar(bucket_counts.index, bucket_counts.values, color=bucket_colors, edgecolor=BG_COLOR)
axes[0].set_title("AQI Bucket Count", fontsize=13, weight="bold")
axes[0].set_xlabel("AQI Bucket")
axes[0].set_ylabel("Count")
axes[0].tick_params(axis="x", rotation=30)

axes[1].pie(bucket_counts.values, labels=bucket_counts.index, colors=bucket_colors,
            autopct="%1.1f%%", startangle=140,
            textprops={"color": FG_COLOR},
            wedgeprops={"edgecolor": BG_COLOR, "linewidth": 1.5})
axes[1].set_title("AQI Bucket Proportion", fontsize=13, weight="bold")

plt.suptitle("AQI Bucket Analysis", fontsize=15, weight="bold", color=FG_COLOR)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/04_aqi_bucket.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"   Saved: {OUTPUT_DIR}/04_aqi_bucket.png")

# 7. TIME-SERIES TREND

print("\n" + "─" * 60)
print("7. TIME-SERIES TREND")
print("─" * 60)

monthly = df.set_index("Date").resample("ME")["AQI"].mean().reset_index()

fig, ax = plt.subplots(figsize=(16, 5))
ax.plot(monthly["Date"], monthly["AQI"], color=ACCENT, linewidth=1.5, label="Monthly Mean AQI")
ax.fill_between(monthly["Date"], monthly["AQI"], alpha=0.15, color=ACCENT)
ax.set_title("Monthly Average AQI Over Time", fontsize=14, weight="bold")
ax.set_xlabel("Date")
ax.set_ylabel("Mean AQI")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/05_aqi_timeseries.png", dpi=150)
plt.close()
print(f"   Saved: {OUTPUT_DIR}/05_aqi_timeseries.png")

# Per-city trend
fig, ax = plt.subplots(figsize=(16, 6))
city_colors = plt.cm.tab10(np.linspace(0, 1, df["City"].nunique()))
for i, city in enumerate(sorted(df["City"].unique())):
    sub = df[df["City"] == city].set_index("Date").resample("ME")["AQI"].mean()
    ax.plot(sub.index, sub.values, label=city, linewidth=1.5, color=city_colors[i])
ax.set_title("Monthly AQI Trend per City", fontsize=14, weight="bold")
ax.set_xlabel("Date")
ax.set_ylabel("AQI")
ax.legend(loc="upper right")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/06_aqi_trend_per_city.png", dpi=150)
plt.close()
print(f"   Saved: {OUTPUT_DIR}/06_aqi_trend_per_city.png")

# 8. SEASONAL ANALYSIS

print("\n" + "─" * 60)
print("8. SEASONAL & MONTHLY PATTERNS")
print("─" * 60)

df["Month"] = df["Date"].dt.month
df["Year"]  = df["Date"].dt.year

monthly_mean = df.groupby("Month")["AQI"].mean()
print(monthly_mean.round(2))

month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

axes[0].bar(range(1, 13), monthly_mean.values, color=plt.cm.plasma(np.linspace(0.1, 0.9, 12)),
            edgecolor=BG_COLOR)
axes[0].set_xticks(range(1, 13))
axes[0].set_xticklabels(month_names)
axes[0].set_title("Average AQI by Month", fontsize=13, weight="bold")
axes[0].set_ylabel("Mean AQI")

pivot = df.pivot_table(index="Month", columns="City", values="AQI", aggfunc="mean")
sns.heatmap(pivot, ax=axes[1], cmap="YlOrRd", annot=True, fmt=".0f",
            linewidths=0.5, linecolor="#21262d",
            cbar_kws={"shrink": 0.8})
axes[1].set_title("Month × City AQI Heatmap", fontsize=13, weight="bold")
axes[1].set_yticklabels(month_names, rotation=0)

plt.suptitle("Seasonal AQI Patterns", fontsize=15, weight="bold", color=FG_COLOR)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/07_seasonal_patterns.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"   Saved: {OUTPUT_DIR}/07_seasonal_patterns.png")

# 9. CORRELATION MATRIX

print("\n" + "─" * 60)
print("9. CORRELATION MATRIX")
print("─" * 60)

num_cols = df.select_dtypes(include=np.number).drop(columns=["Month", "Year"], errors="ignore").columns.tolist()
corr = df[num_cols].corr()
print("\nTop correlations with AQI:")
print(corr["AQI"].drop("AQI").sort_values(ascending=False).round(4))

fig, ax = plt.subplots(figsize=(13, 10))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            linewidths=0.5, linecolor="#21262d",
            annot_kws={"size": 8}, ax=ax,
            cbar_kws={"shrink": 0.8})
ax.set_title("Feature Correlation Matrix", fontsize=14, weight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/08_correlation_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"   Saved: {OUTPUT_DIR}/08_correlation_matrix.png")

# 10. FEATURE DISTRIBUTIONS

print("\n" + "─" * 60)
print("10. FEATURE DISTRIBUTIONS")
print("─" * 60)

pollutants = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3", "Benzene", "Toluene", "Xylene"]
pollutants = [c for c in pollutants if c in df.columns]

ncols = 4
nrows = (len(pollutants) + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(20, nrows * 3.5))
axes = axes.flatten()
pal = plt.cm.viridis(np.linspace(0.2, 0.85, len(pollutants)))

for i, col in enumerate(pollutants):
    data = df[col].dropna()
    axes[i].hist(data, bins=50, color=pal[i], edgecolor=BG_COLOR, alpha=0.85)
    axes[i].set_title(f"{col}", fontsize=11, weight="bold")
    axes[i].set_xlabel("Value")
    axes[i].set_ylabel("Frequency")
    sk = data.skew()
    axes[i].text(0.97, 0.95, f"skew={sk:.2f}", transform=axes[i].transAxes,
                 ha="right", va="top", fontsize=9, color=FG_COLOR)

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle("Pollutant Feature Distributions", fontsize=15, weight="bold", color=FG_COLOR, y=1.01)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/09_feature_distributions.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"   Saved: {OUTPUT_DIR}/09_feature_distributions.png")

# 11. SCATTER PLOTS vs AQI

print("\n" + "─" * 60)
print("11. SCATTER PLOTS – FEATURES vs AQI")
print("─" * 60)

top_corr = corr["AQI"].drop("AQI").abs().sort_values(ascending=False).head(6).index.tolist()

ncols = 3
nrows = 2
fig, axes = plt.subplots(nrows, ncols, figsize=(18, 10))
axes = axes.flatten()

for i, col in enumerate(top_corr):
    valid = df[[col, "AQI"]].dropna()
    axes[i].scatter(valid[col], valid["AQI"], alpha=0.25, s=10,
                    c=valid["AQI"], cmap=PALETTE, rasterized=True)
    m, b, r, p, _ = stats.linregress(valid[col], valid["AQI"])
    x_line = np.linspace(valid[col].min(), valid[col].max(), 200)
    axes[i].plot(x_line, m * x_line + b, color="#f78166", linewidth=2)
    axes[i].set_title(f"{col} vs AQI  (r={r:.3f})", fontsize=11, weight="bold")
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("AQI")

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle("Top-Correlated Features vs AQI", fontsize=15, weight="bold", color=FG_COLOR)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/10_scatter_vs_aqi.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"   Saved: {OUTPUT_DIR}/10_scatter_vs_aqi.png")

# 12. OUTLIER ANALYSIS

print("\n" + "─" * 60)
print("12. OUTLIER ANALYSIS")
print("─" * 60)

fig, ax = plt.subplots(figsize=(16, 6))
data_for_box = [df[c].dropna().values for c in pollutants]
bp = ax.boxplot(data_for_box, patch_artist=True, notch=False,
                flierprops=dict(marker=".", markersize=2, alpha=0.4))
for patch, color in zip(bp["boxes"], pal):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
for median in bp["medians"]:
    median.set_color("#f78166")
    median.set_linewidth(2)
ax.set_xticks(range(1, len(pollutants) + 1))
ax.set_xticklabels(pollutants, rotation=30, ha="right")
ax.set_title("Pollutant Outlier Summary (Boxplots)", fontsize=14, weight="bold")
ax.set_ylabel("Value")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/11_outlier_boxplots.png", dpi=150)
plt.close()
print(f"   Saved: {OUTPUT_DIR}/11_outlier_boxplots.png")

# IQR-based outlier counts
print("\nIQR-based outlier counts per feature:")
for col in pollutants + ["AQI"]:
    if col not in df.columns:
        continue
    q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    iqr = q3 - q1
    n_out = ((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum()
    print(f"  {col:<12}: {n_out:>5} outliers ({n_out/len(df)*100:.2f}%)")


# 13. YEAR-WISE AQI TREND

print("\n" + "─" * 60)
print("13. YEAR-WISE AQI TREND")
print("─" * 60)

yearly = df.groupby(["Year", "City"])["AQI"].mean().unstack()
fig, ax = plt.subplots(figsize=(14, 5))
for i, city in enumerate(yearly.columns):
    ax.plot(yearly.index, yearly[city], marker="o", linewidth=2,
            label=city, color=city_colors[i])
ax.set_title("Year-wise Mean AQI by City", fontsize=14, weight="bold")
ax.set_xlabel("Year")
ax.set_ylabel("Mean AQI")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/12_yearwise_aqi.png", dpi=150)
plt.close()
print(f"   Saved: {OUTPUT_DIR}/12_yearwise_aqi.png")

# 14. MULTICOLLINEARITY (VIF) FOR REGRESSION

print("\n" + "─" * 60)
print("14. MULTICOLLINEARITY (VARIANCE INFLATION FACTOR)")
print("─" * 60)

from statsmodels.stats.outliers_influence import variance_inflation_factor

# Select only numeric pollutant columns and drop rows with NaNs to calculate VIF
vif_cols = [c for c in pollutants if c in df.columns]
vif_data = df[vif_cols].dropna()

# Add a constant term for the intercept
vif_data_with_const = vif_data.copy()
vif_data_with_const["Constant"] = 1

vif_df = pd.DataFrame()
vif_df["Feature"] = vif_data.columns
vif_df["VIF"] = [
    variance_inflation_factor(vif_data_with_const.values, i) 
    for i in range(vif_data.shape[1])
]

print(vif_df.sort_values("VIF", ascending=False).round(2).to_string(index=False))

# Plot VIF
fig, ax = plt.subplots(figsize=(10, 6))
vif_plot_df = vif_df.sort_values("VIF", ascending=True)
ax.barh(vif_plot_df["Feature"], vif_plot_df["VIF"], color="#e3b341", edgecolor=BG_COLOR)
ax.axvline(x=5, color="#da3633", linestyle="--", linewidth=2, label="Threshold (VIF=5)")
ax.axvline(x=10, color="#f78166", linestyle=":", linewidth=2, label="Threshold (VIF=10)")
ax.set_title("Variance Inflation Factor (Multicollinearity)", fontsize=14, weight="bold")
ax.set_xlabel("VIF Score")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/13_vif_multicollinearity.png", dpi=150)
plt.close()
print(f"   Saved: {OUTPUT_DIR}/13_vif_multicollinearity.png")

# 15. AUTOCORRELATION & LAG ANALYSIS

print("\n" + "─" * 60)
print("15. AUTOCORRELATION & LAG ANALYSIS (TIME-SERIES)")
print("─" * 60)

from pandas.plotting import autocorrelation_plot

# Delhi as example for the autocorrelation plot to avoid a messy global plot
delhi_aqi = df[df["City"] == "Delhi"].set_index("Date")["AQI"].dropna()

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Scatter plot: AQI(t) vs AQI(t-1)
axes[0].scatter(delhi_aqi.shift(1), delhi_aqi, alpha=0.3, color=ACCENT, s=10)
axes[0].set_title("Delhi: AQI Today vs AQI Yesterday", fontsize=13, weight="bold")
axes[0].set_xlabel("AQI (t-1)")
axes[0].set_ylabel("AQI (t)")

# Autocorrelation plot
autocorrelation_plot(delhi_aqi.resample("W").mean().dropna(), ax=axes[1], color="#3fb950")
axes[1].set_title("Delhi: AQI Autocorrelation (Weekly)", fontsize=13, weight="bold")

plt.suptitle("Lag & Autocorrelation Exploration", fontsize=15, weight="bold", color=FG_COLOR)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/14_autocorrelation.png", dpi=150)
plt.close()
print(f"   Saved: {OUTPUT_DIR}/14_autocorrelation.png")

# 16. MISSING DATA PATTERN (SENSOR OUTAGES)

print("\n" + "─" * 60)
print("16. MISSING DATA PATTERN MATRIX")
print("─" * 60)

import missingno as msno

fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(111)
# Visualizing missing data for the first 2000 rows as a sample
msno.matrix(df.head(2000), ax=ax, sparkline=False, color=(0.34, 0.65, 1.0))
ax.set_title("Missing Data Pattern Matrix (First 2000 rows)", fontsize=14, weight="bold", color="black")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/15_missing_matrix.png", dpi=150)
plt.close()
print(f"   Saved: {OUTPUT_DIR}/15_missing_matrix.png")

# SUMMARY

print("\n" + "=" * 60)
print(" EDA COMPLETE")
print(f" All plots saved to: {OUTPUT_DIR}/")
print("=" * 60)
print("\nKey Insights:")
print(f"  • Dataset rows    : {len(df):,}")
print(f"  • Date range      : {df['Date'].min().date()}  {df['Date'].max().date()}")
print(f"  • Cities          : {sorted(df['City'].unique())}")
print(f"  • AQI range       : {df['AQI'].min():.1f} – {df['AQI'].max():.1f}")
print(f"  • AQI mean/std    : {df['AQI'].mean():.1f} / {df['AQI'].std():.1f}")
print(f"  • AQI skewness    : {df['AQI'].skew():.4f}")
print(f"  • Best corr w/AQI : {corr['AQI'].drop('AQI').abs().idxmax()} "
      f"({corr['AQI'].drop('AQI').abs().max():.4f})")