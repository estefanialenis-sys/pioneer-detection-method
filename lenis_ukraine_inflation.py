"""
Ukraine Inflation Statistical Analysis
========================================
This script performs a comprehensive statistical analysis of Ukraine's inflation
using panel data stored in panel.csv. It includes descriptive statistics,
rolling statistics, visualizations, and autocorrelation analysis.

Author: Course Project
Date: 2026-02-09
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

# =============================================================================
# SECTION 1: LOAD DATA
# =============================================================================

print("=" * 80)
print("SECTION 1: LOADING DATA")
print("=" * 80)

# Load the panel data from CSV file
df = pd.read_csv('panel.csv')

print(f"Data loaded successfully. Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print()

# =============================================================================
# SECTION 2: IDENTIFY AND EXTRACT UKRAINE DATA
# =============================================================================

print("=" * 80)
print("SECTION 2: EXTRACTING UKRAINE DATA")
print("=" * 80)

# Identify the Ukraine column (UA)
if 'UA' in df.columns:
    ua_ts = df['UA'].dropna()
    print(f"Ukraine column 'UA' found successfully")
    print(f"Number of observations: {len(ua_ts)}")
    print(f"Data type: {ua_ts.dtype}")
else:
    print("Warning: 'UA' column not found. Available columns:")
    print(df.columns.tolist())
    exit()

print()

# =============================================================================
# SECTION 3: DESCRIPTIVE STATISTICS
# =============================================================================

print("=" * 80)
print("SECTION 3: DESCRIPTIVE STATISTICS")
print("=" * 80)

# Calculate basic descriptive statistics
mean_value = ua_ts.mean()
median_value = ua_ts.median()
std_value = ua_ts.std()
min_value = ua_ts.min()
max_value = ua_ts.max()

print(f"Mean:              {mean_value:.4f}")
print(f"Median:            {median_value:.4f}")
print(f"Standard Dev:      {std_value:.4f}")
print(f"Minimum:           {min_value:.4f}")
print(f"Maximum:           {max_value:.4f}")
print(f"Range:             {max_value - min_value:.4f}")
print()

# =============================================================================
# SECTION 4: ROLLING STATISTICS (12-MONTH WINDOWS)
# =============================================================================

print("=" * 80)
print("SECTION 4: ROLLING STATISTICS (12-MONTH WINDOW)")
print("=" * 80)

# Calculate 12-month rolling mean
rolling_mean_12m = ua_ts.rolling(window=12).mean()

# Calculate 12-month rolling volatility (standard deviation)
rolling_volatility_12m = ua_ts.rolling(window=12).std()

print(f"Rolling Mean (12-month):")
print(f"  - Mean of rolling means: {rolling_mean_12m.mean():.4f}")
print(f"  - Std of rolling means:  {rolling_mean_12m.std():.4f}")
print()

print(f"Rolling Volatility (12-month):")
print(f"  - Mean of rolling volatility: {rolling_volatility_12m.mean():.4f}")
print(f"  - Std of rolling volatility:  {rolling_volatility_12m.std():.4f}")
print()

# =============================================================================
# SECTION 5: VISUALIZATIONS
# =============================================================================

print("=" * 80)
print("SECTION 5: CREATING VISUALIZATIONS")
print("=" * 80)

# Create a figure with multiple subplots
fig = plt.figure(figsize=(14, 10))

# Subplot 1: Line plot of inflation over time
ax1 = plt.subplot(3, 2, 1)
ax1.plot(ua_ts.index, ua_ts.values, linewidth=1.5, color='steelblue', label='Inflation')
ax1.plot(rolling_mean_12m.index, rolling_mean_12m.values, linewidth=2, 
         color='red', label='12-Month Rolling Mean', alpha=0.8)
ax1.set_xlabel('Time Period', fontsize=10)
ax1.set_ylabel('Inflation Rate', fontsize=10)
ax1.set_title('Ukraine Inflation Over Time with 12-Month Rolling Mean', fontsize=11, fontweight='bold')
ax1.legend(loc='best')
ax1.grid(True, alpha=0.3)

# Subplot 2: Rolling volatility
ax2 = plt.subplot(3, 2, 2)
ax2.plot(rolling_volatility_12m.index, rolling_volatility_12m.values, 
         linewidth=1.5, color='darkred')
ax2.fill_between(rolling_volatility_12m.index, rolling_volatility_12m.values, 
                  alpha=0.3, color='red')
ax2.set_xlabel('Time Period', fontsize=10)
ax2.set_ylabel('Volatility (Std Dev)', fontsize=10)
ax2.set_title('12-Month Rolling Volatility', fontsize=11, fontweight='bold')
ax2.grid(True, alpha=0.3)

# Subplot 3: Histogram of inflation
ax3 = plt.subplot(3, 2, 3)
ax3.hist(ua_ts.values, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
ax3.axvline(mean_value, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_value:.4f}')
ax3.axvline(median_value, color='green', linestyle='--', linewidth=2, label=f'Median: {median_value:.4f}')
ax3.set_xlabel('Inflation Rate', fontsize=10)
ax3.set_ylabel('Frequency', fontsize=10)
ax3.set_title('Distribution of Ukraine Inflation', fontsize=11, fontweight='bold')
ax3.legend(loc='best')
ax3.grid(True, alpha=0.3, axis='y')

# Subplot 4: Autocorrelation plot (ACF)
ax4 = plt.subplot(3, 2, 4)
plot_acf(ua_ts.values, lags=20, ax=ax4, color='steelblue')
ax4.set_title('Autocorrelation Function (ACF) - First 20 Lags', fontsize=11, fontweight='bold')
ax4.set_xlabel('Lag', fontsize=10)
ax4.set_ylabel('ACF', fontsize=10)

# Subplot 5: Box plot
ax5 = plt.subplot(3, 2, 5)
ax5.boxplot(ua_ts.values, vert=True)
ax5.set_ylabel('Inflation Rate', fontsize=10)
ax5.set_title('Box Plot of Ukraine Inflation', fontsize=11, fontweight='bold')
ax5.grid(True, alpha=0.3, axis='y')

# Subplot 6: Statistics summary text
ax6 = plt.subplot(3, 2, 6)
ax6.axis('off')
stats_text = f"""
SUMMARY STATISTICS

Mean:           {mean_value:.4f}
Median:         {median_value:.4f}
Std Dev:        {std_value:.4f}
Min:            {min_value:.4f}
Max:            {max_value:.4f}
Range:          {max_value - min_value:.4f}

ROLLING STATISTICS (12-month)

Avg Rolling Mean:    {rolling_mean_12m.mean():.4f}
Avg Rolling Vol:     {rolling_volatility_12m.mean():.4f}

Observations:       {len(ua_ts)}
"""
ax6.text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center',
         family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Adjust layout and save figure
plt.tight_layout()
plt.savefig('ukraine_inflation_analysis.png', dpi=300, bbox_inches='tight')
print("Visualization saved as 'ukraine_inflation_analysis.png'")
plt.show()

print()

# =============================================================================
# SECTION 6: SUMMARY AND EXPORT
# =============================================================================

print("=" * 80)
print("SECTION 6: ANALYSIS COMPLETE")
print("=" * 80)

# Create summary DataFrame
summary_df = pd.DataFrame({
    'Statistic': ['Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Range', 
                  'Avg Rolling Mean (12m)', 'Avg Rolling Vol (12m)', 'Observations'],
    'Value': [mean_value, median_value, std_value, min_value, max_value, 
              max_value - min_value, rolling_mean_12m.mean(), 
              rolling_volatility_12m.mean(), len(ua_ts)]
})

# Export summary statistics to CSV
summary_df.to_csv('ukraine_inflation_summary.csv', index=False)
print("Summary statistics exported to 'ukraine_inflation_summary.csv'")

# Export time series with rolling statistics to CSV
export_df = pd.DataFrame({
    'Inflation': ua_ts.values,
    'Rolling_Mean_12m': rolling_mean_12m.values,
    'Rolling_Volatility_12m': rolling_volatility_12m.values
})
export_df.to_csv('ukraine_inflation_timeseries.csv', index=False)
print("Time series data exported to 'ukraine_inflation_timeseries.csv'")

print()
print("=" * 80)
print("Script execution completed successfully!")
print("=" * 80)
