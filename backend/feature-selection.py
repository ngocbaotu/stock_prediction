"""
Feature Selection for Stock Market Data
Statistical Feature Selection Module

This module implements a comprehensive feature selection pipeline including:
1. Variance-based filtering
2. Multicollinearity detection and removal
3. Statistical feature selection using mutual information

The goal is to reduce dimensionality while retaining the most informative features
for stock return prediction.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.feature_selection import VarianceThreshold
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.feature_selection import SelectKBest, mutual_info_regression

# Configuration
TICKER = 'TSLA'  # Change it to the Ticker you want to save
DATA_DIR = '../data/features'
OUTPUT_DIR = '../data/selected'

# Load Stock Data X and y
stocks = pd.read_csv(f"{DATA_DIR}/complete_features.csv")

def create_feature_matrix(df, use_scaled=True):
    """
    Create final feature matrix (X) and target vector (y) for each stock.

    Args:
        df: DataFrame with all features
        use_scaled: Whether to use scaled features (default True)

    Returns:
        Dictionary with stock symbols as keys, containing X, y, and dates
    """
    # Identify feature columns
    exclude_cols = ['date', 'symbol', 'next_day_return']
    
    if use_scaled:
        # Use only scaled features
        feature_cols = [col for col in df.columns if col.endswith('_scaled')]
    else:
        # Use original features
        feature_cols = [col for col in df.columns if col not in exclude_cols and not col.endswith('_scaled')]
    
    # Target variable
    target_col = 'next_day_return'
    
    stock_data = {}
    
    for symbol in df['symbol'].unique():
        stock_df = df[df['symbol'] == symbol].copy()
        stock_df = stock_df.sort_values('date')
        
        # Remove rows with missing target or features
        valid_mask = (
            stock_df[target_col].notnull() & 
            stock_df[feature_cols].notnull().all(axis=1)
        )
        
        stock_df_clean = stock_df[valid_mask].copy()
        
        # Create X and y
        X = stock_df_clean[feature_cols]
        y = stock_df_clean[[target_col]]
        dates = pd.to_datetime(stock_df_clean['date'])
        
        stock_data[symbol] = {
            'X': X,
            'y': y,
            'dates': dates,
            'feature_names': feature_cols,
            'n_samples': len(X),
            'n_features': len(feature_cols)
        }
    
    return stock_data

# create feature matrices X and y
feature_matrices = create_feature_matrix(df=stocks)

# Investigate a particular ticker
print(f"Feature matrix for {TICKER}:")
print(feature_matrices[TICKER]['X'])
print(f"Shape: {feature_matrices[TICKER]['X'].shape}")

# Feature Selection
print("\n" + "="*70)
print("FEATURE SELECTION PIPELINE")
print("="*70)

## Step 1: Drop relatively low-variance columns
print("\nStep 1: Variance-based feature selection")
print("-" * 50)

X = feature_matrices[TICKER]['X']
variances = X.var()
feature_cols = feature_matrices[TICKER]['feature_names']

print(f"Initial number of features: {len(feature_cols)}")
print(f"Variance range: {variances.min():.6f} to {variances.max():.6f}")

# Visualize variance distribution
plt.figure(figsize=(10, 5))
plt.hist(variances, bins=30)
plt.title("Feature Variance Distribution")
plt.xlabel("Feature variance")
plt.ylabel("Count")
plt.axvline(np.percentile(variances, 25), color='red', linestyle='--', 
           label=f'25th percentile: {np.percentile(variances, 25):.3f}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}_variance_distribution.png", dpi=300, bbox_inches='tight')
plt.show()

# Drop bottom 25% of the variance
T = np.percentile(variances, 25) 
selector = VarianceThreshold(threshold=T)
X_cleaned = selector.fit_transform(X)
print(f"Variance threshold: {T:.6f}")
print(f"Features after variance filtering: {X_cleaned.shape[1]}")

# Boolean mask of kept features
support_mask = selector.get_support()

# Features that were kept
kept_features = [f for f, keep in zip(feature_cols, support_mask) if keep]

# Features that were dropped
dropped_features = [f for f, keep in zip(feature_cols, support_mask) if not keep]

print(f"Dropped {len(dropped_features)} low-variance features:")
for feat in dropped_features[:10]:  # Show first 10
    print(f"  - {feat}: variance = {variances[feat]:.6f}")
if len(dropped_features) > 10:
    print(f"  ... and {len(dropped_features) - 10} more")

# Update feature matrices
feature_matrices[TICKER]['X'] = pd.DataFrame(X_cleaned, columns=kept_features)
feature_matrices[TICKER]['feature_names'] = kept_features
feature_matrices[TICKER]['n_features'] = X_cleaned.shape[1]

# Reinvestigate that particular ticker
print(f"\nFeatures after Step 1: {feature_matrices[TICKER]['X'].shape}")

## Step 2: Drop multicollinear columns
print("\nStep 2: Multicollinearity detection and removal")
print("-" * 50)

X = feature_matrices[TICKER]['X']

# Calculate variance inflation factors (higher VIF = stronger multicollinearity)
print("Calculating Variance Inflation Factors (VIF)...")
vif = pd.DataFrame()
vif["feature"] = X.columns
vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

print("\nTop 10 features by VIF:")
print(vif.head(10))

# Check for infinite VIF values
inf_vif_count = np.sum(np.isinf(vif["VIF"]))
print(f"\nFeatures with infinite VIF: {inf_vif_count}")

if inf_vif_count > 0:
    print("Features with infinite VIF (perfect multicollinearity):")
    inf_features = vif[np.isinf(vif["VIF"])]["feature"].tolist()
    for feat in inf_features:
        print(f"  - {feat}")

# Create a boolean mask for infinite VIF
# This is what we are going to drop eventually
drop_mask = np.isinf(vif["VIF"])

# Identify which features to drop
dropped_features_step2 = vif.loc[drop_mask, "feature"].tolist()
kept_features = vif.loc[~drop_mask, "feature"].tolist()

print(f"\nDropping {len(dropped_features_step2)} features with infinite VIF")
print(f"Remaining features: {len(kept_features)}")

# Update feature matrices
X_step2 = X[kept_features]
feature_matrices[TICKER]['X'] = pd.DataFrame(X_step2, columns=kept_features)
feature_matrices[TICKER]['feature_names'] = kept_features
feature_matrices[TICKER]['n_features'] = len(kept_features)

# Reinvestigate that particular ticker
print(f"Features after Step 2: {feature_matrices[TICKER]['X'].shape}")

## Step 3: SelectKBest
print("\nStep 3: Statistical feature selection (SelectKBest with Mutual Information)")
print("-" * 50)

X = feature_matrices[TICKER]['X']
y = feature_matrices[TICKER]['y']
dates = feature_matrices[TICKER]['dates']
feature_cols = feature_matrices[TICKER]['feature_names']

# Apply SelectKBest with mutual information
K_BEST = 20  # Number of features to select
print(f"Selecting top {K_BEST} features using mutual information regression...")

selector = SelectKBest(mutual_info_regression, k=K_BEST)
X_cleaned = selector.fit_transform(X, np.ravel(y))

print(f"Features after SelectKBest: {X_cleaned.shape[1]}")

# Get feature scores
feature_scores = selector.scores_
feature_score_df = pd.DataFrame({
    'feature': feature_cols,
    'score': feature_scores,
    'selected': selector.get_support()
}).sort_values('score', ascending=False)

print(f"\nTop {K_BEST} selected features by mutual information score:")
selected_features = feature_score_df[feature_score_df['selected']]
for idx, row in selected_features.iterrows():
    print(f"  {row['feature']}: {row['score']:.6f}")

# Reconstruct columns
support_mask = selector.get_support()
kept_features = list(np.array(feature_cols)[support_mask])
dropped_features_step3 = list(np.array(feature_cols)[~support_mask])
X_cleaned = pd.DataFrame(X[kept_features], columns=kept_features)

# Update feature matrices
feature_matrices[TICKER]['X'] = pd.DataFrame(X_cleaned, columns=kept_features)
feature_matrices[TICKER]['feature_names'] = kept_features
feature_matrices[TICKER]['n_features'] = X_cleaned.shape[1]

# Reinvestigate final feature matrix
X_final = feature_matrices[TICKER]['X']
X_final.index = dates

print(f"\nFinal feature matrix shape: {X_final.shape}")
print(f"Final features: {list(X_final.columns)}")

# Visualize feature selection results
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Feature scores
axes[0].bar(range(len(feature_score_df)), feature_score_df['score'])
axes[0].axhline(y=feature_score_df[feature_score_df['selected']]['score'].min(), 
                color='red', linestyle='--', label=f'Selection threshold')
axes[0].set_title('Feature Scores (Mutual Information)')
axes[0].set_xlabel('Feature Index')
axes[0].set_ylabel('Mutual Information Score')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Selected vs dropped features
selection_counts = feature_score_df['selected'].value_counts()
axes[1].pie(selection_counts.values, labels=['Dropped', 'Selected'], 
           autopct='%1.1f%%', startangle=90)
axes[1].set_title(f'Feature Selection Results\n(Top {K_BEST} out of {len(feature_cols)})')

plt.tight_layout()
#plt.savefig(f"{OUTPUT_DIR}_feature_selection_results.png", dpi=300, bbox_inches='tight')
plt.show()

# Save X,y as CSV files
print("\nSaving selected features...")
print("-" * 30)

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Save final features and targets
# X_final.to_csv(f"{OUTPUT_DIR}/{TICKER}_X.csv", index=True)
# y.to_csv(f"{OUTPUT_DIR}/{TICKER}_y.csv", index=False)

# print(f"Saved feature matrix: {OUTPUT_DIR}/{TICKER}_X.csv")
# print(f"Saved target vector: {OUTPUT_DIR}/{TICKER}_y.csv")
""" 
# Save feature selection metadata
metadata = {
    'ticker': TICKER,
    'original_features': len(stocks.columns),
    'scaled_features': len([col for col in stocks.columns if col.endswith('_scaled')]),
    'after_variance_filter': len(kept_features) + len(dropped_features_step2) + len(dropped_features_step3),
    'after_multicollinearity_filter': len(kept_features) + len(dropped_features_step3),
    'final_features': len(kept_features),
    'selected_features': kept_features,
    'variance_threshold': float(T),
    'k_best': K_BEST,
}

import json
with open(f"{OUTPUT_DIR}/{TICKER}_selection_metadata.json", 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"Saved metadata: {OUTPUT_DIR}/{TICKER}_selection_metadata.json")
 """
# Summary
print("\n" + "="*70)
print("FEATURE SELECTION SUMMARY")
print("="*70)
print(f"Stock: {TICKER}")
print(f"Original scaled features: {len([col for col in stocks.columns if col.endswith('_scaled')])}")
print(f"After variance filtering: {len(kept_features) + len(dropped_features_step2) + len(dropped_features_step3)}")
print(f"After multicollinearity removal: {len(kept_features) + len(dropped_features_step3)}")
print(f"Final selected features: {len(kept_features)}")
print(f"Feature reduction: {(1 - len(kept_features) / len([col for col in stocks.columns if col.endswith('_scaled')]))*100:.1f}%")
print(f"Data shape: {X_final.shape}")
print("\nFeature-selected data is ready for model training!")
print("="*70)