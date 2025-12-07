"""
Cross-Validation for Stock Market Data
Time Series Cross-Validation Module

This module demonstrates how to properly implement time series cross-validation
for stock market data to avoid data leakage and ensure realistic evaluation.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import root_mean_squared_error
import matplotlib.pyplot as plt
import random

# Configuration
TICKER = "TSLA"

# Create DataFrame with TICKER
tesla = pd.read_csv(f"../data/features/{TICKER}_features.csv")
print(f"Size of {TICKER} dataset: {tesla.shape}")
print(tesla.tail())

# Visualize Time Series Split
def createTimeSeriesSplit(data, n_splits=5, test_size=90, gap=50):
    """
    Generates training and testing indices on each time-series-split data

    Args:
        data: dataset (or feature matrix) of a stock to split.
        n_splits: Number of folds.
        test_size: Size of each test set.
        gap: Number of samples to exclude between train and test (must be equal to test_size).

    Returns:
        List of dicts with keys: (1) "fold": fold number, 0-based (2) "train_idx": training indices (3) "test_idx": testing indices
    """
    tss = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=gap)

    folds = [] 
    for k, (train_idx, test_idx) in enumerate(tss.split(data)): # enum to add k folds 
        folds.append({
            "fold": k,
            "train_idx": train_idx,  
            "test_idx": test_idx,
        })
    return folds

# create time series split
folds = createTimeSeriesSplit(data=tesla)

def previewFold(folds: list):
    """Preview the fold structure to understand the train/test splits"""
    for k in folds:
        print("fold:", k["fold"], "\ntrain_idx", k["train_idx"][:20], "...", k["train_idx"][-20:], 
                        "\ntest_idex", k["test_idx"][:20], "...", k["test_idx"][-15:])

previewFold(folds)

# Visualize each fold in TimeSeriesSplit CV
fig, axs = plt.subplots(5, 1, figsize=(15, 15), sharex=True)

for fold in folds: 
    k = fold["fold"]
    train_idx = fold["train_idx"]
    test_idx  = fold["test_idx"]

    train = tesla.iloc[train_idx]
    test  = tesla.iloc[test_idx]

    ax = axs[k]
    train['next_day_return'].plot(ax=ax, label="Training Set", title=f"Data Train/Test Split Fold {k}")
    test['next_day_return'].plot(ax=ax, label="Test Set")
    ax.legend(loc="upper right")
    ax.axvline(test.index.min(), color="black", ls="--")

plt.tight_layout()
plt.savefig("../data/processed/time_series_cv_visualization.png", dpi=300, bbox_inches='tight')
plt.show()

print("Time series cross-validation visualization saved: ../data/processed/time_series_cv_visualization.png")

# Example: How to use the folds for model training and evaluation
def demonstrate_cv_workflow(folds, data):
    """
    Demonstrate how to use the time series cross-validation folds
    for model training and evaluation workflow.
    """
    print("\nDemonstrating CV workflow:")
    print("="*50)
    
    for fold in folds:
        fold_num = fold["fold"]
        train_idx = fold["train_idx"]
        test_idx = fold["test_idx"]
        
        # Get training and testing data
        train_data = data.iloc[train_idx]
        test_data = data.iloc[test_idx]
        
        print(f"\nFold {fold_num}:")
        print(f"  Training period: {train_data['date'].min()} to {train_data['date'].max()}")
        print(f"  Testing period: {test_data['date'].min()} to {test_data['date'].max()}")
        print(f"  Training samples: {len(train_data)}")
        print(f"  Testing samples: {len(test_data)}")
        
        # Here you would typically:
        # 1. Extract features and targets from train_data
        # 2. Train your model on training data
        # 3. Make predictions on test_data
        # 4. Evaluate performance
        
        # Example placeholder for model training/evaluation
        train_target_mean = train_data['next_day_return'].mean()
        test_target_mean = test_data['next_day_return'].mean()
        
        print(f"  Train target mean: {train_target_mean:.6f}")
        print(f"  Test target mean: {test_target_mean:.6f}")

demonstrate_cv_workflow(folds, tesla)

print("\n" + "="*70)
print("Cross-Validation Setup Complete")
print("="*70)
print(f"Stock: {TICKER}")
print(f"Total data points: {len(tesla)}")
print(f"Number of CV folds: {len(folds)}")
print("Ready for model training and evaluation!")
print("="*70)