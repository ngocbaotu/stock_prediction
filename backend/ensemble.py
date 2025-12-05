import pandas as pd 
import numpy as np 
import json 
import os 
from pathlib import Path 

# Sklearn imports
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error

# Set seed for reproducibility
SEED = 100
np.random.seed(SEED)

print("All imports successful")


#Defining paths & Parameters

# Configuration
TICKER = "TSLA"  # Change this to test other stocks (AAPL, GOOGL, MSFT, AMZN, TSLA)
base_dir = Path(__file__).resolve().parent.parent
TUNING_DIR = base_dir / "model" / "tuning" #store the base model predictions
ENSEMBLE_DIR = base_dir / "model" / "ensemble" #save ensemble results

# Create ensemble output directory, prevent directory not found
os.makedirs(ENSEMBLE_DIR, exist_ok=True)

# Define file paths for base model predictions
xgb_csv_path = os.path.join(TUNING_DIR, f"{TICKER}_xgboost_best_predictions.csv")
lstm_csv_path = os.path.join(TUNING_DIR, f"{TICKER}_lstm_best_predictions.csv")

print(f"Looking for predictions in: {TUNING_DIR}")
print(f"XGBoost CSV: {xgb_csv_path}")
print(f"LSTM CSV: {lstm_csv_path}")



#Loading Predictions

# Load XGBoost predictions
try:
    xgb_df = pd.read_csv(xgb_csv_path)
    print(f"  Loaded XGBoost predictions: {xgb_df.shape}")
    print(f"  Columns: {xgb_df.columns.tolist()}")
except FileNotFoundError:
    print(f"  XGBoost file not found: {xgb_csv_path}")
    print(f"  Make sure you ran 05-models.ipynb with tuning() function first")
    xgb_df = None

# Load LSTM predictions
try:
    lstm_df = pd.read_csv(lstm_csv_path)
    print(f"  Loaded LSTM predictions: {lstm_df.shape}")
    print(f"  Columns: {lstm_df.columns.tolist()}")
except FileNotFoundError:
    print(f"  LSTM file not found: {lstm_csv_path}")
    print(f"  Make sure you ran 05-models.ipynb with tuning() function first")
    lstm_df = None



#Meta Features & Stack Predictions

if xgb_df is not None and lstm_df is not None:
    # Ensure date columns are datetime
    xgb_df['date'] = pd.to_datetime(xgb_df['date'])
    lstm_df['date'] = pd.to_datetime(lstm_df['date'])
    
    # Merge on date to align predictions
    # Using 'inner' join ensures we only keep overlapping dates
    merged_df = pd.merge(
        xgb_df.rename(columns={'predicted': 'xgb_pred', 'actual': 'actual_value'}),
        lstm_df.rename(columns={'predicted': 'lstm_pred', 'actual': 'actual_lstm'}),
        on='date',
        how='inner'
    )
    
    print(f"  Merged predictions shape: {merged_df.shape}")
    print(f"  NaN values: {merged_df.isnull().sum().sum()}")
    
    # Extract meta-features and target
    # Meta-features: predictions from both models
    meta_X = merged_df[['xgb_pred', 'lstm_pred']].values
    # Target: actual values
    meta_y = merged_df['actual_value'].values
    
    print(f"  Created meta-features: {meta_X.shape}")
    print(f"  Target shape: {meta_y.shape}")
else:
    print("  Cannot proceed: prediction files not loaded")
    exit(1)


#Meta models to test

# Define meta-models with different hyperparameters
meta_models = {
    'linear': LinearRegression(),
    'ridge_alpha_1e-5': Ridge(alpha=1e-5),
    'ridge_alpha_1e-4': Ridge(alpha=1e-4),
    'ridge_alpha_1e-3': Ridge(alpha=1e-3),
    'ridge_alpha_1e-2': Ridge(alpha=1e-2),
}

print(f" Defined {len(meta_models)} meta-models to test:")
for name in meta_models.keys():
    print(f"  - {name}")



#Cross Validate Meta Models

# Use Time Series Cross-Validation (consistent with project)
tscv = TimeSeriesSplit(n_splits=3, test_size=60, gap=30)
# Track results
results = []
best_meta_model = None
best_cv_error = float('inf')
best_model_name = None

print("\n" + "="*80)
print("CROSS-VALIDATING META-MODELS")
print("="*80 + "\n")

# Test each meta-model
for name, model in meta_models.items():
    # Cross-validate using mean squared error
    # Note: sklearn uses convention where higher is better, so use negative
    cv_scores = cross_val_score(
        model, 
        meta_X, 
        meta_y, 
        cv=tscv, 
        scoring='neg_mean_squared_error'
    )
    
    # Convert back to positive (actual MSE) and calculate RMSE
    cv_mse = -cv_scores
    cv_rmse = np.sqrt(cv_mse)
    
    avg_rmse = np.mean(cv_rmse)
    std_rmse = np.std(cv_rmse)
    
    # Track results
    results.append({
        'model': name,
        'avg_rmse': avg_rmse,
        'std_rmse': std_rmse,
        'rmse_per_fold': cv_rmse.tolist(),
    })
    
    # Update best model
    if avg_rmse < best_cv_error:
        best_cv_error = avg_rmse
        best_meta_model = model
        best_model_name = name
        best_marker = " ← BEST"
    else:
        best_marker = ""
    
    # Print results for this model
    print(f"{name:30s} | RMSE: {avg_rmse:.6f} ± {std_rmse:.6f}{best_marker}")

print("\n" + "="*80)
print(f"BEST META-MODEL: {best_model_name}")
print(f"Best CV RMSE: {best_cv_error:.6f}")
print("="*80)



#Train best model & Extract coefficients

# Train the best model on all data
best_meta_model.fit(meta_X, meta_y)

# Make ensemble predictions
ensemble_preds = best_meta_model.predict(meta_X)

# Calculate metrics on all data
ensemble_rmse = np.sqrt(mean_squared_error(meta_y, ensemble_preds))

print(f"\n Trained best meta-model: {best_model_name}")
print(f"\nEnsemble Performance:")
print(f"  RMSE: {ensemble_rmse:.6f}")

# Show model coefficients (weights)
print(f"\nModel Coefficients (How to Combine Predictions):")
print(f"  XGBoost weight: {best_meta_model.coef_[0]:.6f}")
print(f"  LSTM weight:    {best_meta_model.coef_[1]:.6f}")
print(f"  Intercept:      {best_meta_model.intercept_:.6f}")

# Interpret weights
total_weight = abs(best_meta_model.coef_[0]) + abs(best_meta_model.coef_[1])
if total_weight > 0:
    xgb_importance = abs(best_meta_model.coef_[0]) / total_weight * 100
    lstm_importance = abs(best_meta_model.coef_[1]) / total_weight * 100
    print(f"\nRelative Importance:")
    print(f"  XGBoost: {xgb_importance:.1f}%")
    print(f"  LSTM:    {lstm_importance:.1f}%")


#Saving best meta model

# Prepare model information
model_info = {
    "ticker": TICKER,
    "meta_model": best_model_name,
    "base_models": ["xgboost", "lstm"],
    "cv_rmse": float(best_cv_error),
    "cv_rmse_std": float(results[0]['std_rmse']),
    "rmse_per_fold": results[0]['rmse_per_fold'],
    "coefficients": {
        "xgboost": float(best_meta_model.coef_[0]),
        "lstm": float(best_meta_model.coef_[1])
    },
    "intercept": float(best_meta_model.intercept_),
    "model_type": str(type(best_meta_model).__name__),
}

# Add model-specific parameters
if hasattr(best_meta_model, 'alpha'):
    model_info["alpha"] = float(best_meta_model.alpha)
if hasattr(best_meta_model, 'l1_ratio'):
    model_info["l1_ratio"] = float(best_meta_model.l1_ratio)

# Save to JSON
output_json_path = os.path.join(ENSEMBLE_DIR, f"{TICKER}_ensemble_meta_model.json")

with open(output_json_path, 'w') as f:
    json.dump(model_info, f, indent=2)

print(f"\n Saved meta-model to: {output_json_path}")



#Save ensemble predictions & the results

# Create dataframe with all predictions
ensemble_results = merged_df[['date']].copy()
ensemble_results['xgb_pred'] = merged_df['xgb_pred'].values
ensemble_results['lstm_pred'] = merged_df['lstm_pred'].values
ensemble_results['ensemble_pred'] = ensemble_preds
ensemble_results['actual'] = meta_y
ensemble_results['ensemble_error'] = np.abs(ensemble_preds - meta_y)

# Save ensemble predictions to CSV
output_csv_path = os.path.join(ENSEMBLE_DIR, f"{TICKER}_ensemble_predictions.csv")
ensemble_results.to_csv(output_csv_path, index=False)

print(f" Saved ensemble predictions to: {output_csv_path}")

# Save all meta-model results for reference
all_results_info = {
    "ticker": TICKER,
    "best_model": best_model_name,
    "best_cv_rmse": float(best_cv_error),
    "all_models_tested": results,
    "ensemble_metrics": {
        "rmse": float(ensemble_rmse),
        "n_samples": int(len(meta_y))
    }
}

output_results_path = os.path.join(ENSEMBLE_DIR, f"{TICKER}_ensemble_all_results.json")

with open(output_results_path, 'w') as f:
    json.dump(all_results_info, f, indent=2)

print(f"✓ Saved all results to: {output_results_path}")


#Create Summary Details

print("\n" + "="*80)
print("TASK 4.4 COMPLETE")
print("="*80)
print(f"\nSummary:")
print(f"  Ticker: {TICKER}")
print(f"  Samples: {len(meta_y)}")
print(f"  Best Meta-Model: {best_model_name}")
print(f"  CV RMSE: {best_cv_error:.6f}")
print(f"  Full Data RMSE: {ensemble_rmse:.6f}")
print(f"\nOutput Files:")
print(f"  1. {output_json_path}")
print(f"  2. {output_csv_path}")
print(f"  3. {output_results_path}")
print("="*80 + "\n")