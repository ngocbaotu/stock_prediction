import numpy as np
import pandas as pd
import gc
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, precision_score, recall_score, f1_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import os
import xgboost as xgb
from sklearn.metrics import root_mean_squared_error
#Import functions from model.py
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


#Calculate Metrics
#Hit rate / Root Mean Squared Error / Information Coefficient

def outputMetrics(actual, preds):
    #Calculating Hit Rate
    hit_rate = (np.sign(actual) == np.sign(preds)).mean()
    
    #Calculating Root Mean Squared Error
    rmse = np.sqrt(mean_squared_error(actual, preds))
    
    #Calculating Information Coefficient
    ic = np.corrcoef(preds, actual)[0, 1]
    
    y_true = np.where(actual >= 0, 1, 0)
    y_pred = np.where(preds >= 0, 1, 0)

    # 2. Calculate the Metrics
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"Precision: {precision:.4f} \t Recall: {recall:.4f} \t F1-Score: {f1:.4f}")
    print(f"hr: {hit_rate:.4f} \t\t rmse: {rmse:.4f} \t\t ic: {ic:.4f}")

    return hit_rate, rmse, ic, precision, recall, f1

#hit_rate, rmse, ic = outputMetrics(tests[-1], preds[-1])




#Generating Graphs
#Bar Chart / Line Plots

#Removed. Lacked meaning to compare the metrics of the same model
#Overall Model Performance
""" 
def metricsGraph(hit_rate, ic):
    metrics = ['Hit Rate', 'Information Coefficient']
    values = [hit_rate, ic]
    
    plt.figure(figsize=(6, 4))
    bars = plt.bar(metrics, values, color=['blue', 'red'])
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.01, f'{height:.2f}', 
                 ha='center', va='bottom', fontsize=11)
    plt.axhline(0, color='black', linewidth=2.5)
    plt.title('Performance Metric Comparison', fontsize=14)
    plt.ylim(-1, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    
    plt.show()
 """
#metricsGraph(hit_rate, ic)

def residualsGraph(stock, actual, pred, directory):
    residuals = actual - pred

    #a, b, c, d = np.polyfit(pred, residuals, 3)
    
    m, b = np.polyfit(pred, residuals, 1)
    #line_of_best_fit = np.poly1d([a, b, c, d])
    line_of_best_fit = np.poly1d([m, b])

    plt.figure(figsize=(10, 6))
    plt.scatter(pred, residuals, 
                color='blue', 
                marker='o', 
                alpha=0.6,
                label='residuals')
    
    plt.plot(pred, line_of_best_fit(pred), color='green',
             linestyle='-', linewidth=3, label="Best Fit Line")

    plt.axhline(0, color='red', linestyle='--', linewidth=2)

    plt.title(f"{stock} Residuals Graph", fontsize=16)
    plt.xlabel("Predicted Values", fontsize=14)
    plt.ylabel("Residuals (Actual - Predicted)", fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()
    #plt.show()
    plt.savefig(directory)

    plt.close()

def compareModelGraphs(metric, xgb_in, lstm_in, ens_in, directory):
    models = ["XGB", "LSTM", "Ensemble"]
    values = [xgb_in, lstm_in, ens_in]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, values)

    for bar in bars: 
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height, f'{height:.4f}',
                ha = 'center', va='bottom', fontsize=11)
    plt.axhline(0, color='black', linewidth=2.5)
    plt.title(f"{metric} Comparison", fontsize=14)
    plt.savefig(directory)


#Data Formatting
STOCKS = ['AAPL', 'AMZN', 'GOOGL', 'MSFT', 'TSLA']

#base_dir = Path(__file__).resolve().parent.parent / "graphs"
script_dir = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(script_dir, "..", "graphs")


for stock in STOCKS:
    #temp_dir = base_dir / stock
    temp_dir = os.path.join(PLOTS_DIR, stock)
    os.makedirs(temp_dir, exist_ok=True)
    print("Current stock: " + stock)
    df = pd.read_csv(f"model/ensemble/{stock}_ensemble_predictions.csv")
    xgbPred = df['xgb_pred']
    lstm_pred = df['lstm_pred']
    ensemble_pred = df['ensemble_pred']
    actual = df['actual']
    ensemble_error = df['ensemble_error']

    #XGB
    print("XGB Metrics")
    xgb_hr, xgb_rmse, xgb_ic, xgb_precision, xgb_recall, xgb_f1 = outputMetrics(actual, xgbPred)
    xgb_dir = os.path.join(temp_dir, "XGB_residual_graph.png")
    residualsGraph(stock, actual, xgbPred, xgb_dir)

    print("-----")

    #LSTM
    print("LSTM Metrics")
    lstm_hr, lstm_rmse, lstm_ic, lstm_precision, lstm_recall, lstm_f1 = outputMetrics(actual, lstm_pred)
    lstm_dir = os.path.join(temp_dir, "LSTM_residual_graph.png")
    residualsGraph(stock, actual, lstm_pred, lstm_dir)

    print("-----")
    #Ensemble
    print("Ensemble Metrics")
    ens_hr, ens_rmse, ens_ic, ens_precision, ens_recall, ens_f1 = outputMetrics(actual, ensemble_pred)
    Ens_dir = os.path.join(temp_dir, "Ensemble_residual_graph.png")
    residualsGraph(stock, actual, ensemble_pred, Ens_dir)
   

    #Compare Models by Metric

    #Hit_Rate Metric
    compare_dir = os.path.join(temp_dir, "Hit_Rate_performance_comparison.png")
    compareModelGraphs("Hit Rate", xgb_hr, lstm_hr, ens_hr, compare_dir)

    #RMSE Metric
    compare_dir = os.path.join(temp_dir, "RMSE_performance_comparison.png")
    compareModelGraphs("RMSE", xgb_rmse, lstm_rmse, ens_rmse, compare_dir)

    #F1 Metric
    compare_dir = os.path.join(temp_dir, "F1_performance_comparison.png")
    compareModelGraphs("F1", xgb_f1, lstm_f1, ens_f1, compare_dir)

    print('\n')
