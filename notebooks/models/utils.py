import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, matthews_corrcoef
)
import json

def calculate_metrics(y_true, y_pred, output_file='metrics.json', rebalanced=False):
    if rebalanced:
        output_file = './results/rebalanced/' + output_file
    else:
        output_file = './results/original/' + output_file
    # Calculate various metrics
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred)
    metrics['recall'] = recall_score(y_true, y_pred)
    metrics['f1_score'] = f1_score(y_true, y_pred)
    
    # Confusion matrix and derived metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['confusion_matrix'] = {'TN': int(tn), 'FP': int(fp), 'FN': int(fn), 'TP': int(tp)}
    metrics['specificity'] = tn / (tn + fp)
    metrics['balanced_accuracy'] = (metrics['recall'] + metrics['specificity']) / 2
    
    # Matthews Correlation Coefficient
    metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
    
    # Save metrics to a JSON file
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"Metrics saved to {output_file}")

# Example usage
if __name__ == "__main__":
    # Sample true labels and predicted probabilities/predictions for demonstration
    y_true = np.array([0, 1, 0, 1, 0, 1, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 0, 1, 0, 0, 1, 0, 1, 0])
    
    calculate_metrics(y_true, y_pred, output_file='metrics.json')