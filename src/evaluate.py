"""
evaluate.py
-----------
Evaluates trained classification models on the held-out test set.
Computes Accuracy, Precision, Recall, F1-Score (Macro & Weighted),
generates confusion matrices, classification reports, and comparison plots.

Author: Raghav (Core ML Lead)
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def evaluate_on_test(pipeline, X_test, y_test, model_name: str) -> dict:
    """
    Evaluates a single fitted pipeline on the held-out test set.
    This function should be called ONLY ONCE for the selected best model.

    Parameters:
        pipeline: Fitted sklearn Pipeline
        X_test: Test feature matrix (unseen during training & CV)
        y_test: True test labels
        model_name (str): Name of the model

    Returns:
        dict: Dictionary of test evaluation metrics
    """
    y_pred = pipeline.predict(X_test)

    metrics = {
        'Model': model_name,
        'Test_Accuracy': accuracy_score(y_test, y_pred),
        'Test_Precision_Macro': precision_score(y_test, y_pred, average='macro', zero_division=0),
        'Test_Recall_Macro': recall_score(y_test, y_pred, average='macro', zero_division=0),
        'Test_F1_Macro': f1_score(y_test, y_pred, average='macro', zero_division=0),
        'Test_Precision_Weighted': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'Test_Recall_Weighted': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'Test_F1_Weighted': f1_score(y_test, y_pred, average='weighted', zero_division=0),
    }

    print(f"\n=== Final Test Evaluation: {model_name} ===")
    print(f"Accuracy:           {metrics['Test_Accuracy']:.4f}")
    print(f"Macro Precision:    {metrics['Test_Precision_Macro']:.4f}")
    print(f"Macro Recall:       {metrics['Test_Recall_Macro']:.4f}")
    print(f"Macro F1-Score:     {metrics['Test_F1_Macro']:.4f}")
    print(f"Weighted F1-Score:  {metrics['Test_F1_Weighted']:.4f}")

    print(f"\nDetailed Classification Report ({model_name}):")
    report = classification_report(y_test, y_pred, zero_division=0)
    print(report)

    return metrics, y_pred, report


def plot_confusion_matrix(
    y_test, y_pred,
    class_labels: list,
    model_name: str,
    save_path: str
):
    """
    Generates and saves a clear, labelled confusion matrix heatmap.

    Parameters:
        y_test: True test labels
        y_pred: Predicted labels
        class_labels (list): Ordered class label names
        model_name (str): Model name for title
        save_path (str): Absolute path to save the figure
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cm = confusion_matrix(y_test, y_pred, labels=class_labels)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_labels, yticklabels=class_labels,
        linewidths=1, linecolor='white', square=True, ax=ax,
        annot_kws={'size': 14, 'fontweight': 'bold'}
    )
    ax.set_title(f'Confusion Matrix — {model_name}\n(Evaluated on Held-Out Test Set, N={len(y_test)})',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Predicted Category', fontsize=12, labelpad=10)
    ax.set_ylabel('Actual Category', fontsize=12, labelpad=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[INFO] Confusion matrix saved to: {save_path}")


def plot_cv_macro_f1_comparison(
    cv_comparison_df: pd.DataFrame,
    save_path: str
):
    """
    Creates a bar chart comparing the 5-fold Stratified CV Macro F1-Score
    across all three models (with error bars showing standard deviation).

    Parameters:
        cv_comparison_df (pd.DataFrame): DataFrame with CV results
        save_path (str): Path to save the comparison figure
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ['#3498db', '#2ecc71', '#e67e22']

    bars = ax.bar(
        cv_comparison_df['Model'],
        cv_comparison_df['CV_Macro_F1_Mean'],
        yerr=cv_comparison_df['CV_Macro_F1_Std'],
        color=colors[:len(cv_comparison_df)],
        edgecolor='black', width=0.5,
        capsize=8, error_kw={'linewidth': 2}
    )

    for bar, mean_val, std_val in zip(bars, cv_comparison_df['CV_Macro_F1_Mean'], cv_comparison_df['CV_Macro_F1_Std']):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + std_val + 0.008,
                f'{mean_val:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_title('Cross-Validation Macro F1-Score Comparison\n(5-Fold Stratified CV on Training Data Only)',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Classification Model', fontsize=12, labelpad=10)
    ax.set_ylabel('CV Macro F1-Score (Mean ± Std)', fontsize=12, labelpad=10)
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[INFO] CV Macro F1 comparison chart saved to: {save_path}")


def save_model(pipeline, save_path: str):
    """
    Serializes the final winning pipeline to disk using joblib.

    Parameters:
        pipeline: Fitted sklearn Pipeline
        save_path (str): Path to save .joblib file
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(pipeline, save_path)
    print(f"[INFO] Model pipeline saved to: {save_path}")

    # Verification: load it back
    loaded = joblib.load(save_path)
    print(f"[INFO] Model loaded back successfully. Type: {type(loaded).__name__}")
    return loaded
