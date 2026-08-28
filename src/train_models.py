"""
train_models.py
---------------
Defines, trains, and cross-validates three classical ML classification
pipelines for predicting student Performance_Category.

Pipeline Architecture:
- Logistic Regression: SimpleImputer(median) → StandardScaler → LogisticRegression
- Decision Tree:       SimpleImputer(median) → DecisionTreeClassifier
- Random Forest:       SimpleImputer(median) → RandomForestClassifier

Tree-based models do NOT receive StandardScaler because their split
decisions depend on feature rank ordering, not distance metrics.

Author: Raghav (Core ML Lead)
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate


def build_model_pipelines(random_state: int = 42) -> dict:
    """
    Creates three end-to-end pipelines with leakage-free preprocessing
    and class-balanced classifiers.

    Logistic Regression receives StandardScaler because gradient-based
    optimization and regularization penalties assume commensurate feature
    magnitudes. Tree-based models skip scaling because split criteria
    are invariant to monotonic feature transformations.

    Parameters:
        random_state (int): Seed for reproducible training.

    Returns:
        dict: {model_name: sklearn Pipeline}
    """
    pipelines = {
        "Logistic Regression": Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(
                solver='lbfgs',
                max_iter=1000,
                class_weight='balanced',
                random_state=random_state
            ))
        ]),

        "Decision Tree": Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('classifier', DecisionTreeClassifier(
                max_depth=5,
                class_weight='balanced',
                random_state=random_state
            ))
        ]),

        "Random Forest": Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('classifier', RandomForestClassifier(
                n_estimators=100,
                max_depth=6,
                class_weight='balanced',
                random_state=random_state
            ))
        ])
    }
    return pipelines


def cross_validate_models(
    pipelines: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_splits: int = 5,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Performs 5-fold Stratified Cross-Validation on the TRAINING partition
    for each pipeline using multiple scoring metrics.

    The test set is NOT used during this evaluation step.

    Parameters:
        pipelines (dict): {model_name: unfitted Pipeline}
        X_train (pd.DataFrame): Training feature matrix
        y_train (pd.Series): Training target labels
        n_splits (int): Number of CV folds (default: 5)
        random_state (int): Seed for fold generation

    Returns:
        pd.DataFrame: CV comparison table with means and std deviations
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    scoring = {
        'accuracy': 'accuracy',
        'precision_macro': 'precision_macro',
        'recall_macro': 'recall_macro',
        'f1_macro': 'f1_macro'
    }

    results = []
    cv_details = {}

    for name, pipeline in pipelines.items():
        print(f"[CV] Running 5-fold Stratified CV for {name}...")
        cv_result = cross_validate(
            pipeline, X_train, y_train,
            cv=cv,
            scoring=scoring,
            return_train_score=False
        )

        row = {
            'Model': name,
            'CV_Accuracy_Mean': np.mean(cv_result['test_accuracy']),
            'CV_Accuracy_Std': np.std(cv_result['test_accuracy']),
            'CV_Macro_Precision_Mean': np.mean(cv_result['test_precision_macro']),
            'CV_Macro_Recall_Mean': np.mean(cv_result['test_recall_macro']),
            'CV_Macro_F1_Mean': np.mean(cv_result['test_f1_macro']),
            'CV_Macro_F1_Std': np.std(cv_result['test_f1_macro']),
        }
        results.append(row)
        cv_details[name] = cv_result
        print(f"       Macro F1 = {row['CV_Macro_F1_Mean']:.4f} ± {row['CV_Macro_F1_Std']:.4f}")

    comparison_df = pd.DataFrame(results)
    return comparison_df, cv_details


def train_final_model(pipeline, X_train, y_train, model_name: str):
    """
    Fits the selected model pipeline on the FULL training partition
    after cross-validation has determined it to be the best candidate.

    Parameters:
        pipeline: Unfitted or previously fitted sklearn Pipeline
        X_train: Full training feature matrix (80% of data)
        y_train: Full training target labels
        model_name (str): Name of the model

    Returns:
        Pipeline: Fitted pipeline ready for test evaluation
    """
    print(f"\n[TRAIN] Fitting final {model_name} pipeline on full training data ({X_train.shape[0]} samples)...")
    pipeline.fit(X_train, y_train)
    print(f"[TRAIN] {model_name} final training completed.")
    return pipeline
