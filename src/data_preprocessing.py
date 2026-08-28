"""
data_preprocessing.py
---------------------
Responsible for loading student performance data, separating features/target,
handling exclusions (Student_ID, Performance_Score), and building
scikit-learn preprocessing pipelines to prevent data leakage.

Author: Raghav (Core ML Lead)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from typing import Tuple, List, Optional


# Verified Schema Constants for Student Performance Dataset
DEFAULT_FEATURE_COLS = [
    'MST_Score',
    'Quiz_Score',
    'Attendance_Percent',
    'Assignment_Score'
]
DEFAULT_TARGET_COL = 'Performance_Category'
DEFAULT_EXCLUDED_COLS = ['Student_ID', 'Performance_Score']


def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Loads student dataset from CSV or Excel file without modifying raw data.
    
    Parameters:
        file_path (str): Path to data file (e.g., 'data/student_performance_data.csv')
        
    Returns:
        pd.DataFrame: Loaded raw dataset
    """
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format. Please provide a .csv or .xlsx file.")
    
    print(f"[INFO] Dataset loaded successfully from '{file_path}' ({df.shape[0]} rows, {df.shape[1]} columns).")
    return df


def extract_features_and_target(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    target_col: str = DEFAULT_TARGET_COL,
    excluded_cols: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Extracts ML feature matrix (X) and target vector (y) while explicitly
    excluding identifier and target leakage columns.
    
    Parameters:
        df (pd.DataFrame): Raw dataframe
        feature_cols (list, optional): Specific features to include (defaults to 4 academic features)
        target_col (str): Target column name ('Performance_Category')
        excluded_cols (list, optional): Columns to exclude ('Student_ID', 'Performance_Score')
        
    Returns:
        tuple: (X: pd.DataFrame, y: pd.Series)
    """
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in dataset columns: {list(df.columns)}")
    
    if feature_cols is None:
        feature_cols = DEFAULT_FEATURE_COLS
    
    if excluded_cols is None:
        excluded_cols = DEFAULT_EXCLUDED_COLS
        
    # Verify feature columns exist in dataframe
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        raise KeyError(f"Specified feature columns not found in dataset: {missing_cols}")
        
    # Extract X and y
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    print(f"[INFO] Features extracted: {feature_cols}")
    print(f"[INFO] Excluded columns (Identifier / Leakage): {excluded_cols}")
    print(f"[INFO] Target extracted: '{target_col}' with {y.nunique()} unique classes.")
    
    return X, y


def split_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.20,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits feature matrix X and target y into training and testing sets.
    Uses stratification to preserve the class imbalance ratio in both splits.
    
    Parameters:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target labels
        test_size (float): Proportion for test partition (default: 0.20 = 20%)
        random_state (int): Seed for exact reproducibility (default: 42)
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    print(f"[INFO] Stratified Train/Test Split completed:")
    print(f"       Training Set: {X_train.shape[0]} samples ({100*(1-test_size):.0f}%)")
    print(f"       Testing Set:  {X_test.shape[0]} samples ({100*test_size:.0f}%)")
    
    return X_train, X_test, y_train, y_test


def create_preprocessing_pipeline() -> Pipeline:
    """
    Builds a leakage-free Scikit-Learn Preprocessing Pipeline.
    
    Chains:
    1. SimpleImputer(strategy='median'): Imputes missing values using training medians.
    2. StandardScaler(): Normalizes features to mean=0, std=1 (useful for Logistic Regression).
    
    Data Leakage Protection:
    The pipeline calculates medians and scaling parameters STRICTLY from X_train
    during `fit()`, and reapplies them unchanged to X_test during `transform()`.
    
    Returns:
        Pipeline: Scikit-learn preprocessing pipeline
    """
    return Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
