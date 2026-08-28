"""
predict.py
----------
Inference module for the Student Performance Classification project.

Loads the saved Logistic Regression pipeline from Phase 3 and provides
a clean function to predict a student's Performance_Category from their
four academic inputs.

WHY A DATAFRAME IS CONSTRUCTED:
    The saved pipeline expects a pandas DataFrame with exact column names
    matching the training data. Constructing the DataFrame explicitly
    prevents feature-order mismatches that would silently produce
    incorrect predictions.

WHY MANUAL SCALING IS NOT REQUIRED:
    The saved pipeline already contains a SimpleImputer and StandardScaler
    as internal steps. When predict() or predict_proba() is called, the
    pipeline automatically applies all preprocessing before classification.

WHY predict_proba() IS USED:
    It returns the model's estimated class probabilities according to
    the trained Logistic Regression model. These are NOT certainty or
    confidence guarantees -- they represent model-estimated probabilities.

WHY model.classes_ IS USED:
    The column order of predict_proba() output depends on the alphabetical
    sort order of the target labels seen during training. Using
    pipeline.classes_ maps each probability column to its correct class
    label, preventing incorrect probability attribution.

Author: Raghav (Core ML Lead)
Integration Note for Aabiya:
    Import `predict_performance` directly in app.py to power the UI.
    Example: from src.predict import predict_performance
"""

import os
import joblib
import pandas as pd

# ---- Constants ----
# These are the exact feature names and order used during model training.
# They MUST match the training DataFrame columns exactly.
FEATURE_NAMES = ['MST_Score', 'Quiz_Score', 'Attendance_Percent', 'Assignment_Score']

# The three valid output classes from the trained model.
VALID_CLASSES = {'Average Performer', 'High Performer', 'Needs Improvement'}

# Default path to the saved model pipeline from Phase 3.
DEFAULT_MODEL_PATH = os.path.join('models', 'best_student_model.joblib')

# Valid range for all four input features.
INPUT_MIN = 0.0
INPUT_MAX = 100.0


# ---- Model Loading ----

def load_model(model_path: str = DEFAULT_MODEL_PATH):
    """
    Loads the saved scikit-learn pipeline from disk using joblib.

    Parameters:
        model_path (str): Path to the .joblib file.

    Returns:
        sklearn.pipeline.Pipeline: The loaded, fitted pipeline ready
                                   for prediction.

    Raises:
        FileNotFoundError: If the model file does not exist.
        RuntimeError: If the loaded object is not a valid sklearn Pipeline,
                      or is missing expected components.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Trained model not found at '{model_path}'. "
            "The Phase 3 pipeline must be trained and saved before inference."
        )

    pipeline = joblib.load(model_path)

    # Verify it is a Pipeline with the expected structure
    from sklearn.pipeline import Pipeline as SklearnPipeline
    if not isinstance(pipeline, SklearnPipeline):
        raise RuntimeError(
            f"Loaded object is of type '{type(pipeline).__name__}', "
            "expected a scikit-learn Pipeline."
        )

    step_names = [name for name, _ in pipeline.steps]
    if 'classifier' not in step_names:
        raise RuntimeError(
            f"Pipeline is missing a 'classifier' step. "
            f"Found steps: {step_names}"
        )

    # Verify predict and predict_proba are available
    if not hasattr(pipeline, 'predict'):
        raise RuntimeError("Pipeline does not support predict().")

    return pipeline


# ---- Input Validation ----

def validate_inputs(mst_score, quiz_score, attendance_percent, assignment_score):
    """
    Validates that all four inputs are numeric and within the range [0, 100].

    Parameters:
        mst_score: MST exam score
        quiz_score: Quiz score
        attendance_percent: Attendance percentage
        assignment_score: Assignment score

    Returns:
        tuple: (float, float, float, float) validated values

    Raises:
        TypeError: If any input is not a numeric type (int or float).
        ValueError: If any input is outside the [0, 100] range.
    """
    inputs = {
        'MST_Score': mst_score,
        'Quiz_Score': quiz_score,
        'Attendance_Percent': attendance_percent,
        'Assignment_Score': assignment_score,
    }

    validated = {}
    errors = []

    for name, value in inputs.items():
        # Check for None / missing
        if value is None:
            errors.append(f"{name}: value is missing (None).")
            continue

        # Check numeric type (reject strings, bools, etc.)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            errors.append(
                f"{name}: expected a numeric value (int or float), "
                f"got {type(value).__name__} = {value!r}."
            )
            continue

        # Convert to float for consistency
        value = float(value)

        # Check range
        if value < INPUT_MIN or value > INPUT_MAX:
            errors.append(
                f"{name}: value {value} is out of valid range "
                f"[{INPUT_MIN}, {INPUT_MAX}]."
            )
            continue

        validated[name] = value

    if errors:
        error_msg = "Input validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        raise ValueError(error_msg)

    return (
        validated['MST_Score'],
        validated['Quiz_Score'],
        validated['Attendance_Percent'],
        validated['Assignment_Score'],
    )


# ---- Single Student Prediction ----

def predict_performance(
    mst_score,
    quiz_score,
    attendance_percent,
    assignment_score,
    model_path: str = DEFAULT_MODEL_PATH
) -> dict:
    """
    Predicts the Performance_Category for a single student.

    This is the PRIMARY INFERENCE INTERFACE for the project.
    The future Python UI should call this function directly.

    Steps performed internally:
        1. Validates all four inputs (numeric, within [0, 100]).
        2. Constructs a one-row pandas DataFrame with exact feature names.
        3. Loads the saved pipeline (Imputer -> Scaler -> LogisticRegression).
        4. Calls pipeline.predict() for the predicted class.
        5. Calls pipeline.predict_proba() for class probability estimates.
        6. Maps probabilities to class labels using pipeline.classes_.

    Parameters:
        mst_score (int or float): MST exam score (0-100)
        quiz_score (int or float): Quiz score (0-100)
        attendance_percent (int or float): Attendance percentage (0-100)
        assignment_score (int or float): Assignment score (0-100)
        model_path (str): Path to saved .joblib pipeline

    Returns:
        dict: {
            "predicted_category": str,
            "probabilities": {
                "Average Performer": float,
                "High Performer": float,
                "Needs Improvement": float
            },
            "predicted_probability": float
        }

    Raises:
        TypeError: If inputs are not numeric.
        ValueError: If inputs are out of range [0, 100].
        FileNotFoundError: If the model file is missing.
        RuntimeError: If the loaded pipeline is invalid.
    """
    # Step 1: Validate inputs
    mst_score, quiz_score, attendance_percent, assignment_score = validate_inputs(
        mst_score, quiz_score, attendance_percent, assignment_score
    )

    # Step 2: Construct a one-row DataFrame with explicit column names.
    # Using exact feature names ensures the pipeline receives columns in the
    # order it was trained on, preventing silent feature-order mismatches.
    input_df = pd.DataFrame(
        [[mst_score, quiz_score, attendance_percent, assignment_score]],
        columns=FEATURE_NAMES
    )

    # Step 3: Load the saved pipeline
    pipeline = load_model(model_path)

    # Step 4: Predict the category
    predicted_category = pipeline.predict(input_df)[0]

    # Step 5 & 6: Get class probabilities and map to labels
    # pipeline.classes_ gives the exact label order for predict_proba() columns.
    probabilities = {}
    predicted_probability = None

    if hasattr(pipeline, 'predict_proba'):
        proba_array = pipeline.predict_proba(input_df)[0]
        class_labels = pipeline.classes_

        for label, prob in zip(class_labels, proba_array):
            probabilities[label] = round(float(prob), 4)

        predicted_probability = probabilities.get(predicted_category)

    result = {
        "predicted_category": predicted_category,
        "probabilities": probabilities,
        "predicted_probability": predicted_probability,
    }

    return result


# ---- Batch Prediction Helper ----

def predict_batch(
    df: pd.DataFrame,
    model_path: str = DEFAULT_MODEL_PATH
) -> pd.DataFrame:
    """
    Predicts Performance_Category for multiple students at once.

    This is a convenience helper. The primary interface remains
    predict_performance() for single-student prediction.

    Parameters:
        df (pd.DataFrame): Must contain exactly the four feature columns:
                           MST_Score, Quiz_Score, Attendance_Percent,
                           Assignment_Score.
        model_path (str): Path to saved pipeline.

    Returns:
        pd.DataFrame: Original DataFrame with added columns:
                      - Predicted_Category
                      - Prob_Average_Performer
                      - Prob_High_Performer
                      - Prob_Needs_Improvement

    Raises:
        ValueError: If required columns are missing or values are invalid.
    """
    # Validate required columns
    missing = [c for c in FEATURE_NAMES if c not in df.columns]
    if missing:
        raise ValueError(
            f"Input DataFrame is missing required feature columns: {missing}. "
            f"Required: {FEATURE_NAMES}"
        )

    # Validate all values are numeric and in range
    feature_df = df[FEATURE_NAMES].copy()
    for col in FEATURE_NAMES:
        if not pd.api.types.is_numeric_dtype(feature_df[col]):
            raise ValueError(f"Column '{col}' contains non-numeric values.")
        if feature_df[col].min() < INPUT_MIN or feature_df[col].max() > INPUT_MAX:
            raise ValueError(
                f"Column '{col}' contains values outside [{INPUT_MIN}, {INPUT_MAX}]."
            )

    # Load model and predict
    pipeline = load_model(model_path)
    predictions = pipeline.predict(feature_df)

    result_df = df.copy()
    result_df['Predicted_Category'] = predictions

    if hasattr(pipeline, 'predict_proba'):
        proba = pipeline.predict_proba(feature_df)
        class_labels = pipeline.classes_
        for i, label in enumerate(class_labels):
            result_df[f'Prob_{label.replace(" ", "_")}'] = proba[:, i].round(4)

    return result_df


# ---- Main (Quick Smoke Test) ----

if __name__ == "__main__":
    print("=" * 60)
    print("predict.py - Quick Smoke Test")
    print("=" * 60)

    result = predict_performance(
        mst_score=70.0,
        quiz_score=65.0,
        attendance_percent=80.0,
        assignment_score=72.0
    )

    print(f"\nInput: MST=70, Quiz=65, Attendance=80, Assignment=72")
    print(f"Predicted Category: {result['predicted_category']}")
    print(f"Predicted Probability: {result['predicted_probability']}")
    print(f"All Probabilities: {result['probabilities']}")
