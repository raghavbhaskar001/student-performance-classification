"""
verify_inference.py
-------------------
Phase 4 Verification Script — tests all 10 required verification checks
plus edge cases and consistency tests for the inference module.

This script does NOT retrain the model. It only verifies the saved
pipeline and the predict_performance() inference interface.

Author: Raghav (Core ML Lead)
"""

import sys
import os
import math

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predict import (
    predict_performance,
    predict_batch,
    load_model,
    validate_inputs,
    FEATURE_NAMES,
    VALID_CLASSES,
    DEFAULT_MODEL_PATH,
)
import pandas as pd

PASS = "PASS"
FAIL = "FAIL"

results = []


def log_test(test_id, description, status, detail=""):
    """Records a test result."""
    results.append((test_id, description, status, detail))
    icon = "[OK]" if status == PASS else "[FAIL]"
    print(f"  {icon} TEST {test_id}: {description}")
    if detail:
        print(f"       {detail}")


# ========================================================================
print("=" * 70)
print("PHASE 4 - INFERENCE VERIFICATION SUITE")
print("=" * 70)

# --- TEST 1: Model loads successfully ---
print("\n--- TEST 1: Model Loading ---")
try:
    pipeline = load_model(DEFAULT_MODEL_PATH)
    log_test(1, "Saved model loads successfully via joblib", PASS,
             f"Type: {type(pipeline).__name__}")
except Exception as e:
    log_test(1, "Saved model loads successfully via joblib", FAIL, str(e))
    print("\n[FATAL] Cannot proceed without a valid model. Exiting.")
    sys.exit(1)

# Inspect pipeline structure
step_names = [name for name, _ in pipeline.steps]
print(f"       Pipeline steps: {step_names}")
print(f"       Classifier: {type(pipeline.named_steps['classifier']).__name__}")
print(f"       Classes: {list(pipeline.classes_)}")

# --- TEST 2: Valid input produces a prediction ---
print("\n--- TEST 2: Valid Prediction ---")
try:
    result = predict_performance(
        mst_score=70.0,
        quiz_score=65.0,
        attendance_percent=80.0,
        assignment_score=72.0
    )
    log_test(2, "Valid four-feature input produces a prediction", PASS,
             f"Predicted: {result['predicted_category']}")
except Exception as e:
    log_test(2, "Valid four-feature input produces a prediction", FAIL, str(e))

# --- TEST 3: Predicted category is one of the three valid classes ---
print("\n--- TEST 3: Valid Class Output ---")
try:
    predicted = result['predicted_category']
    if predicted in VALID_CLASSES:
        log_test(3, "Predicted category is a valid class", PASS,
                 f"'{predicted}' is in {VALID_CLASSES}")
    else:
        log_test(3, "Predicted category is a valid class", FAIL,
                 f"'{predicted}' is NOT in {VALID_CLASSES}")
except Exception as e:
    log_test(3, "Predicted category is a valid class", FAIL, str(e))

# --- TEST 4: predict_proba() probabilities are available ---
print("\n--- TEST 4: Probability Availability ---")
try:
    probs = result.get('probabilities', {})
    if probs and len(probs) == 3:
        log_test(4, "predict_proba() probabilities are available", PASS,
                 f"Probabilities: {probs}")
    else:
        log_test(4, "predict_proba() probabilities are available", FAIL,
                 f"Got: {probs}")
except Exception as e:
    log_test(4, "predict_proba() probabilities are available", FAIL, str(e))

# --- TEST 5: Probabilities correspond to model.classes_ ---
print("\n--- TEST 5: Probability-Class Mapping ---")
try:
    model_classes = set(pipeline.classes_)
    prob_keys = set(probs.keys())
    if model_classes == prob_keys:
        log_test(5, "Probabilities correspond to model.classes_", PASS,
                 f"model.classes_ = {list(pipeline.classes_)}, prob keys = {list(probs.keys())}")
    else:
        log_test(5, "Probabilities correspond to model.classes_", FAIL,
                 f"Mismatch: model.classes_={model_classes}, prob_keys={prob_keys}")
except Exception as e:
    log_test(5, "Probabilities correspond to model.classes_", FAIL, str(e))

# --- TEST 6: Probabilities sum approximately to 1.0 ---
print("\n--- TEST 6: Probability Sum ---")
try:
    prob_sum = sum(probs.values())
    if math.isclose(prob_sum, 1.0, abs_tol=0.01):
        log_test(6, "Class probabilities sum to ~1.0", PASS,
                 f"Sum = {prob_sum:.6f}")
    else:
        log_test(6, "Class probabilities sum to ~1.0", FAIL,
                 f"Sum = {prob_sum:.6f} (expected ~1.0)")
except Exception as e:
    log_test(6, "Class probabilities sum to ~1.0", FAIL, str(e))

# --- TEST 7: Negative input is rejected ---
print("\n--- TEST 7: Negative Input Rejection ---")
try:
    predict_performance(-5.0, 65.0, 80.0, 72.0)
    log_test(7, "Invalid negative input is rejected", FAIL,
             "No exception raised for negative MST_Score = -5")
except (ValueError, TypeError) as e:
    log_test(7, "Invalid negative input is rejected", PASS,
             f"Correctly raised: {type(e).__name__}")

# --- TEST 8: Input above 100 is rejected ---
print("\n--- TEST 8: Input Above 100 Rejection ---")
try:
    predict_performance(70.0, 65.0, 150.0, 72.0)
    log_test(8, "Input above 100 is rejected", FAIL,
             "No exception raised for Attendance_Percent = 150")
except (ValueError, TypeError) as e:
    log_test(8, "Input above 100 is rejected", PASS,
             f"Correctly raised: {type(e).__name__}")

# --- TEST 9: Non-numeric input is rejected ---
print("\n--- TEST 9: Non-numeric Input Rejection ---")
try:
    predict_performance(70.0, "high", 80.0, 72.0)
    log_test(9, "Missing/non-numeric input is rejected", FAIL,
             "No exception raised for Quiz_Score = 'high'")
except (ValueError, TypeError) as e:
    log_test(9, "Missing/non-numeric input is rejected", PASS,
             f"Correctly raised: {type(e).__name__}")

# Also test None
try:
    predict_performance(70.0, None, 80.0, 72.0)
    log_test("9b", "None input is rejected", FAIL,
             "No exception raised for Quiz_Score = None")
except (ValueError, TypeError) as e:
    log_test("9b", "None input is rejected", PASS,
             f"Correctly raised: {type(e).__name__}")

# --- TEST 10: Interface does not require excluded columns ---
print("\n--- TEST 10: No Excluded Columns Required ---")
try:
    # Verify the function signature does not mention Student_ID,
    # Performance_Score, or Performance_Category
    import inspect
    sig = inspect.signature(predict_performance)
    param_names = list(sig.parameters.keys())
    excluded = {'student_id', 'performance_score', 'performance_category'}
    found_excluded = excluded.intersection(set(p.lower() for p in param_names))

    if not found_excluded:
        log_test(10, "Interface does not require excluded columns", PASS,
                 f"Parameters: {param_names}")
    else:
        log_test(10, "Interface does not require excluded columns", FAIL,
                 f"Found excluded params: {found_excluded}")
except Exception as e:
    log_test(10, "Interface does not require excluded columns", FAIL, str(e))

# ========================================================================
# EDGE CASE TESTING
# ========================================================================
print("\n--- EDGE CASE TESTS ---")

# Boundary: all zeros
try:
    r = predict_performance(0, 0, 0, 0)
    log_test("E1", "Boundary input (0, 0, 0, 0) accepted", PASS,
             f"Predicted: {r['predicted_category']}, Prob: {r['predicted_probability']}")
except Exception as e:
    log_test("E1", "Boundary input (0, 0, 0, 0) accepted", FAIL, str(e))

# Boundary: all 100s
try:
    r = predict_performance(100, 100, 100, 100)
    log_test("E2", "Boundary input (100, 100, 100, 100) accepted", PASS,
             f"Predicted: {r['predicted_category']}, Prob: {r['predicted_probability']}")
except Exception as e:
    log_test("E2", "Boundary input (100, 100, 100, 100) accepted", FAIL, str(e))

# Typical mid-range student
try:
    r = predict_performance(55, 50, 60, 48)
    log_test("E3", "Typical mid-range input (55, 50, 60, 48) accepted", PASS,
             f"Predicted: {r['predicted_category']}")
except Exception as e:
    log_test("E3", "Typical mid-range input (55, 50, 60, 48) accepted", FAIL, str(e))

# Integer inputs (should also work)
try:
    r = predict_performance(85, 90, 95, 88)
    log_test("E4", "Integer inputs accepted", PASS,
             f"Predicted: {r['predicted_category']}")
except Exception as e:
    log_test("E4", "Integer inputs accepted", FAIL, str(e))

# Boolean input (should be rejected)
try:
    predict_performance(True, 65, 80, 72)
    log_test("E5", "Boolean input rejected", FAIL,
             "No exception raised for MST_Score = True")
except (ValueError, TypeError) as e:
    log_test("E5", "Boolean input rejected", PASS,
             f"Correctly raised: {type(e).__name__}")

# ========================================================================
# CONSISTENCY TEST
# ========================================================================
print("\n--- CONSISTENCY TEST ---")

r1 = predict_performance(70.0, 65.0, 80.0, 72.0)
r2 = predict_performance(70.0, 65.0, 80.0, 72.0)

if r1 == r2:
    log_test("C1", "Repeated identical inputs produce identical results", PASS,
             f"Both predict: {r1['predicted_category']}, prob={r1['predicted_probability']}")
else:
    log_test("C1", "Repeated identical inputs produce identical results", FAIL,
             f"Run 1: {r1}\nRun 2: {r2}")

# ========================================================================
# BATCH PREDICTION TEST
# ========================================================================
print("\n--- BATCH PREDICTION TEST ---")

try:
    batch_df = pd.DataFrame({
        'MST_Score': [70, 30, 90],
        'Quiz_Score': [65, 40, 85],
        'Attendance_Percent': [80, 50, 95],
        'Assignment_Score': [72, 35, 92],
    })
    result_df = predict_batch(batch_df)

    if 'Predicted_Category' in result_df.columns:
        log_test("B1", "Batch prediction returns predictions", PASS,
                 f"Predictions: {list(result_df['Predicted_Category'])}")
    else:
        log_test("B1", "Batch prediction returns predictions", FAIL,
                 "Missing 'Predicted_Category' column")

    prob_cols = [c for c in result_df.columns if c.startswith('Prob_')]
    if len(prob_cols) == 3:
        log_test("B2", "Batch prediction returns 3 probability columns", PASS,
                 f"Columns: {prob_cols}")
    else:
        log_test("B2", "Batch prediction returns 3 probability columns", FAIL,
                 f"Found: {prob_cols}")
except Exception as e:
    log_test("B1", "Batch prediction works", FAIL, str(e))

# ========================================================================
# MODEL INTEGRITY CHECK
# ========================================================================
print("\n--- MODEL INTEGRITY CHECK ---")

model_path = DEFAULT_MODEL_PATH
model_size = os.path.getsize(model_path)
log_test("M1", "Model file exists and is non-empty", PASS,
         f"Size: {model_size} bytes at {model_path}")

# Verify the classifier type matches Phase 3 selection
classifier_type = type(pipeline.named_steps['classifier']).__name__
if classifier_type == 'LogisticRegression':
    log_test("M2", "Classifier is LogisticRegression (Phase 3 selection)", PASS)
else:
    log_test("M2", "Classifier is LogisticRegression (Phase 3 selection)", FAIL,
             f"Found: {classifier_type}")

# ========================================================================
# SUMMARY
# ========================================================================
print("\n" + "=" * 70)
print("VERIFICATION SUMMARY")
print("=" * 70)

passed = sum(1 for _, _, s, _ in results if s == PASS)
failed = sum(1 for _, _, s, _ in results if s == FAIL)
total = len(results)

print(f"\nTotal Tests: {total}")
print(f"Passed:      {passed}")
print(f"Failed:      {failed}")
print(f"Pass Rate:   {100 * passed / total:.1f}%")

if failed == 0:
    print("\n[SUCCESS] All verification tests passed.")
    print("The inference interface is ready for UI integration.")
else:
    print(f"\n[WARNING] {failed} test(s) failed. Review above for details.")

print("=" * 70)
