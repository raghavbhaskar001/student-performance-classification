# Phase 4 — Inference Module Documentation

## 1. Purpose

This module provides a clean Python interface for predicting a student's
**Performance_Category** using the trained Logistic Regression pipeline
from Phase 3.

It is designed to be called by a future Python-based UI. The UI developer
does **not** need to perform any manual preprocessing — the saved pipeline
handles imputation and scaling internally.

---

## 2. Saved Model Location

```
models/best_student_model.joblib
```

- **Type**: scikit-learn `Pipeline`
- **Steps**: `SimpleImputer(median)` → `StandardScaler` → `LogisticRegression`
- **Classes**: `Average Performer`, `High Performer`, `Needs Improvement`
- **Trained in**: Phase 3 (unchanged, not retrained)

---

## 3. Required Inputs

The inference function accepts exactly **four numeric inputs**:

| Parameter | Description | Valid Range |
|-----------|-------------|:-----------:|
| `mst_score` | Mid-Semester Test score | 0 – 100 |
| `quiz_score` | Quiz score | 0 – 100 |
| `attendance_percent` | Attendance percentage | 0 – 100 |
| `assignment_score` | Assignment score | 0 – 100 |

**Not accepted as inputs** (by design):
- `Student_ID` — identifier column, not a feature
- `Performance_Score` — target leakage column
- `Performance_Category` — this is the prediction target

---

## 4. Input Validation

All inputs are validated before prediction:

| Validation Rule | Behavior |
|-----------------|----------|
| Input is `None` | Raises `ValueError` |
| Input is non-numeric (string, bool, etc.) | Raises `ValueError` |
| Input is negative (< 0) | Raises `ValueError` |
| Input exceeds 100 | Raises `ValueError` |

Invalid inputs produce clear error messages identifying which field
failed and why. **Invalid values are never silently corrected.**

---

## 5. Prediction Process

When `predict_performance()` is called:

1. **Validate** — All four inputs are checked for type and range.
2. **Construct DataFrame** — A one-row `pandas.DataFrame` is built with
   explicit column names (`MST_Score`, `Quiz_Score`, `Attendance_Percent`,
   `Assignment_Score`) to prevent feature-order mismatches.
3. **Load Pipeline** — The saved `.joblib` file is loaded via `joblib.load()`.
4. **Predict** — `pipeline.predict()` returns the predicted class label.
5. **Probabilities** — `pipeline.predict_proba()` returns class probability
   estimates. These are mapped to class labels using `pipeline.classes_`.

**Why manual scaling is not needed:**
The pipeline contains `SimpleImputer` and `StandardScaler` as internal
steps. Calling `predict()` or `predict_proba()` automatically applies
all preprocessing before passing data to the classifier.

---

## 6. Probability Interpretation

The values returned by `predict_proba()` represent the model's **estimated
class probabilities** according to the trained Logistic Regression model.

- They are **not** certainty or confidence guarantees.
- They are model-estimated probabilities based on learned decision boundaries.
- The three probabilities always sum to approximately 1.0.

The probability mapping uses `pipeline.classes_` to ensure each probability
value is assigned to the correct class label, regardless of internal ordering.

---

## 7. Example Usage

### Single Student Prediction

```python
from src.predict import predict_performance

result = predict_performance(
    mst_score=70.0,
    quiz_score=65.0,
    attendance_percent=80.0,
    assignment_score=72.0
)

print(result)
# {
#     "predicted_category": "Average Performer",
#     "probabilities": {
#         "Average Performer": 0.8784,
#         "High Performer": 0.1216,
#         "Needs Improvement": 0.0
#     },
#     "predicted_probability": 0.8784
# }
```

### Batch Prediction (Optional Helper)

```python
from src.predict import predict_batch
import pandas as pd

df = pd.DataFrame({
    'MST_Score': [70, 30, 90],
    'Quiz_Score': [65, 40, 85],
    'Attendance_Percent': [80, 50, 95],
    'Assignment_Score': [72, 35, 92],
})

result_df = predict_batch(df)
print(result_df[['Predicted_Category', 'Prob_Average_Performer',
                  'Prob_High_Performer', 'Prob_Needs_Improvement']])
```

---

## 8. Verification Results

All 21 verification tests passed (100% pass rate):

| Test | Description | Result |
|------|-------------|:------:|
| 1 | Model loads successfully via joblib | PASS |
| 2 | Valid input produces a prediction | PASS |
| 3 | Predicted category is a valid class | PASS |
| 4 | predict_proba() probabilities available | PASS |
| 5 | Probabilities correspond to model.classes_ | PASS |
| 6 | Probabilities sum to ~1.0 | PASS |
| 7 | Negative input rejected | PASS |
| 8 | Input above 100 rejected | PASS |
| 9 | Non-numeric input rejected | PASS |
| 9b | None input rejected | PASS |
| 10 | No excluded columns required | PASS |
| E1 | Boundary (0,0,0,0) accepted | PASS |
| E2 | Boundary (100,100,100,100) accepted | PASS |
| E3 | Mid-range input accepted | PASS |
| E4 | Integer inputs accepted | PASS |
| E5 | Boolean input rejected | PASS |
| C1 | Consistency: identical inputs → identical results | PASS |
| B1 | Batch prediction returns predictions | PASS |
| B2 | Batch prediction returns 3 probability columns | PASS |
| M1 | Model file exists and is non-empty | PASS |
| M2 | Classifier is LogisticRegression | PASS |

---

## 9. Important Restrictions

- The Phase 3 model (`best_student_model.joblib`) must **not** be
  overwritten or retrained without explicit approval.
- The inference module does **not** modify the raw dataset.
- The module does **not** introduce any web frameworks, APIs, or databases.
- All predictions and probabilities come directly from the trained model.
  No values are fabricated or hardcoded.
