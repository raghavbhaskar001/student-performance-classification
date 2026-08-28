# Dataset Audit & ML-Readiness Inspection Report

**Project**: Student Performance Classification using Classical Machine Learning  
**Phase**: Phase 2A — Dataset Ingestion, Audit & ML-Readiness Inspection  
**Audit Date**: August 28, 2026  
**Auditor**: Classical ML Development Team  

---

## 1. Dataset Overview

- **Dataset File Path**: `data/student_performance_data.csv`
- **File Format**: Comma-Separated Values (CSV)
- **Total Records (Rows)**: 1,000
- **Total Columns**: 7
- **Raw Data Status**: Preserved in raw, read-only format. No manual edits, deletions, synthetic rows, or overwrites performed.

---

## 2. Column Inventory & Data Types

| # | Column Name | Pandas Data Type | Non-Null Count | Null Count | Null % | Example Values |
|---|:---|:---:|:---:|:---:|:---:|:---|
| 1 | `Student_ID` | `object` (`str`) | 1,000 | 0 | 0.00% | `STU0001`, `STU0002`, `STU1000` |
| 2 | `MST_Score` | `float64` | 994 | 6 | 0.60% | `33.5`, `78.8`, `98.7` |
| 3 | `Quiz_Score` | `float64` | 994 | 6 | 0.60% | `52.1`, `78.8`, `72.1` |
| 4 | `Attendance_Percent` | `float64` | 994 | 6 | 0.60% | `44.0`, `82.1`, `78.5` |
| 5 | `Assignment_Score` | `float64` | 994 | 6 | 0.60% | `38.3`, `86.3`, `97.7` |
| 6 | `Performance_Score` | `float64` | 1,000 | 0 | 0.00% | `40.28`, `80.96`, `89.14` |
| 7 | `Performance_Category` | `object` (`str`) | 1,000 | 0 | 0.00% | `Needs Improvement`, `High Performer`, `Average Performer` |

---

## 3. Data Quality & Missing Value Report

### Missing Values Summary
- **Total Missing Cells**: 24 out of 7,000 (0.34% of total dataset cells)
- **Rows with At Least One Missing Value**: 24 rows (2.40% of total rows)
- **Columns with Missing Values**:
  - `MST_Score`: 6 missing values (0.60%)
  - `Quiz_Score`: 6 missing values (0.60%)
  - `Attendance_Percent`: 6 missing values (0.60%)
  - `Assignment_Score`: 6 missing values (0.60%)
- **Zero Missing Values**: `Student_ID`, `Performance_Score`, `Performance_Category` have 0 missing values.

### Sample Rows with Missing Entries
| Row Index | Student_ID | MST_Score | Quiz_Score | Attendance_Percent | Assignment_Score | Performance_Score | Performance_Category | Missing Field |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---|:---|
| 76 | `STU0077` | 84.4 | **NaN** | 82.3 | 79.4 | 80.52 | High Performer | `Quiz_Score` |
| 110 | `STU0111` | 41.0 | 47.1 | 79.1 | **NaN** | 53.76 | Average Performer | `Assignment_Score` |
| 136 | `STU0137` | 52.0 | 41.6 | 76.8 | **NaN** | 55.56 | Average Performer | `Assignment_Score` |
| 168 | `STU0169` | 61.8 | 71.0 | 92.9 | **NaN** | 72.12 | Average Performer | `Assignment_Score` |
| 216 | `STU0217` | 34.1 | 25.3 | **NaN** | 21.9 | 40.50 | Needs Improvement | `Attendance_Percent` |
| 395 | `STU0396` | 68.6 | 75.3 | **NaN** | 64.3 | 73.06 | Average Performer | `Attendance_Percent` |
| 442 | `STU0443` | **NaN** | 84.0 | 87.9 | 89.1 | 87.00 | High Performer | `MST_Score` |
| 524 | `STU0525` | 72.8 | 68.2 | **NaN** | 79.4 | 78.26 | High Performer | `Attendance_Percent` |
| 563 | `STU0564` | 54.3 | **NaN** | 70.5 | 56.8 | 56.88 | Average Performer | `Quiz_Score` |
| 598 | `STU0599` | **NaN** | 51.9 | 77.3 | 71.7 | 67.82 | Average Performer | `MST_Score` |

### Duplicate Records
- **Duplicate Rows Count**: 0 (no duplicate entries detected; `Student_ID` contains 1,000 distinct unique keys).

---

## 4. Descriptive Numerical Statistics

All numerical features were checked for ranges, distribution centers, and dispersion:

| Metric | MST_Score | Quiz_Score | Attendance_Percent | Assignment_Score | Performance_Score |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Count** | 994 | 994 | 994 | 994 | 1,000 |
| **Mean** | 64.7767 | 63.8288 | 75.0655 | 66.0030 | 66.8881 |
| **Std Dev** | 16.1657 | 15.9938 | 17.7200 | 15.2253 | 12.0965 |
| **Minimum** | 10.00 | 0.00 | 20.10 | 4.10 | 20.14 |
| **25% (Q1)** | 54.00 | 53.60 | 66.80 | 56.00 | 59.50 |
| **50% (Median)** | 64.35 | 63.50 | 79.10 | 66.30 | 67.15 |
| **75% (Q3)** | 75.40 | 74.90 | 88.60 | 76.50 | 75.90 |
| **Maximum** | 100.00 | 100.00 | 99.90 | 100.00 | 99.74 |
| **Sensible Range Check [0, 100]** | Valid | Valid | Valid | Valid | Valid |

### Range Validation Findings
- No negative values, out-of-scale percentages (>100%), or impossible academic scores exist in the dataset.
- Attendance spans from 20.1% to 99.9%.
- Mid-Semester Tests span from 10.0 to 100.0.
- Quizzes span from 0.0 to 100.0.
- Assignments span from 4.1 to 100.0.

---

## 5. Target Variable & Class Balance Analysis

- **Target Column**: `Performance_Category`
- **Number of Classes**: 3 distinct categories
- **Target Completeness**: 1,000 non-null values (100% complete)

### Class Distribution Table
| Class Label | Sample Count | Proportion (%) | Balance Status |
|:---|:---:|:---:|:---|
| **Average Performer** | 653 | 65.30% | Majority Class |
| **High Performer** | 261 | 26.10% | Moderate Representation |
| **Needs Improvement** | 86 | 8.60% | Minority Class |
| **Total** | **1,000** | **100.00%** | — |

### Imbalance Assessment
- The dataset exhibits **moderate class imbalance** with an imbalance ratio of **7.59 : 1** between the majority class (`Average Performer`) and the minority class (`Needs Improvement`).
- **Evaluation Implication**: Accuracy alone will be misleading (a naive majority-class classifier achieves 65.3% accuracy). Models must be evaluated using **Macro Precision, Macro Recall, Macro F1-score**, and **Per-Class Confusion Matrices**.
- **Splitting Implication**: All train/test splits and cross-validation folds must use **Stratified** sampling (`stratify=y`) to maintain the 8.6% minority class proportion across folds.

---

## 6. Feature Role & Target Leakage Analysis

### Column Role Breakdown

| Column Name | Type | Assigned Role in Project | Justification |
|:---|:---:|:---:|:---|
| `Student_ID` | Identifier | **Exclude from $X$** | Unique arbitrary key (`STU0001` - `STU1000`) with zero generalizable predictive value. |
| `MST_Score` | Numerical Feature | **Include in $X$** | Core academic feature representing mid-semester examination mastery. |
| `Quiz_Score` | Numerical Feature | **Include in $X$** | Core academic feature representing continuous quiz comprehension. |
| `Attendance_Percent` | Numerical Feature | **Include in $X$** | Core behavioral feature representing student lecture engagement. |
| `Assignment_Score` | Numerical Feature | **Include in $X$** | Core practical feature representing project/homework diligence. |
| `Performance_Score` | Numerical Composite | **EXCLUDE from $X$ (LEAKAGE)** | Direct mathematical precursor to target; contains 100% target information. |
| `Performance_Category` | Categorical Target | **Target Variable ($y$)** | Multi-class label for classification (3 tiers). |

---

### Target Leakage Mathematical Investigation

An empirical mathematical analysis was conducted to determine the relationship between `Performance_Score`, the input features, and `Performance_Category`:

#### 1. Mathematical Formula of `Performance_Score`
Using ordinary least squares regression over non-null rows, `Performance_Score` was verified to be an exact linear composite score:
$$\text{Performance\_Score} = 0.40 \times \text{MST\_Score} + 0.20 \times \text{Quiz\_Score} + 0.20 \times \text{Attendance\_Percent} + 0.20 \times \text{Assignment\_Score}$$
- **Residual Sum of Squares**: $\approx 1.08 \times 10^{-24}$ (exact match down to floating point precision).
- **Max absolute difference**: $0.000000$.

#### 2. Deterministic Mapping to `Performance_Category`
Inspecting the range of `Performance_Score` grouped by `Performance_Category`:
- **Needs Improvement**: $\text{Performance\_Score} \in [20.14, 49.96]$ (Threshold: $< 50.0$)
- **Average Performer**: $\text{Performance\_Score} \in [50.04, 74.94]$ (Threshold: $[50.0, 75.0)$)
- **High Performer**: $\text{Performance\_Score} \in [75.00, 99.74]$ (Threshold: $\ge 75.0$)

```mermaid
flowchart LR
    A["Raw Student Inputs<br>(MST, Quiz, Attendance, Assignment)"] -->|40% / 20% / 20% / 20%| B["Performance_Score<br>(Aggregate Continuous Score)"]
    B -->|Thresholds: &lt;50, 50-75, &ge;75| C["Performance_Category<br>(Needs Improvement / Average / High)"]
    style B fill:#ffcccc,stroke:#cc0000,stroke-width:2px
    style C fill:#ccffcc,stroke:#009900,stroke-width:2px
```

> [!CAUTION]
> **CRITICAL ML DECISION — TARGET LEAKAGE**:
> If `Performance_Score` were included in feature matrix $X$, any machine learning model would achieve near 100% accuracy simply by learning threshold splits on `Performance_Score`. This would bypass learning true relationships from raw academic inputs and defeat the purpose of student performance modeling.
> **Therefore, `Performance_Score` MUST be excluded from feature matrix $X$.**

---

## 7. Observed Facts vs. Recommendations

### Observed Facts (Empirical Findings)
1. The dataset contains exactly 1,000 instances and 7 columns.
2. The raw dataset contains 24 missing numerical values (0.6% across `MST_Score`, `Quiz_Score`, `Attendance_Percent`, and `Assignment_Score`).
3. No missing values exist in `Student_ID`, `Performance_Score`, or `Performance_Category`.
4. There are zero duplicate rows.
5. All feature values reside within physically sensible academic boundaries $[0, 100]$.
6. `Performance_Category` consists of three classes with distribution: 65.3% Average, 26.1% High, 8.6% Needs Improvement.
7. `Performance_Score` is a deterministic weighted linear aggregate of the four input features and directly determines `Performance_Category`.

### Recommendations (For Phase 2B & Beyond)
1. **Feature Set Definition**: Set $X = [\text{MST\_Score}, \text{Quiz\_Score}, \text{Attendance\_Percent}, \text{Assignment\_Score}]$ and $y = \text{Performance\_Category}$. Exclude `Student_ID` (identifier) and `Performance_Score` (target leakage).
2. **Missing Value Imputation**: Use `SimpleImputer(strategy='median')` fitted strictly within scikit-learn pipelines on training folds to prevent data leakage.
3. **Feature Scaling**: Apply `StandardScaler` (or `MinMaxScaler`) inside pipelines to normalize feature distributions for distance/gradient-sensitive algorithms (e.g., Logistic Regression).
4. **Stratified Splitting**: Use `StratifiedShuffleSplit` / `train_test_split(stratify=y)` with an 80/20 train/test split ratio to preserve the 8.6% minority class proportion.
5. **Class Imbalance Handling**: Use `class_weight='balanced'` in Logistic Regression, Decision Trees, and Random Forests to ensure the model focuses adequately on detecting "Needs Improvement" students.
6. **Primary Optimization Metric**: Evaluate and rank models based on **Macro F1-Score** rather than raw Accuracy.

---

## 8. ML-Readiness Conclusion

| Inspection Criterion | Status | Notes |
|:---|:---:|:---|
| **Data Integrity** | Passed | 1,000 consistent rows, zero duplicates, clean data types. |
| **Value Ranges** | Passed | All values strictly within valid $[0, 100]$ score/percentage scale. |
| **Target Presence** | Passed | `Performance_Category` is 100% complete with 3 clear tiers. |
| **Leakage Isolation** | Identified & Resolved | `Performance_Score` flagged for mandatory exclusion from $X$. |
| **Missing Data Manageability** | Passed | Missing rate is only 0.6% per feature, easily handled via median imputation in pipeline. |
| **ML Readiness Status** | **READY FOR PHASE 2B** | Preprocessing pipeline can now be designed with full schema alignment. |
