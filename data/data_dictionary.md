# Dataset Dictionary

This document tracks the verified descriptions, data types, value ranges, and feature roles of the audited student performance dataset.

## 1. Dataset Overview
- **Dataset File**: `data/student_performance_data.csv`
- **Total Records (Rows)**: 1,000
- **Total Features (Columns)**: 7 (4 Input Features, 1 Identifier, 1 Leakage Composite Score, 1 Target Label)
- **Collection / Audit Date**: August 28, 2026

---

## 2. Feature Inventory & Role Definitions

| Column Name in Dataset | Description | Pandas Data Type | Value Range | Missing Count (%) | Project Role |
| :--- | :--- | :---: | :---: | :---: | :---|
| `Student_ID` | Unique student identifier key | `object` (`str`) | `STU0001` - `STU1000` | 0 (0.00%) | **Identifier** (Excluded from $X$) |
| `MST_Score` | Mid-Semester Test marks | `float64` | 10.0 - 100.0 | 6 (0.60%) | **Input Feature** (Included in $X$) |
| `Quiz_Score` | Continuous assessment quiz results | `float64` | 0.0 - 100.0 | 6 (0.60%) | **Input Feature** (Included in $X$) |
| `Attendance_Percent` | Student lecture attendance percentage | `float64` | 20.1 - 99.9 | 6 (0.60%) | **Input Feature** (Included in $X$) |
| `Assignment_Score` | Practical homework & assignment marks | `float64` | 4.1 - 100.0 | 6 (0.60%) | **Input Feature** (Included in $X$) |
| `Performance_Score` | Weighted composite academic score ($0.4\times\text{MST} + 0.2\times\text{Quiz} + 0.2\times\text{Att} + 0.2\times\text{Assign}$) | `float64` | 20.14 - 99.74 | 0 (0.00%) | **Target Leakage** (Excluded from $X$) |
| `Performance_Category` | Tiered student performance level | `object` (`str`) | 3 discrete categories | 0 (0.00%) | **Target Label ($y$)** |

---

## 3. Target Categories & Class Distribution

1. **High Performer** (`Performance_Score` $\ge 75.0$):
   - **Count**: 261 samples (26.10%)
   - Characteristics: High exam scores, consistent assignments, high attendance.
2. **Average Performer** ($50.0 \le \text{Performance\_Score} < 75.0$):
   - **Count**: 653 samples (65.30%)
   - Characteristics: Moderate performance meeting baseline academic benchmarks.
3. **Needs Improvement** (`Performance_Score` < $50.0$):
   - **Count**: 86 samples (8.60%)
   - Characteristics: Low examination scores, missing assignments, or poor attendance requiring early academic intervention.
