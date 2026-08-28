# Assignment Flowchart & Methodology Documentation

This document fulfills the mandatory assignment requirement:
> *"A flow diagram/documentation clearly showing preprocessing and visualization steps."*

---

## 1. End-to-End Machine Learning Process Flow

```mermaid
flowchart TD
    A["Raw Student Dataset (CSV/Excel)\n(MST, Quizzes, Attendance, Assignments)"] --> B["1. Data Ingestion & Audit\n(Missing values, duplicates, outliers)"]
    
    subgraph S1["Exploratory Data Analysis (EDA)"]
        B --> C1["Class Distribution Plot"]
        B --> C2["Feature Distribution Histograms"]
        B --> C3["Correlation Heatmap"]
    end
    
    B --> D["2. Feature & Target Separation\n(X: Academic Features, y: Performance Category)"]
    D --> E["3. Stratified Train/Test Split\n(80% Training Data, 20% Testing Data)"]
    
    subgraph S2["Scikit-Learn Preprocessing Pipeline (Prevents Data Leakage)"]
        E --> F1["SimpleImputer(strategy='median')\n(Fitted strictly on Training set)"]
        F1 --> F2["StandardScaler()\n(Fitted strictly on Training set)"]
    end
    
    subgraph S3["Model Training (3 Classical Classifiers)"]
        F2 --> G1["Logistic Regression\n(Baseline Linear Classifier)"]
        F2 --> G2["Decision Tree\n(Rule-based explainable model)"]
        F2 --> G3["Random Forest\n(Ensemble of trees)"]
    end
    
    subgraph S4["Model Evaluation & Performance Comparison"]
        G1 & G2 & G3 --> H1["Compute Test Metrics\n(Accuracy, Precision, Recall, F1-Score)"]
        H1 --> H2["Generate Confusion Matrices\n(Saved to outputs/figures/)"]
        H2 --> H3["Model Comparison Table\n(Saved to outputs/model_comparison.csv)"]
    end
    
    H3 --> I["4. Select Best Model & Save Pipeline (.joblib)\n(models/best_student_model.joblib)"]
    I --> J["5. Student Performance Inference\n(src/predict.py & Optional Streamlit UI)"]
```

---

## 2. Explanation of Flowchart Stages

### Stage 1: Data Ingestion & Cleaning
- **Objective**: Ensure the incoming dataset is free of formatting issues, duplicate entries, and impossible numbers.
- **Actions**:
  - Load raw data via `pandas`.
  - Count and inspect missing values (`isnull().sum()`).
  - Drop duplicate student rows.

### Stage 2: Exploratory Data Analysis (EDA)
- **Objective**: Understand patterns in the data before training any model.
- **Actions**:
  - Bar plot of student counts per performance tier.
  - Distribution histograms for MST scores, Quizzes, Attendance, and Assignments.
  - Correlation heatmap to check which features correlate strongest with academic success.

### Stage 3: Train/Test Split (Stratified)
- **Objective**: Create an unbiased testing partition to measure generalization performance.
- **Methodology**: An 80/20 train/test split with stratification (`stratify=y`) ensures the proportion of High, Average, and Needs Improvement students remains identical in both sets.

### Stage 4: Preprocessing Pipeline (Leakage-Free Normalization)
- **Objective**: Standardize feature ranges and impute any missing entries without leaking test set statistics into training.
- **Methodology**: `SimpleImputer` and `StandardScaler` are wrapped inside a Scikit-Learn `Pipeline`. Parameters ($\mu, \sigma$) are computed **only** on `X_train`.

### Stage 5: Classification Algorithms
- **Logistic Regression**: Linear multiclass baseline.
- **Decision Tree**: Generates transparent, readable decision rules.
- **Random Forest**: Aggregates predictions of multiple decorrelated decision trees to improve stability.

### Stage 6: Evaluation & Selection
- **Metrics**: Accuracy, Precision (Macro/Weighted), Recall (Macro/Weighted), F1-Score (Macro/Weighted), and Confusion Matrix.
- **Selection**: The model with the best Macro F1-score on the test set is saved as `models/best_student_model.joblib`.
