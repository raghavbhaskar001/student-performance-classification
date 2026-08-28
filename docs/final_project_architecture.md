# Final Project Architecture & Methodology Specification

**Project Title**: Student Performance Classification using Classical Machine Learning  
**Lead & Core ML**: Raghav  
**Documentation & Viva**: Divyanshi  
**UI Integration**: Aabiya  

---

## 1. End-to-End System Architecture

```mermaid
flowchart TD
    A["Raw Student Dataset\n(data/student_performance_data.csv)\n1,000 Rows, 7 Columns"] --> B["1. Data Ingestion & Quality Audit\n(Missing values, schema, types)"]
    
    subgraph S1["Exploratory Data Analysis (EDA)"]
        B --> C1["Target Class Distribution\n(High: 26.1%, Avg: 65.3%, Needs: 8.6%)"]
        B --> C2["Feature Distribution Histograms\n(MST, Quiz, Attendance, Assignment)"]
        B --> C3["Multivariate Correlation Heatmap\n(Feature relationships & collinearity)"]
    end
    
    B --> D["2. Feature & Target Isolation\nX: 4 Academic Features\ny: Performance_Category\n(Student_ID & Performance_Score Excluded)"]
    
    D --> E["3. Stratified Train/Test Split\n(80% Train N=800, 20% Test N=200, random_state=42)"]
    
    subgraph S2["Leakage-Free Preprocessing Pipelines"]
        E --> F1["Logistic Regression Pipeline\nSimpleImputer(median) -> StandardScaler() -> LogisticRegression"]
        E --> F2["Decision Tree Pipeline\nSimpleImputer(median) -> DecisionTreeClassifier(max_depth=5)"]
        E --> F3["Random Forest Pipeline\nSimpleImputer(median) -> RandomForestClassifier(n_estimators=100, max_depth=6)"]
    end
    
    subgraph S3["Model Evaluation & Comparison"]
        F1 & F2 & F3 --> G1["5-Fold Stratified Cross-Validation\n(Evaluated strictly on X_train)"]
        G1 --> G2["Comparison Table (outputs/model_comparison_cv.csv)\nMacro F1, Precision, Recall, Accuracy"]
    end
    
    G2 --> H["4. Best Model Selection & Serialization\nWinner: Logistic Regression (CV Macro F1 = 0.9453)\nSaved: models/best_student_model.joblib"]
    
    H --> I["5. Final Test Evaluation (One-time on X_test)\nTest Accuracy: 94.00% | Test Macro F1: 91.18%\noutputs/figures/final_confusion_matrix.png"]
    
    subgraph S4["Production Inference & UI Layer"]
        H --> J["6. Python Inference Module\n(src/predict.py -> predict_performance)\nInput Validation & Probability Extraction"]
        J --> K["7. Python Streamlit Web Interface\n(app.py - Student Performance AI)\nInput Controls -> Prediction Card -> Probability Bars"]
    end
    
    K --> L["Final Output\nPerformance Category + Model Probability"]
```

---

## 2. Component Specifications

### 2.1 Data Layer
- **Source**: `data/student_performance_data.csv` (1,000 observations).
- **Features ($X$)**:
  1. `MST_Score` ($0 - 100$): Mid-Semester Test examination score.
  2. `Quiz_Score` ($0 - 100$): Continuous assessment quiz score.
  3. `Attendance_Percent` ($0 - 100\%$): Classroom lecture attendance percentage.
  4. `Assignment_Score` ($0 - 100$): Homework and assignment mark.
- **Target ($y$)**: `Performance_Category` (Multi-class: *High Performer*, *Average Performer*, *Needs Improvement*).
- **Explicit Exclusions**:
  - `Student_ID`: Unique key (`STU0001` to `STU1000`), excluded to prevent identifier memorization.
  - `Performance_Score`: Deterministic linear combination of inputs ($0.4\times\text{MST} + 0.2\times\text{Quiz} + 0.2\times\text{Att} + 0.2\times\text{Assign}$), excluded to prevent target leakage.

### 2.2 Preprocessing & Partitioning Layer
- **Split Ratio**: 80% Training ($N=800$), 20% Testing ($N=200$).
- **Stratification**: `stratify=y` guarantees exact class balance across partitions.
- **Imputation**: `SimpleImputer(strategy='median')` fitted strictly on training data.
- **Scaling**: `StandardScaler()` applied to Logistic Regression (linear distance-sensitive algorithm). Tree models skip scaling as split points are scale-invariant.

### 2.3 Model Training & Cross-Validation Layer
- **Class Imbalance Management**: `class_weight='balanced'` configured for all models.
- **Validation**: 5-Fold Stratified Cross-Validation on the 80% training set.
- **Model Comparison Results (CV on Training Data)**:
  - **Logistic Regression**: CV Accuracy = $0.9612 \pm 0.0145$, CV Macro F1 = $\mathbf{0.9453 \pm 0.0257}$ (Selected Best Model).
  - **Decision Tree**: CV Accuracy = $0.8625 \pm 0.0198$, CV Macro F1 = $0.8293 \pm 0.0187$.
  - **Random Forest**: CV Accuracy = $0.9175 \pm 0.0160$, CV Macro F1 = $0.8939 \pm 0.0314$.

### 2.4 Final Evaluation Layer (Held-out Test Set)
- **Sample Size**: 200 unseen samples.
- **Logistic Regression Test Results**:
  - Test Accuracy: $\mathbf{94.00\%}$
  - Test Macro Precision: $87.35\%$
  - Test Macro Recall: $96.95\%$
  - Test Macro F1-Score: $\mathbf{91.18\%}$
  - Test Weighted F1-Score: $94.21\%$
  - Minority Class ("Needs Improvement") Recall: $\mathbf{100.0\%}$ (Zero false negatives).

### 2.5 Inference & Presentation Layer
- **Model Serialization**: `models/best_student_model.joblib` (Self-contained Scikit-learn Pipeline).
- **Backend Inference (`src/predict.py`)**:
  - Validates feature ranges ($0 \le \text{value} \le 100$).
  - Assembles single-row DataFrame with explicit feature headers.
  - Automatically executes pipeline preprocessing and estimator inference.
  - Dynamically extracts probability distribution via `pipeline.classes_`.
- **Frontend UI (`app.py`)**:
  - Pure Python Streamlit application with modern glassmorphic interface.
  - Directly consumes `src.predict.predict_performance`.
  - Zero training or data preprocessing logic embedded in the UI layer.
