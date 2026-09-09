# Student Performance Classification using Classical Machine Learning

🌐 **Live Web App**: [https://students-performance-classification.streamlit.app/](https://students-performance-classification.streamlit.app/)

A supervised multi-class Machine Learning system designed to predict and classify student academic performance into three distinct tiers:
- **High Performer**
- **Average Performer**
- **Needs Improvement**

The classification model is built upon four foundational academic indicators:
- **MST Score** (Mid-Semester Test score)
- **Quiz Score** (Continuous assessment score)
- **Attendance Percentage** (Classroom attendance rate)
- **Assignment Score** (Homework and practical assignment mark)

---

## 🚀 Live Interactive Google Colab Notebook

You can inspect, execute, and present the complete end-to-end pipeline interactively in **Google Colab** with a single click:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/raghavbhaskar001/student-performance-classification/blob/main/notebooks/Student_Performance_Classification_Colab.ipynb)

### Google Colab Notebook Features:
- **Zero-Configuration Setup**: Automatically detects the Colab runtime and clones/ingests `student_performance_data.csv` seamlessly without manual file uploads.
- **Visual Exploratory Analysis (EDA)**: Target class imbalance breakdown, continuous assessment KDE curves, Pearson correlation heatmaps, and multi-class separation boxplots.
- **Empirical Leakage Proof**: Mathematically proves why `Performance_Score` and `Student_ID` are strictly excluded from modeling.
- **5-Fold Stratified Cross-Validation**: Comparative benchmark of Logistic Regression, Decision Trees, and Random Forests across Accuracy, Precision, Recall, and Macro F1.
- **Held-Out Test Set Evaluation**: Complete evaluation on $N=200$ unseen test instances, featuring a high-resolution Confusion Matrix and 100% recall on at-risk students.
- **Interactive Slider Demo (`ipywidgets`)**: Embedded interactive UI right inside Colab where examiners can adjust student marks via sliders and instantly view the predicted performance badge and probability distribution bar chart.
- **Model Artifact Export**: Direct serialization and one-click download for `best_student_model.joblib`.

---

## Project Objective

The primary objective of this project is to develop a reliable, explainable, and deployable classical machine learning system for educational performance assessment. By identifying students who require academic support (*Needs Improvement*) or those demonstrating strong mastery (*High Performer*), educational institutions can initiate targeted pedagogical interventions, optimize resource allocation, and deliver proactive mentorship without relying on opaque, computationally heavy architectures.

---

## Team & Contributions

| Member | Role & Contributions |
| :--- | :--- |
| **Raghav** | **ML Lead** — data preprocessing, EDA, model training, evaluation, model serialization, and inference interface |
| **Divyanshi** | **Data & Documentation** — dataset sourcing, data dictionary, project report, presentation, and EDA observations |
| **Aabiya** | **UI & Deployment** — Streamlit interface, UI testing, and live demonstration |

---

## Machine Learning Workflow

The end-to-end pipeline follows standard industry and academic best practices for classical machine learning:

```text
Raw Academic Data
     ↓
Data Audit & EDA
     ↓
Train/Test Split (80/20 Stratified)
     ↓
Missing Value Imputation (Median)
     ↓
Feature Scaling (StandardScaler where appropriate)
     ↓
Classical ML Model Training
     ↓
5-Fold Stratified Cross-Validation
     ↓
Model Comparison & Selection
     ↓
Held-Out Test Evaluation
     ↓
Best Model Serialization (.joblib Pipeline)
     ↓
Inference Module (src/predict.py)
     ↓
Streamlit UI (app.py)
```

---

## Dataset & Features

### Input Features

The predictive model takes exactly four numeric inputs:

| Feature Name | Field Name | Data Type | Valid Range | Description |
| :--- | :--- | :--- | :--- | :--- |
| **MST Score** | `MST_Score` | Numeric (Float/Int) | $0 - 100$ | Mid-Semester Test examination score |
| **Quiz Score** | `Quiz_Score` | Numeric (Float/Int) | $0 - 100$ | Cumulative quiz and continuous assessment score |
| **Attendance %** | `Attendance_Percent` | Numeric (Float/Int) | $0 - 100$ | Classroom lecture attendance percentage |
| **Assignment Score** | `Assignment_Score` | Numeric (Float/Int) | $0 - 100$ | Homework, lab work, and assignment evaluation score |

### Target Variable

- **`Performance_Category`**: Multi-class categorical label (*High Performer*, *Average Performer*, *Needs Improvement*).

### Feature Exclusions & Rationale

- **`Student_ID`**: Excluded from feature sets to prevent the model from memorizing arbitrary identifiers or learning spurious correlations.
- **`Performance_Score`**: Excluded because it represents a direct weighted linear combination of the input metrics used to assign categories, which would cause complete target leakage.

> **Dataset Availability Note**: Complete schema definitions and field metadata are documented in `data/data_dictionary.md`. Raw CSV dataset files are excluded from version control in accordance with `.gitignore` standards.

---

## Models Evaluated

Three classical machine learning algorithms were trained and comparatively evaluated:

1. **Logistic Regression (Multinomial)**: Linear decision boundaries with L2 regularization, standard scaling, and median imputation.
2. **Decision Tree Classifier**: Non-parametric tree-based classifier with `max_depth=5` to prevent overfitting.
3. **Random Forest Classifier**: Ensemble of 100 bagging decision trees with `max_depth=6`.

### Class Imbalance Handling
The target class distribution exhibits moderate natural imbalance (~65.3% Average Performer, ~26.1% High Performer, ~8.6% Needs Improvement). To ensure robust sensitivity on the minority *Needs Improvement* class without resorting to synthetic sample generation, all algorithms were configured with `class_weight='balanced'`.

---

## Model Selection & Results

### 5-Fold Stratified Cross-Validation (Training Set, $N=800$)

| Model | CV Accuracy | CV Macro Precision | CV Macro Recall | CV Macro F1 |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **96.12% ± 1.45%** | **92.26%** | **97.61%** | **0.9453 ± 0.0257** |
| Decision Tree | 86.25% ± 1.98% | 81.12% | 86.49% | 0.8293 ± 0.0187 |
| Random Forest | 91.75% ± 1.60% | 88.64% | 91.05% | 0.8939 ± 0.0314 |

**Selection Decision**: **Logistic Regression** was selected as the final production model based on the highest CV Macro F1-score ($0.9453$) and strong overall validation performance.

### Final Held-Out Test Evaluation ($N=200$)

The winning Logistic Regression pipeline was evaluated on the unseen held-out test split:

- **Test Accuracy**: **94.00%**
- **Test Macro F1-Score**: **91.18%**
- **Test Weighted F1-Score**: **94.21%**
- **Test Macro Recall**: **96.95%**
- **Minority Class (*Needs Improvement*) Recall**: **100.0%** (Zero false negatives on at-risk students)

---

## Leakage Prevention

To ensure academic and methodological rigor, the project enforces strict data leakage prevention protocols:

1. **Target Leakage Elimination**: `Performance_Score` is strictly removed prior to modeling.
2. **Identifier Exclusion**: `Student_ID` is removed from all feature matrices.
3. **Strict Partitioning**: Stratified Train/Test splitting ($80/20$) is performed before fitting any transformations.
4. **Encapsulated Preprocessing**:
   - `SimpleImputer(strategy='median')` is fitted strictly on training data.
   - `StandardScaler()` parameters ($\mu, \sigma$) are computed exclusively on training data.
   - Preprocessing steps and the estimator are encapsulated in an end-to-end `sklearn.pipeline.Pipeline`.
5. **Independent Test Set**: The test set ($N=200$) is used solely for final one-time reporting and never touched during preprocessing or hyperparameter decisions.

---

## Inference System

Production inference is encapsulated in `src/predict.py` via the `predict_performance()` function:

- **Model Loading**: Dynamically loads the serialized pipeline from `models/best_student_model.joblib`.
- **Internal Preprocessing**: All necessary imputation and scaling transformations are executed automatically within the loaded Scikit-learn pipeline.
- **Input Validation**: Validates that all four inputs are within the permissible range ($0 - 100$).
- **Output Format**: Returns both the predicted category string and the corresponding class probability distribution.

---

## Streamlit Application

🌐 **Live Deployed App**: [https://students-performance-classification.streamlit.app/](https://students-performance-classification.streamlit.app/)

The user interface is implemented in `app.py`:

- **Interactive Inputs**: Provides intuitive input fields and sliders for the four academic parameters.
- **Decoupled Architecture**: `app.py` functions purely as a presentation layer, invoking `src.predict.predict_performance()` for predictions.
- **No Training Logic in UI**: The application contains no model training or dataset processing code; it operates entirely on the serialized pipeline artifact.
- **Visual Analytics**: Displays the predicted performance category alongside class probability breakdowns.

---

## Project Structure

```text
Classic ML project/
│
├── data/
│   └── data_dictionary.md             # Feature definitions, schema, and metadata
│
├── docs/                              # Technical reports & academic documentation
│   ├── assignment_flowchart.md        # Pipeline flowchart specification
│   ├── dataset_audit.md               # Data quality and integrity audit
│   ├── final_project_architecture.md  # End-to-end system architecture
│   ├── final_submission_checklist.md  # Submission verification checklist
│   ├── phase_2b_preprocessing_eda.md  # Comprehensive EDA & preprocessing notes
│   ├── phase_4_inference.md           # Inference engine API specification
│   └── presentation_notes.md          # Viva presentation notes & Q&A guide
│
├── models/
│   ├── .gitkeep
│   └── best_student_model.joblib      # Serialized Scikit-learn winning pipeline
│
├── notebooks/                         # Interactive Jupyter notebooks
│   ├── Student_Performance_Classification_Colab.ipynb # Master Colab interactive showcase
│   ├── 01_dataset_audit.ipynb         # Data audit & verification
│   ├── 01_eda_and_preprocessing.ipynb # Exploratory data analysis & pipeline design
│   └── 02_model_training_and_eval.ipynb# Model training, CV & evaluation
│
├── outputs/
│   ├── figures/                       # Saved EDA plots and confusion matrix figures
│   │   └── .gitkeep
│   └── model_comparison_cv.csv        # Cross-validation comparison table
│
├── src/                               # Core Python ML package
│   ├── __init__.py
│   ├── data_preprocessing.py          # Preprocessing functions & pipeline builder
│   ├── evaluate.py                    # Evaluation metrics & performance reports
│   ├── predict.py                     # Standalone production inference engine
│   ├── train_models.py                # Model training & CV benchmark routines
│   └── verify_inference.py            # Comprehensive inference test suite
│
├── app.py                             # Streamlit web application
├── requirements.txt                   # Project dependencies
├── .gitignore                         # Git exclusion rules
└── README.md                          # Main project documentation
```

---

## Installation & Local Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Launch the Streamlit Application

```bash
python -m streamlit run app.py
```

This command starts the local development server and opens the application in your default web browser (typically at `http://localhost:8501`).

---

## Reproducibility & Stack

The project relies exclusively on standard, stable packages listed in `requirements.txt`:

- **Python**: Core programming environment
- **pandas**: Tabular data manipulation and structured analysis
- **numpy**: Numerical operations and array computation
- **scikit-learn**: Machine learning algorithms, pipelines, cross-validation, and metrics
- **matplotlib**: Statistical charting and figure rendering
- **seaborn**: Correlation heatmaps and distribution visualizations
- **joblib**: Model artifact serialization and pipeline persistence
- **streamlit**: Lightweight web application framework

---

## Academic Integrity & ML Principles

- **Authentic Predictions**: All predictions are computed directly by the trained Scikit-learn model; no hardcoded or fabricated outputs exist.
- **Genuine Classical ML**: Developed strictly with classical supervised algorithms (Logistic Regression, Decision Trees, Random Forests), without unwarranted deep learning or LLM claims.
- **Zero Leakage**: Strict methodological boundaries prevent target and feature leakage.
- **Principled Imbalance Handling**: Leverages cost-sensitive balanced loss weighting rather than arbitrary data distortion.
- **Rigorous Validation**: Confirmed via 5-Fold Stratified Cross-Validation on training data and verified on an independent test set.

---

## Current Project Status

The classical ML pipeline, trained model, inference interface, and Streamlit UI are implemented and verified.

---

## Deployment Note

The Streamlit web application requires `models/best_student_model.joblib` to execute predictions. This serialized pipeline file must remain present in the `models/` directory for local execution and cloud deployment.

