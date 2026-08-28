# Presentation & Viva Preparation Notes (For Divyanshi & Team)

Use these concise points for the project PPT and during oral viva questions.

---

## 1. Project Motivation & Goal
- **Problem**: Early identification of students who require academic support or intervention.
- **Objective**: Build an explainable Classical Machine Learning model to categorize students into **High Performer**, **Average Performer**, and **Needs Improvement**.
- **Inputs**: Mid-Semester Test (MST) marks, Quiz results, Attendance records, and Assignment scores.

---

## 2. Key Machine Learning Concepts (Viva Q&A)

### Q1: Why did you use Scikit-Learn Pipelines instead of scaling the whole dataset first?
> **Answer**: Scaling the entire dataset before splitting causes **Data Leakage**. When you fit a scaler on the whole dataset, information about the mean ($\mu$) and standard deviation ($\sigma$) of the test data leaks into the training process. Pipelines ensure the scaler is fitted **only on the training set**, guaranteeing a truly unseen test evaluation.

### Q2: What is Stratified Splitting and why is it important?
> **Answer**: In academic performance data, class distribution might not be evenly balanced (e.g., fewer students may be in "Needs Improvement"). `StratifiedShuffleSplit` / `stratify=y` ensures that both the training and test sets maintain the exact same percentage of each performance tier.

### Q3: Why compare Logistic Regression, Decision Tree, and Random Forest?
> **Answer**:
> - **Logistic Regression**: Serves as an interpretable linear baseline.
> - **Decision Tree**: Provides explicit if-else decision boundaries that educators can directly interpret.
> - **Random Forest**: An ensemble method that combines multiple trees via bagging, reducing variance and resisting overfitting.

### Q4: Why is F1-Score (Macro) prioritized alongside Accuracy?
> **Answer**: Accuracy can be deceptive if one class has significantly more students. **Macro F1-Score** calculates the unweighted average of F1-scores across all three classes, giving equal importance to detecting "Needs Improvement" students as "High Performers".

### Q5: What does the Confusion Matrix tell us?
> **Answer**: The Confusion Matrix reveals exact misclassifications — for instance, whether a "Needs Improvement" student was mistakenly predicted as "Average", allowing us to evaluate false negatives.

---

## 3. Recommended 8-Slide PPT Structure (Divyanshi)
1. **Slide 1**: Title, Team Members (Raghav, Divyanshi, Aabiya), Project Objective.
2. **Slide 2**: Problem Statement & Academic Features (MST, Quizzes, Attendance, Assignments).
3. **Slide 3**: End-to-End Methodology Flowchart (from `docs/assignment_flowchart.md`).
4. **Slide 4**: Data Preprocessing & Leakage Prevention (Pipeline architecture).
5. **Slide 5**: Exploratory Data Analysis (EDA) Highlights & Key Graphs (from `outputs/figures/`).
6. **Slide 6**: Classification Algorithms Overview (Logistic Regression vs. Decision Tree vs. Random Forest).
7. **Slide 7**: Model Evaluation & Comparison Table (Accuracy, Precision, Recall, F1, Confusion Matrix).
8. **Slide 8**: Final Model Selection, Inference Demo, and Conclusion.
