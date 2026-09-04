# DiabetesAI — Explainable Diabetes Risk Assessment

**DiabetesAI** is an explainable machine learning project designed to estimate diabetes risk from commonly available health indicators. The project goes beyond a basic binary classifier by incorporating data-quality handling, model comparison, cross-validation, hyperparameter tuning, probability calibration, explainability, and what-if scenario analysis.

> **Important:** DiabetesAI is an educational and research-oriented risk-assessment system. It is **not a medical diagnostic tool** and should not be used to diagnose diabetes, prescribe treatment, or replace a qualified healthcare professional.

---

## Project Overview

Diabetes is a major global health challenge, and machine learning can help identify patterns associated with increased diabetes risk.

This project explores how supervised machine learning can be used to build an **interpretable diabetes risk assessment pipeline** while addressing practical issues that are often overlooked in introductory ML projects.

The system takes patient-related features such as glucose level, BMI, age, blood pressure, and other health indicators and produces a model-estimated probability of diabetes.

The project focuses on three major goals:

* **Predictive performance** — build and compare multiple ML models.
* **Reliable evaluation** — use stratified cross-validation, multiple metrics, and probability calibration.
* **Explainability** — understand which features influence model predictions using feature importance and SHAP.

---

## Key Features

### 1. Data Quality Handling

The project performs a data-quality audit before model training.

Several variables in the dataset contain zero values that can be physiologically implausible, including:

* Glucose
* Blood Pressure
* Skin Thickness
* Insulin
* BMI

These values are treated as missing observations and handled using **median imputation inside the ML pipeline**.

This prevents preprocessing information from leaking from the test set into the training process.

---

### 2. Multiple Machine Learning Models

The project evaluates several classification algorithms:

* Logistic Regression
* Linear Support Vector Machine
* RBF Support Vector Machine
* Random Forest
* Gradient Boosting

This allows the system to compare different approaches rather than relying on a single algorithm.

---

### 3. Stratified Cross-Validation

Model performance is evaluated using **5-fold Stratified Cross-Validation**.

The following metrics are considered:

* Accuracy
* Precision
* Recall / Sensitivity
* F1 Score
* ROC-AUC

Recall is particularly important in a screening-oriented application because false negatives can cause potentially higher-risk cases to be missed.

---

### 4. Hyperparameter Optimization

The Random Forest model is further optimized using `GridSearchCV`.

The search evaluates combinations of:

* Number of trees
* Maximum tree depth
* Minimum samples per leaf
* Maximum features

ROC-AUC is used as the optimization metric.

---

### 5. Probability Calibration

Instead of treating the raw classifier output as a reliable probability, the project applies probability calibration using `CalibratedClassifierCV`.

The notebook also evaluates:

* ROC-AUC
* Brier Score
* Calibration Curve

This provides a more responsible interpretation of model confidence.

> The resulting probability is a **model-estimated probability**, not a clinically validated diabetes risk score.

---

### 6. Explainable AI

DiabetesAI includes explainability techniques to investigate why the model makes particular predictions.

#### Global Feature Importance

Random Forest feature importance is used to identify which variables contribute most strongly to model predictions across the dataset.

#### SHAP

The project optionally uses **SHAP (SHapley Additive exPlanations)** to provide a more detailed explanation of model behavior.

SHAP can help answer questions such as:

> Which features are pushing the model toward a higher or lower diabetes-risk prediction?

This makes the project more interpretable than a simple black-box prediction system.

---

### 7. Individual Risk Assessment

The notebook provides a reusable function for assessing an individual input:

```python
assess_diabetes_risk(values)
```

The function returns a model-estimated probability and an educational risk category.

The categories are intentionally presented as **model-based UI categories**, not clinical thresholds.

---

### 8. What-If Scenario Analysis

The project also includes a what-if analysis function:

```python
compare_scenarios(base_values, modified_values)
```

This allows users to compare model predictions under different input scenarios.

For example, a user can explore how changing one or more input values affects the model's prediction.

> What-if analysis demonstrates model sensitivity. It should **not** be interpreted as causal evidence that changing a particular health variable will produce a specific medical outcome.

---

## Machine Learning Pipeline

The overall workflow is:

```text
Patient Health Information
          ↓
     Input Validation
          ↓
      Data Cleaning
          ↓
 Missing-Value Treatment
          ↓
   Feature Preprocessing
          ↓
 Model Comparison + CV
          ↓
 Hyperparameter Tuning
          ↓
   Best Model Selection
          ↓
 Probability Calibration
          ↓
   Diabetes Risk Estimate
          ↓
   Explainable AI / SHAP
          ↓
 What-If Scenario Analysis
```

---

## Dataset

The project uses the **Pima Indians Diabetes Dataset**, containing medical and demographic attributes associated with diabetes outcomes.

### Input Features

| Feature                  | Description                  |
| ------------------------ | ---------------------------- |
| Pregnancies              | Number of pregnancies        |
| Glucose                  | Plasma glucose concentration |
| BloodPressure            | Diastolic blood pressure     |
| SkinThickness            | Triceps skin fold thickness  |
| Insulin                  | 2-Hour serum insulin         |
| BMI                      | Body Mass Index              |
| DiabetesPedigreeFunction | Diabetes pedigree function   |
| Age                      | Age of the individual        |

### Target

```text
Outcome
0 → No diabetes outcome recorded
1 → Diabetes outcome recorded
```

The dataset is used for machine learning experimentation and does not represent the full diversity of real-world populations.

---

## Evaluation Strategy

Rather than relying only on accuracy, the project evaluates the model using several complementary metrics.

### Accuracy

Measures the proportion of correct predictions.

### Precision

Measures how many predicted positive cases are actually positive.

### Recall / Sensitivity

Measures how many actual positive cases are successfully identified.

### F1 Score

Balances precision and recall.

### ROC-AUC

Measures the model's ability to distinguish between the two classes across different classification thresholds.

### Brier Score

Evaluates the quality of probabilistic predictions, where lower values generally indicate better calibration.

---

## Technology Stack

### Programming Language

* Python

### Machine Learning

* Scikit-learn
* Random Forest
* Logistic Regression
* Support Vector Machines
* Gradient Boosting

### Data Processing

* Pandas
* NumPy

### Visualization

* Matplotlib
* Seaborn

### Explainable AI

* SHAP

### Model Persistence

* Joblib

### Development Environment

* Jupyter Notebook
* Google Colab
* GitHub

---

## Repository Structure

```text
Diabetes-Prediction-ML/
│
├── DiabetesAI_Risk_Assessment.ipynb
├── diabetes.csv
├── README.md
├── requirements.txt
├── .gitignore
│
└── assets/
    └── screenshots/
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Faseeh555/Diabetes-Prediction-ML.git
cd Diabetes-Prediction-ML
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Then open:

```text
DiabetesAI_Risk_Assessment.ipynb
```

Alternatively, the notebook can be opened directly in Google Colab.

---

## Running the Project

Run the notebook cells sequentially.

The notebook performs:

1. Dataset loading
2. Data-quality inspection
3. Missing-value treatment
4. Exploratory data analysis
5. Train/test splitting
6. Model comparison
7. Stratified cross-validation
8. Hyperparameter tuning
9. Test-set evaluation
10. ROC analysis
11. Probability calibration
12. Feature importance analysis
13. SHAP explainability
14. Individual risk assessment
15. What-if scenario analysis
16. Model serialization

The trained calibrated model can be saved using:

```python
joblib.dump(calibrated_model, "diabetes_risk_model.pkl")
```

---

## Responsible AI Considerations

Healthcare-related machine learning requires careful interpretation.

This project explicitly considers the following limitations:

### Dataset Limitations

The Pima Indians Diabetes Dataset is relatively small and represents a specific population. Therefore, model performance should not automatically be generalized to other populations.

### Clinical Validation

The model has not undergone clinical validation or prospective evaluation.

### Model Probability ≠ Clinical Risk

The probability generated by the model is a statistical model output. It is not a clinically validated probability of developing diabetes.

### Bias and Fairness

Real-world deployment would require extensive evaluation across demographic and clinical subgroups to identify potential performance disparities.

### Data Quality

Medical datasets can contain missing, noisy, or incorrectly recorded values. Data preprocessing decisions can affect model performance.

### Human Oversight

A real healthcare deployment would require qualified medical professionals, appropriate validation, security controls, privacy protections, monitoring, and applicable regulatory review.

---

## Future Development

The current project provides the machine learning foundation for a broader diabetes risk-assessment platform.

Potential future improvements include:

* Interactive Streamlit dashboard
* REST API for model inference
* Patient-friendly visualization
* Individual prediction explanations
* Interactive SHAP dashboard
* More advanced model selection
* Automated model comparison
* External dataset validation
* Fairness and subgroup evaluation
* Model monitoring
* Secure database integration
* Authentication and access control
* Privacy-preserving data handling
* Integration with an AI-powered health education assistant
* Continuous model evaluation and improvement

---

## Hackathon Vision

The long-term goal of DiabetesAI is not simply to answer:

> **"Does this person have diabetes?"**

Instead, the project aims to explore a more useful and responsible question:

> **"What does the model estimate from these health indicators, how confident is it, and what factors contributed to that estimate?"**

This shifts the project from a basic classification notebook toward an **explainable and responsible AI-assisted health risk assessment platform**.

---

## Disclaimer

DiabetesAI is developed for **educational, research, and hackathon purposes**.

It is not intended to:

* Diagnose diabetes
* Replace a doctor or healthcare professional
* Prescribe medication
* Recommend medical treatment
* Provide emergency medical advice

Users should consult qualified healthcare professionals for medical decisions.

---

## Author

**Faseeh ur Rehman Anjum**

BS Computer Science Undergraduate
NUML, Islamabad

Interested in:

* Artificial Intelligence
* Machine Learning
* Explainable AI
* AI for Healthcare
* Research & Responsible AI
* Applied Machine Learning

---

## License

This project is intended for educational and research purposes. Please review the dataset's original licensing/usage terms before redistributing the dataset independently.
