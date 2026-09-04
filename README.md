# DiabetesAI — Explainable Diabetes Risk Assessment

An end-to-end machine learning project for **diabetes risk assessment**, upgraded from a basic SVM notebook into a reproducible and explainable ML pipeline.

> **Responsible-use notice:** This is an educational machine-learning project, not a medical diagnostic system. It has not been clinically validated and should not be used to diagnose, treat, or replace professional medical care.

## What This Version Adds

- Data-quality audit and handling of physiologically implausible zero values
- Leakage-safe preprocessing with scikit-learn pipelines
- Stratified 5-fold cross-validation
- Comparison of Logistic Regression, Linear SVM, RBF SVM, Random Forest, and Gradient Boosting
- Random Forest hyperparameter tuning with GridSearchCV
- Evaluation using Accuracy, Precision, Recall/Sensitivity, F1, ROC-AUC, confusion matrix, and Brier score
- Probability calibration for a model-estimated risk score
- Global feature importance
- Optional SHAP explainability for individual/model behavior
- Interactive what-if scenario helper
- Reusable patient risk-assessment function
- Model serialization with Joblib for future deployment

## Dataset

The project uses the Pima Indians Diabetes Dataset containing 768 observations, eight input features, and the binary `Outcome` target.

Features:
- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age

Target:
- `0` → Non-Diabetic
- `1` → Diabetic

Some zero values in Glucose, BloodPressure, SkinThickness, Insulin, and BMI are treated as potentially missing measurements during preprocessing. The transformation is kept inside the ML pipeline to avoid data leakage.

## Project Workflow

**Patient data → Data validation → Data-quality audit → Invalid/missing-value handling → EDA → Stratified train/test split → Leakage-safe preprocessing → Model comparison with cross-validation → Hyperparameter tuning → Final test evaluation → Probability calibration → Explainability → Risk assessment → What-if analysis → Model export → Future Streamlit/API deployment**

## Notebook

The main notebook is:

`DiabetesAI_Risk_Assessment.ipynb`

It is designed to run in Google Colab or Jupyter Notebook.

## Technologies

Python, Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn, SHAP, Joblib, Jupyter/Google Colab

## Running Locally

```bash
git clone https://github.com/Faseeh555/Diabetes-Prediction-ML.git
cd Diabetes-Prediction-ML
pip install numpy pandas scikit-learn matplotlib seaborn shap joblib
```

Open `DiabetesAI_Risk_Assessment.ipynb` and run the cells sequentially.

## Repository Structure

```text
Diabetes-Prediction-ML/
├── DiabetesAI_Risk_Assessment.ipynb
├── diabetes.csv
├── README.md
└── .gitignore
```

## Limitations

- The dataset is relatively small and represents a specific population, so generalization to other populations is uncertain.
- Historical measurements contain potentially missing values encoded as zero.
- The model is not clinically validated.
- A model-estimated probability is not the same as a clinically validated individual risk percentage.
- Before real-world medical use, the system would require external validation, subgroup/fairness analysis, prospective evaluation, security review, monitoring, and appropriate clinical/regulatory review.

## Planned Product Layer

The next stage is a hackathon-ready **DiabetesAI** application with a Streamlit dashboard, patient input form, calibrated risk estimate, SHAP explanation, what-if simulator, educational health guidance, and a carefully constrained AI health assistant.

## Author

**Faseeh ur Rehman Anjum**  
BS Computer Science — National University of Modern Languages (NUML)

GitHub: https://github.com/Faseeh555
