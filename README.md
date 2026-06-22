# Diabetes Prediction Using Machine Learning

## Overview

This project uses Machine Learning to predict whether a person is diabetic based on various health-related features such as glucose level, blood pressure, BMI, age, and insulin level. The model is built using Python and Scikit-Learn and trained on the Pima Indians Diabetes Dataset.

---

## Dataset

The dataset contains 768 patient records with 8 medical attributes and 1 target variable.

### Features

* Pregnancies
* Glucose
* Blood Pressure
* Skin Thickness
* Insulin
* BMI
* Diabetes Pedigree Function
* Age

### Target Variable

* 0 → Non-Diabetic
* 1 → Diabetic

---

## Technologies Used

* Python
* NumPy
* Pandas
* Scikit-Learn
* Google Colab
* Jupyter Notebook

---

## Machine Learning Algorithm

The project uses the **Support Vector Machine (SVM)** algorithm with a **Linear Kernel** for classification.

```python
classifier = svm.SVC(kernel='linear')
```

---

## Project Workflow

1. Import required libraries.
2. Load and analyze the diabetes dataset.
3. Perform exploratory data analysis.
4. Separate features and target labels.
5. Standardize data using StandardScaler.
6. Split data into training and testing sets.
7. Train the SVM model.
8. Evaluate model performance using accuracy score.
9. Build a predictive system for new patient data.

---

## Model Performance

### Training Accuracy

```text
78.66%
```

### Testing Accuracy

```text
77.27%
```

The model demonstrates good performance in predicting diabetes on unseen data.

---

## Sample Prediction

Input:

```text
(10,115,0,0,0,35.3,0.134,29)
```

Output:

```text
The person is diabetic.
```

---

## Repository Structure

```text
Diabetes-Prediction-ML/
│
├── Diabetes_Prediction.ipynb
├── diabetes.csv
├── README.md
└── requirements.txt
```

---

## How to Run

1. Clone the repository.

```bash
git clone https://github.com/Faseeh555/Diabetes-Prediction-ML.git
```

2. Install dependencies.

```bash
pip install numpy pandas scikit-learn
```

3. Open the notebook in Jupyter Notebook or Google Colab.

4. Run all cells sequentially.

---

## Future Improvements

* Compare multiple machine learning algorithms.
* Perform feature engineering.
* Improve model accuracy through hyperparameter tuning.
* Deploy the model as a web application using Flask or Streamlit.

---

## Author

**Faseeh ur Rehman Anjum**

BS Computer Science Student

National University of Modern Languages (NUML)

GitHub: https://github.com/Faseeh555
