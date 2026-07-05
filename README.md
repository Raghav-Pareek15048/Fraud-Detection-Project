# Financial Transaction Fraud Detection System

A Machine Learning-based system designed to detect fraudulent financial transactions by analyzing transaction patterns and classifying them as either **Legitimate** or **Fraudulent**. This project demonstrates a complete end-to-end machine learning pipeline, including data preprocessing, exploratory data analysis, model training, evaluation, and model persistence.

---

##  Project Overview

Financial fraud is a significant challenge in today's digital payment ecosystem. Traditional rule-based fraud detection systems often struggle to identify evolving fraud patterns. This project leverages Machine Learning algorithms to improve fraud detection accuracy while minimizing false positives and false negatives.

The project was developed as part of an internship to demonstrate practical implementation of supervised machine learning techniques for fraud detection.

---

## Objectives

* Analyze financial transaction data.
* Perform data cleaning and preprocessing.
* Handle class imbalance using SMOTE.
* Explore transaction patterns through Exploratory Data Analysis (EDA).
* Train and compare multiple Machine Learning models.
* Evaluate model performance using industry-standard metrics.
* Save the best-performing model for future predictions.

---

## Dataset

This project uses the **Credit Card Fraud Detection Dataset** from Kaggle.

**Dataset Features**

* Time
* Amount
* PCA-transformed features (V1–V28)
* Class (0 = Legitimate, 1 = Fraudulent)

> **Note:** The dataset is not included in this repository due to GitHub file size limitations. Download the dataset from Kaggle and place `creditcard.csv` inside the `data/` folder before running the notebook.

---

## Project Workflow

1. Dataset Collection
2. Data Cleaning & Preprocessing
3. Exploratory Data Analysis (EDA)
4. Feature Scaling
5. Handling Class Imbalance using SMOTE
6. Model Training
7. Model Evaluation
8. Model Comparison
9. Model Persistence using Joblib
10. Fraud Prediction

---

## Machine Learning Models

The following models were implemented and evaluated:

* Logistic Regression
* Random Forest Classifier
* XGBoost Classifier

---

## Evaluation Metrics

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC Score
* Confusion Matrix
* Classification Report

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Imbalanced-learn (SMOTE)
* XGBoost
* Joblib
* Google Colab
* GitHub

---

## Project Structure

```text
financial-transaction-fraud-detection/
│
├── data/               # Dataset (not included)
├── models/             # Saved trained models
├── notebooks/          # Google Colab/Jupyter notebooks
├── reports/            # Literature review and reports
├── src/                # Source code (if applicable)
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/financial-transaction-fraud-detection.git
```

Navigate to the project directory:

```bash
cd financial-transaction-fraud-detection
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Download the Kaggle dataset and place `creditcard.csv` inside the `data/` directory.

Run the notebook located in the `notebooks/` folder using Jupyter Notebook or Google Colab.

---

## Future Enhancements

* Real-time fraud detection
* Deep Learning-based models
* Streamlit web application
* REST API using Flask/FastAPI
* Cloud deployment

---

## Team Members

* Raghav Pareek
* Aryan Gupta
* Harsh Sinha
* Ayushi Sinha

---

## License

This project was developed for educational and internship purposes.
