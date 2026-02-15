# customer_churn_prediction
Machine learning project predicting telecom customer churn using Random Forest with preprocessing, SMOTE balancing, and performance evaluation.

---

## Project Overview
Customer churn prediction helps companies identify customers likely to leave their service. This project builds a machine learning model to predict churn using telecom customer data such as services subscribed, billing details, and contract information.

The final model helps businesses take proactive measures to retain customers and reduce revenue loss.

---

## Problem Statement
The goal is to predict whether a telecom customer will churn (leave the service) based on their service usage and billing patterns.

Target variable:
Churn → Yes / No

---

## Dataset Description
The dataset includes customer information such as:

### Customer Information
- gender
- SeniorCitizen
- Partner
- Dependents

### Service Details
- tenure
- PhoneService
- MultipleLines
- InternetService
- OnlineSecurity
- OnlineBackup
- DeviceProtection
- TechSupport
- StreamingTV
- StreamingMovies

### Billing Details
- Contract
- PaperlessBilling
- PaymentMethod
- MonthlyCharges
- TotalCharges

### Target Variable
- Churn

---

## Workflow
The project pipeline includes:

1. Data loading and inspection
2. Data cleaning and preprocessing
3. Encoding categorical variables
4. Handling class imbalance using SMOTE
5. Train-test data split
6. Model training using:
   - Decision Tree
   - Random Forest
   - XGBoost
7. Model comparison
8. Selecting Random Forest as final model
9. Model evaluation
10. Saving trained model for reuse

---

## Model Evaluation
Model performance evaluated using:
- Accuracy Score
- Confusion Matrix
- Precision
- Recall
- F1 Score

Random Forest achieved the best performance among tested models.

---

## Model Saving
The trained model and encoders are saved as:

customer_churn_model.pkl  
encoders.pkl  

This allows predictions without retraining.

---

## Installation & Usage
Clone repository:

git clone <repo-link>  
cd customer_churn_prediction  

Install dependencies:

pip install -r requirements.txt  

Run notebook:

jupyter notebook  

or open in Google Colab.

---

## Business Impact
The system can help businesses:
- Identify customers likely to churn
- Improve retention strategies
- Reduce revenue loss

---

## Future Improvements
Potential improvements include:
- Hyperparameter tuning
- Model deployment via web app
- Feature importance analysis
- Real-time prediction system

---

## Technologies Used
Python libraries:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- imbalanced-learn
- xgboost
- pickle

Tools:
- Jupyter Notebook / Google Colab
- GitHub

---

## Project Structure
customer_churn_prediction/

notebooks/ → churn_prediction.ipynb  
models/ → customer_churn_model.pkl, encoders.pkl  
data/ → dataset.csv  
requirements.txt  
README.md  

---

## Author
Naga Vaishnavi  
Aspiring Data Analyst & Data Scientist
---
