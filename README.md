#  Telco Customer Churn Prediction

## Overview
This project predicts whether a telecom customer is likely to churn (leave the company) based on customer demographics, service usage, and billing information. The model also provides churn probability scores to identify high-risk customers.

## Dataset
The dataset contains:
- Customer demographics
- Service subscriptions
- Billing information
- Tenure details
- Churn status (Target Variable)

## Exploratory Data Analysis
Performed EDA using correlation analysis and visualizations to identify patterns related to customer churn.

### Key Insight
📊 Customers who churn have **13.5% higher monthly charges on average** compared to customers who stay, indicating pricing may influence churn behavior.

## Data Preprocessing
- Handled missing values
- Applied Label Encoding and One-Hot Encoding
- Used StandardScaler for feature scaling
- Applied SMOTE to address class imbalance

## Models Used
- Logistic Regression
- XGBoost Classifier

## Results
XGBoost achieved slightly better performance than Logistic Regression and was selected as the final model.

## Technologies
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- SMOTE

## Project Workflow
1. Data Cleaning
2. Exploratory Data Analysis
3. Feature Engineering
4. Data Preprocessing
5. Model Training
6. Model Evaluation
7. Churn Prediction

## Future Improvements
- Hyperparameter tuning
- Model deployment using Flask/FastAPI
- Streamlit dashboard
- SHAP explainability

## Author
**S.Nagavamshi**
