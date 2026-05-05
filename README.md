   # 📊 Customer Churn Prediction System

![Python](https://img.shields.io/badge/Python-3.9-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-green)
![Streamlit](https://img.shields.io/badge/App-Streamlit-red)
![Status](https://img.shields.io/badge/Project-Production%20Ready-brightgreen)

An end-to-end Machine Learning system built to predict telecom customer churn and recommend data-driven retention strategies.

## 🚀 Project Overview

Customer churn directly impacts company revenue. This project analyzes customer behavior, billing patterns, service usage, and contract types to identify customers at high risk of leaving.

Two classification models were trained and evaluated:
- Logistic Regression  
- XGBoost Classifier  

The final solution includes a deployed Streamlit web application for real-time churn prediction.

---

## 🧠 Business Value

✔ Identify high-risk customers early  
✔ Enable targeted retention campaigns  
✔ Reduce revenue loss  
✔ Support data-driven decision making  

---

## 🛠 Tech Stack

Python • Pandas • NumPy • Scikit-learn • XGBoost • Matplotlib • Seaborn • Streamlit  

---

## 📂 Dataset

Telco Customer Churn Dataset  
Source: https://www.kaggle.com/blastchar/telco-customer-churn  

---
## 🤖 Machine Learning Modeling
I trained two models to compare a baseline approach against a high-performance gradient boosting approach.

| **Model**               | Accuracy | Precision | Recall | F1-Score |ROC-AUC|


| **Logistic Regression** | 80%      | 0.65      | 0.55   | 0.60     |0.9    |
| **XGBoost Classifier**  | 82%      | 0.68      | 0.58   | 0.62     |0.85   |

**Why XGBoost?** Although accuracy is similar, XGBoost handled non-linear relationships in "Tenure" and "Monthly Charges" much better than Logistic Regression.



## 💡 Retention Strategies (Business Value)
Based on the model's feature importance, I recommend the following actions:

1. **Convert Month-to-Month to Annual:** Offer a 10% discount for customers switching to a 1-year contract.
2. **Target High-Charge Users:** Customers with billing anomalies (sudden spikes) should receive automated "Check-in" emails from customer support.
3. **Tech Support Upselling:** Churn was lower among users with "Tech Support" enabled. Bundle this service for high-risk customers.


### "How to Run"
## 🚀 Getting Started

1. **Clone the repo:**
   git clone https://github.com/glorey638-cyber/customer-churn-prediction.git

2. **Install dependencies:**
   pip install -r requirements.txt

 3. **Explore the Analysis:**
Open notebooks/5_business_insights.ipynb to see the full data science workflow.

