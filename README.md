<img width="468" height="133" alt="image" src="https://github.com/user-attachments/assets/42b07e7f-6bc2-4f35-8582-6f3b3ef8bcac" />**Problem Statement**
Credit default prediction is a critical problem in financial risk management. Financial institutions must assess whether a customer is likely to default on their credit card payment in the upcoming month. Accurate prediction helps reduce financial losses, improve credit allocation decisions, and manage risk effectively.
This project aims to build and compare multiple machine learning classification models to predict whether a credit card client will default on payment in the next month.

The problem is formulated as a binary classification task, where:
0 → No Default
1 → Default

The objective is to evaluate model performance using multiple evaluation metrics and determine the most effective model for this dataset.

**Dataset Description**

Dataset Name: Default of Credit Card Clients Dataset
Source: UCI Machine Learning Repository

The dataset contains information about credit card clients and includes demographic data, credit information, past payment behavior, and billing/payment amounts.

Dataset Characteristics:
    Total Instances: 30,000
    Number of Input Features: 23
    Target Variable: default payment next month
    Type: Binary Classification
    Data Type: Mostly numerical (some categorical features encoded as integers)

Feature Categories:
  Demographic Features
  Gender
  Education
  Marital Status
  Age
  Credit Information
  Credit Limit
  Repayment History
  Payment status for previous 6 months
  Billing & Payment Amounts
  Bill amounts for previous 6 months
  Payment amounts for previous 6 months

The dataset satisfies the assignment requirements of having more than 500 instances and more than 12 features.

**Models Used**

The following classification models were implemented and evaluated:
a. Logistic Regression
b. Decision Tree Classifier
c. K-Nearest Neighbors (KNN)
d. Gaussian Naive Bayes
e. Random Forest (Ensemble)
f. XGBoost (Ensemble)

All models were trained using the same train-test split to ensure fair comparison. The following evaluation metrics were calculated for each model:
- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)


**Comparison Table of Evaluation Metrics**
ML Model Name	Accuracy	AUC	Precision	Recall	F1	MCC
Logistic Regression	0.8077	0.7076	0.6868	0.2396	0.3553	0.3244
Decision Tree	0.7145	0.6075	0.3694	0.4115	0.3893	0.2042
KNN	0.7928	0.7014	0.5487	0.3564	0.4322	0.3233
Naive Bayes	0.7525	0.7249	0.4515	0.5539	0.4975	0.3386
Random Forest (Ensemble)	0.8135	0.7548	0.6368	0.3647	0.4638	0.3814
XGBoost (Ensemble)	0.8167	0.7774	0.6553	0.3610	0.4655	0.3896

Observations on Model Performance
ML Model Name	Observation about Model Performance
Logistic Regression	Achieved good overall accuracy and high precision but very low recall, indicating conservative prediction of default cases. It tends to miss many actual defaulters.
Decision Tree	Showed the weakest overall performance across most metrics, likely due to overfitting and limited generalization capability.
KNN	Delivered moderate performance with balanced precision and recall but lower discrimination ability compared to ensemble models.
Naive Bayes	Achieved the highest recall among all models, indicating better sensitivity in detecting defaulters, though with reduced precision.
Random Forest (Ensemble)	Demonstrated strong performance with high AUC and MCC, indicating improved stability and generalization compared to individual models.
XGBoost (Ensemble)	Achieved the best overall performance with highest AUC and MCC. It provides superior class separation and balanced predictive capability.
Final Model Selection

Based on the comparison of evaluation metrics, XGBoost is selected as the best-performing model due to its superior AUC and MCC scores, indicating better overall classification quality and balanced performance on imbalanced data.
