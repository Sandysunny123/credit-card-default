1.Problem Statement

    The objective of this project is to predict whether a credit card customer will default on their payment in the next month based on historical financial and demographic data.
    This is a binary classification problem where the target variable indicates default (Yes/No).

2.Dataset Description

    Dataset Name: Default of Credit Card Clients Dataset
    
    Source: Public dataset (Kaggle)
    
    Number of Records: 30,000
    
    Number of Features: 23
    
    Target Variable:
    
    default.payment.next.month
    
    1 → Default
    
    0 → No Default
    
    Feature Categories:
    
    Demographic features: Gender, Education, Marital Status, Age
    
    Financial features: Credit limit, bill amounts, payment amounts
    
    Payment history: Past payment status for previous months
    
    The dataset was manually downloaded and included in the repository to ensure reproducibility during evaluation.

3.Models Used and Evaluation Metrics

    All models were trained and evaluated on the same dataset using an 80–20 train-test split.
    Feature scaling was applied where required.
    
    Evaluation Metrics:
    
    Accuracy
    
    AUC (Area Under ROC Curve)
    
    Precision
    
    Recall
    
    F1 Score
    
    Matthews Correlation Coefficient (MCC)
    
    Model Comparison Table

    | ML Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------|----------|-----|-----------|--------|----|-----|
| Logistic Regression | 0.8080 | 0.7078 | 0.6882 | 0.2411 | 0.3571 | 0.3261 |
| Decision Tree | 0.7233 | 0.6092 | 0.3817 | 0.4047 | 0.3928 | 0.2140 |
| KNN | 0.7935 | 0.6942 | 0.5530 | 0.3459 | 0.4256 | 0.3204 |
| Naive Bayes | 0.7523 | 0.7251 | 0.4513 | 0.5554 | 0.4980 | 0.3391 |
| Random Forest (Ensemble) | 0.8137 | 0.7549 | 0.6370 | 0.3662 | 0.4651 | 0.3824 |
| XGBoost (Ensemble) | 0.8090 | 0.7590 | 0.6162 | 0.3617 | 0.4558 | 0.3676 |



4.Model Performance Observations
 
   ML Model	Observation

   **Logistic Regression**	Provided a strong baseline with good accuracy and interpretability but lower recall for default cases.
   **Decision Tree** Easy to interpret but showed signs of overfitting and comparatively lower overall performance.
   **KNN**	Performed reasonably well after feature scaling but was sensitive to the choice of distance metric.
   **Naive Bayes**	Achieved higher recall, making it suitable for identifying defaulters, though precision was lower.
   **Random Forest**	Delivered the best overall balance across metrics due to ensemble learning and reduced overfitting.
   **XGBoost**	Achieved the highest AUC score, indicating strong discriminatory power for default prediction. 
   
5.Project Structure

   credit-card-default/
   │── app.py
   │── requirements.txt
   │── README.md
   │── data/
   │   └── credit-card.csv
   │── model/
      ├── train_utils.py
      ├── logistic.py
      ├── decision_tree.py
      ├── knn.py
      ├── naive_bayes.py
      ├── random_forest.py
      ├── xgboost.py
      └── saved/
          ├── scaler.pkl
          ├── logistic.pkl
          ├── decision_tree.pkl
          ├── knn.pkl
          ├── naive_bayes.pkl
          ├── random_forest.pkl
          └── xgboost.pkl


6.Streamlit Application

    The project includes an interactive Streamlit web application with the following features:
    
    CSV file upload (test data)
    
    Model selection dropdown
    
    Display of evaluation metrics
    
    Confusion matrix / classification report
    
    The application is deployed using Streamlit Community Cloud.

7.Technologies Used

    Python 3.10
    Pandas, NumPy
    Scikit-learn
    XGBoost
    Streamlit
    GitHub

8.How to Run the Project Locally

    conda activate ml2
    python app.py
    streamlit run app.py


9. Conclusion

This project demonstrates an end-to-end machine learning workflow, including data preprocessing, model training, evaluation, modular code design, and deployment using Streamlit.
Ensemble models such as Random Forest and XGBoost showed superior performance in predicting credit card defaults.