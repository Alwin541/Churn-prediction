# Churn-prediction
This project builds an end-to-end Churn Prediction Machine Learning Model using the BankChurners dataset to identify customers who are likely to leave. The goal is to help businesses take proactive retention actions and improve customer lifetime value.
🎯 Objectives

Perform Exploratory Data Analysis (EDA)

Clean and preprocess the dataset

Handle categorical & numerical features

Check skewness, distribution, outliers

Apply feature engineering

Train ML models (e.g., Random Forest, XGBoost, Logistic Regression)

Evaluate performance using accuracy, F1-score, ROC-AUC

Predict customer churn for test cases

🧠 Dataset Information

The dataset used: BankChurners.csv

It includes customer details such as:

Customer Age

Gender

Dependent count

Education & Income categories

Credit Limit

Card Category

Months on Book

Total Transactions & Total Revolving Balance

Attrition Flag (Target Variable)

📊 Exploratory Data Analysis (EDA)

Key insights observed from the notebook:

Customer Age is nearly normally distributed with a slight right skew

Income and card-related features show right skewness

Outliers were identified and handled where necessary

Categorical features were encoded

Correlation heatmap was used to identify important features

⚙️ Feature Engineering

Handled missing values

Encoded categorical variables (One-Hot / Label Encoding)

Scaled numerical features

Treated skewness using log1p / Box-Cox (where needed)

Balanced data using SMOTE (if class imbalance existed)

🤖 Machine Learning Models Used

Models evaluated in this project:

support vector machines

Random Forest Classifier


📈 Model Evaluation

Typical metrics used:

Accuracy

Precision, Recall, F1-score

ROC-AUC

Confusion Matrix

The final selected model gives a reliable prediction of whether a customer is likely to churn.

📦 Project Structure
├── Churn_prediction.ipynb   # Jupyter Notebook with all steps
├── BankChurners.csv         # Dataset
├── README.md                # Project description

🚀 How to Run the Project
1️⃣ Clone the Repository
git clone https://github.com/yourusername/Churn-Prediction-ML.git
cd Churn-Prediction-ML

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Run the Notebook
jupyter notebook Churn_prediction.ipynb

📌 Results

The trained model successfully predicts churn with good accuracy and can be integrated into dashboards, CRM systems, or automated retention systems.

🛠️ Tools & Technologies

Python

Pandas, NumPy

Matplotlib, Seaborn

Scikit-learn

Jupyter Notebook

📢 Future Improvements

Deploy the model using Flask/FastAPI

Build a Streamlit dashboard

Integrate with real-time customer data

Hyperparameter tuning with Optuna

👤 Author

Alwin Joseph
Machine Learning & AI Developer
