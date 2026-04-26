Titanic Survival Prediction – Machine Learning Project
Project Overview

This project builds a machine learning model to predict whether a passenger survived the Titanic disaster. It follows a complete data science pipeline including data cleaning, feature engineering, feature selection, and model training.

The goal is to extract meaningful insights from passenger data and build an accurate predictive model.

 Dataset Description

The dataset contains information about Titanic passengers:

Pclass – Ticket class (1st, 2nd, 3rd)
Name – Passenger name
Sex – Gender
Age – Age of passenger
SibSp – Siblings/spouses aboard
Parch – Parents/children aboard
Fare – Ticket fare
Cabin – Cabin number
Embarked – Port of embarkation
Files Used:
train.csv – Training dataset (with survival labels)
test.csv – Test dataset (without survival labels)
 Project Workflow
1. Data Cleaning
Handled missing values in Age and Embarked
Removed irrelevant or highly missing features
Ensured dataset consistency
2. Feature Engineering

New meaningful features were created:

Title extracted from names (Mr, Mrs, Miss, etc.)
FamilySize = SibSp + Parch + 1
HasCabin (1 if cabin exists, else 0)
FareLog (log transformation of Fare)
AgeBin (categorical age groups)
3. Feature Selection

To improve model performance:

Correlation analysis was performed
Feature importance ranking was used
Recursive Feature Elimination (RFE) was applied

 Final selected features:

Pclass
Sex
Age
SibSp
Fare
HasCabin
Title
FareLog
FamilySize
4. Model Training
Machine Learning model trained using selected features
Model evaluated using accuracy score
Results
Model Accuracy: ~84.36%
 Best predictive features:
Sex
Age
Fare
Title
Pclass

The model shows strong performance for a baseline classification system.

Technologies Used
Python 
Pandas
NumPy
Scikit-learn
Matplotlib
Project Structure
Titanic_ASSIGNMENT/
│
├── data/
│   ├── train.csv
│   └── test.csv
│
├── notebooks/
│   └── Titanic_Feature_Engineering.ipynb
│
├── scripts/
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   ├── feature_selection.py
│   └── model_training.py
│
└── README.md
How to Run the Project

Run the pipeline step-by-step:

python scripts/data_cleaning.py
python scripts/feature_engineering.py
python scripts/feature_selection.py
python scripts/model_training.py
 Author

Lewis Githinji

 Conclusion

This project demonstrates a full machine learning workflow from raw data preprocessing to model training and evaluation. It highlights the importance of feature engineering and selection in improving model performance.