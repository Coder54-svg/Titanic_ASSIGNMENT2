import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from data_cleaning import clean_data
from feature_engineering import engineer_features

# Load data
train = pd.read_csv('data/train.csv')

# Process pipeline
cleaned = clean_data(train)
engineered = engineer_features(cleaned)

# Select features
X = engineered[['Sex', 'Age', 'FareLog', 'Title', 'Pclass', 'HasCabin', 'FamilySize', 'Embarked']]
y = engineered['Survived']

# Encode categorical variables
X = pd.get_dummies(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")