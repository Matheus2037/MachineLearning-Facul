import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df_012 = pd.read_csv('diabetes_012_health_indicators_BRFSS2015.csv')
df_binary = pd.read_csv('diabetes_binary_health_indicators_BRFSS2015.csv')
df_5050 = pd.read_csv('diabetes_binary_5050split_health_indicators_BRFSS2015.csv')

features_to_remove = ['CholCheck', 'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'Stroke']
X = df_binary.drop(['Diabetes_binary'] + features_to_remove, axis=1)
y = df_binary['Diabetes_binary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=30,
    min_samples_leaf=2,
    min_samples_split=10,
    random_state=42
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("Model Evaluation After Further Feature Removal:")
print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)

importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)
print("\nFeature Importances (After Further Removal):")
print(importances)