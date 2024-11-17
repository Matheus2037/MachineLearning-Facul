import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df_012 = pd.read_csv('diabetes_012_health_indicators_BRFSS2015.csv')
df_binary = pd.read_csv('diabetes_binary_health_indicators_BRFSS2015.csv')
df_5050 = pd.read_csv('diabetes_binary_5050split_health_indicators_BRFSS2015.csv')

columns_to_remove = ['PhysActivity', 'Fruits', 'Veggies', 'AnyHealthcare', 'NoDocbcCost', 'Smoker']
X = df_binary.drop(columns=columns_to_remove + ['Diabetes_binary'])
y = df_binary['Diabetes_binary']

def categorize_bmi(bmi):
    if bmi < 18.5:
        return 'Underweight'
    elif 18.5 <= bmi <= 24.9:
        return 'Normal'
    elif 25 <= bmi <= 29.9:
        return 'Overweight'
    else:
        return 'Obese'

X['BMI_Category'] = X['BMI'].apply(categorize_bmi)

X = pd.get_dummies(X, columns=['BMI_Category'], prefix='BMI', drop_first=True)

def simple_bmi_grouping(bmi):
    if bmi < 25:
        return 1
    elif 25 <= bmi <= 30:
        return 2
    else:
        return 3

X['BMI_Group'] = X['BMI'].apply(simple_bmi_grouping)

X['BMI_Age_Interaction'] = X['BMI'] * X['Age']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

gb_clf = GradientBoostingClassifier(learning_rate=0.05, max_depth=4, n_estimators=300, subsample=1.0, random_state=42)
gb_clf.fit(X_train, y_train)

y_pred_gb = gb_clf.predict(X_test)

accuracy_gb = accuracy_score(y_test, y_pred_gb)
conf_matrix_gb = confusion_matrix(y_test, y_pred_gb)
class_report_gb = classification_report(y_test, y_pred_gb)

print("\nGradient Boosting Classifier Accuracy:", accuracy_gb)
print("Confusion Matrix:\n", conf_matrix_gb)
print("Classification Report:\n", class_report_gb)