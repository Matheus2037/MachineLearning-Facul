import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib  # Import para salvar e carregar o modelo

# Carregar o modelo treinado
try:
    loaded_model = joblib.load('gradient_boosting_model.pkl')
    print("Modelo carregado com sucesso!")
except Exception as e:
    print(f"Erro ao carregar o modelo: {e}")

# Carregar os dados
df_binary = pd.read_csv('diabetes_binary_health_indicators_BRFSS2015.csv')

# Remover colunas irrelevantes
columns_to_remove = ['PhysActivity', 'Fruits', 'Veggies', 'AnyHealthcare', 'NoDocbcCost', 'Smoker']
X = df_binary.drop(columns=columns_to_remove + ['Diabetes_binary'])
y = df_binary['Diabetes_binary']

# Função para categorizar o IMC (BMI)
def categorize_bmi(bmi):
    if bmi < 18.5:
        return 'Underweight'
    elif 18.5 <= bmi <= 24.9:
        return 'Normal'
    elif 25 <= bmi <= 29.9:
        return 'Overweight'
    else:
        return 'Obese'

# Criar a coluna de categorias de BMI
X['BMI_Category'] = X['BMI'].apply(categorize_bmi)

# Transformar a coluna de BMI em variáveis dummy
X = pd.get_dummies(X, columns=['BMI_Category'], prefix='BMI', drop_first=True)

# Criar grupos de BMI
def simple_bmi_grouping(bmi):
    if bmi < 25:
        return 1
    elif 25 <= bmi <= 30:
        return 2
    else:
        return 3

X['BMI_Group'] = X['BMI'].apply(simple_bmi_grouping)

# Interação entre BMI e idade
X['BMI_Age_Interaction'] = X['BMI'] * X['Age']

# Divisão dos dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Verificar se o modelo foi carregado corretamente
if 'loaded_model' in globals():
    # Fazer previsões com o modelo carregado
    y_pred_gb = loaded_model.predict(X_test)

    # Avaliar o modelo
    accuracy_gb = accuracy_score(y_test, y_pred_gb)
    conf_matrix_gb = confusion_matrix(y_test, y_pred_gb)
    class_report_gb = classification_report(y_test, y_pred_gb)

    print("\nGradient Boosting Classifier Accuracy:", accuracy_gb)
    print("Confusion Matrix:\n", conf_matrix_gb)
    print("Classification Report:\n", class_report_gb)
else:
    print("Modelo não carregado! Certifique-se de ter treinado e salvo o modelo corretamente.")

# Função para preparar os dados do paciente para previsão
def prepare_patient_data(input_data, X_train_columns):
    # Processar as colunas do paciente
    input_data['BMI_Category'] = input_data['BMI'].apply(categorize_bmi)
    input_data = pd.get_dummies(input_data, columns=['BMI_Category'], prefix='BMI', drop_first=True)

    # Criar a coluna de grupo de BMI
    input_data['BMI_Group'] = input_data['BMI'].apply(simple_bmi_grouping)

    # Criar a coluna de interação entre BMI e idade
    input_data['BMI_Age_Interaction'] = input_data['BMI'] * input_data['Age']

    # Garantir que todas as colunas do X_train estejam presentes
    for col in X_train_columns:
        if col not in input_data.columns:
            input_data[col] = 0  # Adicionar coluna ausente com valor 0

    # Garantir que as colunas estejam na mesma ordem que no X_train
    input_data = input_data[X_train_columns]

    return input_data


# Exemplo de um novo paciente (dados fornecidos)
new_patient = pd.DataFrame({
    'HighBP': [1],
    'HighChol': [1],
    'CholCheck': [1],
    'BMI': [30.0],
    'MentHlth': [30.0],
    'PhysHlth': [30.0],
    'DiffWalk': [1],
    'Sex': [0],
    'Age': [9],  # Faixa etária (70-74 anos)
    'Education': [5],
    'Income': [1]
})

# Garantir que as colunas do modelo de treino (X_train) sejam carregadas corretamente
columns_to_remove = ['PhysActivity', 'Fruits', 'Veggies', 'AnyHealthcare', 'NoDocbcCost', 'Smoker']
X = df_binary.drop(columns=columns_to_remove + ['Diabetes_binary'])
X['BMI_Category'] = X['BMI'].apply(categorize_bmi)
X = pd.get_dummies(X, columns=['BMI_Category'], prefix='BMI', drop_first=True)
X['BMI_Group'] = X['BMI'].apply(simple_bmi_grouping)
X['BMI_Age_Interaction'] = X['BMI'] * X['Age']
X_train_columns = X.columns  # Colunas do conjunto de treino

# Processar o novo paciente
prepared_patient = prepare_patient_data(new_patient, X_train_columns)

# Fazer a previsão para o novo paciente
if 'loaded_model' in globals():
    prediction = loaded_model.predict(prepared_patient)
    print("\nPredição para novo paciente (0 = Não diabético, 1 = Diabético):", prediction)
    proba = loaded_model.predict_proba(prepared_patient)
    print("Probabilidade (classe 0.0, classe 1.0):", proba)
else:
    print("Modelo não carregado. A previsão não pode ser feita.")