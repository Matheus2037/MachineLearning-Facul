import warnings

warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

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

# Modelo Gradient Boosting
gb_clf = GradientBoostingClassifier(learning_rate=0.05, max_depth=4, n_estimators=300, subsample=1.0, random_state=42)
gb_clf.fit(X_train, y_train)


# Função para prever diabetes (produção)
def predict_diabetes(input_data):
    """
    Função para prever se o paciente tem diabetes (1) ou não (0).

    Parâmetros:
        input_data (pd.DataFrame): DataFrame contendo as características do paciente.

    Retorna:
        List[int]: Lista com as previsões (0 ou 1).
    """
    # Processar o input_data no mesmo formato dos dados de treino
    input_data['BMI_Category'] = input_data['BMI'].apply(categorize_bmi)
    input_data = pd.get_dummies(input_data, columns=['BMI_Category'], prefix='BMI', drop_first=True)

    # Garantir que todas as colunas do treino estão no input_data
    for col in X_train.columns:
        if col not in input_data.columns:
            input_data[col] = 0

    # Ordenar as colunas
    input_data = input_data[X_train.columns]

    # Fazer a previsão
    predictions = gb_clf.predict(input_data)
    return predictions


# Teste do modelo em produção
# Exemplo de um novo paciente
new_patient = pd.DataFrame({
    'HighBP': [0],
    'HighChol': [0],
    'CholCheck': [0],
    'BMI': [45.0],
    'MentHlth': [15],
    'PhysHlth': [30],
    'DiffWalk': [1],
    'Sex': [0],
    'Age': [9],
    'Education': [2],
    'Income': [3]
})

# Prever para o novo paciente
prediction = predict_diabetes(new_patient)
print("Predição para novo paciente (0 = Não diabético, 1 = Diabético):", prediction)
