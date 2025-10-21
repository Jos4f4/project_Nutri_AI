import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import os

# Configurações e Caminhos
DATA_PATH = 'models/synthetic_nutrition_data.csv'
MODEL_PATH = 'models/xgboost_nutrition_model.json'

def train_and_save_model():
    """
    Carrega os dados sintéticos, treina o modelo XGBoost e o salva em formato JSON.
    """
    if not os.path.exists(DATA_PATH):
        print(f"ERRO: Arquivo de dados não encontrado em {DATA_PATH}")
        print("Certifique-se de que você executou 'data_generator.py' primeiro.")
        return

    # 1. Carregar e Preparar os Dados
    data = pd.read_csv(DATA_PATH)
    
    # As Features (X) para o modelo. 
    # Focamos no Z-score e na presença de edema, que são os indicadores chaves.
    FEATURES = ['age_months', 'z_score_imc', 'is_edema']
    TARGET = 'diagnosis_label'
    
    X = data[FEATURES]
    y = data[TARGET]
    
    # 2. Divisão em Conjunto de Treino e Teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2,       # 20% para teste
        random_state=42, 
        stratify=y           # Garante que a proporção de classes seja mantida
    )
    
    print(f"Dados de Treino: {len(X_train)} amostras")
    print(f"Dados de Teste: {len(X_test)} amostras")
    
    # 3. Inicializar e Treinar o Modelo XGBoost
    model = xgb.XGBClassifier(
        objective='multi:softmax',  # Multi-class classification
        num_class=len(y.unique()),  # Número de classes (0, 1, 2, 3)
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        use_label_encoder=False,
        eval_metric='merror',       # Metric for multi-class error rate
        random_state=42
    )
    
    print("\nIniciando Treinamento do XGBoost...")
    model.fit(X_train, y_train)
    print("Treinamento concluído.")
    
    # 4. Avaliação do Modelo
    y_pred = model.predict(X_test)
    
    print("\n--- Relatório de Avaliação ---")
    print(f"Acurácia no Conjunto de Teste: {accuracy_score(y_test, y_pred):.4f}")
    
    # Relatório detalhado por classe (Normal, Risco, Moderada, Grave)
    print("\nRelatório de Classificação (Precision/Recall/F1-Score):")
    print(classification_report(y_test, y_pred, 
                                target_names=['Normal', 'Risco', 'Moderada', 'Grave']))
                                
    # 5. Salvar o Modelo Treinado
    # salvo no formato JSON
    model.save_model(MODEL_PATH)
    print(f"\nModelo XGBoost salvo com sucesso em: {MODEL_PATH}")

if __name__ == '__main__':
    train_and_save_model()