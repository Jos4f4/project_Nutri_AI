import pandas as pd
import xgboost as xgb
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Configurações e Caminhos
MODEL_PATH = 'models/xgboost_nutrition_model.json'

# Mapeamento dos rótulos numéricos para os nomes das categorias
LABEL_MAP = {
    0: "Normal / Sem Risco",
    1: "Risco de Desnutrição",
    2: "Desnutrição Moderada",
    3: "Desnutrição Grave"
}

# Carregamento do Modelo 
try:
    # Cria a instância do modelo e carrega o arquivo JSON
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    print(f"Modelo XGBoost carregado com sucesso de: {MODEL_PATH}")
except Exception as e:
    print(f"ERRO: Falha ao carregar o modelo de '{MODEL_PATH}'.")
    print("Certifique-se de que 'model_trainer.py' foi executado.")
    raise e

# 2. Definição da API e Modelo de Dados

# Inicializa o aplicativo FastAPI
app = FastAPI(
    title="NutriAI Tabular Analysis Service",
    description="Serviço REST para diagnóstico tabular preliminar de desnutrição usando XGBoost."
)

# Modelo Pydantic para validar e tipar os dados de entrada
class TabularDataInput(BaseModel):
    # As features que o modelo XGBoost espera (definidas no model_trainer.py)
    age_months: int = Field(..., description="Idade da criança em meses (6-60).")
    z_score_imc: float = Field(..., description="Z-score de IMC (calculado previamente pelo Spring Boot).")
    is_edema: int = Field(..., description="Presença de edema (1=sim, 0=não).")
    
# Modelo Pydantic para a resposta de saída
class DiagnosisOutput(BaseModel):
    preliminary_diagnosis: str = Field(..., description="O diagnóstico preliminar em texto.")
    risk_category_id: int = Field(..., description="O ID da categoria de risco (0-3).")


# 3. Endpoint REST 
@app.post("/api/v1/diagnose/tabular", response_model=DiagnosisOutput)
async def diagnose_tabular(data: TabularDataInput):
    """
    Recebe dados tabulares (principalmente Z-score) e retorna o diagnóstico preliminar.
    Este resultado será enviado ao Gemini para análise multimodal.
    """
    try:
        # Converte os dados de entrada Pydantic em um formato que o XGBoost entende (DataFrame)
        input_data = {
            'age_months': [data.age_months],
            'z_score_imc': [data.z_score_imc],
            'is_edema': [data.is_edema]
        }
        df = pd.DataFrame(input_data)
        
        # Faz a predição usando o modelo carregado
        prediction_id = model.predict(df)[0]
        
        # Mapeia o ID numérico para o rótulo de texto
        diagnosis_text = LABEL_MAP.get(prediction_id, "Desconhecido")
        
        # Retorna a resposta conforme o modelo DiagnosisOutput
        return DiagnosisOutput(
            preliminary_diagnosis=diagnosis_text,
            risk_category_id=int(prediction_id)
        )
        
    except Exception as e:
        # Em caso de erro na predição
        raise HTTPException(status_code=500, detail=f"Erro interno ao processar a predição: {e}")

# Adiciona um endpoint de health check
@app.get("/health")
def health_check():
    """Verifica se o serviço está ativo."""
    return {"status": "ok", "service": "NutriAI XGBoost Predictor"}