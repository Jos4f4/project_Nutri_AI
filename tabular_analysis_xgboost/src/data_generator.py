import pandas as pd
import numpy as np
from scipy.stats import norm

# Dados de Referência Simplificados da OMS (IMC-por-Idade, 5-19 anos)
# OBS: ESTA É UMA SIMPLIFICAÇÃO CONCEITUAL!
# Os valores M (Mediana) e S (Desvio Padrão) são APROXIMADOS e fixos
# para demonstrar a LÓGICA do cálculo.
# M = Mediana do IMC para determinada idade e sexo
# S = Coeficiente de Variação, usado para calcular o Desvio Padrão (SD = S * M)

# Referências médias aproximadas de IMC (Peso/Altura) para crianças de 5 a 10 anos
IMC_REF_MEDIAN_MALE = 16.5  # M (Mediana)
IMC_REF_S_MALE = 0.08       # S (Coeficiente de Variação)

def calculate_zscore_simplified(age_months, imc):
    """
    Calcula um Z-score simplificado usando uma referência fixa de IMC.
    
    A implementação real requer tabelas M, S, L da OMS
    e o Z-score seria calculado com a fórmula LMS.
    Aqui, usamos a fórmula Z = (X - M) / SD com um SD fixo.
    """
    
    # Usando a referência masculina simplificada
    M = IMC_REF_MEDIAN_MALE
    S = IMC_REF_S_MALE
    SD = S * M  # Desvio Padrão (SD) simplificado
    
    if SD == 0:
        return 0
        
    z_score = (imc - M) / SD
    return z_score

def generate_synthetic_dataset(n_samples=10000):
    """
    Gera um dataset sintético de desnutrição infantil baseado em IMC e Z-score.
    """
    
    # 1. Geração de Variáveis Independentes (Features)
    
    # Idade: Simulação entre 6 e 60 meses (5 anos)
    ages_months = np.random.randint(6, 61, n_samples)
    
    # Altura (cm): Distribuição baseada na idade (com ruído)
    # A altura média aumenta com a idade.
    heights_cm = 65 + (ages_months * 0.5) + np.random.normal(0, 5, n_samples)
    
    # Peso (kg): Distribuição com foco em introduzir casos de desnutrição
    # Usamos uma distribuição skew para simular mais casos abaixo da média.
    # Peso base: 7kg + (idade em meses * 0.2)
    base_weights = 7 + (ages_months * 0.2)
    # Ruído pesado (para simular desnutrição e obesidade)
    weights_kg = base_weights + np.random.normal(-2, 4, n_samples)
    # Garantir que o peso não seja negativo
    weights_kg[weights_kg < 2] = 2 
    
    # IMC: Peso em kg / (Altura em metros)^2
    imc_values = weights_kg / (heights_cm / 100)**2
    
    # Outras features clínicas (ex: presença de Edema - 1=sim, 0=não)
    # Edema é um sinal chave de desnutrição grave (Kwashiorkor), então ele será
    # correlacionado com os casos de Z-score muito baixo.
    is_edema = np.where(imc_values < 13.5, np.random.choice([1, 0], size=n_samples, p=[0.2, 0.8]), 0)
    
    
    # 2. Cálculo do Z-score e Rótulos (Target)
    
    # Calcula o Z-score para cada paciente
    z_scores = [calculate_zscore_simplified(age, imc) for age, imc in zip(ages_months, imc_values)]
    z_scores = np.array(z_scores)

    # Função para atribuir o rótulo (TARGET) com base nas regras da OMS
    def assign_label(z):
        if z <= -3.0:
            return 3  # Grave
        elif z <= -2.0:
            return 2  # Moderada
        elif z <= -1.0:
            return 1  # Risco
        else:
            return 0  # Normal / Sobrepeso
            
    labels = np.array([assign_label(z) for z in z_scores])

    # 3. Criação do DataFrame
    
    data = pd.DataFrame({
        'age_months': ages_months,
        'weight_kg': weights_kg,
        'height_cm': heights_cm,
        'imc': imc_values,
        'z_score_imc': z_scores,
        'is_edema': is_edema,
        'diagnosis_label': labels
    })
    
    # Salva o dataset para ser usado pelo model_trainer.py
    data.to_csv('models/synthetic_nutrition_data.csv', index=False)
    
    print(f"Dataset de {n_samples} amostras gerado e salvo em 'models/synthetic_nutrition_data.csv'")
    print("\nDistribuição dos Rótulos de Diagnóstico:")
    print(data['diagnosis_label'].value_counts())
    
    return data

if __name__ == '__main__':
    # Certifica que o diretório "models" realmente existe
    import os
    if not os.path.exists('models'):
        os.makedirs('models')
        
    generate_synthetic_dataset(n_samples=10000)