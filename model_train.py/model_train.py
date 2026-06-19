import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Criando dados fictícios para o exemplo
np.random.seed(42)
data_size = 1000
X = np.random.rand(data_size, 4) # 4 features: tempo_app, suporte_aberto, compras, idade_conta
y = (X[:, 0] * 0.4 - X[:, 1] * 0.8 + X[:, 2] * 0.3 > 0).astype(int)

# 2. Pré-processamento
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 3. Construindo o Modelo TensorFlow
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(4,)),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 4. Treinamento
model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)

# 5. Salvar o modelo e o scaler
model.save('modelo_churn.h5')
import pickle
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Modelo e Scaler salvos com sucesso!")




import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
import requests  # Para chamar APIs de IA, se necessário

# Configuração da página
st.set_page_config(page_title="Preditor de Churn com IA", layout="centered")
st.title("📊 Sistema Inteligente de Retenção de Clientes")

# Carregar modelo e scaler
@st.cache_resource
def load_resources():
    model = tf.keras.models.load_model('modelo_churn.h5')
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

try:
    model, scaler = load_resources()
except:
    st.error("Por favor, execute o script de treinamento primeiro para gerar o modelo.")

# Formulário de entrada na interface
st.subheader("Insira os dados do cliente:")
tempo_app = st.slider("Tempo diário no App (minutos)", 0, 120, 30)
suporte = st.number_input("Chamados de suporte abertos no mês", min_value=0, max_value=10, value=1)
compras = st.number_input("Número de compras realizadas", min_value=0, max_value=50, value=5)
idade_conta = st.slider("Idade da conta (meses)", 1, 60, 12)

if st.button("Analisar Risco"):
    # Organizar dados para o modelo
    input_data = np.array([[tempo_app/120, suporte/10, compras/50, idade_conta/60]]) # Normalização simples para o exemplo
    
    # Predição do TensorFlow
    probabilidade = model.predict(input_data)[0][0]
    
    st.write("---")
    if probabilidade > 0.5:
        st.error(f"⚠️ Alerta: Alto Risco de Cancelamento! ({probabilidade*100:.2f}%)")
        
        # Simulação da resposta do Gemini/ChatGPT via Prompt estruturado
        st.subheader("💡 Estratégia de Retenção Recomendada (Gerada por IA):")
        
        # Exemplo de resposta que seria gerada pelo Prompt do Framework
        st.info(f"""
        **Ações para o Cliente:**
        1. **Desconto de Urgência:** Oferecer 20% de desconto na próxima mensalidade devido aos {suporte} chamados de suporte abertos.
        2. **Engajamento:** Enviar cupom exclusivo baseado nas suas {compras} compras já realizadas.
        3. **Suporte VIP:** Agendar uma ligação de um gerente de contas para resolver atritos.
        """)
    else:
        st.success(f"✅ Cliente Saudável. Baixo risco de cancelamento. ({probabilidade*100:.2f}%)")