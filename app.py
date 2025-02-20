import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression  # Para previsão com Machine Learning

# Configuração da página
st.set_page_config(page_title="Previsão de Vendas - FMP", layout="wide")

# Aplicação de estilos personalizados
st.markdown("""
    <style>
        body {
            font-family: 'Arial', sans-serif;
        }
        .title {
            font-size: 34px;
            font-weight: bold;
            color: #1E90FF;
            margin-bottom: 25px;
            margin-top: 25px;
        }
        .section {
            font-size: 22px;
            font-weight: bold;
            color: #1E90FF;
            margin-top: 50px;
            margin-bottom: 15px;
        }
        .description {
            font-size: 16px;
            color: #2F4F4F;
            margin-bottom: 10px;
        }
        .stNumberInput>div {
            width: 12rem !important;
        }
        .aligned-container {
            display: flex;
            flex-direction: column;
            align-items: left;
            gap: 5px;
        }
    </style>
""", unsafe_allow_html=True)

# **Título e Introdução**
st.markdown('<p class="title">📊 Simulador de Previsão de Vendas - Função de Massa de Probabilidade (FMP)</p>', unsafe_allow_html=True)

st.markdown('<p class="description">A <b>Função de Massa de Probabilidade (FMP)</b> é uma técnica estatística utilizada para prever eventos <b>discretos</b>, como a quantidade de vendas diárias de um produto no varejo. Com base em dados históricos, conseguimos calcular a probabilidade de cada cenário e tomar decisões estratégicas sobre <b>estoque, promoções e reposição de produtos</b>.</p>', unsafe_allow_html=True)

# **Fórmula**
st.markdown('<p class="section">📌 Como calcular a probabilidade?</p>', unsafe_allow_html=True)
st.latex(r"""
P(X = k) = \frac{\text{Número de dias com } X = k}{\text{Total de dias analisados}}
""")

# **Exemplo Prático**
st.markdown('<p class="section">✏ Exemplo Prático</p>', unsafe_allow_html=True)
st.markdown("""
<p class="description">
Se analisarmos as vendas de um <b>tênis específico</b> nos últimos <b>20 dias</b>, temos:
</p>
<ul>
    <li>🔴 <b>0 vendas</b> em 2 dias</li>
    <li>🔴 <b>1 venda</b> em 5 dias</li>
    <li>🔴 <b>2 vendas</b> em 6 dias</li>
    <li>🔴 <b>3 vendas</b> em 4 dias</li>
    <li>🔴 <b>4 vendas</b> em 3 dias</li>
</ul>
""", unsafe_allow_html=True)

st.write("Agora, queremos calcular a **probabilidade de vender exatamente 2 pares de tênis em um dia**. Substituímos os valores na fórmula:")

st.latex(r"""
P(X = 2) = \frac{6}{20} = 0.30
""")

st.write("Isso significa que há uma **chance de 30%** de vender exatamente **2 pares de tênis** em um dia.")

# **Seção de Entrada de Dados**
st.markdown('<p class="section">📝 Insira os dados históricos de vendas</p>', unsafe_allow_html=True)
st.write("Preencha os campos abaixo com a quantidade de dias que ocorreram cada número de vendas:")

# Criando layout responsivo e ajustando largura dos inputs
valores = [0, 1, 2, 3, 4, 5]
frequencias = []

for valor in valores:
    st.markdown(f'<div class="aligned-container"><p class="description"><b>Quantidade de dias com {valor} vendas:</b></p></div>', unsafe_allow_html=True)
    freq = st.number_input("", min_value=0, value=10, step=1, key=f"freq_{valor}")
    frequencias.append(freq)

# **Cálculo da FMP**
total_dias = sum(frequencias)
probabilidades = [freq / total_dias if total_dias > 0 else 0 for freq in frequencias]

# **Exibição dos Resultados**
st.markdown('<p class="section">🎯 Distribuição de Probabilidade das Vendas</p>', unsafe_allow_html=True)
resultado = {valores[i]: f"{probabilidades[i]*100:.2f}%" for i in range(len(valores))}
st.write(resultado)

# **🔹 Machine Learning para análise e sugestões**
st.markdown('<p class="section">🧠 Análise e Sugestão com Machine Learning</p>', unsafe_allow_html=True)

# Transformando os dados para ML
X = np.array(valores).reshape(-1, 1)  # Transformando em matriz 2D
y = np.array(probabilidades)

# Criando e treinando um modelo de regressão linear
if total_dias > 0:
    model = LinearRegression()
    model.fit(X, y)

    # Predição da tendência futura
    next_sale = max(valores) + 1
    predicted_prob = model.predict([[next_sale]])[0] * 100

    # Exibir recomendação
    st.write(f"📈 Se a tendência continuar, a **probabilidade estimada de vender {next_sale} pares** pode ser de aproximadamente **{predicted_prob:.2f}%**.")
    
    # Analisando se as vendas estão aumentando ou diminuindo
    trend = "aumentando 📊" if model.coef_[0] > 0 else "diminuindo 📉"
    st.write(f"🔍 A tendência das suas vendas parece estar **{trend}** com o tempo.")
else:
    st.write("⚠️ Insira dados para ver sugestões com IA.")

# **Gráfico da FMP - Mais Compacto e Elegante**
st.markdown('<p class="section">📊 Gráfico - Função de Massa de Probabilidade</p>', unsafe_allow_html=True)

fig, ax = plt.subplots(figsize=(4, 2.5))  # Gráfico mais compacto
ax.bar(valores, probabilidades, color='#1E90FF', alpha=0.8, edgecolor='#2F4F4F')
ax.set_xlabel("Quantidade de Vendas", fontsize=10)
ax.set_ylabel("Probabilidade", fontsize=10)
ax.set_title("Distribuição de Probabilidade", fontsize=12)
ax.tick_params(axis='both', labelsize=8)  # Reduzindo os textos dos eixos

# Exibir gráfico no Streamlit
st.pyplot(fig)
