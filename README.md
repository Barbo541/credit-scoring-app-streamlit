# 📊 Credit Scoring App com Streamlit

Este é um aplicativo interativo para análise de risco de crédito, construído com Streamlit, usando um modelo de regressão logística treinado com PyCaret.

O objetivo é prever a probabilidade de inadimplência (`mau = 1`) com base em dados socioeconômicos dos clientes e fornecer visualizações claras, métricas de performance e explicabilidade do modelo.

---

## ⚙️ Funcionalidades

- Upload de base de dados `.ftr` com informações dos clientes
- Previsão de inadimplência com base em um modelo treinado
- Visualização de resultados e scores
- Análise exploratória dos dados
- Métricas de classificação (Accuracy, Recall, F1)
- Curva ROC e AUC
- Explicabilidade via coeficientes da regressão
- Insights estratégicos por tipo de renda e escolaridade
- Baixar resultado e resetar análise

---

## 🎥 Demonstração do App

<!-- Substitua o link abaixo pelo seu vídeo -->
[▶️ Clique aqui para assistir à demonstração](https://www.youtube.com/watch?v=SEU_VIDEO_ID)

---

## 🚀 Como rodar localmente

```bash
# Clone o repositório
git clone https://github.com/Barbo541/credit-scoring-app-streamlit.git
cd credit-scoring-app-streamlit

# (Recomendado) Crie um ambiente virtual
conda create -n credit-env python=3.10 -y
conda activate credit-env

# Instale as dependências
pip install -r requirements.txt

# Rode o app
streamlit run app_credit_scoring.py
```

---

## ☁️ Deploy no Render

1. Faça login no [Render](https://render.com)
2. Crie um novo projeto a partir do seu repositório GitHub
3. Use o seguinte comando de start:  
```bash
streamlit run app_credit_scoring.py
```
4. Certifique-se de que o arquivo `requirements.txt` está no repositório

---

## 🧪 Tecnologias Utilizadas

- [Streamlit](https://streamlit.io/)
- [Scikit-learn](https://scikit-learn.org/)
- [PyCaret (no notebook)](https://pycaret.org/)
- Pandas, Seaborn, Matplotlib

---

## 📁 Organização dos Arquivos

```
.
├── app_credit_scoring.py       # Código principal do app
├── model_final.pkl             # Modelo treinado com regressão logística
├── requirements.txt            # Dependências do projeto
├── .gitignore                  # Arquivos ignorados no versionamento
└── README.md                   # Documentação
```

---

## 📄 Licença

Este projeto foi desenvolvido para fins educacionais como parte de um portfólio de ciência de dados.