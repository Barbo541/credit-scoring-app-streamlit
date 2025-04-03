# 📊 Credit Scoring App com Streamlit

✅ **Acesse o app online agora:**  
🔗 [https://credit-scoring-app-streamlit.onrender.com](https://credit-scoring-app-streamlit.onrender.com)

---

Este projeto realiza uma análise de risco de crédito utilizando um modelo de regressão logística treinado com dados públicos. Desenvolvido com Streamlit, ele é ideal para fins de demonstração e portfólio.

### 🧠 Funcionalidades principais:
- Upload de dados no formato `.ftr`
- Carregamento de uma base de exemplo leve
- Geração de previsões de inadimplência (`mau=1`)
- Visualização de métricas, curva ROC e explicabilidade com coeficientes do modelo
- Interface com navegação por seções e filtros interativos

### 📁 Estrutura do repositório:
```
📦 credit-scoring-app-streamlit
├── credit_scoring.py              # Código principal do app Streamlit
├── model_final.pkl                # Modelo final treinado com sklearn
├── credit-score-exemplo-pequeno.ftr  # Base de exemplo otimizada
├── requirements.txt              # Dependências do projeto
├── demo/
│   └── streamlit-demo.webm       # Gravação da demo do app (opcional)
└── notebooks/
    └── CreditScoring.Proj.ipynb # Notebook de análise e construção do modelo
```

### 🚀 Tecnologias utilizadas:
- `Python`, `pandas`, `scikit-learn`, `Streamlit`
- Visualizações com `matplotlib` e `seaborn`

---

📌 **Observações**:
- A base completa foi substituída por uma amostra menor para melhor performance no deploy.
- O modelo utiliza transformações como PCA, padronização e agrupamento de categorias.

Sinta-se à vontade para clonar, testar localmente ou fazer fork para adaptar ao seu portfólio!

---

🔗 Repositório original: [github.com/Barbo541/credit-scoring-app-streamlit](https://github.com/Barbo541/credit-scoring-app-streamlit)
