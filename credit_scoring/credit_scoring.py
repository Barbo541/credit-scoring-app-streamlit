import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
import io
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

st.set_page_config(page_title="Credit Scoring App", layout="wide")

@st.cache_data
def carregar_modelo():
    with open("model_final.pkl", "rb") as f:
        return pickle.load(f)

selecao = st.sidebar.selectbox("📑 Selecione uma etapa do projeto:", [
    "📊 Análises - 📥 Upload e Previsão",
    "📊 Análises - 📊 Resultados",
    "📊 Análises - 📂 Análise dos Dados",
    "📈 Métricas - 🔢 Métricas",
    "📈 Métricas - 📈 Curva ROC",
    "📈 Métricas - 📉 Explicabilidade",
    "📈 Métricas - 📣 Insights Estratégicos",
    "📘 Informações - 📚 Sobre o Modelo"
])

selecao = selecao.split("-", 1)[-1].strip()

# 📥 Upload e Previsão
if selecao == "📥 Upload e Previsão":
    st.subheader("📥 Upload e Previsão")
    uploaded_file = st.file_uploader("Envie a base de dados (.ftr)", type=["ftr"])
    if uploaded_file is not None:
        try:
            df_raw = pd.read_feather(uploaded_file)
            st.success(f"Base carregada com sucesso: {df_raw.shape[0]} linhas × {df_raw.shape[1]} colunas")
            st.dataframe(df_raw.head())
            df_input = df_raw.copy()

            if 'educacao' in df_input.columns:
                df_input['educacao'] = df_input['educacao'].replace({
                    'Fundamental': 'Baixa', 'Médio': 'Média', 'Superior incompleto': 'Média',
                    'Superior completo': 'Alta', 'Pós graduação': 'Alta'
                })
            if 'tipo_renda' in df_input.columns:
                df_input['tipo_renda'] = df_input['tipo_renda'].replace({
                    'Assalariado': 'Trabalho', 'Servidor público': 'Trabalho',
                    'Empresário': 'Trabalho', 'Pensionista': 'Aposentadoria', 'Bolsista': 'Outros'
                })

            colunas_remover = ['index', 'data_ref', 'mau']
            X_modelo = df_input.drop(columns=[col for col in colunas_remover if col in df_input.columns])

            modelo = carregar_modelo()
            pred = modelo.predict(X_modelo)
            proba = modelo.predict_proba(X_modelo)[:, 1]

            df_resultado = df_raw.copy()
            df_resultado['score'] = proba
            df_resultado['previsao_mau'] = pred

            st.session_state['df_resultado'] = df_resultado
            st.session_state['modelo'] = modelo

            st.success("Previsões geradas com sucesso! Vá para as próximas abas para visualizar resultados.")
        except Exception as e:
            st.error(f"Erro ao processar o arquivo: {e}")

# 📊 Resultados
elif selecao == "📊 Resultados":
    if 'df_resultado' in st.session_state:
        df = st.session_state['df_resultado']
        st.subheader("📊 Resultados da Previsão")
        st.dataframe(df[['score', 'previsao_mau']].join(df.drop(columns=['score', 'previsao_mau'])).head())
        hue_col = 'mau' if 'mau' in df.columns else 'previsao_mau'
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.histplot(data=df, x='score', hue=hue_col, bins=30, kde=True, ax=ax, stat='percent')
        ax.set_title("Distribuição de Scores de Risco")
        st.pyplot(fig)
    else:
        st.warning("Você precisa gerar as previsões na aba 📥 Upload e Previsão.")

# 📂 Análise dos Dados
elif selecao == "📂 Análise dos Dados":
    if 'df_resultado' in st.session_state:
        df = st.session_state['df_resultado']
        st.subheader("📂 Análise Exploratória dos Dados")
        col1, col2 = st.columns(2)
        with col1:
            fig_idade, ax1 = plt.subplots()
            sns.histplot(df['idade'], bins=30, kde=True, ax=ax1)
            ax1.set_title("Distribuição de Idade")
            st.pyplot(fig_idade)
        with col2:
            fig_renda, ax2 = plt.subplots()
            sns.histplot(df['renda'], bins=30, kde=True, ax=ax2)
            ax2.set_title("Distribuição de Renda")
            st.pyplot(fig_renda)

        if 'mau' in df.columns:
            col3, col4 = st.columns(2)
            with col3:
                fig1, ax3 = plt.subplots()
                sns.barplot(data=df, x='educacao', y='mau', ax=ax3)
                ax3.set_title("Inadimplência por Escolaridade")
                st.pyplot(fig1)
            with col4:
                fig2, ax4 = plt.subplots()
                sns.barplot(data=df, x='tipo_renda', y='mau', ax=ax4)
                ax4.set_title("Inadimplência por Tipo de Renda")
                st.pyplot(fig2)
            fig3, ax5 = plt.subplots()
            sns.boxplot(data=df, x='mau', y='tempo_emprego', ax=ax5)
            ax5.set_title("Tempo de Emprego por Classe")
            st.pyplot(fig3)

# 🔢 Métricas
elif selecao == "🔢 Métricas":
    if 'df_resultado' in st.session_state:
        df = st.session_state['df_resultado']
        threshold = st.slider("Escolha o threshold (limiar):", 0.0, 1.0, 0.5, 0.01)
        pred_bin = (df['score'] >= threshold).astype(int)
        if 'mau' in df.columns:
            cm = confusion_matrix(df['mau'], pred_bin)
            report = classification_report(df['mau'], pred_bin, output_dict=True, zero_division=0)
            st.text("Matriz de Confusão")
            st.write(cm)
            st.text("Relatório de Classificação Completo")
            st.text(classification_report(df['mau'], pred_bin))

            st.markdown("### 📊 Principais Métricas:")
            st.markdown(f"- **Accuracy:** {report['accuracy']:.2f}")
            st.markdown(f"- **Precision (mau=1):** {report.get('1', {}).get('precision', 0.0):.2f}")
            st.markdown(f"- **Recall (mau=1):** {report.get('1', {}).get('recall', 0.0):.2f}")
            st.markdown(f"- **F1-score (mau=1):** {report.get('1', {}).get('f1-score', 0.0):.2f}")

# 📈 Curva ROC
elif selecao == "📈 Curva ROC":
    if 'df_resultado' in st.session_state and 'mau' in st.session_state['df_resultado'].columns:
        df = st.session_state['df_resultado']
        fpr, tpr, _ = roc_curve(df['mau'], df['score'])
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}", color='blue')
        ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
        ax.set_title("Curva ROC")
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.legend()
        st.pyplot(fig)
    else:
        st.warning("Você precisa gerar as previsões primeiro e garantir que a coluna 'mau' esteja presente na base.")

# 📉 Explicabilidade
elif selecao == "📉 Explicabilidade":
    if 'modelo' in st.session_state:
        modelo = st.session_state['modelo']
        etapa_modelo = None
        for nome, etapa in modelo.named_steps.items():
            if hasattr(etapa, 'coef_'):
                etapa_modelo = etapa
                break
        if etapa_modelo is not None:
            st.subheader("📉 Coeficientes da Regressão Logística")
            coef = etapa_modelo.coef_[0]
            features = etapa_modelo.feature_names_in_ if hasattr(etapa_modelo, 'feature_names_in_') else [f"Var{i}" for i in range(len(coef))]
            coef_df = pd.DataFrame({'Variável': features, 'Peso': coef}).sort_values(by='Peso', key=abs, ascending=False)
            st.dataframe(coef_df)
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=coef_df, x='Peso', y='Variável', palette='coolwarm', ax=ax)
            ax.set_title("Importância das Variáveis")
            st.pyplot(fig)
        else:
            st.warning("Modelo não possui coeficientes disponíveis.")

# 📣 Insights Estratégicos
elif selecao == "📣 Insights Estratégicos":
    if 'df_resultado' in st.session_state:
        df = st.session_state['df_resultado']
        if 'mau' in df.columns:
            inad_pct = df['mau'].mean() * 100
            score_mau = df[df['mau'] == 1]['score'].mean()
            score_bom = df[df['mau'] == 0]['score'].mean()
            st.markdown(f"- **Inadimplência geral:** {inad_pct:.2f}%")
            st.markdown(f"- **Score médio inadimplentes:** {score_mau:.2f}")
            st.markdown(f"- **Score médio adimplentes:** {score_bom:.2f}")
            if 'tipo_renda' in df.columns:
                st.markdown("**Risco por tipo de renda:**")
                for tipo, pct in df.groupby('tipo_renda')['mau'].mean().sort_values(ascending=False).items():
                    st.markdown(f"- {tipo}: {pct*100:.2f}%")
            if 'educacao' in df.columns:
                st.markdown("**Risco por escolaridade:**")
                for edu, pct in df.groupby('educacao')['mau'].mean().sort_values(ascending=False).items():
                    st.markdown(f"- {edu}: {pct*100:.2f}%")
        else:
            st.warning("A coluna 'mau' não está disponível para gerar os insights estratégicos.")

# 📚 Sobre o Modelo
elif selecao == "📚 Sobre o Modelo":
    st.subheader("📚 Sobre o Modelo")
    st.markdown("""
    Este modelo foi treinado com Regressão Logística usando um pipeline completo com:
    - Imputação de dados
    - Normalização
    - Redução de dimensionalidade com PCA
    - Classificação supervisionada com LogisticRegression

    O objetivo é prever a probabilidade de inadimplência (`mau = 1`) com base em variáveis socioeconômicas.
    """)
    if 'df_resultado' in st.session_state:
        buffer = io.BytesIO()
        st.session_state['df_resultado'].to_csv(buffer, index=False)
        st.download_button("⬇️ Baixar resultado (.csv)", buffer.getvalue(), "resultado.csv", mime="text/csv")
    if st.button("🔁 Resetar análise"):
        st.session_state.clear()
        st.rerun()




