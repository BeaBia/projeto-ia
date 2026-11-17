import pandas as pd
import streamlit as st

from ai.config import MODEL_DIR
from services.analise_service import obter_modelo_disponivel


st.title("Validacao experimental")
pacote = None
try:
    pacote = obter_modelo_disponivel()
except RuntimeError as erro:
    st.error(str(erro))

if pacote is None:
    st.info(f"Nenhum modelo local disponivel. Execute `python -m ai.treinamento`; artefatos serao salvos em `{MODEL_DIR}`.")
else:
    stats = pacote["stats"]
    metricas = pacote["metricas"]
    st.warning(
        "Resultados experimentais no PAD-UFES-20, nao constituem validacao clinica. "
        "A classe negativa agrega todos os diagnosticos fora de BCC/SCC/MEL, inclusive queratose actinica."
    )
    st.subheader("Particoes por paciente")
    st.dataframe(pd.DataFrame([
        {"Conjunto": "Treino", "Imagens": f"{stats['tr_imgs']} ({stats['tr_aug']} usadas)", "Pacientes": stats["tr_pacs"]},
        {"Conjunto": "Validacao (calibracao do limiar)", "Imagens": stats["va_imgs"], "Pacientes": stats["va_pacs"]},
        {"Conjunto": "Teste cego", "Imagens": stats["te_imgs"], "Pacientes": stats["te_pacs"]},
    ]), hide_index=True, use_container_width=True)
    st.subheader("Metricas na validacao")
    metricas_validacao = pacote["metricas_validacao"]
    auc_validacao = "Indisponivel: validacao com uma classe" if metricas_validacao["auc"] is None else f"{metricas_validacao['auc']:.3f}"
    st.dataframe(pd.DataFrame([{
        "ROC-AUC": auc_validacao,
        "Sensibilidade": f"{metricas_validacao['sens'] * 100:.1f}%",
        "Especificidade": f"{metricas_validacao['esp'] * 100:.1f}%",
        "Precisao": f"{metricas_validacao['prec'] * 100:.1f}%",
        "F1": f"{metricas_validacao['f1']:.3f}",
    }]), hide_index=True, use_container_width=True)
    st.subheader("Metricas no teste cego")
    auc = "Indisponivel: teste com uma classe" if metricas["auc"] is None else f"{metricas['auc']:.3f}"
    st.dataframe(pd.DataFrame([{
        "ROC-AUC": auc,
        "Sensibilidade": f"{metricas['sens'] * 100:.1f}%",
        "Especificidade": f"{metricas['esp'] * 100:.1f}%",
        "Precisao": f"{metricas['prec'] * 100:.1f}%",
        "F1": f"{metricas['f1']:.3f}",
        "Limiar (validacao)": f"{pacote['limiar']:.2f}",
    }]), hide_index=True, use_container_width=True)
    st.subheader("Matriz de confusao")
    st.dataframe(pd.DataFrame(
        metricas["cm"],
        index=["Real: demais diagnosticos", "Real: BCC/SCC/MEL"],
        columns=["Predito: abaixo do limiar", "Predito: acima do limiar"],
    ), use_container_width=True)
    st.caption("Escores do MLP nao foram calibrados como probabilidades clinicas.")