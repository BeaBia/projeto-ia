import streamlit as st


st.title("Visao geral")
st.write("Prototipo academico para exploracao experimental de imagens do PAD-UFES-20.")
st.info(
	"Esta aplicacao nao realiza diagnostico, nao descarta cancer e nao substitui avaliacao "
	"por profissional de saude. Os resultados sao experimentais e nao validados para uso clinico."
)
st.markdown(
	"**Alvo atual:** BCC, SCC e MEL versus demais diagnosticos. "
	"O grupo de comparacao pode incluir queratose actinica, uma condicao pre-cancerosa; "
	"por isso, um resultado abaixo do limiar nao significa lesao benigna."
)
