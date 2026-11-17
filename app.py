import streamlit as st


st.set_page_config(page_title="Analise dermatologica experimental", page_icon="🔬", layout="wide")

paginas = {
    "Aplicacao": [
        st.Page("pages/dashboard.py", title="Visao geral", default=True),
        st.Page("pages/nova_analise.py", title="Nova analise"),
        st.Page("pages/validacao.py", title="Validacao experimental"),
    ],
    "Demonstracao": [
        st.Page("pages/pacientes.py", title="Pacientes"),
        st.Page("pages/historico.py", title="Historico"),
    ],
}
navegacao = st.navigation(paginas, position="sidebar")
st.sidebar.caption("Prototipo academico; nao destinado a diagnostico ou uso clinico.")
navegacao.run()