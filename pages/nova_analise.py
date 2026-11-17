import streamlit as st
from PIL import Image, UnidentifiedImageError

from services.analise_service import analisar_imagem, obter_modelo_disponivel


st.title("Nova analise de imagem")
st.warning(
	"Uso exclusivamente experimental. O resultado nao diagnostica nem descarta cancer; "
	"nao substitui avaliacao dermatologica. O grupo negativo pode conter queratose actinica pre-cancerosa."
)
try:
	pacote = obter_modelo_disponivel()
except RuntimeError as erro:
	st.error(str(erro))
	pacote = None

if pacote is None:
	st.info("Modelo treinado nao encontrado. Para treinar localmente, execute `python -m ai.treinamento` no terminal.")
else:
	arquivo = st.file_uploader("Selecione uma fotografia", type=["png", "jpg", "jpeg", "webp"])
	if arquivo is not None:
		try:
			with Image.open(arquivo) as imagem:
				original = imagem.convert("RGB")
			resultado = analisar_imagem(original, pacote)
		except (OSError, UnidentifiedImageError, ValueError, RuntimeError) as erro:
			st.error(f"Nao foi possivel processar a imagem: {erro}")
		else:
			colunas = st.columns(3)
			colunas[0].image(original, caption="Imagem enviada", use_container_width=True)
			colunas[1].image(resultado["imagem_box"], caption=f"Bounding box {resultado['bbox']}", use_container_width=True)
			colunas[2].image(resultado["imagem_segmentada"], caption="Mascara experimental", use_container_width=True)

			if resultado["fora_distribuicao"]:
				st.warning("Imagem potencialmente fora da distribuicao de treino (iluminacao ou enquadramento atipico).")
			st.metric("Escore do modelo (nao calibrado)", f"{resultado['escore'] * 100:.1f}%")
			st.caption(f"Limiar experimental selecionado na validacao: {resultado['limiar']:.2f}")
			if resultado["acima_limiar"]:
				st.error("Escore acima do limiar do grupo BCC/SCC/MEL. Isto nao e um diagnostico.")
			else:
				st.info("Escore abaixo do limiar. Isto nao comprova benignidade nem descarta cancer.")

			geometria = resultado["geometria"]
			medidas = st.columns(4)
			medidas[0].metric("Assimetria (A)", f"{(geometria[8] + geometria[9]) / 2:.3f}")
			medidas[1].metric("Borda irregular (B)", f"{geometria[7]:.2f}")
			medidas[2].metric("Variacao de cor (C)", f"{geometria[3:6].mean():.3f}")
			medidas[3].metric("Area segmentada", f"{geometria[6] * 100:.1f}%")
