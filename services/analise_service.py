from ai.config import MANIFEST_PATH, MODEL_PATH


def obter_modelo_disponivel():
	"""Retorna o modelo local confiado ou None; nunca inicia treinamento."""
	if not MODEL_PATH.is_file() or not MANIFEST_PATH.is_file():
		return None
	try:
		from ai.treinamento import carregar_modelo
	except ModuleNotFoundError as erro:
		raise RuntimeError("Instale as dependencias de requirements.txt para carregar o modelo.") from erro
	return carregar_modelo()


def analisar_imagem(imagem, pacote_modelo):
	try:
		from ai.predicao import predizer
	except ModuleNotFoundError as erro:
		raise RuntimeError("Instale as dependencias de requirements.txt para executar a analise.") from erro
	return predizer(imagem, pacote_modelo)
