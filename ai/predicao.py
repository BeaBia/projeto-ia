import numpy as np

from ai.extrator_cnn import extrair_cnn
from ai.preprocessamento import preparar_visualizacoes


def preparar_caracteristicas(imagem):
	geometria, imagem_box, imagem_segmentada, bbox, imagem_corrigida = preparar_visualizacoes(imagem)
	caracteristicas = np.concatenate([geometria, extrair_cnn([imagem_corrigida])[0]]).astype(np.float32)
	return caracteristicas, geometria, imagem_box, imagem_segmentada, bbox


def predizer(imagem, pacote):
	caracteristicas, geometria, imagem_box, imagem_segmentada, bbox = preparar_caracteristicas(imagem)
	normalizado = pacote["scaler"].transform([caracteristicas])
	fora_distribuicao = bool(np.percentile(np.abs(normalizado), 95) > 4.5)
	escore = float(pacote["modelo"].predict_proba(np.clip(normalizado, -3.5, 3.5))[0][1])
	limiar = float(pacote["limiar"])
	return {
		"escore": escore,
		"limiar": limiar,
		"acima_limiar": escore >= limiar,
		"fora_distribuicao": fora_distribuicao,
		"geometria": geometria,
		"imagem_box": imagem_box,
		"imagem_segmentada": imagem_segmentada,
		"bbox": bbox,
	}
