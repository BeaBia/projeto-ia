import hashlib
import json
import pickle
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from ai.avaliacao import calcular_metricas, selecionar_limiar_f2
from ai.config import (
	MALIGNAS, LIMITE_POR_CLASSE, MANIFEST_PATH, MODEL_DIR, MODEL_PATH,
	USAR_AUGMENTATION,
)
from ai.dataset import carregar_dataset, ler_imagem
from ai.extrator_cnn import extrair_cnn
from ai.preprocessamento import extrair_geometria


def _ler_item(origem):
	imagem = ler_imagem(origem)
	geometria, _, _, _, imagem_corrigida = extrair_geometria(imagem)
	return geometria, imagem_corrigida


def _sha256(caminho):
	digest = hashlib.sha256()
	with caminho.open("rb") as arquivo:
		for bloco in iter(lambda: arquivo.read(1024 * 1024), b""):
			digest.update(bloco)
	return digest.hexdigest()


def carregar_modelo():
	"""Carrega somente artefato local acompanhado de manifesto consistente."""
	if not MODEL_PATH.is_file() or not MANIFEST_PATH.is_file():
		return None
	try:
		manifesto = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
		if manifesto.get("created_by") != "ai.treinamento" or manifesto.get("artifact_sha256") != _sha256(MODEL_PATH):
			return None
		with MODEL_PATH.open("rb") as arquivo:
			return pickle.load(arquivo)
	except (OSError, ValueError, pickle.UnpicklingError, EOFError, AttributeError, ImportError, IndexError, KeyError) as erro:
		raise RuntimeError("Artefato de modelo local invalido; gere-o novamente com python -m ai.treinamento.") from erro


def treinar_modelo(forcar=False):
	"""Treina e persiste o classificador; pode ser executado sem Streamlit."""
	if not forcar:
		existente = carregar_modelo()
		if existente is not None:
			return existente

	df, imagens = carregar_dataset()
	if df is None or "img_id" not in df or "diagnostic" not in df:
		raise FileNotFoundError("PAD-UFES-20 nao encontrado: verifique data/ ou caminho_kaggle.txt.")
	df = df[df["img_id"].astype(str).isin(imagens)].reset_index(drop=True)
	if df.empty:
		raise ValueError("Nenhuma imagem do metadata.csv foi localizada.")
	if LIMITE_POR_CLASSE is not None:
		malignos = df[df["diagnostic"].isin(MALIGNAS)]
		demais = df[~df["diagnostic"].isin(MALIGNAS)]
		df = pd.concat([
			malignos.sample(n=min(len(malignos), LIMITE_POR_CLASSE), random_state=42),
			demais.sample(n=min(len(demais), LIMITE_POR_CLASSE), random_state=42),
		]).sample(frac=1.0, random_state=42).reset_index(drop=True)

	grupos = (
		df["patient_id"].fillna(df["img_id"]).astype(str).to_numpy()
		if "patient_id" in df.columns else df["img_id"].astype(str).to_numpy()
	)
	y = df["diagnostic"].isin(MALIGNAS).astype(int).to_numpy()
	if len(np.unique(y)) != 2:
		raise ValueError("O conjunto localizado precisa conter exemplos dos dois grupos binarios.")
	try:
		idx_tv, idx_te = next(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42).split(np.zeros(len(y)), y, grupos))
		r_tr, r_va = next(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42).split(np.zeros(len(idx_tv)), y[idx_tv], grupos[idx_tv]))
	except ValueError as erro:
		raise ValueError("Nao foi possivel criar divisao estratificada por paciente; confira a quantidade de grupos por classe.") from erro
	idx_tr, idx_va = idx_tv[r_tr], idx_tv[r_va]
	if len(np.unique(y[idx_tr])) != 2:
		raise ValueError("A divisao de treino nao contem ambas as classes; nao e possivel treinar o MLP.")

	origens = [imagens[img_id] for img_id in df["img_id"].astype(str)]
	with ThreadPoolExecutor(max_workers=8) as executor:
		resultados = list(executor.map(_ler_item, origens))
	geometria_base = np.vstack([resultado[0] for resultado in resultados])
	imagens_base = [resultado[1] for resultado in resultados]
	X_base = np.hstack([geometria_base, extrair_cnn(imagens_base)])

	if USAR_AUGMENTATION:
		imagens_aug, geometria_aug, y_aug = [], [], []
		for indice in idx_tr:
			imagem = imagens_base[indice]
			imagens_aug.extend([
				imagem,
				imagem.transpose(Image.Transpose.FLIP_LEFT_RIGHT),
				ImageEnhance.Brightness(imagem).enhance(1.12),
			])
			geometria_aug.extend([geometria_base[indice]] * 3)
			y_aug.extend([int(y[indice])] * 3)
		X_tr_raw = np.hstack([np.vstack(geometria_aug), extrair_cnn(imagens_aug)])
		y_tr_raw = np.array(y_aug, dtype=int)
	else:
		X_tr_raw, y_tr_raw = X_base[idx_tr], y[idx_tr]

	scaler = StandardScaler()
	X_tr = np.clip(scaler.fit_transform(X_tr_raw), -3.5, 3.5)
	X_va = np.clip(scaler.transform(X_base[idx_va]), -3.5, 3.5)
	X_te = np.clip(scaler.transform(X_base[idx_te]), -3.5, 3.5)

	rng = np.random.default_rng(42)
	positivos, negativos = np.where(y_tr_raw == 1)[0], np.where(y_tr_raw == 0)[0]
	if not len(positivos) or not len(negativos):
		raise ValueError("Balanceamento impossivel: classe ausente no treino.")
	idx_balanceado = np.concatenate(
		(negativos, rng.choice(positivos, size=len(negativos), replace=True))
		if len(positivos) < len(negativos)
		else (positivos, rng.choice(negativos, size=len(positivos), replace=True))
	)
	rng.shuffle(idx_balanceado)

	modelo = MLPClassifier(hidden_layer_sizes=(128, 32), alpha=0.10, early_stopping=True, max_iter=400, random_state=42)
	modelo.fit(X_tr[idx_balanceado], y_tr_raw[idx_balanceado])
	escores_va = modelo.predict_proba(X_va)[:, 1]
	limiar, _ = selecionar_limiar_f2(y[idx_va], escores_va)
	escores_te = modelo.predict_proba(X_te)[:, 1]

	pacote = {
		"modelo": modelo,
		"scaler": scaler,
		"limiar": limiar,
		"stats": {
			"total_imgs": len(X_base), "total_pacs": len(np.unique(grupos)),
			"tr_imgs": len(idx_tr), "tr_aug": len(X_tr_raw), "tr_pacs": len(np.unique(grupos[idx_tr])),
			"va_imgs": len(idx_va), "va_pacs": len(np.unique(grupos[idx_va])),
			"te_imgs": len(idx_te), "te_pacs": len(np.unique(grupos[idx_te])),
		},
		"metricas_validacao": calcular_metricas(y[idx_va], escores_va, limiar),
		"metricas": calcular_metricas(y[idx_te], escores_te, limiar),
	}
	MODEL_DIR.mkdir(parents=True, exist_ok=True)
	with MODEL_PATH.open("wb") as arquivo:
		pickle.dump(pacote, arquivo)
	manifesto = {
		"created_by": "ai.treinamento",
		"artifact_sha256": _sha256(MODEL_PATH),
		"warning": "Somente use artefatos produzidos localmente por este projeto.",
	}
	MANIFEST_PATH.write_text(json.dumps(manifesto, indent=2), encoding="utf-8")
	return pacote


if __name__ == "__main__":
	pacote_treinado = treinar_modelo(forcar=True)
	print(f"Treinamento concluido. Limiar de validacao: {pacote_treinado['limiar']:.2f}")
	print(f"Artefato salvo em: {MODEL_PATH}")
