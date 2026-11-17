import os
import zipfile
from pathlib import Path

import pandas as pd
from PIL import Image

from ai.config import ROOT_DIR

EXTENSOES_IMAGEM = {".png", ".jpg", ".jpeg"}
ARQUIVO_TESTE = "amostra_teste_pad_ufes_20.zip"


def _ler_raizes():
	raizes = [ROOT_DIR]
	arquivo_caminho = ROOT_DIR / "caminho_kaggle.txt"
	if arquivo_caminho.exists():
		extra = arquivo_caminho.read_text(encoding="utf-8").strip()
		if extra:
			caminho = Path(extra)
			raizes.insert(0, caminho if caminho.is_absolute() else ROOT_DIR / caminho)
	return raizes


def carregar_dataset(tentar_kaggle=True):
	"""Localiza metadata.csv e imagens locais; KaggleHub e opcional."""
	raizes = _ler_raizes()
	for tentativa in range(2 if tentar_kaggle else 1):
		metadata, imagens = None, {}
		for base in raizes:
			if not base.exists():
				continue
			for diretorio, subdirs, arquivos in os.walk(base):
				subdirs[:] = [
					nome for nome in subdirs
					if nome not in {".git", ".venv", "__pycache__", "models", ".streamlit"}
				]
				for nome in arquivos:
					caminho = Path(diretorio) / nome
					if nome == "metadata.csv" and metadata is None:
						metadata = pd.read_csv(caminho)
					elif caminho.suffix.lower() in EXTENSOES_IMAGEM:
						imagens[nome] = ("file", str(caminho))
					elif caminho.suffix.lower() == ".zip" and nome != ARQUIVO_TESTE:
						try:
							with zipfile.ZipFile(caminho, "r") as arquivo_zip:
								for item in arquivo_zip.namelist():
									nome_item = os.path.basename(item)
									if nome_item == "metadata.csv" and metadata is None:
										with arquivo_zip.open(item) as conteudo:
											metadata = pd.read_csv(conteudo)
									elif Path(nome_item).suffix.lower() in EXTENSOES_IMAGEM:
										imagens[nome_item] = ("zip", str(caminho), item)
						except (OSError, zipfile.BadZipFile):
							continue
			if metadata is not None and imagens:
				return metadata, imagens

		if tentativa == 0:
			try:
				import kagglehub
				raizes = [Path(kagglehub.dataset_download("mahdavi1202/skin-cancer"))]
			except Exception:
				break
	return None, {}


def ler_imagem(origem):
	"""Abre imagem indicada por carregar_dataset e materializa os pixels."""
	if origem[0] == "file":
		with Image.open(origem[1]) as imagem:
			return imagem.convert("RGB").copy()
	with zipfile.ZipFile(origem[1], "r") as arquivo_zip:
		with arquivo_zip.open(origem[2]) as conteudo:
			with Image.open(conteudo) as imagem:
				return imagem.convert("RGB").copy()
