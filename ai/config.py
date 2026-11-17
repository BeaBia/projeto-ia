from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
MODEL_PATH = MODEL_DIR / "modelo_visao_1000.pkl"
MANIFEST_PATH = MODEL_DIR / "modelo_visao_1000.json"

IMG_SIZE = (128, 128)
CNN_SIZE = (224, 224)
MALIGNAS = {"BCC", "SCC", "MEL"}
LIMITE_POR_CLASSE = 500
USAR_AUGMENTATION = False