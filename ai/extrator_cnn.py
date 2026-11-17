from functools import lru_cache

import numpy as np
import torch
from torchvision import transforms
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from ai.config import CNN_SIZE


@lru_cache(maxsize=1)
def carregar_cnn():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    cnn.classifier = torch.nn.Identity()
    cnn.eval().to(device)
    prep = transforms.Compose([
        transforms.Resize(CNN_SIZE, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return cnn, prep, device


def extrair_cnn(imagens, batch_size=64):
    if not imagens:
        return np.zeros((0, 576), dtype=np.float32)
    cnn, prep, device = carregar_cnn()
    tensores = [prep(imagem.convert("RGB")) for imagem in imagens]
    with torch.inference_mode():
        saidas = [
            cnn(torch.stack(tensores[i:i + batch_size]).to(device)).cpu().numpy().astype(np.float32)
            for i in range(0, len(tensores), batch_size)
        ]
    return np.vstack(saidas)