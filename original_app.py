"""Copia de referencia do script monolitico recebido; nao e usado pelo app."""

import io
import os
import pickle
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image, ImageDraw, ImageEnhance
from scipy import ndimage
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from torchvision import transforms
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

IMG_SIZE = (128, 128)
CNN_SIZE = (224, 224)
MALIGNAS = {"BCC", "SCC", "MEL"}
LIMITE_POR_CLASSE = 500
USAR_AUGMENTATION = False
CACHE_PATH = "modelo_visao_1000.pkl"


def extrair_geometria(imagem):
    arr = np.asarray(imagem.convert("RGB").resize(IMG_SIZE, Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    norm = np.power(np.mean(np.power(np.maximum(arr, 1e-4), 6), axis=(0, 1)), 1.0 / 6.0)
    illum = norm / np.maximum(np.linalg.norm(norm), 1e-6)
    arr = np.clip(arr * ((1.0 / np.sqrt(3.0)) / np.maximum(illum, 1e-4)), 0.0, 1.0).astype(np.float32)
    suave = ndimage.gaussian_filter(arr, sigma=(1.2, 1.2, 0))
    bordas = np.vstack([suave[:10].reshape(-1, 3), suave[-10:].reshape(-1, 3), suave[:, :10].reshape(-1, 3), suave[:, -10:].reshape(-1, 3)])
    dist = np.linalg.norm(suave - np.median(bordas, axis=0), axis=2)
    hist, edges = np.histogram(dist.ravel(), bins=256)
    centers = 0.5 * (edges[:-1] + edges[1:])
    w1 = np.cumsum(hist)
    w2 = w1[-1] - w1
    csum = np.cumsum(hist * centers)
    limiar = float(centers[np.argmax(w1[:-1] * w2[:-1] * (csum[:-1] / np.maximum(w1[:-1], 1) - (csum[-1] - csum[:-1]) / np.maximum(w2[:-1], 1)) ** 2)])
    mask = ndimage.binary_closing(dist > limiar, structure=np.ones((5, 5)))
    mask = ndimage.binary_opening(ndimage.binary_fill_holes(mask), structure=np.ones((5, 5)))
    rotulos, n_obj = ndimage.label(mask)
    if n_obj > 0:
        mask = rotulos == (int(np.argmax(ndimage.sum(mask, rotulos, range(1, n_obj + 1)))) + 1)
    else:
        mask = np.zeros(IMG_SIZE, dtype=bool)
        mask[32:96, 32:96] = True
    ys, xs = np.where(mask)
    bbox = (max(0, int(xs.min()) - 4), max(0, int(ys.min()) - 4), min(127, int(xs.max()) + 4), min(127, int(ys.max()) + 4)) if len(xs) > 0 else (24, 24, 104, 104)
    area = float(np.sum(mask)) / float(IMG_SIZE[0] * IMG_SIZE[1])
    perim = float(np.sum(mask ^ ndimage.binary_erosion(mask, structure=np.ones((3, 3)))))
    borda_irreg = min((perim ** 2) / (4.0 * np.pi * max(float(np.sum(mask)), 1.0)), 3.0)
    mf = mask.astype(np.float32)
    assim_h, assim_v = float(np.mean(np.abs(mf - np.fliplr(mf)))), float(np.mean(np.abs(mf - np.flipud(mf))))
    pixels = arr[mask] if np.any(mask) else arr.reshape(-1, 3)
    vetor = np.concatenate([pixels.mean(axis=0), pixels.std(axis=0), [area, borda_irreg, assim_h, assim_v]]).astype(np.float32)
    return vetor, arr, mask, bbox, Image.fromarray((arr * 255.0).astype(np.uint8))


@st.cache_resource(show_spinner=False)
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
    cnn, prep, device = carregar_cnn()
    tensores = [prep(img) for img in imagens]
    with torch.inference_mode():
        out = [cnn(torch.stack(tensores[i:i + batch_size]).to(device)).cpu().numpy().astype(np.float32) for i in range(0, len(tensores), batch_size)]
    return np.vstack(out) if out else np.zeros((0, 576), dtype=np.float32)


def analisar_imagem(imagem):
    geom, arr, mask, bbox, img_corr = extrair_geometria(imagem)
    feats = np.concatenate([geom, extrair_cnn([img_corr])[0]]).astype(np.float32)
    img_box = imagem.convert("RGB").resize(IMG_SIZE, Image.Resampling.BILINEAR)
    ImageDraw.Draw(img_box).rectangle(bbox, outline=(255, 50, 50), width=3)
    overlay = arr.copy()
    contorno = mask ^ ndimage.binary_erosion(mask, structure=np.ones((3, 3)))
    overlay[mask] = overlay[mask] * 0.65 + np.array([0.0, 0.75, 0.85]) * 0.35
    overlay[contorno] = np.array([1.0, 0.9, 0.0])
    return feats, geom, img_box, Image.fromarray((np.clip(overlay, 0, 1) * 255).astype(np.uint8)), bbox


def carregar_dataset():
    raizes = [Path.cwd()]
    if os.path.exists("caminho_kaggle.txt"):
        extra = Path("caminho_kaggle.txt").read_text(encoding="utf-8").strip()
        if extra:
            raizes.insert(0, Path(extra))
    for tentativa in range(2):
        df, imgs = None, {}
        for base in raizes:
            if not base.exists():
                continue
            for arq in base.rglob("*"):
                if not arq.is_file() or arq.name == "amostra_teste_pad_ufes_20.zip":
                    continue
                if arq.name == "metadata.csv" and df is None:
                    df = pd.read_csv(arq)
                elif arq.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                    imgs[arq.name] = ("file", str(arq))
                elif arq.suffix.lower() == ".zip":
                    with zipfile.ZipFile(arq, "r") as zf:
                        for item in zf.namelist():
                            nome = os.path.basename(item)
                            if nome == "metadata.csv" and df is None:
                                with zf.open(item) as f:
                                    df = pd.read_csv(f)
                            elif nome.lower().endswith((".png", ".jpg", ".jpeg")):
                                imgs[nome] = ("zip", str(arq), item)
            if df is not None and imgs:
                return df, imgs
        if tentativa == 0:
            try:
                import kagglehub
                raizes = [Path(kagglehub.dataset_download("mahdavi1202/skin-cancer"))]
            except Exception:
                break
    return None, {}


def _ler_item(args):
    raw, path = args
    try:
        with Image.open(io.BytesIO(raw) if raw is not None else path) as img:
            geom, _, _, _, img_corr = extrair_geometria(img)
            return geom, img_corr
    except Exception:
        return np.zeros(10, dtype=np.float32), Image.new("RGB", IMG_SIZE, (128, 128, 128))


@st.cache_resource(show_spinner=False)
def treinar_modelo():
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "rb") as f:
            return pickle.load(f)
    df, imgs = carregar_dataset()
    if df is None:
        st.error("Base PAD-UFES-20 nao encontrada.")
        st.stop()
    df = df[df["img_id"].astype(str).isin(imgs)].reset_index(drop=True)
    if LIMITE_POR_CLASSE is not None:
        malignos = df[df["diagnostic"].isin(MALIGNAS)]
        benignos = df[~df["diagnostic"].isin(MALIGNAS)]
        df = pd.concat([
            malignos.sample(n=min(len(malignos), LIMITE_POR_CLASSE), random_state=42),
            benignos.sample(n=min(len(benignos), LIMITE_POR_CLASSE), random_state=42),
        ]).sample(frac=1.0, random_state=42).reset_index(drop=True)
    grupos = df["patient_id"].fillna(df["img_id"]).astype(str).to_numpy() if "patient_id" in df.columns else df["img_id"].astype(str).to_numpy()
    y = df["diagnostic"].isin(MALIGNAS).astype(int).to_numpy()
    idx_tv, idx_te = next(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42).split(np.zeros(len(y)), y, grupos))
    r_tr, r_va = next(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42).split(np.zeros(len(idx_tv)), y[idx_tv], grupos[idx_tv]))
    idx_tr, idx_va = idx_tv[r_tr], idx_tv[r_va]
    zips, tarefas = {}, []
    for img_id in df["img_id"].astype(str):
        origem = imgs[img_id]
        if origem[0] == "file":
            tarefas.append((None, origem[1]))
        else:
            zips.setdefault(origem[1], zipfile.ZipFile(origem[1], "r"))
            tarefas.append((zips[origem[1]].read(origem[2]), None))
    for zf in zips.values():
        zf.close()
    with ThreadPoolExecutor(max_workers=8) as ex:
        res = list(ex.map(_ler_item, tarefas))
    geom_base = np.vstack([r[0] for r in res])
    imgs_base = [r[1] for r in res]
    X_base = np.hstack([geom_base, extrair_cnn(imgs_base)])
    if USAR_AUGMENTATION:
        aug_imgs, aug_geom, aug_y = [], [], []
        for i in idx_tr:
            im = imgs_base[i]
            aug_imgs.extend([im, im.transpose(Image.Transpose.FLIP_LEFT_RIGHT), ImageEnhance.Brightness(im).enhance(1.12)])
            aug_geom.extend([geom_base[i]] * 3)
            aug_y.extend([int(y[i])] * 3)
        X_tr_raw = np.hstack([np.vstack(aug_geom), extrair_cnn(aug_imgs)])
        y_tr_raw = np.array(aug_y, dtype=int)
    else:
        X_tr_raw, y_tr_raw = X_base[idx_tr], y[idx_tr]
    scaler = StandardScaler()
    X_tr = np.clip(scaler.fit_transform(X_tr_raw), -3.5, 3.5)
    X_va = np.clip(scaler.transform(X_base[idx_va]), -3.5, 3.5)
    X_te = np.clip(scaler.transform(X_base[idx_te]), -3.5, 3.5)
    rng = np.random.default_rng(42)
    pos, neg = np.where(y_tr_raw == 1)[0], np.where(y_tr_raw == 0)[0]
    idx_bal = np.concatenate([neg, rng.choice(pos, size=len(neg), replace=True)]) if len(pos) < len(neg) else np.concatenate([pos, rng.choice(neg, size=len(pos), replace=True)])
    rng.shuffle(idx_bal)
    modelo = MLPClassifier(hidden_layer_sizes=(128, 32), alpha=0.10, early_stopping=True, max_iter=400, random_state=42)
    modelo.fit(X_tr[idx_bal], y_tr_raw[idx_bal])
    probs_va = modelo.predict_proba(X_va)[:, 1]
    limiar, melhor_f2 = 0.50, -1.0
    for t in np.linspace(0.25, 0.65, 41):
        preds = (probs_va >= t).astype(int)
        rec = recall_score(y[idx_va], preds, zero_division=0)
        prec = precision_score(y[idx_va], preds, zero_division=0)
        f2 = (5.0 * prec * rec) / (4.0 * prec + rec) if (prec + rec) > 0 else 0.0
        if f2 > melhor_f2:
            melhor_f2, limiar = f2, float(t)
    probs_te = modelo.predict_proba(X_te)[:, 1]
    preds_te = (probs_te >= limiar).astype(int)
    tn, fp, fn, tp = confusion_matrix(y[idx_te], preds_te, labels=[0, 1]).ravel()
    pacote = {
        "modelo": modelo,
        "scaler": scaler,
        "limiar": limiar,
        "stats": {"total_imgs": len(X_base), "total_pacs": len(np.unique(grupos)), "tr_imgs": len(idx_tr), "tr_aug": len(X_tr_raw), "tr_pacs": len(np.unique(grupos[idx_tr])), "va_imgs": len(idx_va), "va_pacs": len(np.unique(grupos[idx_va])), "te_imgs": len(idx_te), "te_pacs": len(np.unique(grupos[idx_te]))},
        "metricas": {"auc": float(roc_auc_score(y[idx_te], probs_te)), "sens": float(tp / max(tp + fn, 1)), "esp": float(tn / max(tn + fp, 1)), "prec": float(precision_score(y[idx_te], preds_te, zero_division=0)), "f1": float(f1_score(y[idx_te], preds_te, zero_division=0)), "cm": [[int(tn), int(fp)], [int(fn), int(tp)]]},
    }
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(pacote, f)
    return pacote


def main():
    st.set_page_config(page_title="Triagem Dermatologica Visual", page_icon="🔬", layout="centered")
    st.title("Triagem Dermatológica de manchas na pele")
    pacote = treinar_modelo()
    s = pacote["stats"]
    st.caption(f"PAD-UFES-20 ({s['total_imgs']} imagens / {s['total_pacs']} pacientes) | MobileNetV3 + Otsu ABC")
    aba1, aba2 = st.tabs(["Analise por Imagem", "Validacao Cientifica"])
    with aba1:
        foto = st.file_uploader("Envie a fotografia da lesao cutanea", type=["png", "jpg", "jpeg", "webp"])
        if foto is not None:
            img = Image.open(foto)
            feats, geom, img_box, img_seg, bbox = analisar_imagem(img)
            c1, c2, c3 = st.columns(3)
            c1.image(img.resize(IMG_SIZE), caption="Original", use_container_width=True)
            c2.image(img_box, caption=f"Deteccao {bbox}", use_container_width=True)
            c3.image(img_seg, caption="Segmentacao Otsu", use_container_width=True)
            z = pacote["scaler"].transform([feats])
            if np.percentile(np.abs(z), 95) > 4.5:
                st.warning("Aviso OOD: Iluminacao ou enquadramento atipico em relacao ao treino.")
            prob = float(pacote["modelo"].predict_proba(np.clip(z, -3.5, 3.5))[0][1])
            lim = pacote["limiar"]
            if prob >= lim:
                st.error(f"SUSPEITA DE CANCER DE PELE (Probabilidade: {prob * 100:.1f}% | Limiar: {lim * 100:.0f}%)")
            else:
                st.success(f"LESAO DE PELE BENIGNA (Probabilidade de Malignidade: {prob * 100:.1f}% | Limiar: {lim * 100:.0f}%)")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Assimetria (A)", f"{(geom[8] + geom[9]) / 2.0:.3f}")
            m2.metric("Borda Irregular (B)", f"{geom[7]:.2f}")
            m3.metric("Variacao Cor (C)", f"{np.mean(geom[3:6]):.3f}")
            m4.metric("Area Segmentada", f"{geom[6] * 100:.1f}%")
    with aba2:
        m = pacote["metricas"]
        st.subheader("Divisao Sem Vazamento por Paciente")
        st.table(pd.DataFrame([
            {"Conjunto": "Treino", "Imagens": f"{s['tr_imgs']} ({s['tr_aug']} usadas)", "Pacientes": s["tr_pacs"]},
            {"Conjunto": "Validacao (Calibracao)", "Imagens": s["va_imgs"], "Pacientes": s["va_pacs"]},
            {"Conjunto": "Teste Cego", "Imagens": s["te_imgs"], "Pacientes": s["te_pacs"]},
        ]))
        st.subheader("Desempenho no Teste Cego")
        st.table(pd.DataFrame([{"ROC-AUC": f"{m['auc']:.3f}", "Sensibilidade": f"{m['sens'] * 100:.1f}%", "Especificidade": f"{m['esp'] * 100:.1f}%", "Precisao": f"{m['prec'] * 100:.1f}%", "F1-Score": f"{m['f1']:.3f}", "Limiar": f"{pacote['limiar']:.2f}"}]))
        st.subheader("Matriz de Confusao")
        st.table(pd.DataFrame(m["cm"], index=["Real: Benigno", "Real: Maligno (BCC/SCC/MEL)"], columns=["Predito: Benigno", "Predito: Suspeita Cancer"]))


if __name__ == "__main__":
    main()