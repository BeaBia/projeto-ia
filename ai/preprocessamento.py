import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from ai.config import IMG_SIZE


def extrair_geometria(imagem):
	arr = np.asarray(imagem.convert("RGB").resize(IMG_SIZE, Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
	norm = np.power(np.mean(np.power(np.maximum(arr, 1e-4), 6), axis=(0, 1)), 1.0 / 6.0)
	illum = norm / np.maximum(np.linalg.norm(norm), 1e-6)
	arr = np.clip(arr * ((1.0 / np.sqrt(3.0)) / np.maximum(illum, 1e-4)), 0.0, 1.0).astype(np.float32)
	suave = ndimage.gaussian_filter(arr, sigma=(1.2, 1.2, 0))

	bordas = np.vstack([
		suave[:10].reshape(-1, 3), suave[-10:].reshape(-1, 3),
		suave[:, :10].reshape(-1, 3), suave[:, -10:].reshape(-1, 3),
	])
	dist = np.linalg.norm(suave - np.median(bordas, axis=0), axis=2)

	hist, edges = np.histogram(dist.ravel(), bins=256)
	centers = 0.5 * (edges[:-1] + edges[1:])
	w1 = np.cumsum(hist)
	w2 = w1[-1] - w1
	csum = np.cumsum(hist * centers)
	criterio = w1[:-1] * w2[:-1] * (
		csum[:-1] / np.maximum(w1[:-1], 1)
		- (csum[-1] - csum[:-1]) / np.maximum(w2[:-1], 1)
	) ** 2
	limiar = float(centers[np.argmax(criterio)])

	mask = ndimage.binary_closing(dist > limiar, structure=np.ones((5, 5)))
	mask = ndimage.binary_opening(ndimage.binary_fill_holes(mask), structure=np.ones((5, 5)))
	rotulos, n_obj = ndimage.label(mask)
	if n_obj > 0:
		areas = ndimage.sum(mask, rotulos, range(1, n_obj + 1))
		mask = rotulos == (int(np.argmax(areas)) + 1)
	else:
		mask = np.zeros(IMG_SIZE, dtype=bool)
		mask[32:96, 32:96] = True

	ys, xs = np.where(mask)
	bbox = (
		max(0, int(xs.min()) - 4), max(0, int(ys.min()) - 4),
		min(127, int(xs.max()) + 4), min(127, int(ys.max()) + 4),
	) if len(xs) > 0 else (24, 24, 104, 104)
	area = float(np.sum(mask)) / float(IMG_SIZE[0] * IMG_SIZE[1])
	perim = float(np.sum(mask ^ ndimage.binary_erosion(mask, structure=np.ones((3, 3)))))
	borda_irreg = min((perim ** 2) / (4.0 * np.pi * max(float(np.sum(mask)), 1.0)), 3.0)
	mf = mask.astype(np.float32)
	assim_h = float(np.mean(np.abs(mf - np.fliplr(mf))))
	assim_v = float(np.mean(np.abs(mf - np.flipud(mf))))

	pixels = arr[mask] if np.any(mask) else arr.reshape(-1, 3)
	vetor = np.concatenate([
		pixels.mean(axis=0), pixels.std(axis=0),
		[area, borda_irreg, assim_h, assim_v],
	]).astype(np.float32)
	return vetor, arr, mask, bbox, Image.fromarray((arr * 255.0).astype(np.uint8))


def preparar_visualizacoes(imagem):
	geometria, arr, mask, bbox, imagem_corrigida = extrair_geometria(imagem)
	imagem_box = imagem.convert("RGB").resize(IMG_SIZE, Image.Resampling.BILINEAR)
	ImageDraw.Draw(imagem_box).rectangle(bbox, outline=(255, 50, 50), width=3)

	overlay = arr.copy()
	contorno = mask ^ ndimage.binary_erosion(mask, structure=np.ones((3, 3)))
	overlay[mask] = overlay[mask] * 0.65 + np.array([0.0, 0.75, 0.85]) * 0.35
	overlay[contorno] = np.array([1.0, 0.9, 0.0])
	imagem_segmentada = Image.fromarray((np.clip(overlay, 0, 1) * 255).astype(np.uint8))
	return geometria, imagem_box, imagem_segmentada, bbox, imagem_corrigida
