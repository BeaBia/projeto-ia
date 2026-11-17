import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


def selecionar_limiar_f2(y_validacao, escores):
    limiar, melhor_f2 = 0.50, -1.0
    for candidato in np.linspace(0.25, 0.65, 41):
        predicoes = (escores >= candidato).astype(int)
        recall = recall_score(y_validacao, predicoes, zero_division=0)
        precision = precision_score(y_validacao, predicoes, zero_division=0)
        f2 = (5.0 * precision * recall) / (4.0 * precision + recall) if precision + recall > 0 else 0.0
        if f2 > melhor_f2:
            melhor_f2, limiar = f2, float(candidato)
    return limiar, float(melhor_f2)


def calcular_metricas(y_real, escores, limiar):
    predicoes = (escores >= limiar).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_real, predicoes, labels=[0, 1]).ravel()
    try:
        auc = float(roc_auc_score(y_real, escores))
    except ValueError:
        auc = None
    return {
        "auc": auc,
        "sens": float(tp / max(tp + fn, 1)),
        "esp": float(tn / max(tn + fp, 1)),
        "prec": float(precision_score(y_real, predicoes, zero_division=0)),
        "f1": float(f1_score(y_real, predicoes, zero_division=0)),
        "cm": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "classes_teste": sorted(int(valor) for valor in np.unique(y_real)),
    }