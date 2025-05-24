import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, hamming_loss, accuracy_score

def evaluate_multilabel(y_true, y_pred, threshold=0.5):
    y_pred_bin = (y_pred >= threshold).astype(int)
    metrics = {
        'precision_per_sdg': precision_score(y_true, y_pred_bin, average=None, zero_division=0),
        'recall_per_sdg': recall_score(y_true, y_pred_bin, average=None, zero_division=0),
        'f1_per_sdg': f1_score(y_true, y_pred_bin, average=None, zero_division=0),
        'macro_f1': f1_score(y_true, y_pred_bin, average='macro', zero_division=0),
        'micro_f1': f1_score(y_true, y_pred_bin, average='micro', zero_division=0),
        'hamming_loss': hamming_loss(y_true, y_pred_bin),
        'exact_match': accuracy_score(y_true, y_pred_bin)
    }
    return metrics

def find_optimal_thresholds(y_true, y_probs):
    thresholds = []
    for i in range(y_true.shape[1]):
        best_thr, best_f1 = 0.5, 0
        for t in np.linspace(0.1, 0.9, 17):
            f1 = f1_score(y_true[:, i], (y_probs[:, i] >= t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thr = t
        thresholds.append(best_thr)
    return np.array(thresholds)

def aggregate_logits_chunkwise(chunk_logits):
    return np.max(chunk_logits, axis=0)
