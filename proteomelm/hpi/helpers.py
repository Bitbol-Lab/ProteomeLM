import numpy as np
import torch
from typing import Dict


def _metrics_from_arrays(preds: np.ndarray, labs: np.ndarray) -> Dict[str, float]:
    """Compute cosine similarity and MSE between two aligned arrays."""
    if preds.shape[0] == 0:
        return {"cosine_similarity": 0.0, "cosine_loss": 1.0, "mse": 0.0}
    pnorm = np.linalg.norm(preds, axis=-1, keepdims=True)
    lnorm = np.linalg.norm(labs, axis=-1, keepdims=True)
    pnorm[pnorm == 0] = 1e-7
    lnorm[lnorm == 0] = 1e-7
    cosines = (preds / pnorm * labs / lnorm).sum(axis=-1)
    return {
        "cosine_similarity": float(np.mean(cosines)),
        "cosine_loss": float(1 - np.mean(cosines)),
        "mse": float(np.mean((preds - labs) ** 2)),
    }


def compute_metrics(eval_pred) -> Dict[str, float]:
    """
    Compute evaluation metrics separated by host and pathogen.

    The model's forward pass already filters predictions to masked positions
    only (shape ``[num_masked, dim]``).  Labels come in full padded form
    ``[batch, max_seq_len, dim]`` with ``-100`` at ignored positions.
    source_ids (``[batch, max_seq_len]``) marks host (0) vs pathogen (1);
    when present, metrics are split accordingly.
    """
    predictions, label_ids = eval_pred

    # label_ids is a tuple (labels, source_ids) when source_ids in label_names
    if isinstance(label_ids, (tuple, list)):
        labels, source_ids = label_ids[0], label_ids[1]
    else:
        labels, source_ids = label_ids, None

    # predictions may be a tuple (prediction_scores, prediction_norm, ...)
    preds = predictions[0] if isinstance(predictions, tuple) else predictions

    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    # Flatten labels to 2-D; keep only non-ignored (masked) positions
    labs_flat = labels.reshape(-1, labels.shape[-1])
    valid_mask = labs_flat[:, 0] != -100
    labs = labs_flat[valid_mask]

    # preds are already masked-only; flatten to 2-D
    preds = preds.reshape(-1, preds.shape[-1])

    # Align lengths (safety for accumulation rounding)
    n = min(preds.shape[0], labs.shape[0])
    preds, labs = preds[:n], labs[:n]

    metrics = {f"overall_{k}": v for k, v in _metrics_from_arrays(preds, labs).items()}

    if source_ids is not None:
        if isinstance(source_ids, torch.Tensor):
            source_ids = source_ids.detach().cpu().numpy()
        # Flatten source_ids and keep positions that match valid_mask
        src_flat = source_ids.reshape(-1)[valid_mask][:n]
        for label, name in ((0, "host"), (1, "pathogen")):
            sel = src_flat == label
            sub_metrics = _metrics_from_arrays(preds[sel], labs[sel])
            for k, v in sub_metrics.items():
                metrics[f"{name}_{k}"] = v

    return metrics
