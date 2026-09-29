import copy
import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, matthews_corrcoef

from proteomelm.modeling_proteomelm import ProteomeLMForMaskedLM
from proteomelm.utils.embedding import build_genome_esmc
from proteomelm.utils.proteome import (
    load_orthodb_group_vectors,
    build_group_embeddings_for_proteome,
    download_orthodb_tsv_for_accessions,
)
from proteomelm.utils.io import ensure_dir, parse_fasta

logger = logging.getLogger(__name__)


def _get_device(device: Optional[torch.device] = None) -> torch.device:
    if device is not None:
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------
# PPI Models
# --------------------------

class EnhancedPPIModel(nn.Module):
    """PPI classifier with three generalization improvements:

    1. Symmetric pair encoder — [E1+E2, |E1-E2|, E1*E2] so f(A,B)==f(B,A) exactly.
    2. Residual skip in protein branch — preserves a direct linear path from raw
       embedding to the interaction layer, zero-initialised so training starts
       from the same point as without it.
    3. FiLM conditioning — attention features modulate protein features before
       interaction, allowing the model to learn which embedding dimensions matter
       given the observed cross-protein attention pattern.
    """

    def __init__(self,
                 protein_embed_dim=640,
                 pair_feature_dim=48,
                 protein_layer1_dim=512,
                 protein_layer2_dim=256,
                 dropout_protein1=0.3,
                 dropout_protein2=0.4,
                 pair_layer1_dim=64,
                 pair_layer2_dim=32,
                 dropout_pair=0.1,
                 interaction_layer1_dim=128,
                 interaction_layer2_dim=64,
                 dropout_interaction=0.3,
                 classifier_layer1_dim=128,
                 classifier_layer2_dim=64,
                 dropout_classifier1=0.2,
                 dropout_classifier2=0.1,
                 **kwargs):
        super().__init__()
        self.protein_embed_dim = protein_embed_dim
        self.pair_feature_dim = pair_feature_dim

        # Protein branch
        self.protein_branch = nn.Sequential(
            nn.Linear(protein_embed_dim, protein_layer1_dim),
            nn.BatchNorm1d(protein_layer1_dim),
            nn.ReLU(),
            nn.Dropout(dropout_protein1),
            nn.Linear(protein_layer1_dim, protein_layer2_dim),
            nn.BatchNorm1d(protein_layer2_dim),
            nn.ReLU(),
            nn.Dropout(dropout_protein2),
        )
        # Residual skip: direct linear projection, zero-init so it starts inert.
        self.protein_skip = nn.Linear(protein_embed_dim, protein_layer2_dim, bias=False)

        # Pair branch
        self.pair_branch = nn.Sequential(
            nn.Linear(pair_feature_dim, pair_layer1_dim),
            nn.BatchNorm1d(pair_layer1_dim),
            nn.ReLU(),
            nn.Dropout(dropout_pair),
            nn.Linear(pair_layer1_dim, pair_layer2_dim),
            nn.BatchNorm1d(pair_layer2_dim),
            nn.ReLU(),
        )

        # FiLM: attention context → scale and shift for protein features.
        # Zero-init → identity at the start (scale=1+tanh(0)=1, shift=0).
        self.film_scale = nn.Linear(pair_layer2_dim, protein_layer2_dim)
        self.film_shift = nn.Linear(pair_layer2_dim, protein_layer2_dim)

        # Symmetric interaction processor: 3× dim instead of 4× (no [E1,E2] terms)
        self.interaction_processor = nn.Sequential(
            nn.Linear(protein_layer2_dim * 3, interaction_layer1_dim),
            nn.BatchNorm1d(interaction_layer1_dim),
            nn.ReLU(),
            nn.Dropout(dropout_interaction),
            nn.Linear(interaction_layer1_dim, interaction_layer2_dim),
            nn.BatchNorm1d(interaction_layer2_dim),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(interaction_layer2_dim + pair_layer2_dim, classifier_layer1_dim),
            nn.BatchNorm1d(classifier_layer1_dim),
            nn.ReLU(),
            nn.Dropout(dropout_classifier1),
            nn.Linear(classifier_layer1_dim, classifier_layer2_dim),
            nn.BatchNorm1d(classifier_layer2_dim),
            nn.ReLU(),
            nn.Dropout(dropout_classifier2),
            nn.Linear(classifier_layer2_dim, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        # Zero-init so the model starts identically to the pre-improvement baseline.
        nn.init.zeros_(self.protein_skip.weight)
        nn.init.zeros_(self.film_scale.weight)
        nn.init.zeros_(self.film_scale.bias)
        nn.init.zeros_(self.film_shift.weight)
        nn.init.zeros_(self.film_shift.bias)

    def _encode(self, E: torch.Tensor) -> torch.Tensor:
        return self.protein_branch(E) + self.protein_skip(E)

    def _film(self, E: torch.Tensor, F_proj: torch.Tensor) -> torch.Tensor:
        """Attention-conditioned modulation of protein features."""
        scale = 1.0 + torch.tanh(self.film_scale(F_proj))  # ≈ 1 at init
        shift = self.film_shift(F_proj)                     # ≈ 0 at init
        return E * scale + shift

    def forward(self, F=None, E1=None, E2=None):
        if F is None:
            F = torch.zeros((E1.shape[0], self.pair_feature_dim), device=E1.device, dtype=E1.dtype)
        if E1 is None:
            E1 = torch.zeros((F.shape[0], self.protein_embed_dim), device=F.device)
        if E2 is None:
            E2 = torch.zeros((F.shape[0], self.protein_embed_dim), device=F.device)

        F_proj = self.pair_branch(F)

        # Encode E1 and E2 in one pass so BatchNorm sees a single batch: separate
        # passes would normalize each side by its own statistics in training
        # (e.g. host-only vs pathogen-only in HPI) but by one shared running
        # average at eval, and a pair's training output would depend on which
        # slot each of its proteins occupies.
        E1_enc, E2_enc = self._encode(torch.cat([E1, E2], dim=0)).chunk(2, dim=0)
        E1_cond = self._film(E1_enc, F_proj)
        E2_cond = self._film(E2_enc, F_proj)

        # Symmetric: f(A,B) == f(B,A) by construction
        interaction_terms = torch.cat([
            E1_cond + E2_cond,
            (E1_cond - E2_cond).abs(),
            E1_cond * E2_cond,
        ], dim=1)

        interaction_features = self.interaction_processor(interaction_terms)
        return self.classifier(torch.cat([interaction_features, F_proj], dim=1))


class SimpleMLP(nn.Module):
    def __init__(self, protein_embed_dim, pair_feature_dim, hidden_dims=[64], dropout=0.3):
        super().__init__()
        layers = []
        prev_dim = 2 * protein_embed_dim + pair_feature_dim
        self.protein_embed_dim = protein_embed_dim
        self.pair_feature_dim = pair_feature_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, F=None, E1=None, E2=None):
        if F is None:
            F = torch.zeros((E1.shape[0], self.pair_feature_dim), device=E1.device, dtype=E1.dtype)
        if E1 is None:
            E1 = torch.zeros((F.shape[0], self.protein_embed_dim), device=F.device)
        if E2 is None:
            E2 = torch.zeros((F.shape[0], self.protein_embed_dim), device=F.device)
        return self.network(torch.cat([E1, E2, F], dim=1))


# --------------------------
# Loss Functions
# --------------------------

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, pos_weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction='none', pos_weight=self.pos_weight
        )
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        return (focal_weight * bce_loss).mean()


# --------------------------
# Training & Evaluation
# --------------------------

def _seed_everything(seed: int):
    """Set all relevant random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_model_cv(
    X_train, X_test, y_train, y_test,
    n_epochs=50,
    patience=10,
    model_type="simplemlp",
    verbose=False,
    replica_seed: int = 0,
    batch_size: int = 256,
    weight_decay: float = 1e-2,
    max_grad_norm: float = 5.0,
    use_class_weights: bool = False,
    loss_type: str = "bce",
    label_smoothing: float = 0.1,
    focal_gamma: float = 2.0,
    lr: float = 1e-3,
    warmup_epochs: int = 10,
    device: Optional[torch.device] = None,
):
    """Train a PPI classifier with early stopping on validation ROC-AUC.

    AdamW with a linear-warmup + cosine LR schedule; ``loss_type`` is ``"bce"``
    (with ``label_smoothing``) or ``"focal"``. Both AUC and AUPR are logged.

    Key defaults versus the original version:
      - loss_type: "bce" (was "aupr" — BCE is more stable for balanced data)
      - max_grad_norm: 5.0 (was 1.0 — old value clipped >50% of gradients)
      - scheduler: cosine (was ReduceLROnPlateau — collapsed LR to ~0 by ep.40)
      - use_class_weights: False (was True — Bernett/DScript are balanced 1:1)
      - patience: 10 (was 5)
    """
    _device = _get_device(device)
    seed = 42 + replica_seed
    _seed_everything(seed)

    g = torch.Generator()
    g.manual_seed(seed)

    def seed_worker(worker_id):
        import random
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    PPIModel = EnhancedPPIModel if model_type == "enhancedppi" else SimpleMLP

    def _to_tensor(arr, n_fallback, dim_fallback):
        if arr is not None:
            return torch.tensor(arr, dtype=torch.float32).view(arr.shape[0], -1).to(_device)
        return torch.zeros((n_fallback, dim_fallback), device=_device)

    n_train, n_test = len(y_train), len(y_test)

    f_train = _to_tensor(X_train.get("edges"), n_train, 1)
    e1_train = _to_tensor(X_train.get("x1"), n_train, 1)
    e2_train = _to_tensor(X_train.get("x2"), n_train, 1)
    f_test = _to_tensor(X_test.get("edges"), n_test, 1)
    e1_test = _to_tensor(X_test.get("x1"), n_test, 1)
    e2_test = _to_tensor(X_test.get("x2"), n_test, 1)

    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(_device)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(_device)

    embed_dim = e1_train.shape[-1]
    edges_dim = f_train.shape[-1]
    logger.debug("embed_dim=%d  edges_dim=%d", embed_dim, edges_dim)

    _seed_everything(seed)  # Re-seed before weight init for reproducibility
    model = PPIModel(protein_embed_dim=embed_dim, pair_feature_dim=edges_dim).to(_device)

    pos_weight = None
    if use_class_weights:
        n_neg = int((y_train == 0).sum())
        n_pos = int((y_train == 1).sum())
        if n_pos > 0:
            pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32, device=_device)
            logger.debug("pos_weight=%.2f  (neg=%d pos=%d)", pos_weight.item(), n_neg, n_pos)

    if loss_type == "focal":
        _base_loss = FocalLoss(alpha=0.25, gamma=focal_gamma, pos_weight=pos_weight)
    else:
        _base_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    if label_smoothing > 0.0 and loss_type == "bce":
        # Apply label smoothing: push targets toward 0.5 by epsilon
        _eps = label_smoothing
        def loss_fn(logits, targets):
            return _base_loss(logits, targets * (1 - _eps) + _eps / 2)
    else:
        loss_fn = _base_loss

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_dataset = TensorDataset(f_train, e1_train, e2_train, y_train_t)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size, shuffle=True,
        # BatchNorm cannot train on a single-sample batch; drop only a lone
        # trailing sample so every other dataset size batches exactly as before.
        drop_last=len(train_dataset) % batch_size == 1,
        worker_init_fn=seed_worker, generator=g
    )
    val_loader = DataLoader(
        TensorDataset(f_test, e1_test, e2_test, y_test_t),
        batch_size=batch_size, shuffle=False
    )

    # Cosine schedule with linear warmup: warmup prevents the model from making
    # large destructive steps in the first few epochs before the loss landscape
    # has been explored.
    if warmup_epochs > 0:
        _warmup = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0 / max(warmup_epochs, 1),
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        _cosine = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(n_epochs - warmup_epochs, 1)
        )
        _scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[_warmup, _cosine],
            milestones=[warmup_epochs],
        )
    else:
        _scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    best_auc = 0.0
    epochs_no_improve = 0
    best_state = None

    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0
        for f, e1, e2, labels in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(f, e1, e2), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        y_pred_list, y_true_list = [], []
        with torch.no_grad():
            for f, e1, e2, labels in val_loader:
                y_pred_list.extend(model(f, e1, e2).cpu().numpy())
                y_true_list.extend(labels.cpu().numpy())

        val_auc  = roc_auc_score(y_true_list, y_pred_list)
        val_aupr = average_precision_score(y_true_list, y_pred_list)
        _scheduler.step()

        if verbose:
            logger.info(
                "Epoch %d/%d  train_loss=%.4f  val_auc=%.4f  val_aupr=%.4f  lr=%.2e",
                epoch + 1, n_epochs, running_loss / len(train_loader),
                val_auc, val_aupr, optimizer.param_groups[0]['lr']
            )

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.debug("Early stopping at epoch %d (best_auc=%.4f)", epoch + 1, best_auc)
                break

    model.load_state_dict(best_state if best_state is not None else model.state_dict())
    model.eval()

    with torch.no_grad():
        logits = model(f_test, e1_test, e2_test).cpu().numpy().flatten()

    y_pred_proba = 1.0 / (1.0 + np.exp(-logits))
    y_test_bin = y_test_t.cpu().numpy().flatten().astype(int)

    auc_final  = roc_auc_score(y_test_bin, y_pred_proba)
    aupr_final = average_precision_score(y_test_bin, y_pred_proba)
    logger.debug("Val AUC=%.4f  AUPR=%.4f", auc_final, aupr_final)

    return model.cpu(), {
        "auc": auc_final,
        "aupr": aupr_final,
        "best_val_score": best_auc,   # best validation ROC-AUC (the early-stopping metric)
    }


def test_model_cv(model, X_test, y_test, device: Optional[torch.device] = None):
    """Evaluate a trained model on held-out data. Returns (X_test, y_test, model, metrics)."""
    _device = _get_device(device)
    n_test = len(y_test)

    def _to_tensor(arr, dim_fallback):
        if arr is not None:
            return torch.tensor(arr, dtype=torch.float32).view(arr.shape[0], -1).to(_device)
        return torch.zeros((n_test, dim_fallback), device=_device)

    f_test = _to_tensor(X_test.get("edges"), 1)
    e1_test = _to_tensor(X_test.get("x1"), 1)
    e2_test = _to_tensor(X_test.get("x2"), 1)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    model = model.to(_device).eval()
    with torch.no_grad():
        logits = model(f_test, e1_test, e2_test).cpu().numpy().flatten()

    y_pred_proba = 1.0 / (1.0 + np.exp(-logits))
    y_test_bin = y_test_t.numpy().flatten().astype(int)
    y_pred_bin = (y_pred_proba >= 0.5).astype(int)

    auc = roc_auc_score(y_test_bin, y_pred_proba)
    aupr = average_precision_score(y_test_bin, y_pred_proba)
    f1 = f1_score(y_test_bin, y_pred_bin, zero_division=0)
    mcc = matthews_corrcoef(y_test_bin, y_pred_bin)

    logger.debug("Test  AUC=%.4f  AUPR=%.4f  F1=%.4f  MCC=%.4f", auc, aupr, f1, mcc)
    return X_test, y_test, model, {"auc": auc, "aupr": aupr, "f1": f1, "mcc": mcc}


# --------------------------
# Weight averaging (SWA)
# --------------------------

def average_state_dicts(states: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Uniform average of floating-point tensors across state dicts from one training run.

    Integer buffers (BatchNorm's ``num_batches_tracked``) are taken from the last state;
    BatchNorm running statistics must be recomputed afterwards (``recompute_batchnorm_stats``).
    """
    return {
        key: (sum(s[key].float() for s in states) / len(states)).to(states[0][key].dtype)
        if states[0][key].is_floating_point() else states[-1][key].clone()
        for key in states[0]
    }


@torch.no_grad()
def recompute_batchnorm_stats(model: nn.Module, batches) -> None:
    """Re-estimate BatchNorm running statistics as a cumulative average over ``batches``.

    ``batches`` yields ``(F, E1, E2)`` tuples on the model's device. Needed after weight
    averaging, since averaged weights no longer match any epoch's running statistics.
    """
    bns = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    if not bns:
        return
    momenta = [m.momentum for m in bns]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None  # cumulative moving average
    was_training = model.training
    model.train()
    for F, E1, E2 in batches:
        model(F, E1, E2)
    for m, momentum in zip(bns, momenta):
        m.momentum = momentum
    model.train(was_training)


# --------------------------
# ProteomeLM feature extraction
# --------------------------

def _compute_encoded_genome_signature(fasta_path: Union[Path, str], checkpoint: Union[Path, str]) -> str:
    """Content hash of the FASTA sequences + model checkpoint identity.

    Used to detect a stale `encoded_genome_file` cache: the cache is keyed by
    filesystem path only, and a path can end up reused across different FASTA
    content (or a different checkpoint) — without this check `prepare_ppi`
    would silently pair new pair-labels with old, mismatched embeddings.
    """
    sequences = parse_fasta(str(fasta_path))
    parts = [str(checkpoint)] + [f"{label}:{seq}" for label, seq in sorted(sequences.items())]
    return hashlib.sha1("\n".join(parts).encode("utf-8")).hexdigest()


def prepare_ppi(
    checkpoint: Union[Path, str],
    fasta_file: Union[Path, str],
    encoded_genome_file: Optional[Union[Path, str]] = None,
    esm_device: Optional[str] = None,
    proteomelm_device: Optional[str] = None,
    include_attention: bool = False,
    include_all_hidden_states: bool = False,
    reload_if_possible: bool = False,
    orthodb_db_path: Optional[Union[Path, str]] = None,
    orthodb_tsv_path: Optional[Union[Path, str]] = None,
    orthodb_min_group_size: int = 0,
    orthodb_fetch_online: bool = True,
    model: Optional[ProteomeLMForMaskedLM] = None,
) -> Dict[str, Any]:
    """Run ProteomeLM on a FASTA file and return attention + embedding outputs.

    ``esm_device`` and ``proteomelm_device`` default to ``"cuda"`` when a GPU
    is available, ``"cpu"`` otherwise. ``model`` is an already loaded
    ``checkpoint`` (e.g. from ``notebook_inference.load_proteomelm_backbone``)
    to use instead of loading it again; it is cast to bfloat16 on
    ``proteomelm_device`` and put in eval mode. ``checkpoint`` still keys the
    encoded-genome cache.
    """
    if esm_device is None:
        esm_device = str(_get_device())
    if proteomelm_device is None:
        proteomelm_device = str(_get_device())

    fasta_path = Path(fasta_file)
    assert fasta_path.exists(), f"FASTA file not found: {fasta_path}"

    expected_signature = _compute_encoded_genome_signature(fasta_path, checkpoint)
    data = None
    if encoded_genome_file is not None and Path(encoded_genome_file).exists() and reload_if_possible:
        cached = torch.load(encoded_genome_file)
        if cached.get("_cache_signature") == expected_signature:
            cached.pop("_cache_signature", None)
            data = cached
            logger.info("Reloaded ESM embeddings from %s", encoded_genome_file)
        else:
            logger.warning(
                "Encoded genome cache at %s doesn't match the current FASTA/checkpoint "
                "(content changed since it was written) — recomputing instead of reusing it.",
                encoded_genome_file,
            )

    if data is None:
        with torch.no_grad():
            data = build_genome_esmc(fasta_path, device=esm_device)
        if encoded_genome_file is not None:
            ensure_dir(str(Path(encoded_genome_file).parent))
            torch.save({**data, "_cache_signature": expected_signature}, encoded_genome_file)
            logger.info("Saved ESM embeddings to %s", encoded_genome_file)

    assert "inputs_embeds" in data and "group_embeds" in data

    if orthodb_db_path:
        orthodb_db_path = str(orthodb_db_path)
        if "," in orthodb_db_path:
            orthodb_db_path = orthodb_db_path.split(",")[0]
        if orthodb_db_path.endswith(".pkl"):
            orthodb_db_path = str(Path(orthodb_db_path).parent)

        tsv_path = str(orthodb_tsv_path) if orthodb_tsv_path else str(fasta_path.with_suffix("")) + "_orthodb.tsv"

        if not Path(tsv_path).exists() and orthodb_fetch_online:
            sequences = parse_fasta(str(fasta_path))
            try:
                download_orthodb_tsv_for_accessions(list(sequences.keys()), tsv_path)
            except RuntimeError as e:
                logger.warning("OrthoDB TSV download failed: %s", e)

        if Path(tsv_path).exists() and Path(tsv_path).stat().st_size == 0:
            logger.warning("OrthoDB TSV is empty; using ESM group embeddings")
            tsv_path = None

        if tsv_path and Path(tsv_path).exists() and Path(orthodb_db_path).exists():
            orthodb_means = load_orthodb_group_vectors(orthodb_db_path, min_group_size=orthodb_min_group_size)
            group_embeds, mask = build_group_embeddings_for_proteome(
                fasta_path=str(fasta_path),
                orthodb_tsv_path=tsv_path,
                orthodb_group_means=orthodb_means,
                esm_embeddings=data["inputs_embeds"],
            )
            if mask.sum().item() == 0:
                logger.warning("OrthoDB TSV has no mappings; using ESM group embeddings")
            else:
                data["group_embeds"] = group_embeds
                data["orthodb_mask"] = mask
        else:
            logger.warning("OrthoDB TSV or group vectors missing; using ESM group embeddings")

    if model is None:
        if "/" in str(checkpoint) and not Path(checkpoint).exists():
            checkpoint_path = str(checkpoint)
        else:
            cp = Path(checkpoint)
            assert cp.exists(), f"Checkpoint not found: {cp}"
            checkpoint_path = str(cp)
        model = ProteomeLMForMaskedLM.from_pretrained(checkpoint_path)
    model = model.to(dtype=torch.bfloat16, device=proteomelm_device).eval()

    with torch.no_grad():
        inputs_embeds = data["inputs_embeds"][None].to(proteomelm_device, dtype=torch.bfloat16)
        group_embeds = data["group_embeds"][None].to(proteomelm_device, dtype=torch.bfloat16)
        output = model(
            inputs_embeds=inputs_embeds,
            group_embeds=group_embeds,
            output_attentions=include_attention,
            output_hidden_states=include_all_hidden_states,
        )
        data["plm_attentions"] = [x.cpu() for x in output.attentions] if include_attention else None
        data["plm_representations"] = output.last_hidden_states.cpu()
        data["plm_logits"] = output.logits.cpu()
        data["plm_all_representations"] = (
            torch.cat([x.cpu() for x in output.hidden_states], 0)
            if include_all_hidden_states else None
        )

    return data
