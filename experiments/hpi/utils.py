"""Shared utilities for ProteomeLM-HPI clean analysis.

Provides:
  - DATASETS / DATASET_GROUP / CLEAN_LABELS / ABLATION_MODELS constants
  - load_attention()  — load averaged cross-attention arrays from .npz files
  - lr_auroc()        — cross-validated logistic-regression AUROC
  - lr_auroc_grouped_identity_cv() — LR AUROC with sequence-identity-grouped CV
  - style_ax()        — consistent matplotlib axis style (rcParams set on import)
  - encode_sequences() / encode_split_cached() / add_gene_name_aliases()
                      — ESM-C encoding of combined host+pathogen proteomes
  - load_finetuned() / load_base() / run_inference()
                      — ProteomeLM (+ LoRA) loading and full-proteome forward pass

torch / esm / peft / Bio are imported lazily inside the functions that need
them, so the plotting scripts only need numpy / sklearn / matplotlib.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import hashlib

import numpy as np
import matplotlib.pyplot as plt
import requests
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# In-memory CV-split cache
# ---------------------------------------------------------------------------
# Keys: (dataset_key, pos_hash, neg_hash, n_folds, min_seq_id_int, seed)
# Values: list of (train_indices, test_indices) arrays
# The group_keys array is also cached separately.
_GROUP_KEY_CACHE: Dict[Tuple, np.ndarray] = {}   # (dataset_key, pos_hash, neg_hash, min_seq_id_int) -> group_keys
_SPLIT_CACHE: Dict[Tuple, List[Tuple[np.ndarray, np.ndarray]]] = {}   # + (n_folds, seed)

# ---------------------------------------------------------------------------
# Matplotlib style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family":      "Arial",
    "font.size":        12,
    "axes.labelsize":   13,
    "axes.titlesize":   14,
    "xtick.labelsize":  11,
    "ytick.labelsize":  11,
    "legend.fontsize":  11,
})

COLOR_BASE      = "#aaaaaa"
COLOR_FINETUNED = "#E74C3C"


def style_ax(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ---------------------------------------------------------------------------
# Dataset list  (key → display label used in all three scripts)
# ---------------------------------------------------------------------------
# Viruses first, then bacteria — order within each group by AUROC in plots
DATASETS: List[Tuple[str, str]] = [
    # Viruses
    ("sars_cov2_zhou2022", "$\\mathit{SARS\\text{-}CoV\\text{-}2}$"),
    ("hiv1_jager2012",     "$\\mathit{HIV}$-1"),
    ("influenza_a",        "$\\mathit{Influenza\\ A}$"),
    ("ebv",                "$\\mathit{EBV}$"),
    ("hpv",                "$\\mathit{HPV}$-16"),
    ("hsv1",               "$\\mathit{HSV}$-1"),
    # Bacteria
    ("yersinia",           "$\\mathit{Y.\\ pestis}$\n(Yersinia)"),
    ("salmonella",         "$\\mathit{S.\\ }$Typhimurium\n(Salmonella)"),
    ("chlamydia",          "$\\mathit{C.\\ trachomatis}$"),
    ("tuberculosis",       "$\\mathit{M.\\ tuberculosis}$"),
]

# Group membership for each dataset key
DATASET_GROUP: Dict[str, str] = {
    "sars_cov2_zhou2022": "Virus",
    "hiv1_jager2012": "Virus",
    "influenza_a":    "Virus",
    "ebv":            "Virus",
    "hpv":            "Virus",
    "hsv1":           "Virus",
    "yersinia":       "Bacteria",
    "salmonella":     "Bacteria",
    "chlamydia":      "Bacteria",
    "tuberculosis":   "Bacteria",
}

# Human-readable species names (no LaTeX), shared by the figure scripts
CLEAN_LABELS: Dict[str, str] = {
    "sars_cov2_zhou2022": "SARS-CoV-2",
    "hiv1_jager2012": "HIV-1",
    "influenza_a": "Influenza A virus",
    "ebv": "Epstein-Barr virus",
    "hpv": "Human papillomavirus 16",
    "hsv1": "Herpes simplex virus 1",
    "yersinia": "Yersinia pestis",
    "salmonella": "Salmonella enterica Typhimurium",
    "chlamydia": "Chlamydia trachomatis",
    "tuberculosis": "Mycobacterium tuberculosis",
}

# ---------------------------------------------------------------------------
# Ablation model registry  (subdir name → display label)
# ---------------------------------------------------------------------------
ABLATION_MODELS: Dict[str, str] = {
    "proteomelm_hpi_bigdataset":                "asym mask, block, extended dataset",
    "proteomelm_hpi_s_abl1_seed_sym_noblock":   "sym mask, no block",
    "proteomelm_hpi_s_abl2_seed_asym_noblock":  "asym mask, no block",
    "proteomelm_hpi_s_abl3_seed_inv_noblock":   "inv mask, no block",
    "proteomelm_hpi_s_abl4_seed_total_noblock": "total mask, no block",
    "proteomelm_hpi_s_abl5_seed_sym_block":     "sym mask, block",
    "proteomelm_hpi_s_abl6_seed_asym_block":    "asym mask, block",
    "proteomelm_hpi_s_abl7_seed_inv_block":     "inv mask, block",
    "proteomelm_hpi_s_abl8_seed_total_block":   "total mask, block",
}

# ---------------------------------------------------------------------------
# Core data-loading helper
# ---------------------------------------------------------------------------

def load_attention(
    attention_dir: Path,
    dataset: str,
    suffix: str = "",
) -> Dict[str, np.ndarray]:
    """Load averaged cross-attention arrays for a dataset.

    Looks for files: {dataset}_{type}{suffix}_attention.npz
    Types loaded: hpi, random_cross, intra_host, random_intra

    Returns dict  type -> array of shape (n_pairs, n_layers, n_heads)
    where each entry is the average of both cross-attention directions.
    Missing files are silently skipped.
    """
    result: Dict[str, np.ndarray] = {}
    for itype in ("hpi", "random_cross", "intra_host", "random_intra"):
        path = attention_dir / f"{dataset}_{itype}{suffix}_attention.npz"
        if not path.exists():
            continue
        npz = np.load(path, allow_pickle=True)
        # Average both cross-attention directions → symmetric pair score
        result[itype] = (npz["attention_a_to_b"] + npz["attention_b_to_a"]) / 2.0
    return result


def load_attention_with_pairs(
    attention_dir: Path,
    dataset: str,
    itype: str,
    suffix: str = "",
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Load one attention type plus pair identifiers from disk.

    Returns (attention_avg, pair_ids) where pair_ids has shape (n_pairs, 2).
    Returns None when the file does not exist.
    """
    path = attention_dir / f"{dataset}_{itype}{suffix}_attention.npz"
    if not path.exists():
        return None
    npz = np.load(path, allow_pickle=True)
    attn = (npz["attention_a_to_b"] + npz["attention_b_to_a"]) / 2.0
    pair_ids = npz["pair_ids"]
    return attn, pair_ids


def _load_uniprot_sequence_cache(cache_file: Path) -> Dict[str, str]:
    seqs: Dict[str, str] = {}
    if not cache_file.exists():
        return seqs
    for line in cache_file.read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) != 2:
            continue
        acc, seq = parts
        if acc and seq:
            seqs[acc] = seq
    return seqs


def _save_uniprot_sequence_cache(cache_file: Path, seqs: Dict[str, str]) -> None:
    lines = [f"{acc}\t{seq}" for acc, seq in sorted(seqs.items()) if seq]
    cache_file.write_text("\n".join(lines) + ("\n" if lines else ""))


def _fetch_uniprot_sequence(accession: str, timeout: int = 4) -> Optional[str]:
    url = f"https://rest.uniprot.org/uniprotkb/{accession}.fasta"
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code != 200:
            return None
        seq = "".join(line.strip() for line in r.text.splitlines() if line and not line.startswith(">"))
        return seq if seq else None
    except Exception:
        return None


def _parse_fasta_to_map(text: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    current_acc: Optional[str] = None
    chunks: List[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current_acc and chunks:
                out[current_acc] = "".join(chunks)
            chunks = []
            parts = line[1:].split("|")
            if len(parts) > 1:
                current_acc = parts[1].strip()
            else:
                current_acc = line[1:].split()[0]
        else:
            chunks.append(line)
    if current_acc and chunks:
        out[current_acc] = "".join(chunks)
    return out


def _fetch_uniprot_sequences_batch(accessions: List[str], timeout: int = 45) -> Dict[str, str]:
    """Fetch many UniProt accessions in one request using the stream API."""
    if not accessions:
        return {}

    acc_query = " OR ".join(f"accession:{acc}" for acc in accessions)
    url = "https://rest.uniprot.org/uniprotkb/stream"
    params = {
        "format": "fasta",
        "query": f"({acc_query})",
    }
    try:
        r = requests.get(url, params=params, timeout=timeout)
        if r.status_code != 200 or not r.text.strip():
            return {}
        return _parse_fasta_to_map(r.text)
    except Exception:
        return {}


def _ensure_sequences(accessions: List[str], cache_dir: Path) -> Dict[str, str]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "uniprot_sequence_cache.tsv"
    seqs = _load_uniprot_sequence_cache(cache_file)

    missing = [acc for acc in accessions if acc not in seqs]
    if missing:
        # Batch queries are much faster than one-request-per-accession.
        # Keep URL length moderate to avoid request rejection by upstream.
        batch_size = 40
        for i in range(0, len(missing), batch_size):
            batch = missing[i:i + batch_size]
            batch_map = _fetch_uniprot_sequences_batch(batch)
            for acc, seq in batch_map.items():
                if seq:
                    seqs[acc] = seq

        # Fallback for any accessions not returned by batch query.
        still_missing = [acc for acc in missing if acc not in seqs]
        for acc in still_missing:
            seq = _fetch_uniprot_sequence(acc)
            if seq:
                seqs[acc] = seq
        _save_uniprot_sequence_cache(cache_file, seqs)
    return {acc: seqs[acc] for acc in accessions if acc in seqs}


def _cluster_sequences_mmseqs(
    sequences: Dict[str, str],
    cache_dir: Path,
    dataset: str,
    side: str,
    min_seq_id: float = 0.5,
) -> Dict[str, str]:
    """Cluster sequences with MMseqs2 and return accession -> cluster_id map."""
    if not sequences:
        return {}

    cache_dir.mkdir(parents=True, exist_ok=True)
    # v2 cache: ensures we do not reuse older cache files from buggy cluster-id mapping.
    cache_file = cache_dir / f"{dataset}_{side}_clusters_id{int(min_seq_id*100)}_v2.tsv"
    if cache_file.exists():
        cached: Dict[str, str] = {}
        for line in cache_file.read_text().splitlines():
            if not line.strip():
                continue
            acc, cid = line.split("\t")
            cached[acc] = cid
        if all(acc in cached for acc in sequences):
            return {acc: cached[acc] for acc in sequences}

    if len(sequences) == 1:
        only = next(iter(sequences.keys()))
        return {only: f"{dataset}_{side}_c0"}

    with tempfile.TemporaryDirectory(prefix=f"mmseqs_{dataset}_{side}_") as td:
        td_path = Path(td)
        in_fa = td_path / "input.fasta"
        out_prefix = td_path / "out"
        tmp_dir = td_path / "tmp"

        with in_fa.open("w") as fh:
            for acc, seq in sequences.items():
                fh.write(f">{acc}\n{seq}\n")

        cmd = [
            "mmseqs", "easy-cluster",
            str(in_fa), str(out_prefix), str(tmp_dir),
            "--min-seq-id", str(min_seq_id),
            "-c", "0.8",
            "--cov-mode", "0",
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        cluster_tsv = Path(f"{out_prefix}_cluster.tsv")
        if not cluster_tsv.exists():
            # Conservative fallback: each sequence becomes its own cluster.
            return {acc: f"{dataset}_{side}_{acc}" for acc in sequences}

        clusters: Dict[str, str] = {}
        rep_to_cid: Dict[str, str] = {}
        with cluster_tsv.open() as fh:
            for line in fh:
                rep, member = line.strip().split("\t")
                # All members of the same representative must share one cluster id.
                cid = rep_to_cid.setdefault(rep, f"{dataset}_{side}_{rep}")
                clusters[member] = cid

        # Ensure every accession has a cluster label.
        for acc in sequences:
            clusters.setdefault(acc, f"{dataset}_{side}_{acc}")

        lines = [f"{acc}\t{clusters[acc]}" for acc in sorted(clusters)]
        cache_file.write_text("\n".join(lines) + "\n")
        return {acc: clusters[acc] for acc in sequences}


def _orient_pairs_with_hpi_reference(
    pos_pairs: np.ndarray,
    neg_pairs: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Orient pair columns as (pathogen, host) using positive HPI reference IDs."""
    path_ids = {str(x) for x in pos_pairs[:, 0].tolist()}
    host_ids = {str(x) for x in pos_pairs[:, 1].tolist()}

    def _orient(arr: np.ndarray) -> np.ndarray:
        out = np.array(arr, dtype=object, copy=True)
        for i in range(len(out)):
            a = str(out[i, 0])
            b = str(out[i, 1])
            if a in path_ids and b in host_ids:
                continue
            if b in path_ids and a in host_ids:
                out[i, 0], out[i, 1] = out[i, 1], out[i, 0]
        return out

    return _orient(pos_pairs), _orient(neg_pairs)


def _pairs_hash(pairs: np.ndarray) -> str:
    """Stable hash of a 2-D pair-ID array (order-independent within rows, order-sensitive across rows)."""
    flat = pairs.astype(str).flatten().tobytes()
    return hashlib.sha1(flat).hexdigest()[:16]


def lr_auroc_grouped_identity_cv(
    pos_attn: np.ndarray,
    neg_attn: np.ndarray,
    pos_pair_ids: np.ndarray,
    neg_pair_ids: np.ndarray,
    dataset_key: str,
    n_folds: int = 2,
    return_folds: bool = False,
    min_seq_id: float = 0.5,
    cache_dir: Optional[Path] = None,
) -> float | Tuple[float, float]:
    """Group-aware LR AUROC with CV split isolation by sequence-identity clusters.

    Host and pathogen proteins are clustered at *min_seq_id* using MMseqs2.
    Each pair gets a group key from (pathogen_cluster, host_cluster), and CV
    uses StratifiedGroupKFold so similar proteins do not leak across folds.
    """
    if len(pos_attn) == 0 or len(neg_attn) == 0:
        return (np.nan, np.nan) if return_folds else np.nan

    pos_pairs, neg_pairs = _orient_pairs_with_hpi_reference(pos_pair_ids, neg_pair_ids)

    path_ids = sorted({str(x) for x in pos_pairs[:, 0].tolist()} | {str(x) for x in neg_pairs[:, 0].tolist()})
    host_ids = sorted({str(x) for x in pos_pairs[:, 1].tolist()} | {str(x) for x in neg_pairs[:, 1].tolist()})

    cdir = cache_dir or (Path(__file__).resolve().parent / "data" / "sequence_identity")

    # ---- group_keys: cached in memory across model calls for the same dataset ----
    pos_h = _pairs_hash(pos_pairs)
    neg_h = _pairs_hash(neg_pairs)
    min_seq_id_int = int(round(min_seq_id * 100))
    gk_key = (dataset_key, pos_h, neg_h, min_seq_id_int)

    if gk_key not in _GROUP_KEY_CACHE:
        path_seqs = _ensure_sequences(path_ids, cdir)
        host_seqs = _ensure_sequences(host_ids, cdir)
        path_clusters = _cluster_sequences_mmseqs(path_seqs, cdir, dataset_key, "pathogen", min_seq_id=min_seq_id)
        host_clusters = _cluster_sequences_mmseqs(host_seqs, cdir, dataset_key, "host", min_seq_id=min_seq_id)
        for acc in path_ids:
            path_clusters.setdefault(acc, f"{dataset_key}_pathogen_{acc}")
        for acc in host_ids:
            host_clusters.setdefault(acc, f"{dataset_key}_host_{acc}")
        all_pairs = np.vstack([pos_pairs, neg_pairs])
        _GROUP_KEY_CACHE[gk_key] = np.array([
            f"{path_clusters[str(p)]}|{host_clusters[str(h)]}" for p, h in all_pairs
        ])

    group_keys = _GROUP_KEY_CACHE[gk_key]

    X = np.vstack([
        pos_attn.reshape(len(pos_attn), -1),
        neg_attn.reshape(len(neg_attn), -1),
    ])
    y = np.concatenate([np.ones(len(pos_attn), dtype=int), np.zeros(len(neg_attn), dtype=int)])

    min_class = int(np.bincount(y).min())
    n_splits = min(n_folds, min_class)
    if n_splits < 2:
        return (np.nan, np.nan) if return_folds else np.nan

    # ---- CV splits: cached in memory (only indices, not X) ----
    fold_scores: List[float] = []
    for seed in (42, 43, 44, 45, 46):
        split_key = (*gk_key, n_folds, seed)
        if split_key not in _SPLIT_CACHE:
            cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            _SPLIT_CACHE[split_key] = list(cv.split(X, y, groups=group_keys))
        for tr, te in _SPLIT_CACHE[split_key]:
            if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
                continue
            sc = StandardScaler()
            clf = LogisticRegression(C=0.1, random_state=seed)
            clf.fit(sc.fit_transform(X[tr]), y[tr])
            prob = clf.predict_proba(sc.transform(X[te]))[:, 1]
            fold_scores.append(float(roc_auc_score(y[te], prob)))

    if not fold_scores:
        return (np.nan, np.nan) if return_folds else np.nan
    mean = float(np.mean(fold_scores))
    std = float(np.std(fold_scores))
    return (mean, std) if return_folds else mean


# ---------------------------------------------------------------------------
# AUROC metrics
# ---------------------------------------------------------------------------

def lr_auroc(
    pos_attn: np.ndarray,
    neg_attn: np.ndarray,
    n_folds: int = 2,
    return_folds: bool = False,
) -> float | Tuple[float, float]:
    """Cross-validated logistic-regression AUROC using all heads as features.

    Flattens (n_layers × n_heads) attention per pair into a feature vector
    and fits a regularised LR with StratifiedKFold CV.

    Returns mean AUROC (float), or (mean, std_across_folds) when
    *return_folds=True*.  Returns NaN when there are too few samples.
    """
    X = np.vstack([
        pos_attn.reshape(len(pos_attn), -1),
        neg_attn.reshape(len(neg_attn), -1),
    ])
    y = np.concatenate([np.ones(len(pos_attn)), np.zeros(len(neg_attn))])
    min_class = int(np.bincount(y.astype(int)).min())
    n_splits = min(n_folds, min_class)
    if n_splits < 2:
        return (np.nan, np.nan) if return_folds else np.nan

    fold_scores: List[float] = []
    for seed in (42, 43, 44, 45, 46):
        for tr, te in StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(X, y):
            sc = StandardScaler()
            clf = LogisticRegression(C=.1, random_state=seed)
            clf.fit(sc.fit_transform(X[tr]), y[tr])
            prob = clf.predict_proba(sc.transform(X[te]))[:, 1]
            if len(set(y[te].tolist())) > 1:
                fold_scores.append(float(roc_auc_score(y[te], prob)))

    if not fold_scores:
        return (np.nan, np.nan) if return_folds else np.nan
    mean = float(np.mean(fold_scores))
    std  = float(np.std(fold_scores))
    return (mean, std) if return_folds else mean


# ---------------------------------------------------------------------------
# ESM-C encoding of combined host + pathogen proteomes
# ---------------------------------------------------------------------------

def encode_sequences(
    sequences: List[str],
    device: str,
    max_padded_tokens: int = 16000,
    truncate: Optional[int] = None,
    to_float32: bool = False,
):
    """Encode sequences with ESM-C 600M; returns mean-pooled embeddings (N, 1152) on CPU.

    Sequences are sorted by length and packed into batches of at most
    *max_padded_tokens* padded tokens. *truncate* (if set) caps each sequence
    after sorting (the sort uses the untruncated length); *to_float32* casts
    each batch's pooled output to float32 (otherwise it stays bfloat16).
    """
    import torch
    from esm.models.esmc import ESMC
    from tqdm import tqdm

    model = ESMC.from_pretrained("esmc_600m")
    model.eval().to(device, dtype=torch.bfloat16)

    # Sort by length to minimise padding waste
    order = sorted(range(len(sequences)), key=lambda i: len(sequences[i]))
    sorted_seqs = [sequences[i] for i in order]

    all_embeddings_sorted = []

    def process_batch(esm, batch):
        input_ids = esm._tokenize(batch).long().to(device)
        with torch.no_grad():
            output = esm(input_ids)
        emb = output.embeddings
        pad_id = esm.tokenizer.pad_token_id
        mask = input_ids != pad_id
        emb[~mask] = 0.0
        counts = mask.sum(dim=1, keepdim=True).clamp(min=1)
        pooled = (emb.sum(dim=1) / counts).detach().cpu()
        return pooled.float() if to_float32 else pooled

    current_batch: List[str] = []
    for seq in tqdm(sorted_seqs, desc=f"ESM-C ({len(sequences)} proteins)"):
        if truncate is not None:
            seq = seq[:truncate]
        seq_len = len(seq) + 2
        if current_batch:
            new_max = max(len(current_batch[-1]) + 2, seq_len)
            if (len(current_batch) + 1) * new_max > max_padded_tokens:
                all_embeddings_sorted.append(process_batch(model, current_batch))
                current_batch = []
                torch.cuda.empty_cache()
        current_batch.append(seq)
    if current_batch:
        all_embeddings_sorted.append(process_batch(model, current_batch))

    embeddings_sorted = torch.cat(all_embeddings_sorted, dim=0)
    embeddings = torch.empty_like(embeddings_sorted)
    for new_idx, orig_idx in enumerate(order):
        embeddings[orig_idx] = embeddings_sorted[new_idx]

    del model, embeddings_sorted, all_embeddings_sorted
    torch.cuda.empty_cache()
    return embeddings


def encode_split_cached(
    combined_fasta: Path,
    esm_cache_dir: Path,
    device: str,
) -> Tuple[Dict, Dict[str, int]]:
    """Encode host and pathogen proteomes separately with per-species caching.

    The host proteome (e.g. human, ~20K proteins) is cached under
    ``{esm_cache_dir}/host_{host_tag}_esm.pt`` and reused across all
    datasets that share the same host.  The pathogen proteome is cached
    under ``{esm_cache_dir}/pathogen_{pathogen_tag}_esm.pt``.

    Returns the concatenated esm_data dict and the full protein_to_idx mapping,
    both in host-first order (not FASTA order).
    """
    import torch
    from Bio import SeqIO

    host_labels, host_seqs = [], []
    path_labels, path_seqs = [], []
    host_tag, path_tag = None, None

    for record in SeqIO.parse(combined_fasta, "fasta"):
        header = record.description
        parts = header.split("|")
        uid = parts[1] if len(parts) > 1 else header.split()[0]
        tag3 = parts[2] if len(parts) > 2 else ""
        # Host tag: extract the proteome ID (e.g. UP000005640_HOST_HUMAN)
        if "HOST" in tag3:
            host_labels.append(uid)
            host_seqs.append(str(record.seq)[:4096])
            if host_tag is None:
                # use the part after HOST_ as the cache key, e.g. HUMAN / MOUSE
                after = tag3.split("HOST")[-1].strip("_").split("_")[0]
                host_tag = after if after else "host"
        elif "PATHOGEN" in tag3:
            path_labels.append(uid)
            path_seqs.append(str(record.seq)[:4096])
            if path_tag is None:
                after = tag3.split("PATHOGEN")[-1].strip("_").split("_")[0]
                path_tag = after if after else combined_fasta.parent.name

    # Fall back: if tags not found, encode the whole FASTA as the host
    if not host_labels:
        print("  WARNING: HOST/PATHOGEN tags not found — encoding combined FASTA as-is.")
        for record in SeqIO.parse(combined_fasta, "fasta"):
            header = record.description
            parts = header.split("|")
            host_labels.append(parts[1] if len(parts) > 1 else header.split()[0])
            host_seqs.append(str(record.seq)[:4096])
        host_tag = combined_fasta.parent.name
        path_labels, path_seqs, path_tag = [], [], "none"

    esm_cache_dir.mkdir(parents=True, exist_ok=True)

    def _load_or_encode(kind: str, labels: List[str], seqs: List[str], cache_path: Path):
        if cache_path.exists():
            emb = torch.load(cache_path, map_location="cpu")
            if emb.shape[0] == len(labels):
                print(f"  {kind} cache hit: {cache_path.name} ({len(labels)} proteins)")
                return emb
            print(f"  {kind} cache size mismatch ({emb.shape[0]} cached vs {len(labels)} in FASTA) — re-encoding")
        else:
            print(f"  Encoding {kind.lower()} proteome ({len(labels)} proteins) -> {cache_path.name}")
        emb = encode_sequences(seqs, device)
        torch.save(emb, cache_path)
        return emb

    host_emb = _load_or_encode("Host", host_labels, host_seqs, esm_cache_dir / f"host_{host_tag}_esm.pt")
    path_emb = None
    if path_labels:
        path_emb = _load_or_encode("Pathogen", path_labels, path_seqs,
                                   esm_cache_dir / f"pathogen_{path_tag}_esm.pt")

    all_labels = host_labels + path_labels
    combined_emb = torch.cat([host_emb, path_emb], dim=0) if path_emb is not None else host_emb

    esm_data = {
        "inputs_embeds": combined_emb,
        "group_embeds": combined_emb,
        "group_labels": all_labels,
    }
    protein_to_idx = {uid: i for i, uid in enumerate(all_labels)}
    print(f"  Combined proteome: {len(all_labels)} proteins")
    return esm_data, protein_to_idx


def add_gene_name_aliases(fasta_file: Path, protein_to_idx: Dict[str, int]) -> Dict[str, int]:
    """Return a copy of *protein_to_idx* with GN= gene-name aliases added.

    Each alias points at the index its UniProt accession already has in
    *protein_to_idx*, i.e. the host-first embedding order returned by
    ``encode_split_cached``, never at the FASTA position. The two differ for
    pathogen-first FASTAs (raw/ebv and raw/hsv1), where FASTA positions index
    the wrong embedding rows.
    """
    mapping = dict(protein_to_idx)
    with open(fasta_file) as f:
        for line in f:
            if line.startswith(">"):
                header = line[1:].strip()
                parts = header.split("|")
                uid = parts[1] if len(parts) > 1 else header.split()[0]
                if uid in protein_to_idx and "GN=" in header:
                    gene = header.split("GN=")[1].split()[0]
                    mapping.setdefault(gene, protein_to_idx[uid])
    return mapping


# ---------------------------------------------------------------------------
# ProteomeLM loading + full-proteome inference
# ---------------------------------------------------------------------------

def load_finetuned(checkpoint_path: str, base_model_path: str, device: str = "cpu"):
    """Load ProteomeLM base + LoRA adapters in bfloat16 (CPU by default: attention is O(n²))."""
    import torch
    from peft import PeftModel
    from proteomelm.modeling_proteomelm import ProteomeLMForMaskedLM

    print(f"  Loading base: {base_model_path}")
    base = ProteomeLMForMaskedLM.from_pretrained(base_model_path)
    checkpoint_path = str(Path(checkpoint_path).resolve())
    print(f"  Loading LoRA: {checkpoint_path}")
    model = PeftModel.from_pretrained(base, checkpoint_path)
    return model.to(dtype=torch.bfloat16, device=device).eval()


def load_base(base_model_path: str):
    """Load ProteomeLM base model (no LoRA) in bfloat16 on CPU."""
    import torch
    from proteomelm.modeling_proteomelm import ProteomeLMForMaskedLM

    print(f"  Loading base (no LoRA): {base_model_path}")
    model = ProteomeLMForMaskedLM.from_pretrained(base_model_path)
    return model.to(dtype=torch.bfloat16, device="cpu").eval()


def run_inference(model, esm_data: Dict, return_logits: bool = False):
    """Forward pass over the whole proteome on CPU with output_attentions=True.

    Returns the per-layer attention as float32 numpy arrays ``(n_heads, N, N)``,
    or ``(contextual_embeddings, attentions)`` when *return_logits* is set
    (contextual embeddings = the model's per-protein output, float32 ``(N, D)``).
    """
    import torch

    seq_len = esm_data["inputs_embeds"].shape[0]
    print(f"  Running inference (seq_len={seq_len})...")
    with torch.no_grad():
        output = model(
            inputs_embeds=esm_data["inputs_embeds"][None].to(dtype=torch.bfloat16, device="cpu"),
            group_embeds=esm_data["group_embeds"][None].to(dtype=torch.bfloat16, device="cpu"),
            output_attentions=True,
        )
    contextual = None
    if return_logits:
        contextual = output.logits.squeeze(0).float().cpu().numpy().astype(np.float32)
    attentions = [attn_t.squeeze(0).float().numpy() for attn_t in output.attentions]
    del output
    n_l, n_h = len(attentions), attentions[0].shape[0]
    mem_gb = n_l * n_h * seq_len * seq_len * 4 / 1e9
    print(f"  -> {n_l} layers × {n_h} heads, seq_len={seq_len} ({mem_gb:.3f} GB)")
    if return_logits:
        print(f"  -> contextual embeddings: {contextual.shape}")
        return contextual, attentions
    return attentions
