#!/usr/bin/env python3
"""
Gold Benchmark Evaluation — Balanced Three-Fold Cross-Evaluation
================================================================
Attention features are computed *inline* by running ProteomeLM on each
pathogen's combined proteome, so the checkpoint used for evaluation is
explicit and verifiable. The same forward pass is also used to extract
contextualized ProteomeLM protein embeddings. Results are cached per
checkpoint+pathogen to avoid repeating expensive inference.

By default, the script reads explicit fold directories built by
build_gold_benchmark.py:

    fold_a/{train,val,test}.tsv
    fold_b/{train,val,test}.tsv
    fold_c/{train,val,test}.tsv

val.tsv is used for LR hyperparameter selection and MLP early stopping.

Output
------
  figures/gold_benchmark_per_species.tsv
  figures/gold_benchmark_per_species.{pdf,png}

Models reported:
    - Cosine baseline (ESM-C embedding cosine similarity)
    - Embedding LR baseline ([e_a, e_b, |e_a-e_b|, e_a*e_b])
    - ProteomeLM-Attn-LR on attention only
    - ProteomeLM-LR on contextualized ProteomeLM embeddings + attention
    - EnhancedPPI on ProteomeLM attention only (optional)
    - MLP (EnhancedPPI) on ProteomeLM embeddings + attention (optional)

Usage
-----
    python 9_gold_benchmark.py \\
        --checkpoint /path/to/ProteomeLM-HPI/checkpoint-XXXX \\
        --base-model Bitbol-Lab/ProteomeLM-M
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_SCRIPT_DIR   = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_SCRIPT_DIR))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

# utils sets its own figure rcParams on import; keep this script's style (set below).
with plt.rc_context():
    from utils import add_gene_name_aliases, encode_split_cached, load_finetuned, run_inference

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
_DEFAULT_GOLD      = _SCRIPT_DIR / "data" / "gold_benchmark"
_DEFAULT_BENCH_RAW = _SCRIPT_DIR / "data" / "benchmarks" / "raw"
_DEFAULT_CKPT      = (_PROJECT_ROOT / "output"
                      / "ProteomeLM-HPI-S-abl8-seed-total-block"
                      / "checkpoint-20000")
_DEFAULT_BASE      = "Bitbol-Lab/ProteomeLM-S"
_DEFAULT_ESM_CACHE = _SCRIPT_DIR / "data" / "esm_cache"
_DEFAULT_ATTN_CACHE = _SCRIPT_DIR / "data" / "gold_attn_cache"
_FIGS_DIR          = _SCRIPT_DIR / "figures"

FOLD_COLORS = {"A": "#E74C3C", "B": "#4A90E2", "C": "#27AE60"}

SPECIES_LABELS: dict[str, str] = {
    "sars_cov2_zhou2022": "SARS-CoV-2",
    "hiv1_jager2012":     "HIV-1",
    "influenza_a":        "Influenza A",
    "ebv":                "EBV",
    "hpv":                "HPV-16",
    "hsv1":               "HSV-1",
    "yersinia":           r"$\it{Y.~pestis}$",
    "salmonella":         r"$\it{S.~enterica}$",
    "chlamydia":          r"$\it{C.~trachomatis}$",
    "tuberculosis":       r"$\it{M.~tuberculosis}$",
}

plt.rcParams.update({
    "font.family":    "Arial",
    "font.size":      12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
})


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _macro_species_auroc(
    y_true: np.ndarray,
    probs: np.ndarray,
    pathogens: np.ndarray,
) -> float:
    """Mean AUROC across pathogens with at least one positive and one negative.

    This matches the reporting target better than pooled AUROC, which can be
    dominated by a single large validation pathogen.
    """
    aucs: List[float] = []
    for pat in sorted(set(pathogens)):
        mask = pathogens == pat
        y_pat = y_true[mask]
        if mask.sum() < 2 or len(np.unique(y_pat)) < 2:
            continue
        aucs.append(float(roc_auc_score(y_pat, probs[mask])))
    return float(np.mean(aucs)) if aucs else np.nan


def _per_species_auroc(
    y_true: np.ndarray,
    probs: np.ndarray,
    pathogens: np.ndarray,
    fold: str,
    orientation_invariant: bool = False,
) -> list[dict]:
    rows = []
    for pat in sorted(set(pathogens)):
        mask  = pathogens == pat
        n_pos = int(y_true[mask].sum())
        n_neg = int((1 - y_true[mask]).sum())
        if mask.sum() < 2 or n_pos == 0:
            print(f"  Skipping {pat}: insufficient positive pairs ({n_pos} pos)")
            continue
        raw_auc = float(roc_auc_score(y_true[mask], probs[mask]))
        if orientation_invariant:
            rows.append({
                "pathogen": pat,
                "fold":     fold,
                "n_pos":    n_pos,
                "n_neg":    n_neg,
                "auroc_raw": raw_auc,
                "auroc":    max(raw_auc, 1.0 - raw_auc),
                "direction": "normal" if raw_auc >= 0.5 else "inverted",
            })
        else:
            rows.append({
            "pathogen": pat,
            "fold":     fold,
            "n_pos":    n_pos,
            "n_neg":    n_neg,
            "auroc":    raw_auc,
            })
    return rows


def _print_per_species(rows: list[dict], label: str) -> None:
    print(f"\n  Per-species AUROC  [{label}]")
    if rows and "auroc_raw" in rows[0]:
        col = 35
        print(f"    {'Pathogen':<{col}}{'Fold':>5}{'AUROC':>8}{'Raw':>8}{'Dir':>11}{'n_pos':>7}{'n_neg':>7}")
        print("    " + "─" * (col + 46))
        for r in sorted(rows, key=lambda x: x["auroc"], reverse=True):
            print(f"    {r['pathogen']:<{col}}{r['fold']:>5}"
                  f"{r['auroc']:>8.4f}{r['auroc_raw']:>8.4f}{r['direction']:>11}"
                  f"{r['n_pos']:>7}{r['n_neg']:>7}")
        return

    col = 35
    print(f"    {'Pathogen':<{col}}{'Fold':>5}{'AUROC':>8}{'n_pos':>7}{'n_neg':>7}")
    print("    " + "─" * (col + 27))
    for r in sorted(rows, key=lambda x: x["auroc"], reverse=True):
        print(f"    {r['pathogen']:<{col}}{r['fold']:>5}"
              f"{r['auroc']:>8.4f}{r['n_pos']:>7}{r['n_neg']:>7}")


def _cosine_similarity_pairs(
    split_df: pd.DataFrame,
    esm_cache: Dict[str, np.ndarray],
) -> np.ndarray:
    """Return cosine similarity scores in the same per-pathogen order as features."""
    if not esm_cache:
        return np.zeros(len(split_df), dtype=np.float32)

    esm_dim  = next(iter(esm_cache.values())).shape[0]
    zero_emb = np.zeros(esm_dim, dtype=np.float32)

    scores_parts: List[np.ndarray] = []
    for pathogen in split_df["pathogen"].unique():
        sub = split_df[split_df["pathogen"] == pathogen]
        a = np.stack([
            esm_cache.get(str(r.protein_a), zero_emb)
            for r in sub.itertuples()
        ]).astype(np.float32)
        b = np.stack([
            esm_cache.get(str(r.protein_b), zero_emb)
            for r in sub.itertuples()
        ]).astype(np.float32)

        num = np.sum(a * b, axis=1)
        den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
        den = np.clip(den, 1e-8, None)
        scores_parts.append((num / den).astype(np.float32))

    return np.concatenate(scores_parts) if scores_parts else np.zeros(0, dtype=np.float32)


def _embedding_pair_features(
    split_df: pd.DataFrame,
    esm_cache: Dict[str, np.ndarray],
) -> np.ndarray:
    """Return pairwise ESM feature matrix in per-pathogen split order.

    Features per pair: [e_a, e_b, |e_a-e_b|, e_a*e_b].
    """
    if not esm_cache:
        return np.zeros((len(split_df), 4), dtype=np.float32)

    esm_dim  = next(iter(esm_cache.values())).shape[0]
    zero_emb = np.zeros(esm_dim, dtype=np.float32)

    x_parts: List[np.ndarray] = []
    for pathogen in split_df["pathogen"].unique():
        sub = split_df[split_df["pathogen"] == pathogen]
        a = np.stack([
            esm_cache.get(str(r.protein_a), zero_emb)
            for r in sub.itertuples()
        ]).astype(np.float32)
        b = np.stack([
            esm_cache.get(str(r.protein_b), zero_emb)
            for r in sub.itertuples()
        ]).astype(np.float32)
        x_parts.append(np.concatenate([a, b, np.abs(a - b), a * b], axis=1))

    if not x_parts:
        return np.zeros((0, esm_dim * 4), dtype=np.float32)
    return np.concatenate(x_parts, axis=0).astype(np.float32)


def _pathogen_embedding_pair_features(
    split_df: pd.DataFrame,
    protein_lookups: Dict[str, Dict[str, np.ndarray]],
    fallback_dim: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return per-pair features from per-pathogen contextual protein embeddings."""
    x_parts, y_parts, pat_parts = [], [], []
    n_found = n_total = 0

    for pathogen in split_df["pathogen"].unique():
        sub = split_df[split_df["pathogen"] == pathogen]
        lookup = protein_lookups.get(pathogen, {})
        zero_emb = np.zeros(fallback_dim, dtype=np.float32)
        a = np.stack([
            lookup.get(str(r.protein_a), zero_emb)
            for r in sub.itertuples()
        ]).astype(np.float32)
        b = np.stack([
            lookup.get(str(r.protein_b), zero_emb)
            for r in sub.itertuples()
        ]).astype(np.float32)
        mask = np.array([
            str(r.protein_a) in lookup and str(r.protein_b) in lookup
            for r in sub.itertuples()
        ], dtype=bool)

        x_parts.append(np.concatenate([a, b, np.abs(a - b), a * b], axis=1))
        y_parts.append(sub["label"].to_numpy(dtype=np.int32))
        pat_parts.append(np.full(len(sub), pathogen, dtype=object))
        n_total += len(sub)
        n_found += int(mask.sum())

    if not x_parts:
        return (
            np.zeros((0, fallback_dim * 4), dtype=np.float32),
            np.zeros(0, dtype=np.int32),
            np.zeros(0, dtype=object),
            0.0,
        )
    return (
        np.concatenate(x_parts, axis=0).astype(np.float32),
        np.concatenate(y_parts),
        np.concatenate(pat_parts),
        n_found / n_total if n_total else 0.0,
    )


def _split_protein_embeddings(
    split_df: pd.DataFrame,
    protein_lookups: Dict[str, Dict[str, np.ndarray]],
    fallback_dim: int,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Return per-pair protein embeddings (protein_a, protein_b) for a split."""
    x1_parts, x2_parts = [], []
    n_found = n_total = 0

    for pathogen in split_df["pathogen"].unique():
        sub = split_df[split_df["pathogen"] == pathogen]
        lookup = protein_lookups.get(pathogen, {})
        zero_emb = np.zeros(fallback_dim, dtype=np.float32)
        x1 = np.stack([
            lookup.get(str(r.protein_a), zero_emb)
            for r in sub.itertuples()
        ]).astype(np.float32)
        x2 = np.stack([
            lookup.get(str(r.protein_b), zero_emb)
            for r in sub.itertuples()
        ]).astype(np.float32)
        mask = np.array([
            str(r.protein_a) in lookup and str(r.protein_b) in lookup
            for r in sub.itertuples()
        ], dtype=bool)

        x1_parts.append(x1)
        x2_parts.append(x2)
        n_total += len(sub)
        n_found += int(mask.sum())

    if not x1_parts:
        return (
            np.zeros((0, fallback_dim), dtype=np.float32),
            np.zeros((0, fallback_dim), dtype=np.float32),
            0.0,
        )
    return (
        np.concatenate(x1_parts, axis=0).astype(np.float32),
        np.concatenate(x2_parts, axis=0).astype(np.float32),
        n_found / n_total if n_total else 0.0,
    )


def _select_lr_regularization(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    val_pathogens: np.ndarray,
    label: str,
) -> float:
    best_C, best_val_auc = 1.0, -1.0
    for C in (0.01, 0.1, 1.0, 10.0):
        clf = LogisticRegression(C=C, class_weight="balanced",
                                 max_iter=1000, random_state=42)
        clf.fit(X_tr, y_tr)
        auc = _macro_species_auroc(y_va, clf.predict_proba(X_va)[:, 1], val_pathogens)
        if auc > best_val_auc:
            best_val_auc, best_C = auc, C
    print(f"  {label} best C={best_C}  val macro-AUROC = {best_val_auc:.4f}")
    return best_C


# ---------------------------------------------------------------------------
# Per-pathogen attention cache
# ---------------------------------------------------------------------------

def _ckpt_key(checkpoint_path: str) -> str:
    """8-char sha256 prefix of the resolved checkpoint path."""
    return hashlib.sha256(
        str(Path(checkpoint_path).resolve()).encode()
    ).hexdigest()[:8]


def build_pathogen_feature_cache(
    pathogen: str,
    all_pairs: List[Tuple[str, str]],
    benchmark_raw_dir: Path,
    esm_cache_dir: Path,
    cache_path: Path,
    model,
    device: str,
) -> Tuple[Dict[Tuple[str, str], np.ndarray], Dict[str, np.ndarray]]:
    """
    Compute (or load from cache) flattened attention features and contextual
    ProteomeLM protein embeddings for a pathogen.
    """
    if cache_path.exists():
        npz = np.load(cache_path, allow_pickle=True)
        if {"pair_ids", "pair_features", "protein_ids", "protein_features"}.issubset(npz.files):
            pair_lookup: Dict[Tuple[str, str], np.ndarray] = {}
            for ids, feat in zip(npz["pair_ids"], npz["pair_features"]):
                a, b = str(ids[0]), str(ids[1])
                pair_lookup[(a, b)] = feat
                pair_lookup[(b, a)] = feat
            protein_lookup = {
                str(pid): feat
                for pid, feat in zip(npz["protein_ids"], npz["protein_features"])
            }
            print(f"  [{pathogen}] cache hit: {len(npz['pair_ids'])} pairs, "
                  f"{len(npz['protein_ids'])} proteins from {cache_path.name}")
            return pair_lookup, protein_lookup

        # Legacy cache format: ignore and rebuild with contextual protein embeddings.
        print(f"  [{pathogen}] legacy cache format detected — rebuilding {cache_path.name}")

    combined_fasta = benchmark_raw_dir / pathogen / "combined_proteome.fasta"
    if not combined_fasta.exists():
        print(f"  [{pathogen}] WARNING: no combined_proteome.fasta — skipping")
        return {}, {}

    print(f"\n  [{pathogen}] — ESM-C encoding")
    esm_data, protein_to_idx = encode_split_cached(combined_fasta, esm_cache_dir, device)
    protein_to_idx = add_gene_name_aliases(combined_fasta, protein_to_idx)

    print(f"  [{pathogen}] — ProteomeLM forward pass")
    protein_features, attentions = run_inference(model, esm_data, return_logits=True)

    valid_pairs: List[Tuple[str, str]] = []
    idx_a_list: List[int] = []
    idx_b_list: List[int] = []
    for a, b in all_pairs:
        ia = protein_to_idx.get(a)
        ib = protein_to_idx.get(b)
        if ia is not None and ib is not None:
            valid_pairs.append((a, b))
            idx_a_list.append(ia)
            idx_b_list.append(ib)

    if not valid_pairs:
        print(f"  [{pathogen}] WARNING: no pairs resolved in proteome")
        return {}, {}

    n_missing = len(all_pairs) - len(valid_pairs)
    if n_missing:
        print(f"  [{pathogen}] {n_missing} pairs not found in proteome index")

    idx_a = np.array(idx_a_list)
    idx_b = np.array(idx_b_list)
    attn_ab = np.stack([a[:, idx_a, idx_b] for a in attentions]).transpose(2, 0, 1)
    attn_ba = np.stack([a[:, idx_b, idx_a] for a in attentions]).transpose(2, 0, 1)
    avg_attn = (attn_ab + attn_ba) / 2.0
    pair_features = avg_attn.reshape(len(valid_pairs), -1).astype(np.float32)

    protein_ids = np.array(esm_data["group_labels"], dtype=object)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        pair_ids=np.array(valid_pairs, dtype=object),
        pair_features=pair_features,
        protein_ids=protein_ids,
        protein_features=protein_features,
    )
    print(f"  [{pathogen}] saved {len(valid_pairs)} pairs + {len(protein_ids)} proteins "
          f"→ {cache_path.name}")

    pair_lookup: Dict[Tuple[str, str], np.ndarray] = {}
    for (a, b), feat in zip(valid_pairs, pair_features):
        pair_lookup[(a, b)] = feat
        pair_lookup[(b, a)] = feat

    protein_lookup = {
        str(pid): protein_features[i]
        for i, pid in enumerate(protein_ids)
    }
    for alias, idx in protein_to_idx.items():
        protein_lookup.setdefault(str(alias), protein_features[idx])

    return pair_lookup, protein_lookup


def build_all_lookups(
    all_dfs: List[pd.DataFrame],
    benchmark_raw_dir: Path,
    esm_cache_dir: Path,
    attn_cache_dir: Path,
    checkpoint_path: str,
    base_model: str,
    device: str,
) -> Tuple[Dict[str, Dict[Tuple[str, str], np.ndarray]], Dict[str, Dict[str, np.ndarray]]]:
    """
    Load ProteomeLM once, then extract + cache attention features for every
    pathogen that appears in any of the supplied DataFrames.
    """
    combined = pd.concat(all_dfs, ignore_index=True)

    pathogen_pairs: Dict[str, set] = {}
    for _, row in combined.iterrows():
        p = row["pathogen"]
        pathogen_pairs.setdefault(p, set()).add(
            (str(row["protein_a"]), str(row["protein_b"]))
        )

    ckpt_key   = _ckpt_key(checkpoint_path)
    ckpt_dir   = attn_cache_dir / ckpt_key

    print(f"\n{'='*65}")
    print(f"Loading ProteomeLM checkpoint")
    print(f"  key: {ckpt_key}  ({checkpoint_path})")
    print(f"{'='*65}")
    model = load_finetuned(checkpoint_path, base_model)

    attn_lookups: Dict[str, Dict] = {}
    protein_lookups: Dict[str, Dict] = {}
    for pathogen in sorted(pathogen_pairs):
        pair_lookup, protein_lookup = build_pathogen_feature_cache(
            pathogen        = pathogen,
            all_pairs       = list(pathogen_pairs[pathogen]),
            benchmark_raw_dir = benchmark_raw_dir,
            esm_cache_dir   = esm_cache_dir,
            cache_path      = ckpt_dir / f"{pathogen}_goldfeatures.npz",
            model           = model,
            device          = device,
        )
        attn_lookups[pathogen] = pair_lookup
        protein_lookups[pathogen] = protein_lookup

    del model
    torch.cuda.empty_cache()
    return attn_lookups, protein_lookups


# ---------------------------------------------------------------------------
# Feature matrix assembly
# ---------------------------------------------------------------------------

def pairs_to_features(
    df: pd.DataFrame,
    lookup: Dict[Tuple[str, str], np.ndarray],
    fallback_dim: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    actual_dim = next(iter(lookup.values())).shape[0] if lookup else fallback_dim
    X_list, y_list, found = [], [], []
    for _, row in df.iterrows():
        key  = (str(row["protein_a"]), str(row["protein_b"]))
        feat = lookup.get(key)
        if feat is None:
            feat = lookup.get((key[1], key[0]))
        X_list.append(feat if feat is not None else np.zeros(actual_dim, dtype=np.float32))
        y_list.append(int(row["label"]))
        found.append(feat is not None)
    return (np.array(X_list, dtype=np.float32),
            np.array(y_list, dtype=np.int32),
            np.array(found, dtype=bool))


def load_split_features(
    split_df: pd.DataFrame,
    lookups: Dict[str, Dict],
    fallback_dim: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return (X, y, pathogens, coverage) for a split DataFrame.

    Features are z-score normalized *per pathogen* before concatenation so
    that the classifier sees relative attention signal (above/below this
    pathogen's baseline) rather than absolute values.  Absolute magnitudes
    are not comparable across pathogens because the softmax partition
    function scales with proteome size.
    """
    X_parts, y_parts, pat_parts = [], [], []
    deferred: List[Tuple[int, np.ndarray, np.ndarray]] = []
    actual_feat_dim: Optional[int] = None
    n_found = n_total = 0

    for pathogen in split_df["pathogen"].unique():
        sub    = split_df[split_df["pathogen"] == pathogen]
        lookup = lookups.get(pathogen, {})
        y_arr  = sub["label"].to_numpy(dtype=np.int32)
        p_arr  = np.full(len(sub), pathogen, dtype=object)
        n_total += len(sub)

        if not lookup:
            print(f"  No lookup for {pathogen} — deferred to zeros")
            deferred.append((len(sub), y_arr, p_arr))
            continue

        X, y, mask = pairs_to_features(sub, lookup, fallback_dim)
        if actual_feat_dim is None:
            actual_feat_dim = X.shape[1]

        # Per-pathogen z-score: makes attention signal comparable across
        # proteomes of different sizes and compositions.
        if X.shape[0] > 1:
            mu  = X.mean(axis=0, keepdims=True)
            std = X.std(axis=0, keepdims=True).clip(min=1e-8)
            X   = (X - mu) / std

        X_parts.append(X)
        y_parts.append(y_arr)
        pat_parts.append(p_arr)
        n_found += int(mask.sum())

    true_dim = actual_feat_dim if actual_feat_dim is not None else fallback_dim
    for n, y_arr, p_arr in deferred:
        X_parts.append(np.zeros((n, true_dim), dtype=np.float32))
        y_parts.append(y_arr)
        pat_parts.append(p_arr)

    if not X_parts:
        return (np.zeros((0, true_dim), dtype=np.float32),
                np.zeros(0, dtype=np.int32),
                np.zeros(0, dtype=object), 0.0)
    return (np.concatenate(X_parts),
            np.concatenate(y_parts),
            np.concatenate(pat_parts),
            n_found / n_total if n_total else 0.0)


# ---------------------------------------------------------------------------
# ESM-C per-protein cache (used only by supervised MLP)
# ---------------------------------------------------------------------------

def build_protein_esm_cache(
    all_dfs: List[pd.DataFrame],
    cache_path: Path,
    device: str,
    benchmark_raw_dir: Path,
) -> Dict[str, np.ndarray]:
    """Build/load a flat {accession: embedding} dict for all proteins in all_dfs."""
    cache: Dict[str, np.ndarray] = {}
    if cache_path.exists():
        loaded = torch.load(cache_path, map_location="cpu")
        if isinstance(loaded, dict):
            cache = {k: (v.numpy() if isinstance(v, torch.Tensor) else v)
                     for k, v in loaded.items()}
        print(f"  Loaded {len(cache):,} ESM embeddings from {cache_path}")

    combined = pd.concat(all_dfs, ignore_index=True)
    needed   = (set(combined["protein_a"].astype(str)) |
                set(combined["protein_b"].astype(str)))
    missing  = needed - set(cache.keys())
    if not missing:
        return cache

    print(f"  {len(missing):,} accessions missing — scanning FASTAs …")
    acc_to_seq: Dict[str, str] = {}
    for fasta_path in sorted(benchmark_raw_dir.glob("*/combined_proteome.fasta")):
        current_id, chunks = None, []
        with fasta_path.open() as fh:
            for line in fh:
                line = line.rstrip()
                if line.startswith(">"):
                    if current_id and current_id in missing:
                        acc_to_seq[current_id] = "".join(chunks)
                    chunks = []
                    parts = line[1:].split("|")
                    current_id = parts[1] if len(parts) >= 2 else line[1:].split()[0]
                else:
                    chunks.append(line)
            if current_id and current_id in missing:
                acc_to_seq[current_id] = "".join(chunks)

    still_missing = missing - set(acc_to_seq.keys())
    if still_missing:
        print(f"  Warning: {len(still_missing):,} accession(s) not found in any FASTA "
              f"(will use zero embedding):")
        for acc in sorted(still_missing)[:20]:
            print(f"    {acc}")

    from esm.models.esmc import ESMC
    from tqdm import tqdm
    esm_model = ESMC.from_pretrained("esmc_600m").eval().to(device, dtype=torch.bfloat16)
    pad_id    = esm_model.tokenizer.pad_token_id
    for acc in tqdm(sorted(acc_to_seq, key=lambda a: len(acc_to_seq[a])),
                    desc="ESM-C encode"):
        ids = esm_model._tokenize([acc_to_seq[acc][:4096]]).long().to(device)
        with torch.no_grad():
            out = esm_model(ids)
        emb  = out.embeddings
        mask = ids != pad_id
        emb[~mask] = 0.0
        pooled = emb.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
        cache[acc] = pooled[0].float().cpu().numpy()
    del esm_model
    torch.cuda.empty_cache()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: torch.tensor(v) for k, v in cache.items()}, cache_path)
    print(f"  ESM cache saved → {cache_path}  ({len(cache):,} entries)")
    return cache


# ---------------------------------------------------------------------------
# Core fold runner
# ---------------------------------------------------------------------------

def run_fold(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    attn_lookups: Dict[str, Dict],
    protein_lookups: Dict[str, Dict],
    attn_dim: int,
    protein_dim: int,
    fold_name: str,
    no_supervised: bool,
    esm_cache: Dict[str, np.ndarray],
    n_replicas: int,
    n_epochs: int,
    patience: int,
    device: str,
) -> Dict[str, Optional[List[dict]]]:
    print(f"\n  {'─' * 60}")
    print(f"  Fold {fold_name}")
    print(f"    train: {sorted(train_df['pathogen'].unique())}")
    print(f"    test:  {sorted(test_df['pathogen'].unique())}")
    print(f"  {'─' * 60}")

    Xatt_tr, y_tr, _,      cov_att_tr = load_split_features(train_df, attn_lookups, attn_dim)
    Xatt_va, y_va, _,      cov_att_va = load_split_features(val_df,   attn_lookups, attn_dim)
    Xatt_te, y_te, pat_te, cov_att_te = load_split_features(test_df,  attn_lookups, attn_dim)

    Xplm_tr, _, _, cov_plm_pair_tr = _pathogen_embedding_pair_features(train_df, protein_lookups, protein_dim)
    Xplm_va, _, _, cov_plm_pair_va = _pathogen_embedding_pair_features(val_df,   protein_lookups, protein_dim)
    Xplm_te, _, _, cov_plm_pair_te = _pathogen_embedding_pair_features(test_df,  protein_lookups, protein_dim)

    X_tr = np.concatenate([Xatt_tr, Xplm_tr], axis=1)
    X_va = np.concatenate([Xatt_va, Xplm_va], axis=1)
    X_te = np.concatenate([Xatt_te, Xplm_te], axis=1)

    print(f"  Attention coverage   — train {cov_att_tr:.1%}  val {cov_att_va:.1%}  test {cov_att_te:.1%}")
    print(f"  ProteomeLM coverage  — train {cov_plm_pair_tr:.1%}  val {cov_plm_pair_va:.1%}  test {cov_plm_pair_te:.1%}")
    print(f"  Shapes               — attn train {Xatt_tr.shape}  plm-pair train {Xplm_tr.shape}")
    print(f"                         combined train {X_tr.shape}  val {X_va.shape}  test {X_te.shape}")

    # ---- ProteomeLM-Attn-LR (attention only) ----
    scaler_attn = StandardScaler()
    Xatt_tr_sc = scaler_attn.fit_transform(Xatt_tr)
    Xatt_va_sc = scaler_attn.transform(Xatt_va)
    Xatt_te_sc = scaler_attn.transform(Xatt_te)

    best_C_attn = _select_lr_regularization(
        Xatt_tr_sc, y_tr, Xatt_va_sc, y_va,
        val_df["pathogen"].to_numpy(dtype=object),
        label="ProteomeLM-Attn-LR",
    )

    clf_attn = LogisticRegression(C=best_C_attn, class_weight="balanced",
                                  max_iter=1000, random_state=42)
    clf_attn.fit(Xatt_tr_sc, y_tr)
    probs_attn = clf_attn.predict_proba(Xatt_te_sc)[:, 1]
    attn_lr_rows = _per_species_auroc(y_te, probs_attn, pat_te, fold_name)
    _print_per_species(attn_lr_rows, f"ProteomeLM-Attn-LR, fold {fold_name}")

    # ---- ProteomeLM-LR (contextual embeddings + attention) ----
    scaler  = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_va_sc = scaler.transform(X_va)
    X_te_sc = scaler.transform(X_te)

    best_C = _select_lr_regularization(
        X_tr_sc, y_tr, X_va_sc, y_va,
        val_df["pathogen"].to_numpy(dtype=object),
        label="ProteomeLM-LR",
    )

    clf = LogisticRegression(C=best_C, class_weight="balanced",
                             max_iter=1000, random_state=42)
    clf.fit(X_tr_sc, y_tr)

    test_probs = clf.predict_proba(X_te_sc)[:, 1]
    lr_rows = _per_species_auroc(y_te, test_probs, pat_te, fold_name)
    _print_per_species(lr_rows, f"ProteomeLM-LR, fold {fold_name}")

    # ---- Cosine baseline on ESM-C embeddings ----
    cos_scores = _cosine_similarity_pairs(test_df, esm_cache)
    if len(cos_scores) != len(y_te):
        print("  Warning: cosine score length mismatch; defaulting to zeros")
        cos_scores = np.zeros_like(y_te, dtype=np.float32)
    cos_rows = _per_species_auroc(
        y_te,
        cos_scores,
        pat_te,
        fold_name,
        orientation_invariant=False,
    )
    _print_per_species(cos_rows, f"Cosine, fold {fold_name}")

    # ---- Embedding-only LR baseline ----
    Xemb_tr = _embedding_pair_features(train_df, esm_cache)
    Xemb_va = _embedding_pair_features(val_df, esm_cache)
    Xemb_te = _embedding_pair_features(test_df, esm_cache)

    scaler_emb = StandardScaler()
    Xemb_tr_sc = scaler_emb.fit_transform(Xemb_tr)
    Xemb_va_sc = scaler_emb.transform(Xemb_va)
    Xemb_te_sc = scaler_emb.transform(Xemb_te)

    best_C_emb = _select_lr_regularization(
        Xemb_tr_sc, y_tr, Xemb_va_sc, y_va,
        val_df["pathogen"].to_numpy(dtype=object),
        label="ESM-LR",
    )

    clf_emb = LogisticRegression(C=best_C_emb, class_weight="balanced",
                                 max_iter=1000, random_state=42)
    clf_emb.fit(Xemb_tr_sc, y_tr)
    probs_emb = clf_emb.predict_proba(Xemb_te_sc)[:, 1]
    esm_lr_rows = _per_species_auroc(y_te, probs_emb, pat_te, fold_name)
    _print_per_species(esm_lr_rows, f"ESM-LR, fold {fold_name}")

    # ---- Supervised MLP (optional) ----
    attn_mlp_rows: Optional[List[dict]] = None
    mlp_rows: Optional[List[dict]] = None
    if not no_supervised:
        plm_tr_a, plm_tr_b, cov_prot_tr = _split_protein_embeddings(train_df, protein_lookups, protein_dim)
        plm_va_a, plm_va_b, cov_prot_va = _split_protein_embeddings(val_df,   protein_lookups, protein_dim)
        plm_te_a, plm_te_b, cov_prot_te = _split_protein_embeddings(test_df,  protein_lookups, protein_dim)
        print(f"  ProteomeLM protein coverage — train {cov_prot_tr:.1%}  val {cov_prot_va:.1%}  test {cov_prot_te:.1%}")

        try:
            from proteomelm.ppi.model import train_model_cv
            import torch as _torch

            def _run_enhancedppi_variant(
                *,
                label: str,
                train_inputs: Dict[str, Optional[np.ndarray]],
                val_inputs: Dict[str, Optional[np.ndarray]],
                test_inputs: Dict[str, Optional[np.ndarray]],
            ) -> List[dict]:
                all_probs: List[np.ndarray] = []
                for rep in range(n_replicas):
                    print(f"  {label} replica {rep + 1}/{n_replicas}")
                    model_sup, val_m = train_model_cv(
                        train_inputs,
                        val_inputs,
                        y_tr,
                        y_va,
                        n_epochs=n_epochs,
                        patience=patience,
                        model_type="enhancedppi",
                        verbose=False,
                        replica_seed=rep,
                        use_class_weights=True,
                        loss_type="focal",
                    )
                    model_sup.eval()
                    _dev = next(model_sup.parameters()).device
                    with _torch.no_grad():
                        logits = model_sup(
                            None if test_inputs["edges"] is None else _torch.tensor(test_inputs["edges"], dtype=_torch.float32).to(_dev),
                            None if test_inputs["x1"] is None else _torch.tensor(test_inputs["x1"], dtype=_torch.float32).to(_dev),
                            None if test_inputs["x2"] is None else _torch.tensor(test_inputs["x2"], dtype=_torch.float32).to(_dev),
                        ).cpu().numpy().flatten()
                    all_probs.append(1.0 / (1.0 + np.exp(-logits)))
                    print(f"    val AUROC={val_m['auc']:.4f}")

                avg_probs = np.mean(all_probs, axis=0)
                rows = _per_species_auroc(y_te, avg_probs, pat_te, fold_name)
                _print_per_species(rows, f"{label} ×{n_replicas}, fold {fold_name}")
                return rows

            attn_mlp_rows = _run_enhancedppi_variant(
                label="ProteomeLM-Attn-EnhancedPPI",
                train_inputs={"edges": Xatt_tr, "x1": None, "x2": None},
                val_inputs={"edges": Xatt_va, "x1": None, "x2": None},
                test_inputs={"edges": Xatt_te, "x1": None, "x2": None},
            )

            mlp_rows = _run_enhancedppi_variant(
                label="ProteomeLM-MLP",
                train_inputs={"edges": Xatt_tr, "x1": plm_tr_a, "x2": plm_tr_b},
                val_inputs={"edges": Xatt_va, "x1": plm_va_a, "x2": plm_va_b},
                test_inputs={"edges": Xatt_te, "x1": plm_te_a, "x2": plm_te_b},
            )

        except ImportError as e:
            print(f"  Skipping MLP: {e}")

    return {
        "attn_lr": attn_lr_rows,
        "lr": lr_rows,
        "cosine": cos_rows,
        "esm_lr": esm_lr_rows,
        "attn_mlp": attn_mlp_rows,
        "mlp": mlp_rows,
    }


# ---------------------------------------------------------------------------
# Figure + TSV output
# ---------------------------------------------------------------------------

def _draw_panel(ax: plt.Axes, rows: List[dict], title: str) -> None:
    rows_sorted = sorted(rows, key=lambda r: r["auroc"])
    labels = [SPECIES_LABELS.get(r["pathogen"], r["pathogen"]) for r in rows_sorted]
    aurocs = [r["auroc"] for r in rows_sorted]
    colors = [FOLD_COLORS[r["fold"]] for r in rows_sorted]
    y = np.arange(len(rows_sorted))

    ax.barh(y, aurocs, height=0.6, color=colors, zorder=3)
    ax.axvline(0.5, color="gray", linestyle=":", lw=1, zorder=2)
    for i, a in enumerate(aurocs):
        ax.text(a + 0.005, i, f"{a:.3f}", va="center", fontsize=9)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("AUROC (HPI vs. random cross-species)")
    ax.set_title(title)
    ax.set_xlim(0.4, 1.07)
    ax.xaxis.grid(True, alpha=0.3, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fold_names = sorted({row["fold"] for row in rows})
    ax.legend(
        handles=[
            mpatches.Patch(color=FOLD_COLORS[fold], label=f"Fold {fold}")
            for fold in fold_names
        ],
        loc="lower right",
        fontsize=9,
    )


def load_fold_tables(gold_dir: Path) -> List[Tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """Load explicit fold directories, with fallback to the legacy two-fold layout."""
    fold_dirs = sorted(
        path for path in gold_dir.glob("fold_*")
        if path.is_dir()
    )
    if fold_dirs:
        folds = []
        for fold_dir in fold_dirs:
            fold_name = fold_dir.name.split("_", 1)[-1].upper()
            missing = [name for name in ("train.tsv", "val.tsv", "test.tsv") if not (fold_dir / name).exists()]
            if missing:
                sys.exit(f"ERROR: {fold_dir} is missing {missing}. Run build_gold_benchmark.py first.")
            folds.append((
                fold_name,
                pd.read_csv(fold_dir / "train.tsv", sep="\t"),
                pd.read_csv(fold_dir / "val.tsv", sep="\t"),
                pd.read_csv(fold_dir / "test.tsv", sep="\t"),
            ))
        return folds

    missing = [name for name in ("train.tsv", "val.tsv", "test.tsv") if not (gold_dir / name).exists()]
    if missing:
        sys.exit(
            f"ERROR: {gold_dir} is missing {missing}. Run build_gold_benchmark.py first."
        )

    train_df = pd.read_csv(gold_dir / "train.tsv", sep="\t")
    val_df = pd.read_csv(gold_dir / "val.tsv", sep="\t")
    test_df = pd.read_csv(gold_dir / "test.tsv", sep="\t")
    return [
        ("A", train_df, val_df, test_df),
        ("B", test_df, val_df, train_df),
    ]


def save_figures(
    cosine_rows: List[dict],
    esm_lr_rows: List[dict],
    attn_lr_rows: List[dict],
    plm_lr_rows: List[dict],
    attn_mlp_rows: Optional[List[dict]],
    mlp_rows: Optional[List[dict]],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    tsv_path = out_dir / "gold_benchmark_per_species.tsv"
    df_cos_cols = ["pathogen", "fold", "n_pos", "n_neg", "auroc"]
    if cosine_rows and "auroc_raw" in cosine_rows[0]:
        df_cos_cols += ["auroc_raw", "direction"]
    df_cos = pd.DataFrame(cosine_rows)[df_cos_cols]
    df_cos["model"] = "cosine"
    df_attn_lr = pd.DataFrame(attn_lr_rows)[["pathogen", "fold", "n_pos", "n_neg", "auroc"]]
    df_attn_lr["model"] = "proteomelm_attn_lr"
    df_lr = pd.DataFrame(plm_lr_rows)[["pathogen", "fold", "n_pos", "n_neg", "auroc"]]
    df_lr["model"] = "proteomelm_lr"
    df_esm_lr = pd.DataFrame(esm_lr_rows)[["pathogen", "fold", "n_pos", "n_neg", "auroc"]]
    df_esm_lr["model"] = "esm_lr"
    to_concat = [df_cos, df_esm_lr, df_attn_lr, df_lr]
    if attn_mlp_rows:
        df_attn_mlp = pd.DataFrame(attn_mlp_rows)[["pathogen", "fold", "n_pos", "n_neg", "auroc"]]
        df_attn_mlp["model"] = "proteomelm_attn_enhancedppi"
        to_concat.append(df_attn_mlp)
    if mlp_rows:
        df_mlp = pd.DataFrame(mlp_rows)[["pathogen", "fold", "n_pos", "n_neg", "auroc"]]
        df_mlp["model"] = "proteomelm_mlp"
        to_concat.append(df_mlp)
    pd.concat(to_concat, ignore_index=True).to_csv(tsv_path, sep="\t", index=False)
    print(f"\n  Saved → {tsv_path}")

    panels: List[Tuple[List[dict], str]] = [
        (cosine_rows, "Gold benchmark — cosine baseline (ESM-C)"),
        (esm_lr_rows, "Gold benchmark — ESM-LR baseline"),
        (attn_lr_rows, "Gold benchmark — ProteomeLM-Attn-LR"),
        (plm_lr_rows, "Gold benchmark — ProteomeLM-LR (embeddings + attention)"),
    ]
    if attn_mlp_rows:
        panels.append((attn_mlp_rows, "Gold benchmark — ProteomeLM-Attn-EnhancedPPI"))
    if mlp_rows:
        panels.append((mlp_rows, "Gold benchmark — ProteomeLM-MLP (embeddings + attention)"))

    n_panels = len(panels)
    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(5.6 * n_panels, max(4, len(plm_lr_rows) * 0.65)),
    )
    if n_panels == 1:
        axes = [axes]

    for ax, (rows, title) in zip(axes, panels):
        _draw_panel(ax, rows, title)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"gold_benchmark_per_species.{ext}",
                    dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out_dir}/gold_benchmark_per_species.{{pdf,png}}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--gold-dir",      type=Path, default=_DEFAULT_GOLD)
    parser.add_argument("--benchmark-dir", type=Path, default=_DEFAULT_BENCH_RAW,
                        help="Path to benchmarks/raw/ (contains {pathogen}/combined_proteome.fasta)")
    parser.add_argument("--checkpoint",    default=str(_DEFAULT_CKPT),
                        help="ProteomeLM-HPI LoRA checkpoint directory")
    parser.add_argument("--base-model",    default=_DEFAULT_BASE,
                        help="HuggingFace name or local path of ProteomeLM base model")
    parser.add_argument("--esm-cache-dir", type=Path, default=_DEFAULT_ESM_CACHE,
                        help="Directory for per-proteome ESM-C embedding caches")
    parser.add_argument("--attn-cache-dir", type=Path, default=_DEFAULT_ATTN_CACHE,
                        help="Directory for per-checkpoint attention feature caches")
    parser.add_argument("--device",        default="cuda:0",
                        help="Device for ESM-C encoding (ProteomeLM always on CPU)")
    parser.add_argument("--no-supervised", action="store_true",
                        help="Skip MLP, run LR only")
    parser.add_argument("--n-replicas",    type=int, default=3)
    parser.add_argument("--n-epochs",      type=int, default=100)
    parser.add_argument("--patience",      type=int, default=15)
    parser.add_argument("--out-dir",       type=Path, default=_FIGS_DIR)
    args = parser.parse_args()

    # ---- Load splits ----
    print("=" * 65)
    print("Loading gold benchmark splits")
    print("=" * 65)
    fold_tables = load_fold_tables(args.gold_dir)
    for fold_name, train_df, val_df, test_df in fold_tables:
        print(f"  Fold {fold_name}")
        for split_name, df in (("train", train_df), ("val", val_df), ("test", test_df)):
            pos = int((df.label == 1).sum())
            print(f"    {split_name}: {len(df):,} pairs  ({pos:,} pos)  "
                  f"pathogens: {sorted(df['pathogen'].unique())}")

    all_dfs = [df for _, train_df, val_df, test_df in fold_tables for df in (train_df, val_df, test_df)]

    # ---- Build attention feature lookups (runs ProteomeLM inline) ----
    attn_lookups, protein_lookups = build_all_lookups(
        all_dfs           = all_dfs,
        benchmark_raw_dir = args.benchmark_dir,
        esm_cache_dir     = args.esm_cache_dir,
        attn_cache_dir    = args.attn_cache_dir,
        checkpoint_path   = args.checkpoint,
        base_model        = args.base_model,
        device            = args.device,
    )

    # Infer feature dimensions from any available lookup
    attn_dim = 144
    for lk in attn_lookups.values():
        if lk:
            attn_dim = next(iter(lk.values())).shape[0]
            break
    protein_dim = 1152
    for lk in protein_lookups.values():
        if lk:
            protein_dim = next(iter(lk.values())).shape[0]
            break
    print(f"\n  Attention feature dim: {attn_dim}")
    print(f"  ProteomeLM embedding dim: {protein_dim}")

    # ---- ESM-C per-protein cache (supervised MLP only) ----
    esm_cache: Dict[str, np.ndarray] = {}
    print("\n" + "=" * 65)
    print("Building / loading per-protein ESM-C cache (simple baselines only)")
    print("=" * 65)
    esm_cache_path = args.esm_cache_dir / "protein_cache.pt"
    esm_cache = build_protein_esm_cache(
        all_dfs,
        esm_cache_path,
        args.device,
        args.benchmark_dir,
    )

    # ---- Fold evaluation ----
    print("\n" + "=" * 65)
    print(f"Evaluating {len(fold_tables)} fold(s)")
    print("=" * 65)

    fold_kwargs = dict(
        attn_lookups = attn_lookups,
        protein_lookups = protein_lookups,
        attn_dim     = attn_dim,
        protein_dim  = protein_dim,
        no_supervised = args.no_supervised,
        esm_cache    = esm_cache,
        n_replicas   = args.n_replicas,
        n_epochs     = args.n_epochs,
        patience     = args.patience,
        device       = args.device,
    )

    fold_results = []
    for fold_name, train_df, val_df, test_df in fold_tables:
        fold_results.append(
            run_fold(train_df, val_df, test_df, fold_name=fold_name, **fold_kwargs)
        )

    # ---- Combine and save ----
    cosine_rows = [row for result in fold_results for row in result["cosine"]]
    esm_lr_rows = [row for result in fold_results for row in result["esm_lr"]]
    attn_lr_rows = [row for result in fold_results for row in result["attn_lr"]]
    plm_lr_rows = [row for result in fold_results for row in result["lr"]]
    attn_mlp_present = [result["attn_mlp"] for result in fold_results if result["attn_mlp"]]
    attn_mlp_rows = [row for rows in attn_mlp_present for row in rows] if attn_mlp_present else None
    mlp_present = [result["mlp"] for result in fold_results if result["mlp"]]
    mlp_rows = [row for rows in mlp_present for row in rows] if mlp_present else None

    print("\n" + "=" * 65)
    print(f"Combined per-species AUROC ({len(plm_lr_rows)} fold-test pathogen results)")
    print("=" * 65)
    _print_per_species(cosine_rows, "Cosine")
    _print_per_species(esm_lr_rows, "ESM-LR")
    _print_per_species(attn_lr_rows, "ProteomeLM-Attn-LR")
    _print_per_species(plm_lr_rows, "ProteomeLM-LR")
    if attn_mlp_rows:
        _print_per_species(attn_mlp_rows, f"ProteomeLM-Attn-EnhancedPPI ×{args.n_replicas}")
    if mlp_rows:
        _print_per_species(mlp_rows, f"ProteomeLM-MLP ×{args.n_replicas}")

    save_figures(cosine_rows, esm_lr_rows, attn_lr_rows, plm_lr_rows, attn_mlp_rows, mlp_rows, args.out_dir)


if __name__ == "__main__":
    main()
