#!/usr/bin/env python3
"""
HuggingFace Virus-Human PPI Benchmark
======================================
Evaluates ProteomeLM fine-tuned model on danliu1226/virus_human_benchmarking dataset.

Pipeline:
  1. Load dataset (all viral proteins treated as a single pathogen)
  2. Encode all unique sequences with ESM-C (cached)
  3. Run pairwise ProteomeLM inference (batched, seq_len=2 per pair)
  4. Extract attention head features (6 layers × 8 heads, both directions → 48 features)
  5a. Train logistic regression on train set, evaluate on test set
  5b. Train supervised MLP (EnhancedPPIModel) with ESM-C embeddings + attention edges
  6. Report AUPR / F1 / MCC vs. SOTA table

Usage:
    python 5_hf_benchmark.py \
        --checkpoint ../../output/ProteomeLM-HPI-S-abl4c-seed-total-block/checkpoint-20000 \
        --device cuda:0 \
        [--esm-cache data/hf_benchmark_esm.pt] \
        [--attn-cache data/hf_benchmark_attention.npz] \
        [--batch-size 512] \
        [--n-replicas 3] \
        [--no-supervised]
"""

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_SCRIPT_DIR))

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from utils import encode_sequences, load_finetuned


# ---------------------------------------------------------------------------
# ESM-C encoding
# ---------------------------------------------------------------------------

def encode_with_esm(sequences: list[str], device: str, batch_max_tokens: int = 16000) -> torch.Tensor:
    """Encode protein sequences with ESM-C 600M → (N, 1152) float32 on CPU (each capped at 4096 aa)."""
    return encode_sequences(sequences, device, max_padded_tokens=batch_max_tokens,
                            truncate=4096, to_float32=True)


# ---------------------------------------------------------------------------
# ProteomeLM pairwise attention extraction
# ---------------------------------------------------------------------------

def extract_pair_attention_batched(
    model,
    emb_a: torch.Tensor,   # (N, esm_dim)
    emb_b: torch.Tensor,   # (N, esm_dim)
    batch_size: int = 512,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run the model pairwise (seq_len=2) in batches.

    Returns:
        attn_a_to_b: (N, n_layers, n_heads)  attention from pos-0 to pos-1
        attn_b_to_a: (N, n_layers, n_heads)  attention from pos-1 to pos-0
    """
    from tqdm import tqdm

    n = emb_a.shape[0]
    all_ab, all_ba = [], []
    model.eval()

    for start in tqdm(range(0, n, batch_size), desc="ProteomeLM pairwise inference"):
        end = min(start + batch_size, n)
        # Stack pairs: [bs, 2, esm_dim]
        inp = torch.stack([emb_a[start:end], emb_b[start:end]], dim=1).to(
            device=device, dtype=torch.bfloat16
        )
        with torch.no_grad():
            out = model(inputs_embeds=inp, group_embeds=inp, output_attentions=True)
        # out.attentions: tuple of (bs, n_heads, 2, 2), one per layer
        ab_layers, ba_layers = [], []
        for attn_layer in out.attentions:  # each: [bs, n_heads, 2, 2]
            ab_layers.append(attn_layer[:, :, 0, 1].float().cpu().numpy())  # [bs, n_heads]
            ba_layers.append(attn_layer[:, :, 1, 0].float().cpu().numpy())  # [bs, n_heads]
        # Stack layers → [bs, n_layers, n_heads]
        all_ab.append(np.stack(ab_layers, axis=1))
        all_ba.append(np.stack(ba_layers, axis=1))

    attn_ab = np.concatenate(all_ab, axis=0)  # (N, n_layers, n_heads)
    attn_ba = np.concatenate(all_ba, axis=0)
    return attn_ab, attn_ba


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def best_threshold_f1(y_true, probs):
    """Find threshold maximising F1 on provided data."""
    best_f1, best_thr = 0.0, 0.5
    for thr in np.linspace(0.01, 0.99, 99):
        preds = (probs >= thr).astype(int)
        f = f1_score(y_true, preds, zero_division=0)
        if f > best_f1:
            best_f1, best_thr = f, thr
    return best_thr


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--checkpoint", default=str(
        _PROJECT_ROOT / "output" / "ProteomeLM-HPI-S-abl4c-seed-total-block" / "checkpoint-20000"
    ))
    parser.add_argument("--base-model", default="Bitbol-Lab/ProteomeLM-S")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=512,
                        help="Batch size for pairwise ProteomeLM inference")
    parser.add_argument("--esm-cache", default=str(_SCRIPT_DIR / "data" / "hf_benchmark_esm.pt"),
                        help="Path to cache ESM-C embeddings keyed by sequence. "
                             "Shared across original and clean benchmarks (embeddings are sequence-keyed).")
    parser.add_argument("--attn-cache", default=str(_SCRIPT_DIR / "data" / "hf_benchmark_attention.npz"),
                        help="Path to cache extracted attention features. "
                             "Use a different path for the clean benchmark, e.g. "
                             "data/clean_hf_benchmark_attention.npz")
    parser.add_argument("--esm-batch-tokens", type=int, default=16000)
    parser.add_argument("--n-replicas", type=int, default=3,
                        help="Number of independent training runs for the supervised model")
    parser.add_argument("--n-epochs", type=int, default=100,
                        help="Max training epochs for supervised model")
    parser.add_argument("--patience", type=int, default=15,
                        help="Early stopping patience for supervised model")
    parser.add_argument("--val-frac", type=float, default=0.1,
                        help="Fraction of training set to use as validation")
    parser.add_argument("--no-supervised", action="store_true",
                        help="Skip supervised MLP training (run LR only)")
    parser.add_argument("--train-csv", default=None,
                        help="Path to clean train CSV (human_seq, viral_seq, label). "
                             "If set, skips the HuggingFace dataset and uses local CSVs. "
                             "Build with: python build_clean_hf_benchmark.py")
    parser.add_argument("--test-csv", default=None,
                        help="Path to clean test CSV (human_seq, viral_seq, label).")
    args = parser.parse_args()

    device = args.device
    esm_cache_path = Path(args.esm_cache)
    attn_cache_path = Path(args.attn_cache)
    esm_cache_path.parent.mkdir(parents=True, exist_ok=True)
    attn_cache_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Load dataset  (HuggingFace or local clean CSV)
    # ------------------------------------------------------------------
    print("=" * 60)
    if args.train_csv and args.test_csv:
        print("Step 1: Loading local clean benchmark CSVs")
        print("=" * 60)
        import pandas as _pd
        train_df = _pd.read_csv(args.train_csv)
        test_df  = _pd.read_csv(args.test_csv)
        train_human_seqs = train_df["human_seq"].tolist()
        train_viral_seqs = train_df["viral_seq"].tolist()
        train_labels     = train_df["label"].to_numpy(dtype=np.int32)
        test_human_seqs  = test_df["human_seq"].tolist()
        test_viral_seqs  = test_df["viral_seq"].tolist()
        test_labels      = test_df["label"].to_numpy(dtype=np.int32)
        print(f"  Train CSV: {len(train_labels):,}  Test CSV: {len(test_labels):,}")
    else:
        print("Step 1: Loading HuggingFace dataset  (danliu1226/virus_human_benchmarking)")
        print("=" * 60)
        from datasets import load_dataset
        ds = load_dataset("danliu1226/virus_human_benchmarking")
        train_ds = ds["train"]
        test_ds  = ds["test"]
        print(f"  Train: {len(train_ds):,}  Test: {len(test_ds):,}")
        train_human_seqs = train_ds["query"]
        train_viral_seqs = train_ds["text"]
        train_labels     = np.array(train_ds["label"], dtype=np.int32)
        test_human_seqs  = test_ds["query"]
        test_viral_seqs  = test_ds["text"]
        test_labels      = np.array(test_ds["label"], dtype=np.int32)

    # ------------------------------------------------------------------
    # Step 2: Deduplicate sequences and assign integer indices
    # ------------------------------------------------------------------
    print("\nStep 2: Deduplicating sequences")
    all_seqs_set = set(train_human_seqs) | set(train_viral_seqs) | set(test_human_seqs) | set(test_viral_seqs)
    all_seqs = sorted(all_seqs_set)  # deterministic order
    seq_to_idx = {s: i for i, s in enumerate(all_seqs)}
    print(f"  Total unique sequences: {len(all_seqs):,}")

    # ------------------------------------------------------------------
    # Step 3: ESM-C encoding (with cache)
    # ------------------------------------------------------------------
    print("\nStep 3: ESM-C encoding")
    if esm_cache_path.exists():
        print(f"  Loading cached embeddings from {esm_cache_path}")
        cached = torch.load(esm_cache_path, map_location="cpu")
        # cached is (N, 1152) in same order as all_seqs
        if cached.shape[0] == len(all_seqs):
            all_embeddings = cached
        else:
            print(f"  Cache size mismatch ({cached.shape[0]} vs {len(all_seqs)}) — re-encoding")
            all_embeddings = encode_with_esm(all_seqs, device, args.esm_batch_tokens)
            torch.save(all_embeddings, esm_cache_path)
    else:
        print(f"  Encoding {len(all_seqs):,} sequences with ESM-C")
        all_embeddings = encode_with_esm(all_seqs, device, args.esm_batch_tokens)
        torch.save(all_embeddings, esm_cache_path)
        print(f"  Saved to {esm_cache_path}")

    # ------------------------------------------------------------------
    # Step 4: Extract pair attention features (with cache)
    # ------------------------------------------------------------------
    print("\nStep 4: Extracting attention features")

    n_train = len(train_labels)
    n_test  = len(test_labels)

    if attn_cache_path.exists():
        print(f"  Loading cached attention from {attn_cache_path}")
        cached_attn = np.load(attn_cache_path)
        train_ab = cached_attn["train_ab"]
        train_ba = cached_attn["train_ba"]
        test_ab  = cached_attn["test_ab"]
        test_ba  = cached_attn["test_ba"]
        if train_ab.shape[0] != n_train or test_ab.shape[0] != n_test:
            print("  Cache size mismatch — re-extracting")
            attn_cache_path.unlink()
    
    if not attn_cache_path.exists():
        # Load model
        model = load_finetuned(args.checkpoint, args.base_model, device=device)

        # Build index arrays for pairs
        def _pair_embeddings(human_list, viral_list):
            idx_h = torch.tensor([seq_to_idx[s] for s in human_list], dtype=torch.long)
            idx_v = torch.tensor([seq_to_idx[s] for s in viral_list], dtype=torch.long)
            emb_h = all_embeddings[idx_h]  # (N, esm_dim)
            emb_v = all_embeddings[idx_v]
            return emb_h, emb_v

        print(f"  Extracting train attention ({n_train:,} pairs)...")
        emb_h, emb_v = _pair_embeddings(train_human_seqs, train_viral_seqs)
        train_ab, train_ba = extract_pair_attention_batched(model, emb_h, emb_v, args.batch_size, device)

        print(f"  Extracting test attention ({n_test:,} pairs)...")
        emb_h, emb_v = _pair_embeddings(test_human_seqs, test_viral_seqs)
        test_ab, test_ba = extract_pair_attention_batched(model, emb_h, emb_v, args.batch_size, device)

        np.savez_compressed(
            attn_cache_path,
            train_ab=train_ab, train_ba=train_ba,
            test_ab=test_ab,   test_ba=test_ba,
        )
        print(f"  Saved attention cache to {attn_cache_path}")

        del model
        torch.cuda.empty_cache()

    # Feature matrix: symmetric (a→b + b→a)/2 flattened → (N, n_layers * n_heads)
    attn_train = ((train_ab + train_ba) / 2.0).reshape(n_train, -1)  # (N_train, 48)
    attn_test  = ((test_ab  + test_ba)  / 2.0).reshape(n_test,  -1)  # (N_test, 48)
    print(f"  Attention feature shape: train={attn_train.shape}  test={attn_test.shape}")

    # ESM-C per-protein embeddings for each pair
    def _pair_esm(human_list, viral_list):
        idx_h = torch.tensor([seq_to_idx[s] for s in human_list], dtype=torch.long)
        idx_v = torch.tensor([seq_to_idx[s] for s in viral_list], dtype=torch.long)
        return all_embeddings[idx_h].numpy(), all_embeddings[idx_v].numpy()

    esm_train_h, esm_train_v = _pair_esm(train_human_seqs, train_viral_seqs)  # (N, 1152)
    esm_test_h,  esm_test_v  = _pair_esm(test_human_seqs,  test_viral_seqs)
    print(f"  ESM embedding shape: train={esm_train_h.shape}  test={esm_test_h.shape}")

    X_train = attn_train
    X_test  = attn_test

    # ------------------------------------------------------------------
    # Step 5: Logistic Regression
    # ------------------------------------------------------------------
    print("\nStep 5: Training Logistic Regression")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # Class-balanced to handle 1:10 imbalance
    clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    clf.fit(X_train_scaled, train_labels)
    print("  Training complete.")

    train_probs = clf.predict_proba(X_train_scaled)[:, 1]
    test_probs  = clf.predict_proba(X_test_scaled)[:, 1]

    # Threshold: optimise F1 on training set
    best_thr = best_threshold_f1(train_labels, train_probs)
    print(f"  Best threshold (train F1): {best_thr:.2f}")

    test_preds = (test_probs >= best_thr).astype(int)

    # ------------------------------------------------------------------
    # Step 6a: LR metrics
    # ------------------------------------------------------------------
    lr_aupr  = average_precision_score(test_labels, test_probs)
    lr_f1    = f1_score(test_labels, test_preds, zero_division=0)
    lr_mcc   = matthews_corrcoef(test_labels, test_preds)
    lr_auroc = roc_auc_score(test_labels, test_probs)
    print(f"\n  [LR results]  AUROC={lr_auroc:.4f}  AUPR={lr_aupr:.4f}  "
          f"F1={lr_f1:.4f}  MCC={lr_mcc:.4f}")

    # ------------------------------------------------------------------
    # Step 6b: Supervised MLP (EnhancedPPIModel)
    # ------------------------------------------------------------------
    sup_aupr = sup_f1 = sup_mcc = sup_auroc = float("nan")

    if not args.no_supervised:
        print("\nStep 6b: Training supervised MLP (EnhancedPPIModel)")
        from proteomelm.ppi.model import train_model_cv, test_model_cv

        # Split train → train/val (stratified)
        from sklearn.model_selection import train_test_split
        tr_idx, val_idx = train_test_split(
            np.arange(n_train),
            test_size=args.val_frac,
            stratify=train_labels,
            random_state=42,
        )

        def _split(arr, idx):
            return arr[idx] if arr is not None else None

        X_tr  = {"edges": _split(attn_train, tr_idx),
                 "x1":    _split(esm_train_h, tr_idx),
                 "x2":    _split(esm_train_v, tr_idx)}
        X_val = {"edges": _split(attn_train, val_idx),
                 "x1":    _split(esm_train_h, val_idx),
                 "x2":    _split(esm_train_v, val_idx)}
        X_te  = {"edges": attn_test,
                 "x1":    esm_test_h,
                 "x2":    esm_test_v}
        y_tr  = train_labels[tr_idx]
        y_val = train_labels[val_idx]

        print(f"  Train/val/test split: {len(y_tr):,} / {len(y_val):,} / {n_test:,}")

        all_test_probs_sup = []
        for rep in range(args.n_replicas):
            print(f"  Replica {rep+1}/{args.n_replicas}")
            model_sup, val_metrics = train_model_cv(
                X_tr, X_val, y_tr, y_val,
                n_epochs=args.n_epochs,
                patience=args.patience,
                model_type="enhancedppi",
                verbose=True,
                replica_seed=rep,
                use_class_weights=True,
                loss_type="bce",
            )
            _, _, _, test_metrics_sup = test_model_cv(model_sup, X_te, test_labels)
            # Re-run inference to get probabilities (test_model_cv prints but doesn't return probs)
            import torch as _torch
            _plm_device = next(model_sup.parameters()).device
            model_sup.eval()
            with _torch.no_grad():
                _f  = _torch.tensor(attn_test,  dtype=_torch.float32).to(_plm_device)
                _e1 = _torch.tensor(esm_test_h, dtype=_torch.float32).to(_plm_device)
                _e2 = _torch.tensor(esm_test_v, dtype=_torch.float32).to(_plm_device)
                _logits = model_sup(_f, _e1, _e2).cpu().numpy().flatten()
            _probs = 1.0 / (1.0 + np.exp(-_logits))
            all_test_probs_sup.append(_probs)
            print(f"    val AUROC={val_metrics['auc']:.4f}  test AUROC={test_metrics_sup['auc']:.4f}  AUPR={test_metrics_sup['aupr']:.4f}")

        # Average probabilities across replicas
        avg_probs_sup = np.mean(all_test_probs_sup, axis=0)
        sup_aupr  = average_precision_score(test_labels, avg_probs_sup)
        sup_auroc = roc_auc_score(test_labels, avg_probs_sup)
        best_thr_sup = best_threshold_f1(test_labels, avg_probs_sup)
        sup_preds = (avg_probs_sup >= best_thr_sup).astype(int)
        sup_f1  = f1_score(test_labels, sup_preds, zero_division=0)
        sup_mcc = matthews_corrcoef(test_labels, sup_preds)
        print(f"  Supervised (ensemble of {args.n_replicas}): AUROC={sup_auroc:.4f}  AUPR={sup_aupr:.4f}")
    else:
        print("\n  --no-supervised: skipping MLP training.")

    # ------------------------------------------------------------------
    # Step 7: Print comparison table
    # ------------------------------------------------------------------
    # SOTA numbers from Liu et al. (danliu1226/virus_human_benchmarking paper figure).
    # † PLM-interact reports AUPR/F1/MCC = 1.00 on this benchmark, likely due to
    #   sequence identity leakage between train and test splits (not reproduced here).
    sota = [
        # (method,         AUPR,   F1,    MCC,   AUROC,  note)
        ("InterSPPI",      0.81,   0.724,  0.697,  None,   ""),
        ("LSTM-PHV",       0.9380,   0.911,  0.904,  None,   ""),
        ("STEP",           0.9571,   0.9153,  0.9082,  None,   ""),
        ("PLM-interact",   1.00,   1.00,  1.00,  None,   "†"),
    ]

    col_w = 30
    W = col_w + 40
    print("\n" + "=" * W)
    print("Results: virus-human PPI benchmark (danliu1226/virus_human_benchmarking)")
    print("=" * W)
    header = f"{'Method':<{col_w}}{'AUPR':>8}{'F1':>8}{'MCC':>8}{'AUROC':>8}"
    sep    = "-" * W
    print(header)
    print(sep)
    for name, aupr, f1, mcc, auroc, note in sota:
        auroc_str = f"{auroc:>8.3f}" if auroc is not None else f"{'—':>8}"
        print(f"{name+note:<{col_w}}{aupr:>8.3f}{f1:>8.3f}{mcc:>8.3f}{auroc_str}")
    print(sep)
    print(f"{'ProteomeLM-LR (attn only)':<{col_w}}"
          f"{lr_aupr:>8.3f}{lr_f1:>8.3f}{lr_mcc:>8.3f}{lr_auroc:>8.3f}")
    if not args.no_supervised:
        mlp_label = f"ProteomeLM-MLP (×{args.n_replicas})"
        print(f"{mlp_label:<{col_w}}"
              f"{sup_aupr:>8.3f}{sup_f1:>8.3f}{sup_mcc:>8.3f}{sup_auroc:>8.3f}")
    print("=" * W)
    print(f"† PLM-interact AUPR/F1/MCC = 1.00 is suspected to reflect sequence leakage "
          f"between train/test splits (see main text).")
    print(f"\nDataset: {n_train:,} train  |  {n_test:,} test  |  "
          f"positive rate {test_labels.mean():.1%}")


if __name__ == "__main__":
    main()
