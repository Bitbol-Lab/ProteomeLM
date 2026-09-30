#!/usr/bin/env python3
"""
Extract Attention Patterns for HPI Evaluation
==============================================
Run this script to (re-)compute attention .npz files for any checkpoint.
Outputs go to data/attention/ (used by 1_base_vs_finetuned.py, 2_ablations.py
and 6_head_auc_maps.py).

Usage:
    python 0_extract_attention.py \
        --pathogen sars_cov2_zhou2022 \
        --checkpoint /path/to/output/ProteomeLM-HPI/checkpoint-XXXX \
        --also-base

    # Run for the 10 benchmark datasets (utils.DATASETS):
    for p in sars_cov2_zhou2022 hiv1_jager2012 influenza_a ebv hpv hsv1 \
              yersinia salmonella chlamydia tuberculosis; do
        python 0_extract_attention.py --pathogen $p --checkpoint <ckpt> --also-base
    done

Prerequisites:
    - Benchmark data in data/benchmarks/ (not rebuildable from this repo;
      download the benchmark data, see README.md)
    - ESM-C 600M model accessible (loaded via esm package)
    - proteomelm package installed (pip install -e /path/to/ProteomeLM)
"""

import sys
from pathlib import Path

# Add project root (../../) so proteomelm package can be imported, and this
# directory so the shared utils module can be
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_SCRIPT_DIR))

import argparse
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

# Shared with 9_gold_benchmark.py; re-exported here (tests/ load this module and call encode_split_cached / add_gene_name_aliases).
from utils import (
    add_gene_name_aliases,
    encode_split_cached,
    load_base,
    load_finetuned,
    run_inference,
)


# ---------------------------------------------------------------------------
# Protein index utilities
# ---------------------------------------------------------------------------

def create_protein_mapping(fasta_file: Path) -> Dict[str, int]:
    """Map protein IDs (and gene names) to FASTA position.

    Do not use this to index the ESM/attention tensors of a combined HPI
    proteome: ``encode_split_cached`` orders them host-first, which is not the
    FASTA order when the pathogen comes first. Use ``add_gene_name_aliases``.
    """
    mapping = {}
    idx = 0
    with open(fasta_file) as f:
        for line in f:
            if line.startswith(">"):
                header = line[1:].strip()
                parts = header.split("|")
                uniprot_id = parts[1] if len(parts) > 1 else header.split()[0]
                mapping[uniprot_id] = idx
                if "GN=" in header:
                    gene = header.split("GN=")[1].split()[0]
                    mapping[gene] = idx
                idx += 1
    return mapping


# ---------------------------------------------------------------------------
# Pair utilities
# ---------------------------------------------------------------------------

def generate_negatives(
    host_proteins: List[str],
    pathogen_proteins: List[str],
    positives_set: set,
    n: int,
    mode: str = "cross",
    seed: int = 42,
) -> List[Tuple[str, str]]:
    """Random negative pairs: cross-species or intra-host."""
    rng = np.random.RandomState(seed)
    negatives = []
    for _ in range(n * 10):
        if mode == "cross":
            a = rng.choice(host_proteins)
            b = rng.choice(pathogen_proteins)
        else:
            a, b = tuple(sorted(rng.choice(host_proteins, 2, replace=False)))
        pair = tuple(sorted([a, b]))
        if pair not in positives_set:
            negatives.append(pair)
            if len(negatives) >= n:
                break
    return negatives


def extract_attention(
    pairs: List[Tuple[str, str]],
    protein_to_idx: Dict[str, int],
    attentions: List[np.ndarray],
    max_pairs: int = 10000,
) -> Optional[Dict]:
    """Extract (n_pairs, n_layers, n_heads) attention arrays for given pairs."""
    valid_pairs, idx_a, idx_b = [], [], []
    for a, b in pairs:
        if a in protein_to_idx and b in protein_to_idx:
            valid_pairs.append((a, b))
            idx_a.append(protein_to_idx[a])
            idx_b.append(protein_to_idx[b])
            if len(valid_pairs) >= max_pairs:
                break
    if not valid_pairs:
        return None
    idx_a, idx_b = np.array(idx_a), np.array(idx_b)
    attn_ab = np.stack([attn[:, idx_a, idx_b] for attn in attentions]).transpose(2, 0, 1)
    attn_ba = np.stack([attn[:, idx_b, idx_a] for attn in attentions]).transpose(2, 0, 1)
    return {"pair_ids": valid_pairs, "attention_a_to_b": attn_ab, "attention_b_to_a": attn_ba}


# ---------------------------------------------------------------------------
# Targeted inference (large proteomes)
# ---------------------------------------------------------------------------

def run_inference_targeted(
    model,
    esm_data: Dict,
    idx_a: np.ndarray,
    idx_b: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Memory-efficient forward pass using forward hooks.

    Instead of storing the full (n_layers, n_heads, seq, seq) attention tensor,
    hooks intercept each layer's attention weights, extract only the needed
    (idx_a, idx_b) entries, then replace the full tensor with None so Python's
    GC can free it immediately.  Peak memory is one layer's attention matrix
    (~seq²×bfloat16) which is freed after each layer, rather than the sum of
    all layers.

    Returns
    -------
    attn_ab : np.ndarray  shape (n_pairs, n_layers, n_heads)  weights[i→j]
    attn_ba : np.ndarray  shape (n_pairs, n_layers, n_heads)  weights[j→i]
    """
    seq_len = esm_data["inputs_embeds"].shape[0]

    # Locate MultiHeadSelfAttention layers in forward-pass order.
    # Identify by structural attributes (q_lin + n_heads) rather than class name,
    # which is robust to PeftModel wrapping and dynamic class generation by PEFT.
    # ProteomeLMForMaskedLM has TWO Transformer stacks (one inherited from DistilBERT,
    # one custom); we want only the ones under 'transformer.transformer.layer.*'
    # i.e. the stack that the model's forward() actually calls.
    import re as _re
    def _layer_sort_key(pair):
        name, _ = pair
        m = _re.search(r"\.layer\.(\d+)\.", name)
        return int(m.group(1)) if m else 9999
    named_attn = [(name, mod) for name, mod in model.named_modules()
                  if hasattr(mod, "q_lin") and hasattr(mod, "n_heads") and "distilbert" not in name]
    named_attn.sort(key=_layer_sort_key)
    attention_modules = [mod for _, mod in named_attn]
    n_layers = len(attention_modules)
    if n_layers == 0:
        raise RuntimeError("Could not find any MultiHeadSelfAttention modules in model.")

    n_heads = attention_modules[0].n_heads
    n_pairs = len(idx_a)
    attn_ab = np.zeros((n_pairs, n_layers, n_heads), dtype=np.float32)
    attn_ba = np.zeros((n_pairs, n_layers, n_heads), dtype=np.float32)

    # Convert indices to tensors once
    t_ia = torch.from_numpy(idx_a).long()
    t_ib = torch.from_numpy(idx_b).long()

    hooks = []
    for layer_idx, attn_mod in enumerate(attention_modules):
        def _make_hook(lidx):
            def _hook(module, inp, output):
                # output = (context, weights) when output_attentions=True
                # weights shape: (batch=1, n_heads, seq, seq)
                if not isinstance(output, (tuple, list)):
                    return output
                weights = output[1] if len(output) >= 2 else None
                if weights is None or not isinstance(weights, torch.Tensor):
                    return output
                w = weights[0].float()  # (n_heads, seq, seq)
                attn_ab[:, lidx, :] = w[:, t_ia, t_ib].T.cpu().numpy()
                attn_ba[:, lidx, :] = w[:, t_ib, t_ia].T.cpu().numpy()
                # Drop the full weight tensor; only the pair extracts are kept
                return (output[0], None) + output[2:]
            return _hook
        hooks.append(attn_mod.register_forward_hook(_make_hook(layer_idx)))

    mem_peak_gb = n_heads * seq_len * seq_len * 2 / 1e9  # one layer, bfloat16
    print(
        f"  Running targeted inference (seq_len={seq_len}, {n_pairs} pairs, "
        f"{n_layers}L×{n_heads}H, peak≈{mem_peak_gb:.2f} GB/layer)..."
    )
    try:
        with torch.no_grad():
            model(
                inputs_embeds=esm_data["inputs_embeds"][None].to(dtype=torch.bfloat16, device="cpu"),
                group_embeds=esm_data["group_embeds"][None].to(dtype=torch.bfloat16, device="cpu"),
                output_attentions=True,
            )
    finally:
        for h in hooks:
            h.remove()

    return attn_ab, attn_ba


def _collect_pairs(
    pairs: List[Tuple[str, str]],
    protein_to_idx: Dict[str, int],
    max_pairs: int,
) -> Tuple[List[Tuple[str, str]], np.ndarray, np.ndarray]:
    """Filter pairs to those present in protein_to_idx and return indices."""
    valid, ia, ib = [], [], []
    for a, b in pairs:
        if a in protein_to_idx and b in protein_to_idx:
            valid.append((a, b))
            ia.append(protein_to_idx[a])
            ib.append(protein_to_idx[b])
            if len(valid) >= max_pairs:
                break
    return valid, np.array(ia, dtype=np.int64), np.array(ib, dtype=np.int64)


# ---------------------------------------------------------------------------
# Proteome subsampling (for large proteomes that exceed memory)
# ---------------------------------------------------------------------------

def _subsample_proteome(
    esm_data: Dict,
    protein_to_idx: Dict[str, int],
    host_proteins: List[str],
    pathogen_proteins: List[str],
    must_keep_ids: set,
    max_size: int,
    seed: int = 42,
) -> Tuple[Dict, Dict[str, int], List[str], List[str]]:
    """Subsample combined proteome to at most *max_size* proteins.

    Priority order:
      1. All pathogen proteins (kept in full).
      2. All proteins that appear in positive pairs (must_keep_ids).
      3. Random host proteins to fill the remaining budget.

    Returns updated (esm_data, protein_to_idx, sub_host_proteins, sub_path_proteins).
    """
    n_total = len(host_proteins) + len(pathogen_proteins)
    if n_total <= max_size:
        return esm_data, protein_to_idx, host_proteins, pathogen_proteins

    rng = np.random.RandomState(seed)

    selected_set: set = set(pathogen_proteins) | must_keep_ids
    n_remaining = max(0, max_size - len(selected_set))

    host_optional = [p for p in host_proteins if p not in selected_set]
    if n_remaining > 0 and host_optional:
        n_sample = min(n_remaining, len(host_optional))
        sampled = rng.choice(host_optional, n_sample, replace=False).tolist()
        selected_set |= set(sampled)

    sub_host = [p for p in host_proteins if p in selected_set]
    sub_path = [p for p in pathogen_proteins if p in selected_set]
    all_ordered = sub_host + sub_path

    # Build reverse map: old_index -> new_index (handles gene-name aliases too)
    old_to_new: Dict[int, int] = {}
    for new_i, pid in enumerate(all_ordered):
        if pid in protein_to_idx:
            old_to_new[protein_to_idx[pid]] = new_i

    new_idx_map: Dict[str, int] = {}
    for pid, old_idx in protein_to_idx.items():
        if old_idx in old_to_new:
            new_idx_map[pid] = old_to_new[old_idx]

    old_tensor = esm_data["inputs_embeds"]
    old_indices = [protein_to_idx[p] for p in all_ordered if p in protein_to_idx]
    new_tensor = old_tensor[old_indices]

    new_esm_data = {
        "inputs_embeds": new_tensor,
        "group_embeds": new_tensor,
        "group_labels": all_ordered,
    }

    print(
        f"  Subsampled proteome: {n_total} -> {len(all_ordered)} proteins "
        f"(host: {len(sub_host)}, pathogen: {len(sub_path)}, cap={max_size})"
    )
    return new_esm_data, new_idx_map, sub_host, sub_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract attention patterns for one pathogen dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pathogen", required=True,
                        help="Pathogen key (e.g. sars_cov2_zhou2022, hiv1_jager2012, influenza_a)")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to fine-tuned LoRA checkpoint directory")
    parser.add_argument("--base-model", default="Bitbol-Lab/ProteomeLM-S",
                        help="Base HuggingFace model name or local path")
    parser.add_argument("--benchmark-dir",
                        default=str(_SCRIPT_DIR / "data" / "benchmarks"),
                        help="Directory with raw/ and processed/ subdirs")
    parser.add_argument("--output-dir",
                        default=str(_SCRIPT_DIR / "data" / "attention"),
                        help="Where to write output .npz files")
    parser.add_argument("--esm-cache-dir", default=None,
                        help="Cache dir for host_*_esm.pt / pathogen_*_esm.pt (default: output-dir)")
    parser.add_argument("--device", default="cuda:0",
                        help="Device for ESM-C encoding (attention always on CPU)")
    parser.add_argument("--max-pairs", type=int, default=10000)
    parser.add_argument("--n-negative", type=int, default=5000)
    parser.add_argument(
        "--max-proteome-size", type=int, default=None,
        help="(Fallback) Cap the combined proteome to this many proteins by randomly "
             "subsampling host proteins. Trades context completeness for speed. "
             "Prefer --targeted-inference-threshold for lossless memory reduction.",
    )
    parser.add_argument(
        "--targeted-inference-threshold", type=int, default=10000,
        help="Use hook-based targeted inference (no subsampling, lossless) when the "
             "combined proteome exceeds this many proteins. Peak memory ≈ one layer's "
             "attention matrix (~seq²×bfloat16) instead of all layers combined. "
             "Set to 0 to always use targeted inference, or a large number to disable.",
    )
    parser.add_argument("--also-base", action="store_true",
                        help="Also extract from base model (no LoRA); adds _base suffix")
    args = parser.parse_args()

    benchmark_dir = Path(args.benchmark_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    esm_cache_dir = Path(args.esm_cache_dir) if args.esm_cache_dir else output_dir
    esm_cache_dir.mkdir(parents=True, exist_ok=True)

    combined_fasta = benchmark_dir / "raw" / args.pathogen / "combined_proteome.fasta"
    pairs_dir = benchmark_dir / "processed" / args.pathogen

    print("=" * 60)
    print(f"Extracting Attention: {args.pathogen.upper()}")
    print("=" * 60)

    if not combined_fasta.exists():
        raise FileNotFoundError(
            f"Combined proteome not found: {combined_fasta}\n"
            "Download the benchmark data first (see README.md)."
        )

    # Step 1: ESM embeddings (split-cached: host once, pathogen per species)
    esm_data, protein_to_idx = encode_split_cached(combined_fasta, esm_cache_dir, args.device)

    # Step 2: Protein index is already returned by encode_split_cached (host-first
    #         order, matching the embedding rows); only add gene-name aliases onto
    #         it. Rebuilding the map in FASTA order mis-indexed the pathogen-first
    #         ebv/hsv1 FASTAs.
    protein_to_idx = add_gene_name_aliases(combined_fasta, protein_to_idx)
    print(f"\n{len(set(protein_to_idx.values()))} proteins in combined proteome")

    # Step 3: Host / pathogen split
    host_proteins, pathogen_proteins = [], []
    with open(combined_fasta) as f:
        for line in f:
            if line.startswith(">"):
                header = line[1:].strip()
                parts = header.split("|")
                uid = parts[1] if len(parts) > 1 else header.split()[0]
                tag = parts[2] if len(parts) > 2 else ""
                if "HOST" in tag:
                    host_proteins.append(uid)
                elif "PATHOGEN" in tag:
                    pathogen_proteins.append(uid)
    print(f"  Host: {len(host_proteins)}  Pathogen: {len(pathogen_proteins)}")

    # Step 4: Load positive pairs to determine must-keep proteins
    all_positive_pairs: set = set()
    must_keep_ids: set = set()
    preloaded_pairs: Dict[str, List[Tuple[str, str]]] = {}
    for bm_type in ["hpi", "intra_host"]:
        path = pairs_dir / f"{bm_type}_pairs.tsv"
        if not path.exists():
            continue
        df = pd.read_csv(path, sep="\t")
        pairs = list(zip(df["protein_a"], df["protein_b"]))
        preloaded_pairs[bm_type] = pairs
        all_positive_pairs.update(tuple(sorted(p)) for p in pairs)
        must_keep_ids.update(p for pair in pairs for p in pair)
        print(f"  {bm_type}: {len(pairs)} pairs")

    # Step 5: Optionally subsample proteome before inference (fallback/legacy)
    sub_esm, sub_idx = esm_data, protein_to_idx
    if args.max_proteome_size is not None:
        sub_esm, sub_idx, host_proteins, pathogen_proteins = _subsample_proteome(
            esm_data, protein_to_idx,
            host_proteins, pathogen_proteins,
            must_keep_ids,
            args.max_proteome_size,
        )

    neg_cross = generate_negatives(host_proteins, pathogen_proteins, all_positive_pairs,
                                   args.n_negative, mode="cross")
    neg_intra = generate_negatives(host_proteins, pathogen_proteins, all_positive_pairs,
                                   args.n_negative, mode="intra_host")

    # Step 6: Decide inference mode
    seq_len = sub_esm["inputs_embeds"].shape[0]
    use_targeted = seq_len >= args.targeted_inference_threshold

    def _save(result, name):
        if result:
            output_file = output_dir / f"{args.pathogen}_{name}_attention.npz"
            np.savez_compressed(
                output_file,
                pair_ids=np.array(result["pair_ids"], dtype=object),
                attention_a_to_b=result["attention_a_to_b"],
                attention_b_to_a=result["attention_b_to_a"],
            )
            print(f"  ✓ {output_file.name}  ({len(result['pair_ids'])} pairs)")

    # Collect all pair groups
    pair_groups: Dict[str, List[Tuple[str, str]]] = {}
    for bm_type, pairs in preloaded_pairs.items():
        pair_groups[bm_type] = pairs
    pair_groups["random_cross"] = neg_cross
    pair_groups["random_intra"] = neg_intra

    def _run_and_save(model, suffix=""):
        if use_targeted:
            # Build union of all (valid) pairs to extract in a single forward pass
            all_valid: Dict[str, Tuple[List, np.ndarray, np.ndarray]] = {}
            union_pairs: List[Tuple[str, str]] = []
            seen: set = set()
            for name, pairs in pair_groups.items():
                vp, ia, ib = _collect_pairs(pairs, sub_idx, args.max_pairs)
                all_valid[name] = (vp, ia, ib)
                for p in zip(vp, ia.tolist(), ib.tolist()):
                    key = (p[1], p[2])
                    if key not in seen:
                        seen.add(key)
                        union_pairs.append((p[1], p[2]))  # store indices directly

            # Single forward pass
            ab_all, ba_all, lookup = None, None, {}
            if union_pairs:
                u_ia = np.array([p[0] for p in union_pairs], dtype=np.int64)
                u_ib = np.array([p[1] for p in union_pairs], dtype=np.int64)
                ab_all, ba_all = run_inference_targeted(model, sub_esm, u_ia, u_ib)
                # Build lookup: (ia, ib) -> row in ab_all / ba_all
                lookup = {(int(u_ia[k]), int(u_ib[k])): k for k in range(len(u_ia))}
                print(f"  Extracted {len(union_pairs)} unique pairs across all groups")
            else:
                print(f"  WARNING: no pairs to extract")

            for name, (vp, ia, ib) in all_valid.items():
                if not vp:
                    continue
                if ab_all is None:
                    print(f"  SKIP {name}: no union pairs")
                    continue
                rows = [lookup[(int(ia[k]), int(ib[k]))] for k in range(len(vp))]
                result = {
                    "pair_ids": vp,
                    "attention_a_to_b": ab_all[rows],
                    "attention_b_to_a": ba_all[rows],
                }
                _save(result, name if not suffix else f"{name}{suffix}")
        else:
            attentions = run_inference(model, sub_esm)
            for name, pairs in pair_groups.items():
                out_name = name if not suffix else f"{name}{suffix}"
                _save(extract_attention(pairs, sub_idx, attentions, args.max_pairs), out_name)

    print(f"\n--- Fine-tuned model {'(targeted)' if use_targeted else ''} ---")
    try:
        model_ft = load_finetuned(args.checkpoint, args.base_model)
        _run_and_save(model_ft)
        del model_ft
        print(f"  ✓ Fine-tuned model extraction complete")
    except Exception as e:
        print(f"  ✗ Fine-tuned model extraction failed: {e}")
        import traceback
        traceback.print_exc()
        raise

    if args.also_base:
        print(f"\n--- Base model (no LoRA) {'(targeted)' if use_targeted else ''} ---")
        try:
            model_base = load_base(args.base_model)
            _run_and_save(model_base, suffix="_base")
            del model_base
            print(f"  ✓ Base model extraction complete")
        except Exception as e:
            print(f"  ✗ Base model extraction failed: {e}")
            import traceback
            traceback.print_exc()

    torch.cuda.empty_cache()
    print(f"\n{'='*60}")
    print(f"Done. Output: {output_dir}")


if __name__ == "__main__":
    main()
