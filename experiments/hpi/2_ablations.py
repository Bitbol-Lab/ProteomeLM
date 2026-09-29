"""Part 2 — Ablation study: comparing masking strategies and block attention.

For every ablation model the script computes a logistic-regression AUROC
(all attention heads as features) per dataset (HPI vs. random cross-species),
with sequence-identity-grouped CV (MMseqs2 clusters on pathogen and host
proteins, see utils.lr_auroc_grouped_identity_cv), and reports the
virus/bacteria-balanced mean. The base model (no LoRA) is included for
reference, from the _base attention files in data/ablation_attention/base/.

Produces two figures:  figures/ablations.{pdf,png,svg}
                       figures/ablations_per_dataset.{pdf,png,svg}

Data required (written by 0_extract_attention.py):
    data/ablation_attention/<model_key>/checkpoint-<step>/   (one subdir per model)
    data/ablation_attention/base/                            (_base files)

Usage:
    python 2_ablations.py
    python 2_ablations.py --ablation-dir /path/to/ablation_attention --checkpoint 20000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    DATASETS, DATASET_GROUP, ABLATION_MODELS,
    load_attention_with_pairs,
    lr_auroc_grouped_identity_cv,
    style_ax,
)

ROOT           = Path(__file__).resolve().parent
ABL_DIR        = ROOT / "data" / "ablation_attention"
FLAT_ATTN_DIR  = ROOT / "data" / "ablation_attention" / "base"  # holds _base files
FIGS_DIR       = ROOT / "figures"

# ---------------------------------------------------------------------------
# Human-readable labels + 4-category colors
# ---------------------------------------------------------------------------
MODEL_READABLE = {
    "proteomelm_hpi_bigdataset":                "Full Pathogen masking + block + extended data",
    "proteomelm_hpi_s_abl1_seed_sym_noblock":   "Mask 50% host / 50% pathogen + no block",
    "proteomelm_hpi_s_abl2_seed_asym_noblock":  "Host-biased masking + no block",
    "proteomelm_hpi_s_abl3_seed_inv_noblock":   "Pathogen-biased masking + no block",
    "proteomelm_hpi_s_abl4_seed_total_noblock": "Full Pathogen masking + no block",
    "proteomelm_hpi_s_abl5_seed_sym_block":     "Mask 50% host / 50% pathogen + block",
    "proteomelm_hpi_s_abl6_seed_asym_block":    "Host-biased masking + block",
    "proteomelm_hpi_s_abl7_seed_inv_block":     "Pathogen-biased masking + block",
    "proteomelm_hpi_s_abl8_seed_total_block":   "Full Pathogen masking + block",
}

CATEGORY_COLORS = {
    "base": "#9E9E9E",              # gray
    "no_block": "#4C78A8",          # blue
    "block": "#54A24B",             # green
    "block_extended": "#F58518",    # orange
}

CATEGORY_LABELS = {
    "base": "Base model",
    "no_block": "No block",
    "block": "Block",
    "block_extended": "Block + extended data",
}


def model_category(model_key: str) -> str:
    if model_key == "proteomelm_hpi_bigdataset":
        return "block_extended"
    if "noblock" in model_key:
        return "no_block"
    return "block"


def balanced_group_mean(ds_scores: Dict[str, float]) -> tuple[float, float, float]:
    """Return (virus_mean, bacteria_mean, balanced_mean).

    balanced_mean = (virus_mean + bacteria_mean) / 2 when both groups are present.
    If one group is missing, falls back to the available group mean.
    """
    virus_vals = [v for k, v in ds_scores.items() if DATASET_GROUP.get(k) == "Virus" and np.isfinite(v)]
    bacteria_vals = [v for k, v in ds_scores.items() if DATASET_GROUP.get(k) == "Bacteria" and np.isfinite(v)]

    virus_mean = float(np.mean(virus_vals)) if virus_vals else np.nan
    bacteria_mean = float(np.mean(bacteria_vals)) if bacteria_vals else np.nan
    if np.isfinite(virus_mean) and np.isfinite(bacteria_mean):
        balanced = 0.5 * (virus_mean + bacteria_mean)
    elif np.isfinite(virus_mean):
        balanced = virus_mean
    elif np.isfinite(bacteria_mean):
        balanced = bacteria_mean
    else:
        balanced = np.nan
    return virus_mean, bacteria_mean, float(balanced)


def log_cv_progress(model_label: str, dataset_key: str, phase: str) -> None:
    print(f"    [{phase}] {model_label} :: {dataset_key}")


# ---------------------------------------------------------------------------
# Compute per-model mean AUROC
# ---------------------------------------------------------------------------

def resolve_attn_dir(model_dir: Path) -> Path:
    """Return the checkpoint subdir with the highest step, or model_dir itself."""
    ckpts = sorted(model_dir.glob("checkpoint-*"),
                   key=lambda p: int(p.name.split("-")[1]))
    return ckpts[-1] if ckpts else model_dir


def normalize_checkpoint_name(checkpoint: str | None) -> str | None:
    """Normalize checkpoint values from CLI (e.g., '5000' -> 'checkpoint-5000')."""
    if checkpoint is None:
        return None
    ckpt = str(checkpoint).strip()
    if not ckpt or ckpt.lower() in {"latest", "none"}:
        return None
    if ckpt.startswith("checkpoint-"):
        return ckpt
    if ckpt.isdigit():
        return f"checkpoint-{ckpt}"
    return ckpt


def resolve_attn_dir_from_spec(model_dir: Path, checkpoint: str | None = None) -> Path | None:
    """Resolve attention dir from an optional explicit checkpoint name."""
    checkpoint = normalize_checkpoint_name(checkpoint)
    if checkpoint is None:
        return resolve_attn_dir(model_dir)
    ckpt_dir = model_dir / checkpoint
    if ckpt_dir.exists():
        return ckpt_dir
    return None


def grouped_cv_scores(
    attn_dir: Path,
    suffix: str,
    active_datasets: List[Tuple[str, str]],
    progress_label: str,
    n_folds: int,
    min_seq_id: float,
) -> Dict[str, Tuple[float, float]]:
    """Grouped-identity LR AUROC for every dataset whose hpi + random_cross files exist.

    Returns {ds_key: (mean, fold_std)} in *active_datasets* order; NaN scores are kept.
    """
    out: Dict[str, Tuple[float, float]] = {}
    for ds_key, _ in active_datasets:
        pos = load_attention_with_pairs(attn_dir, ds_key, "hpi", suffix=suffix)
        neg = load_attention_with_pairs(attn_dir, ds_key, "random_cross", suffix=suffix)
        if pos is None or neg is None:
            continue
        log_cv_progress(progress_label, ds_key, "running")
        m, s = lr_auroc_grouped_identity_cv(
            pos[0], neg[0], pos[1], neg[1],
            dataset_key=ds_key,
            n_folds=n_folds,
            return_folds=True,
            min_seq_id=min_seq_id,
        )
        if np.isfinite(m):
            log_cv_progress(progress_label, ds_key, f"done m={m:.3f}")
        else:
            log_cv_progress(progress_label, ds_key, "done m=nan")
        out[ds_key] = (m, s)
    return out


def summarize_model(scores: Dict[str, Tuple[float, float]]) -> Dict | None:
    """Aggregate per-dataset (mean, fold_std) into the balanced mean ± error row fields."""
    finite = {k: v for k, v in scores.items() if not np.isnan(v[0])}
    if not finite:
        return None
    means = [m for m, _ in finite.values()]
    fold_stds = [s for _, s in finite.values()]
    vmean, bmean, mean = balanced_group_mean({k: float(m) for k, (m, _) in finite.items()})
    n = len(means)
    sem = float(np.std(means) / np.sqrt(n))
    # combine cross-dataset SEM with mean within-dataset fold std
    err = float(np.sqrt(sem**2 + np.mean(np.array(fold_stds)**2)))
    return {"mean": mean, "virus_mean": vmean, "bacteria_mean": bmean, "std": err, "n": n}


def compute_model_aurocs(
    abl_dir: Path,
    dataset_keys: List[str] | None = None,
    checkpoint: str | None = None,
) -> List[Dict]:
    """Return one row per model with the balanced mean ± error AUROC across datasets.

    If *dataset_keys* is given, only those datasets are included.
    """
    rows: List[Dict] = []
    active_datasets = [(k, l) for k, l in DATASETS if dataset_keys is None or k in dataset_keys]

    # ---- Base model (_base files live in ablation_attention/base/) ----
    if FLAT_ATTN_DIR.exists():
        summary = summarize_model(grouped_cv_scores(
            FLAT_ATTN_DIR, "_base", active_datasets, "base", n_folds=4, min_seq_id=0.5))
        if summary is not None:
            rows.append({
                "model_key": "base",
                "label": "Base model (no LoRA)",
                **summary,
                "category": "base",
                "color": CATEGORY_COLORS["base"],
            })
            print(f"  {'Base model (no LoRA)':50s}  n={summary['n']}  "
                  f"mean={summary['mean']:.3f}  err={summary['std']:.3f}")

    # ---- Ablation models ----
    for model_key, label in ABLATION_MODELS.items():
        model_dir = abl_dir / model_key
        if model_key not in MODEL_READABLE:
            continue
        if not model_dir.exists():
            print(f"  Skipping {model_key}: directory not found")
            continue
        attn_dir = resolve_attn_dir_from_spec(model_dir, checkpoint)
        if attn_dir is None:
            ckpt_name = normalize_checkpoint_name(checkpoint)
            print(f"  Skipping {model_key}: checkpoint not found ({ckpt_name})")
            continue
        summary = summarize_model(grouped_cv_scores(
            attn_dir, "", active_datasets, model_key, n_folds=4, min_seq_id=0.5))
        if summary is None:
            print(f"  Skipping {model_key}: no data found")
            continue
        rows.append({
            "model_key": model_key,
            "label": MODEL_READABLE.get(model_key, label),
            **summary,
            "category": model_category(model_key),
            "color": CATEGORY_COLORS[model_category(model_key)],
        })
        print(f"  {model_key:50s}  n={summary['n']}  "
              f"mean={summary['mean']:.3f}  err={summary['std']:.3f}")

    return rows


# ---------------------------------------------------------------------------
# Per-dataset scores (for the grouped bar figure)
# ---------------------------------------------------------------------------

def compute_per_dataset_aurocs(
    abl_dir: Path,
    dataset_keys: List[str] | None = None,
    checkpoint: str | None = None,
) -> Dict:
    """Return {model_label: {"scores": {ds_label: auroc}, "category": ...}} incl. the base model.

    Uses grouped logistic-regression AUROC with sequence-identity-clean CV
    (base model: 3 folds, 40% clustering; ablation models: 2 folds, 50%).
    """
    active_datasets = [(k, l) for k, l in DATASETS if dataset_keys is None or k in dataset_keys]
    ds_labels = dict(active_datasets)

    result: Dict[str, Dict[str, float]] = {}

    # Base model
    if FLAT_ATTN_DIR.exists():
        scores = grouped_cv_scores(FLAT_ATTN_DIR, "_base", active_datasets, "base/per-ds",
                                   n_folds=3, min_seq_id=0.4)
        base_row = {ds_labels[k]: m for k, (m, _) in scores.items()}
        if base_row:
            result["Base model"] = {
                "scores": base_row,
                "category": "base",
            }

    # Ablation models
    for model_key, label in ABLATION_MODELS.items():
        model_dir = abl_dir / model_key
        if not model_dir.exists():
            continue
        attn_dir = resolve_attn_dir_from_spec(model_dir, checkpoint)
        if attn_dir is None:
            continue
        scores = grouped_cv_scores(attn_dir, "", active_datasets, f"{model_key}/per-ds",
                                   n_folds=2, min_seq_id=0.5)
        row = {ds_labels[k]: m for k, (m, _) in scores.items()}
        if row:
            result[MODEL_READABLE.get(model_key, label)] = {
                "scores": row,
                "category": model_category(model_key),
            }

    return result


# ---------------------------------------------------------------------------
# Figure 1 — mean AUROC per model (horizontal bars)
# ---------------------------------------------------------------------------

def plot(rows: List[Dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Sort descending so strongest model is shown first.
    rows = sorted(rows, key=lambda r: r["mean"], reverse=True)

    labels = [r["label"] for r in rows]
    means  = np.array([r["mean"]  for r in rows])
    colors = [r["color"] for r in rows]

    fig, ax = plt.subplots(figsize=(7.2, max(2.8, len(rows) * 0.34)))
    y = np.arange(len(rows))

    # Compact bars (touching/near-touching) and no error bars for visual clarity.
    ax.barh(y, means, height=0.96, color=colors, zorder=3)
    ax.axvline(0.5, color="gray", linestyle=":", lw=1, zorder=2)

    # Annotate each bar with its value
    for i, m in enumerate(means):
        ax.text(m + 0.003, i, f"{m:.3f}", va="center", fontsize=8.5)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel("Mean Logistic Regression AUROC")
    ax.set_title("Ablation Study: Mask Type and Block Attention")
    x_min = float(np.nanmin(means)) - 0.01
    x_max = float(np.nanmax(means)) + 0.01
    ax.set_xlim(max(0.5, x_min), min(1.0, x_max))
    ax.xaxis.grid(True, alpha=0.3, zorder=0)
    style_ax(ax)

    # Category legend (4 colors only).
    import matplotlib.patches as mpatches
    legend_order = ["base", "no_block", "block", "block_extended"]
    present = {r["category"] for r in rows}
    legend_handles = [
        mpatches.Patch(color=CATEGORY_COLORS[c], label=CATEGORY_LABELS[c])
        for c in legend_order if c in present
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8.5, frameon=False, ncol=2)

    plt.tight_layout()
    for ext in ("pdf", "png", "svg"):
        fig.savefig(out_dir / f"ablations.{ext}", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\nSaved → {out_dir}/ablations.{{pdf,png,svg}}")


# ---------------------------------------------------------------------------
# Figure 2 — per-dataset grouped bars (one group per dataset, one bar per model)
# ---------------------------------------------------------------------------

def plot_per_dataset(per_ds: Dict, out_dir: Path) -> None:
    """Grouped bar chart: x = dataset, bars = models (base + ablations)."""
    out_dir.mkdir(parents=True, exist_ok=True)

    model_labels = list(per_ds.keys())
    # Collect all dataset labels that appear in at least one model
    all_ds_labels: List[str] = []
    seen: set = set()
    for ds_key, ds_label in DATASETS:
        if ds_label not in seen:
            for row in per_ds.values():
                if ds_label in row["scores"]:
                    all_ds_labels.append(ds_label)
                    seen.add(ds_label)
                    break

    n_ds     = len(all_ds_labels)
    n_models = len(model_labels)
    width    = 0.9 / n_models
    x        = np.arange(n_ds)

    fig, ax = plt.subplots(figsize=(max(9.2, n_ds * 0.95), 4.2))

    for i, label in enumerate(model_labels):
        offsets = x + (i - n_models / 2 + 0.5) * width
        vals = [per_ds[label]["scores"].get(ds, np.nan) for ds in all_ds_labels]
        color = CATEGORY_COLORS[per_ds[label]["category"]]
        ax.bar(offsets, vals, width, color=color, alpha=0.9,
               label=label.replace("\n", " "), zorder=3)

    ax.axhline(0.5, color="gray", linestyle=":", lw=1, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(all_ds_labels, fontsize=10)
    ax.set_ylabel("Logistic Regression AUROC\n(HPI vs. random cross-species)")
    ax.set_title("Ablation Study: Per Dataset")
    ax.set_ylim(0.45, 1.05)
    ax.yaxis.grid(True, alpha=0.3, zorder=0)
    style_ax(ax)

    # Category-only legend (compact and readable).
    import matplotlib.patches as mpatches
    present = {per_ds[m]["category"] for m in model_labels}
    legend_order = ["base", "no_block", "block", "block_extended"]
    handles = [
        mpatches.Patch(color=CATEGORY_COLORS[c], label=CATEGORY_LABELS[c])
        for c in legend_order if c in present
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=8.5, frameon=False, ncol=2)

    plt.tight_layout()
    for ext in ("pdf", "png", "svg"):
        fig.savefig(out_dir / f"ablations_per_dataset.{ext}", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out_dir}/ablations_per_dataset.{{pdf,png,svg}}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ablation-dir",
        default=str(ABL_DIR),
        help=f"Directory containing one subdir per model (default: {ABL_DIR})",
    )
    parser.add_argument(
        "--datasets",
        default="sars_cov2_zhou2022,hiv1_jager2012,influenza_a,hpv,ebv,hsv1,chlamydia,tuberculosis,yersinia,salmonella",
        help="Comma-separated dataset keys to include (default: all). "
             f"Available: {', '.join(k for k, _ in DATASETS)}",
    )
    parser.add_argument(
        "--checkpoint",
        default="20000",
        help="Checkpoint to use for ablation models: latest, 5000, or checkpoint-5000 (default: latest)",
    )
    args = parser.parse_args()

    abl_dir = Path(args.ablation_dir)
    if not abl_dir.exists():
        sys.exit(f"ERROR: ablation directory not found: {abl_dir}\n"
                 "Run 0_extract_attention.py first.")

    dataset_keys = [d.strip() for d in args.datasets.split(",")] if args.datasets else None
    if dataset_keys:
        print(f"Using datasets: {', '.join(dataset_keys)}")
    print(f"Using checkpoint: {normalize_checkpoint_name(args.checkpoint) or 'latest'}")

    print("Computing mean AUROC per model…")
    rows = compute_model_aurocs(
        abl_dir,
        dataset_keys,
        checkpoint=args.checkpoint,
    )
    if not rows:
        sys.exit("No data found — check that .npz files are present.")

    plot(rows, FIGS_DIR)

    print("\nComputing per-dataset scores…")
    per_ds = compute_per_dataset_aurocs(
        abl_dir,
        dataset_keys,
        checkpoint=args.checkpoint,
    )
    plot_per_dataset(per_ds, FIGS_DIR)
