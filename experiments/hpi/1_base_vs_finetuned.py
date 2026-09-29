"""Part 1 — Base model vs. fine-tuned model across all datasets.

For each dataset the script computes the cross-validated logistic-regression
AUROC (all attention heads as features, 3-fold StratifiedKFold × 5 seeds) for
distinguishing true HPI pairs from random cross-species pairs, for both
the base ProteomeLM-S and the fine-tuned (LoRA) checkpoint.

Produces ONE figure:  figures/base_vs_finetuned.{pdf,png,svg}

Data required (written by 0_extract_attention.py):
    data/ablation_attention/proteomelm_hpi_s_abl8_seed_total_block/checkpoint-20000/
        {dataset}_{hpi,random_cross}_attention.npz
    data/ablation_attention/base/
        {dataset}_{hpi,random_cross}_base_attention.npz

Usage:
    python 1_base_vs_finetuned.py
    python 1_base_vs_finetuned.py --attention-dir /path/to/finetuned_attention
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt

# Ensure the utils module is importable when the script is run directly
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    CLEAN_LABELS, COLOR_BASE, COLOR_FINETUNED, DATASET_GROUP, DATASETS,
    load_attention, lr_auroc, style_ax,
)

ROOT          = Path(__file__).resolve().parent
ATTN_DIR      = ROOT / "data/ablation_attention/proteomelm_hpi_s_abl8_seed_total_block/checkpoint-20000"
BASE_ATTN_DIR = ROOT / "data/ablation_attention/base"
FIGS_DIR      = ROOT / "figures"

# Compact x-axis labels for small figure footprints
DISPLAY_LABELS = {
    "sars_cov2_zhou2022": "SARS-CoV-2",
    "hiv1_jager2012": "HIV-1",
    "influenza_a": "Influenza A",
    "ebv": "EBV",
    "hpv": "HPV-16",
    "hsv1": "HSV-1",
    "yersinia": "Y. pestis",
    "salmonella": "S. enterica",
    "chlamydia": "C. trachomatis",
    "tuberculosis": "M. tuberculosis",
}


# ---------------------------------------------------------------------------
# Compute metrics
# ---------------------------------------------------------------------------

def compute_aurocs(attn_dir: Path, base_attn_dir: Path) -> List[Dict]:
    """Return a row per dataset with base and finetuned AUROC."""
    rows = []
    for ds_key, _ds_label in DATASETS:
        ft   = load_attention(attn_dir, ds_key, suffix="")
        base = load_attention(base_attn_dir, ds_key, suffix="_base")
        if "hpi" not in ft or "random_cross" not in ft:
            print(f"  Skipping {ds_key}: finetuned attention not found")
            continue
        ft_score   = lr_auroc(ft["hpi"],   ft["random_cross"],n_folds=3, return_folds=True)[0]
        base_score = (lr_auroc(base["hpi"], base["random_cross"], n_folds=3, return_folds=True)[0]
                      if ("hpi" in base and "random_cross" in base)
                      else np.nan)
        rows.append({"key": ds_key, "label": CLEAN_LABELS.get(ds_key, ds_key),
                     "finetuned": ft_score, "base": base_score})
        print(f"  {ds_key:30s}  base={base_score:.3f}  finetuned={ft_score:.3f}")
    return rows


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def plot(rows: List[Dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Sort: viruses first (by finetuned AUROC desc), then bacteria, then fungi
    GROUP_ORDER = ["Virus", "Bacteria", "Fungus"]
    def sort_key(r):
        g = DATASET_GROUP.get(r["key"], "Other")
        g_idx = GROUP_ORDER.index(g) if g in GROUP_ORDER else len(GROUP_ORDER)
        return (g_idx, -r["finetuned"])
    rows = sorted(rows, key=sort_key)

    # Locate group boundaries and label positions
    groups = [DATASET_GROUP.get(r["key"], "Other") for r in rows]
    group_boundaries = [i for i in range(1, len(groups)) if groups[i] != groups[i - 1]]
    group_spans: Dict[str, tuple] = {}
    prev = 0
    for b in group_boundaries + [len(rows)]:
        g = groups[prev]
        group_spans[g] = (prev, b)
        prev = b

    n     = len(rows)
    x     = np.arange(n)
    width = 0.34

    fig, ax = plt.subplots(figsize=(max(5, n * 0.5), 4.35))

    ax.bar(x - width / 2,
           [r["base"]      for r in rows],
           width, color=COLOR_BASE,      label="Base model (no LoRA)", zorder=3)
    ax.bar(x + width / 2,
           [r["finetuned"] for r in rows],
           width, color=COLOR_FINETUNED, label="Fine-tuned (LoRA)", zorder=3)

    # Dashed separator lines between groups
    for b in group_boundaries:
        ax.axvline(b - 0.5, color="#aaaaaa", lw=1.2, ls="--", zorder=4)

    # Group label annotations just below the top
    ymax = 1.02
    for g, (lo, hi) in group_spans.items():
        cx = (lo + hi - 1) / 2
        ax.text(cx, ymax - 0.005, g + "s", ha="center", va="top",
                fontsize=9.3, color="#555555", fontstyle="italic", zorder=5)

    ax.axhline(0.5, color="gray", linestyle=":", lw=1, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY_LABELS.get(r["key"], r["label"]) for r in rows],
                       rotation=28, ha="right", fontsize=9.2)
    ax.set_ylabel("Logistic Regression AUROC\n(HPI vs. random cross-species)")
    ax.set_ylim(0.45, ymax)
    ax.set_title("ProteomeLM Base vs Fine-tuned", fontsize=12)
    ax.legend(loc="lower right", frameon=False, fontsize=9.2)
    ax.yaxis.grid(True, alpha=0.3, zorder=1)
    style_ax(ax)

    plt.tight_layout()
    for ext in ("pdf", "png", "svg"):
        fig.savefig(out_dir / f"base_vs_finetuned.{ext}", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\nSaved → {out_dir}/base_vs_finetuned.{{pdf,png,svg}}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--attention-dir",
        default=str(ATTN_DIR),
        help="Directory containing {dataset}_{type}[_base]_attention.npz files "
             f"(default: {ATTN_DIR})",
    )
    args = parser.parse_args()

    attn_dir = Path(args.attention_dir)
    if not attn_dir.exists():
        sys.exit(f"ERROR: attention directory not found: {attn_dir}\n"
                 "Run 0_extract_attention.py first.")

    print("Computing AUROC per dataset…")
    rows = compute_aurocs(attn_dir, BASE_ATTN_DIR)
    if not rows:
        sys.exit("No data found — check that attention .npz files are present.")

    plot(rows, FIGS_DIR)
