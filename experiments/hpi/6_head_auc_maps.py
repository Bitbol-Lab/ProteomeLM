#!/usr/bin/env python3
"""Per-head AUROC maps for base and fine-tuned ProteomeLM attention.

Produces two figures (fine-tuned and base) in the same heatmap style as the
user-provided notebook snippet, adapted to all available species.

Usage:
    python 6_head_auc_maps.py
    python 6_head_auc_maps.py --metric AUC
    python 6_head_auc_maps.py --finetuned-dir data/ablation_attention/proteomelm_hpi_s_abl8_seed_total_block/checkpoint-20000 \
                              --base-dir data/ablation_attention/base
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import CLEAN_LABELS, DATASETS, load_attention

SCRIPT_DIR = Path(__file__).resolve().parent

ROOT = SCRIPT_DIR

FT_ATTN_DIR = ROOT / "data/ablation_attention/proteomelm_hpi_s_abl8_seed_total_block/checkpoint-20000"
BASE_ATTN_DIR = ROOT / "data/ablation_attention/base"
FIGS_DIR = ROOT / "figures"


# Compact but clean subplot titles (manual line breaks avoid awkward wrapping)
TITLE_LABELS: Dict[str, str] = {
    "sars_cov2_zhou2022": "SARS-CoV-2",
    "hiv1_jager2012": "HIV-1",
    "influenza_a": "Influenza A\nvirus",
    "ebv": "Epstein-Barr\nvirus",
    "hpv": "Human papilloma-\nvirus 16",
    "hsv1": "Herpes simplex\nvirus 1",
    "yersinia": "Yersinia\npestis",
    "salmonella": "Salmonella\nTyphimurium",
    "chlamydia": "Chlamydia\ntrachomatis",
    "tuberculosis": "Mycobacterium\ntuberculosis",
}

# Species-specific palette endpoints (light -> dark), to mimic per-species color maps
SPECIES_PALETTES: Dict[str, Tuple[str, str]] = {
    "sars_cov2_zhou2022": ("#fff7ec", "#7f2704"),
    "hiv1_jager2012": ("#f7f4f9", "#3f007d"),
    "influenza_a": ("#f7fcf5", "#00441b"),
    "ebv": ("#f7f4f9", "#4a1486"),
    "hpv": ("#f7fbff", "#08306b"),
    "hsv1": ("#fff5f0", "#67000d"),
    "yersinia": ("#fff7ec", "#8c2d04"),
    "salmonella": ("#f7fbff", "#084594"),
    "chlamydia": ("#f7fcfd", "#014636"),
    "tuberculosis": ("#f7fcf0", "#00441b"),
}


def auc_matrix(pos_attn: np.ndarray, neg_attn: np.ndarray, metric: str = "AUC") -> np.ndarray:
    """Compute per-(layer,head) AUROC matrix."""
    n_layers, n_heads = pos_attn.shape[1], pos_attn.shape[2]
    y = np.concatenate([np.ones(len(pos_attn)), np.zeros(len(neg_attn))])

    mat = np.zeros((n_layers, n_heads), dtype=np.float32)
    for i in range(n_layers):
        for j in range(n_heads):
            scores = np.concatenate([pos_attn[:, i, j], neg_attn[:, i, j]])
            mat[i, j] = float(roc_auc_score(y, scores))

    if metric.upper() == "AUC":
        # Orientation-invariant head quality in [0.5, 1.0], as in your snippet
        mat = np.abs(mat - 0.5) + 0.5
    return mat


def available_species(
    finetuned_dir: Path,
    base_dir: Path,
) -> Tuple[List[str], List[str], List[str]]:
    """Return species available for finetuned, base, and either model.

    Finetuned expects {dataset}_hpi_attention.npz and {dataset}_random_cross_attention.npz.
    Base expects      {dataset}_hpi_base_attention.npz and {dataset}_random_cross_base_attention.npz.
    """
    ordered = [k for k, _ in DATASETS]

    ft_keys, base_keys = [], []
    for ds in ordered:
        ft_ok = (finetuned_dir / f"{ds}_hpi_attention.npz").exists() and (
            finetuned_dir / f"{ds}_random_cross_attention.npz"
        ).exists()
        base_ok = (base_dir / f"{ds}_hpi_base_attention.npz").exists() and (
            base_dir / f"{ds}_random_cross_base_attention.npz"
        ).exists()
        if ft_ok:
            ft_keys.append(ds)
        if base_ok:
            base_keys.append(ds)

    either = [ds for ds in ordered if ds in ft_keys or ds in base_keys]
    return ft_keys, base_keys, either


def species_cmap(species_key: str) -> LinearSegmentedColormap:
    low, high = SPECIES_PALETTES.get(species_key, ("#f3f4f6", "#1f2937"))
    return LinearSegmentedColormap.from_list(
        f"cmap_{species_key}",
        [low, "#ffffff", high],
        N=256,
    )


def draw_maps(
    matrices: Dict[str, np.ndarray],
    out_file: Path,
    title_prefix: str,
    metric: str,
) -> None:
    """Draw one multi-panel heatmap figure with notebook-style formatting."""
    species = list(matrices.keys())
    if not species:
        print(f"No matrices to plot for {title_prefix}.")
        return

    n = len(species)
    ncols = 4 if n >= 8 else (3 if n >= 5 else 2)
    nrows = int(math.ceil(n / ncols))

    fig_w = 4 * ncols
    fig_h = 4 * nrows + 0.7
    fig = plt.figure(figsize=(fig_w, fig_h))

    for k, spec in enumerate(species):
        ax = plt.subplot(nrows, ncols, k + 1)
        plt.subplots_adjust(hspace=0.40, wspace=0.26)

        matrix = matrices[spec]
        n_layers, n_heads = matrix.shape

        cmap = species_cmap(spec)
        # Adaptive scaling and nonlinear norm increase visible contrast.
        vmax = max(0.70, float(np.percentile(matrix, 95)))
        norm = mcolors.PowerNorm(gamma=0.65, vmin=0.5, vmax=vmax)
        im = ax.imshow(matrix, cmap=cmap, aspect="auto", norm=norm)

        # Cell borders (linewidths=1.0 look)
        ax.set_xticks(np.arange(-0.5, n_heads, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n_layers, 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=1.0)
        ax.tick_params(which="minor", bottom=False, left=False)

        # Annotations (fmt=".2f", annot_kws size ~11.5)
        for i in range(n_layers):
            for j in range(n_heads):
                val = float(matrix[i, j])
                txt = "white" if norm(val) > 0.60 else "#111111"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9.0, color=txt)

        clean_spec = TITLE_LABELS.get(spec, CLEAN_LABELS.get(spec, spec))
        ax.set_title(clean_spec, fontweight="bold", fontstyle="italic", fontsize=9.8)
        ax.set_xlabel("Head", fontsize=9.0)
        if k % ncols == 0:
            ax.set_ylabel("Layer", fontsize=9.0)

        # Match 1-indexed ticks from your snippet
        ax.set_xticks(np.arange(n_heads))
        ax.set_xticklabels([str(i + 1) for i in range(n_heads)], rotation=0, fontsize=8.5)
        ax.set_yticks(np.arange(n_layers))
        ax.set_yticklabels([str(i + 1) for i in range(n_layers)], rotation=0, fontsize=8.5)

        # Explicitly disable colorbar (as in your snippet)
        _ = im

    # Hide any unused panels
    total_axes = nrows * ncols
    for idx in range(n + 1, total_axes + 1):
        ax = plt.subplot(nrows, ncols, idx)
        ax.axis("off")

    fig.suptitle(f"{title_prefix} — per-head {metric.upper()} maps", y=1.01, fontsize=12.5)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg"):
        fig.savefig(out_file.with_suffix(f".{ext}"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved: {out_file.with_suffix('.pdf')}, {out_file.with_suffix('.svg')}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric", default="AUC", choices=["AUC", "auc"], help="Metric to display")
    parser.add_argument("--finetuned-dir", type=Path, default=FT_ATTN_DIR)
    parser.add_argument("--base-dir", type=Path, default=BASE_ATTN_DIR)
    parser.add_argument("--out-dir", type=Path, default=FIGS_DIR)
    args = parser.parse_args()

    ft_keys, base_keys, either_keys = available_species(args.finetuned_dir, args.base_dir)
    if not either_keys:
        raise SystemExit("No available species found in finetuned/base attention directories.")

    print(f"Available species (either model): {len(either_keys)}")
    print(f"  Fine-tuned: {len(ft_keys)}")
    print(f"  Base:       {len(base_keys)}")

    ft_mats: Dict[str, np.ndarray] = {}
    for ds in ft_keys:
        d = load_attention(args.finetuned_dir, ds, suffix="")
        if "hpi" not in d or "random_cross" not in d:
            continue
        ft_mats[ds] = auc_matrix(d["hpi"], d["random_cross"], metric=args.metric)

    base_mats: Dict[str, np.ndarray] = {}
    for ds in base_keys:
        d = load_attention(args.base_dir, ds, suffix="_base")
        if "hpi" not in d or "random_cross" not in d:
            continue
        base_mats[ds] = auc_matrix(d["hpi"], d["random_cross"], metric=args.metric)

    draw_maps(
        matrices=ft_mats,
        out_file=args.out_dir / "per_head_auc_finetuned_proteomelmS.pdf",
        title_prefix="Fine-tuned ProteomeLM-S",
        metric=args.metric,
    )
    draw_maps(
        matrices=base_mats,
        out_file=args.out_dir / "per_head_auc_base_proteomelmS.pdf",
        title_prefix="Base ProteomeLM-S",
        metric=args.metric,
    )


if __name__ == "__main__":
    main()
