#!/usr/bin/env python3
"""
=============================================================================
TRiC/CCT CHAPERONIN VALIDATION FOR PROTEOMELM (supplementary figure)
=============================================================================

Tests whether ProteomeLM attention over the full yeast proteome singles out
the eukaryotic chaperonin TRiC/CCT and its ring topology.

The TRiC complex consists of 8 paralogous subunits (CCT1-8) arranged in a
specific circular order within each ring (Kalisman et al. 2012, PNAS):

    CCT2 → CCT4 → CCT1 → CCT3 → CCT6 → CCT8 → CCT7 → CCT5 → (CCT2)

Two analyses build the supplementary figure (supp_cct_figure.{pdf,png,svg}):
  A. Complex membership: layer/head-averaged attention for the 28 CCT–CCT
     pairs vs 8 × 1000 CCT–background pairs (violin + ROC).
  B. Per-head AUC for adjacent (8) vs non-adjacent (20) ring pairs, with a
     permutation test on the best head.

Usage:
    python cct.py --prepare                   # Download yeast proteome, locate CCT subunits
    python cct.py --predict --model <path>    # ProteomeLM attention, analyses, figure

Reference:
    Kalisman et al. (2012) PNAS 109(8):2884-2889
    "Subunit order of eukaryotic TRiC/CCT chaperonin by cross-linking,
    mass spectrometry, and combinatorial homology modeling"
"""

import os
import sys
import json
import argparse
import itertools
import numpy as np
import torch
from pathlib import Path
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import shared utilities
from proteomelm.utils.io import ensure_dir, parse_fasta, save_json
from proteomelm.utils.embedding import compute_esm_embeddings, compute_proteomelm_embeddings, load_attentions
from proteomelm.utils.proteome import download_proteome, find_proteins_in_proteome
from experiments.examples.common import (
    COLOR_BLUE, COLOR_GRAY, COLOR_GREEN, COLOR_RED,
    apply_plot_style, build_orthodb_group_embeds, pair_attn, save_figure, style_axes, to_colormap,
)

apply_plot_style(16)

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    'proteome_id': 'UP000002311',  # Yeast reference proteome
    'esm_device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'proteome_device': 'cpu',
}

# =============================================================================
# TRiC/CCT SUBUNIT INFORMATION
# =============================================================================

# Yeast TRiC/CCT subunits with UniProt IDs
CCT_SUBUNITS = {
    'CCT1': {'uniprot': 'P12612', 'gene': 'TCP1',  'alt_name': 'TCP-1-alpha'},
    'CCT2': {'uniprot': 'P39076', 'gene': 'CCT2',  'alt_name': 'TCP-1-beta'},
    'CCT3': {'uniprot': 'P39077', 'gene': 'CCT3',  'alt_name': 'TCP-1-gamma'},
    'CCT4': {'uniprot': 'P39078', 'gene': 'CCT4',  'alt_name': 'TCP-1-delta'},
    'CCT5': {'uniprot': 'P40413', 'gene': 'CCT5',  'alt_name': 'TCP-1-epsilon'},
    'CCT6': {'uniprot': 'P39079', 'gene': 'CCT6',  'alt_name': 'TCP-1-zeta'},
    'CCT7': {'uniprot': 'P42943', 'gene': 'CCT7',  'alt_name': 'TCP-1-eta'},
    'CCT8': {'uniprot': 'P47079', 'gene': 'CCT8',  'alt_name': 'TCP-1-theta'},
}

# The established ring order (Kalisman et al. 2012, confirmed by cryo-EM)
# Reading clockwise around the ring
RING_ORDER = ['CCT6', 'CCT8', 'CCT7', 'CCT5', 'CCT2', 'CCT4', 'CCT1', 'CCT3', ]


# Generate adjacent pairs (8 pairs that are neighbors in the ring)
def get_adjacent_pairs():
    """Get the 8 pairs that are adjacent in the ring."""
    pairs = []
    for i in range(len(RING_ORDER)):
        a = RING_ORDER[i]
        b = RING_ORDER[(i + 1) % len(RING_ORDER)]
        pairs.append(tuple(sorted([a, b])))
    return set(pairs)


ADJACENT_PAIRS = get_adjacent_pairs()


# Generate all pairs (28 total). The order fixes the permutation test's draws.
def get_all_pairs():
    """Get all 28 possible pairs."""
    subunits = list(CCT_SUBUNITS.keys())
    return [tuple(sorted(p)) for p in itertools.combinations(subunits, 2)]


ALL_PAIRS = get_all_pairs()


# =============================================================================
# DATA PREPARATION
# =============================================================================

def find_cct_in_proteome_local(proteome_seqs: dict, output_dir: str) -> dict:
    """Find CCT subunits in proteome and return their indices."""

    # Use shared utility function
    cct_info = find_proteins_in_proteome(
        target_proteins=CCT_SUBUNITS,
        proteome_sequences=proteome_seqs,
        uniprot_key='uniprot'
    )

    # Add gene info
    for subunit in cct_info:
        cct_info[subunit]['gene'] = CCT_SUBUNITS[subunit]['gene']

    # Save mapping
    save_json(cct_info, os.path.join(output_dir, "cct_proteome_mapping.json"))

    return cct_info


# =============================================================================
# ANALYSES
# =============================================================================

def analyze_attention_heads(attentions: torch.Tensor, cct_mapping: dict, output_dir: str):
    """Analyze how well each attention head discriminates adjacent vs non-adjacent pairs."""
    from sklearn.metrics import roc_auc_score, average_precision_score

    print("  Analyzing attention head performance...")

    num_layers, num_heads, seq_len, _ = attentions.shape

    # Extract attention values for each pair
    pairs = ALL_PAIRS
    attn_values = np.zeros((len(pairs), num_layers, num_heads))
    labels = np.zeros(len(pairs))

    for i, (a, b) in enumerate(pairs):
        if a not in cct_mapping or b not in cct_mapping:
            continue

        idx_a = cct_mapping[a]['proteome_idx']
        idx_b = cct_mapping[b]['proteome_idx']

        for layer in range(num_layers):
            for head in range(num_heads):
                attn_values[i, layer, head] = pair_attn(attentions[layer, head], idx_a, idx_b)

        labels[i] = 1 if (a, b) in ADJACENT_PAIRS else 0

    # Compute AUC for each head
    auc_matrix = np.zeros((num_layers, num_heads))
    ap_matrix = np.zeros((num_layers, num_heads))

    for layer in range(num_layers):
        for head in range(num_heads):
            scores = attn_values[:, layer, head]
            auc_matrix[layer, head] = roc_auc_score(labels, scores)
            ap_matrix[layer, head] = average_precision_score(labels, scores)

    # Permutation test for significance
    print("  Running permutation test...")
    n_permutations = 1000
    np.random.seed(42)

    permuted_max_aucs = []

    for _ in range(n_permutations):
        perm_labels = np.random.permutation(labels)
        perm_max = 0
        for layer in range(num_layers):
            for head in range(num_heads):
                scores = attn_values[:, layer, head]
                auc = roc_auc_score(perm_labels, scores)
                perm_max = max(perm_max, auc)
        permuted_max_aucs.append(perm_max)

    permuted_max_aucs = np.array(permuted_max_aucs)

    observed_max_auc = auc_matrix.max()
    expected_max_null = permuted_max_aucs.mean()
    pvalue_max = (np.sum(permuted_max_aucs >= observed_max_auc) + 1) / (n_permutations + 1)

    print(f"  Observed best head AUC: {observed_max_auc:.4f}")
    print(f"  Expected max under null: {expected_max_null:.4f}")
    print(f"  P-value (corrected for {num_layers*num_heads} heads): {pvalue_max:.4f}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))

    sns.heatmap(
        auc_matrix,
        annot=True,
        fmt='.2f',
        cmap=to_colormap(COLOR_GREEN),
        vmin=0.3,
        vmax=0.9,
        center=0.5,
        ax=axes[0],
        cbar_kws={'label': 'ROC AUC'},
        annot_kws={'size': 12}
    )
    axes[0].set_xlabel('Head', fontsize=18)
    axes[0].set_ylabel('Layer', fontsize=18)
    axes[0].set_title('Attention Head Performance (ROC AUC)\nAdjacent vs Non-adjacent Discrimination', fontsize=20)
    cbar0 = axes[0].collections[0].colorbar
    cbar0.ax.tick_params(labelsize=14)
    cbar0.set_label('ROC AUC', fontsize=18)

    sns.heatmap(
        ap_matrix,
        annot=True,
        fmt='.2f',
        cmap=to_colormap(COLOR_GREEN),
        vmin=labels.mean(),
        vmax=0.8,
        ax=axes[1],
        cbar_kws={'label': 'Avg Precision'},
        annot_kws={'size': 12}
    )
    axes[1].set_xlabel('Head', fontsize=16)
    axes[1].set_ylabel('Layer', fontsize=16)
    axes[1].set_title('Attention Head Performance (Avg Precision)', fontsize=18)

    axes[2].hist(permuted_max_aucs, bins=40, alpha=0.65, color=COLOR_GRAY, density=True)
    axes[2].axvline(observed_max_auc, color=COLOR_RED, lw=2, linestyle='--',
                    label=f'Observed max = {observed_max_auc:.2f}')
    axes[2].axvline(expected_max_null, color=COLOR_BLUE, lw=2, linestyle=':',
                    label=f'Expected max = {expected_max_null:.2f}')
    axes[2].set_xlabel('Max AUC across all heads')
    axes[2].set_ylabel('Density')
    axes[2].set_title(f'Permutation Test (n={n_permutations})\nP-value = {pvalue_max:.2f}', fontsize=17)
    axes[2].legend(fontsize=13)
    axes[0].tick_params(axis='both', labelsize=15)
    axes[1].tick_params(axis='both', labelsize=14)
    axes[2].tick_params(axis='both', labelsize=14)

    style_axes(axes[0])
    style_axes(axes[1])
    style_axes(axes[2])

    plt.tight_layout()
    save_figure(fig, output_dir, "cct_attention_head_analysis", also_png=True)

    # Save results
    best_layer, best_head = np.unravel_index(auc_matrix.argmax(), auc_matrix.shape)

    results = {
        'auc_matrix': auc_matrix.tolist(),
        'ap_matrix': ap_matrix.tolist(),
        'best_head': {
            'layer': int(best_layer),
            'head': int(best_head),
            'auc': float(auc_matrix.max()),
            'ap': float(ap_matrix[best_layer, best_head])
        },
        'permutation_test': {
            'n_permutations': n_permutations,
            'observed_max_auc': float(observed_max_auc),
            'expected_max_null': float(expected_max_null),
            'pvalue_corrected': float(pvalue_max),
            'significant': bool(pvalue_max < 0.05)
        }
    }

    with open(os.path.join(output_dir, "cct_attention_head_results.json"), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"  Best head: Layer {best_layer}, Head {best_head} (AUC={auc_matrix.max():.4f})")
    print(f"  Saved: {os.path.join(output_dir, 'cct_attention_head_analysis.pdf')}")

    return results


def analyze_cct_membership(attentions: torch.Tensor, cct_mapping: dict, output_dir: str,
                           n_background_proteins: int = 1000):
    """
    Test whether ProteomeLM attention distinguishes CCT complex members
    from non-members (intra-complex vs inter-complex).

    Mirrors the ribosome analyze_membership() approach:
      - Intra-complex:  CCT_i <-> CCT_j  (all 28 pairs)
      - Inter-complex:  CCT_i <-> nonCCT_k  (sampled)

    Produces: violin plot + ROC curve figure, and saves metrics JSON.
    """
    from sklearn.metrics import roc_curve, auc as sk_auc
    from scipy import stats

    print("\n" + "=" * 70)
    print("CCT COMPLEX MEMBERSHIP ANALYSIS")
    print("=" * 70)
    print("  Comparing intra-complex (CCT-CCT) vs inter-complex (CCT-nonCCT)\n")

    num_layers, num_heads, seq_len, _ = attentions.shape

    cct_indices = sorted([info['proteome_idx'] for info in cct_mapping.values()])
    cct_set = set(cct_indices)
    non_cct_indices = [i for i in range(seq_len) if i not in cct_set]

    if len(non_cct_indices) < 100:
        print("  Skipping: insufficient background proteins")
        return None

    # ---- Intra-complex scores (CCT-CCT, 28 pairs) ----
    intra_scores = []
    for i in range(len(cct_indices)):
        for j in range(i + 1, len(cct_indices)):
            intra_scores.append(pair_attn(attentions, cct_indices[i], cct_indices[j]))

    # ---- Inter-complex scores (CCT-nonCCT, sampled) ----
    np.random.seed(42)
    sampled_bg = np.random.choice(
        non_cct_indices,
        min(len(non_cct_indices), n_background_proteins),
        replace=False
    )

    inter_scores = []
    for r_idx in cct_indices:
        for n_idx in sampled_bg:
            inter_scores.append(pair_attn(attentions, r_idx, n_idx))

    # ---- Statistics ----
    mw_stat, mw_pval = stats.mannwhitneyu(
        intra_scores, inter_scores, alternative='greater'
    )
    intra_mean = np.mean(intra_scores)
    inter_mean = np.mean(inter_scores)
    fold_enrich = intra_mean / inter_mean if inter_mean > 0 else float('inf')

    y_true = [0] * len(inter_scores) + [1] * len(intra_scores)
    y_score = inter_scores + intra_scores
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_val = sk_auc(fpr, tpr)

    print(f"  Intra-complex (CCT-CCT):   {intra_mean:.6f} (n={len(intra_scores)})")
    print(f"  Inter-complex (CCT-other): {inter_mean:.6f} (n={len(inter_scores)})")
    print(f"  Fold enrichment:           {fold_enrich:.2f}x")
    print(f"  Mann-Whitney p-value:      {mw_pval:.2e}")
    print(f"  Membership AUC:            {roc_val:.3f}")

    # ---- Figure: Violin + ROC (matching ribosome style) ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.5))

    # Panel: Violin plot
    parts = ax1.violinplot(
        [inter_scores, intra_scores], showmeans=False, showmedians=True
    )
    for pc in parts['bodies']:
        pc.set_facecolor(COLOR_RED)
        pc.set_alpha(0.7)
    parts['bodies'][1].set_facecolor(COLOR_BLUE)
    ax1.set_xticks([1, 2])
    ax1.set_xticklabels(['Inter-Complex\n(CCT–other)', 'Intra-Complex\n(CCT–CCT)'], fontsize=16)
    ax1.set_ylabel('Attention Score', fontsize=18)
    ax1.set_title(f'Complex Membership\np = {mw_pval:.2e}, fold = {fold_enrich:.1f}×', fontsize=20)
    ax1.tick_params(axis='y', labelsize=15)
    style_axes(ax1)

    # Panel: ROC
    ax2.plot(fpr, tpr, color=COLOR_BLUE, lw=3, label=f'AUC = {roc_val:.2f}')
    ax2.plot([0, 1], [0, 1], 'k--')
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('Membership ROC', fontsize=17)
    ax2.legend(loc='lower right', fontsize=13)
    style_axes(ax2)

    plt.tight_layout()

    save_figure(fig, output_dir, "cct_membership_analysis", also_png=True)
    print(f"\n  Figure saved: {os.path.join(output_dir, 'cct_membership_analysis.pdf')}")

    # ---- Save results ----
    results = {
        'intra_complex': {
            'n_pairs': len(intra_scores),
            'mean': float(intra_mean),
            'std': float(np.std(intra_scores)),
        },
        'inter_complex': {
            'n_pairs': len(inter_scores),
            'mean': float(inter_mean),
            'std': float(np.std(inter_scores)),
        },
        'comparison': {
            'fold_enrichment': float(fold_enrich),
            'mannwhitney_pvalue': float(mw_pval),
            'membership_auc': float(roc_val),
        }
    }

    with open(os.path.join(output_dir, "cct_membership_results.json"), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"  Results saved: {output_dir}/cct_membership_results.json")
    return results


def compose_cct_supplementary_figure(output_dir: str, fig_output_path: str = None):
    """
        Compose a 2-panel CCT supplementary figure from individual panel outputs.

        Panels:
            A: Complex Membership (violin panel only)
            B: Per-head AUC heatmap
    """
    from PIL import Image, ImageDraw, ImageFont

    membership_path = os.path.join(output_dir, "cct_membership_analysis.png")
    head_path = os.path.join(output_dir, "cct_attention_head_analysis.png")

    for p in [membership_path, head_path]:
        if not os.path.exists(p):
            print(f"  Warning: Missing {p} — skipping figure composition")
            return None

    if fig_output_path is None:
        fig_output_path = os.path.join(output_dir, "supp_cct_figure.pdf")

    print("\n  Composing supplementary figure...")

    def resize_to_height(img, h):
        ratio = h / img.height
        return img.resize((int(img.width * ratio), h), Image.LANCZOS)

    membership_img = Image.open(membership_path)
    head_img = Image.open(head_path)

    # Crop panel A from membership figure (left half: violin only)
    w_m = membership_img.width
    panel_a = membership_img.crop((0, 0, int(w_m * 0.50), membership_img.height))

    # Crop panel B from attention-head figure (left third: AUC heatmap + colorbar)
    w_h = head_img.width
    panel_b = head_img.crop((0, 0, int(w_h * 0.37), head_img.height))

    target_h = 2000
    gap = 60
    margin_top = 90

    pa = resize_to_height(panel_a, target_h)
    pb = resize_to_height(panel_b, target_h)

    row_w = pa.width + pb.width + gap
    total_w = row_w
    total_h = target_h + 2 * margin_top

    canvas = Image.new('RGB', (total_w, total_h), 'white')

    # Single row: two heatmaps
    x = 0
    canvas.paste(pa, (x, margin_top))
    x += pa.width + gap
    canvas.paste(pb, (x, margin_top))

    # Labels
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 84)
    except OSError:
        font = ImageFont.load_default()

    draw.text((0, 8), "A", fill='black', font=font)
    draw.text((pa.width + gap, 8), "B", fill='black', font=font)

    # Save raster/PDF composite
    os.makedirs(os.path.dirname(fig_output_path) or '.', exist_ok=True)
    canvas.save(fig_output_path, dpi=(450, 450))
    canvas.save(fig_output_path.replace('.pdf', '.png'), dpi=(450, 450))

    # Also save a true vector SVG (editable text/elements)
    svg_output_path = fig_output_path.replace('.pdf', '.svg')
    fig_svg, ax_svg = plt.subplots(figsize=(total_w / 450, total_h / 450), dpi=450)
    ax_svg.imshow(np.asarray(canvas))
    ax_svg.axis('off')
    fig_svg.subplots_adjust(0, 0, 1, 1)
    fig_svg.savefig(svg_output_path, format='svg', bbox_inches='tight', pad_inches=0)
    plt.close(fig_svg)

    print(f"  Saved supplementary figure: {fig_output_path} ({canvas.size[0]}x{canvas.size[1]})")
    print(f"  Saved supplementary figure (SVG): {svg_output_path}")
    return fig_output_path


# =============================================================================
# COMMAND FUNCTIONS
# =============================================================================

def cmd_prepare(output_dir: str):
    """Download the yeast proteome and locate the CCT subunits in it."""

    print("="*70)
    print("TRiC/CCT VALIDATION - PREPARE")
    print("="*70)

    print("\n[1/2] Downloading yeast proteome...")
    proteome_path = download_proteome(
        proteome_id=CONFIG['proteome_id'],
        output_dir=output_dir,
        organism_name='yeast',
        reviewed_only=True,
        include_isoforms=False
    )
    proteome_seqs = parse_fasta(proteome_path)
    print(f"  Proteome size: {len(proteome_seqs)} proteins")

    print("\n[2/2] Locating CCT subunits in proteome...")
    find_cct_in_proteome_local(proteome_seqs, output_dir)

    print("\n" + "="*70)
    print("READY - Run: python cct.py --predict --model <path_to_proteomelm>")
    print("="*70)


def cmd_predict(output_dir: str, model_path: str = None, orthodb_db_path: str = None,
                orthodb_tsv: str = None, orthodb_min_group_size: int = 10):
    """Run ProteomeLM on the yeast proteome, then the analyses and the figure."""

    print("="*70)
    print("TRiC/CCT VALIDATION - PREDICT")
    print("="*70)

    # Check prerequisites
    proteome_path = os.path.join(output_dir, "yeast_proteome.fasta")
    mapping_path = os.path.join(output_dir, "cct_proteome_mapping.json")

    if not os.path.exists(proteome_path) or not os.path.exists(mapping_path):
        print("Error: Run --prepare first")
        return

    # Compute ESM embeddings
    print("\n[1/5] Computing ESM embeddings...")
    esm_embeddings = compute_esm_embeddings(
        fasta_path=proteome_path,
        output_dir=output_dir,
        prefix="yeast",
        device=CONFIG['esm_device']
    )

    # OrthoDB group embeddings (without them, ESM embeddings serve as group embeddings)
    group_embeds = None
    if orthodb_db_path:
        print("\n[1b/5] Creating OrthoDB group embeddings...")
        try:
            group_embeds = build_orthodb_group_embeds(
                fasta_path=proteome_path,
                esm_embeddings=esm_embeddings,
                output_dir=output_dir,
                organism='yeast',
                proteome_id=CONFIG['proteome_id'],
                db_path=orthodb_db_path,
                tsv_path=orthodb_tsv,
                min_group_size=orthodb_min_group_size,
                cache_path=os.path.join(output_dir, "yeast_orthodb_group_embeddings.pt"),
            )
        except Exception as e:
            print(f"  Warning: Failed to create OrthoDB group embeddings: {e}")
            print("  Falling back to default (ESM as group embeddings)")
            group_embeds = None

    # Compute ProteomeLM embeddings and attentions (cached after the first run)
    print("\n[2/5] Computing ProteomeLM embeddings and attentions...")
    _, attentions = compute_proteomelm_embeddings(
        esm_embeddings=esm_embeddings,
        output_dir=output_dir,
        model_path=model_path,
        prefix="yeast",
        device=CONFIG['proteome_device'],
        group_embeds=group_embeds
    )
    if attentions is None:
        attentions = load_attentions(output_dir, prefix="yeast")

    with open(mapping_path) as f:
        cct_mapping = json.load(f)

    print("\n[3/5] Analyzing attention heads (adjacent vs non-adjacent ring pairs)...")
    analyze_attention_heads(attentions, cct_mapping, output_dir)

    print("\n[4/5] Analyzing CCT complex membership (intra vs inter)...")
    analyze_cct_membership(attentions, cct_mapping, output_dir)

    print("\n[5/5] Composing supplementary figure...")
    compose_cct_supplementary_figure(output_dir)

    print("\n" + "="*70)
    print("DONE")
    print(f"  Attention head analysis: {output_dir}/cct_attention_head_analysis.png")
    print(f"  CCT membership: {output_dir}/cct_membership_analysis.png")
    print(f"  Supplementary figure: {output_dir}/supp_cct_figure.pdf")
    print("="*70)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='TRiC/CCT validation for ProteomeLM (supplementary figure)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download the yeast proteome and locate the CCT subunits
  python %(prog)s --prepare

  # Run ProteomeLM with OrthoDB group embeddings, analyses and figure
  python %(prog)s --predict --model Bitbol-Lab/ProteomeLM-M \\
      --orthodb-db-path /path/to/training   # directory with group_vectors_*.pkl
"""
    )

    parser.add_argument('--prepare', '-p', action='store_true',
                        help='Download the yeast proteome and locate the CCT subunits')
    parser.add_argument('--predict', action='store_true',
                        help='Run ProteomeLM on the yeast proteome, the analyses and the figure')
    parser.add_argument('--output', '-o', type=str, default='./tric_validation',
                        help='Output directory')
    parser.add_argument('--model', '-m', type=str, default=None,
                        help='Path to ProteomeLM model')

    # OrthoDB functional embedding arguments
    parser.add_argument('--orthodb-group-vectors', type=str, default=None,
                        help='Path to an OrthoDB group_vectors_*.pkl file or its directory '
                             '(alias of --orthodb-db-path)')
    parser.add_argument('--orthodb-db-path', type=str, default=None,
                        help='Directory with group_vectors_*.pkl files')
    parser.add_argument('--orthodb-tsv', type=str, default=None,
                        help='OrthoDB TSV from download_proteome (optional)')
    parser.add_argument('--orthodb-min-group-size', type=int, default=10,
                        help='Minimum OrthoDB group size (default: 10)')

    args = parser.parse_args()

    ensure_dir(args.output)

    if args.prepare:
        cmd_prepare(args.output)

    elif args.predict:
        cmd_predict(
            args.output,
            args.model,
            orthodb_db_path=args.orthodb_db_path or args.orthodb_group_vectors,
            orthodb_tsv=args.orthodb_tsv,
            orthodb_min_group_size=args.orthodb_min_group_size,
        )

    else:
        parser.print_help()
        print("\n⚠️  Specify --prepare or --predict")


if __name__ == "__main__":
    main()
