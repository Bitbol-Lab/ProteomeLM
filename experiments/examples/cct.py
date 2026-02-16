#!/usr/bin/env python3
"""
=============================================================================
TRiC/CCT CHAPERONIN RING ORDERING VALIDATION FOR PROTEOMELM
=============================================================================

Tests whether ProteomeLM attention can recover the correct subunit arrangement
of the eukaryotic chaperonin TRiC/CCT.

The TRiC complex consists of 8 paralogous subunits (CCT1-8) arranged in a 
specific circular order within each ring. This order was determined by 
Kalisman et al. 2012 (PNAS) using cross-linking mass spectrometry:

    CCT2 → CCT4 → CCT1 → CCT3 → CCT6 → CCT8 → CCT7 → CCT5 → (CCT2)

This is a stringent test of whether ProteomeLM learned "emergent" structural
organization, because:
1. All 8 subunits are paralogs (similar sequences)
2. All are in the same complex (should all have high attention)
3. But only 8 pairs are adjacent in the ring (out of 28 total)

If ProteomeLM learned the ring topology, adjacent pairs should have higher
attention than non-adjacent pairs.

Usage:
    python tric_cct_validation.py --prepare      # Setup and ground truth
    python tric_cct_validation.py --predict      # Run ProteomeLM
    python tric_cct_validation.py --evaluate     # Analyze results

Reference:
    Kalisman et al. (2012) PNAS 109(8):2884-2889
    "Subunit order of eukaryotic TRiC/CCT chaperonin by cross-linking, 
    mass spectrometry, and combinatorial homology modeling"

Author: For ProteomeLM reviewer response
"""

import os
import sys
import json
import argparse
import itertools
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Get workspace root
WORKSPACE_ROOT = Path(__file__).parent.parent.parent

# Import shared utilities
from proteomelm.utils.io import ensure_dir, parse_fasta, save_json
from proteomelm.utils.embedding import compute_esm_embeddings, compute_proteomelm_embeddings, load_attentions
from proteomelm.utils.proteome import (
    download_proteome,
    find_proteins_in_proteome,
    load_orthodb_group_vectors,
    build_group_embeddings_for_proteome,
)

# =============================================================================
# PLOTTING STYLE
# =============================================================================

colorspal6 = [
    (0.25098039215686274, 0.3254901960784314, 0.8274509803921568),
    (0.8666666666666667, 0.7019607843137254, 0.06274509803921569),
    (0.7098039215686275, 0.11372549019607843, 0.0784313725490196),
    (0.0, 0.7450980392156863, 1.0),
    (0.984313725490196, 0.28627450980392155, 0.6901960784313725),
    (0.0, 0.6980392156862745, 0.36470588235294116),
    (0.792156862745098, 0.792156862745098, 0.792156862745098),
]

colorspal12 = [
    (0.9215686274509803, 0.6745098039215687, 0.13725490196078433),
    (0.7215686274509804, 0.0, 0.34509803921568627),
    (0.0, 0.5490196078431373, 0.9764705882352941),
    (0.0, 0.43137254901960786, 0.0),
    (0.0, 0.7333333333333333, 0.6784313725490196),
    (0.8196078431372549, 0.38823529411764707, 0.9019607843137255),
    (0.6980392156862745, 0.27058823529411763, 0.00784313725490196),
    (1.0, 0.5725490196078431, 0.5294117647058824),
    (0.34901960784313724, 0.32941176470588235, 0.8392156862745098),
    (0.0, 0.7764705882352941, 0.9725490196078431),
    (0.5294117647058824, 0.5215686274509804, 0.0),
    (0.0, 0.6549019607843137, 0.4235294117647059),
    (0.7411764705882353, 0.7411764705882353, 0.7411764705882353),
]

COLOR_BLUE = colorspal6[0]
COLOR_YELLOW = colorspal6[1]
COLOR_RED = colorspal6[2]
COLOR_CYAN = colorspal6[3]
COLOR_MAGENTA = colorspal6[4]
COLOR_GREEN = colorspal6[5]
COLOR_GRAY = colorspal6[6]


def to_colormap(base_color, dark=0.2, light=0.9, name="custom_colormap"):
    import colorsys
    import matplotlib.colors as mcolors

    rgb = mcolors.to_rgb(base_color)
    h, l, s = colorsys.rgb_to_hls(*rgb)
    colors = [
        colorsys.hls_to_rgb(h, light, s),
        colorsys.hls_to_rgb(h, l, s),
        colorsys.hls_to_rgb(h, dark, s),
    ]
    rgb_colors = [mcolors.to_rgb(c) for c in colors]
    return mcolors.LinearSegmentedColormap.from_list(name, rgb_colors)


def apply_plot_style():
    plt.rcParams.update({
        "font.family": "Arial",
        "text.usetex": False,
        "font.size": 16,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "legend.fontsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "svg.fonttype": "none",
    })


def style_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def save_figure(fig, output_dir, name, also_png=False):
    pdf_path = os.path.join(output_dir, f"{name}.pdf")
    svg_path = os.path.join(output_dir, f"{name}.svg")
    fig.savefig(pdf_path, dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
    if also_png:
        png_path = os.path.join(output_dir, f"{name}.png")
        fig.savefig(png_path, dpi=400, bbox_inches="tight", facecolor="white")
    plt.close(fig)


apply_plot_style()

# =============================================================================
# ORTHODB FUNCTIONAL EMBEDDING SUPPORT
# =============================================================================

def create_orthodb_group_embeddings(
    fasta_path: str,
    output_dir: str,
    orthodb_config: dict,
    esm_embeddings: torch.Tensor = None,
    prefix: str = "proteome",
    use_online: bool = False
) -> torch.Tensor:
    """
    Create OrthoDB group embeddings for each protein in a proteome.
    
    For each protein, looks up its OrthoDB group and uses the mean embedding
    of that group as the functional embedding. For proteins without a known
    OrthoDB group, falls back to their ESM embedding.
    
    Args:
        fasta_path: Path to FASTA file with protein sequences
        output_dir: Directory to save/cache group embeddings
        orthodb_config: Configuration dict with paths to OrthoDB data
        esm_embeddings: ESM embeddings to use as fallback (optional)
        prefix: Prefix for output files
        
    Returns:
        Tensor of group embeddings (n_proteins, hidden_dim)
    """
    cache_path = os.path.join(output_dir, f"{prefix}_orthodb_group_embeddings.pt")
    
    # Load from cache if available
    if os.path.exists(cache_path):
        print(f"  Loading cached OrthoDB group embeddings: {cache_path}")
        return torch.load(cache_path, map_location='cpu')
    
    print(f"  Creating OrthoDB group embeddings for proteome...")

    db_path = orthodb_config.get('orthodb_db_path') or orthodb_config.get('group_vectors_path')
    if db_path and os.path.isfile(db_path):
        db_path = os.path.dirname(db_path.split(',')[0])

    tsv_path = orthodb_config.get('orthodb_tsv_path')
    if tsv_path is None:
        safe_name = os.path.basename(fasta_path).split('_')[0]
        tsv_path = os.path.join(output_dir, f"{safe_name}_orthodb.tsv")
        if not os.path.exists(tsv_path):
            download_proteome(
                proteome_id=CONFIG['proteome_id'],
                output_dir=output_dir,
                organism_name='yeast',
                reviewed_only=True,
                include_isoforms=False,
                download_orthodb=True
            )

    if not db_path or not os.path.exists(db_path):
        raise FileNotFoundError(f"OrthoDB group vectors path not found: {db_path}")
    if not tsv_path or not os.path.exists(tsv_path):
        raise FileNotFoundError(f"OrthoDB TSV not found: {tsv_path}")

    orthodb_means = load_orthodb_group_vectors(
        db_path,
        min_group_size=orthodb_config.get('min_group_size', 10)
    )
    group_embeddings, mask = build_group_embeddings_for_proteome(
        fasta_path=fasta_path,
        orthodb_tsv_path=tsv_path,
        orthodb_group_means=orthodb_means,
        esm_embeddings=esm_embeddings,
    )
    n_found = mask.sum().item()
    print(f"    Found OrthoDB groups for {n_found}/{len(mask)} proteins ({100*n_found/len(mask):.1f}%)")
    
    # Save to cache
    torch.save(group_embeddings, cache_path)
    print(f"  Saved OrthoDB group embeddings: {group_embeddings.shape} -> {cache_path}")
    
    return group_embeddings


def compute_orthodb_functional_similarity(output_dir: str, orthodb_config: dict = None) -> pd.DataFrame:
    """
    Compute similarity matrix based on OrthoDB functional embeddings.
    
    This provides a functional baseline: proteins in the same ortholog group
    should have similar functional embeddings.
    
    Args:
        output_dir: Output directory
        orthodb_config: Configuration dict with paths to OrthoDB data:
            - group_vectors_path: Path to OrthoDB group vectors pickle
            - uniprot_to_og_path: Path to UniProt-to-OG mapping
            
    Returns:
        DataFrame with functional similarity matrix for CCT subunits
    """
    from sklearn.metrics.pairwise import cosine_similarity
    if orthodb_config is None:
        print("  Warning: orthodb_config not provided for functional similarity")
        return None

    print("\n  Computing OrthoDB functional embeddings...")

    db_path = orthodb_config.get('orthodb_db_path') or orthodb_config.get('group_vectors_path')
    if db_path and os.path.isfile(db_path):
        db_path = os.path.dirname(db_path.split(',')[0])
    tsv_path = orthodb_config.get('orthodb_tsv_path')

    if not db_path or not os.path.exists(db_path):
        print("  Warning: OrthoDB group vectors path not found")
        return None
    if not tsv_path or not os.path.exists(tsv_path):
        print("  Warning: OrthoDB TSV not found")
        return None
    
    # Get UniProt IDs for CCT subunits
    subunits = list(CCT_SUBUNITS.keys())
    uniprot_ids = [CCT_SUBUNITS[s]['uniprot'] for s in subunits]
    
    proteome_path = os.path.join(output_dir, "yeast_proteome.fasta")
    if not os.path.exists(proteome_path):
        print("  Warning: yeast_proteome.fasta not found")
        return None

    orthodb_means = load_orthodb_group_vectors(
        db_path,
        min_group_size=orthodb_config.get('min_group_size', 10)
    )
    group_embeds, mask = build_group_embeddings_for_proteome(
        fasta_path=proteome_path,
        orthodb_tsv_path=tsv_path,
        orthodb_group_means=orthodb_means,
        esm_embeddings=None,
    )

    # Extract embeddings for the CCT subunits
    sequences = parse_fasta(proteome_path)
    proteome_ids = list(sequences.keys())
    id_to_idx = {uid: i for i, uid in enumerate(proteome_ids)}
    indices = [id_to_idx.get(uid) for uid in uniprot_ids]
    if any(i is None for i in indices):
        print("  Warning: Some CCT subunits missing in proteome mapping")
        return None

    embeddings = group_embeds[indices]
    mapped_count = int(mask[indices].sum().item())
    print(f"  Found functional embedding vectors for {mapped_count}/{len(uniprot_ids)} CCT subunits")
        
    # Compute cosine similarity
    similarity_matrix = cosine_similarity(embeddings.numpy())
    
    # Create DataFrame
    sim_df = pd.DataFrame(similarity_matrix, index=subunits, columns=subunits)
    
    # Save
    out_path = os.path.join(output_dir, "cct_orthodb_functional_similarity.csv")
    sim_df.to_csv(out_path)
    print(f"  Saved OrthoDB functional similarity: {out_path}")
    
    return sim_df


# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    'output_dir': './tric_validation',
    'organism': 'Saccharomyces cerevisiae',
    'proteome_id': 'UP000002311',  # Yeast reference proteome
    'esm_device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'proteome_device': 'cpu',
    'proteomelm_model': None,  # Path to ProteomeLM model
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

# Generate all pairs (28 total)
def get_all_pairs():
    """Get all 28 possible pairs."""
    subunits = list(CCT_SUBUNITS.keys())
    return [tuple(sorted(p)) for p in itertools.combinations(subunits, 2)]

ALL_PAIRS = get_all_pairs()

# Non-adjacent pairs (20 pairs)
NON_ADJACENT_PAIRS = set(ALL_PAIRS) - ADJACENT_PAIRS


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_ring_distance(a, b):
    """
    Get the minimum distance between two subunits in the ring.
    Adjacent = 1, opposite = 4 (for 8-member ring).
    """
    try:
        i = RING_ORDER.index(a)
        j = RING_ORDER.index(b)
    except ValueError:
        return None
    
    diff = abs(i - j)
    ring_size = len(RING_ORDER)
    return min(diff, ring_size - diff)


# =============================================================================
# STEP 1: PREPARE DATA
# =============================================================================

def prepare_data(output_dir: str):
    """Prepare ground truth and data files."""
    
    print("="*70)
    print("TRiC/CCT VALIDATION: PREPARATION")
    print("="*70)
    
    # Create ground truth matrices
    subunits = list(CCT_SUBUNITS.keys())
    n = len(subunits)
    
    # Adjacency matrix (binary: 1 if adjacent in ring)
    adj_matrix = np.zeros((n, n), dtype=int)
    for i, a in enumerate(subunits):
        for j, b in enumerate(subunits):
            if i != j and tuple(sorted([a, b])) in ADJACENT_PAIRS:
                adj_matrix[i, j] = 1
    
    adj_df = pd.DataFrame(adj_matrix, index=subunits, columns=subunits)
    adj_df.to_csv(os.path.join(output_dir, "adjacency_matrix.csv"))
    
    # Distance matrix (ring distance: 1-4)
    dist_matrix = np.zeros((n, n), dtype=int)
    for i, a in enumerate(subunits):
        for j, b in enumerate(subunits):
            if i != j:
                dist_matrix[i, j] = get_ring_distance(a, b)
    
    dist_df = pd.DataFrame(dist_matrix, index=subunits, columns=subunits)
    dist_df.to_csv(os.path.join(output_dir, "ring_distance_matrix.csv"))
    
    # Save subunit info
    with open(os.path.join(output_dir, "cct_subunits.json"), 'w') as f:
        json.dump(CCT_SUBUNITS, f, indent=2)
    
    # Save ring order
    with open(os.path.join(output_dir, "ring_order.json"), 'w') as f:
        json.dump({
            'order': RING_ORDER,
            'adjacent_pairs': [list(p) for p in ADJACENT_PAIRS],
            'reference': 'Kalisman et al. 2012 PNAS 109(8):2884-2889'
        }, f, indent=2)
    
    # Create pairs to evaluate
    pairs_data = []
    for pair in ALL_PAIRS:
        a, b = pair
        pairs_data.append({
            'subunit_a': a,
            'subunit_b': b,
            'uniprot_a': CCT_SUBUNITS[a]['uniprot'],
            'uniprot_b': CCT_SUBUNITS[b]['uniprot'],
            'is_adjacent': 1 if pair in ADJACENT_PAIRS else 0,
            'ring_distance': get_ring_distance(a, b)
        })
    
    pairs_df = pd.DataFrame(pairs_data)
    pairs_df.to_csv(os.path.join(output_dir, "pairs_to_evaluate.csv"), index=False)
    
    with open(os.path.join(output_dir, "pairs_to_evaluate.json"), 'w') as f:
        json.dump(pairs_data, f, indent=2)
    
    # Summary
    print(f"\n[1/2] Ground truth prepared:")
    print(f"  - 8 CCT subunits")
    print(f"  - 28 total pairs")
    print(f"  - 8 adjacent pairs (ring neighbors)")
    print(f"  - 20 non-adjacent pairs")
    print(f"\n[2/2] Ring order (Kalisman et al. 2012):")
    print(f"  {' → '.join(RING_ORDER)} → (CCT2)")
    print(f"\nAdjacent pairs:")
    for p in sorted(ADJACENT_PAIRS):
        print(f"  {p[0]} - {p[1]}")
    
    print(f"\nOutput: {output_dir}/")
    print(f"  - adjacency_matrix.csv")
    print(f"  - ring_distance_matrix.csv")
    print(f"  - pairs_to_evaluate.csv")
    print(f"  - cct_subunits.json")
    print(f"  - ring_order.json")


# =============================================================================
# STEP 2: PROTEOMELM PREDICTION (using shared utilities)
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



def extract_cct_attention_matrix(output_dir: str) -> pd.DataFrame:
    """Extract 8x8 attention matrix for CCT subunits from full proteome attention."""
    
    mapping_path = os.path.join(output_dir, "cct_proteome_mapping.json")
    
    # Load attention using shared utility
    print("  Loading attention weights...")
    try:
        attentions = load_attentions(output_dir, prefix="yeast")
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Attention file not found. Run --predict first.") from e
    
    with open(mapping_path) as f:
        cct_mapping = json.load(f)
    
    subunits = list(CCT_SUBUNITS.keys())
    n = len(subunits)
    
    # Extract attention submatrix
    attention_matrix = np.zeros((n, n))
    
    for i, a in enumerate(subunits):
        for j, b in enumerate(subunits):
            if a not in cct_mapping or b not in cct_mapping:
                continue
            
            idx_a = cct_mapping[a]['proteome_idx']
            idx_b = cct_mapping[b]['proteome_idx']
            
            # Average attention across all layers and heads (both directions)
            attn_ab = attentions[:, :, idx_a, idx_b].mean().item()
            attn_ba = attentions[:, :, idx_b, idx_a].mean().item()
            attention_matrix[i, j] = (attn_ab + attn_ba) / 2
    
    attention_df = pd.DataFrame(attention_matrix, index=subunits, columns=subunits)
    attention_df.to_csv(os.path.join(output_dir, "cct_attention_matrix.csv"))
    
    print(f"  Saved CCT attention matrix: {output_dir}/cct_attention_matrix.csv")
    return attention_df


def analyze_attention_heads(output_dir: str):
    """Analyze how well each attention head discriminates adjacent vs non-adjacent pairs."""
    from sklearn.metrics import roc_auc_score, average_precision_score
    from scipy import stats
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    attn_path = os.path.join(output_dir, "yeast_proteomelm_attentions.pt")
    mapping_path = os.path.join(output_dir, "cct_proteome_mapping.json")    
    print("  Analyzing attention head performance...")
    
    attentions = torch.load(attn_path, map_location='cpu')
    num_layers, num_heads, seq_len, _ = attentions.shape
    
    with open(mapping_path) as f:
        cct_mapping = json.load(f)
    
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
                attn_ab = attentions[layer, head, idx_a, idx_b].item()
                attn_ba = attentions[layer, head, idx_b, idx_a].item()
                attn_values[i, layer, head] = (attn_ab + attn_ba) / 2
        
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


def analyze_ring_distance_correlation(output_dir: str):
    """Analyze correlation between attention and ring distance for each head."""
    from scipy import stats
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    attn_path = os.path.join(output_dir, "yeast_proteomelm_attentions.pt")
    mapping_path = os.path.join(output_dir, "cct_proteome_mapping.json")
    
    if not os.path.exists(attn_path):
        print("  Error: Run --predict first")
        return
    
    print("  Analyzing attention-distance correlation per head...")
    
    attentions = torch.load(attn_path, map_location='cpu')
    num_layers, num_heads, seq_len, _ = attentions.shape
    
    with open(mapping_path) as f:
        cct_mapping = json.load(f)
    
    # Extract attention and distance for each pair
    pairs = ALL_PAIRS
    attn_values = np.zeros((len(pairs), num_layers, num_heads))
    distances = np.zeros(len(pairs))
    
    for i, (a, b) in enumerate(pairs):
        if a not in cct_mapping or b not in cct_mapping:
            continue
        
        idx_a = cct_mapping[a]['proteome_idx']
        idx_b = cct_mapping[b]['proteome_idx']
        
        for layer in range(num_layers):
            for head in range(num_heads):
                attn_ab = attentions[layer, head, idx_a, idx_b].item()
                attn_ba = attentions[layer, head, idx_b, idx_a].item()
                attn_values[i, layer, head] = (attn_ab + attn_ba) / 2
        
        distances[i] = get_ring_distance(a, b)
    
    # Compute Spearman correlation for each head
    # Negative correlation = as distance increases, attention decreases = good (close pairs have high attention)
    corr_matrix = np.zeros((num_layers, num_heads))
    pval_matrix = np.zeros((num_layers, num_heads))
    
    for layer in range(num_layers):
        for head in range(num_heads):
            rho, p = stats.spearmanr(distances, attn_values[:, layer, head])
            corr_matrix[layer, head] = rho  # Keep raw correlation
            pval_matrix[layer, head] = p
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    # Note: corr_matrix is [num_layers, num_heads], which maps to [rows, cols] = [Y, X]
    sns.heatmap(corr_matrix, annot=False, fmt='.3f', cmap='RdBu_r', 
                vmin=-0.6, vmax=0.6, center=0, ax=axes[0], 
                cbar_kws={'label': 'Correlation (Distance vs Attention)'},
                xticklabels=list(range(num_heads)),
                yticklabels=list(range(num_layers)))
    axes[0].set_xlabel('Head')
    axes[0].set_ylabel('Layer')
    axes[0].set_title('Attention-Distance Correlation\n(Negative = Higher attention for closer subunits, Red = Good)')
    
    # Significance mask
    sig_mask = pval_matrix < 0.05
    sig_matrix = np.where(sig_mask, corr_matrix, np.nan)
    
    sns.heatmap(sig_matrix, annot=False, fmt='.3f', cmap='RdBu_r', 
                vmin=-0.6, vmax=0.6, center=0, ax=axes[1], 
                cbar_kws={'label': 'Significant Correlations (p<0.05)'},
                xticklabels=list(range(num_heads)),
                yticklabels=list(range(num_layers)))
    axes[1].set_xlabel('Head')
    axes[1].set_ylabel('Layer')
    axes[1].set_title(f'Significant Correlations Only\n({sig_mask.sum()}/{num_layers*num_heads} heads)')
    axes[0].tick_params(axis='both', labelsize=13)
    axes[1].tick_params(axis='both', labelsize=13)
    
    style_axes(axes[0])
    style_axes(axes[1])

    plt.tight_layout()
    save_figure(fig, output_dir, "cct_distance_correlation", also_png=True)
    
    # Save results
    # Find best correlated head (most negative = strongest signal for proximity)
    best_layer, best_head = np.unravel_index(corr_matrix.argmin(), corr_matrix.shape)
    
    results = {
        'correlation_matrix': corr_matrix.tolist(),
        'pvalue_matrix': pval_matrix.tolist(),
        'best_correlated_head': {
            'layer': int(best_layer),
            'head': int(best_head),
            'correlation': float(corr_matrix[best_layer, best_head]),
            'pvalue': float(pval_matrix[best_layer, best_head])
        },
        'n_significant_heads': int(sig_mask.sum()),
        'total_heads': num_layers * num_heads
    }
    
    with open(os.path.join(output_dir, "cct_distance_correlation_results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"  Best correlated head: Layer {best_layer}, Head {best_head} (ρ={corr_matrix[best_layer, best_head]:.4f})")
    print(f"  Significant heads: {sig_mask.sum()}/{num_layers*num_heads}")
    print(f"  Saved: {os.path.join(output_dir, 'cct_distance_correlation.pdf')}")
    
    return results


def analyze_cct_vs_background(output_dir: str, n_random_pairs: int = 1000):
    """
    Compare CCT intra-complex attention against random background protein pairs.
    
    This tests whether CCT subunits have elevated attention to each other compared
    to random protein pairs from the human proteome (excluding CCT proteins).
    
    Similar to ribosome validation but for the CCT complex.
    """
    from sklearn.metrics import roc_auc_score, average_precision_score
    from scipy import stats
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    attn_path = os.path.join(output_dir, "yeast_proteomelm_attentions.pt")
    mapping_path = os.path.join(output_dir, "cct_proteome_mapping.json")
    
    if not os.path.exists(attn_path):
        print("  Error: Run --predict first")
        return None
    
    print("\n" + "="*70)
    print("CCT vs BACKGROUND ANALYSIS")
    print("="*70)
    print("  Testing if CCT proteins have elevated attention to each other")
    print("  compared to random background protein pairs.\n")
    
    # Load data
    print("  Loading attention weights...")
    attentions = torch.load(attn_path, map_location='cpu')
    num_layers, num_heads, seq_len, _ = attentions.shape
    
    with open(mapping_path) as f:
        cct_mapping = json.load(f)
    
    # Get CCT protein indices
    cct_indices = set()
    for subunit, info in cct_mapping.items():
        cct_indices.add(info['proteome_idx'])
    
    print(f"  CCT protein indices: {sorted(cct_indices)}")
    print(f"  Total proteome size: {seq_len}")
    
    # Non-CCT indices for random sampling
    non_cct_indices = [i for i in range(seq_len) if i not in cct_indices]
    print(f"  Non-CCT proteins available: {len(non_cct_indices)}")
    
    # Check if we have enough background proteins (skip if ring-only mode)
    if len(non_cct_indices) < 100:
        print("\n  ⚠️  Insufficient background proteins (likely ring-only mode).")
        print("  Skipping CCT vs background analysis (requires full proteome).\n")
        return None
    
    # ==========================================================================
    # EXTRACT CCT-CCT ATTENTION (all 28 pairs)
    # ==========================================================================
    print("\n  Extracting CCT-CCT attention (intra-complex)...")
    
    cct_cct_attention = []
    for pair in ALL_PAIRS:
        a, b = pair
        if a not in cct_mapping or b not in cct_mapping:
            continue
        
        idx_a = cct_mapping[a]['proteome_idx']
        idx_b = cct_mapping[b]['proteome_idx']
        
        # Average attention across all layers and heads
        attn_ab = attentions[:, :, idx_a, idx_b].mean().item()
        attn_ba = attentions[:, :, idx_b, idx_a].mean().item()
        attn = (attn_ab + attn_ba) / 2
        
        cct_cct_attention.append({
            'pair': (a, b),
            'idx_pair': (idx_a, idx_b),
            'attention': attn,
            'type': 'cct_cct'
        })
    
    print(f"  CCT-CCT pairs: {len(cct_cct_attention)}")
    
    # ==========================================================================
    # GENERATE RANDOM BACKGROUND PAIRS (excluding CCT proteins)
    # ==========================================================================
    print(f"\n  Generating {n_random_pairs} random background pairs...")
    np.random.seed(42)
    
    random_attention = []
    sampled_pairs = set()
    
    attempts = 0
    while len(random_attention) < n_random_pairs and attempts < n_random_pairs * 10:
        # Sample two distinct non-CCT proteins
        i, j = np.random.choice(non_cct_indices, 2, replace=False)
        pair = (min(i, j), max(i, j))
        
        if pair not in sampled_pairs:
            sampled_pairs.add(pair)
            
            # Extract attention
            attn_ab = attentions[:, :, i, j].mean().item()
            attn_ba = attentions[:, :, j, i].mean().item()
            attn = (attn_ab + attn_ba) / 2
            
            random_attention.append({
                'idx_pair': pair,
                'attention': attn,
                'type': 'random'
            })
        
        attempts += 1
    
    print(f"  Random pairs sampled: {len(random_attention)}")
    
    # ==========================================================================
    # STATISTICAL COMPARISON
    # ==========================================================================
    print("\n  Computing statistics...")
    
    cct_scores = [p['attention'] for p in cct_cct_attention]
    random_scores = [p['attention'] for p in random_attention]
    
    cct_mean = np.mean(cct_scores)
    random_mean = np.mean(random_scores)
    cct_std = np.std(cct_scores)
    random_std = np.std(random_scores)
    
    # Mann-Whitney U test (CCT > random)
    stat, mw_pval = stats.mannwhitneyu(cct_scores, random_scores, alternative='greater')
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt(((len(cct_scores)-1)*cct_std**2 + (len(random_scores)-1)*random_std**2) / 
                         (len(cct_scores) + len(random_scores) - 2))
    cohens_d = (cct_mean - random_mean) / pooled_std if pooled_std > 0 else 0
    
    # T-test (for comparison)
    t_stat, t_pval = stats.ttest_ind(cct_scores, random_scores, alternative='greater')
    
    # ROC AUC (treating CCT as positive class)
    all_scores = cct_scores + random_scores
    labels = [1] * len(cct_scores) + [0] * len(random_scores)
    auc = roc_auc_score(labels, all_scores)
    ap = average_precision_score(labels, all_scores)
    
    print(f"\n  CCT-CCT attention:     {cct_mean:.6f} ± {cct_std:.6f} (n={len(cct_scores)})")
    print(f"  Random background:     {random_mean:.6f} ± {random_std:.6f} (n={len(random_scores)})")
    print(f"  Ratio (CCT/random):    {cct_mean/random_mean:.2f}×")
    print(f"  Cohen's d:             {cohens_d:.3f}")
    print(f"  Mann-Whitney U p-val:  {mw_pval:.2e}")
    print(f"  ROC AUC:               {auc:.4f}")
    print(f"  Average Precision:     {ap:.4f}")
    
    # ==========================================================================
    # PER-HEAD ANALYSIS
    # ==========================================================================
    print("\n  Analyzing per-head discrimination (CCT vs background)...")
    
    # Extract per-head attention for CCT pairs
    cct_attn_per_head = np.zeros((len(ALL_PAIRS), num_layers, num_heads))
    for i, pair in enumerate(ALL_PAIRS):
        a, b = pair
        if a not in cct_mapping or b not in cct_mapping:
            continue
        idx_a = cct_mapping[a]['proteome_idx']
        idx_b = cct_mapping[b]['proteome_idx']
        
        attn_ab = attentions[:, :, idx_a, idx_b].numpy()
        attn_ba = attentions[:, :, idx_b, idx_a].numpy()
        cct_attn_per_head[i] = (attn_ab + attn_ba) / 2
    
    # Extract per-head attention for random pairs
    random_attn_per_head = np.zeros((len(random_attention), num_layers, num_heads))
    for i, p in enumerate(random_attention):
        idx_a, idx_b = p['idx_pair']
        attn_ab = attentions[:, :, idx_a, idx_b].numpy()
        attn_ba = attentions[:, :, idx_b, idx_a].numpy()
        random_attn_per_head[i] = (attn_ab + attn_ba) / 2
    
    # Compute AUC for each head
    auc_matrix = np.zeros((num_layers, num_heads))
    
    all_labels = [1] * len(ALL_PAIRS) + [0] * len(random_attention)
    
    for layer in range(num_layers):
        for head in range(num_heads):
            head_scores = np.concatenate([
                cct_attn_per_head[:, layer, head],
                random_attn_per_head[:, layer, head]
            ])
            try:
                auc_matrix[layer, head] = roc_auc_score(all_labels, head_scores)
            except:
                auc_matrix[layer, head] = 0.5
    
    best_layer, best_head = np.unravel_index(auc_matrix.argmax(), auc_matrix.shape)
    
    print(f"  Best head for CCT vs background:")
    print(f"    Layer {best_layer}, Head {best_head}: AUC = {auc_matrix.max():.4f}")
    print(f"  Mean AUC across all heads: {auc_matrix.mean():.4f}")
    
    # ==========================================================================
    # VISUALIZATION
    # ==========================================================================
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # Panel A: Distribution comparison
    ax = axes[0, 0]
    ax.hist(random_scores, bins=50, alpha=0.7, label=f'Random (n={len(random_scores)})',
            color=COLOR_GRAY, density=True)
    ax.hist(cct_scores, bins=20, alpha=0.7, label=f'CCT-CCT (n={len(cct_scores)})',
            color=COLOR_BLUE, density=True)
    ax.axvline(random_mean, color=COLOR_GRAY, linestyle='--', linewidth=2)
    ax.axvline(cct_mean, color=COLOR_BLUE, linestyle='--', linewidth=2)
    ax.set_xlabel('Attention Score')
    ax.set_ylabel('Density')
    ax.set_title(f'CCT vs Random Background\np = {mw_pval:.2e}, d = {cohens_d:.2f}', fontweight='bold')
    ax.legend()
    
    # Panel B: Box plot
    ax = axes[0, 1]
    box_data = [random_scores, cct_scores]
    bp = ax.boxplot(box_data, labels=['Random\nBackground', 'CCT\nComplex'], 
                    patch_artist=True)
    bp['boxes'][0].set_facecolor(COLOR_GRAY)
    bp['boxes'][1].set_facecolor(COLOR_BLUE)
    ax.set_ylabel('Attention Score')
    ax.set_title(f'CCT Attention is {cct_mean/random_mean:.2f}× Higher', fontweight='bold')
    
    # Add individual points
    ax.scatter([1]*len(random_scores[:100]), random_scores[:100], alpha=0.3, color=COLOR_GRAY, s=10)
    ax.scatter([2]*len(cct_scores), cct_scores, alpha=0.5, color=COLOR_BLUE, s=30)
    
    # Panel C: Per-head AUC heatmap
    ax = axes[0, 2]
    sns.heatmap(auc_matrix, annot=True, fmt='.3f', cmap='RdYlGn', 
                vmin=0.5, vmax=1.0, center=0.75, ax=ax, cbar_kws={'label': 'ROC AUC'})
    ax.set_xlabel('Head')
    ax.set_ylabel('Layer')
    ax.set_title('Per-Head AUC (CCT vs Background)', fontweight='bold')
    
    # Panel D: CCT attention matrix heatmap
    ax = axes[1, 0]
    # Create attention matrix for the 8 CCT subunits
    n_subunits = len(RING_ORDER)
    attn_matrix = np.zeros((n_subunits, n_subunits))
    subunits = list(CCT_SUBUNITS.keys())
    for i, a in enumerate(subunits):
        for j, b in enumerate(subunits):
            if i == j:
                continue
            if a in cct_mapping and b in cct_mapping:
                idx_a = cct_mapping[a]['proteome_idx']
                idx_b = cct_mapping[b]['proteome_idx']
                attn_ab = attentions[:, :, idx_a, idx_b].mean().item()
                attn_ba = attentions[:, :, idx_b, idx_a].mean().item()
                attn_matrix[i, j] = (attn_ab + attn_ba) / 2
    
    im = ax.imshow(attn_matrix, cmap='Blues')
    ax.set_xticks(range(n_subunits))
    ax.set_yticks(range(n_subunits))
    ax.set_xticklabels(subunits, rotation=45, ha='right')
    ax.set_yticklabels(subunits)
    ax.set_title('CCT Intra-complex Attention', fontweight='bold')
    plt.colorbar(im, ax=ax, label='Attention')
    
    # Mark diagonal
    for i in range(n_subunits):
        ax.add_patch(plt.Rectangle((i-0.5, i-0.5), 1, 1, fill=True, 
                                    facecolor='white', edgecolor='black'))
    
    # Panel E: Cumulative distribution
    ax = axes[1, 1]
    random_sorted = np.sort(random_scores)
    cct_sorted = np.sort(cct_scores)
    
    ax.plot(random_sorted, np.linspace(0, 1, len(random_sorted)),
            label='Random', color=COLOR_GRAY, linewidth=2)
    ax.plot(cct_sorted, np.linspace(0, 1, len(cct_sorted)),
            label='CCT-CCT', color=COLOR_BLUE, linewidth=2)
    ax.set_xlabel('Attention Score')
    ax.set_ylabel('Cumulative Fraction')
    ax.set_title('Cumulative Distribution', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Panel F: Summary
    ax = axes[1, 2]
    ax.axis('off')
    
    summary_text = f"""
CCT vs Background Analysis
{'='*40}

SAMPLES
  CCT-CCT pairs:    {len(cct_scores)} (all 8 subunits)
  Random pairs:     {len(random_scores)} (non-CCT proteome)

ATTENTION SCORES
  CCT mean:         {cct_mean:.6f} ± {cct_std:.6f}
  Random mean:      {random_mean:.6f} ± {random_std:.6f}
  Fold enrichment:  {cct_mean/random_mean:.2f}×

STATISTICAL TESTS
  Mann-Whitney U:   p = {mw_pval:.2e}
  Cohen's d:        {cohens_d:.3f}
  ROC AUC:          {auc:.4f}
  Avg Precision:    {ap:.4f}

PER-HEAD ANALYSIS
  Best head:        L{best_layer}H{best_head} (AUC={auc_matrix.max():.3f})
  Mean head AUC:    {auc_matrix.mean():.3f}

INTERPRETATION
  {'✓ CCT proteins show elevated attention' if mw_pval < 0.05 else '✗ No significant elevation'}
  {'  (complex members attend to each other more)' if mw_pval < 0.05 else ''}
"""
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    for ax in axes.flat:
        if hasattr(ax, "spines"):
            style_axes(ax)

    plt.tight_layout()
    save_figure(fig, output_dir, "cct_vs_background_analysis")

    print(f"\n  Figure saved: {os.path.join(output_dir, 'cct_vs_background_analysis.pdf')}")
    
    # Save results
    results = {
        'cct_cct': {
            'n_pairs': len(cct_scores),
            'mean': float(cct_mean),
            'std': float(cct_std),
            'scores': cct_scores
        },
        'random_background': {
            'n_pairs': len(random_scores),
            'mean': float(random_mean),
            'std': float(random_std)
        },
        'comparison': {
            'fold_enrichment': float(cct_mean / random_mean),
            'cohens_d': float(cohens_d),
            'mannwhitney_pvalue': float(mw_pval),
            'ttest_pvalue': float(t_pval),
            'roc_auc': float(auc),
            'average_precision': float(ap),
            'significant': bool(mw_pval < 0.05)
        },
        'per_head_analysis': {
            'auc_matrix': auc_matrix.tolist(),
            'best_head': {
                'layer': int(best_layer),
                'head': int(best_head),
                'auc': float(auc_matrix.max())
            },
            'mean_auc': float(auc_matrix.mean())
        }
    }
    
    with open(os.path.join(output_dir, "cct_vs_background_results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"  Results saved: {output_dir}/cct_vs_background_results.json")
    
    return results


def analyze_cct_membership(output_dir: str, n_background_proteins: int = 1000):
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
    import matplotlib.pyplot as plt

    attn_path = os.path.join(output_dir, "yeast_proteomelm_attentions.pt")
    mapping_path = os.path.join(output_dir, "cct_proteome_mapping.json")

    if not os.path.exists(attn_path):
        print("  Error: Run --predict first")
        return None

    print("\n" + "=" * 70)
    print("CCT COMPLEX MEMBERSHIP ANALYSIS")
    print("=" * 70)
    print("  Comparing intra-complex (CCT-CCT) vs inter-complex (CCT-nonCCT)\n")

    # Load data
    attentions = torch.load(attn_path, map_location='cpu')
    num_layers, num_heads, seq_len, _ = attentions.shape

    with open(mapping_path) as f:
        cct_mapping = json.load(f)

    cct_indices = sorted([info['proteome_idx'] for info in cct_mapping.values()])
    cct_set = set(cct_indices)
    non_cct_indices = [i for i in range(seq_len) if i not in cct_set]

    if len(non_cct_indices) < 100:
        print("  Skipping: insufficient background proteins (ring-only mode)")
        return None

    # ---- Intra-complex scores (CCT-CCT, 28 pairs) ----
    intra_scores = []
    for i in range(len(cct_indices)):
        for j in range(i + 1, len(cct_indices)):
            idx_a, idx_b = cct_indices[i], cct_indices[j]
            score = (attentions[:, :, idx_a, idx_b].mean().item() +
                     attentions[:, :, idx_b, idx_a].mean().item()) / 2
            intra_scores.append(score)

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
            score = (attentions[:, :, r_idx, n_idx].mean().item() +
                     attentions[:, :, n_idx, r_idx].mean().item()) / 2
            inter_scores.append(score)

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
    ax1.set_xticklabels(['Inter-Complex\n(CCT\u2013other)', 'Intra-Complex\n(CCT\u2013CCT)'], fontsize=16)
    ax1.set_ylabel('Attention Score', fontsize=18)
    ax1.set_title(f'Complex Membership\np = {mw_pval:.2e}, fold = {fold_enrich:.1f}\u00d7', fontsize=20)
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

    print(f"\n  Composing supplementary figure...")

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
# STEP 3: ANALYSIS FUNCTIONS
# =============================================================================

def reconstruct_ring_order_analysis(output_dir: str, orthodb_config: dict = None):
    """
    Comprehensive analysis of ring order reconstruction from attention.
    Finds the best order and compares it with ground truth using multiple scoring methods.
    """
    import matplotlib.pyplot as plt
    from scipy import stats
    from itertools import permutations
    
    attn_path = os.path.join(output_dir, "cct_attention_matrix.csv")
    
    if not os.path.exists(attn_path):
        print("  Error: Run --predict first to generate attention matrix")
        return
    
    print("\n" + "="*70)
    print("RING ORDER RECONSTRUCTION ANALYSIS")
    print("="*70)
    
    attention_df = pd.read_csv(attn_path, index_col=0)
    subunits = list(attention_df.index)
    
    # Load ESM embeddings for baseline comparison
    esm_emb_path = os.path.join(output_dir, "yeast_esm_embeddings.pt")
    esm_sim_df = None
    
    if os.path.exists(esm_emb_path):
        print("\n[0b/6] Computing ESM embedding similarity baseline...")
        
        mapping_path = os.path.join(output_dir, "cct_proteome_mapping.json")
        if os.path.exists(mapping_path):
            esm_embeddings = torch.load(esm_emb_path, map_location='cpu')
            with open(mapping_path) as f:
                cct_mapping = json.load(f)
            
            # Filter subunits to only those found in mapping
            mapped_subunits = [s for s in subunits if s in cct_mapping]
            
            if len(mapped_subunits) < len(subunits):
                missing = set(subunits) - set(mapped_subunits)
                print(f"  Warning: {len(missing)} subunits not found in mapping: {missing}")
            
            # Extract ESM embeddings for CCT proteins
            cct_esm_embeddings = []
            
            for subunit in mapped_subunits:
                idx = cct_mapping[subunit]['proteome_idx']
                # Convert from BFloat16 to Float32 before numpy conversion
                cct_esm_embeddings.append(esm_embeddings[idx].float().numpy())
            
            cct_esm_embeddings = np.array(cct_esm_embeddings)
            
            # Compute cosine similarity matrix
            from sklearn.metrics.pairwise import cosine_similarity
            esm_sim_matrix = cosine_similarity(cct_esm_embeddings)
            
            esm_sim_df = pd.DataFrame(esm_sim_matrix, index=mapped_subunits, columns=mapped_subunits)
            print(f"  ESM similarity matrix computed (shape: {esm_sim_matrix.shape})")
            print(f"  Mean similarity: {esm_sim_matrix[np.triu_indices_from(esm_sim_matrix, k=1)].mean():.4f}")
        else:
            print(f"  Warning: Mapping file not found")
    else:
        print(f"  Note: ESM embeddings not available, skipping baseline")
    
    # Load full attention tensor for best head analysis
    full_attn_path = os.path.join(output_dir, "yeast_proteomelm_attentions.pt")
    mapping_path = os.path.join(output_dir, "cct_proteome_mapping.json")
    
    best_head_df = None
    best_head_info = None
    
    if os.path.exists(full_attn_path) and os.path.exists(mapping_path):
        print("\n[0c/6] Loading best attention head...")
        
        # Load attention head analysis results
        head_results_path = os.path.join(output_dir, "cct_attention_head_results.json")
        if os.path.exists(head_results_path):
            with open(head_results_path) as f:
                head_results = json.load(f)
            
            best_layer = head_results['best_head']['layer']
            best_head = head_results['best_head']['head']
            best_auc = head_results['best_head']['auc']
            
            print(f"  Best head: Layer {best_layer}, Head {best_head} (AUC={best_auc:.4f})")
            
            # Extract attention for best head
            attentions = torch.load(full_attn_path, map_location='cpu')
            with open(mapping_path) as f:
                cct_mapping = json.load(f)
            
            # Create attention matrix for best head
            n = len(subunits)
            best_head_attn = np.zeros((n, n))
            
            for i, a in enumerate(subunits):
                for j, b in enumerate(subunits):
                    if a not in cct_mapping or b not in cct_mapping:
                        continue
                    idx_a = cct_mapping[a]['proteome_idx']
                    idx_b = cct_mapping[b]['proteome_idx']
                    
                    # Use only the best head
                    attn_ab = attentions[best_layer, best_head, idx_a, idx_b].item()
                    attn_ba = attentions[best_layer, best_head, idx_b, idx_a].item()
                    best_head_attn[i, j] = (attn_ab + attn_ba) / 2
            
            best_head_df = pd.DataFrame(best_head_attn, index=subunits, columns=subunits)
            best_head_info = {
                'layer': best_layer,
                'head': best_head,
                'auc': best_auc
            }
            print(f"  Extracted attention matrix for best head")
        else:
            print(f"  Warning: Best head results not found, using average attention only")
    else:
        print(f"  Note: Full attention tensor not available, using average attention only")
    
    # Apply APC (Average Product Correct) to reduce phylogenetic bias
    def apply_apc(matrix_df):
        """
        Apply Average Product Correct (APC) to a similarity/attention matrix.
        APC(i,j) = M(i,j) - (mean(M(i,:)) * mean(M(:,j))) / mean(M)
        """
        M = matrix_df.values.copy()
        
        # Set diagonal to zero for mean calculations (avoid self-similarity)
        np.fill_diagonal(M, 0)
        
        # Calculate means
        row_means = M.mean(axis=1, keepdims=True)
        col_means = M.mean(axis=0, keepdims=True)
        total_mean = M.mean()
        
        # Apply APC correction
        if total_mean != 0:
            M_apc = M - (row_means @ col_means) / total_mean
        else:
            M_apc = M
        
        # Set diagonal back to zero
        np.fill_diagonal(M_apc, 0)
        
        return pd.DataFrame(M_apc, index=matrix_df.index, columns=matrix_df.columns)
    
    # Create APC-corrected versions
    attention_apc_df = apply_apc(attention_df)
    print(f"  Applied APC to average attention")
    
    best_head_apc_df = None
    if best_head_df is not None:
        best_head_apc_df = apply_apc(best_head_df)
        print(f"  Applied APC to best head attention")
    
    esm_sim_apc_df = None
    if esm_sim_df is not None:
        esm_sim_apc_df = apply_apc(esm_sim_df)
        print(f"  Applied APC to ESM similarity")
    
    # Define multiple scoring functions
    def score_sum(order, attn_df):
        """Sum of attention for adjacent pairs"""
        score = 0
        for i in range(len(order)):
            a, b = order[i], order[(i + 1) % len(order)]
            score += (attn_df.loc[a, b] + attn_df.loc[b, a]) / 2
        return score
    
    def score_product(order, attn_df):
        """Product of attention (geometric strength)"""
        score = 1.0
        for i in range(len(order)):
            a, b = order[i], order[(i + 1) % len(order)]
            score *= (attn_df.loc[a, b] + attn_df.loc[b, a]) / 2
        return score
    
    def score_min(order, attn_df):
        """Minimum attention (weakest link)"""
        scores = []
        for i in range(len(order)):
            a, b = order[i], order[(i + 1) % len(order)]
            scores.append((attn_df.loc[a, b] + attn_df.loc[b, a]) / 2)
        return min(scores)
    
    def score_harmonic(order, attn_df):
        """Harmonic mean (penalizes weak links)"""
        scores = []
        for i in range(len(order)):
            a, b = order[i], order[(i + 1) % len(order)]
            s = (attn_df.loc[a, b] + attn_df.loc[b, a]) / 2
            scores.append(s)
        return len(scores) / sum(1/(s+1e-10) for s in scores)
    
    def score_variance_penalty(order, attn_df):
        """Sum minus variance (prefer uniform attention)"""
        scores = []
        for i in range(len(order)):
            a, b = order[i], order[(i + 1) % len(order)]
            scores.append((attn_df.loc[a, b] + attn_df.loc[b, a]) / 2)
        return sum(scores) - np.std(scores)
    
    def score_exponential(order, attn_df):
        """Sum of exponential attention (emphasize strong links)"""
        score = 0
        for i in range(len(order)):
            a, b = order[i], order[(i + 1) % len(order)]
            attn = (attn_df.loc[a, b] + attn_df.loc[b, a]) / 2
            score += np.exp(attn * 1000)  # Scale up for numerical stability
        return score
    
    scoring_methods = {
        'sum': ('Sum', score_sum),
        'product': ('Product', score_product),
        'min': ('Min (Weakest Link)', score_min),
        'harmonic': ('Harmonic Mean', score_harmonic),
        'var_penalty': ('Sum - Variance', score_variance_penalty),
        'exponential': ('Exponential Sum', score_exponential)
    }
    
    # Create list of attention sources to test
    attention_sources = [
        ('average', 'Average (All Heads)', attention_df),
        ('average_apc', 'Average + APC', attention_apc_df)
    ]
    
    if best_head_df is not None:
        attention_sources.append(('best_head', f'Best Head (L{best_head_info["layer"]}H{best_head_info["head"]})', best_head_df))
        if best_head_apc_df is not None:
            attention_sources.append(('best_head_apc', f'Best Head + APC', best_head_apc_df))
    
    if esm_sim_df is not None:
        attention_sources.append(('esm_baseline', 'ESM Similarity (Baseline)', esm_sim_df))
        if esm_sim_apc_df is not None:
            attention_sources.append(('esm_baseline_apc', 'ESM + APC (Baseline)', esm_sim_apc_df))
    
    # Add OrthoDB functional similarity if available
    if orthodb_config is not None:
        orthodb_sim_df = compute_orthodb_functional_similarity(output_dir, orthodb_config)
        if orthodb_sim_df is not None:
            attention_sources.append(('orthodb_func', 'OrthoDB Functional', orthodb_sim_df))
            orthodb_apc_df = apply_apc(orthodb_sim_df)
            attention_sources.append(('orthodb_func_apc', 'OrthoDB Functional + APC', orthodb_apc_df))
            print(f"  Added OrthoDB functional similarity sources")
    
    # Find common subunits across all sources (some might be missing from ESM)
    common_subunits = set(subunits)
    for src_key, src_name, src_df in attention_sources:
        common_subunits = common_subunits.intersection(set(src_df.index))
    
    common_subunits = sorted(common_subunits)  # Keep consistent order
    
    if len(common_subunits) < len(subunits):
        missing = set(subunits) - set(common_subunits)
        print(f"  Note: Using {len(common_subunits)}/{len(subunits)} common subunits (missing: {missing})")
        # Update subunits list to only common ones
        subunits = common_subunits
        # Also update ring order to only include common subunits
        ring_order_common = [s for s in RING_ORDER if s in common_subunits]
    else:
        ring_order_common = RING_ORDER
    
    n_subunits = len(ring_order_common)
    
    print(f"\n[1/5] Exhaustive search over all ring permutations...")
    print(f"  Testing {len(scoring_methods)} scoring methods × {len(attention_sources)} attention sources...")
    
    # Fix first element to avoid counting rotations multiple times
    fixed = subunits[0]
    others = subunits[1:]
    
    # Calculate scores for all permutations with all methods and sources
    all_permutations = {src[0]: {method: [] for method in scoring_methods.keys()} for src in attention_sources}
    
    for perm in permutations(others):
        order = [fixed] + list(perm)
        order_str = '→'.join(order)
        
        for src_key, src_name, src_attn_df in attention_sources:
            for method, (method_name, score_func) in scoring_methods.items():
                score = score_func(order, src_attn_df)
                all_permutations[src_key][method].append({
                    'order': order,
                    'score': score,
                    'order_str': order_str
                })
    
    # Sort by score for each method and source
    for src_key in all_permutations.keys():
        for method in scoring_methods.keys():
            all_permutations[src_key][method].sort(key=lambda x: x['score'], reverse=True)
    
    print(f"  Total permutations evaluated: {len(all_permutations[attention_sources[0][0]]['sum'])} per method/source")
    
    # Report results for each method and source
    print(f"\n[2/5] Evaluating ground truth order with all methods and sources...")
    
    results_by_method = {}
    
    for src_key, src_name, src_attn_df in attention_sources:
        print(f"\n  === {src_name} ===")
        results_by_method[src_key] = {}
        
        for method, (method_name, score_func) in scoring_methods.items():
            perms = all_permutations[src_key][method]
            best_order = perms[0]['order']
            best_score = perms[0]['score']
            
            # Calculate ground truth score using common subunits
            true_score = score_func(ring_order_common, src_attn_df)
            
            # Find rank of ground truth
            true_rank = None
            for idx, perm in enumerate(perms):
                # Check exact match or reverse
                if perm['order'] == ring_order_common or perm['order'] == list(reversed(ring_order_common)):
                    true_rank = idx + 1
                    break
            
            if true_rank is None:
                # Check for rotations
                for idx, perm in enumerate(perms):
                    for rot in range(len(ring_order_common)):
                        rotated = ring_order_common[rot:] + ring_order_common[:rot]
                        if perm['order'] == rotated or perm['order'] == list(reversed(rotated)):
                            true_rank = idx + 1
                            break
                    if true_rank:
                        break
            
            percentile = (true_rank / len(perms)) * 100 if true_rank else None
            similarity = ring_similarity(best_order, ring_order_common)
            
            # Statistics
            scores = [p['score'] for p in perms]
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            z_true = (true_score - mean_score) / std_score if std_score > 0 else 0
            
            results_by_method[src_key][method] = {
                'source_name': src_name,
                'method_name': method_name,
                'best_order': best_order,
                'best_score': best_score,
                'true_score': true_score,
                'true_rank': true_rank,
                'percentile': percentile,
                'similarity': similarity,
                'z_score': z_true,
                'all_perms': perms,
                'mean_score': mean_score,
                'std_score': std_score
            }
            
            print(f"    {method_name}:")
            print(f"      Best: {' → '.join(best_order[:4])}...{best_order[-1]} (score={best_score:.3e})")
            print(f"      GT rank: {true_rank}/{len(perms)} (top {percentile:.2f}%) - {int(similarity*n_subunits)}/{n_subunits} pairs - z={z_true:+.2f}σ")
    
    # Find best method (highest ground truth rank)
    print(f"\n[3/5] Comparing methods and sources...")
    
    method_comparison = []
    for src_key in results_by_method.keys():
        for method, res in results_by_method[src_key].items():
            if res['true_rank']:
                method_comparison.append({
                    'source': src_key,
                    'source_name': res['source_name'],
                    'method': res['method_name'],
                    'rank': res['true_rank'],
                    'percentile': res['percentile'],
                    'similarity': res['similarity'],
                    'z_score': res['z_score']
                })
    
    method_comparison.sort(key=lambda x: x['rank'])
    
    print(f"\n  All combinations ranked by ground truth rank:")
    for i, mc in enumerate(method_comparison[:10]):  # Show top 10
        print(f"    {i+1:2d}. {mc['source_name']:30s} + {mc['method']:20s} - Rank {mc['rank']:4d} (top {mc['percentile']:5.2f}%) - {int(mc['similarity']*8)}/8 pairs - z={mc['z_score']:+.2f}σ")
    
    # Use average attention with sum method as default for main visualization
    default_source = 'average'
    default_method = 'sum'
    res = results_by_method[default_source][default_method]
    best_order = res['best_order']
    best_score = res['best_score']
    true_score = res['true_score']
    true_rank = res['true_rank']
    percentile = res['percentile']
    similarity = res['similarity']
    perms = res['all_perms']
    mean_score = res['mean_score']
    std_score = res['std_score']
    z_best = (best_score - mean_score) / std_score if std_score > 0 else 0
    z_true = res['z_score']
    
    print(f"\n[4/5] Statistical analysis (using {res['source_name']} + {res['method_name']} for plots)...")
    
    print(f"  Mean permutation score: {mean_score:.6e} ± {std_score:.6e}")
    print(f"  Best order z-score: {z_best:.2f}")
    print(f"  Ground truth z-score: {z_true:.2f}")
    
    # Get best head results if available
    best_head_res = None
    if best_head_df is not None and 'best_head' in results_by_method:
        best_head_res = results_by_method['best_head'][default_method]
        print(f"\n  Best Head (L{best_head_info['layer']}H{best_head_info['head']}) Performance:")
        print(f"    Ground truth rank: {best_head_res['true_rank']}/{len(best_head_res['all_perms'])} (top {best_head_res['percentile']:.2f}%)")
        print(f"    Similarity: {int(best_head_res['similarity']*8)}/8 pairs")
        print(f"    Z-score: {best_head_res['z_score']:+.2f}σ")
        
        # Compare to average
        avg_rank = res['true_rank']
        improvement = avg_rank - best_head_res['true_rank']
        print(f"    Improvement over average: {improvement:+d} ranks ({'better' if improvement > 0 else 'worse'})")
    
    # Visualization
    print(f"\n[5/5] Creating visualizations...")
    fig, axes = plt.subplots(3, 3, figsize=(20, 18))
    
    # Panel A: Score distribution
    ax = axes[0, 0]
    scores = [p['score'] for p in perms]
    ax.hist(scores, bins=50, alpha=0.7, color=COLOR_GRAY, density=True)
    ax.axvline(best_score, color=COLOR_RED, linewidth=2, linestyle='--', label=f'Best: {best_score:.4e}')
    ax.axvline(true_score, color=COLOR_BLUE, linewidth=2, linestyle='--', label=f'Ground truth: {true_score:.4e}')
    ax.axvline(mean_score, color=COLOR_GREEN, linewidth=1, linestyle=':', label=f'Mean: {mean_score:.4e}')
    ax.set_xlabel('Total Adjacent Pair Attention')
    ax.set_ylabel('Density')
    ax.set_title(f'Distribution of Ring Order Scores ({default_method})\n(n={len(perms)} permutations)', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Panel B: Cumulative distribution
    ax = axes[0, 1]
    sorted_scores = np.sort(scores)[::-1]  # Descending
    cumulative = np.arange(1, len(sorted_scores) + 1)
    ax.plot(sorted_scores, cumulative, color=COLOR_GRAY, linewidth=2)
    if true_rank:
        ax.axhline(true_rank, color=COLOR_BLUE, linewidth=2, linestyle='--', 
                   label=f'Ground truth rank: {true_rank}')
        ax.axvline(true_score, color=COLOR_BLUE, linewidth=2, linestyle='--')
    ax.set_xlabel('Score')
    ax.set_ylabel('Rank')
    ax.set_title('Rank vs Score', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Panel C: Source + Method comparison (top 10)
    ax = axes[0, 2]
    if method_comparison:
        top_10 = method_comparison[:10]
        labels = [f"{mc['source_name'].split()[0]} + {mc['method']}" for mc in top_10]
        ranks = [mc['rank'] for mc in top_10]
        colors = [COLOR_GREEN if r == 1 else COLOR_YELLOW if r <= 10 else COLOR_RED for r in ranks]
        
        bars = ax.barh(range(len(labels)), ranks, color=colors, alpha=0.7)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel('Ground Truth Rank')
        ax.set_title('Top 10 Source + Method Combinations\n(Lower rank = Better)', fontweight='bold')
        ax.axvline(1, color=COLOR_GREEN, linestyle=':', linewidth=2, alpha=0.5, label='Optimal')
        ax.grid(alpha=0.3, axis='x')
        ax.legend()
        ax.invert_yaxis()
    else:
        ax.axis('off')
        ax.text(0.5, 0.5, 'No method comparison data', ha='center', va='center')
    
    # Panel D: Top 10 orders
    ax = axes[1, 0]
    ax.axis('off')
    
    top_text = f"TOP 10 RING ORDERS ({default_method})\n" + "="*50 + "\n\n"
    for i, perm in enumerate(perms[:10]):
        marker = "★" if perm['order'] == RING_ORDER else " "
        marker = "✓" if i == 0 else marker
        marker = "✓" if i == 0 else marker
        top_text += f"{i+1:2d}. {marker} {perm['order_str']}\n"
        top_text += f"     Score: {perm['score']:.6e}\n"
        if i < 9:
            top_text += "\n"
    
    ax.text(0.05, 0.95, top_text, transform=ax.transAxes, fontsize=8,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # Panel E: Attention matrix ordered by best reconstruction (average)
    ax = axes[1, 1]
    # Use the attention source from the default
    src_attn = attention_df if default_source == 'average' else best_head_df
    best_ordered = src_attn.loc[best_order, best_order]
    im = ax.imshow(best_ordered.values, cmap='Reds', aspect='auto')
    n_subunits = len(best_order)
    ax.set_xticks(range(n_subunits))
    ax.set_yticks(range(n_subunits))
    ax.set_xticklabels(best_order, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(best_order, fontsize=8)
    ax.set_title(f'Attention Matrix (Average)\n(Ordered by Best {default_method})', fontweight='bold')
    plt.colorbar(im, ax=ax, label='Attention')
    
    # Highlight diagonal adjacency
    for i in range(n_subunits):
        j = (i + 1) % n_subunits
        ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=False,
                        edgecolor=COLOR_BLUE, linewidth=2))
        ax.add_patch(plt.Rectangle((i-0.5, j-0.5), 1, 1, fill=False,
                        edgecolor=COLOR_BLUE, linewidth=2))
    
    # Panel F: Attention matrix ordered by ground truth
    ax = axes[1, 2]
    true_ordered = attention_df.loc[ring_order_common, ring_order_common]
    im = ax.imshow(true_ordered.values, cmap='Reds', aspect='auto')
    n_subunits = len(ring_order_common)
    ax.set_xticks(range(n_subunits))
    ax.set_yticks(range(n_subunits))
    ax.set_xticklabels(ring_order_common, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(ring_order_common, fontsize=8)
    ax.set_title('Attention Matrix\n(Ordered by Ground Truth)', fontweight='bold')
    plt.colorbar(im, ax=ax, label='Attention')
    
    # Highlight diagonal adjacency
    for i in range(n_subunits):
        j = (i + 1) % n_subunits
        ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=False,
                        edgecolor=COLOR_BLUE, linewidth=2))
        ax.add_patch(plt.Rectangle((i-0.5, j-0.5), 1, 1, fill=False,
                        edgecolor=COLOR_BLUE, linewidth=2))
    
    # Panel G: Comparison of sources - ground truth ranks
    ax = axes[2, 0]
    if len(attention_sources) > 1:
        # Compare all sources for each method
        method_names = list(scoring_methods.keys())
        
        # Prepare data for grouped bar chart
        n_sources = len(attention_sources)
        source_keys = [src[0] for src in attention_sources]
        source_labels = [src[1].split()[0] for src in attention_sources]  # Shortened labels
        
        x = np.arange(len(method_names))
        width = 0.8 / n_sources
        
        colors_src = colorspal12
        
        for i, src_key in enumerate(source_keys):
            if src_key in results_by_method:
                ranks = [results_by_method[src_key][m]['true_rank'] for m in method_names]
                offset = (i - n_sources/2 + 0.5) * width
                ax.bar(x + offset, ranks, width, label=source_labels[i], alpha=0.8, color=colors_src[i % len(colors_src)])
        
        ax.set_ylabel('Ground Truth Rank')
        ax.set_xlabel('Scoring Method')
        ax.set_title('Performance by Source and Method\n(Lower = Better)', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([scoring_methods[m][0] for m in method_names], rotation=45, ha='right', fontsize=7)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3, axis='y')
        ax.axhline(1, color=COLOR_GREEN, linestyle=':', linewidth=1, alpha=0.5)
    else:
        # Just show average results
        avg_results = results_by_method['average']
        method_names = list(scoring_methods.keys())
        percentiles = [avg_results[m]['percentile'] for m in method_names]
        colors_bar = ['green' if p < 5 else 'orange' if p < 20 else 'red' for p in percentiles]
        
        bars = ax.bar(range(len(method_names)), percentiles, color=colors_bar, alpha=0.7)
        ax.set_xticks(range(len(method_names)))
        ax.set_xticklabels([scoring_methods[m][0] for m in method_names], rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Percentile Rank (%)')
        ax.set_title('Ground Truth Percentile by Scoring Method\n(Lower = Better)', fontweight='bold')
        ax.axhline(1, color=COLOR_GREEN, linestyle=':', linewidth=2, alpha=0.5, label='Top 1%')
        ax.axhline(5, color=COLOR_YELLOW, linestyle=':', linewidth=2, alpha=0.5, label='Top 5%')
        ax.grid(alpha=0.3, axis='y')
        ax.legend()
    
    # Panel H: Summary for default method
    ax = axes[2, 1]
    ax.axis('off')
    
    n_subunits = len(ring_order_common)
    mid_point = (n_subunits + 1) // 2
    
    summary_text = f"""
RING RECONSTRUCTION ({default_method.upper()})
{'='*50}

SEARCH SPACE
  Total permutations: {len(perms)}

BEST RECONSTRUCTION
  Order: {' → '.join(best_order[:mid_point])}
         {' → '.join(best_order[mid_point:])}
  Score: {best_score:.6e}
  Z-score: {z_best:.2f}σ

GROUND TRUTH (Kalisman 2012)
  Order: {' → '.join(ring_order_common[:mid_point])}
         {' → '.join(ring_order_common[mid_point:])}
  Score: {true_score:.6e}
  Rank: {true_rank}/{len(perms)} (top {percentile:.2f}%)
  Z-score: {z_true:.2f}σ
  Score ratio: {true_score/best_score:.4f}

SIMILARITY
  Matching pairs: {int(similarity * n_subunits)}/{n_subunits}
  Similarity: {similarity*100:.1f}%

INTERPRETATION
  {'✓ GT is OPTIMAL' if true_rank == 1 else f'✗ GT is rank {true_rank}'}
  {'✓ Strong signal' if similarity >= 0.75 else '~ Partial' if similarity >= 0.5 else '✗ Weak signal'}
"""
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Panel I: Summary of all methods
    ax = axes[2, 2]
    ax.axis('off')
    
    all_methods_text = "SOURCE + METHOD SUMMARY\n" + "="*50 + "\n\n"
    
    for src_key in results_by_method.keys():
        src_name = results_by_method[src_key][list(scoring_methods.keys())[0]]['source_name']
        
        # Add marker for baseline
        if 'baseline' in src_key.lower() or 'esm' in src_key.lower():
            all_methods_text += f"{src_name} [BASELINE]:\n"
        else:
            all_methods_text += f"{src_name}:\n"
        
        for method in scoring_methods.keys():
            res_item = results_by_method[src_key][method]
            all_methods_text += f"  {res_item['method_name']:12s}: "
            all_methods_text += f"R{res_item['true_rank']:3d} "
            all_methods_text += f"({res_item['percentile']:4.1f}%) "
            all_methods_text += f"{int(res_item['similarity']*n_subunits)}/{n_subunits} "
            all_methods_text += f"z={res_item['z_score']:+.1f}\n"
        all_methods_text += "\n"
    
    # Add best combination and baseline comparison
    if method_comparison:
        best = method_comparison[0]
        all_methods_text += f"BEST COMBINATION:\n"
        all_methods_text += f"  {best['source_name']}\n"
        all_methods_text += f"  + {best['method']}\n"
        all_methods_text += f"  Rank: {best['rank']} (top {best['percentile']:.2f}%)\n"
        
        # Compare to ESM baseline if available
        if esm_sim_df is not None and 'esm_baseline' in results_by_method:
            esm_best_rank = min(results_by_method['esm_baseline'][m]['true_rank'] 
                               for m in scoring_methods.keys())
            improvement = esm_best_rank - best['rank']
            all_methods_text += f"\nVS ESM BASELINE:\n"
            all_methods_text += f"  ESM best rank: {esm_best_rank}\n"
            all_methods_text += f"  ProteomeLM best: {best['rank']}\n"
            all_methods_text += f"  Improvement: {improvement:+d} ranks\n"
            all_methods_text += f"  {'✓ Learns structure!' if improvement > 0 else '✗ No improvement'}\n"
    
        ax.text(0.05, 0.95, all_methods_text, transform=ax.transAxes, fontsize=7,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor=COLOR_CYAN, alpha=0.2))
    
    plt.tight_layout()
    save_figure(fig, output_dir, "cct_ring_reconstruction")

    print(f"\n  Figure saved: {os.path.join(output_dir, 'cct_ring_reconstruction.pdf')}")
    
    # Save results
    results = {
        'attention_sources': [src[1] for src in attention_sources],
        'scoring_methods': {
            src_key: {
                method: {
                    'method_name': res['method_name'],
                    'source_name': res['source_name'],
                    'best_order': res['best_order'],
                    'best_score': float(res['best_score']),
                    'ground_truth_score': float(res['true_score']),
                    'ground_truth_rank': res['true_rank'],
                    'ground_truth_percentile': float(res['percentile']) if res['percentile'] else None,
                    'similarity': float(res['similarity']),
                    'z_score': float(res['z_score']),
                    'score_ratio': float(res['true_score'] / res['best_score'])
                }
                for method, res in results_by_method[src_key].items()
            }
            for src_key in results_by_method.keys()
        },
        'best_combination': {
            'source': method_comparison[0]['source_name'],
            'method': method_comparison[0]['method'],
            'rank': method_comparison[0]['rank'],
            'percentile': method_comparison[0]['percentile']
        } if method_comparison else None,
        'best_head_info': best_head_info,
        'esm_baseline_included': esm_sim_df is not None,
        'default_method_used': f"{default_source} + {default_method}",
        'ground_truth': {
            'order': RING_ORDER,
        }
    }
    
    # Add ESM baseline comparison if available
    if esm_sim_df is not None and 'esm_baseline' in results_by_method:
        esm_best_rank = min(results_by_method['esm_baseline'][m]['true_rank'] 
                           for m in scoring_methods.keys())
        proteomelm_best_rank = method_comparison[0]['rank'] if method_comparison else None
        
        results['esm_baseline_comparison'] = {
            'esm_best_rank': esm_best_rank,
            'proteomelm_best_rank': proteomelm_best_rank,
            'improvement': esm_best_rank - proteomelm_best_rank if proteomelm_best_rank else None,
            'proteomelm_outperforms': (esm_best_rank > proteomelm_best_rank) if proteomelm_best_rank else False
        }
    
    with open(os.path.join(output_dir, "cct_ring_reconstruction_results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"  Results saved: {output_dir}/cct_ring_reconstruction_results.json")
    print("\n" + "="*70)
    
    return results


def find_best_ring_order(attention_matrix: pd.DataFrame):
    """
    Find the ring order that maximizes total attention between adjacent pairs.
    Uses brute force over all 7!/2 = 2520 unique circular permutations.
    
    Returns:
        best_order: list of subunits in best order
        best_score: total attention for adjacent pairs
        all_scores: dict of all permutations and scores
    """
    from itertools import permutations
    
    subunits = list(attention_matrix.index)
    
    # Fix first element to avoid counting rotations multiple times
    fixed = subunits[0]
    others = subunits[1:]
    
    best_order = None
    best_score = -np.inf
    all_scores = {}
    
    for perm in permutations(others):
        order = [fixed] + list(perm)
        
        # Calculate score (sum of attention for adjacent pairs)
        score = 0
        for i in range(len(order)):
            a = order[i]
            b = order[(i + 1) % len(order)]
            # Use symmetric attention
            score += (attention_matrix.loc[a, b] + attention_matrix.loc[b, a]) / 2
        
        # Store (use tuple for hashing, normalized to start with min element)
        # Also check reverse to avoid double counting
        order_key = tuple(order)
        all_scores[order_key] = score
        
        if score > best_score:
            best_score = score
            best_order = order
    
    return best_order, best_score, all_scores


def ring_similarity(order1, order2):
    """
    Calculate similarity between two ring orders.
    Returns fraction of adjacent pairs that match (0-1).
    """
    def get_adj_set(order):
        pairs = set()
        for i in range(len(order)):
            a = order[i]
            b = order[(i + 1) % len(order)]
            pairs.add(tuple(sorted([a, b])))
        return pairs
    
    adj1 = get_adj_set(order1)
    adj2 = get_adj_set(order2)
    
    return len(adj1 & adj2) / len(adj1)


def evaluate_predictions(attention_df: pd.DataFrame, output_dir: str, suffix: str = ""):
    """
    Evaluate ProteomeLM attention against ring ground truth.
    
    Args:
        attention_df: Attention matrix (8x8 for CCT subunits)
        output_dir: Directory to save results
        suffix: Optional suffix for output files
    """
    import matplotlib.pyplot as plt
    from scipy import stats
    
    print("\n" + "="*70)
    print(f"TRiC/CCT VALIDATION: EVALUATION{suffix.upper()}")
    print("="*70)
    
    subunits = list(CCT_SUBUNITS.keys())
    
    # Ensure we have all subunits
    missing = set(subunits) - set(attention_df.index)
    if missing:
        print(f"Warning: Missing subunits in attention matrix: {missing}")
        return
    
    # Reorder to match our subunit list
    attention_df = attention_df.loc[subunits, subunits]
    
    # 1. Compare adjacent vs non-adjacent attention
    print("\n[1/4] Adjacent vs Non-adjacent Attention")
    
    adj_scores = []
    non_adj_scores = []
    
    for pair in ALL_PAIRS:
        a, b = pair
        score = (attention_df.loc[a, b] + attention_df.loc[b, a]) / 2
        
        if pair in ADJACENT_PAIRS:
            adj_scores.append(score)
        else:
            non_adj_scores.append(score)
    
    adj_mean = np.mean(adj_scores)
    non_adj_mean = np.mean(non_adj_scores)
    
    # Statistical test
    stat, pval = stats.mannwhitneyu(adj_scores, non_adj_scores, alternative='greater')
    
    print(f"  Adjacent pairs (n=8):     mean = {adj_mean:.6f}")
    print(f"  Non-adjacent pairs (n=20): mean = {non_adj_mean:.6f}")
    print(f"  Ratio: {adj_mean/non_adj_mean:.2f}x")
    print(f"  Mann-Whitney U test (adjacent > non-adjacent): p = {pval:.4e}")
    
    # 2. Correlation with ring distance
    print("\n[2/4] Correlation with Ring Distance")
    
    distances = []
    scores = []
    
    for pair in ALL_PAIRS:
        a, b = pair
        score = (attention_df.loc[a, b] + attention_df.loc[b, a]) / 2
        dist = get_ring_distance(a, b)
        
        distances.append(dist)
        scores.append(score)
    
    # Negative correlation expected (closer = higher attention)
    rho, p_corr = stats.spearmanr(distances, scores)
    print(f"  Spearman correlation (distance vs attention): ρ = {rho:.4f} (p = {p_corr:.4e})")
    print(f"  Expected: negative (closer subunits → higher attention)")
    
    # 3. Ring reconstruction
    print("\n[3/4] Ring Order Reconstruction")
    
    best_order, best_score, all_scores = find_best_ring_order(attention_df)
    
    # Get score for true order
    true_score = 0
    for i in range(len(RING_ORDER)):
        a = RING_ORDER[i]
        b = RING_ORDER[(i + 1) % len(RING_ORDER)]
        true_score += (attention_df.loc[a, b] + attention_df.loc[b, a]) / 2
    
    similarity = ring_similarity(best_order, RING_ORDER)
    
    print(f"  True ring order:  {' → '.join(RING_ORDER)}")
    print(f"  Best order found: {' → '.join(best_order)}")
    print(f"  True order score: {true_score:.6f}")
    print(f"  Best order score: {best_score:.6f}")
    print(f"  Adjacent pairs recovered: {int(similarity * 8)}/8 ({similarity*100:.1f}%)")
    
    # Rank of true order among all permutations
    sorted_scores = sorted(all_scores.values(), reverse=True)
    true_rank = sorted_scores.index(true_score) + 1 if true_score in sorted_scores else None
    
    # Count how many permutations score >= true
    n_better_or_equal = sum(1 for s in sorted_scores if s >= true_score)
    percentile = (1 - n_better_or_equal / len(sorted_scores)) * 100
    
    print(f"  Rank of true order: {true_rank}/{len(sorted_scores)} (top {100-percentile:.1f}%)")
    
    # 4. Permutation test
    print("\n[4/4] Permutation Test (Adjacent vs Non-adjacent)")
    
    n_perms = 10000
    observed_diff = adj_mean - non_adj_mean
    all_scores_flat = adj_scores + non_adj_scores
    
    null_diffs = []
    for _ in range(n_perms):
        np.random.shuffle(all_scores_flat)
        perm_adj = np.mean(all_scores_flat[:8])
        perm_non = np.mean(all_scores_flat[8:])
        null_diffs.append(perm_adj - perm_non)
    
    p_perm = np.mean([d >= observed_diff for d in null_diffs])
    
    print(f"  Observed difference: {observed_diff:.6f}")
    print(f"  Null distribution mean: {np.mean(null_diffs):.6f}")
    print(f"  P-value (permutation): {p_perm:.4f}")
    print(f"  Significant (p < 0.05): {p_perm < 0.05}")
    
    # ==========================================================================
    # VISUALIZATION
    # ==========================================================================
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Panel A: Attention matrix
    ax = axes[0, 0]
    # Reorder by ring order for visualization
    ring_attention = attention_df.loc[RING_ORDER, RING_ORDER]
    im = ax.imshow(ring_attention.values, cmap='Reds', aspect='auto')
    ax.set_xticks(range(8))
    ax.set_yticks(range(8))
    ax.set_xticklabels(RING_ORDER, rotation=45, ha='right')
    ax.set_yticklabels(RING_ORDER)
    ax.set_title('Attention Matrix\n(ordered by ring)', fontweight='bold')
    plt.colorbar(im, ax=ax, label='Attention')
    
    # Highlight adjacent pairs
    for i in range(8):
        j = (i + 1) % 8
        ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=False,
                        edgecolor=COLOR_BLUE, linewidth=2))
        ax.add_patch(plt.Rectangle((i-0.5, j-0.5), 1, 1, fill=False,
                        edgecolor=COLOR_BLUE, linewidth=2))
    
    # Panel B: Adjacent vs Non-adjacent distribution
    ax = axes[0, 1]
    ax.hist(adj_scores, bins=8, alpha=0.7, label=f'Adjacent (n=8)', color=COLOR_BLUE, density=True)
    ax.hist(non_adj_scores, bins=10, alpha=0.7, label=f'Non-adjacent (n=20)', color=COLOR_RED, density=True)
    ax.axvline(adj_mean, color=COLOR_BLUE, linestyle='--', linewidth=2)
    ax.axvline(non_adj_mean, color=COLOR_RED, linestyle='--', linewidth=2)
    ax.set_xlabel('Attention Score')
    ax.set_ylabel('Density')
    ax.set_title(f'Adjacent vs Non-adjacent\np = {pval:.4e}', fontweight='bold')
    ax.legend()
    
    # Panel C: Score vs Ring Distance
    ax = axes[0, 2]
    ax.scatter(distances, scores, alpha=0.7, s=100, color=COLOR_CYAN)
    
    # Add regression line
    z = np.polyfit(distances, scores, 1)
    p = np.poly1d(z)
    x_line = np.linspace(1, 4, 100)
    ax.plot(x_line, p(x_line), color=COLOR_RED, linestyle='--', linewidth=2)
    
    ax.set_xlabel('Ring Distance')
    ax.set_ylabel('Attention Score')
    ax.set_title(f'Attention vs Ring Distance\nρ = {rho:.3f} (p = {p_corr:.3e})', fontweight='bold')
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(['1\n(adjacent)', '2', '3', '4\n(opposite)'])
    
    # Panel D: Ring diagram with attention
    ax = axes[1, 0]
    ax.set_aspect('equal')
    
    # Draw ring
    angles = np.linspace(0, 2*np.pi, 9)[:-1]  # 8 positions
    radius = 1
    positions = {RING_ORDER[i]: (radius * np.cos(angles[i] - np.pi/2), 
                                  radius * np.sin(angles[i] - np.pi/2)) 
                 for i in range(8)}
    
    # Draw edges colored by attention
    max_att = max(adj_scores)
    min_att = min(adj_scores)
    
    for i in range(8):
        a = RING_ORDER[i]
        b = RING_ORDER[(i + 1) % 8]
        att = (attention_df.loc[a, b] + attention_df.loc[b, a]) / 2
        
        # Normalize attention to color
        norm_att = (att - min_att) / (max_att - min_att) if max_att > min_att else 0.5
        
        x = [positions[a][0], positions[b][0]]
        y = [positions[a][1], positions[b][1]]
        ax.plot(x, y, linewidth=3 + 5*norm_att, color=plt.cm.Blues(0.3 + 0.7*norm_att))
    
    # Draw nodes
    for subunit, (x, y) in positions.items():
        ax.scatter(x, y, s=800, c='lightblue', edgecolors='black', linewidth=2, zorder=10)
        ax.text(x, y, subunit.replace('CCT', ''), ha='center', va='center', 
                fontweight='bold', fontsize=10, zorder=11)
    
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.axis('off')
    ax.set_title('Ring Topology\n(edge width = attention)', fontweight='bold')
    
    # Panel E: Permutation test
    ax = axes[1, 1]
    ax.hist(null_diffs, bins=50, alpha=0.7, color=COLOR_GRAY, density=True)
    ax.axvline(observed_diff, color=COLOR_RED, linestyle='--', linewidth=2, 
               label=f'Observed = {observed_diff:.4f}')
    ax.axvline(np.mean(null_diffs), color=COLOR_BLUE, linestyle=':', linewidth=2,
               label=f'Null mean = {np.mean(null_diffs):.4f}')
    ax.set_xlabel('Mean Difference (Adjacent - Non-adjacent)')
    ax.set_ylabel('Density')
    ax.set_title(f'Permutation Test (n={n_perms})\np = {p_perm:.4f}', fontweight='bold')
    ax.legend()
    
    # Panel F: Summary statistics
    ax = axes[1, 2]
    ax.axis('off')
    
    summary_text = f"""
TRiC/CCT Ring Ordering Validation
{'='*40}

GROUND TRUTH
  Ring order: CCT2→CCT4→CCT1→CCT3→CCT6→CCT8→CCT7→CCT5
  Reference: Kalisman et al. 2012 PNAS

RESULTS
  Adjacent pairs: {adj_mean:.6f} (mean attention)
  Non-adjacent:   {non_adj_mean:.6f} (mean attention)
  Ratio:          {adj_mean/non_adj_mean:.2f}×
  
  Mann-Whitney U: p = {pval:.2e}
  Spearman ρ:     {rho:.3f} (p = {p_corr:.2e})
  
  Ring reconstruction:
    Adjacent pairs recovered: {int(similarity * 8)}/8
    True order rank: {true_rank}/{len(sorted_scores)}
  
  Permutation test: p = {p_perm:.4f}
  
INTERPRETATION
  {'✓ SIGNIFICANT' if p_perm < 0.05 else '✗ NOT SIGNIFICANT'}
  {'ProteomeLM captures ring topology' if p_perm < 0.05 else 'Ring topology not captured'}
"""
    ax.text(0.1, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    save_figure(fig, output_dir, f"tric_validation_results{suffix}")

    print(f"\n  Figure saved: {os.path.join(output_dir, f'tric_validation_results{suffix}.pdf')}")
    
    # Save metrics
    metrics = {
        'adjacent_mean': float(adj_mean),
        'non_adjacent_mean': float(non_adj_mean),
        'ratio': float(adj_mean / non_adj_mean),
        'mannwhitney_pvalue': float(pval),
        'spearman_rho': float(rho),
        'spearman_pvalue': float(p_corr),
        'ring_similarity': float(similarity),
        'adjacent_pairs_recovered': int(similarity * 8),
        'true_order_rank': true_rank,
        'total_permutations': len(sorted_scores),
        'permutation_pvalue': float(p_perm),
        'best_order': best_order,
        'true_order': RING_ORDER,
        'significant': bool(p_perm < 0.05)
    }
    
    with open(os.path.join(output_dir, f"metrics{suffix}.json"), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return metrics


def load_attention_from_proteomelm(predictions_file: str) -> pd.DataFrame:
    """
    Load attention matrix from ProteomeLM output.
    
    Expected format (JSON):
    [
        {"subunit_a": "CCT1", "subunit_b": "CCT2", "attention": 0.0035},
        ...
    ]
    
    Or as an 8x8 matrix CSV with CCT1-CCT8 as row/column names.
    """
    if predictions_file.endswith('.json'):
        with open(predictions_file) as f:
            preds = json.load(f)
        
        subunits = list(CCT_SUBUNITS.keys())
        matrix = pd.DataFrame(0.0, index=subunits, columns=subunits)
        
        for p in preds:
            a = p.get('subunit_a') or p.get('protein_a')
            b = p.get('subunit_b') or p.get('protein_b')
            score = p.get('attention') or p.get('score')
            
            if a in subunits and b in subunits and score is not None:
                matrix.loc[a, b] = score
                matrix.loc[b, a] = score
        
        return matrix
    
    elif predictions_file.endswith('.csv'):
        return pd.read_csv(predictions_file, index_col=0)
    
    else:
        raise ValueError(f"Unknown format: {predictions_file}")


# =============================================================================
# COMMAND FUNCTIONS
# =============================================================================

def create_ring_only_fasta(output_dir: str):
    """Create a FASTA file with only the 8 CCT ring subunits from UniProt."""
    import requests
    from time import sleep
    
    print("\n[2/3] Fetching CCT sequences from UniProt...")
    proteome_path = os.path.join(output_dir, "yeast_proteome.fasta")
    
    sequences = {}
    subunit_order = []  # Track order of subunits in FASTA
    
    for subunit, info in CCT_SUBUNITS.items():
        uniprot_id = info['uniprot']
        gene = info['gene']
        
        print(f"  Fetching {subunit} ({uniprot_id})...")
        
        # Fetch sequence from UniProt
        url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            # Parse FASTA response
            lines = response.text.strip().split('\n')
            header = lines[0]
            sequence = ''.join(lines[1:])
            
            # Store with simplified header including subunit name
            sequences[subunit] = {
                'uniprot_id': uniprot_id,
                'header': f"sp|{uniprot_id}|{gene}_YEAST {subunit}",
                'sequence': sequence,
                'gene': gene
            }
            subunit_order.append(subunit)
            
            sleep(0.1)  # Be nice to UniProt API
            
        except Exception as e:
            print(f"    Warning: Could not fetch {uniprot_id}: {e}")
            continue
    
    # Write FASTA file
    with open(proteome_path, 'w') as f:
        for subunit in subunit_order:
            data = sequences[subunit]
            f.write(f">{data['header']}\n")
            f.write(f"{data['sequence']}\n")
    
    print(f"  Created ring-only FASTA with {len(sequences)} sequences")
    print(f"  Saved to: {proteome_path}")
    
    # Create mapping for ring-only mode
    mapping = {}
    for idx, subunit in enumerate(subunit_order):
        mapping[subunit] = {
            'proteome_idx': idx,
            'uniprot': sequences[subunit]['uniprot_id'],
            'gene': sequences[subunit]['gene'],
            'header': sequences[subunit]['header']
        }
    
    # Save mapping
    mapping_path = os.path.join(output_dir, "cct_proteome_mapping.json")
    save_json(mapping, mapping_path)
    print(f"  Created ring-only mapping with {len(mapping)} proteins")
    
    return proteome_path, sequences


def cmd_prepare(output_dir: str, ring_only: bool = False):
    """Prepare data and download human proteome or create ring-only FASTA."""
    
    print("="*70)
    print("TRiC/CCT VALIDATION - PREPARE")
    print("="*70)
    
    # Prepare ground truth
    print("\n[1/3] Preparing ground truth...")
    prepare_data(output_dir)
    
    if ring_only:
        # Create FASTA with only the 8 CCT ring subunits
        print("\n[MODE: Ring-only (8 CCT subunits)]")
        proteome_path, sequences = create_ring_only_fasta(output_dir)
        # Mapping already created in create_ring_only_fasta
        print("\n[3/3] CCT subunits located (ring-only mode)")
    else:
        # Download full yeast proteome using shared utility
        print("\n[MODE: Full proteome]")
        print("\n[2/3] Downloading yeast proteome...")
        proteome_path = download_proteome(
            proteome_id=CONFIG['proteome_id'],
            output_dir=output_dir,
            organism_name='yeast',
            reviewed_only=True,
            include_isoforms=False
        )
        proteome_seqs = parse_fasta(proteome_path)
        print(f"  Proteome size: {len(proteome_seqs)} proteins")
        
        # Find CCT subunits
        print("\n[3/3] Locating CCT subunits in proteome...")
        cct_info = find_cct_in_proteome_local(proteome_seqs, output_dir)
    
    print("\n" + "="*70)
    print("READY - Run: python cct.py --predict --model <path_to_proteomelm>")
    print("="*70)


def cmd_predict(output_dir: str, model_path: str = None, orthodb_config: dict = None):
    """Run ProteomeLM predictions."""
    
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
    
    # Create OrthoDB group embeddings if config provided
    group_embeds = None
    if orthodb_config is not None and (orthodb_config.get('group_vectors_path') or orthodb_config.get('orthodb_db_path')):
        print("\n[1b/5] Creating OrthoDB group embeddings...")
        try:
            group_embeds = create_orthodb_group_embeddings(
                fasta_path=proteome_path,
                output_dir=output_dir,
                orthodb_config=orthodb_config,
                esm_embeddings=esm_embeddings,
                prefix="yeast",
                use_online=False
            )
        except Exception as e:
            print(f"  Warning: Failed to create OrthoDB group embeddings: {e}")
            print("  Falling back to default (ESM as group embeddings)")
            group_embeds = None
    
    # Compute ProteomeLM embeddings and attentions
    print("\n[2/5] Computing ProteomeLM embeddings and attentions...")
    proteomelm_embeddings, _ = compute_proteomelm_embeddings(
        esm_embeddings=esm_embeddings,
        output_dir=output_dir,
        model_path=model_path,
        prefix="yeast",
        device=CONFIG['proteome_device'],
        group_embeds=group_embeds
    )
    
    # Extract CCT attention matrix
    print("\n[3/5] Extracting CCT attention matrix...")
    attention_df = extract_cct_attention_matrix(output_dir)
    
    # Run attention head analysis
    print("\n[4/6] Analyzing attention heads...")
    analyze_attention_heads(output_dir)
    analyze_ring_distance_correlation(output_dir)
    
    # CCT vs background analysis
    print("\n[5/6] Analyzing CCT vs background...")
    analyze_cct_vs_background(output_dir)
    
    # CCT membership analysis (intra vs inter-complex)
    print("\n[5b/6] Analyzing CCT complex membership (intra vs inter)...")
    analyze_cct_membership(output_dir)
    
    # Ring reconstruction analysis
    print("\n[6/6] Ring order reconstruction...")
    reconstruct_ring_order_analysis(output_dir, orthodb_config=orthodb_config)
    
    print("\n" + "="*70)
    print("PREDICTIONS SAVED")
    print(f"  CCT attention matrix: {output_dir}/cct_attention_matrix.csv")
    print(f"  Attention head analysis: {output_dir}/cct_attention_head_analysis.png")
    print(f"  Distance correlation: {output_dir}/cct_distance_correlation.png")
    print(f"  CCT vs background: {output_dir}/cct_vs_background_analysis.png")
    print(f"  CCT membership: {output_dir}/cct_membership_analysis.png")
    
    # Compose supplementary figure
    print("\n[9/9] Composing supplementary figure...")
    compose_cct_supplementary_figure(output_dir)
    
    print("\nEvaluate with:")
    print(f"  python cct.py --evaluate {output_dir}/cct_attention_matrix.csv")
    print("  or")
    print(f"  python cct.py --evaluate  # uses auto-generated matrix")
    print("="*70)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='TRiC/CCT ring ordering validation for ProteomeLM',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Prepare with full yeast proteome
  python %(prog)s --prepare
  
  # Prepare with only the 8 CCT ring subunits (faster, less memory)
  python %(prog)s --prepare --ring-only
  
  # After running ProteomeLM, evaluate
  python %(prog)s --evaluate attention_matrix.csv
  
  # Evaluate with OrthoDB functional embeddings (using local files)
  python %(prog)s --evaluate auto \\
      --orthodb-group-vectors /path/to/group_vectors_50.pkl,/path/to/group_vectors_200.pkl \\
      --orthodb-uniprot-mapping /path/to/uniprot_to_og.pkl
  
  # Evaluate with OrthoDB using online API (no local files needed)
  python %(prog)s --evaluate auto --orthodb-fetch-online
  
  # Demo with random data
  python %(prog)s --demo
"""
    )
    
    parser.add_argument('--prepare', '-p', action='store_true',
                        help='Prepare ground truth data and download human proteome')
    parser.add_argument('--predict', action='store_true',
                        help='Run ProteomeLM predictions on human proteome')
    parser.add_argument('--evaluate', '-e', type=str, metavar='FILE', nargs='?', const='auto',
                        help='Evaluate predictions file (or use auto-generated from --predict)')
    parser.add_argument('--demo', action='store_true',
                        help='Run demo with synthetic data')
    parser.add_argument('--ring-only', action='store_true',
                        help='Work with only the 8 CCT ring subunits instead of full proteome (use with --prepare)')
    parser.add_argument('--output', '-o', type=str, default='./tric_validation',
                        help='Output directory')
    parser.add_argument('--model', '-m', type=str, default=None,
                        help='Path to ProteomeLM model')
    
    # OrthoDB functional embedding arguments
    parser.add_argument('--orthodb-group-vectors', type=str, default=None,
                        help='Path to OrthoDB group vectors file or directory')
    parser.add_argument('--orthodb-db-path', type=str, default=None,
                        help='Directory with group_vectors_*.pkl files')
    parser.add_argument('--orthodb-tsv', type=str, default=None,
                        help='OrthoDB TSV from download_proteome (optional)')
    parser.add_argument('--orthodb-min-group-size', type=int, default=10,
                        help='Minimum OrthoDB group size (default: 10)')
    
    args = parser.parse_args()
    
    CONFIG['output_dir'] = args.output
    CONFIG['proteomelm_model'] = args.model
    ensure_dir(args.output)
    
    # Build OrthoDB config if paths provided
    orthodb_config = None
    if args.orthodb_group_vectors is not None or args.orthodb_db_path is not None:
        orthodb_config = {
            'group_vectors_path': args.orthodb_group_vectors,
            'orthodb_db_path': args.orthodb_db_path,
            'orthodb_tsv_path': args.orthodb_tsv,
            'min_group_size': args.orthodb_min_group_size,
        }
    
    if args.prepare:
        cmd_prepare(args.output, ring_only=args.ring_only)
        
    elif args.predict:
        cmd_predict(args.output, args.model, orthodb_config=orthodb_config)
        
    elif args.evaluate:
        if args.evaluate == 'auto':
            # Use the auto-generated attention matrix
            attn_file = os.path.join(args.output, "cct_attention_matrix.csv")
            if not os.path.exists(attn_file):
                print(f"Error: {attn_file} not found. Run --predict first.")
                sys.exit(1)
            args.evaluate = attn_file
        
        attention_df = load_attention_from_proteomelm(args.evaluate)
        
        # Run ring ordering evaluation
        evaluate_predictions(attention_df, args.output)
        
        # Ring reconstruction analysis
        print("\n" + "="*70)
        print("Additional Analysis: Ring Order Reconstruction")
        print("="*70)
        reconstruct_ring_order_analysis(args.output, orthodb_config=orthodb_config)
        
        # Also run CCT vs background analysis if attention file exists
        attn_path = os.path.join(args.output, "yeast_proteomelm_attentions.pt")
        if os.path.exists(attn_path):
            print("\n" + "="*70)
            print("Additional Analysis: CCT vs Background")
            print("="*70)
            analyze_cct_vs_background(args.output)
        else:
            print("\n  Note: Run --predict to get full CCT vs background analysis")
        
    elif args.demo:
        print("Running demo with synthetic data...")
        prepare_data(args.output)
        
        # Create synthetic attention that partially captures ring structure
        subunits = list(CCT_SUBUNITS.keys())
        n = len(subunits)
        
        # Base: random attention
        np.random.seed(42)
        attention = np.random.uniform(0.001, 0.003, (n, n))
        
        # Add signal: adjacent pairs get higher attention
        for i, a in enumerate(subunits):
            for j, b in enumerate(subunits):
                if i != j:
                    pair = tuple(sorted([a, b]))
                    dist = get_ring_distance(a, b)
                    # Higher attention for closer pairs
                    attention[i, j] += 0.002 * (5 - dist) / 4
        
        # Symmetrize
        attention = (attention + attention.T) / 2
        np.fill_diagonal(attention, 0)
        
        attention_df = pd.DataFrame(attention, index=subunits, columns=subunits)
        attention_df.to_csv(os.path.join(args.output, "demo_attention.csv"))
        
        print("\nEvaluating synthetic data with ring signal...")
        evaluate_predictions(attention_df, args.output)
        
    else:
        parser.print_help()
        print("\n⚠️  Specify --prepare, --predict, --evaluate, or --demo")


if __name__ == "__main__":
    main()