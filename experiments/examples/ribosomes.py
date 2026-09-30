#!/usr/bin/env python3
"""
E. coli Ribosome Validation & Figure Generation for ProteomeLM
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MultipleLocator
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr, linregress
from sklearn.metrics import roc_curve, auc, average_precision_score, precision_recall_curve
import torch
import networkx as nx

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from proteomelm.utils.io import ensure_dir, parse_fasta, download_pdb_structure
from proteomelm.utils.embedding import compute_esm_embeddings, compute_proteomelm_embeddings, load_attentions
from proteomelm.utils.proteome import download_proteome
from experiments.examples.common import (
    COLOR_BLUE, COLOR_CYAN, COLOR_GRAY, COLOR_GREEN, COLOR_RED,
    apply_plot_style, build_orthodb_group_embeds, pair_attn, save_figure, style_axes, to_colormap,
)

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION & MAPPING
# =============================================================================

CONFIG = {
    'pdb_id': '7K00',
    'proteome_id': 'UP000000625',
    'contact_threshold': 8.0, 
    'output_dir': './ribosome_validation',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}

apply_plot_style(22)

# Hardcoded mapping for 7K00 -> UniProt (E. coli K-12)
# Essential for mapping structure chains to proteome indices
RIBOSOMAL_MAP = {
    # Small Subunit (30S)
    'B': 'P0A7V0', 'C': 'P0A7V3', 'D': 'P0A7V8', 'E': 'P0A7W1', 'F': 'P02358',
    'G': 'P02359', 'H': 'P0A7W7', 'I': 'P0A7X3', 'J': 'P0A7R5', 'K': 'P0A7R9',
    'L': 'P0A7S3', 'M': 'P0A7S9', 'N': 'P0AG59', 'O': 'P0ADZ4', 'P': 'P0A7T3',
    'Q': 'P0AG63', 'R': 'P0A7T7', 'S': 'P0A7U3', 'T': 'P0A7U7', 'U': 'P68679',
    # Large Subunit (50S)
    'X': 'P60422', 'Y': 'P60438', 'Z': 'P60723', 'a': 'P62399', 'b': 'P0AG55',
    'c': 'P0A7R1', 'd': 'P0AA10', 'e': 'P0ADY3', 'f': 'P02413', 'g': 'P0ADY7',
    'h': 'P0AG44', 'i': 'P0C018', 'j': 'P0A7K6', 'k': 'P0A7L3', 'l': 'P0AG48',
    'm': 'P61175', 'n': 'P0ADZ0', 'o': 'P60624', 'p': 'P68919', 'q': 'P0A7L8',
    'r': 'P0A7M2', 's': 'P0A7M6', 't': 'P0AG51', 'u': 'P0A7N4', 'v': 'P0A7N9',
    'w': 'P0A7P5', 'x': 'P0A7Q1', 'y': 'P0A7Q6', 'z': 'P0A7M9'
}

# =============================================================================
# DATA PREPARATION
# =============================================================================

def parse_structure(cif_path):
    """Extracts CA coordinates from PDB/CIF."""
    from Bio.PDB import MMCIFParser, is_aa
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("ribosome", cif_path)
    
    chain_coords = {}
    for model in structure:
        for chain in model:
            cid = chain.get_id()
            if cid not in RIBOSOMAL_MAP: continue
            
            residues = [r for r in chain if is_aa(r, standard=True) and 'CA' in r]
            if len(residues) > 20:
                coords = np.array([r['CA'].get_coord() for r in residues])
                chain_coords[cid] = coords
        break 
    return chain_coords

def compute_ground_truth(chain_coords):
    """Computes binary contact matrix (<8A)."""
    chains = sorted(chain_coords.keys())
    n = len(chains)
    matrix = np.zeros((n, n))
    
    for i, c1 in enumerate(chains):
        for j, c2 in enumerate(chains):
            if i >= j: continue
            dists = cdist(chain_coords[c1], chain_coords[c2])
            if np.min(dists) < CONFIG['contact_threshold']:
                matrix[i, j] = 1
                matrix[j, i] = 1
    return pd.DataFrame(matrix, index=chains, columns=chains)

def get_mappings(proteome_seqs):
    """Creates map: Chain ID -> Proteome Index."""
    proteome_ids = list(proteome_seqs.keys())
    id_to_idx = {uid: i for i, uid in enumerate(proteome_ids)}
    
    chain_to_idx = {}
    ribo_indices = []
    
    for chain_id, uniprot_id in RIBOSOMAL_MAP.items():
        # Handle cases where PDB ID might differ slightly or isoform issues
        # Simple lookup for now
        if uniprot_id in id_to_idx:
            idx = id_to_idx[uniprot_id]
            chain_to_idx[chain_id] = idx
            ribo_indices.append(idx)
            
    return chain_to_idx, sorted(list(set(ribo_indices))), proteome_ids

# =============================================================================
# PLOTTING ROUTINES
# =============================================================================

def analyze_membership(attentions, proteome_ids, ribo_indices, output_dir):
    print("   [1/3] Plotting Membership (Violin & ROC)...")
    
    ribo_set = set(ribo_indices)
    non_ribo_indices = [i for i in range(len(proteome_ids)) if i not in ribo_set]
    
    # 1. Ribo-Ribo (Positive)
    pos_scores = []
    for i in range(len(ribo_indices)):
        for j in range(i+1, len(ribo_indices)):
            idx_a, idx_b = ribo_indices[i], ribo_indices[j]
            # Mean attention, one direction only (a_ij, i < j), unlike the
            # symmetrized pair_attn used by the other panels
            score = attentions[:, :, idx_a, idx_b].mean().item()
            pos_scores.append(score)
            
    # 2. Ribo-NonRibo (Negative) - Sampled
    neg_scores = []
    np.random.seed(42)
    # Sample enough negatives to be representative
    sampled_neg = np.random.choice(non_ribo_indices, min(len(non_ribo_indices), 1000), replace=False)
    
    # Calculate scores for Ribo vs Sampled Non-Ribo
    for r_idx in ribo_indices:
        for n_idx in sampled_neg:
            score = attentions[:, :, r_idx, n_idx].mean().item()  # ribosomal -> other only
            neg_scores.append(score)
            
    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Violin
    parts = ax1.violinplot([neg_scores, pos_scores], showmeans=False, showmedians=True)
    for pc in parts['bodies']:
        pc.set_facecolor(COLOR_RED)
        pc.set_alpha(0.7)
    parts['bodies'][1].set_facecolor(COLOR_BLUE)
    parts['cmaxes'].set_color('black')
    parts['cmins'].set_color('black')
    parts['cmedians'].set_color('black')
    parts['cbars'].set_color('black')
    pc.set_facecolor(COLOR_RED)
    pc.set_alpha(0.7)
    parts['bodies'][1].set_facecolor(COLOR_BLUE)
    ax1.set_xticks([1, 2]); ax1.set_xticklabels(['Inter-complex', 'Intra-complex'], fontsize=22)
    ax1.set_ylabel('Attention scores', fontsize=24); ax1.set_title('Complex membership', fontsize=24)
    ax1.minorticks_on()
    ax1.yaxis.set_minor_locator(MultipleLocator(0.0005))
    ax1.tick_params(axis='y', which='minor', length=3, width=0.8)
    ax1.tick_params(axis='y', which='major', width=1.0)
    
    # ROC
    y_true = [0]*len(neg_scores) + [1]*len(pos_scores)
    y_score = neg_scores + pos_scores
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_val = auc(fpr, tpr)
    
    ax2.plot(fpr, tpr, color=COLOR_BLUE, lw=3, label=f'AUC = {roc_val:.3f}')
    ax2.plot([0, 1], [0, 1], 'k--'); ax2.legend(loc="lower right", fontsize=13)
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('Membership ROC', fontsize=18)
    ax2.minorticks_on()
    ax2.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax2.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax2.tick_params(which='minor', length=3, width=0.8)
    ax2.tick_params(which='major', width=1.0)

    style_axes(ax1)
    style_axes(ax2)
    
    save_figure(fig, output_dir, "Fig1_Membership")
    print(f"         -> Saved Fig1 (AUC={roc_val:.3f})")

def analyze_specificity(attentions, gt_df, chain_to_idx, output_dir):
    print("   [2/3] Plotting Specificity (Heatmaps)...")
    
    valid_chains = [c for c in gt_df.index if c in chain_to_idx]
    if not valid_chains: return
    
    n = len(valid_chains)
    attn_matrix = np.zeros((n, n))
    
    for i, c1 in enumerate(valid_chains):
        idx1 = chain_to_idx[c1]
        for j, c2 in enumerate(valid_chains):
            idx2 = chain_to_idx[c2]
            s = pair_attn(attentions, idx1, idx2)
            attn_matrix[i, j] = s
            
    gt_matrix = gt_df.loc[valid_chains, valid_chains].values
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    sns.heatmap(gt_matrix, ax=ax1, cmap="Greys", cbar=False, xticklabels=False, yticklabels=False)
    ax1.set_xlabel('Chain ID')
    ax1.set_ylabel('Chain ID')
    ax1.set_title('Ground Truth Contacts', fontsize=18)
    sns.heatmap(attn_matrix, ax=ax2, cmap=to_colormap(COLOR_BLUE), xticklabels=False, yticklabels=False)
    ax2.set_xlabel('Chain ID')
    ax2.set_ylabel('Chain ID')
    ax2.set_title('ProteomeLM Attention', fontsize=18)

    style_axes(ax1)
    style_axes(ax2)
    
    save_figure(fig, output_dir, "Fig2_Specificity")
    print("         -> Saved Fig2")

def analyze_distance(attentions, chain_coords, chain_to_idx, output_dir):
    print("   [2/4] Plotting Distance Correlation (Å)...")
    
    valid_chains = [c for c in chain_coords.keys() if c in chain_to_idx]
    if len(valid_chains) < 2:
        print("         -> Skipping (insufficient mapped chains)")
        return
    
    dists, scores = [], []
    for i, c1 in enumerate(valid_chains):
        for j, c2 in enumerate(valid_chains):
            if i < j:
                coords1 = chain_coords[c1]
                coords2 = chain_coords[c2]
                d = np.min(cdist(coords1, coords2))
                idx1, idx2 = chain_to_idx[c1], chain_to_idx[c2]
                s = pair_attn(attentions, idx1, idx2)
                dists.append(d); scores.append(s)
                
    if not dists:
        print("         -> Skipping (no distances computed)")
        return
    
    rho, _ = pearsonr(dists, scores)
    
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(np.array(dists), scores, alpha=0.3, color=COLOR_GRAY)
    
    slope, intercept, _, _, _ = linregress(dists, scores)
    x = np.array([min(dists), max(dists)])
    ax.plot(x, slope*x + intercept, color=COLOR_RED, lw=2)
    
    ax.set_title(f'Distance Correlation (Pearson r={rho:.2f})', fontsize=18)
    ax.set_xlabel('Min Cα Distance (Å)')
    ax.set_ylabel('Attention')    
    ax.minorticks_on()
    ax.tick_params(which='minor', length=3, width=0.8)
    ax.tick_params(which='major', width=1.0)
    style_axes(ax)
    
    save_figure(fig, output_dir, "Fig3_Distance")
    print(f"         -> Saved Fig3 (Pearson r={rho:.2f})")

def analyze_reconstruction(attn_matrix, gt_df, valid_chains, output_dir, tag=""):
    print(f"   [4/4] Reconstructing Complex{tag} (PR/ROC + Top-k graph)...")

    if len(valid_chains) < 3:
        print("         -> Skipping (insufficient mapped chains)")
        return

    n = len(valid_chains)

    gt_matrix = gt_df.loc[valid_chains, valid_chains].values

    # Upper triangle edge lists
    triu_idx = np.triu_indices(n, k=1)
    y_true = gt_matrix[triu_idx].astype(int)
    y_score = attn_matrix[triu_idx]

    n_true = int(y_true.sum())
    n_edges = len(y_true)

    if n_true == 0:
        print("         -> Skipping (no ground-truth contacts)")
        return

    # ROC / PR metrics
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_val = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap_val = average_precision_score(y_true, y_score)

    # Top-k reconstruction (k = number of true contacts)
    k = n_true
    edge_scores = []
    for i in range(n):
        for j in range(i + 1, n):
            edge_scores.append((valid_chains[i], valid_chains[j], attn_matrix[i, j], gt_matrix[i, j]))

    edge_scores.sort(key=lambda x: x[2], reverse=True)
    top_edges = edge_scores[:k]

    tp = sum(1 for _, _, _, gt in top_edges if gt == 1)
    precision_at_k = tp / k if k else 0
    recall_at_k = tp / n_true if n_true else 0
    f1_at_k = (2 * precision_at_k * recall_at_k / (precision_at_k + recall_at_k)) if (precision_at_k + recall_at_k) else 0

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(21, 7.5))

    # Panel A: ROC
    ax = axes[0]
    ax.plot(fpr, tpr, color=COLOR_BLUE, lw=3, label=f'AUC = {roc_val:.3f}')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Contact Recovery (ROC)', fontsize=18)
    ax.legend(loc='lower right', fontsize=13)
    ax.minorticks_on()
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.tick_params(which='minor', length=3, width=0.8)
    ax.tick_params(which='major', width=1.0)
    style_axes(ax)

    # Panel B: PR
    ax = axes[1]
    ax.plot(recall, precision, color=COLOR_RED, lw=3, label=f'AP = {ap_val:.3f}')
    ax.axhline(y=n_true / n_edges, color='gray', linestyle='--', lw=1, label='Random')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Contact Recovery (PR)', fontsize=18)
    ax.legend(loc='lower left', fontsize=13)
    ax.minorticks_on()
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.tick_params(which='minor', length=3, width=0.8)
    ax.tick_params(which='major', width=1.0)
    style_axes(ax)

    # Panel C: Top-k graph
    ax = axes[2]
    G = nx.Graph()
    G.add_nodes_from(valid_chains)
    for a, b, score, gt in top_edges:
        G.add_edge(a, b, score=score, gt=gt)

    pos = nx.spring_layout(G, seed=42)
    edge_colors = [COLOR_GREEN if G[u][v]['gt'] == 1 else COLOR_RED for u, v in G.edges()]
    edge_widths = [2.5 if G[u][v]['gt'] == 1 else 1.0 for u, v in G.edges()]

    nx.draw_networkx_nodes(G, pos, node_size=120, node_color=COLOR_CYAN, ax=ax, alpha=0.9)
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths, ax=ax, alpha=0.8)
    nx.draw_networkx_labels(G, pos, font_size=9, ax=ax)

    ax.set_title(f"Top-k Reconstruction (k={k})\nP@k={precision_at_k:.2f}, R@k={recall_at_k:.2f}", fontsize=17)
    ax.axis('off')

    plt.tight_layout()
    fig_name = "Fig4_Reconstruction" + (f"_{tag}" if tag else "")
    save_figure(fig, output_dir, fig_name)

    # Save metrics
    metrics = {
        'n_chains': n,
        'n_true_contacts': n_true,
        'n_edges': n_edges,
        'roc_auc': float(roc_val),
        'avg_precision': float(ap_val),
        'precision_at_k': float(precision_at_k),
        'recall_at_k': float(recall_at_k),
        'f1_at_k': float(f1_at_k)
    }

    metrics_name = "reconstruction_metrics" + (f"_{tag}" if tag else "") + ".json"
    with open(os.path.join(output_dir, metrics_name), 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"         -> Saved Fig4{tag} + metrics (AUC={roc_val:.3f}, AP={ap_val:.3f})")

def analyze_attention_heads(attentions, gt_df, chain_coords, chain_to_idx, output_dir):
    print("   [3/4] Analyzing per-head performance (AUC + correlation)...")

    valid_chains = [c for c in gt_df.index if c in chain_to_idx and c in chain_coords]
    if len(valid_chains) < 3:
        print("         -> Skipping (insufficient mapped chains)")
        return None

    n = len(valid_chains)
    num_layers, num_heads, _, _ = attentions.shape

    # Precompute distances and GT labels for upper triangle
    dist_matrix = np.zeros((n, n))
    gt_matrix = gt_df.loc[valid_chains, valid_chains].values

    for i, c1 in enumerate(valid_chains):
        for j, c2 in enumerate(valid_chains):
            if i < j:
                d = np.min(cdist(chain_coords[c1], chain_coords[c2]))
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d

    triu_idx = np.triu_indices(n, k=1)
    y_true = gt_matrix[triu_idx].astype(int)
    dists = dist_matrix[triu_idx]

    if y_true.sum() == 0:
        print("         -> Skipping (no ground-truth contacts)")
        return None

    auc_matrix = np.zeros((num_layers, num_heads))
    ap_matrix = np.zeros((num_layers, num_heads))
    corr_matrix = np.zeros((num_layers, num_heads))
    scores_per_head = np.zeros((num_layers, num_heads, len(y_true)))

    for layer in range(num_layers):
        for head in range(num_heads):
            scores = []
            for i, c1 in enumerate(valid_chains):
                idx1 = chain_to_idx[c1]
                for j, c2 in enumerate(valid_chains):
                    if i < j:
                        idx2 = chain_to_idx[c2]
                        s = pair_attn(attentions[layer, head], idx1, idx2)
                        scores.append(s)

            scores = np.array(scores)
            scores_per_head[layer, head, :] = scores
            fpr, tpr, _ = roc_curve(y_true, scores)
            auc_matrix[layer, head] = auc(fpr, tpr)
            ap_matrix[layer, head] = average_precision_score(y_true, scores)
            corr_matrix[layer, head] = pearsonr(dists, scores)[0]

    # Best head by AUC
    best_layer, best_head = np.unravel_index(auc_matrix.argmax(), auc_matrix.shape)
    best_info = {
        'layer': int(best_layer),
        'head': int(best_head),
        'auc': float(auc_matrix[best_layer, best_head]),
        'ap': float(ap_matrix[best_layer, best_head]),
        'pearson_r': float(corr_matrix[best_layer, best_head])
    }

    # Permutation test (max AUC across heads)
    n_permutations = 1000
    np.random.seed(42)
    permuted_max_aucs = []

    for _ in range(n_permutations):
        perm_labels = np.random.permutation(y_true)
        perm_auc = np.zeros((num_layers, num_heads))
        for layer in range(num_layers):
            for head in range(num_heads):
                fpr, tpr, _ = roc_curve(perm_labels, scores_per_head[layer, head, :])
                perm_auc[layer, head] = auc(fpr, tpr)
        permuted_max_aucs.append(perm_auc.max())

    permuted_max_aucs = np.array(permuted_max_aucs)
    observed_max_auc = auc_matrix.max()
    expected_max_null = permuted_max_aucs.mean()
    pvalue_max = (np.sum(permuted_max_aucs >= observed_max_auc) + 1) / (n_permutations + 1)

    # Plot
    fig, axes = plt.subplots(1, 4, figsize=(36, 8.5))

    hm_auc = sns.heatmap(
        auc_matrix,
        annot=True,
        fmt='.2f',
        cmap=to_colormap(COLOR_BLUE),
        vmin=0.3,
        vmax=0.9,
        center=0.5,
        ax=axes[0],
        cbar_kws={'label': 'AUC'},
        annot_kws={'size': 17}
    )
    axes[0].set_xlabel('Head', fontsize=24)
    axes[0].set_ylabel('Layer', fontsize=24)
    axes[0].set_title('AUC of PPI recovery', fontsize=24)
    hm_auc.collections[0].colorbar.ax.tick_params(labelsize=20)
    hm_auc.collections[0].colorbar.set_label('AUC', size=22)
    style_axes(axes[0])

    hm_ap = sns.heatmap(
        ap_matrix,
        annot=True,
        fmt='.2f',
        cmap=to_colormap(COLOR_BLUE),
        vmin=y_true.mean(),
        vmax=0.8,
        ax=axes[1],
        cbar_kws={'label': 'Avg Precision'},
        annot_kws={'size': 17}
    )
    axes[1].set_xlabel('Head', fontsize=24)
    axes[1].set_ylabel('Layer', fontsize=24)
    axes[1].set_title('Per-head avg precision', fontsize=24)
    hm_ap.collections[0].colorbar.ax.tick_params(labelsize=20)
    hm_ap.collections[0].colorbar.set_label('Avg precision', size=22)
    style_axes(axes[1])

    hm_corr = sns.heatmap(
        corr_matrix,
        annot=True,
        fmt='.2f',
        cmap=to_colormap(COLOR_BLUE),
        vmin=-0.6,
        vmax=0.6,
        center=0,
        ax=axes[2],
        cbar_kws={'label': 'Pearson r (Å vs Attention)'},
        annot_kws={'size': 17}
    )
    axes[2].set_xlabel('Head', fontsize=24)
    axes[2].set_ylabel('Layer', fontsize=24)
    axes[2].set_title('Per-head distance correlation', fontsize=24)
    hm_corr.collections[0].colorbar.ax.tick_params(labelsize=20)
    hm_corr.collections[0].colorbar.set_label('Pearson r (Å vs attention)', size=22)
    style_axes(axes[2])

    axes[3].hist(permuted_max_aucs, bins=50, alpha=0.7, color=COLOR_GRAY, density=False)
    axes[3].axvline(observed_max_auc, color=COLOR_RED, lw=2, linestyle='--',
                    label=f'Observed max = {observed_max_auc:.2f}')
    axes[3].axvline(expected_max_null, color=COLOR_BLUE, lw=2, linestyle=':',
                    label=f'Null model max = {expected_max_null:.2f}')
    axes[3].set_xlabel('Maxiumum AUC across all heads', fontsize=24)
    axes[3].set_ylabel('Counts', fontsize=24)
    axes[3].set_title(f'Permutation test (n={n_permutations})\nP-value = {pvalue_max:.2f}', fontsize=24)
    axes[3].legend(fontsize=20)
    for ax in axes:
        ax.tick_params(axis='both', labelsize=20)
    style_axes(axes[3])

    plt.tight_layout()
    save_figure(fig, output_dir, "Fig3_Head_Performance")

    results = {
        'auc_matrix': auc_matrix.tolist(),
        'ap_matrix': ap_matrix.tolist(),
        'corr_matrix': corr_matrix.tolist(),
        'best_head': best_info,
        'permutation_test': {
            'n_permutations': n_permutations,
            'observed_max_auc': float(observed_max_auc),
            'expected_max_null': float(expected_max_null),
            'pvalue_corrected': float(pvalue_max),
            'significant': bool(pvalue_max < 0.05)
        }
    }

    with open(os.path.join(output_dir, "head_performance_metrics.json"), 'w') as f:
        json.dump(results, f, indent=2)

    print(
        f"         -> Saved Fig3_Head_Performance + metrics (best AUC={best_info['auc']:.3f}, "
        f"expected null max={expected_max_null:.3f}, p={pvalue_max:.3f})"
    )

    # Build best-head attention matrix for reconstruction
    attn_matrix = np.zeros((n, n))
    for i, c1 in enumerate(valid_chains):
        idx1 = chain_to_idx[c1]
        for j, c2 in enumerate(valid_chains):
            idx2 = chain_to_idx[c2]
            s = pair_attn(attentions[best_layer, best_head], idx1, idx2)
            attn_matrix[i, j] = s

    return {
        'best_info': best_info,
        'best_attn_matrix': attn_matrix,
        'valid_chains': valid_chains
    }

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--predict', action='store_true')
    parser.add_argument('--model', type=str)
    parser.add_argument('--orthodb-db-path', type=str, default=None,
                        help='Directory with group_vectors_*.pkl files')
    parser.add_argument('--orthodb-tsv', type=str, default=None,
                        help='OrthoDB TSV from download_proteome (optional)')
    parser.add_argument('--orthodb-min-group-size', type=int, default=0,
                        help='Minimum OrthoDB group size (default: 0)')
    args = parser.parse_args()
    
    output_dir = ensure_dir(CONFIG['output_dir'])
    
    if args.prepare:
        print(">>> PREPARING DATA")
        # 1. Download
        proteome_path = download_proteome(CONFIG['proteome_id'], output_dir, 'ecoli')
        pdb_path = download_pdb_structure(CONFIG['pdb_id'], output_dir, 'cif')
        
        # 2. Parse & Map
        proteome_seqs = parse_fasta(proteome_path)
        chain_coords = parse_structure(pdb_path)
        gt_df = compute_ground_truth(chain_coords)
        
        # Save GT
        gt_df.to_csv(os.path.join(output_dir, "ground_truth.csv"))
        
        # Save Helper Mappings
        chain_to_idx, ribo_indices, proteome_ids = get_mappings(proteome_seqs)
        with open(os.path.join(output_dir, "mappings.json"), 'w') as f:
            json.dump({
                'chain_to_idx': chain_to_idx,
                'ribo_indices': ribo_indices,
                'proteome_ids': proteome_ids
            }, f)
            
        print(">>> Done. Run --predict next.")

    if args.predict:
        print(">>> GENERATING PREDICTIONS & FIGURES")
        if not args.model: print("Error: --model required"); return
        
        # 1. Load Data
        proteome_path = os.path.join(output_dir, "ecoli_proteome.fasta")
        with open(os.path.join(output_dir, "mappings.json")) as f:
            mappings = json.load(f)
        gt_df = pd.read_csv(os.path.join(output_dir, "ground_truth.csv"), index_col=0)
        pdb_path = os.path.join(output_dir, f"{CONFIG['pdb_id']}.cif")
        if not os.path.exists(pdb_path):
            pdb_path = download_pdb_structure(CONFIG['pdb_id'], output_dir, 'cif')
        chain_coords = parse_structure(pdb_path)
        
        # 2. Run Model
        esm_embs = compute_esm_embeddings(proteome_path, output_dir, "ecoli", device=CONFIG['device'])

        group_embeds = None
        if args.orthodb_db_path:
            group_embeds = build_orthodb_group_embeds(
                fasta_path=proteome_path,
                esm_embeddings=esm_embs,
                output_dir=output_dir,
                organism='ecoli',
                proteome_id=CONFIG['proteome_id'],
                db_path=args.orthodb_db_path,
                tsv_path=args.orthodb_tsv,
                min_group_size=args.orthodb_min_group_size,
            )

        compute_proteomelm_embeddings(
            esm_embs,
            output_dir,
            args.model,
            "ecoli",
            device=CONFIG['device'],
            group_embeds=group_embeds,
        )
        
        # 3. Load Attentions
        attentions = load_attentions(output_dir, "ecoli")
        
        # 4. EXECUTE PLOTS
        analyze_membership(attentions, mappings['proteome_ids'], mappings['ribo_indices'], output_dir)
        analyze_specificity(attentions, gt_df, mappings['chain_to_idx'], output_dir)
        analyze_distance(attentions, chain_coords, mappings['chain_to_idx'], output_dir)

        head_results = analyze_attention_heads(attentions, gt_df, chain_coords, mappings['chain_to_idx'], output_dir)

        # Reconstruction (all-head average)
        valid_chains = [c for c in gt_df.index if c in mappings['chain_to_idx']]
        n = len(valid_chains)
        attn_matrix = np.zeros((n, n))
        for i, c1 in enumerate(valid_chains):
            idx1 = mappings['chain_to_idx'][c1]
            for j, c2 in enumerate(valid_chains):
                idx2 = mappings['chain_to_idx'][c2]
                s = pair_attn(attentions, idx1, idx2)
                attn_matrix[i, j] = s

        analyze_reconstruction(attn_matrix, gt_df, valid_chains, output_dir, tag="")

        # Reconstruction (best head)
        if head_results is not None:
            analyze_reconstruction(
                head_results['best_attn_matrix'],
                gt_df,
                head_results['valid_chains'],
                output_dir,
                tag="BestHead"
            )
        
        print(f">>> All figures saved to {output_dir}")

if __name__ == "__main__":
    main()