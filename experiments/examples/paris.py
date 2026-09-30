#!/usr/bin/env python3
"""
=============================================================================
PARIS PHAGE DEFENSE SYSTEM VALIDATION FOR PROTEOMELM
=============================================================================

Tests whether ProteomeLM can predict which phage peptides are recognized by
bacterial PARIS (Phage Anti-Restriction Induced System) AriA proteins.

The PARIS system consists of AriA proteins that detect specific phage peptides
and trigger a defense response. Each AriA variant (Sys 2, 3, 6, 7, 8, 9) 
recognizes a different set of phage peptides.

This is a stringent test because:
1. AriA proteins are similar across systems but recognize different peptides
2. Phage peptides are short (~40 aa) and diverse
3. Binary classification with very imbalanced data (1-3% positive)

Benchmark AP scores (ESM2/ProstT5 + XGBoost):
    Sys 2: 0.37-0.45
    Sys 3: 0.38-0.45
    Sys 6: 0.16-0.19
    Sys 7: 0.47-0.55
    Sys 8: 0.02-0.12  (few positives)
    Sys 9: 0.01-0.01  (few positives)

Reference structure: PDB 8V45 (AriA System 2 with T7 Ocr)

Inputs (read by --prepare from --data-dir, default experiments/examples/data/paris_validation/):
    binary_toxicity_hits.csv    per-peptide sequences ('Gene #', 'Protein_Sequence')
                                and binary hits per system ('Sys 2' ... 'Sys 9')
    protein_clusters_3mer.csv   'Gene #' -> 'Cluster_ID', used for the cluster-based split
    PARIS_systems.faa           AriA protein sequences, headers containing 'Sys2' ... 'Sys9'
These files are not included in the repository (data/ directories are
gitignored); they are available from the authors on request.

Usage:
    python paris.py --prepare                   # Setup data
    python paris.py --predict --model <path>    # Run ProteomeLM
    python paris.py --evaluate                  # Evaluate predictions
"""

import os
import sys
import json
import argparse
import warnings
from typing import Dict, List, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import shared utilities
from proteomelm.utils.io import ensure_dir, parse_fasta, write_fasta
from proteomelm.utils.embedding import compute_esm_embeddings, compute_proteomelm_embeddings, load_attentions
from experiments.examples.common import pair_attn

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    'output_dir': './paris_validation',
    'data_dir': str(Path(__file__).parent / 'data' / 'paris_validation'),
    'esm_device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'proteomelm_model': None,
}

# =============================================================================
# PARIS SYSTEM INFORMATION
# =============================================================================

# PARIS AriA systems - from the FASTA file
PARIS_SYSTEMS = {
    'Sys2': {'name': 'B185 AriA', 'fasta_header': 'Sys2 B185 AriA'},
    'Sys3': {'name': 'O42 AriA', 'fasta_header': 'Sys3 O42 AriA'},
    # 'Sys5': {'name': '41-1Ti9 AriA', 'fasta_header': 'Sys5 41-1Ti9 AriA'},  # Poor data
    'Sys6': {'name': '55989 AriA', 'fasta_header': 'Sys6 55989 AriA'},
    'Sys7': {'name': 'H617 AriA', 'fasta_header': 'Sys7 H617 AriA'},
    'Sys8': {'name': 'S2-12 AriAB', 'fasta_header': 'Sys8 S2-12 AriAB'},  # Few positives
    'Sys9': {'name': 'E1167 AriAB', 'fasta_header': 'Sys9 E1167 AriAB'},  # Few positives
}

# Column names in the CSV
SYSTEM_COLUMNS = {
    'Sys2': 'Sys 2',
    'Sys3': 'Sys 3',
    'Sys6': 'Sys 6',
    'Sys7': 'Sys 7',
    'Sys8': 'Sys 8',
    'Sys9': 'Sys 9',
}

# Baseline AP scores from ESM2/ProstT5
BASELINE_SCORES = {
    'Sys2': {'ESM': 0.374, 'ProstT5': 0.447, 'Combined': 0.412},
    'Sys3': {'ESM': 0.392, 'ProstT5': 0.382, 'Combined': 0.396},
    'Sys6': {'ESM': 0.165, 'ProstT5': 0.160, 'Combined': 0.185},
    'Sys7': {'ESM': 0.473, 'ProstT5': 0.551, 'Combined': 0.502},
    'Sys8': {'ESM': 0.115, 'ProstT5': 0.025, 'Combined': 0.064},
    'Sys9': {'ESM': 0.012, 'ProstT5': 0.011, 'Combined': 0.008},
}


# =============================================================================
# STEP 1: PREPARE DATA
# =============================================================================

def prepare_data(output_dir: str):
    """Load and prepare PARIS data for validation."""
    
    print("="*70)
    print("PARIS VALIDATION: PREPARATION")
    print("="*70)
    
    data_dir = CONFIG['data_dir']
    print(f"  Input data: {data_dir}")
    
    # Load toxicity data
    print("\n[1/4] Loading toxicity data...")
    tox_path = os.path.join(data_dir, 'binary_toxicity_hits.csv')
    tox_df = pd.read_csv(tox_path)
    
    # Clean sequences (remove stop codon *)
    tox_df['Protein_Sequence'] = tox_df['Protein_Sequence'].str.replace('*', '', regex=False)
    
    print(f"  Total phage peptides: {len(tox_df)}")
    
    # Subsample for faster testing
    if len(tox_df) > 10000:
        print(f"  Subsampling to 10,000 peptides for testing...")
        tox_df = tox_df.sample(n=10000, random_state=42).reset_index(drop=True)
        print(f"  After subsampling: {len(tox_df)}")
    
    # Load cluster info for train/test split
    print("\n[2/4] Loading cluster information...")
    cluster_path = os.path.join(data_dir, 'protein_clusters_3mer.csv')
    cluster_df = pd.read_csv(cluster_path)
    
    # Merge
    tox_df = tox_df.merge(cluster_df, on='Gene #')
    
    n_clusters = cluster_df['Cluster_ID'].nunique()
    print(f"  Unique clusters: {n_clusters}")
    
    # Load AriA sequences
    print("\n[3/4] Loading AriA protein sequences...")
    fasta_path = os.path.join(data_dir, 'PARIS_systems.faa')
    aria_seqs = parse_fasta(fasta_path)
    
    # Map to system IDs
    aria_by_system = {}
    for header, seq in aria_seqs.items():
        for sys_id, info in PARIS_SYSTEMS.items():
            if sys_id.lower().replace('sys', 'sys') in header.lower():
                aria_by_system[sys_id] = {
                    'header': header,
                    'sequence': seq,
                    'length': len(seq)
                }
                break
    
    print(f"  AriA proteins loaded: {len(aria_by_system)}")
    for sys_id, info in aria_by_system.items():
        print(f"    {sys_id}: {info['length']} aa")
    
    # Statistics per system
    print("\n[4/4] Computing statistics per system...")
    stats = {}
    for sys_id, col_name in SYSTEM_COLUMNS.items():
        if col_name in tox_df.columns:
            n_pos = tox_df[col_name].sum()
            n_total = len(tox_df)
            stats[sys_id] = {
                'n_positives': int(n_pos),
                'n_total': n_total,
                'prevalence': float(n_pos / n_total),
                'baseline_ap': BASELINE_SCORES.get(sys_id, {})
            }
            print(f"  {sys_id}: {n_pos:,} positives ({100*n_pos/n_total:.2f}%)")
    
    # Save processed data
    tox_df.to_csv(os.path.join(output_dir, 'peptides_processed.csv'), index=False)
    
    with open(os.path.join(output_dir, 'aria_sequences.json'), 'w') as f:
        json.dump(aria_by_system, f, indent=2)
    
    with open(os.path.join(output_dir, 'system_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n  Saved: {output_dir}/")
    print(f"    - peptides_processed.csv")
    print(f"    - aria_sequences.json")
    print(f"    - system_stats.json")
    
    return tox_df, aria_by_system, stats


# =============================================================================
# STEP 2: COMPUTE EMBEDDINGS
# =============================================================================


def extract_attention_scores(output_dir: str, aria_indices: Dict[str, int], 
                              peptide_indices: List[int]) -> Dict[str, np.ndarray]:
    """Extract attention scores from AriA to each peptide."""
    
    try:
        print("  Loading attention weights...")
        attentions = load_attentions(output_dir, prefix="paris")
    except FileNotFoundError:
        print("  Attention file not found. Run --predict first.")
        return None
    # attentions: [num_layers, num_heads, seq_len, seq_len]
    
    scores_by_system = {}
    
    for sys_id, aria_idx in aria_indices.items():
        # Get attention from AriA to each peptide, averaged over layers and heads
        attn_to_peptides = []
        for pep_idx in peptide_indices:
            # Average attention (both directions) across layers and heads
            attn_to_peptides.append(pair_attn(attentions, aria_idx, pep_idx))
        
        scores_by_system[sys_id] = np.array(attn_to_peptides)
    
    return scores_by_system


def compute_paris_embeddings(output_dir: str, model_path: str = None):
    """
    Compute all embeddings needed for evaluation.
    
    Creates a combined FASTA with AriA proteins + all peptides,
    then computes ESM and ProteomeLM embeddings (same as CCT approach).
    """
    
    print("\n" + "="*70)
    print("PARIS VALIDATION: COMPUTING EMBEDDINGS")
    print("="*70)
    
    # Load data
    peptides_df = pd.read_csv(os.path.join(output_dir, 'peptides_processed.csv'))
    with open(os.path.join(output_dir, 'aria_sequences.json')) as f:
        aria_seqs = json.load(f)
    
    # Create combined FASTA: AriA proteins first, then peptides
    print("\n[1/4] Creating combined FASTA file...")
    combined_seqs = {}
    
    # Add AriA sequences (will be at indices 0 to n_aria-1)
    aria_indices = {}
    for i, (sys_id, info) in enumerate(aria_seqs.items()):
        seq_id = f"AriA_{sys_id}"
        combined_seqs[seq_id] = info['sequence']
        aria_indices[sys_id] = i
    
    n_aria = len(aria_seqs)
    print(f"  AriA proteins: {n_aria} (indices 0-{n_aria-1})")
    
    # Add peptide sequences (will be at indices n_aria to n_aria + n_peptides - 1)
    peptide_indices = []
    for i, row in peptides_df.iterrows():
        seq_id = row['Gene #']
        combined_seqs[seq_id] = row['Protein_Sequence']
        peptide_indices.append(n_aria + len(peptide_indices))
    
    print(f"  Peptides: {len(peptide_indices)} (indices {n_aria}-{n_aria + len(peptide_indices) - 1})")
    
    # Write combined FASTA
    fasta_path = os.path.join(output_dir, 'paris_combined.fasta')
    write_fasta(combined_seqs, fasta_path)
    print(f"  Saved: {fasta_path}")
    
    # Save index mapping
    index_mapping = {
        'aria_indices': aria_indices,
        'peptide_indices': peptide_indices,
        'n_aria': n_aria,
        'n_peptides': len(peptide_indices),
        'total': len(combined_seqs)
    }
    with open(os.path.join(output_dir, 'index_mapping.json'), 'w') as f:
        json.dump(index_mapping, f, indent=2)
    
    # Compute ESM embeddings
    print("\n[2/4] Computing ESM-C embeddings...")
    esm_embeddings = compute_esm_embeddings(
        fasta_path=fasta_path,
        output_dir=output_dir,
        prefix="paris",
        device=CONFIG['esm_device']
    )
    
    # Compute ProteomeLM embeddings
    if model_path:
        print("\n[3/4] Computing ProteomeLM embeddings and attentions...")
        proteomelm_embeddings, _ = compute_proteomelm_embeddings(
            esm_embeddings=esm_embeddings,
            output_dir=output_dir,
            model_path=model_path,
            prefix="paris",
            device='cpu'
        )
        
        # Extract attention scores for each AriA system
        print("\n[4/4] Extracting attention scores per system...")
        scores = extract_attention_scores(output_dir, aria_indices, peptide_indices)
        
        if scores:
            for sys_id, attn_scores in scores.items():
                save_path = os.path.join(output_dir, f"{sys_id}_attention_scores.npy")
                np.save(save_path, attn_scores)
                print(f"  Saved {sys_id}: {save_path}")
    else:
        print("\n[3/4] Skipping ProteomeLM (no model path provided)")
        print("[4/4] Skipped")
    
    print("\n" + "="*70)
    print("EMBEDDINGS COMPUTED")
    print("="*70)


# =============================================================================
# STEP 3: TRAIN-TEST SPLIT (CLUSTER-BASED)
# =============================================================================

def create_train_test_split(df: pd.DataFrame, test_fraction: float = 0.2, 
                            seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create train/test split based on sequence clusters.
    Ensures no information leakage between similar sequences.
    """
    np.random.seed(seed)
    
    # Get unique clusters
    clusters = df['Cluster_ID'].unique()
    np.random.shuffle(clusters)
    
    # Assign clusters to test set
    n_test_clusters = int(len(clusters) * test_fraction)
    test_clusters = set(clusters[:n_test_clusters])
    
    # Split
    test_mask = df['Cluster_ID'].isin(test_clusters)
    train_df = df[~test_mask].copy()
    test_df = df[test_mask].copy()
    
    print(f"  Train: {len(train_df):,} samples ({len(clusters) - n_test_clusters} clusters)")
    print(f"  Test:  {len(test_df):,} samples ({n_test_clusters} clusters)")
    
    return train_df, test_df


# =============================================================================
# STEP 4: EVALUATION
# =============================================================================

def run_evaluation(output_dir: str, use_proteomelm: bool = True):
    """Run full evaluation pipeline."""
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import average_precision_score, roc_auc_score
    
    print("\n" + "="*70)
    print("PARIS VALIDATION: EVALUATION")
    print("="*70)
    
    # Load data
    df = pd.read_csv(os.path.join(output_dir, 'peptides_processed.csv'))
    
    # Load ESM embeddings (combined file)
    esm_emb = torch.load(os.path.join(output_dir, 'paris_esm_embeddings.pt'))
    
    # Load index mapping
    with open(os.path.join(output_dir, 'index_mapping.json')) as f:
        idx_map = json.load(f)
    
    n_aria = idx_map['n_aria']
    
    # Extract peptide embeddings (skip first n_aria which are AriA proteins)
    peptide_emb = esm_emb[n_aria:].float()  # [n_peptides, emb_dim] - convert to float32
    
    # Create train/test split
    print("\n[1/4] Creating cluster-based train/test split...")
    train_df, test_df = create_train_test_split(df)
    
    # Map dataframe indices to embedding indices
    train_idx = train_df.index.tolist()
    test_idx = test_df.index.tolist()
    
    all_results = []
    
    # Evaluate each system
    systems_to_eval = ['Sys2', 'Sys3', 'Sys6', 'Sys7']  # Skip 8, 9 for now (too few positives)
    
    for sys_id in systems_to_eval:
        print(f"\n[2/4] Evaluating {sys_id}...")
        
        col_name = SYSTEM_COLUMNS[sys_id]
        
        # Get labels
        y_train = train_df[col_name].values
        y_test = test_df[col_name].values
        
        print(f"  Train positives: {y_train.sum():,}/{len(y_train):,}")
        print(f"  Test positives: {y_test.sum():,}/{len(y_test):,}")
        
        # Method 1: ESM embeddings only
        print(f"\n  Method 1: ESM embeddings + classifier")
        X_train = peptide_emb[train_idx].numpy()
        X_test = peptide_emb[test_idx].numpy()
        
        clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        
        y_pred_esm = clf.predict_proba(X_test)[:, 1]
        ap_esm = average_precision_score(y_test, y_pred_esm)
        auc_esm = roc_auc_score(y_test, y_pred_esm)
        
        print(f"    ESM-only AP: {ap_esm:.4f} (baseline ESM: {BASELINE_SCORES[sys_id]['ESM']:.3f})")
        print(f"    ESM-only AUC: {auc_esm:.4f}")
        
        # Method 2: ProteomeLM attention scores
        attn_path = os.path.join(output_dir, f"{sys_id}_attention_scores.npy")
        
        if use_proteomelm and os.path.exists(attn_path):
            print(f"\n  Method 2: ProteomeLM attention")
            attn_scores = np.load(attn_path)  # [n_peptides]
            
            # Direct attention as score
            y_pred_attn = attn_scores[test_idx]
            ap_attn = average_precision_score(y_test, y_pred_attn)
            auc_attn = roc_auc_score(y_test, y_pred_attn)
            
            print(f"    Attention-only AP: {ap_attn:.4f}")
            print(f"    Attention-only AUC: {auc_attn:.4f}")
            
            # Method 3: Combined features
            print(f"\n  Method 3: ESM + ProteomeLM attention")
            
            X_train_combined = np.column_stack([X_train, attn_scores[train_idx].reshape(-1, 1)])
            X_test_combined = np.column_stack([X_test, attn_scores[test_idx].reshape(-1, 1)])
            
            clf_combined = GradientBoostingClassifier(n_estimators=100, random_state=42)
            clf_combined.fit(X_train_combined, y_train)
            
            y_pred_combined = clf_combined.predict_proba(X_test_combined)[:, 1]
            ap_combined = average_precision_score(y_test, y_pred_combined)
            auc_combined = roc_auc_score(y_test, y_pred_combined)
            
            print(f"    Combined AP: {ap_combined:.4f} (baseline combined: {BASELINE_SCORES[sys_id]['Combined']:.3f})")
            print(f"    Combined AUC: {auc_combined:.4f}")
            
            result = {
                'system': sys_id,
                'n_test': len(y_test),
                'n_positives': int(y_test.sum()),
                'esm_ap': float(ap_esm),
                'esm_auc': float(auc_esm),
                'attention_ap': float(ap_attn),
                'attention_auc': float(auc_attn),
                'combined_ap': float(ap_combined),
                'combined_auc': float(auc_combined),
                'baseline_esm': BASELINE_SCORES[sys_id]['ESM'],
                'baseline_combined': BASELINE_SCORES[sys_id]['Combined'],
            }
        else:
            result = {
                'system': sys_id,
                'n_test': len(y_test),
                'n_positives': int(y_test.sum()),
                'esm_ap': float(ap_esm),
                'esm_auc': float(auc_esm),
                'baseline_esm': BASELINE_SCORES[sys_id]['ESM'],
                'baseline_combined': BASELINE_SCORES[sys_id]['Combined'],
            }
        
        all_results.append(result)
    
    # Save results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(output_dir, 'evaluation_results.csv'), index=False)
    
    with open(os.path.join(output_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Visualize
    print("\n[3/4] Creating visualizations...")
    create_evaluation_plots(all_results, output_dir)
    
    # Summary
    print("\n[4/4] Summary")
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(results_df.to_string(index=False))
    
    return all_results


def create_evaluation_plots(results: List[Dict], output_dir: str):
    """Create visualization of evaluation results."""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Panel A: AP comparison
    ax = axes[0]
    systems = [r['system'] for r in results]
    x = np.arange(len(systems))
    width = 0.25
    
    baseline_esm = [r.get('baseline_esm', 0) for r in results]
    baseline_combined = [r.get('baseline_combined', 0) for r in results]
    our_esm = [r.get('esm_ap', 0) for r in results]
    our_combined = [r.get('combined_ap', r.get('esm_ap', 0)) for r in results]
    
    ax.bar(x - width, baseline_esm, width, label='Baseline ESM', color='lightblue', edgecolor='black')
    ax.bar(x, our_esm, width, label='Our ESM', color='blue', edgecolor='black')
    ax.bar(x + width, our_combined, width, label='Our Combined', color='darkblue', edgecolor='black')
    
    ax.set_ylabel('Average Precision')
    ax.set_xlabel('PARIS System')
    ax.set_xticks(x)
    ax.set_xticklabels(systems)
    ax.legend()
    ax.set_title('Average Precision by System', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add baseline line
    for i, (be, bc) in enumerate(zip(baseline_esm, baseline_combined)):
        ax.axhline(y=bc, xmin=(i-0.3)/len(systems), xmax=(i+0.5)/len(systems), 
                   color='red', linestyle='--', alpha=0.5)
    
    # Panel B: Improvement over baseline
    ax = axes[1]
    
    improvements = []
    for r in results:
        if 'combined_ap' in r:
            imp = r['combined_ap'] - r['baseline_combined']
        else:
            imp = r['esm_ap'] - r['baseline_esm']
        improvements.append(imp)
    
    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    ax.bar(systems, improvements, color=colors, edgecolor='black')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.set_ylabel('AP Improvement vs Baseline')
    ax.set_xlabel('PARIS System')
    ax.set_title('Improvement over ESM/XGBoost Baseline', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    out_path = os.path.join(output_dir, 'evaluation_comparison.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(out_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {out_path}")


# =============================================================================
# COMMAND FUNCTIONS
# =============================================================================

def cmd_prepare(output_dir: str):
    """Prepare data for validation."""
    ensure_dir(output_dir)
    prepare_data(output_dir)
    
    print("\n" + "="*70)
    print("READY - Run: python paris.py --predict --model <path_to_proteomelm>")
    print("="*70)


def cmd_predict(output_dir: str, model_path: str = None):
    """Run predictions."""
    ensure_dir(output_dir)
    
    # Check if data is prepared
    if not os.path.exists(os.path.join(output_dir, 'peptides_processed.csv')):
        print("Error: Run --prepare first")
        return
    
    compute_paris_embeddings(output_dir, model_path)
    
    print("\n" + "="*70)
    print("PREDICTIONS COMPLETE")
    print("Run: python paris.py --evaluate")
    print("="*70)


def cmd_evaluate(output_dir: str):
    """Evaluate predictions."""
    
    # Check if embeddings exist
    if not os.path.exists(os.path.join(output_dir, 'paris_esm_embeddings.pt')):
        print("Error: Run --predict first")
        return
    
    # Check if ProteomeLM scores exist
    use_proteomelm = os.path.exists(os.path.join(output_dir, 'Sys2_attention_scores.npy'))
    
    run_evaluation(output_dir, use_proteomelm)
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print(f"Results saved to: {output_dir}/")
    print("="*70)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='PARIS phage defense system validation for ProteomeLM',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Step 1: Prepare data
  python paris.py --prepare
  
  # Step 2: Run predictions (with ProteomeLM)
  python paris.py --predict --model /path/to/proteomelm
  
  # Step 3: Evaluate
  python paris.py --evaluate

Systems evaluated:
  Sys2, Sys3, Sys6, Sys7 (sufficient positives)
  Sys8, Sys9 (few positives: attention scores extracted, not evaluated)
  Sys5 (skipped - poor data quality)
"""
    )
    
    parser.add_argument('--prepare', '-p', action='store_true',
                        help='Prepare data files')
    parser.add_argument('--predict', action='store_true',
                        help='Run predictions')
    parser.add_argument('--evaluate', '-e', action='store_true',
                        help='Evaluate predictions')
    parser.add_argument('--output', '-o', type=str, default='./paris_validation',
                        help='Output directory')
    parser.add_argument('--model', '-m', type=str, default=None,
                        help='Path to ProteomeLM model')
    parser.add_argument('--data-dir', type=str, default=CONFIG['data_dir'],
                        help='Directory with the input files (see module docstring)')
    
    args = parser.parse_args()
    
    CONFIG['output_dir'] = args.output
    CONFIG['proteomelm_model'] = args.model
    CONFIG['data_dir'] = args.data_dir
    
    if args.prepare:
        cmd_prepare(args.output)
    elif args.predict:
        cmd_predict(args.output, args.model)
    elif args.evaluate:
        cmd_evaluate(args.output)
    else:
        parser.print_help()
        print("\n⚠️  Specify --prepare, --predict, or --evaluate")


if __name__ == "__main__":
    main()
