#!/usr/bin/env python3
"""
Differential-interactome analysis (step 3 of 3): reads the attention files from
extract_attention.py and writes tables and figures.

Produces:
- Table S1: AUROC for discriminating interaction types from random pairs
- Table S2: Pairwise binary classification AUROC between interaction types
- Figures: Attention head AUROC vs. Cosine similarity vs. PCA removal
- All figures in PDF and SVG format
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Plotting style
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['text.usetex'] = False
plt.rcParams['font.size'] = 16
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 13
plt.rcParams['figure.titlesize'] = 20

def style_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# === CONFIGURATION ===
SPECIES_LIST = ['ecoli', 'yeast', 'human']
SPECIES_NAMES = {'ecoli': 'E. coli', 'yeast': 'S. cerevisiae', 'human': 'H. sapiens'}
SPECIES_NAMES_ITALIC = {
    'ecoli': r'$\it{E.\ coli}$',
    'yeast': r'$\it{S.\ cerevisiae}$',
    'human': r'$\it{H.\ sapiens}$',
}
SPECIES_PLAIN_TO_ITALIC = {SPECIES_NAMES[s]: SPECIES_NAMES_ITALIC[s] for s in SPECIES_LIST}


def italicize_species_label(label: str) -> str:
    return SPECIES_PLAIN_TO_ITALIC.get(label, label)

# Interaction types for tables
TABLE_TYPES = ['pdb', 'pdb_physical', 'coexpression']
TYPE_LABELS = {
    'pdb': 'Direct (PDB)',
    'pdb_physical': 'Same complex (PDB)',
    'coexpression': 'Coexpression (STRING)',
    'random': 'Random'
}

# Colors for plots
COLORS = {
    'pdb': '#2B3AC0',
    'pdb_physical': '#E74C3C',
    'coexpression': '#F39C12',
    'random': '#7F8C8D'
}

colorspal6 = [(0.25098039215686274, 0.3254901960784314, 0.8274509803921568),
                (0.8666666666666667, 0.7019607843137254, 0.06274509803921569),
                (0.7098039215686275, 0.11372549019607843, 0.0784313725490196),
                (0.0, 0.7450980392156863, 1.0),
                (0.984313725490196, 0.28627450980392155, 0.6901960784313725),
                (0.0, 0.6980392156862745, 0.36470588235294116),
                (0.792156862745098, 0.792156862745098, 0.792156862745098)]



def load_attention_data(attention_dir: Path, species: str, types: List[str]) -> Dict:
    """Load attention data for specified types."""
    data = {}
    for itype in types:
        path = attention_dir / f"{species}_{itype}_attention.npz"
        if path.exists():
            npz = np.load(path, allow_pickle=True)
            # Average both directions
            attn = (npz['attention_a_to_b'] + npz['attention_b_to_a']) / 2
            data[itype] = {
                'attention': attn,
                'pairs': [tuple(p) for p in npz['pair_ids']]
            }
    return data


def load_functional_encodings(attention_dir: Path, species: str) -> Optional[Dict]:
    """Load functional encodings (ESM inputs + functional group embeddings)."""
    encoding_file = attention_dir / f"{species}_functional_encodings.npz"
    if not encoding_file.exists():
        return None
    
    data = np.load(encoding_file, allow_pickle=True)

    if 'group_equals_inputs' in data.files:
        group_equals_inputs = bool(data['group_equals_inputs'])
    else:
        inputs_arr = data['inputs_embeds']
        group_arr = data['group_embeds']
        group_equals_inputs = inputs_arr.shape == group_arr.shape and np.allclose(inputs_arr, group_arr, atol=1e-6)

    group_source = str(data['group_source']) if 'group_source' in data.files else 'unknown'

    return {
        'inputs_embeds': data['inputs_embeds'],
        'group_embeds': data['group_embeds'],
        'protein_to_idx': data['protein_to_idx'].item(),
        'group_source': group_source,
        'group_equals_inputs': group_equals_inputs,
    }


def compute_encoding_similarity(
    pairs: List[Tuple[str, str]],
    protein_to_idx: Dict[str, int],
    embeddings: np.ndarray
) -> Optional[np.ndarray]:
    """Compute cosine similarity between functional encodings for protein pairs."""
    indices_a, indices_b = [], []
    
    for prot_a, prot_b in pairs:
        if prot_a in protein_to_idx and prot_b in protein_to_idx:
            indices_a.append(protein_to_idx[prot_a])
            indices_b.append(protein_to_idx[prot_b])
    
    if not indices_a:
        return None
    
    indices_a, indices_b = np.array(indices_a), np.array(indices_b)
    
    # Get embeddings and normalize for cosine similarity
    emb_a = normalize(embeddings[indices_a], axis=1)
    emb_b = normalize(embeddings[indices_b], axis=1)
    
    return np.sum(emb_a * emb_b, axis=1)


def compute_embedding_auroc_vs_random(
    attention_dir: Path,
    species: str,
    embeddings: np.ndarray,
    protein_to_idx: Dict[str, int],
) -> Dict[str, float]:
    """Compute AUROC vs. random for all interaction types using cosine similarity."""
    auroc_results: Dict[str, float] = {}
    random_sims = None

    for itype in TABLE_TYPES + ['random']:
        path = attention_dir / f"{species}_{itype}_attention.npz"
        if not path.exists():
            continue
        npz = np.load(path, allow_pickle=True)
        pairs = [tuple(p) for p in npz['pair_ids']]
        sims = compute_encoding_similarity(pairs, protein_to_idx, embeddings)

        if sims is None:
            continue
        if itype == 'random':
            random_sims = sims
        else:
            auroc_results[itype] = sims

    if random_sims is None:
        return {t: np.nan for t in TABLE_TYPES}

    final_aurocs = {}
    for itype, sims in auroc_results.items():
        y_true = np.concatenate([np.ones(len(sims)), np.zeros(len(random_sims))])
        y_score = np.concatenate([sims, random_sims])
        final_aurocs[itype] = roc_auc_score(y_true, y_score)

    return final_aurocs


def compute_auroc_vs_random(attention_data: Dict, itype: str) -> float:
    """Compute best AUROC for interaction type vs. random using attention heads."""
    if itype not in attention_data or 'random' not in attention_data:
        return np.nan
    
    pos_attn = attention_data[itype]['attention']  # (n_pos, n_layers, n_heads)
    neg_attn = attention_data['random']['attention']
    
    n_layers, n_heads = pos_attn.shape[1], pos_attn.shape[2]
    y_true = np.concatenate([np.ones(len(pos_attn)), np.zeros(len(neg_attn))])
    
    best_auroc = 0.5
    for layer in range(n_layers):
        for head in range(n_heads):
            y_score = np.concatenate([pos_attn[:, layer, head], neg_attn[:, layer, head]])
            auroc = roc_auc_score(y_true, y_score)
            best_auroc = max(best_auroc, auroc)
    
    return best_auroc


def compute_per_head_auroc(attention_data: Dict, itype: str) -> Optional[pd.DataFrame]:
    """Compute AUROC for each individual attention head."""
    if itype not in attention_data or 'random' not in attention_data:
        return None
    
    pos_attn = attention_data[itype]['attention']
    neg_attn = attention_data['random']['attention']
    
    n_layers, n_heads = pos_attn.shape[1], pos_attn.shape[2]
    y_true = np.concatenate([np.ones(len(pos_attn)), np.zeros(len(neg_attn))])
    
    head_aurocs = []
    for layer in range(n_layers):
        for head in range(n_heads):
            y_score = np.concatenate([pos_attn[:, layer, head], neg_attn[:, layer, head]])
            auroc = roc_auc_score(y_true, y_score)
            head_aurocs.append({
                'layer': layer, 'head': head,
                'head_id': f'L{layer}H{head}',
                'auroc': auroc
            })
    
    return pd.DataFrame(head_aurocs)


def train_pairwise_classifier(attention_data: Dict, type_a: str, type_b: str) -> Dict:
    """Train logistic regression to distinguish between two interaction types."""
    if type_a not in attention_data or type_b not in attention_data:
        return {'accuracy': np.nan, 'auc': np.nan}
    
    # Prepare features (flattened attention)
    X_a = attention_data[type_a]['attention'].reshape(len(attention_data[type_a]['attention']), -1)
    X_b = attention_data[type_b]['attention'].reshape(len(attention_data[type_b]['attention']), -1)
    
    X = np.vstack([X_a, X_b])
    y = np.concatenate([np.ones(len(X_a)), np.zeros(len(X_b))])
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Scale and train
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    clf = LogisticRegression(max_iter=1000, C=0.1, random_state=42)
    clf.fit(X_train_scaled, y_train)
    
    # Evaluate
    accuracy = clf.score(X_test_scaled, y_test)
    y_prob = clf.predict_proba(X_test_scaled)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    
    return {'accuracy': accuracy, 'auc': auc, 'model': clf, 'scaler': scaler}


def compute_pc_removal_auroc_for_embeddings(
    embeddings: np.ndarray,
    protein_to_idx: Dict[str, int],
    attention_dir: Path,
    species: str,
    n_remove: int
) -> Dict:
    """Compute AUROC after removing top n principal components for a given embedding set."""
    pca = PCA(n_components=200)
    transformed = pca.fit_transform(embeddings)

    transformed_removed = transformed.copy()
    transformed_removed[:, :n_remove] = 0
    modified_embeddings = pca.inverse_transform(transformed_removed)

    return compute_embedding_auroc_vs_random(
        attention_dir=attention_dir,
        species=species,
        embeddings=modified_embeddings,
        protein_to_idx=protein_to_idx,
    )


def generate_table_s1(attention_dir: Path, output_dir: Path):
    """Generate Table S1: AUROC for discriminating interaction types from random pairs."""
    print("\n" + "=" * 70)
    print("TABLE S1: AUROC for discriminating interaction types from random pairs")
    print("=" * 70)
    
    results = []
    for species in SPECIES_LIST:
        row = {'Species': SPECIES_NAMES[species]}
        attention_data = load_attention_data(attention_dir, species, TABLE_TYPES + ['random'])
        
        for itype in TABLE_TYPES:
            auroc = compute_auroc_vs_random(attention_data, itype)
            row[TYPE_LABELS[itype]] = auroc
        
        results.append(row)
        print(f"  {species}: {row}")
    
    df = pd.DataFrame(results)
    df.to_csv(output_dir / 'table_s1_auroc_vs_random.csv', index=False)
    print(f"\n✓ Saved: {output_dir / 'table_s1_auroc_vs_random.csv'}")
    
    return df


def generate_table_s1_embeddings(attention_dir: Path, output_dir: Path):
    """Generate AUROC tables for cosine similarity baselines (inputs vs. functional)."""
    print("\n" + "=" * 70)
    print("TABLE S1B: AUROC vs. random (Cosine similarity baselines)")
    print("=" * 70)

    inputs_rows = []
    functional_rows = []

    for species in SPECIES_LIST:
        row_inputs = {'Species': SPECIES_NAMES[species]}
        row_functional = {'Species': SPECIES_NAMES[species]}

        encodings = load_functional_encodings(attention_dir, species)
        if encodings is None:
            inputs_rows.append(row_inputs)
            functional_rows.append(row_functional)
            continue

        inputs_aurocs = compute_embedding_auroc_vs_random(
            attention_dir=attention_dir,
            species=species,
            embeddings=encodings['inputs_embeds'],
            protein_to_idx=encodings['protein_to_idx'],
        )
        functional_aurocs = {}
        if not encodings.get('group_equals_inputs', False):
            functional_aurocs = compute_embedding_auroc_vs_random(
                attention_dir=attention_dir,
                species=species,
                embeddings=encodings['group_embeds'],
                protein_to_idx=encodings['protein_to_idx'],
            )
        else:
            print(f"  {species}: functional group_embeds identical to inputs_embeds (source={encodings.get('group_source', 'unknown')}); writing NaN for functional baseline")

        for itype in TABLE_TYPES:
            row_inputs[TYPE_LABELS[itype]] = inputs_aurocs.get(itype, np.nan)
            row_functional[TYPE_LABELS[itype]] = functional_aurocs.get(itype, np.nan)

        inputs_rows.append(row_inputs)
        functional_rows.append(row_functional)

        print(f"  {species}: inputs={row_inputs}, functional={row_functional}")

    df_inputs = pd.DataFrame(inputs_rows)
    df_functional = pd.DataFrame(functional_rows)

    df_inputs.to_csv(output_dir / 'table_s1b_inputs_cosine.csv', index=False)
    df_functional.to_csv(output_dir / 'table_s1b_functional_cosine.csv', index=False)
    print(f"\n✓ Saved: {output_dir / 'table_s1b_inputs_cosine.csv'}")
    print(f"✓ Saved: {output_dir / 'table_s1b_functional_cosine.csv'}")

    return df_inputs, df_functional


def generate_table_s2(attention_dir: Path, output_dir: Path):
    """Generate Table S2: Pairwise binary classification AUROC between interaction types."""
    print("\n" + "=" * 70)
    print("TABLE S2: Pairwise binary classification AUROC")
    print("=" * 70)
    
    # Comparisons: Direct vs. Same complex, Direct vs. Coexpression, Direct vs. Random
    comparisons = [
        ('pdb', 'pdb_physical', 'Direct vs. Same complex'),
        ('pdb', 'coexpression', 'Direct vs. Coexpression'),
        ('pdb', 'random', 'Direct vs. Random')
    ]
    
    results = []
    for species in SPECIES_LIST:
        row = {'Species': SPECIES_NAMES[species]}
        attention_data = load_attention_data(attention_dir, species, TABLE_TYPES + ['random'])
        
        for type_a, type_b, label in comparisons:
            clf_result = train_pairwise_classifier(attention_data, type_a, type_b)
            row[label] = clf_result['auc']
        
        results.append(row)
        print(f"  {species}: Direct vs. Same complex={row['Direct vs. Same complex']:.3f}, "
              f"Direct vs. Coexpression={row['Direct vs. Coexpression']:.3f}, "
              f"Direct vs. Random={row['Direct vs. Random']:.3f}")
    
    df = pd.DataFrame(results)
    df.to_csv(output_dir / 'table_s2_pairwise_classification.csv', index=False)
    print(f"\n✓ Saved: {output_dir / 'table_s2_pairwise_classification.csv'}")
    
    return df


def plot_attention_by_type(attention_dir: Path, output_dir: Path):
    """Plot attention head AUROC by interaction type for each species."""
    print("\n" + "=" * 70)
    print("Generating attention by interaction type figures...")
    print("=" * 70)
    
    for species in SPECIES_LIST:
        attention_data = load_attention_data(attention_dir, species, TABLE_TYPES + ['random'])
        
        if 'random' not in attention_data:
            print(f"  {species}: No random data, skipping")
            continue
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'ProteomeLM Attention by Interaction Type - {SPECIES_NAMES_ITALIC[species]}', 
                     fontsize=18, fontweight='bold')
        
        for idx, itype in enumerate(TABLE_TYPES):
            if itype not in attention_data:
                axes[idx].text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=14, transform=axes[idx].transAxes)
                axes[idx].set_title(TYPE_LABELS[itype])
                continue
            
            head_aurocs = compute_per_head_auroc(attention_data, itype)
            if head_aurocs is None:
                continue
            
            # Sort by AUROC
            sorted_aurocs = head_aurocs.sort_values('auroc', ascending=False)['auroc'].values
            
            # Plot
            ax = axes[idx]
            ax.plot(range(len(sorted_aurocs)), sorted_aurocs, color=COLORS[itype], lw=2, label='Attention heads')
            ax.axhline(y=0.5, color='gray', linestyle='--', lw=1, alpha=0.7, label='Random baseline')
            ax.fill_between(range(len(sorted_aurocs)), 0.5, sorted_aurocs, alpha=0.3, color=COLORS[itype])
            
            ax.set_xlabel('Attention head (sorted)')
            ax.set_ylabel('AUROC vs. Random')
            ax.set_title(f'{TYPE_LABELS[itype]}\nBest: {sorted_aurocs[0]:.3f}, Mean: {sorted_aurocs.mean():.3f}')
            ax.set_ylim(0.4, 1.0)
            ax.legend(loc='lower left', fontsize=12)
            ax.grid(True, alpha=0.3)
            style_axes(ax)
        
        plt.tight_layout()
        
        # Save in multiple formats
        for fmt in ['pdf', 'svg', 'png']:
            fig.savefig(output_dir / f'{species}_attention_by_type.{fmt}', dpi=300, bbox_inches='tight')
        
        plt.close()
        print(f"  ✓ {species}: Saved attention_by_type figures")


def plot_heads_vs_cosine_vs_pca(attention_dir: Path, output_dir: Path):
    """
    Key figure: Compare attention heads AUROC vs. cosine similarity vs. PCA removal.
    Addresses Reviewer 2's MirrorTree concern.
    """
    print("\n" + "=" * 70)
    print("Generating heads vs. cosine vs. PCA figures...")
    print("=" * 70)
    
    n_remove_list = [0, 1, 2, 3, 5, 10, 20, 50, 100, 150]
    
    for species in SPECIES_LIST:
        attention_data = load_attention_data(attention_dir, species, TABLE_TYPES + ['random'])
        encodings = load_functional_encodings(attention_dir, species)
        
        if encodings is None or 'random' not in attention_data:
            print(f"  {species}: Missing data, skipping")
            continue

        has_distinct_functional = not encodings.get('group_equals_inputs', False)
        if not has_distinct_functional:
            print(f"  {species}: functional group_embeds identical to inputs_embeds; skipping functional-only curves")
        
        # Compute PC removal AUROC (inputs + functional)
        pc_results_inputs = {}
        pc_results_functional = {}
        for n_remove in n_remove_list:
            pc_inputs = compute_pc_removal_auroc_for_embeddings(
                embeddings=encodings['inputs_embeds'],
                protein_to_idx=encodings['protein_to_idx'],
                attention_dir=attention_dir,
                species=species,
                n_remove=n_remove,
            )
            pc_functional = {}
            if has_distinct_functional:
                pc_functional = compute_pc_removal_auroc_for_embeddings(
                    embeddings=encodings['group_embeds'],
                    protein_to_idx=encodings['protein_to_idx'],
                    attention_dir=attention_dir,
                    species=species,
                    n_remove=n_remove,
                )

            for itype, auroc in pc_inputs.items():
                if itype not in pc_results_inputs:
                    pc_results_inputs[itype] = {}
                pc_results_inputs[itype][n_remove] = auroc

            for itype, auroc in pc_functional.items():
                if itype not in pc_results_functional:
                    pc_results_functional[itype] = {}
                pc_results_functional[itype][n_remove] = auroc
        
        # Plot for each interaction type
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        fig.suptitle(f'Attention vs. Encoding Similarity vs. PCA Removal - {SPECIES_NAMES_ITALIC[species]}',
                     fontsize=18, fontweight='bold')
        
        for idx, itype in enumerate(TABLE_TYPES):
            ax = axes[idx]
            
            # 1. Attention heads (coral line)
            head_aurocs = compute_per_head_auroc(attention_data, itype)
            if head_aurocs is not None:
                sorted_aurocs = head_aurocs.sort_values('auroc', ascending=False)['auroc'].values
                ax.plot(range(len(sorted_aurocs)), sorted_aurocs, 
                       color=colorspal6[1], lw=2, label=f'Attention heads (best={sorted_aurocs[0]:.3f})')
            
            # 2. Encoding similarity baselines (inputs + functional)
            if itype in pc_results_inputs and 0 in pc_results_inputs[itype]:
                enc_sim_auroc = pc_results_inputs[itype][0]
                ax.axhline(y=enc_sim_auroc, color=colorspal6[4], linestyle='-', lw=2,
                          label=f'Cosine inputs ({enc_sim_auroc:.3f})')

            if itype in pc_results_functional and 0 in pc_results_functional[itype]:
                enc_func_auroc = pc_results_functional[itype][0]
                ax.axhline(y=enc_func_auroc, color=colorspal6[5], linestyle='-', lw=2,
                          label=f'Cosine functional ({enc_func_auroc:.3f})')
            
            # 3. PCA removal curves (inputs + functional)
            if itype in pc_results_inputs:
                pc_x = []
                pc_y = []
                for n_rem in n_remove_list:
                    if n_rem in pc_results_inputs[itype]:
                        pc_x.append(n_rem)
                        pc_y.append(pc_results_inputs[itype][n_rem])
                
                if pc_x:
                    # Scale x to fit head axis
                    n_heads = len(sorted_aurocs) if head_aurocs is not None else 216
                    ax.plot(pc_x[:n_heads], pc_y, color=colorspal6[4], linestyle='--', lw=2, marker='o', 
                           label='PCA removal (inputs)')
                    
            if itype in pc_results_functional:
                pc_x = []
                pc_y = []
                for n_rem in n_remove_list:
                    if n_rem in pc_results_functional[itype]:
                        pc_x.append(n_rem)
                        pc_y.append(pc_results_functional[itype][n_rem])

                if pc_x:
                    n_heads = len(sorted_aurocs) if head_aurocs is not None else 216
                    ax.plot(pc_x[:n_heads], pc_y, color=colorspal6[5], linestyle='--', lw=2, marker='o',
                           label='PCA removal (functional)')
            
            ax.axhline(y=0.5, color='gray', linestyle=':', lw=1, alpha=0.5)
            ax.set_xlabel('Attention head (sorted) / # PCA removed')
            ax.set_ylabel('AUROC vs. Random')
            ax.set_title(TYPE_LABELS[itype])
            ax.set_ylim(0.3, 1.0)
            ax.legend(loc='lower left', fontsize=12)
            style_axes(ax)
        
        plt.tight_layout()
        
        for fmt in ['pdf', 'svg', 'png']:
            fig.savefig(output_dir / f'{species}_heads_vs_cosine_vs_pca.{fmt}', dpi=300, bbox_inches='tight')
        
        plt.close()
        print(f"  ✓ {species}: Saved heads_vs_cosine_vs_pca figures")


def plot_pairwise_classification_heads(attention_dir: Path, output_dir: Path):
    """Plot attention head importance for pairwise classification."""
    print("\n" + "=" * 70)
    print("Generating pairwise classification head importance figures...")
    print("=" * 70)
    
    comparisons = [
        ('pdb', 'pdb_physical', 'Direct vs. Same complex'),
        ('pdb', 'coexpression', 'Direct vs. Coexpression'),
        ('pdb', 'random', 'Direct vs. Random')
    ]
    
    for species in SPECIES_LIST:
        attention_data = load_attention_data(attention_dir, species, TABLE_TYPES + ['random'])
        
        if len(attention_data) < 2:
            print(f"  {species}: Insufficient data, skipping")
            continue
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'Logistic Regression Coefficients by Attention Head - {SPECIES_NAMES_ITALIC[species]}',
                     fontsize=18, fontweight='bold')
        
        for idx, (type_a, type_b, label) in enumerate(comparisons):
            ax = axes[idx]
            
            if type_a not in attention_data or type_b not in attention_data:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=14, transform=ax.transAxes)
                ax.set_title(label)
                continue
            
            # Train classifier
            clf_result = train_pairwise_classifier(attention_data, type_a, type_b)
            
            if 'model' not in clf_result:
                continue
            
            # Get coefficients
            coefs = clf_result['model'].coef_[0]
            
            # Reshape to (layers, heads) if possible
            n_layers = attention_data[type_a]['attention'].shape[1]
            n_heads = attention_data[type_a]['attention'].shape[2]
            
            if len(coefs) == n_layers * n_heads:
                coef_matrix = coefs.reshape(n_layers, n_heads)
                
                # Heatmap
                im = ax.imshow(coef_matrix, aspect='auto', cmap='RdBu_r', 
                              vmin=-np.abs(coef_matrix).max(), vmax=np.abs(coef_matrix).max())
                ax.set_xlabel('Head')
                ax.set_ylabel('Layer')
                ax.set_title(f'{label}\nAcc={clf_result["accuracy"]:.3f}, AUC={clf_result["auc"]:.3f}')
                cbar = plt.colorbar(im, ax=ax, label='Coefficient')
                cbar.ax.tick_params(labelsize=13)
                style_axes(ax)
            else:
                # Bar plot for flattened coefficients
                sorted_idx = np.argsort(np.abs(coefs))[::-1][:50]
                ax.bar(range(len(sorted_idx)), coefs[sorted_idx])
                ax.set_xlabel('Top 50 features')
                ax.set_ylabel('Coefficient')
                ax.set_title(f'{label}\nAcc={clf_result["accuracy"]:.3f}')
                style_axes(ax)
        
        plt.tight_layout()
        
        for fmt in ['pdf', 'svg', 'png']:
            fig.savefig(output_dir / f'{species}_pairwise_classification_heads.{fmt}', dpi=300, bbox_inches='tight')
        
        plt.close()
        print(f"  ✓ {species}: Saved pairwise_classification_heads figures")


# Species-level colors (matching notebook)
SPECIES_COLORS = {
    'ecoli':  colorspal6[0],
    'yeast':  colorspal6[5],
    'human':  colorspal6[2],
}


def plot_auroc_summary_bars(attention_dir: Path, output_dir: Path):
    """
    Combined bar chart:
      Left block  — each interaction type vs. Random (AUROC, best attention head)
      Right block — pairwise classification AUC between interaction types
    One group of bars per species, coloured by species.
    """
    print("\n" + "=" * 70)
    print("Generating AUROC summary bar figure...")
    print("=" * 70)

    # --- define the two blocks of comparisons ---
    vs_random_types = TABLE_TYPES  # pdb, pdb_physical, coexpression
    vs_random_labels = [f'{TYPE_LABELS[t]} vs. Random' for t in vs_random_types]

    pairwise_comparisons = [
        ('pdb', 'pdb_physical',  'Direct vs.\nSame complex'),
        ('pdb', 'coexpression',  'Direct vs.\nCoexpression'),
        ('pdb_physical', 'coexpression', 'Same complex vs.\nCoexpression'),
    ]
    pairwise_labels = [lbl for _, _, lbl in pairwise_comparisons]

    n_left  = len(vs_random_labels)
    n_right = len(pairwise_labels)
    gap = 0.8  # extra space between the two blocks

    # x positions with a gap between the two blocks
    x_left  = np.arange(n_left)
    x_right = np.arange(n_right) + n_left + gap
    x_all   = np.concatenate([x_left, x_right])

    available_species = [s for s in SPECIES_LIST if s in SPECIES_COLORS]

    n_species = len(available_species)
    bar_width = 0.8 / n_species
    offsets = np.linspace(-(n_species - 1) * bar_width / 2,
                           (n_species - 1) * bar_width / 2,
                           n_species)

    fig, ax = plt.subplots(figsize=(11, 5))
    plt.rcParams['font.family'] = 'Arial'

    for sp_idx, species in enumerate(available_species):
        attention_data = load_attention_data(
            attention_dir, species, TABLE_TYPES + ['random'])
        if 'random' not in attention_data:
            print(f"  {species}: no random data, skipping")
            continue

        color = SPECIES_COLORS[species]

        vals = []

        # --- left block: type vs. random (best head AUROC) ---
        for itype in vs_random_types:
            auroc = compute_auroc_vs_random(attention_data, itype)
            vals.append(auroc)

        # --- right block: pairwise classification AUC ---
        for type_a, type_b, _ in pairwise_comparisons:
            res = train_pairwise_classifier(attention_data, type_a, type_b)
            vals.append(res['auc'])

        vals = np.array(vals, dtype=float)

        bars = ax.bar(
            x_all + offsets[sp_idx],
            vals,
            width=bar_width,
            color=color,
            alpha=0.85,
            edgecolor='white',
            linewidth=0.5,
            label=SPECIES_NAMES_ITALIC[species],
        )

        # value labels on top of bars
        for bar_rect, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar_rect.get_x() + bar_rect.get_width() / 2, v + 0.008,
                        f'{v:.2f}', ha='center', va='bottom', fontsize=14,
                        color='#333333', fontfamily='Arial')

    # --- cosmetics ---
    ax.set_xticks(x_all)
    ax.set_xticklabels(
        [f'{TYPE_LABELS[t]}\nvs. Random' for t in vs_random_types] + pairwise_labels,
        fontsize=15, fontfamily='Arial')
    ax.set_ylabel('AUROC', fontsize=16, fontfamily='Arial')
    ax.tick_params(axis='both', labelsize=14)
    ax.set_ylim(0.3, 1.05)
    ax.axhline(0.5, color='gray', linestyle=':', lw=1, alpha=0.6)

    # add a vertical separator between the two blocks
    sep_x = (x_left[-1] + x_right[0]) / 2
    ax.axvline(sep_x, color='#cccccc', linestyle='-', lw=0.8)
    ax.text(np.mean(x_left), 1.02, 'vs. Random', ha='center', fontsize=16,
            fontweight='bold', fontfamily='Arial', transform=ax.get_xaxis_transform())
    ax.text(np.mean(x_right), 1.02, 'Pairwise', ha='center', fontsize=16,
            fontweight='bold', fontfamily='Arial', transform=ax.get_xaxis_transform())

    ax.legend(fontsize=14, frameon=False, loc='lower right', prop={'family': 'Arial'})
    style_axes(ax)
    ax.grid(True, alpha=0.15, axis='y')
    ax.grid(False, axis='x')
    plt.title('ProteomeLM Attention: Interaction-Type Discrimination',
              fontweight='bold', fontsize=18, fontfamily='Arial', pad=20)
    plt.tight_layout()

    for fmt in ['pdf', 'svg', 'png']:
        fig.savefig(output_dir / f'auroc_summary_bars.{fmt}', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved auroc_summary_bars figure")


def generate_summary_figure(attention_dir: Path, output_dir: Path, table_s1: pd.DataFrame, table_s2: pd.DataFrame):
    """Generate combined summary figure with tables and key metrics."""
    print("\n" + "=" * 70)
    print("Generating summary figure...")
    print("=" * 70)
    
    fig = plt.figure(figsize=(16, 10))
    
    # Create grid
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Table S1 heatmap
    ax1 = fig.add_subplot(gs[0, 0])
    s1_data = table_s1.set_index('Species')[[TYPE_LABELS[t] for t in TABLE_TYPES]]
    im1 = ax1.imshow(s1_data.values, aspect='auto', cmap='YlOrRd', vmin=0.5, vmax=1.0)
    ax1.set_xticks(range(len(s1_data.columns)))
    ax1.set_xticklabels(s1_data.columns, rotation=45, ha='right', fontsize=14)
    ax1.set_yticks(range(len(s1_data.index)))
    ax1.set_yticklabels([italicize_species_label(s) for s in s1_data.index], fontsize=14)
    ax1.set_title('Table S1: AUROC vs. Random', fontweight='bold', fontsize=17)
    
    # Add values
    for i in range(len(s1_data.index)):
        for j in range(len(s1_data.columns)):
            val = s1_data.values[i, j]
            if not np.isnan(val):
                ax1.text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=12,
                        color='white' if val > 0.75 else 'black')
    
    cbar1 = plt.colorbar(im1, ax=ax1, label='AUROC')
    cbar1.ax.tick_params(labelsize=13)
    style_axes(ax1)
    
    # Table S2 heatmap
    ax2 = fig.add_subplot(gs[0, 1])
    s2_data = table_s2.set_index('Species')
    im2 = ax2.imshow(s2_data.values, aspect='auto', cmap='YlGnBu', vmin=0.5, vmax=1.0)
    ax2.set_xticks(range(len(s2_data.columns)))
    ax2.set_xticklabels(s2_data.columns, rotation=45, ha='right', fontsize=14)
    ax2.set_yticks(range(len(s2_data.index)))
    ax2.set_yticklabels([italicize_species_label(s) for s in s2_data.index], fontsize=14)
    ax2.set_title('Table S2: Pairwise Classification AUROC', fontweight='bold', fontsize=17)
    
    for i in range(len(s2_data.index)):
        for j in range(len(s2_data.columns)):
            val = s2_data.values[i, j]
            if not np.isnan(val):
                ax2.text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=12,
                        color='white' if val > 0.75 else 'black')
    
    cbar2 = plt.colorbar(im2, ax=ax2, label='AUROC')
    cbar2.ax.tick_params(labelsize=13)
    style_axes(ax2)
    
    # Species comparison bar chart (Table S1)
    ax3 = fig.add_subplot(gs[1, 0])
    x = np.arange(len(SPECIES_LIST))
    width = 0.25
    
    for i, itype in enumerate(TABLE_TYPES):
        col_name = TYPE_LABELS[itype]
        if col_name in table_s1.columns:
            values = table_s1[col_name].values
            ax3.bar(x + i*width, values, width, label=col_name, color=COLORS[itype])
    
    ax3.set_xlabel('Species')
    ax3.set_ylabel('AUROC vs. Random')
    ax3.set_title('AUROC by Species and Interaction Type', fontweight='bold')
    ax3.set_xticks(x + width)
    ax3.set_xticklabels([SPECIES_NAMES_ITALIC[s] for s in SPECIES_LIST])
    ax3.legend(loc='lower right', fontsize=13)
    ax3.set_ylim(0.5, 1.0)
    ax3.grid(True, alpha=0.3, axis='y')
    style_axes(ax3)
    
    # Pairwise classification comparison
    ax4 = fig.add_subplot(gs[1, 1])
    comparisons_short = ['Direct vs.\nSame complex', 'Direct vs.\nCoexpression', 'Direct vs.\nRandom']
    
    for i, species in enumerate(SPECIES_LIST):
        species_row = table_s2[table_s2['Species'] == SPECIES_NAMES[species]]
        if not species_row.empty:
            values = species_row.iloc[0, 1:].values
            ax4.bar(x + i*width, values, width, label=SPECIES_NAMES_ITALIC[species])
    
    ax4.set_xlabel('Classification Task')
    ax4.set_ylabel('AUROC')
    ax4.set_title('Pairwise Classification by Species', fontweight='bold')
    ax4.set_xticks(x + width)
    ax4.set_xticklabels(comparisons_short)
    ax4.legend(loc='lower right', fontsize=13)
    ax4.set_ylim(0.5, 1.0)
    ax4.grid(True, alpha=0.3, axis='y')
    style_axes(ax4)
    
    plt.suptitle('ProteomeLM Benchmark Summary', fontsize=20, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    for fmt in ['pdf', 'svg', 'png']:
        fig.savefig(output_dir / f'summary_figure.{fmt}', dpi=300, bbox_inches='tight')
    
    plt.close()
    print(f"  ✓ Saved summary figure")


def print_latex_tables(table_s1: pd.DataFrame, table_s2: pd.DataFrame):
    """Print LaTeX formatted tables for manuscript."""
    print("\n" + "=" * 70)
    print("LATEX TABLES FOR MANUSCRIPT")
    print("=" * 70)
    
    print("\n% Table S1")
    print("\\begin{table}[h]")
    print("\\caption{AUROC for discriminating interaction types from random pairs.}")
    print("\\label{tab:auroc_vs_random}")
    print("\\centering")
    print("\\begin{tabular}{lccc}")
    print("\\toprule")
    print("Species & Direct (PDB) & Same complex (PDB) & Coexpression (STRING) \\\\")
    print("\\midrule")
    
    for _, row in table_s1.iterrows():
        values = [f"{row[TYPE_LABELS[t]]:.3f}" if not pd.isna(row.get(TYPE_LABELS[t], np.nan)) else "-" 
                  for t in TABLE_TYPES]
        species_label = f"\\textit{{{row['Species']}}}"
        print(f"{species_label} & {' & '.join(values)} \\")
    
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")
    
    print("\n% Table S2")
    print("\\begin{table}[h]")
    print("\\caption{Pairwise binary classification AUROC between interaction types.}")
    print("\\label{tab:pairwise_classification}")
    print("\\centering")
    print("\\begin{tabular}{lccc}")
    print("\\toprule")
    print("Species & Direct vs. Same complex & Direct vs. Coexpression & Direct vs. Random \\\\")
    print("\\midrule")
    
    for _, row in table_s2.iterrows():
        values = [f"{row[col]:.3f}" if not pd.isna(row.get(col, np.nan)) else "-" 
                  for col in ['Direct vs. Same complex', 'Direct vs. Coexpression', 'Direct vs. Random']]
        species_label = f"\\textit{{{row['Species']}}}"
        print(f"{species_label} & {' & '.join(values)} \\")
    
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="ProteomeLM differential-interactome analysis")
    parser.add_argument('--attention-dir', type=str, default='attention_patterns',
                       help='Directory containing attention pattern files')
    parser.add_argument('--output-dir', type=str, default='figures_minimal',
                       help='Output directory for figures and tables')
    args = parser.parse_args()
    
    attention_dir = Path(args.attention_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("ProteomeLM Minimal Analysis Pipeline")
    print("=" * 70)
    print(f"Attention directory: {attention_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Species: {', '.join(SPECIES_LIST)}")
    print(f"Interaction types: {', '.join(TABLE_TYPES)}")
    
    # Check for data
    available_species = []
    for species in SPECIES_LIST:
        random_file = attention_dir / f"{species}_random_attention.npz"
        if random_file.exists():
            available_species.append(species)
    
    if not available_species:
        print("\n⚠ No attention data found. Run extract_attention.py first.")
        print(f"  Looking in: {attention_dir}")
        return
    
    print(f"\nAvailable species: {', '.join(available_species)}")
    
    # Set matplotlib style
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = (12, 7)
    plt.rcParams['font.size'] = 16
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['axes.titlesize'] = 18
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14
    plt.rcParams['legend.fontsize'] = 13
    plt.rcParams['figure.titlesize'] = 20
    
    # Generate tables
    table_s1 = generate_table_s1(attention_dir, output_dir)
    table_s2 = generate_table_s2(attention_dir, output_dir)
    table_s1_inputs, table_s1_functional = generate_table_s1_embeddings(attention_dir, output_dir)
    
    # Generate figures
    plot_attention_by_type(attention_dir, output_dir)
    plot_heads_vs_cosine_vs_pca(attention_dir, output_dir)
    plot_pairwise_classification_heads(attention_dir, output_dir)
    plot_auroc_summary_bars(attention_dir, output_dir)
    generate_summary_figure(attention_dir, output_dir, table_s1, table_s2)
    
    # Print LaTeX tables
    print_latex_tables(table_s1, table_s2)
    
    print("\n" + "=" * 70)
    print("✓ Analysis Complete!")
    print("=" * 70)
    print(f"\nOutput files in {output_dir}:")
    for f in sorted(output_dir.glob('*')):
        print(f"  - {f.name}")


if __name__ == '__main__':
    main()
