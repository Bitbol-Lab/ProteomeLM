#!/usr/bin/env python3
"""
Ablation utilities for comparing attention AUROC between models.

Subcommands:
  extract  - generate attention patterns for a model checkpoint
  compare  - compare per-head AUROC and best-head heatmaps between two models
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import roc_auc_score

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

SPECIES_PROTEOME_IDS = {
    'yeast': 'UP000002311',
    'human': 'UP000005640',
    'ecoli': 'UP000000625',
}

SPECIES_LIST = ['ecoli', 'yeast', 'human']
SPECIES_NAMES = {'ecoli': 'E. coli', 'yeast': 'S. cerevisiae', 'human': 'H. sapiens'}

colorspal6 = [
    (0.25098039215686274, 0.3254901960784314, 0.8274509803921568),
    (0.8666666666666667, 0.7019607843137254, 0.06274509803921569),
    (0.7098039215686275, 0.11372549019607843, 0.0784313725490196),
    (0.0, 0.7450980392156863, 1.0),
    (0.984313725490196, 0.28627450980392155, 0.6901960784313725),
    (0.0, 0.6980392156862745, 0.36470588235294116),
    (0.792156862745098, 0.792156862745098, 0.792156862745098),
]

SPECIES_COLORS = {
    'ecoli': colorspal6[0],
    'yeast': colorspal6[5],
    'human': colorspal6[2],
}

SPECIES_LEGEND_LABELS = {
    'ecoli': r'$\it{E.\ coli}$',
    'yeast': r'$\it{S.\ cerevisiae}$',
    'human': r'$\it{H.\ sapiens}$',
}

TABLE_TYPES = ['pdb', 'pdb_physical', 'coexpression']
TYPE_LABELS = {
    'pdb': 'Direct (PDB)',
    'pdb_physical': 'Same complex (PDB)',
    'coexpression': 'Coexpression (STRING)',
    'random': 'Random',
}

COLORS = {
    'pdb': '#2B3AC0',
    'pdb_physical': '#E74C3C',
    'coexpression': '#F39C12',
    'random': '#7F8C8D',
}


def parse_fasta(fasta_file: Path) -> Dict[str, str]:
    """Parse FASTA file to dict of {id: sequence}."""
    sequences = {}
    current_id, current_seq = None, []

    with open(fasta_file) as f:
        for line in f:
            if line.startswith('>'):
                if current_id:
                    sequences[current_id] = ''.join(current_seq)
                header = line[1:].split()[0]
                current_id = header.split('|')[1] if '|' in header else header
                current_seq = []
            else:
                current_seq.append(line.strip())
        if current_id:
            sequences[current_id] = ''.join(current_seq)

    return sequences


def create_protein_mapping(fasta_file: Path) -> Dict[str, int]:
    """Create mapping from protein IDs to proteome index."""
    mapping = {}
    idx = 0

    with open(fasta_file) as f:
        for line in f:
            if line.startswith('>'):
                header = line[1:].strip()
                parts = header.split('|')
                uniprot_id = parts[1] if len(parts) > 1 else header.split()[0]
                mapping[uniprot_id] = idx

                if 'GN=' in header:
                    gene = header.split('GN=')[1].split()[0]
                    mapping[gene] = idx

                if match := re.search(r'Y[A-P][LR]\d{3}[WC](?:-[A-Z])?', header):
                    mapping[match.group(0)] = idx

                idx += 1

    return mapping


def extract_attention(
    pairs: List[Tuple[str, str]],
    protein_to_idx: Dict[str, int],
    attentions: List[np.ndarray],
    max_pairs: int = 10000,
) -> Optional[Dict]:
    """Extract attention values for protein pairs."""
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

    return {'pair_ids': valid_pairs, 'attention_a_to_b': attn_ab, 'attention_b_to_a': attn_ba}


def generate_negatives(
    proteins: List[str],
    positives: pd.DataFrame,
    n: int,
    seed: int = 42,
) -> List[Tuple[str, str]]:
    """Generate random negative pairs."""
    np.random.seed(seed)
    pos_set = set(zip(positives['protein_a'], positives['protein_b']))
    negatives = []

    for _ in range(n * 10):
        a, b = tuple(sorted(np.random.choice(proteins, 2, replace=False)))
        if (a, b) not in pos_set:
            negatives.append((a, b))
            if len(negatives) >= n:
                break

    return negatives


def build_orthodb_ids_for_proteome(
    fasta_file: Path,
    orthodb_tsv: Path,
    orthodb_vocab: Dict[str, int],
) -> torch.Tensor:
    """Build OrthoDB ID indices aligned to proteome order."""
    from proteomelm.utils.proteome import parse_orthodb_tsv

    sequences = parse_fasta(fasta_file)
    protein_ids = list(sequences.keys())
    uniprot_to_ogs = parse_orthodb_tsv(str(orthodb_tsv))

    orthodb_ids: List[int] = []
    n_mapped = 0
    for pid in protein_ids:
        og_ids = uniprot_to_ogs.get(pid, [])
        mapped_idx = 0
        for og_id in og_ids:
            if og_id in orthodb_vocab:
                mapped_idx = orthodb_vocab[og_id]
                break
        if mapped_idx != 0:
            n_mapped += 1
        orthodb_ids.append(mapped_idx)

    print(f"  OrthoDB IDs mapped: {n_mapped}/{len(orthodb_ids)}")
    return torch.tensor(orthodb_ids, dtype=torch.long)


def load_attention_data(attention_dir: Path, species: str, types: List[str]) -> Dict:
    """Load attention data for specified types."""
    data = {}
    for itype in types:
        path = attention_dir / f"{species}_{itype}_attention.npz"
        if path.exists():
            npz = np.load(path, allow_pickle=True)
            attn = (npz['attention_a_to_b'] + npz['attention_b_to_a']) / 2
            data[itype] = {
                'attention': attn,
                'pairs': [tuple(p) for p in npz['pair_ids']],
            }
    return data


def compute_auroc_vs_random(attention_data: Dict, itype: str) -> float:
    """Compute best AUROC for interaction type vs random using attention heads."""
    if itype not in attention_data or 'random' not in attention_data:
        return np.nan

    pos_attn = attention_data[itype]['attention']
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
                'layer': layer,
                'head': head,
                'head_id': f'L{layer}H{head}',
                'auroc': auroc,
            })

    return pd.DataFrame(head_aurocs)


def compute_table_s1_data(attention_dir: Path) -> pd.DataFrame:
    """Compute Table S1 data without writing files."""
    results = []
    for species in SPECIES_LIST:
        row = {'Species': SPECIES_NAMES[species]}
        attention_data = load_attention_data(attention_dir, species, TABLE_TYPES + ['random'])

        for itype in TABLE_TYPES:
            auroc = compute_auroc_vs_random(attention_data, itype)
            row[TYPE_LABELS[itype]] = auroc

        results.append(row)

    return pd.DataFrame(results)


def plot_attention_by_type_compare(
    attention_dir_a: Path,
    attention_dir_b: Path,
    label_a: str,
    label_b: str,
    output_dir: Path,
):
    """Compact comparison: one row of interaction types with species-overlaid curves."""
    print("\n" + "=" * 70)
    print("Generating attention by interaction type comparison figures...")
    print("=" * 70)

    title_fs = 36
    subplot_title_fs = 28
    axis_label_fs = 28
    tick_fs = 20
    legend_fs = 20
    legend_title_fs = 22

    fig, axes = plt.subplots(1, len(TABLE_TYPES), figsize=(24, 6), sharey=True)
    fig.suptitle('Attention by Interaction Type (All Species)', fontsize=title_fs, fontweight='bold')

    for tidx, itype in enumerate(TABLE_TYPES):
        ax = axes[tidx]
        plotted_any = False

        for species in SPECIES_LIST:
            attention_a = load_attention_data(attention_dir_a, species, TABLE_TYPES + ['random'])
            attention_b = load_attention_data(attention_dir_b, species, TABLE_TYPES + ['random'])

            if 'random' not in attention_a or 'random' not in attention_b:
                continue
            if itype not in attention_a or itype not in attention_b:
                continue

            head_aurocs_a = compute_per_head_auroc(attention_a, itype)
            head_aurocs_b = compute_per_head_auroc(attention_b, itype)
            if head_aurocs_a is None or head_aurocs_b is None:
                continue

            sorted_a = head_aurocs_a.sort_values('auroc', ascending=False)['auroc'].values
            sorted_b = head_aurocs_b.sort_values('auroc', ascending=False)['auroc'].values
            species_color = SPECIES_COLORS[species]

            ax.plot(
                range(len(sorted_a)),
                sorted_a,
                color=species_color,
                lw=2.8,
            )
            ax.plot(
                range(len(sorted_b)),
                sorted_b,
                color=species_color,
                lw=2.8,
                linestyle='--',
            )
            plotted_any = True

        ax.axhline(y=0.5, color='gray', linestyle='--', lw=1.6, alpha=0.7)
        ax.set_title(TYPE_LABELS[itype], fontsize=subplot_title_fs, fontweight='bold')
        ax.set_xlabel('Attention head (sorted)', fontsize=axis_label_fs)
        if tidx == 0:
            ax.set_ylabel('AUROC vs Random', fontsize=axis_label_fs)
        ax.set_ylim(0.4, 1.0)
        ax.tick_params(labelsize=tick_fs)
        if not plotted_any:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes, fontsize=axis_label_fs)

    standard_handles = [
        Line2D([0], [0], color=SPECIES_COLORS[s], lw=3.2, linestyle='-', label=SPECIES_LEGEND_LABELS[s])
        for s in SPECIES_LIST
    ]
    orthodb_discrete_handles = [
        Line2D([0], [0], color=SPECIES_COLORS[s], lw=3.2, linestyle='--', label=SPECIES_LEGEND_LABELS[s])
        for s in SPECIES_LIST
    ]

    legend_standard = fig.legend(
        handles=standard_handles,
        loc='upper center',
        bbox_to_anchor=(0.33, 1.02),
        ncol=3,
        fontsize=legend_fs,
        frameon=True,
        title='ProteomeLM',
        title_fontsize=legend_title_fs,
    )
    legend_orthodb_discrete = fig.legend(
        handles=orthodb_discrete_handles,
        loc='upper center',
        bbox_to_anchor=(0.77, 1.02),
        ncol=3,
        fontsize=legend_fs,
        frameon=True,
        title='ProteomeLM-Discrete',
        title_fontsize=legend_title_fs,
    )
    legend_standard.get_title().set_fontweight('bold')
    legend_orthodb_discrete.get_title().set_fontweight('bold')
    fig.add_artist(legend_standard)
    fig.add_artist(legend_orthodb_discrete)

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    for fmt in ['pdf', 'svg', 'png']:
        fig.savefig(
            output_dir / f'attention_by_type_compare_all_species.{fmt}',
            dpi=300,
            bbox_inches='tight',
        )
    plt.close()
    print("  Saved attention_by_type_compare_all_species figure")


def plot_table_s1_heatmap_compare(
    attention_dir_a: Path,
    attention_dir_b: Path,
    label_a: str,
    label_b: str,
    output_dir: Path,
):
    """Side-by-side heatmaps of best-head AUROC vs random for two models."""
    print("\n" + "=" * 70)
    print("Generating Table S1 heatmap comparison...")
    print("=" * 70)

    table_a = compute_table_s1_data(attention_dir_a)
    table_b = compute_table_s1_data(attention_dir_b)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, table, title in zip(axes, [table_a, table_b], [label_a, label_b]):
        s1_data = table.set_index('Species')[[TYPE_LABELS[t] for t in TABLE_TYPES]]
        im = ax.imshow(s1_data.values, aspect='auto', cmap='YlOrRd', vmin=0.5, vmax=1.0)
        ax.set_xticks(range(len(s1_data.columns)))
        ax.set_xticklabels(s1_data.columns, rotation=45, ha='right')
        ax.set_yticks(range(len(s1_data.index)))
        ax.set_yticklabels(s1_data.index)
        ax.set_title(f'{title}: AUROC vs Random', fontweight='bold')

        for i in range(len(s1_data.index)):
            for j in range(len(s1_data.columns)):
                val = s1_data.values[i, j]
                if not np.isnan(val):
                    ax.text(
                        j,
                        i,
                        f'{val:.3f}',
                        ha='center',
                        va='center',
                        fontsize=10,
                        color='white' if val > 0.75 else 'black',
                    )

        plt.colorbar(im, ax=ax, label='AUROC')

    plt.tight_layout()
    for fmt in ['pdf', 'svg', 'png']:
        fig.savefig(output_dir / f'table_s1_heatmap_compare.{fmt}', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved table_s1_heatmap_compare figure")


def run_extract(args: argparse.Namespace) -> None:
    benchmark_dir = Path(args.benchmark_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    proteome_file = benchmark_dir / "raw" / args.species / f"{args.species}_proteome.fasta"
    pairs_dir = benchmark_dir / "processed" / f"{args.species}_simplified"
    raw_dir = benchmark_dir / "raw" / args.species

    print("=" * 60)
    print(f"Extracting Attention: {args.species.upper()}")
    print("=" * 60)

    if not proteome_file.exists():
        raise FileNotFoundError(
            f"Proteome not found: {proteome_file}\nRun build_benchmark_minimal.py first!"
        )

    print(f"Proteome: {proteome_file}")

    esm_file = output_dir / f"{args.species}_esm_full.pt"
    if esm_file.exists():
        print("Loading cached ESM embeddings...")
        esm_data = torch.load(esm_file)
    else:
        print("Computing ESM embeddings...")
        sys.path.insert(0, str(Path(args.checkpoint).parent.parent))
        from proteomelm.utils.embedding import build_genome_esmc

        with torch.no_grad():
            esm_data = build_genome_esmc(proteome_file, device=args.device)
        torch.save(esm_data, esm_file)

    group_embeds = esm_data["group_embeds"]
    orthodb_ids = None

    if args.model_type == 'standard' and args.orthodb_db_path:
        orthodb_tsv = args.orthodb_tsv
        if orthodb_tsv is None:
            proteome_id = SPECIES_PROTEOME_IDS.get(args.species)
            if proteome_id is None:
                print(f"Warning: No proteome ID for species '{args.species}'")
            else:
                from proteomelm.utils.proteome import download_proteome

                print("Downloading OrthoDB TSV via UniProt...")
                download_proteome(
                    proteome_id=proteome_id,
                    output_dir=str(raw_dir),
                    organism_name=args.species,
                    reviewed_only=True,
                    download_orthodb=True,
                )
                orthodb_tsv = str(raw_dir / f"{args.species}_orthodb.tsv")

        if orthodb_tsv and not Path(orthodb_tsv).exists():
            print(f"Warning: OrthoDB TSV not found: {orthodb_tsv}")
            print("  Falling back to ESM group embeddings.")
        elif not Path(args.orthodb_db_path).exists():
            print(f"Warning: OrthoDB DB path not found: {args.orthodb_db_path}")
            print("  Falling back to ESM group embeddings.")
        elif orthodb_tsv:
            from proteomelm.utils.proteome import load_orthodb_group_vectors, build_group_embeddings_for_proteome

            print("Building OrthoDB functional group embeddings...")
            orthodb_means = load_orthodb_group_vectors(
                args.orthodb_db_path,
                min_group_size=args.orthodb_min_group_size,
            )
            group_embeds, mask = build_group_embeddings_for_proteome(
                fasta_path=str(proteome_file),
                orthodb_tsv_path=orthodb_tsv,
                orthodb_group_means=orthodb_means,
                esm_embeddings=esm_data["inputs_embeds"],
            )
            print(f"  OrthoDB functional embeddings: {mask.sum().item()}/{mask.shape[0]} mapped")

    if args.model_type == 'alternate':
        from proteomelm.alternate.modeling_naive import (
            ProteomeLMWithOrthoDBEmbedding,
            build_orthodb_vocab,
        )

        orthodb_tsv = args.orthodb_tsv
        if orthodb_tsv is None:
            proteome_id = SPECIES_PROTEOME_IDS.get(args.species)
            if proteome_id is None:
                raise ValueError(f"No proteome ID for species '{args.species}'")
            from proteomelm.utils.proteome import download_proteome

            print("Downloading OrthoDB TSV via UniProt...")
            download_proteome(
                proteome_id=proteome_id,
                output_dir=str(raw_dir),
                organism_name=args.species,
                reviewed_only=True,
                download_orthodb=True,
            )
            orthodb_tsv = str(raw_dir / f"{args.species}_orthodb.tsv")

        if orthodb_tsv is None or not Path(orthodb_tsv).exists():
            raise FileNotFoundError("OrthoDB TSV is required for OrthoDBDiscrete model")
        if args.orthodb_db_path is None or not Path(args.orthodb_db_path).exists():
            raise FileNotFoundError("OrthoDB DB path is required for OrthoDBDiscrete model")

        if args.orthodb_vocab_path and Path(args.orthodb_vocab_path).exists():
            import pickle

            with open(args.orthodb_vocab_path, "rb") as f:
                orthodb_vocab = pickle.load(f)
            print(f"Loaded OrthoDB vocab: {len(orthodb_vocab)} entries")
        else:
            orthodb_vocab = build_orthodb_vocab(
                db_path=args.orthodb_db_path,
                min_taxid_size=args.orthodb_min_group_size,
                vocab_cache_path=args.orthodb_vocab_path,
            )

        orthodb_ids = build_orthodb_ids_for_proteome(
            fasta_file=proteome_file,
            orthodb_tsv=Path(orthodb_tsv),
            orthodb_vocab=orthodb_vocab,
        )

        model = ProteomeLMWithOrthoDBEmbedding.from_pretrained(
            args.checkpoint, orthodb_vocab_size=len(orthodb_vocab)
        )
    else:
        from proteomelm.modeling_proteomelm import ProteomeLMForMaskedLM
        model = ProteomeLMForMaskedLM.from_pretrained(args.checkpoint)

    model = model.to(dtype=torch.bfloat16, device='cpu').eval()

    print("Running inference...")
    with torch.no_grad():
        model_inputs = {
            "inputs_embeds": esm_data["inputs_embeds"][None].to(dtype=torch.bfloat16),
            "output_attentions": True,
        }
        if args.model_type == 'alternate':
            model_inputs["orthodb_ids"] = orthodb_ids[None]
        else:
            model_inputs["group_embeds"] = group_embeds[None].to(dtype=torch.bfloat16)

        output = model(**model_inputs)
        attentions = [attn.squeeze(0).float().numpy() for attn in output.attentions]

    print(f"  {len(attentions)} layers, shape: {attentions[0].shape}")

    protein_to_idx = create_protein_mapping(proteome_file)
    print(f"  {len(set(protein_to_idx.values()))} proteins mapped")

    print("Saving functional encodings...")
    np.savez_compressed(
        output_dir / f"{args.species}_functional_encodings.npz",
        inputs_embeds=esm_data["inputs_embeds"].cpu().float().numpy(),
        group_embeds=group_embeds.cpu().float().numpy(),
        protein_to_idx=protein_to_idx,
    )

    all_pos = []
    for bm_type in ['pdb', 'pdb_physical', 'coexpression']:
        print(f"\n[{bm_type.upper()}]")
        path = pairs_dir / f"{bm_type}_pairs.tsv"

        if not path.exists():
            print(f"  Not found: {path}")
            continue

        df = pd.read_csv(path, sep='\t')
        if 'uniprot_a' in df.columns:
            df = df.rename(columns={'uniprot_a': 'protein_a', 'uniprot_b': 'protein_b'})

        all_pos.append(df)
        pairs = list(zip(df['protein_a'], df['protein_b']))
        print(f"  {len(pairs)} pairs loaded")

        result = extract_attention(pairs, protein_to_idx, attentions, args.max_pairs)
        if result:
            np.savez_compressed(
                output_dir / f"{args.species}_{bm_type}_attention.npz",
                pair_ids=np.array(result['pair_ids'], dtype=object),
                attention_a_to_b=result['attention_a_to_b'],
                attention_b_to_a=result['attention_b_to_a'],
            )
            print(f"  Saved {len(result['pair_ids'])} pairs")
        else:
            print("  No valid pairs found in proteome mapping")

    print("\n[RANDOM]")
    if all_pos:
        all_positive = pd.concat(all_pos, ignore_index=True)
        neg_pairs = generate_negatives(list(protein_to_idx.keys()), all_positive, args.n_negative)

        result = extract_attention(neg_pairs, protein_to_idx, attentions, args.max_pairs)
        if result:
            np.savez_compressed(
                output_dir / f"{args.species}_random_attention.npz",
                pair_ids=np.array(result['pair_ids'], dtype=object),
                attention_a_to_b=result['attention_a_to_b'],
                attention_b_to_a=result['attention_b_to_a'],
            )
            print(f"  Saved {len(result['pair_ids'])} pairs")

    print("\nDone. Output:")
    print(f"  {output_dir}")


def run_compare(args: argparse.Namespace) -> None:
    attention_dir_a = Path(args.attention_dir_a)
    attention_dir_b = Path(args.attention_dir_b)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ProteomeLM Attention Comparison")
    print("=" * 70)
    print(f"Attention A: {attention_dir_a}")
    print(f"Attention B: {attention_dir_b}")
    print(f"Labels: {args.label_a} vs {args.label_b}")
    print(f"Output directory: {output_dir}")

    plot_attention_by_type_compare(
        attention_dir_a,
        attention_dir_b,
        args.label_a,
        args.label_b,
        output_dir,
    )
    plot_table_s1_heatmap_compare(
        attention_dir_a,
        attention_dir_b,
        args.label_a,
        args.label_b,
        output_dir,
    )

    table_a = compute_table_s1_data(attention_dir_a)
    table_b = compute_table_s1_data(attention_dir_b)
    safe_a = args.label_a.lower().replace(" ", "_")
    safe_b = args.label_b.lower().replace(" ", "_")
    table_a.to_csv(output_dir / f'table_s1_auroc_vs_random_{safe_a}.csv', index=False)
    table_b.to_csv(output_dir / f'table_s1_auroc_vs_random_{safe_b}.csv', index=False)

    print("\nDone.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ablation: attention AUROC comparison")
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract_parser = subparsers.add_parser("extract", help="Extract attention patterns")
    extract_parser.add_argument('--species', required=True, choices=['yeast', 'human', 'ecoli'])
    extract_parser.add_argument('--checkpoint', required=True, help='ProteomeLM checkpoint')
    extract_parser.add_argument('--model-type', default='standard', choices=['standard', 'alternate'])
    extract_parser.add_argument('--benchmark-dir', default='data/benchmarks')
    extract_parser.add_argument('--output-dir', default='attention_patterns')
    extract_parser.add_argument('--device', default='cuda:0')
    extract_parser.add_argument('--max-pairs', type=int, default=10000)
    extract_parser.add_argument('--n-negative', type=int, default=5000)
    extract_parser.add_argument('--orthodb-tsv', default=None)
    extract_parser.add_argument('--orthodb-db-path', default=None)
    extract_parser.add_argument('--orthodb-min-group-size', type=int, default=100)
    extract_parser.add_argument('--orthodb-vocab-path', default=None)

    compare_parser = subparsers.add_parser("compare", help="Compare two attention directories")
    compare_parser.add_argument('--attention-dir-a', required=True)
    compare_parser.add_argument('--attention-dir-b', required=True)
    compare_parser.add_argument('--label-a', default='ProteomeLM-S')
    compare_parser.add_argument('--label-b', default='ProteomeLM-S-OrthoDBDiscrete')
    compare_parser.add_argument('--output-dir', default='figures_minimal/orthodb_discrete_vs_standard')

    args = parser.parse_args()

    if args.command == "extract":
        run_extract(args)
    elif args.command == "compare":
        run_compare(args)


if __name__ == '__main__':
    main()
