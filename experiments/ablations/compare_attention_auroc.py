#!/usr/bin/env python3
"""
Ablation utilities for comparing attention AUROC between models.

Subcommands:
  extract  - generate attention patterns for a model checkpoint (standard ProteomeLM,
             or the learned-OrthoDB-ID ablation with --model-type alternate)
  compare  - compare per-head AUROC and best-head heatmaps between two models

Pair lists come from experiments/differential_interactomes/build_benchmark.py; the
extraction and AUROC helpers are shared with experiments/differential_interactomes/.
"""

import argparse
import sys
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from proteomelm.utils.io import parse_fasta
from experiments.differential_interactomes.extract_attention import (
    build_group_embeds,
    compute_attentions,
    create_protein_mapping,
    embeddings_are_identical,
    load_esm_data,
    resolve_orthodb_tsv,
    save_pair_attentions,
)

# analyze_proteomelm sets global rcParams and warning filters on import; keep this script's defaults.
with plt.rc_context(), warnings.catch_warnings():
    from experiments.differential_interactomes.analyze_proteomelm import (
        SPECIES_COLORS,
        SPECIES_LIST,
        SPECIES_NAMES,
        SPECIES_NAMES_ITALIC as SPECIES_LEGEND_LABELS,
        TABLE_TYPES,
        TYPE_LABELS,
        compute_auroc_vs_random,
        compute_per_head_auroc,
        load_attention_data,
    )


def build_orthodb_ids_for_proteome(
    fasta_file: Path,
    orthodb_tsv: Path,
    orthodb_vocab: Dict[str, int],
) -> torch.Tensor:
    """Build OrthoDB ID indices aligned to proteome order."""
    from proteomelm.utils.proteome import parse_orthodb_tsv

    sequences = parse_fasta(str(fasta_file))
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
            f"Proteome not found: {proteome_file}\nRun differential_interactomes/build_benchmark.py first!"
        )

    print(f"Proteome: {proteome_file}")

    esm_data = load_esm_data(proteome_file, output_dir / f"{args.species}_esm_full.pt", args.device)

    if args.model_type == 'alternate':
        from proteomelm.alternate.modeling_naive import (
            ProteomeLMWithOrthoDBEmbedding,
            build_orthodb_vocab,
        )

        group_embeds = esm_data["group_embeds"]
        orthodb_tsv = resolve_orthodb_tsv(args.species, raw_dir, args.orthodb_tsv)
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
        attentions = compute_attentions(model, esm_data["inputs_embeds"], orthodb_ids=orthodb_ids[None])
    else:
        group_embeds, _ = build_group_embeds(
            esm_data, proteome_file, args.species, raw_dir,
            orthodb_db_path=args.orthodb_db_path,
            orthodb_tsv=args.orthodb_tsv,
            min_group_size=args.orthodb_min_group_size,
        )
        if embeddings_are_identical(esm_data["inputs_embeds"], group_embeds):
            # extract_attention.py refuses this case; kept here to reproduce earlier ablation runs.
            print("Warning: group_embeds are identical to inputs_embeds (ESM-C used as group embeddings).")
            print("  Pass --orthodb-db-path for OrthoDB group embeddings, as in pretraining.")

        from proteomelm.modeling_proteomelm import ProteomeLMForMaskedLM
        model = ProteomeLMForMaskedLM.from_pretrained(args.checkpoint)
        attentions = compute_attentions(
            model, esm_data["inputs_embeds"], group_embeds=group_embeds[None].to(dtype=torch.bfloat16)
        )

    protein_to_idx = create_protein_mapping(proteome_file)
    print(f"  {len(set(protein_to_idx.values()))} proteins mapped")

    print("Saving functional encodings...")
    np.savez_compressed(
        output_dir / f"{args.species}_functional_encodings.npz",
        inputs_embeds=esm_data["inputs_embeds"].cpu().float().numpy(),
        group_embeds=group_embeds.cpu().float().numpy(),
        protein_to_idx=protein_to_idx,
    )

    save_pair_attentions(
        attentions, protein_to_idx, pairs_dir, output_dir, args.species, args.max_pairs, args.n_negative
    )

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
