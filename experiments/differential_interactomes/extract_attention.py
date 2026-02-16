#!/usr/bin/env python3
"""
Minimal Attention Extraction for ProteomeLM Analysis

Uses proteome from build_benchmark_minimal.py output.
Extracts attention patterns for: pdb, pdb_physical, coexpression, random
"""

import sys
from pathlib import Path
# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
from typing import Dict, List, Tuple, Optional
import re
import numpy as np
import pandas as pd
import torch

SPECIES_PROTEOME_IDS = {
    'yeast': 'UP000002311',
    'human': 'UP000005640',
    'ecoli': 'UP000000625',
}


def embeddings_are_identical(a: torch.Tensor, b: torch.Tensor, atol: float = 1e-6) -> bool:
    """Return True when two embedding tensors are numerically identical."""
    if a.shape != b.shape:
        return False
    return torch.allclose(a.float(), b.float(), atol=atol)


def parse_fasta(fasta_file: Path) -> Dict[str, str]:
    """Parse FASTA file to dict of {id: sequence}."""
    sequences = {}
    current_id, current_seq = None, []
    
    with open(fasta_file) as f:
        for line in f:
            if line.startswith('>'):
                if current_id:
                    sequences[current_id] = ''.join(current_seq)
                # Parse: >sp|P12345|NAME or >P12345
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
                
                # Primary ID: sp|P12345|NAME -> P12345
                parts = header.split('|')
                uniprot_id = parts[1] if len(parts) > 1 else header.split()[0]
                
                mapping[uniprot_id] = idx
                
                # Gene name from GN=
                if 'GN=' in header:
                    gene = header.split('GN=')[1].split()[0]
                    mapping[gene] = idx
                
                # Yeast systematic names
                if match := re.search(r'Y[A-P][LR]\d{3}[WC](?:-[A-Z])?', header):
                    mapping[match.group(0)] = idx
                
                idx += 1
    
    return mapping


def extract_attention(
    pairs: List[Tuple[str, str]],
    protein_to_idx: Dict[str, int],
    attentions: List[np.ndarray],
    max_pairs: int = 10000
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


def generate_negatives(proteins: List[str], positives: pd.DataFrame, n: int, seed: int = 42) -> List[Tuple[str, str]]:
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


def main():
    parser = argparse.ArgumentParser(description="Extract attention for ProteomeLM benchmark")
    parser.add_argument('--species', required=True, choices=['yeast', 'human', 'ecoli'])
    parser.add_argument('--checkpoint', required=True, help='ProteomeLM checkpoint')
    parser.add_argument('--benchmark-dir', default='data/benchmarks', help='Benchmark directory (from build_benchmark)')
    parser.add_argument('--output-dir', default='attention_patterns')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--max-pairs', type=int, default=10000)
    parser.add_argument('--n-negative', type=int, default=5000)
    parser.add_argument('--orthodb-tsv', default=None,
                        help='OrthoDB TSV from download_proteome (optional)')
    parser.add_argument('--orthodb-db-path', default=None,
                        help='Directory with group_vectors_*.pkl (optional)')
    parser.add_argument('--orthodb-min-group-size', type=int, default=0,
                        help='Min OrthoDB group size threshold for loading vectors')
    parser.add_argument('--allow-identical-group-embeds', action='store_true',
                        help='Allow running when group_embeds == inputs_embeds (not recommended)')
    args = parser.parse_args()
    
    benchmark_dir = Path(args.benchmark_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Paths from build_benchmark_minimal.py
    proteome_file = benchmark_dir / "raw" / args.species / f"{args.species}_proteome.fasta"
    pairs_dir = benchmark_dir / "processed" / f"{args.species}_simplified"
    raw_dir = benchmark_dir / "raw" / args.species
    
    print("=" * 60)
    print(f"Extracting Attention: {args.species.upper()}")
    print("=" * 60)
    
    # Check proteome exists
    if not proteome_file.exists():
        raise FileNotFoundError(f"Proteome not found: {proteome_file}\nRun build_benchmark_minimal.py first!")
    
    print(f"Proteome: {proteome_file}")
    
    # Compute ESM embeddings
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
    
    # Optionally build functional group embeddings
    group_embeds = esm_data["group_embeds"]
    group_source = "esm_fallback"
    if args.orthodb_db_path:
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
            group_source = "orthodb"
            print(f"  OrthoDB functional embeddings: {mask.sum().item()}/{mask.shape[0]} mapped")

    group_equals_inputs = embeddings_are_identical(esm_data["inputs_embeds"], group_embeds)
    if group_equals_inputs:
        if args.allow_identical_group_embeds:
            print("Warning: group_embeds are identical to inputs_embeds.")
            print("  Continuing because --allow-identical-group-embeds was set.")
        else:
            raise RuntimeError(
                "group_embeds are identical to inputs_embeds. "
                "This makes the functional baseline invalid. "
                "Provide OrthoDB vectors via --orthodb-db-path (and optional --orthodb-tsv), "
                "or pass --allow-identical-group-embeds to force-run."
            )
    else:
        print("  Using distinct group_embeds for functional baseline.")

    # Load ProteomeLM
    print(f"Loading ProteomeLM: {args.checkpoint}")
    sys.path.insert(0, str(Path(args.checkpoint).parent.parent))
    from proteomelm.modeling_proteomelm import ProteomeLMForMaskedLM
    
    model = ProteomeLMForMaskedLM.from_pretrained(args.checkpoint)
    model = model.to(dtype=torch.bfloat16, device='cpu').eval()
    
    print("Running inference...")
    with torch.no_grad():
        output = model(
            inputs_embeds=esm_data["inputs_embeds"][None].to(dtype=torch.bfloat16),
            group_embeds=group_embeds[None].to(dtype=torch.bfloat16),
            output_attentions=True
        )
        attentions = [attn.squeeze(0).float().numpy() for attn in output.attentions]
    
    print(f"  {len(attentions)} layers, shape: {attentions[0].shape}")
    
    # Create protein mapping
    protein_to_idx = create_protein_mapping(proteome_file)
    print(f"  {len(set(protein_to_idx.values()))} proteins mapped")
    
    # Save functional encodings
    print("Saving functional encodings...")
    np.savez_compressed(
        output_dir / f"{args.species}_functional_encodings.npz",
        inputs_embeds=esm_data["inputs_embeds"].cpu().float().numpy(),
        group_embeds=group_embeds.cpu().float().numpy(),
        protein_to_idx=protein_to_idx,
        group_source=np.array(group_source),
        group_equals_inputs=np.array(group_equals_inputs)
    )
    
    # Process each benchmark type
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
                attention_b_to_a=result['attention_b_to_a']
            )
            print(f"  Saved {len(result['pair_ids'])} pairs")
        else:
            print(f"  No valid pairs found in proteome mapping!")
    
    # Generate negatives
    print(f"\n[RANDOM]")
    if all_pos:
        all_positive = pd.concat(all_pos, ignore_index=True)
        neg_pairs = generate_negatives(list(protein_to_idx.keys()), all_positive, args.n_negative)
        
        result = extract_attention(neg_pairs, protein_to_idx, attentions, args.max_pairs)
        if result:
            np.savez_compressed(
                output_dir / f"{args.species}_random_attention.npz",
                pair_ids=np.array(result['pair_ids'], dtype=object),
                attention_a_to_b=result['attention_a_to_b'],
                attention_b_to_a=result['attention_b_to_a']
            )
            print(f"  Saved {len(result['pair_ids'])} pairs")
    
    print(f"\n{'=' * 60}")
    print(f"✓ Complete. Output: {output_dir}")


if __name__ == '__main__':
    main()