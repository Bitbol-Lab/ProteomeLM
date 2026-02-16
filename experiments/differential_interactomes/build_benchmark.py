#!/usr/bin/env python3
"""
Minimal Benchmark Builder for ProteomeLM Analysis

Approach:
1. Download full UniProt proteome for species
2. For PINDER/STRING pairs, match exact UniProt IDs first
3. If no exact match, use MMseqs2 to find best sequence match

Interaction types:
- pdb: Direct contacts from PINDER/PDB (gold standard)
- pdb_physical: Same complex from PINDER/PDB  
- coexpression: Co-expression from STRING
"""

import argparse
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Set, Tuple, Optional
import pandas as pd

SPECIES_CONFIG = {
    'yeast': {'proteome_id': 'UP000002311', 'string_id': '4932', 'pinder_taxon': '559292'},
    'human': {'proteome_id': 'UP000005640', 'string_id': '9606', 'pinder_taxon': '9606'},
    'ecoli': {'proteome_id': 'UP000000625', 'string_id': '511145', 'pinder_taxon': '83333'},
}


class BenchmarkBuilder:
    def __init__(self, species: str, output_dir: str = "data/benchmarks"):
        self.species = species
        self.config = SPECIES_CONFIG[species]
        self.output_dir = Path(output_dir)
        self.raw_dir = self.output_dir / "raw" / species
        self.processed_dir = self.output_dir / "processed" / f"{species}_simplified"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Proteome data
        self.proteome_file: Optional[Path] = None
        self.proteome_ids: Set[str] = set()  # All UniProt IDs in proteome
        self.id_to_seq: Dict[str, str] = {}  # UniProt ID -> sequence
        self.gene_to_uniprot: Dict[str, str] = {}
        self.string_to_uniprot: Dict[str, str] = {}
        
        # MMseqs2 database for unmatched sequences
        self.mmseqs_db: Optional[Path] = None

    def download_proteome(self):
        """Download full UniProt proteome."""
        print(f"\n📥 Downloading proteome {self.config['proteome_id']}...")
        
        self.proteome_file = self.raw_dir / f"{self.species}_proteome.fasta"
        
        if not self.proteome_file.exists():
            url = f"https://rest.uniprot.org/uniprotkb/stream?format=fasta&query=proteome:{self.config['proteome_id']}+AND+reviewed:true"
            subprocess.run(["wget", "-q", "-O", str(self.proteome_file), url], check=True)
        
        # Parse proteome
        self._parse_proteome()
        print(f"  ✓ {len(self.proteome_ids)} proteins in proteome")

    def _parse_proteome(self):
        """Parse proteome FASTA and build indices."""
        import re
        
        current_id, current_seq, current_gene = None, [], None
        
        with open(self.proteome_file) as f:
            for line in f:
                if line.startswith('>'):
                    if current_id:
                        seq = ''.join(current_seq)
                        self.proteome_ids.add(current_id)
                        self.id_to_seq[current_id] = seq
                        if current_gene:
                            self.gene_to_uniprot[current_gene] = current_id
                    
                    # Parse header: >sp|P12345|GENE_SPECIES Description GN=GeneName
                    header = line[1:].strip()
                    parts = header.split('|')
                    current_id = parts[1] if len(parts) > 1 else header.split()[0]
                    current_seq = []
                    current_gene = None
                    
                    # Extract gene name
                    if 'GN=' in header:
                        current_gene = header.split('GN=')[1].split()[0]
                    # Yeast systematic names
                    if match := re.search(r'Y[A-P][LR]\d{3}[WC](?:-[A-Z])?', header):
                        current_gene = match.group(0)
                else:
                    current_seq.append(line.strip())
        
        # Last entry
        if current_id:
            self.proteome_ids.add(current_id)
            self.id_to_seq[current_id] = ''.join(current_seq)
            if current_gene:
                self.gene_to_uniprot[current_gene] = current_id

    def download_string_data(self):
        """Download STRING data."""
        print(f"\n📥 Downloading STRING data...")
        
        string_id = self.config['string_id']
        string_file = self.raw_dir / f"string_{string_id}.txt.gz"
        string_extracted = self.raw_dir / f"string_{string_id}.txt"
        
        if not string_extracted.exists():
            if not string_file.exists():
                url = f"https://stringdb-downloads.org/download/protein.links.detailed.v12.0/{string_id}.protein.links.detailed.v12.0.txt.gz"
                subprocess.run(["wget", "-q", "-O", str(string_file), url], check=True)
            subprocess.run(["gunzip", "-f", "-k", str(string_file)], check=True)
        
        # Load STRING ID mapping
        info_file = self.raw_dir / f"string_{string_id}_info.txt.gz"
        info_extracted = self.raw_dir / f"string_{string_id}_info.txt"
        
        if not info_extracted.exists():
            if not info_file.exists():
                url = f"https://stringdb-downloads.org/download/protein.info.v12.0/{string_id}.protein.info.v12.0.txt.gz"
                subprocess.run(["wget", "-q", "-O", str(info_file), url], check=True)
            subprocess.run(["gunzip", "-f", "-k", str(info_file)], check=True)
        
        # Build STRING -> UniProt mapping
        prefix = f"{string_id}."
        with open(info_extracted) as f:
            f.readline()
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    sid = parts[0].replace(prefix, '')
                    name = parts[1]
                    # Try gene name first, then direct UniProt
                    if name in self.gene_to_uniprot:
                        self.string_to_uniprot[sid] = self.gene_to_uniprot[name]
                    elif name in self.proteome_ids:
                        self.string_to_uniprot[sid] = name
        
        print(f"  ✓ STRING ready, {len(self.string_to_uniprot)} IDs mapped")

    def build_mmseqs_db(self):
        """Build MMseqs2 database from proteome for sequence matching."""
        print("\n🔧 Building MMseqs2 database...")
        
        self.mmseqs_db = self.raw_dir / f"{self.species}_mmseqs_db"
        
        if not Path(f"{self.mmseqs_db}.index").exists():
            subprocess.run([
                "mmseqs", "createdb", str(self.proteome_file), str(self.mmseqs_db)
            ], check=True, capture_output=True)
        
        print(f"  ✓ MMseqs2 database ready")

    def match_by_sequence(self, query_id: str, query_seq: str, min_identity: float = 0.9) -> Optional[str]:
        """Find best matching proteome protein by sequence similarity."""
        if not query_seq or len(query_seq) < 10:
            return None
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            query_fasta = tmpdir / "query.fasta"
            query_db = tmpdir / "query_db"
            result_db = tmpdir / "result_db"
            result_tsv = tmpdir / "result.tsv"
            
            # Write query
            with open(query_fasta, 'w') as f:
                f.write(f">{query_id}\n{query_seq}\n")
            
            # Create query db and search
            subprocess.run(["mmseqs", "createdb", str(query_fasta), str(query_db)], 
                          capture_output=True, check=True)
            subprocess.run([
                "mmseqs", "search", str(query_db), str(self.mmseqs_db), str(result_db), tmpdir,
                "--min-seq-id", str(min_identity), "-s", "7.5", "--max-seqs", "1"
            ], capture_output=True, check=True)
            subprocess.run([
                "mmseqs", "convertalis", str(query_db), str(self.mmseqs_db), str(result_db), str(result_tsv),
                "--format-output", "query,target,pident"
            ], capture_output=True, check=True)
            
            # Parse result
            if result_tsv.exists():
                with open(result_tsv) as f:
                    for line in f:
                        parts = line.strip().split('\t')
                        if len(parts) >= 2:
                            return parts[1]  # Target UniProt ID
        
        return None

    def resolve_uniprot(self, uniprot_id: str, sequence: str = None) -> Optional[str]:
        """Resolve UniProt ID to proteome, using MMseqs2 if needed."""
        if not uniprot_id:
            return None
        
        # Clean ID (remove isoform suffix)
        clean_id = uniprot_id.split('-')[0]
        
        # Direct match
        if clean_id in self.proteome_ids:
            return clean_id
        
        # Try via gene name
        if clean_id in self.gene_to_uniprot:
            return self.gene_to_uniprot[clean_id]
        
        # Sequence-based matching with MMseqs2
        if sequence and self.mmseqs_db:
            matched = self.match_by_sequence(clean_id, sequence)
            if matched:
                return matched
        
        return None

    def process_pinder(self, min_sasa: float = 500.0, min_residues: int = 10) -> Dict[str, Set[Tuple[str, str]]]:
        """Process PINDER with proteome-based ID resolution."""
        print(f"\n🔬 Processing PINDER...")
        
        pdb_pairs: Set[Tuple[str, str]] = set()
        pdb_physical: Set[Tuple[str, str]] = set()
        
        cache = self.raw_dir / f"pinder_{self.species}_pairs.tsv"
        cache_phys = self.raw_dir / f"pinder_{self.species}_physical_pairs.tsv"
        
        if cache.exists():
            df = pd.read_csv(cache, sep='\t')
            pdb_pairs = {tuple(sorted([r['uniprot_a'], r['uniprot_b']])) for _, r in df.iterrows() if pd.notna(r['uniprot_a'])}
            if cache_phys.exists():
                df_phys = pd.read_csv(cache_phys, sep='\t')
                pdb_physical = {tuple(sorted([r['uniprot_a'], r['uniprot_b']])) for _, r in df_phys.iterrows() if pd.notna(r['uniprot_a'])}
            print(f"  ✓ Loaded {len(pdb_pairs)} direct, {len(pdb_physical)} same-complex from cache")
            return {'pdb': pdb_pairs, 'pdb_physical': pdb_physical}
        
        from pinder.core import get_index, get_metadata
        print("  Loading PINDER index...")
        
        index = get_index()
        metadata = get_metadata()
        df = index.merge(metadata, on='id', how='left')
        print(f"  Total systems: {len(df):,}")
        
        # Filter by species
        target_taxon = self.config['pinder_taxon']
        if 'taxid_R' in df.columns and 'taxid_L' in df.columns:
            mask = (df['taxid_R'].astype(str).str.contains(target_taxon, na=False) &
                    df['taxid_L'].astype(str).str.contains(target_taxon, na=False))
            df = df[mask]
            print(f"  After species filter: {len(df):,}")
        
        # Quality split
        if 'buried_sasa' in df.columns:
            mask = df['buried_sasa'] >= min_sasa
            if 'n_residue_pairs' in df.columns:
                mask &= df['n_residue_pairs'] >= min_residues
            df_high, df_low = df[mask], df[~mask]
        else:
            df_high, df_low = df, pd.DataFrame()
        
        print(f"  High quality: {len(df_high):,}, Lower: {len(df_low):,}")
        
        # Extract and resolve pairs
        n_exact, n_mmseqs, n_failed = 0, 0, 0
        
        def process_df(dataframe, target_set):
            nonlocal n_exact, n_mmseqs, n_failed
            for _, row in dataframe.iterrows():
                uid_r = str(row['uniprot_R']).split('-')[0] if pd.notna(row.get('uniprot_R')) else None
                uid_l = str(row['uniprot_L']).split('-')[0] if pd.notna(row.get('uniprot_L')) else None
                
                # Resolve to proteome
                resolved_r = self.resolve_uniprot(uid_r)
                resolved_l = self.resolve_uniprot(uid_l)
                
                if resolved_r and resolved_l and resolved_r != resolved_l:
                    target_set.add(tuple(sorted([resolved_r, resolved_l])))
                    if resolved_r == uid_r and resolved_l == uid_l:
                        n_exact += 1
                    else:
                        n_mmseqs += 1
                else:
                    n_failed += 1
        
        process_df(df_high, pdb_pairs)
        process_df(df_low, pdb_physical)
        
        print(f"  Exact matches: {n_exact}, MMseqs2 resolved: {n_mmseqs}, Failed: {n_failed}")
        print(f"  ✓ {len(pdb_pairs)} high-quality, {len(pdb_physical)} lower-quality pairs")
        
        # Cache
        if pdb_pairs:
            pd.DataFrame([{'uniprot_a': p[0], 'uniprot_b': p[1]} for p in pdb_pairs]).to_csv(cache, sep='\t', index=False)
        if pdb_physical:
            pd.DataFrame([{'uniprot_a': p[0], 'uniprot_b': p[1]} for p in pdb_physical]).to_csv(cache_phys, sep='\t', index=False)
        
        return {'pdb': pdb_pairs, 'pdb_physical': pdb_physical}

    def process_string_coexpression(self, threshold: int = 900) -> Set[Tuple[str, str]]:
        """Process STRING co-expression with proteome-based resolution."""
        print(f"\n🧬 Processing STRING coexpression (threshold={threshold})...")
        
        string_file = self.raw_dir / f"string_{self.config['string_id']}.txt"
        prefix = f"{self.config['string_id']}."
        pairs: Set[Tuple[str, str]] = set()
        
        if not string_file.exists():
            print("  ⚠ STRING file not found")
            return pairs
        
        for chunk in pd.read_csv(string_file, sep=' ', chunksize=100000):
            if 'coexpression' not in chunk.columns:
                continue
            
            filtered = chunk[chunk['coexpression'] > threshold]
            for _, row in filtered.iterrows():
                sid_a = row['protein1'].replace(prefix, '')
                sid_b = row['protein2'].replace(prefix, '')
                
                # Resolve via STRING mapping or gene name
                uid_a = self.string_to_uniprot.get(sid_a) or self.gene_to_uniprot.get(sid_a)
                uid_b = self.string_to_uniprot.get(sid_b) or self.gene_to_uniprot.get(sid_b)
                
                if uid_a and uid_b and uid_a != uid_b:
                    if uid_a in self.proteome_ids and uid_b in self.proteome_ids:
                        pairs.add(tuple(sorted([uid_a, uid_b])))
        
        print(f"  ✓ {len(pairs)} coexpression pairs")
        return pairs

    def save_pairs(self, pairs: Set[Tuple[str, str]], name: str):
        """Save pairs to TSV."""
        if not pairs:
            return
        data = [{'uniprot_a': p[0], 'uniprot_b': p[1]} for p in pairs]
        path = self.processed_dir / f"{name}_pairs.tsv"
        pd.DataFrame(data).to_csv(path, sep='\t', index=False)
        print(f"  Saved {len(data)} pairs → {path.name}")

    def run(self, coexp_threshold: int = 900, use_mmseqs: bool = True):
        """Run benchmark building pipeline."""
        self.download_proteome()
        self.download_string_data()
        
        if use_mmseqs:
            self.build_mmseqs_db()
        
        pinder = self.process_pinder()
        coexp = self.process_string_coexpression(coexp_threshold)
        
        # Remove overlaps (PDB is gold standard)
        pdb_set = pinder.get('pdb', set())
        coexp_clean = coexp - pdb_set - pinder.get('pdb_physical', set())
        
        print(f"\n💾 Saving to {self.processed_dir}...")
        self.save_pairs(pdb_set, 'pdb')
        self.save_pairs(pinder.get('pdb_physical', set()), 'pdb_physical')
        self.save_pairs(coexp_clean, 'coexpression')
        
        print(f"\n✓ Benchmark complete for {self.species}")


def main():
    parser = argparse.ArgumentParser(description="Build minimal ProteomeLM benchmark")
    parser.add_argument('--species', default='yeast', choices=list(SPECIES_CONFIG.keys()))
    parser.add_argument('--output-dir', default='data/benchmarks')
    parser.add_argument('--coexp-threshold', type=int, default=900)
    parser.add_argument('--no-mmseqs', action='store_true', help='Skip MMseqs2 sequence matching')
    args = parser.parse_args()
    
    builder = BenchmarkBuilder(args.species, args.output_dir)
    builder.run(args.coexp_threshold, use_mmseqs=not args.no_mmseqs)


if __name__ == '__main__':
    main()
