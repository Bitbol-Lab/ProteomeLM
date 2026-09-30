# Differential interactomes

Do ProteomeLM attention heads tell interaction types apart? For E. coli, yeast
and human, three scripts build pair benchmarks, extract whole-proteome
attention for every pair, and score each head against random pairs and against
the other types. Run them in order from this directory:

```bash
# 1. Pair lists per interaction type (downloads UniProt, STRING, PINDER)
python build_benchmark.py --species ecoli --output-dir data/benchmarks

# 2. ProteomeLM attention for every pair (+ random negatives)
python extract_attention.py --species ecoli --checkpoint Bitbol-Lab/ProteomeLM-M \
    --benchmark-dir data/benchmarks --output-dir data/attention_patterns \
    --orthodb-db-path /path/to/training   # directory with group_vectors_*.pkl

# 3. Tables and figures for every species found in data/attention_patterns
python analyze_proteomelm.py --attention-dir data/attention_patterns --output-dir data/figures
```

Repeat steps 1–2 with `--species yeast` and `--species human`. `data/` is gitignored.

| Type | Source | Definition |
|------|--------|------------|
| `pdb` | PINDER | Same-species dimers with buried SASA ≥ 500 Å² and ≥ 10 residue pairs |
| `pdb_physical` | PINDER | Remaining same-species dimers (smaller interfaces) |
| `coexpression` | STRING v12 | Coexpression score > `--coexp-threshold` (900), minus the two PDB sets |
| `random` | Sampled | Proteome pairs outside the positive sets, made by step 2 (`--n-negative`, seed 42) |

**`build_benchmark.py`** (`--species {yeast,human,ecoli}`, `--output-dir`, `--coexp-threshold`, `--no-mmseqs`)
writes `raw/{species}/` (proteome FASTA, STRING files, PINDER pair caches) and
`processed/{species}_simplified/{pdb,pdb_physical,coexpression}_pairs.tsv`. IDs are
matched to the reviewed UniProt proteome by accession (isoform suffix dropped) or gene name.
An MMseqs2 database is built (skip with `--no-mmseqs`), but the sequence fallback is not
reached: the PINDER path passes no sequences to it.

**`extract_attention.py`** (`--species`, `--checkpoint`, `--benchmark-dir`, `--output-dir`,
`--device` for ESM-C, `--max-pairs` 10000, `--n-negative` 5000, `--orthodb-tsv`,
`--orthodb-db-path`, `--orthodb-min-group-size`, `--allow-identical-group-embeds`) runs
ProteomeLM in bf16 on CPU over the full proteome. It writes `{species}_esm_full.pt`,
`{species}_functional_encodings.npz` and `{species}_{type}_attention.npz` (both attention
directions, `[n_pairs, layers, heads]`). Without OrthoDB group vectors the group embeddings
would equal the ESM-C inputs, so the script stops unless `--allow-identical-group-embeds` is set.

**`analyze_proteomelm.py`** (`--attention-dir`, `--output-dir`) writes
`table_s1_auroc_vs_random.csv` (best-head AUROC vs random),
`table_s1b_{inputs,functional}_cosine.csv` (cosine-similarity baselines),
`table_s2_pairwise_classification.csv` (logistic regression on all heads), per-species
`{species}_attention_by_type`, `{species}_heads_vs_cosine_vs_pca` and
`{species}_pairwise_classification_heads` figures, plus `auroc_summary_bars` and
`summary_figure`. It also prints the tables as LaTeX.

The learned-OrthoDB-ID ablation (`experiments/ablations/compare_attention_auroc.py`) reuses
steps 1–2 and this directory's AUROC helpers.

Requirements: the `proteomelm` package, `pinder` (`pip install pinder`, step 1 only),
`wget`, `gunzip`, and optionally `mmseqs` on `PATH`.
