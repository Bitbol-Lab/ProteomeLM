# experiments/hpi/

Analysis pipeline for the host-pathogen interaction (HPI) fine-tuning work
(`proteomelm/hpi/`), producing the ablation/benchmark figures and tables for
the GenBio/ICML 2026 workshop paper.

## Benchmark data

The per-pathogen benchmark data in `data/benchmarks/`
(`raw/{pathogen}/combined_proteome.fasta` and
`processed/{pathogen}/hpi_pairs.tsv` for the 10 pathogens in
`utils.DATASETS`) **cannot be rebuilt from this repository**: download the
benchmark data (see the top-level README) and unpack it into `data/`.

The following scripts document or refresh parts of that data:

- **`build_gold_benchmark.py`** — builds the stricter, multi-level-holdout
  `data/gold_benchmark/` (balanced 3-fold species split) from
  `data/benchmarks/`; `--leave-one-out` builds `data/gold_benchmark_loo/`
  (one fold per held-out pathogen).
- **`fetch_pathogen_proteomes.py`** — downloads a benchmark pathogen's full
  UniProt reference proteome and rebuilds its `combined_proteome.fasta`
  (host first, then pathogen). Does not handle EBV / HSV-1 (see next item).
- **`process_raw_ebv_hsv1.py`** — processes the raw EBV (AP-MS/CompPASS) and
  HSV-1 (XL-MS) supplementary tables into `hpi_pairs.tsv`, and writes their
  `combined_proteome.fasta` (pathogen first, interacting pathogen proteins
  only) — the version used in the paper.

## Pipeline (numbered scripts)

| Script | Produces | Paper artifact |
| --- | --- | --- |
| `0_extract_attention.py` | Pair attention `.npz` files for a checkpoint (and, with `--also-base`, the base model) | Input to scripts 1, 2, 6 |
| `1_base_vs_finetuned.py` | Logistic-regression AUROC (all heads), base vs. fine-tuned, per dataset | Fig. 1B |
| `2_ablations.py` | Balanced (virus/bacteria) LR AUROC per ablation condition, with sequence-identity-grouped CV | Fig. 1A |
| `5_hf_benchmark.py` | Evaluation on the HF `danliu1226/virus_human_benchmarking` dataset | Table 1 |
| `6_head_auc_maps.py` | Per-head AUROC heatmaps, base and fine-tuned | Supplementary per-head heatmaps |
| `9_gold_benchmark.py` | Balanced three-fold cross-species evaluation on the gold benchmark | Table 2 |

Gaps in the numbering (3, 4, 7, 8) are analyses that are not part of this
release.

### Attention layout

`0_extract_attention.py --pathogen <key> --checkpoint <lora_ckpt> [--also-base]`
writes `{pathogen}_{hpi,random_cross,intra_host,random_intra}[_base]_attention.npz`
to `--output-dir` (default `data/attention/`). Scripts 1, 2 and 6 read:

- fine-tuned models: `data/ablation_attention/<model_key>/checkpoint-<step>/`
  (`<model_key>` as in `utils.ABLATION_MODELS`, e.g.
  `proteomelm_hpi_s_abl8_seed_total_block`);
- base model: `data/ablation_attention/base/` (the `_base` files).

Each `.npz` has keys `attention_a_to_b`, `attention_b_to_a`
(shape `n_pairs × n_layers × n_heads`) and `pair_ids`; load the symmetrized
attention with `(npz["attention_a_to_b"] + npz["attention_b_to_a"]) / 2`
(`utils.load_attention`).

## Shared utilities

`utils.py` holds constants and helpers reused across the numbered scripts:

- `DATASETS` (the 10-pathogen list), `DATASET_GROUP` (virus/bacteria),
  `CLEAN_LABELS`, `ABLATION_MODELS` (attention subdir → label);
- `load_attention()`, `load_attention_with_pairs()`;
- `lr_auroc()` (StratifiedKFold) and `lr_auroc_grouped_identity_cv()`
  (StratifiedGroupKFold on MMseqs2 sequence-identity clusters);
- ESM-C / ProteomeLM helpers shared by scripts 0, 5 and 9:
  `encode_sequences()`, `encode_split_cached()` (host-first, per-species
  embedding caches), `add_gene_name_aliases()`, `load_finetuned()`,
  `load_base()`, `run_inference()`.

## Dependencies

```text
numpy pandas matplotlib scikit-learn scipy requests biopython
esm peft datasets        # scripts 0, 5, 9 (ESM-C encoding, LoRA, HF benchmark)
mmseqs2 (binary on PATH) # sequence-identity clustering (2_ablations.py, build_gold_benchmark.py)
```
