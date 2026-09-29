# PPI benchmarks (Bernett, D-SCRIPT)

Batch evaluation of ProteomeLM attention as a protein-protein interaction
signal on the Bernett et al. (2024) gold standard and the D-SCRIPT species
datasets. It used to live in the package as `proteomelm/ppi/{main,experiment_runner,evaluation}.py`.

- `run_benchmarks.py`: the command-line entry point and the runners.
- `evaluation.py`: the metrics. `AttentionAnalyzer` computes, on the test split,
  the AUC/AUPR of every attention head used alone as a score, and the same
  metrics for per-layer sums, the whole-model sum and the top 10 PCA
  components. This per-head AUC is the unsupervised PPI computation of the
  ProteomeLM paper. `PerformanceEvaluator` also trains `EnhancedPPIModel`
  heads on several feature combinations: `Att`, `ProteomeLM`, `ESM`,
  combinations of these, and `ProteomeLM-Layer<i>`.

Pair features come from `proteomelm.ppi.feature_extraction.PPIFeatureExtractor`.
It runs one forward pass over the whole proteome and takes `a_ij + a_ji` per
layer and head. ProteomeLM runs on the CPU and ESM-C on
`cuda:0` (see `ExtractionConfig`). The full N x N attention of every layer is
held in RAM, which takes about 135 GB for the mouse proteome.

## Inputs (under `PROTEOMELM_DATA_ROOT`)

| `--dataset` | Directory | Files |
| --- | --- | --- |
| `bernett` | `bernett/` | `human_gold.faa`, `Intra{0,1,2}_{pos,neg}_rr.txt` (Intra-0 = val, Intra-1 = train, Intra-2 = test) |
| `dscript:<species>` | `dscript/<species>/` | `<species>.faa`, `<species>_{train,val,test}_interaction.tsv` (`id1 <tab> id2 <tab> label`) |
| `benchmark:<species>` | `benchmark/<species>/` | same format as D-SCRIPT |
| `cross-species` | `dscript/<species>/` for each `--species` | same as `dscript:<species>` |

ESM-C embeddings are cached next to the data (`dump_dict_esm_<dataset>.pt`,
validated by a content hash). Pair features are written to
`dump_dict.pkl` in the same directory.

## Usage

Run from the repository root:

```bash
# per-head attention AUCs of ProteomeLM-S on Bernett
python -m experiments.ppi_benchmarks.run_benchmarks --dataset bernett --checkpoint Bitbol-Lab/ProteomeLM-S

# also train supervised heads (5 replicas per feature combination)
python -m experiments.ppi_benchmarks.run_benchmarks --dataset dscript:yeast --mode supervised

# every saved checkpoint of a training run: <run_dir>/checkpoint-<n>
python -m experiments.ppi_benchmarks.run_benchmarks --dataset dscript:human \
    --checkpoint /path/to/run_dir --checkpoint-numbers 15 30 45 --model-name my-run

# supervised heads trained on D-SCRIPT human, tested on each species
python -m experiments.ppi_benchmarks.run_benchmarks --dataset cross-species --mode supervised
```

## Outputs

Written to `--results-dir`, which defaults to the dataset directory
(`DATA_ROOT/dscript` for `cross-species`). Rows of the same model and
checkpoint are replaced on re-runs.

- `unsupervised_results.csv`: `model_name, checkpoint, layer, head, auc, aupr`,
  plus `sum`/`mean`/`max` summary rows (`head = all`).
- `supervised_results.csv`: `model_name, checkpoint, feature_combo, replica, auc, aupr`
  (test split).
- Trained heads: `<dataset>/trained_models/<model_name>/checkpoint_<n>/<combo>_replica_<i>.pt`
  and a `model_registry.pkl`. For cross-species runs they go to
  `dscript/cross_species_models/<model_name>/checkpoint_<n>/`. Disable saving
  with `--no-save-models`.

A checkpoint given without `--checkpoint-numbers` is labeled `final`.

## Known limitations

- Cross-species metrics are only logged. The CSVs have no species column:
  supervised cross-species AUCs are not written at all, and the unsupervised
  per-head rows of all species land in one table without a way to tell them
  apart.
- The bundled supervised models `data/interactomes/enhanced_ppi_model_*.pt`
  are not produced here. They come from
  `experiments/ppi_bundled_models/train_bundled.py`.
