# experiments/essentiality/

Gene-essentiality prediction from frozen ProteomeLM embeddings (ProteomeLM PNAS paper,
Fig. 5; code by Gionata Zalaffi). A small classifier is trained on each layer's
per-protein hidden states over the OGEE genomes. This folder reproduces:

- **Fig. 5A**: test AUROC vs relative layer depth for ESM-C 600M and ProteomeLM-XS/S/M/L
  (2-layer classifier, bootstrap-weighted mean over seeds 42–46).
- **Fig. 5B**: labelled vs predicted essential genes (donuts) for the held-out genomes
  *S. cerevisiae* (580240) and *E. coli* K-12 (83333), and the minimal cells JCVI-Syn1.0
  (766747) and JCVI-Syn3A (2144189). The N highest-scoring proteins are called
  essential, with N the number of labelled essential genes.
- **SI**: trained vs random vs resampled ("statistics") ProteomeLM weights; 1/2/3-layer
  classifiers; per-layer intrinsic dimension, PCA ID and matrix entropy; comparison with
  Evo (`notebooks/comparison_othermethods.ipynb`).

| file | role |
|---|---|
| `config.yaml` | final published configuration + data layout (paths relative to `--data-dir`) |
| `data.py` | `download`, `labels`, `split` (mmseqs2 40% clusters → 5 folds: 0 test, 1 val, 2–4 train) |
| `train.py` | ESM-C + ProteomeLM embeddings, per-layer classifiers, `make-baselines` |
| `evaluate.py` | test-fold metric pickles; whole-genome predictions for Fig. 5B |
| `figures.py` | `fig5a`, `fig5b`, `baselines`, `depth`, `interpretability` |
| `interpretability.py` | Two-NN ID / entropy / PCA per layer and genome |
| `run_all.py` | driver for all stages (`--dry-run` prints the commands) |

## Requirements

The `proteomelm` environment, plus: `mmseqs` (split), `gdown` (OGEE tables, Syn1.0 labels
and one Fitness Browser table are mirrored on Google Drive), the NCBI
[`datasets` CLI](https://www.ncbi.nlm.nih.gov/datasets/docs/v2/command-line-tools/download-and-install/)
(`$NCBI_DATASETS`, else `~/datasets`, else on `PATH`), `dadapy` (interpretability),
`ncbi-taxonomist` (superkingdoms in the interpretability figures) and, for the notebook
only, `evo-model` in its own environment. `wandb` is optional (`--wandb`).

## Data

Everything lives under `--data-dir` (default `$PROTEOMELM_DATA_ROOT/essentiality`) with the
file names of the original server, so an existing data folder is reused as is:

- **OGEE** `gene_essentiality.txt`, `genes.txt`, `datasets.txt` → `ogee_data/` (Google Drive
  mirror, IDs in `data.get_dataset_info_df`; the OGEE site's certificate had expired).
- **Proteomes** → `all_fasta3/`: UniProt reference proteomes (FTP), else UniProtKB (REST), else
  NCBI (`datasets`). Custom sources: SGD `orf_trans.fasta.gz` (yeast), Fitness Browser
  `orgSeqs.cgi` (6 bacteria), fixed NCBI accessions (`data.get_data_from_othersources`).
- **Labels** → `label_to_ess7/labeled_essentiality_taxid{t}.pkl` and duplicate-free
  FASTAs → `all_fasta_noduplicates2/`.
- **Minimal cells** → `inference_data/minimalcell/`: NCBI proteomes, SynWiki (Syn3A) and
  Hutchison et al. 2016 Database S1 (Syn1.0), downloaded on first use.
- **Evo comparison**: SGD `S288C_reference_genome_R64-5-1_20240529` (genome release
  tarball) and `saccharomyces_cerevisiae.gff`, unpacked into `data/comparison_data/`.

## Pipeline

```bash
PY="python -m experiments.essentiality"          # run from the repo root
$PY.run_all data split                           # download + labels, then the 40% cluster split
$PY.run_all train evaluate --gpus 0,1 --max-jobs 6   # ESMC,XS,S,M,L x seeds 42-47 x 1/2 layers
$PY.run_all figures                              # Fig. 5A (2- and 1-layer) and Fig. 5B
$PY.run_all baselines interpretability           # SI
```

Stages can be run on subsets (`--checkpoints S --seeds 42 --classifier-layers 2`). The
ProteomeLM weights default to Hugging Face `Bitbol-Lab/ProteomeLM-{size}`, which are
tensor-identical to the `checkpoint-210` weights used for the paper (checked against a copy of those checkpoints; `--checkpoint-dir DIR` reads
`DIR/ProteomeLM-{size}/checkpoint-210`). Each module also has its own CLI (`--help`).

**Embeddings.** `group_embeds` is the protein's own ESM-C embedding (no OrthoDB lookup), and
ProteomeLM runs in bfloat16. Layer `i` of a ProteomeLM run is `hidden_states[i]`, for
`i < n_layers`: index 0 is the input projection, and the output of the last block is never
used. ESM-C layer `i` is the mean-pooled output of block `i + 1` (all 36 blocks).

## Known differences from the paper text

- Normalization: the paper describes genome-wide mean/std normalization; the final config
  has `normalize_genome: False`, and when on, `_normalize_tensor` standardizes each protein
  over its features, not across the genome.
- Excluded genomes: 7 taxids (580240, 83333, 199310, 679895, 941322, 1380365, 1124478) vs 4
  in the paper; 580240 and 83333 are the Fig. 5B held-out genomes.
