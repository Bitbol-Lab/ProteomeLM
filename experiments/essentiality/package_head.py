"""Package a trained essentiality classifier as a ProteomeLM-ess head folder.

Writes ``config.json`` + ``model.safetensors`` (the format read by
``proteomelm.essentiality.load_head``) and a ``README.md`` model card, ready for
``huggingface-cli upload Bitbol-Lab/ProteomeLM-ess <folder>``. The default input
is the classifier of Fig. 5B (``250801-2layer-082ev5el``: ProteomeLM-L,
``hidden_states[8]``, 2-layer head, seed 42)::

    python -m experiments.essentiality.package_head            # -> {data-dir}/hf_release/ProteomeLM-ess
"""
import argparse
import json
import os
from pathlib import Path
from typing import Dict, Tuple

import torch

from experiments.essentiality.common import HF_REPO, load_config, plm_size_from_checkpoint, resolve
from proteomelm.essentiality import DEFAULT_HEAD, EssentialityHeadConfig, save_head

PUBLISHED_HEAD = "250801-2layer-082ev5el"


def head_config_from_classifier(cfg: dict) -> EssentialityHeadConfig:
    """Head config from a training ``config.json`` (only trained-ProteomeLM, 2-layer heads
    without layer norm or genome normalization can be packaged)."""
    problems = []
    if cfg.get("model_id") != "2layer":
        problems.append(f"model_id={cfg.get('model_id')!r} (need '2layer')")
    if cfg.get("use_esmc_as_input"):
        problems.append("ESM-C input (need a ProteomeLM backbone)")
    if (cfg.get("which_weights") or "trained") != "trained":
        problems.append(f"which_weights={cfg.get('which_weights')!r} (need 'trained')")
    if cfg.get("use_layernorm") or cfg.get("normalize_genome"):
        problems.append("layer norm / genome normalization is not supported by the inference head")
    size = plm_size_from_checkpoint(cfg.get("proteomeLM_checkpoint"))
    if size is None:
        problems.append(f"cannot parse the ProteomeLM size from {cfg.get('proteomeLM_checkpoint')!r}")
    if problems:
        raise ValueError("Cannot package this classifier: " + "; ".join(problems))
    return EssentialityHeadConfig(
        backbone=HF_REPO.format(size=size),
        layer=int(cfg["which_hidden_layer"]),
        input_dim=int(cfg["hidden_dim"]),
        hidden_dim=int(cfg["classifier_hidden_dim"]),
        dropout=float(cfg["dropout"]),
        num_labels=int(cfg.get("num_labels", 2)),
        id2label={"0": "essential", "1": "non-essential"},  # training labels: E=0, NE=1
    )


def is_classifier_checkpoint(folder) -> bool:
    """A training-pipeline classifier folder (``config.json`` + ``pytorch_model.bin``)."""
    return os.path.isfile(os.path.join(folder, "pytorch_model.bin")) and os.path.isfile(os.path.join(folder, "config.json"))


def load_classifier_checkpoint(folder) -> Tuple[EssentialityHeadConfig, Dict[str, torch.Tensor]]:
    with open(os.path.join(folder, "config.json")) as f:
        cfg = json.load(f)
    state_dict = torch.load(os.path.join(folder, "pytorch_model.bin"), map_location="cpu", weights_only=True)
    return head_config_from_classifier(cfg), state_dict


MODEL_CARD = """---
license: apache-2.0
library_name: pytorch
base_model: {backbone}
tags:
- biology
- proteomics
- gene-essentiality
- protein-language-model
- proteomelm
---

# ProteomeLM-ess

Gene-essentiality classifier of [ProteomeLM](https://github.com/Bitbol-Lab/ProteomeLM)
([PNAS 2026](https://www.pnas.org/doi/10.1073/pnas.2524201123), Fig. 5). Given a whole
proteome, it gives each protein a probability of being essential.

## Model

- **Input**: a whole proteome (one genome), as a protein FASTA file.
- **Embeddings**: every protein is embedded with ESM-C 600M (mean over residues, first
  {esm_max_length} residues). The frozen backbone [{backbone}](https://huggingface.co/{backbone}) then reads the
  whole proteome at once, in bfloat16, with each protein's own ESM-C embedding as input and
  group embedding.
- **Head**: a two-layer classifier (linear {input_dim} → {hidden_dim}, ReLU, dropout {dropout}, linear
  {hidden_dim} → 2) on the backbone's `hidden_states[{layer}]` (index 0 is the input projection).
  Class 0 is *essential*, class 1 *non-essential*; the output is `p_essential` = softmax class 0.
- **Files**: `config.json` (backbone, layer, dimensions, labels) and `model.safetensors`
  ({n_params:,} parameters, stored in {weight_dtype}; inference runs the head in float32).

## Training data

Essentiality labels (E / NE) from [OGEE](https://v3.ogee.info) for 82 genomes, mostly
bacteria plus several eukaryotes (e.g. mouse, zebrafish, *Drosophila*, *C. elegans*,
*Arabidopsis*, fission yeast). The head was trained on the training folds of the paper's
gene-level train / validation / test split, with Adam (learning rate 1e-3) and early stopping
on the validation-fold AUPR (ProteomeLM-L layer 8, seed 42).

Seven OGEE genomes were held out of training entirely: *S. cerevisiae* S288C (taxid 580240),
*E. coli* K-12 MG1655 (83333), and taxids 199310, 679895, 941322, 1380365 and 1124478.

## Performance

Whole-genome predictions on genomes held out of training (as in Fig. 5B of the paper): the N
highest-ranked proteins are called essential, with N the number of labelled essential genes.
AUROC is computed on the genes labelled essential or non-essential.

| genome | proteins | labelled essential (N) | AUROC | labelled essential genes in the top N |
|---|---|---|---|---|
{performance_rows}

Labels: OGEE (*E. coli*, *S. cerevisiae*), Hutchison et al. 2016 (JCVI-Syn1.0) and SynWiki
(JCVI-Syn3A). The embeddings are computed in bfloat16, so scores vary slightly across GPUs
and library versions (top-N counts by a gene or two).

## Usage

```bash
pip install "proteomelm @ git+https://github.com/Bitbol-Lab/ProteomeLM"
python -m proteomelm.essentiality --fasta proteome.fasta --out scores.tsv --top-fraction 0.1
```

The TSV lists every protein, most essential first: `protein_id`, `p_essential`, `rank` and,
with `--top-n N`, `--top-fraction F` or `--threshold T`, `predicted_essential`. Ranks are
more reliable than the raw probabilities, which are not calibrated across organisms (the
fraction of essential genes ranges from a few percent in free-living bacteria to about half
in minimal cells), so a top-N or top-fraction rule is recommended.

or, from Python:

```python
from proteomelm.essentiality import EssentialityPredictor, read_fasta

ids, sequences = read_fasta("proteome.fasta")
predictor = EssentialityPredictor.from_pretrained("{repo_id}", device="cuda")
table = predictor.predict(ids, sequences, top_fraction=0.1)   # pandas DataFrame, sorted by rank
```

The input should be one complete proteome: ProteomeLM scores every protein in the context
of all the others. A GPU is recommended (a bacterial proteome takes about a minute, most of
it ESM-C); the first run downloads ESM-C 600M and {backbone}.

## Citation

```bibtex
@article{{malbranke2026proteomelm,
  title={{ProteomeLM: A proteome-scale language model enables accurate and rapid prediction of protein-protein interactions and gene essentiality across taxa}},
  author={{Malbranke, Cyril and Zalaffi, Gionata Paolo and Bitbol, Anne-Florence}},
  journal={{Proceedings of the National Academy of Sciences}},
  volume={{123}},
  number={{21}},
  pages={{e2524201123}},
  year={{2026}},
  publisher={{National Academy of Sciences}},
  doi={{10.1073/pnas.2524201123}},
  url={{https://www.pnas.org/doi/10.1073/pnas.2524201123}}
}}
```
"""


def write_model_card(folder, config: EssentialityHeadConfig, state_dict, performance_rows: str,
                     repo_id: str = DEFAULT_HEAD) -> Path:
    from proteomelm.essentiality import ESMC_MAX_LENGTH

    dtypes = sorted({str(v.dtype).replace("torch.", "") for v in state_dict.values()})
    text = MODEL_CARD.format(backbone=config.backbone, layer=config.layer, input_dim=config.input_dim,
                             hidden_dim=config.hidden_dim, dropout=config.dropout, esm_max_length=ESMC_MAX_LENGTH,
                             n_params=sum(v.numel() for v in state_dict.values()), weight_dtype="/".join(dtypes),
                             performance_rows=performance_rows.strip(), repo_id=repo_id)
    path = Path(folder) / "README.md"
    path.write_text(text)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Package an essentiality classifier as a ProteomeLM-ess head folder.")
    parser.add_argument("--data-dir", default=None, help="Root of all data paths (default DATA_ROOT/essentiality)")
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint", default=None,
                        help=f"Classifier folder (default: the published {PUBLISHED_HEAD})")
    parser.add_argument("--out", default="hf_release/ProteomeLM-ess", help="Output folder, relative to --data-dir")
    parser.add_argument("--performance-rows", default=None,
                        help="Markdown table rows for the model card (a file); default: a placeholder row")
    args = parser.parse_args(argv)
    cfg = load_config(args.config, args.data_dir)
    checkpoint = resolve(cfg["data_dir"], args.checkpoint) if args.checkpoint else \
        os.path.join(cfg["paths"]["published_classifier_checkpoints_folder"], PUBLISHED_HEAD)
    out = resolve(cfg["data_dir"], args.out)
    config, state_dict = load_classifier_checkpoint(checkpoint)
    save_head(out, config, state_dict)
    rows = Path(args.performance_rows).read_text() if args.performance_rows else "| (to fill) | | | | |"
    write_model_card(out, config, state_dict, rows)
    for name in sorted(os.listdir(out)):
        print(f"{os.path.join(out, name)}  ({os.path.getsize(os.path.join(out, name)) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
