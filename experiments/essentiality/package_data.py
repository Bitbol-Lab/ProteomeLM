"""Package the essentiality data as the Hugging Face dataset ``Bitbol-Lab/ProteomeLM-ess-data``.

Reads the outputs of the ``download``/``labels``/``split`` stages under ``--data-dir`` and
writes, ready for ``huggingface-cli upload --repo-type dataset``::

    raw/ogee_v3/{gene_essentiality,genes,datasets}.txt   OGEE v3 tables, unmodified
    data/proteins.parquet        one row per protein of the 89 OGEE genomes (FASTA order)
    data/genomes.tsv             one row per genome
    data/minimal_cells.parquet   JCVI-Syn1.0 / Syn3A proteomes and labels (Fig. 5B)
    data/info.json               split parameters
    README.md                    dataset card

``python -m experiments.essentiality.data fetch`` turns it back into the pipeline's files::

    python -m experiments.essentiality.package_data   # -> {data-dir}/hf_release/ProteomeLM-ess-data
"""
import argparse
import json
import os
import pickle
import re
import shutil
from typing import Dict, List, Optional, Sequence

import pandas as pd
import requests
from Bio import SeqIO

from experiments.essentiality.common import DATA_REPO, load_config, resolve, splits_filename
from experiments.essentiality.data import MINIMAL_CELL_TAXIDS, OGEE_FILES, OGEE_HUB_DIR

HOLDOUT_TAXIDS = (580240, 83333)  # S. cerevisiae S288C, E. coli K-12 MG1655 (Fig. 5B)
MINIMAL_CELL_NAMES = {766747: "JCVI-Syn1.0", 2144189: "JCVI-Syn3A"}
MINIMAL_CELL_SOURCES = {766747: "Hutchison et al. 2016, Science 351:aad6253, Database S1",
                        2144189: "SynWiki (synwiki.uni-goettingen.de)"}


def training_label(calls: Sequence[str]) -> Optional[str]:
    """Label used for training: the last E/NE call among a gene's OGEE calls (None if none)."""
    ess = [c for c in calls if c in ("E", "NE")]
    return ess[-1] if ess else None


def proteins_table(fasta_folder: str, label_folder: str, split: Optional[Dict[str, int]] = None) -> pd.DataFrame:
    """One row per protein of every ``{source}_taxid{t}.fasta`` in ``fasta_folder``, in FASTA
    order, with its OGEE calls from ``labeled_essentiality_taxid{t}.pkl`` and its fold."""
    rows = []
    for name in sorted(os.listdir(fasta_folder), key=lambda n: int(re.search(r"taxid(\d+)", n).group(1))):
        taxid = int(re.search(r"taxid(\d+)", name).group(1))
        with open(os.path.join(label_folder, f"labeled_essentiality_taxid{taxid}.pkl"), "rb") as f:
            labels = pickle.load(f)
        for record in SeqIO.parse(os.path.join(fasta_folder, name), "fasta"):
            entry = labels[record.id]
            assert entry["tax id"] == taxid, (name, record.id)
            rows.append({"taxid": taxid,
                         "protein_id": record.id,
                         "header": record.description,
                         "sequence": str(record.seq),
                         "gene": entry["gene"],
                         "synonyms": entry["synonims"],
                         "ogee_calls": list(entry["Essentiality"]),
                         "label": training_label(entry["Essentiality"]),
                         "fold": None if split is None else split[record.id],
                         "fasta_file": name})
    table = pd.DataFrame(rows)
    if split is not None:
        table["fold"] = table["fold"].astype("int8")
    return table


def ncbi_names(taxids: Sequence[int]) -> Dict[int, str]:
    """Scientific names from NCBI Taxonomy (empty on network errors)."""
    try:
        r = requests.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
                         params={"db": "taxonomy", "id": ",".join(map(str, taxids)), "retmode": "json"},
                         timeout=60)
        r.raise_for_status()
        result = r.json()["result"]
        return {int(t): result[str(t)]["scientificname"] for t in result["uids"]}
    except Exception as e:
        print(f"Warning: no organism names from NCBI ({e})")
        return {}


def genomes_table(proteins: pd.DataFrame, excluded: Sequence[int], names: Dict[int, str]) -> pd.DataFrame:
    """Per-genome counts and role: cross-validation, held out (Fig. 5B) or excluded."""
    # NCBI's taxid 580240 is W303, but its proteome here is the SGD S288C reference (as in the paper)
    names = {**names, 580240: "Saccharomyces cerevisiae S288C"} if 580240 in names else names
    rows = []
    for taxid, g in proteins.groupby("taxid", sort=False):
        n_e, n_ne = int((g["label"] == "E").sum()), int((g["label"] == "NE").sum())
        if taxid in HOLDOUT_TAXIDS:
            role = "held out (Fig. 5B)"
        elif taxid in excluded or len(g) < 10 or n_e + n_ne < 10:
            role = "excluded"
        else:
            role = "cross-validation"
        rows.append({"taxid": taxid, "organism": names.get(taxid, ""), "fasta_file": g["fasta_file"].iloc[0],
                     "proteins": len(g), "E": n_e, "NE": n_ne, "role": role})
    return pd.DataFrame(rows)


def minimal_cells_table(folder: str) -> pd.DataFrame:
    """Proteomes and labels of JCVI-Syn1.0 and Syn3A, as used for Fig. 5B."""
    from experiments.essentiality.evaluate import GetMinimalCellLabels
    rows = []
    for taxid in MINIMAL_CELL_TAXIDS:
        fasta = os.path.join(folder, f"ncbi_data_taxid{taxid}.fasta")
        _, labels = GetMinimalCellLabels()(taxid=taxid, folder_path=folder, fasta_file=fasta,
                                           return_gene_to_labels=True)
        for record in SeqIO.parse(fasta, "fasta"):
            rows.append({"taxid": taxid, "organism": MINIMAL_CELL_NAMES[taxid], "protein_id": record.id,
                         "header": record.description, "sequence": str(record.seq),
                         "label": labels[record.id], "label_source": MINIMAL_CELL_SOURCES[taxid]})
    return pd.DataFrame(rows)


def write_dataset_card(out: str, proteins: pd.DataFrame, genomes: pd.DataFrame, minimal: pd.DataFrame,
                       info: dict) -> None:
    roles = genomes["role"].value_counts()
    labelled = proteins["label"].notna()
    folds = proteins["fold"].value_counts().sort_index()
    held = genomes[genomes["role"] != "cross-validation"]
    held_rows = "\n".join(f"| {r.taxid} | {r.organism or '?'} | {r.role} | {r.proteins:,} | {r.E:,} | {r.NE:,} |"
                          for r in held.itertuples())
    mc = minimal.groupby(["organism", "taxid"])["label"].value_counts().unstack(fill_value=0)
    mc_rows = "\n".join(f"| {org} | {taxid} | {int(row.sum()):,} | {int(row.get('E', 0)):,} | {int(row.get('NE', 0)):,} |"
                        for (org, taxid), row in mc.iterrows())
    card = f"""---
license: cc-by-3.0
pretty_name: ProteomeLM gene-essentiality data (OGEE v3)
tags:
- biology
- proteomics
- gene-essentiality
- proteomelm
configs:
- config_name: proteins
  data_files: data/proteins.parquet
  default: true
- config_name: minimal_cells
  data_files: data/minimal_cells.parquet
---

# ProteomeLM gene-essentiality data

The data used to train and evaluate the gene-essentiality classifier of
[ProteomeLM](https://github.com/Bitbol-Lab/ProteomeLM)
([PNAS 2026](https://www.pnas.org/doi/10.1073/pnas.2524201123), Fig. 5; trained head:
[Bitbol-Lab/ProteomeLM-ess](https://huggingface.co/Bitbol-Lab/ProteomeLM-ess)).
The essentiality calls come from **OGEE v3**, the Online GEne Essentiality database
([Gurumayum et al., NAR 2021](https://doi.org/10.1093/nar/gkaa884)). The OGEE website
(v3.ogee.info) is currently unavailable, so its tables are mirrored here unmodified under
its CC BY 3.0 license. Please cite OGEE, and the primary screens it compiles, when you
use these labels.

## Files

| file | content |
|---|---|
| `raw/ogee_v3/gene_essentiality.txt` | OGEE essentiality calls (`dataset`, `taxaID`, `locus`, `gene`, `score`, `essentiality`, `pmid`, `Ref_db`) |
| `raw/ogee_v3/genes.txt` | OGEE gene annotations |
| `raw/ogee_v3/datasets.txt` | OGEE datasets (source, technique, condition, definition of essential) |
| `data/proteins.parquet` | {len(proteins):,} proteins of {len(genomes)} genomes, with labels and folds (config `proteins`) |
| `data/genomes.tsv` | one row per genome: organism, protein and label counts, role |
| `data/minimal_cells.parquet` | JCVI-Syn1.0 and JCVI-Syn3A proteomes and labels (config `minimal_cells`) |
| `data/info.json` | split parameters |

## `proteins`

Each OGEE genome's proteome (UniProt reference proteome, else UniProtKB, else NCBI or the
source named in OGEE) with exact duplicates (on the first 4,096 residues) removed. OGEE
calls were matched to proteins by gene name, locus tag and synonyms.

| column | |
|---|---|
| `taxid` | NCBI taxonomy id of the genome |
| `protein_id`, `header` | FASTA id and full header line |
| `sequence` | amino-acid sequence |
| `gene`, `synonyms` | gene name and synonyms used for matching |
| `ogee_calls` | all OGEE calls for the gene (`E` essential, `NE` non-essential, `C` conditional, ...) |
| `label` | training label: the last `E`/`NE` call, null if none |
| `fold` | cross-validation fold: 0 = test, 1 = validation, 2-4 = train |
| `fasta_file` | proteome file name in the original pipeline |

{int(labelled.sum()):,} proteins have a label ({int((proteins['label'] == 'E').sum()):,} E, \
{int((proteins['label'] == 'NE').sum()):,} NE). {roles.get('cross-validation', 0)} genomes form the \
cross-validation set; the others were not used for training:

| taxid | organism | role | proteins | E | NE |
|---|---|---|---|---|---|
{held_rows}

"Excluded" genomes have fewer than 10 labelled proteins or problematic labels.

**Split.** All {len(proteins):,} sequences were clustered with MMseqs2 \
(`easy-cluster --min-seq-id {info['split_threshold'] / 100:.2f}`), and whole clusters were assigned \
at random to {info['n_splits']} folds (seed {info['split_seed']}), so proteins sharing \
≥{info['split_threshold']}% identity are always in the same fold. Fold sizes: \
{", ".join(f"{int(k)}: {v:,}" for k, v in folds.items())}. The released head
[ProteomeLM-ess](https://huggingface.co/Bitbol-Lab/ProteomeLM-ess) was trained with an earlier fold
assignment, so evaluate it on the held-out genomes and the minimal cells, not on fold 0.

## `minimal_cells`

The minimal cells of Fig. 5B (NCBI proteomes, locus tags as ids). Labels: `E`, `NE`, `QE`
(quasi-essential), uncertain calls (`E?`, `NE?`), `Not_a_Gene` and `No_label`.

| organism | taxid | proteins | E | NE |
|---|---|---|---|---|
{mc_rows}

Sources: Hutchison et al. 2016, *Science* 351:aad6253, Database S1 (JCVI-Syn1.0) and
[SynWiki](https://synwiki.uni-goettingen.de) (JCVI-Syn3A).

## Usage

```python
import pandas as pd
proteins = pd.read_parquet("hf://datasets/{DATA_REPO}/data/proteins.parquet")
```

or `datasets.load_dataset("{DATA_REPO}", "proteins")`. To rebuild the input files of the
ProteomeLM pipeline (`experiments/essentiality` in the GitHub repository):

```bash
python -m experiments.essentiality.data fetch
```

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
  doi={{10.1073/pnas.2524201123}}
}}

@article{{gurumayum2021ogee,
  title={{OGEE v3: Online GEne Essentiality database with increased coverage of organisms and human cell lines}},
  author={{Gurumayum, Sanathoi and Jiang, Puzi and Hao, Xiaowen and Campos, Tulio L and Young, Neil D and Korhonen, Pasi K and Gasser, Robin B and Bork, Peer and Zhao, Xing-Ming and He, Li-jie and Chen, Wei-Hua}},
  journal={{Nucleic Acids Research}},
  volume={{49}},
  number={{D1}},
  pages={{D998--D1003}},
  year={{2021}},
  doi={{10.1093/nar/gkaa884}}
}}
```
"""
    with open(os.path.join(out, "README.md"), "w") as f:
        f.write(card)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Package the essentiality data as a Hugging Face dataset folder.")
    parser.add_argument("--data-dir", default=None, help="Root of all data paths (default DATA_ROOT/essentiality)")
    parser.add_argument("--config", default=None)
    parser.add_argument("--split-seed", type=int, default=None, help="Split to package (default: config)")
    parser.add_argument("--out", default="hf_release/ProteomeLM-ess-data", help="Output folder, relative to --data-dir")
    args = parser.parse_args(argv)
    cfg = load_config(args.config, args.data_dir)
    p, s = cfg["paths"], cfg["split"]
    seed = s["seed"] if args.split_seed is None else args.split_seed
    out = resolve(cfg["data_dir"], args.out)
    os.makedirs(os.path.join(out, OGEE_HUB_DIR), exist_ok=True)
    os.makedirs(os.path.join(out, "data"), exist_ok=True)

    for name in OGEE_FILES:
        shutil.copyfile(os.path.join(p["ogee_data_dir"], name), os.path.join(out, OGEE_HUB_DIR, name))
    with open(os.path.join(p["splits_folder"], splits_filename(s["prefix"], s["threshold"], seed)), "rb") as f:
        split = pickle.load(f)
    proteins = proteins_table(p["fasta_folder"], p["label_folder"], split)
    proteins.to_parquet(os.path.join(out, "data", "proteins.parquet"), index=False, compression="zstd")
    genomes = genomes_table(proteins, cfg["classifier"]["which_taxids_to_exclude"],
                            ncbi_names(proteins["taxid"].unique().tolist()))
    genomes.to_csv(os.path.join(out, "data", "genomes.tsv"), sep="\t", index=False)
    minimal = minimal_cells_table(p["minimalcell_folder"])
    minimal.to_parquet(os.path.join(out, "data", "minimal_cells.parquet"), index=False, compression="zstd")
    info = {"split_threshold": s["threshold"], "n_splits": s["n_splits"], "split_seed": seed}
    with open(os.path.join(out, "data", "info.json"), "w") as f:
        json.dump(info, f, indent=2)
    write_dataset_card(out, proteins, genomes, minimal, info)
    for root, _, files in sorted(os.walk(out)):
        for name in sorted(files):
            path = os.path.join(root, name)
            print(f"{os.path.relpath(path, out)}  ({os.path.getsize(path) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
