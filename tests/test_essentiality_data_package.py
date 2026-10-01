"""experiments.essentiality: the Hugging Face dataset (package_data) and its `fetch` step
rebuild the pipeline's FASTAs, label pickles, fold pickle and minimal-cell labels."""
import json
import math
import os
import pickle
import sys
import types

import pandas as pd

from experiments.essentiality.data import (OGEE_FILES, OGEE_HUB_DIR, run_fetch, write_minimal_cells,
                                           write_pipeline_files)
from experiments.essentiality.evaluate import GetMinimalCellLabels
from experiments.essentiality.package_data import genomes_table, proteins_table, training_label

GENOMES = {
    111: ("uniprotkb_data_taxid111.fasta",
          [("sp|P1|A_X", "sp|P1|A_X Protein A OS=X", "MKV", "geneA", ["a1"], ["NE", "E"]),
           ("sp|P2|B_X", "sp|P2|B_X Protein B OS=X", "MAAL", None, None, []),
           ("sp|P3|C_X", "sp|P3|C_X Protein C OS=X", "MW", "geneC", ["c1", "c2"], ["E", "C"])]),
    580240: ("othersource_data_taxid580240.fasta",
             [("YAL001C", "YAL001C TFC3 SGDID:S000000001, Chr I", "MSTQ", "S000000001", ["YAL001C"], [math.nan]),
              ("YAL002W", "YAL002W VPS8 SGDID:S000000002, Chr I", "MEQN", "S000000002", ["YAL002W"], ["NE"])]),
}


def write_source(tmp_path):
    fasta_folder, label_folder = tmp_path / "fasta", tmp_path / "labels"
    fasta_folder.mkdir()
    label_folder.mkdir()
    split = {}
    for taxid, (name, proteins) in GENOMES.items():
        with open(fasta_folder / name, "w") as f:
            for pid, header, seq, *_ in proteins:
                f.write(f">{header}\n{seq[:2]}\n{seq[2:]}\n")  # wrapped, as in the originals
        labels = {pid: {"tax id": taxid, "gene": gene, "synonims": syn, "Essentiality": calls}
                  for pid, _, _, gene, syn, calls in proteins}
        with open(label_folder / f"labeled_essentiality_taxid{taxid}.pkl", "wb") as f:
            pickle.dump(labels, f)
        split.update({pid: i % 5 for i, (pid, *_) in enumerate(proteins)})
    return fasta_folder, label_folder, split


def same_labels(a, b):
    def norm(d):
        return {k: {**v, "Essentiality": ["nan" if isinstance(c, float) and math.isnan(c) else c
                                          for c in v["Essentiality"]]} for k, v in d.items()}
    return norm(a) == norm(b)


def read_fasta(path):
    from Bio import SeqIO
    return [(r.id, r.description, str(r.seq)) for r in SeqIO.parse(str(path), "fasta")]


def test_training_label_is_last_e_or_ne_call():
    assert training_label(["NE", "E", "C"]) == "E"
    assert training_label(["C", "U"]) is None
    assert training_label([]) is None


def test_proteins_table_round_trips_through_parquet(tmp_path):
    fasta_folder, label_folder, split = write_source(tmp_path)
    table = proteins_table(str(fasta_folder), str(label_folder), split)
    assert list(table["label"]) == ["E", None, "E", None, "NE"]
    table.to_parquet(tmp_path / "proteins.parquet", index=False)
    table = pd.read_parquet(tmp_path / "proteins.parquet")

    out = tmp_path / "out"
    written = write_pipeline_files(table, str(out / "fasta"), str(out / "labels"))
    assert sorted(written) == sorted(GENOMES)
    for taxid, (name, _) in GENOMES.items():
        assert read_fasta(out / "fasta" / name) == read_fasta(fasta_folder / name)
        with open(label_folder / f"labeled_essentiality_taxid{taxid}.pkl", "rb") as f:
            original = pickle.load(f)
        with open(out / "labels" / f"labeled_essentiality_taxid{taxid}.pkl", "rb") as f:
            rebuilt = pickle.load(f)
        assert same_labels(original, rebuilt)
    # existing files are kept
    assert write_pipeline_files(table, str(out / "fasta"), str(out / "labels")) == []


def test_genomes_table_roles(tmp_path):
    fasta_folder, label_folder, split = write_source(tmp_path)
    table = proteins_table(str(fasta_folder), str(label_folder), split)
    genomes = genomes_table(table, excluded=[], names={580240: "W303"}).set_index("taxid")
    assert genomes.loc[580240, "role"] == "held out (Fig. 5B)"
    assert genomes.loc[580240, "organism"] == "Saccharomyces cerevisiae S288C"
    assert genomes.loc[111, "role"] == "excluded"  # fewer than 10 proteins


def test_minimal_cell_labels_from_tsv(tmp_path):
    minimal = pd.DataFrame({"taxid": [2144189] * 3, "protein_id": ["JCVISYN3A_0001", "JCVISYN3A_0002", "JCVISYN3A_0003"],
                            "header": ["JCVISYN3A_0001 dnaA", "JCVISYN3A_0002 dnaN", "JCVISYN3A_0003 x"],
                            "sequence": ["MK", "MA", "MW"], "label": ["E", "QE", "No_label"]})
    write_minimal_cells(minimal, str(tmp_path))
    fasta = tmp_path / "ncbi_data_taxid2144189.fasta"
    coarse, labels = GetMinimalCellLabels()(taxid=2144189, folder_path=str(tmp_path), fasta_file=str(fasta),
                                            return_gene_to_labels=True)
    assert labels == {"JCVISYN3A_0001": "E", "JCVISYN3A_0002": "QE", "JCVISYN3A_0003": "No_label"}
    assert coarse["E"] == 1 and coarse["QE"] == 1


def test_run_fetch_writes_pipeline_inputs(tmp_path, monkeypatch):
    fasta_folder, label_folder, split = write_source(tmp_path)
    hub = tmp_path / "hub"
    (hub / "data").mkdir(parents=True)
    (hub / OGEE_HUB_DIR).mkdir(parents=True)
    for name in OGEE_FILES:
        (hub / OGEE_HUB_DIR / name).write_text(f"{name}\n")
    proteins_table(str(fasta_folder), str(label_folder), split).to_parquet(hub / "data" / "proteins.parquet")
    pd.DataFrame({"taxid": [766747], "protein_id": ["MMSYN1_0001"], "header": ["MMSYN1_0001"],
                  "sequence": ["MK"], "label": ["E"]}).to_parquet(hub / "data" / "minimal_cells.parquet")
    (hub / "data" / "info.json").write_text(json.dumps({"split_threshold": 40, "n_splits": 5, "split_seed": 0}))
    monkeypatch.setitem(sys.modules, "huggingface_hub",
                        types.SimpleNamespace(snapshot_download=lambda *a, **k: str(hub)))

    d = tmp_path / "data_dir"
    cfg = {"paths": {"ogee_data_dir": str(d / "ogee"), "fasta_folder": str(d / "fasta"),
                     "label_folder": str(d / "labels"), "splits_folder": str(d),
                     "minimalcell_folder": str(d / "minimalcell")},
           "split": {"prefix": "all_sequences2_labelled_splits", "threshold": 40, "n_splits": 5, "seed": 0}}
    os.makedirs(d)
    split_path = run_fetch(cfg)
    assert os.path.basename(split_path) == "all_sequences2_labelled_splits_40_seed0.pkl"
    with open(split_path, "rb") as f:
        assert pickle.load(f) == split
    assert sorted(os.listdir(d / "ogee")) == sorted(OGEE_FILES)
    assert sorted(os.listdir(d / "fasta")) == sorted(name for name, _ in GENOMES.values())
    assert (d / "minimalcell" / "minimalcell_taxid766747_labels.tsv").exists()
