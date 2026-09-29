"""experiments.essentiality.data: cluster-to-fold assignment of the essentiality split.

The mmseqs TSV is grouped by cluster representative, not in FASTA order, so fold
labels must come from the TSV member ids for each cluster to land in one fold.
"""
import random
from collections import defaultdict

import pytest

from experiments.essentiality.data import (assign_folds, cluster_fold_rows, label_folds, mmseqs_to_fasta_ids,
                                           read_mmseqs_clusters, split_for_crossval, write_indexed_fasta)


def make_clusters(n_clusters=60, seed=0):
    """Synthetic mmseqs output: (reps, members) grouped by representative, and a FASTA
    order of the same ids that interleaves clusters (as genomes do)."""
    rng = random.Random(seed)
    reps, members = [], []
    for c in range(n_clusters):
        size = rng.randint(1, 8)
        ids = [f"c{c}_m{k}" for k in range(size)]
        reps += [ids[0]] * size
        members += ids
    fasta_order = members[:]
    rng.shuffle(fasta_order)
    return reps, members, fasta_order


def folds_per_cluster(reps, members, labelling):
    by_cluster = defaultdict(set)
    for rep, member in zip(reps, members):
        by_cluster[rep].add(labelling[member])
    return by_cluster


def test_every_cluster_in_exactly_one_fold():
    reps, members, _ = make_clusters()
    labelling = assign_folds(reps, members, n_splits=5, seed=0)
    assert set(labelling) == set(members)
    assert all(len(f) == 1 for f in folds_per_cluster(reps, members, labelling).values())


def test_fold_sizes_close_to_one_over_n():
    reps, members, _ = make_clusters(n_clusters=400)
    labelling = assign_folds(reps, members, n_splits=5, seed=1)
    counts = [list(labelling.values()).count(k) for k in range(5)]
    n = len(members)
    # Folds 0-3 stop right after reaching n/5 (overshoot < one cluster, <= 8 rows);
    # the last fold takes the remainder.
    for k in range(4):
        assert n / 5 <= counts[k] < n / 5 + 8
    assert sum(counts) == n
    assert counts[4] > 0


def test_deterministic_given_seed():
    reps, members, _ = make_clusters()
    assert assign_folds(reps, members, seed=3) == assign_folds(reps, members, seed=3)
    assert assign_folds(reps, members, seed=3) != assign_folds(reps, members, seed=4)


def test_fold_rows_partition_all_rows():
    reps, _, _ = make_clusters()
    rows = cluster_fold_rows(reps, n_splits=5, rng=random.Random(0))
    flat = sorted(r for fold in rows for r in fold)
    assert flat == list(range(len(reps)))


def test_fasta_order_mapping_would_split_clusters():
    """Labelling TSV rows with FASTA-order ids would split clusters; member ids do not."""
    reps, members, fasta_order = make_clusters()
    rows = cluster_fold_rows(reps, n_splits=5, rng=random.Random(0))
    legacy = label_folds(rows, fasta_order)
    fixed = label_folds(rows, members)
    straddling = [c for c, f in folds_per_cluster(reps, members, legacy).items() if len(f) > 1]
    assert straddling, "FASTA-order mapping should put some cluster in several folds"
    assert all(len(f) == 1 for f in folds_per_cluster(reps, members, fixed).values())


def test_split_for_crossval_from_existing_tsv(tmp_path):
    reps, members, fasta_order = make_clusters(n_clusters=20)
    fasta = tmp_path / "all.fasta"
    fasta.write_text("".join(f">{i} desc\nMKV\n" for i in fasta_order))
    tsv = tmp_path / "clusters40.tsv_cluster.tsv"
    tsv.write_text("".join(f"{r}\t{m}\n" for r, m in zip(reps, members)))
    assert read_mmseqs_clusters(str(tsv)) == (reps, members)
    out = tmp_path / "splits_40_seed0.pkl"
    labelling = split_for_crossval(str(fasta), str(out), str(tmp_path), threshold=40, split_seed=0,
                                   cluster_tsv=str(tsv))
    assert out.exists()
    assert labelling == assign_folds(reps, members, seed=0)


def test_split_for_crossval_rejects_tsv_of_other_fasta(tmp_path):
    fasta = tmp_path / "all.fasta"
    fasta.write_text(">a\nMKV\n>b\nMKV\n")
    tsv = tmp_path / "c.tsv"
    tsv.write_text("a\ta\n")
    with pytest.raises(ValueError, match="absent"):
        split_for_crossval(str(fasta), str(tmp_path / "o.pkl"), str(tmp_path), cluster_tsv=str(tsv))


def test_ids_that_look_like_nan_are_kept_as_strings(tmp_path):
    tsv = tmp_path / "c.tsv"
    tsv.write_text("NA\tNA\nNA\tnan\n")
    assert read_mmseqs_clusters(str(tsv)) == (["NA", "NA"], ["NA", "nan"])


def test_mmseqs_ids_map_back_to_uniprot_headers():
    """mmseqs writes the accession for sp|ACC|NAME / tr|ACC|NAME headers."""
    fasta_ids = ["sp|P12345|A_ECOLI", "tr|Q9XYZ1|B_ECOLI", "YAL001C", "b0001"]
    assert mmseqs_to_fasta_ids(["P12345", "Q9XYZ1", "YAL001C", "b0001"], fasta_ids) == fasta_ids
    with pytest.raises(ValueError, match="match no FASTA record"):
        mmseqs_to_fasta_ids(["unknown"], fasta_ids)


def test_write_indexed_fasta(tmp_path):
    src = tmp_path / "in.fasta"
    src.write_text(">sp|P1|X desc\nMKV\n>g2\nMAA\n")
    ids = write_indexed_fasta(str(src), str(tmp_path / "out.fasta"))
    assert ids == ["sp|P1|X", "g2"]
    assert (tmp_path / "out.fasta").read_text() == ">0\nMKV\n>1\nMAA\n"
