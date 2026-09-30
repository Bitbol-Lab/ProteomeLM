#!/usr/bin/env python3
"""
Build a rigorous gold-standard virus-human (and bacteria-human) PPI benchmark.

Multi-level holdout strategy
-----------------------------
1. PATHOGEN-LEVEL  (most important): entire pathogen species held out from
   training.  Tests true generalisation to unseen pathogens / pandemic targets.
2. PROTEIN-LEVEL:  entire proteins (all their pairs) live in one split only.
   Guaranteed by construction from level 1.
3. SEQUENCE-LEVEL: MMseqs2 easy-cluster at 40 % identity on pathogen proteins
   across splits. Train proteins similar to val/test are removed; val proteins
   similar to test are removed.
4. RANDOM NEGATIVES: random cross-species pairs (any pathogen protein × any
   host protein not in the known positive set), length-matched to positives.
   Bait negatives (known-HPI pathogen proteins paired with random host proteins)
   are excluded to avoid false negatives from untested pairs.
5. LENGTH MATCHING: negative pairs are sampled to match the length distribution
   of the positive pairs (separate bins for pathogen and host lengths).

Input
-----
    data/benchmarks/raw/{pathogen}/combined_proteome.fasta
    data/benchmarks/processed/{pathogen}/hpi_pairs.tsv

Output
------
    data/gold_benchmark/
        fold_a/
            train.tsv
            val.tsv
            test.tsv
            stats.txt
        fold_b/
            ...
        fold_c/
            ...

    data/gold_benchmark_loo/
        fold_hiv1_jager2012/
            train.tsv
            val.tsv
            test.tsv
            stats.txt
        ...

Usage
-----
    # Default balanced 3-fold species split:
    python build_gold_benchmark.py

    # Legacy custom single split:
    python build_gold_benchmark.py \\
        --train-pathogens hiv1_jager2012 hpv hsv1 salmonella chlamydia \\
        --val-pathogens   ebv \\
        --test-pathogens  influenza_a tuberculosis yersinia sars_cov2_zhou2022

    # Stricter sequence identity threshold:
    python build_gold_benchmark.py --seq-id 0.30

    # True leave-one-pathogen-out benchmark (no validation split):
    python build_gold_benchmark.py --leave-one-out
"""

from __future__ import annotations

import argparse
import random
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR   = Path(__file__).resolve().parent
BENCH_DIR    = SCRIPT_DIR / "data" / "benchmarks"
PROC_DIR     = BENCH_DIR  / "processed"
RAW_DIR      = BENCH_DIR  / "raw"
OUT_DIR      = SCRIPT_DIR / "data" / "gold_benchmark"
OUT_DIR_LOO  = SCRIPT_DIR / "data" / "gold_benchmark_loo"

VIRUS_PATHOGENS = {
    "hiv1_jager2012",
    "hpv",
    "hsv1",
    "ebv",
    "influenza_a",
    "sars_cov2_zhou2022",
}

BACTERIA_PATHOGENS = {
    "salmonella",
    "chlamydia",
    "yersinia",
    "tuberculosis",
}

# Default balanced 3-fold split built from three pathogen blocks.
# Folds A/B/C are permutations of the same three groups so the benchmark stays
# symmetric while each validation and test split still mixes viruses and
# bacteria.
DEFAULT_FOLD_SPECS = {
    "A": {
        "train": ["hiv1_jager2012", "hsv1", "sars_cov2_zhou2022", "yersinia"],
        "val":   ["ebv", "salmonella", "chlamydia"],
        "test":  ["influenza_a", "hpv", "tuberculosis"],
    },
    "B": {
        "train":  ["influenza_a", "hpv", "tuberculosis"],
        "val": ["hiv1_jager2012", "hsv1", "sars_cov2_zhou2022", "yersinia"],
        "test":   ["ebv", "salmonella", "chlamydia"],
    },
    "C": {
        "train":   ["ebv", "salmonella", "chlamydia"],
        "val":  ["influenza_a", "hpv", "tuberculosis"],
        "test": ["hiv1_jager2012", "hsv1", "sars_cov2_zhou2022", "yersinia"],
    },
}


# ---------------------------------------------------------------------------
# FASTA parsing
# ---------------------------------------------------------------------------

def parse_fasta(path: Path) -> dict[str, str]:
    """Return {uniprot_id: sequence} from a UniProt-format FASTA.

    Header format: >sp|UNIPROTID|HOST  or  >sp|UNIPROTID|PATHOGEN
    """
    seqs: dict[str, str] = {}
    current_id: str | None = None
    current_seq: list[str] = []
    with path.open() as f:
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                if current_id:
                    seqs[current_id] = "".join(current_seq)
                parts = line[1:].split("|")
                current_id = parts[1] if len(parts) >= 2 else line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)
    if current_id:
        seqs[current_id] = "".join(current_seq)
    return seqs


def parse_fasta_split(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    """Parse a combined HOST+PATHOGEN FASTA in one pass.

    Returns (pathogen_seqs, host_seqs) dicts keyed by UniProt accession.
    """
    pathogen_seqs: dict[str, str] = {}
    host_seqs:     dict[str, str] = {}
    current_id:    str | None = None
    current_is_host: bool = False
    current_seq:   list[str] = []

    def _flush():
        if current_id is None:
            return
        seq = "".join(current_seq)
        if current_is_host:
            host_seqs[current_id] = seq
        else:
            pathogen_seqs[current_id] = seq

    with path.open() as f:
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                _flush()
                parts = line[1:].split("|")
                current_id = parts[1] if len(parts) >= 2 else line[1:].split()[0]
                current_is_host = "|HOST" in line
                current_seq = []
            else:
                current_seq.append(line)
    _flush()
    return pathogen_seqs, host_seqs


def load_pathogen_data(pathogen: str) -> tuple[dict[str, str], dict[str, str], pd.DataFrame]:
    """Load sequences and HPI pairs for one pathogen.

    Returns (pathogen_seqs, host_seqs, hpi_pairs_df).
    hpi_pairs_df columns: protein_a (pathogen), protein_b (host).
    """
    fasta_path = RAW_DIR / pathogen / "combined_proteome.fasta"
    pairs_path = PROC_DIR / pathogen / "hpi_pairs.tsv"

    if not fasta_path.exists():
        raise FileNotFoundError(f"FASTA not found: {fasta_path}")
    if not pairs_path.exists():
        raise FileNotFoundError(f"HPI pairs not found: {pairs_path}")

    pathogen_seqs, host_seqs = parse_fasta_split(fasta_path)
    pairs = pd.read_csv(pairs_path, sep="\t")
    return pathogen_seqs, host_seqs, pairs


# ---------------------------------------------------------------------------
# Sequence clustering with MMseqs2
# ---------------------------------------------------------------------------

def cluster_seqs(seqs: dict[str, str], seq_id: float,
                 label: str = "sequences") -> dict[str, str]:
    """Cluster *seqs* with MMseqs2 easy-cluster and return {uid: rep_uid}."""
    if not seqs:
        return {}
    if len(seqs) == 1:
        only = next(iter(seqs))
        return {only: only}

    print(f"  Clustering {len(seqs):,} unique {label} at {seq_id*100:.0f}% …", end=" ",
          flush=True)
    with tempfile.TemporaryDirectory(prefix="mmseqs_bench_") as td:
        td_path    = Path(td)
        in_fa      = td_path / "input.fasta"
        out_prefix = td_path / "out"
        tmp_dir    = td_path / "tmp"

        with in_fa.open("w") as fh:
            for uid, seq in seqs.items():
                fh.write(f">{uid}\n{seq}\n")

        cmd = [
            "mmseqs", "easy-cluster",
            str(in_fa), str(out_prefix), str(tmp_dir),
            "--min-seq-id", str(seq_id),
            "-c", "0.8",
            "--cov-mode", "0",
        ]
        subprocess.run(cmd, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        cluster_tsv = Path(f"{out_prefix}_cluster.tsv")
        if not cluster_tsv.exists():
            return {uid: uid for uid in seqs}

        member_to_rep: dict[str, str] = {}
        with cluster_tsv.open() as fh:
            for line in fh:
                rep, member = line.strip().split("\t")
                member_to_rep[member] = rep
        for uid in seqs:
            member_to_rep.setdefault(uid, uid)

    n_clusters = len(set(member_to_rep.values()))
    print(f"→ {n_clusters} clusters from {len(seqs)} sequences")
    return member_to_rep


# ---------------------------------------------------------------------------
# Hard negative sampling
# ---------------------------------------------------------------------------

def _build_length_bins(
    host_seq_pool: dict[str, str],
    pos_host_lens: list[int],
    bins: tuple = (100, 300, 600, 1200),
) -> tuple:
    def _bin(length: int) -> int:
        for i, b in enumerate(bins):
            if length <= b:
                return i
        return len(bins)

    host_by_bin: dict[int, list[str]] = defaultdict(list)
    for uid in host_seq_pool:
        host_by_bin[_bin(len(host_seq_pool[uid]))].append(uid)
    return _bin, host_by_bin


def sample_random_negatives(
    positives: pd.DataFrame,
    pathogen_seqs: dict[str, str],
    host_seq_pool: dict[str, str],
    neg_ratio: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """Random cross-species pairs, length-matched to positives.

    Bait negatives are excluded: pairing a known-interacting pathogen protein
    with untested host proteins likely introduces false negatives.
    """
    rng           = random.Random(seed)
    all_pos_set   = set(zip(positives["protein_a"], positives["protein_b"]))
    all_host_ids  = list(host_seq_pool.keys())
    pathogen_ids  = list(pathogen_seqs.keys())
    n_neg_needed  = len(positives) * neg_ratio
    pos_host_lens = [len(host_seq_pool.get(p, "")) for p in positives["protein_b"]]

    _bin, host_by_bin = _build_length_bins(host_seq_pool, pos_host_lens)

    negatives: list[tuple[str, str]] = []
    neg_set: set[tuple[str, str]]    = set()
    for _ in range(n_neg_needed * 10):
        if len(negatives) >= n_neg_needed:
            break
        path_prot  = rng.choice(pathogen_ids)
        target_bin = _bin(pos_host_lens[rng.randint(0, len(pos_host_lens) - 1)])
        candidates = host_by_bin.get(target_bin, all_host_ids)
        if not candidates:
            continue
        host_prot = rng.choice(candidates)
        pair = (path_prot, host_prot)
        if pair in all_pos_set or pair in neg_set:
            continue
        neg_set.add(pair)
        negatives.append(pair)

    df = pd.DataFrame(negatives, columns=["protein_a", "protein_b"])
    df["neg_type"] = "random"
    df["label"]    = 0
    return df


def sample_hard_negatives(
    positives: pd.DataFrame,
    pathogen_seqs: dict[str, str],
    host_seq_pool: dict[str, str],
    neg_ratio: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """Half bait negatives (known-HPI pathogen protein + random host) +
    half random cross-species pairs, both length-matched.

    Bait negatives are harder for the classifier but may include untested
    true positives — use this variant to measure sensitivity to hard negatives.
    """
    rng = random.Random(seed)

    known_targets: dict[str, set[str]] = defaultdict(set)
    for _, row in positives.iterrows():
        known_targets[row["protein_a"]].add(row["protein_b"])

    all_pos_set            = set(zip(positives["protein_a"], positives["protein_b"]))
    all_host_ids           = list(host_seq_pool.keys())
    pathogen_ids_all       = list(pathogen_seqs.keys())
    pathogen_ids_with_hpi  = list(known_targets.keys())
    n_neg_needed           = len(positives) * neg_ratio
    pos_host_lens          = [len(host_seq_pool.get(p, "")) for p in positives["protein_b"]]

    _bin, host_by_bin = _build_length_bins(host_seq_pool, pos_host_lens)

    negatives: list[tuple[str, str, str]] = []
    neg_set: set[tuple[str, str]]         = set()

    # Phase 1: bait negatives (up to half the budget)
    n_bait = min(n_neg_needed // 2, len(pathogen_ids_with_hpi) * 50)
    for _ in range(n_bait * 3):
        if len(negatives) >= n_bait:
            break
        path_prot  = rng.choice(pathogen_ids_with_hpi)
        target_bin = _bin(pos_host_lens[rng.randint(0, len(pos_host_lens) - 1)])
        candidates = host_by_bin.get(target_bin, all_host_ids)
        if not candidates:
            continue
        host_prot = rng.choice(candidates)
        if host_prot in known_targets[path_prot]:
            continue
        pair = (path_prot, host_prot)
        if pair in all_pos_set or pair in neg_set:
            continue
        neg_set.add(pair)
        negatives.append((path_prot, host_prot, "bait"))

    # Phase 2: random cross-species to fill remainder
    for _ in range((n_neg_needed - len(negatives)) * 5):
        if len(negatives) >= n_neg_needed:
            break
        path_prot  = rng.choice(pathogen_ids_all)
        target_bin = _bin(pos_host_lens[rng.randint(0, len(pos_host_lens) - 1)])
        candidates = host_by_bin.get(target_bin, all_host_ids)
        if not candidates:
            continue
        host_prot = rng.choice(candidates)
        pair = (path_prot, host_prot)
        if pair in all_pos_set or pair in neg_set:
            continue
        neg_set.add(pair)
        negatives.append((path_prot, host_prot, "random"))

    df = pd.DataFrame(negatives, columns=["protein_a", "protein_b", "neg_type"])
    df["label"] = 0
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _validate_default_fold_specs() -> None:
    all_pathogens = VIRUS_PATHOGENS | BACTERIA_PATHOGENS
    unique_blocks = {
        frozenset(pathogens)
        for split_spec in DEFAULT_FOLD_SPECS.values()
        for pathogens in split_spec.values()
    }
    if len(unique_blocks) != 3:
        raise ValueError(
            "Default fold specs must be permutations of exactly three pathogen blocks"
        )

    for fold_name, split_spec in DEFAULT_FOLD_SPECS.items():
        role_sets = {role: set(pathogens) for role, pathogens in split_spec.items()}
        if {frozenset(pathogens) for pathogens in role_sets.values()} != unique_blocks:
            raise ValueError(
                f"Fold {fold_name} must be a permutation of the shared 3-block split design"
            )
        for first_role, second_role in (("train", "val"), ("train", "test"), ("val", "test")):
            overlap = role_sets[first_role] & role_sets[second_role]
            if overlap:
                raise ValueError(
                    f"Fold {fold_name} reuses pathogens across {first_role}/{second_role}: {sorted(overlap)}"
                )
        union = role_sets["train"] | role_sets["val"] | role_sets["test"]
        if union != all_pathogens:
            missing = sorted(all_pathogens - union)
            extra = sorted(union - all_pathogens)
            raise ValueError(
                f"Fold {fold_name} does not cover the 10-species benchmark exactly "
                f"(missing={missing}, extra={extra})"
            )
        for role in ("val", "test"):
            members = role_sets[role]
            if not (members & VIRUS_PATHOGENS) or not (members & BACTERIA_PATHOGENS):
                raise ValueError(
                    f"Fold {fold_name} {role} split must include both virus and bacteria species"
                )


def _collect_split(
    pathogens: list[str],
    pathogen_seqs_all: dict[str, dict[str, str]],
    hpi_pairs_all: dict[str, pd.DataFrame],
) -> tuple[dict[str, str], pd.DataFrame]:
    all_path_seqs: dict[str, str] = {}
    all_pairs_list: list[pd.DataFrame] = []
    for pathogen in pathogens:
        all_path_seqs.update(pathogen_seqs_all[pathogen])
        df = hpi_pairs_all[pathogen].copy()
        df["pathogen"] = pathogen
        all_pairs_list.append(df)
    if all_pairs_list:
        pairs_df = pd.concat(all_pairs_list, ignore_index=True)
    else:
        pairs_df = pd.DataFrame(columns=["protein_a", "protein_b", "pathogen"])
    return all_path_seqs, pairs_df


def _build_split_dataset(
    *,
    split_name: str,
    split_spec: dict[str, list[str]],
    out_dir: Path,
    pathogen_seqs_all: dict[str, dict[str, str]],
    hpi_pairs_all: dict[str, pd.DataFrame],
    host_pool: dict[str, str],
    args: argparse.Namespace,
) -> dict[str, object]:
    train_pathogens = split_spec["train"]
    val_pathogens = split_spec["val"]
    test_pathogens = split_spec["test"]
    _sample_negatives = sample_hard_negatives if args.hard_negatives else sample_random_negatives

    print("\n" + "=" * 65)
    print(f"Building {split_name}")
    print("=" * 65)

    print("\nStep 2 — Pathogen-level split (virus/bacteria holdout)")
    train_path_seqs, train_pos = _collect_split(train_pathogens, pathogen_seqs_all, hpi_pairs_all)
    val_path_seqs, val_pos = _collect_split(val_pathogens, pathogen_seqs_all, hpi_pairs_all)
    test_path_seqs, test_pos = _collect_split(test_pathogens, pathogen_seqs_all, hpi_pairs_all)

    print(f"  Train pathogens: {train_pathogens}")
    print(f"  Train positives: {len(train_pos):,}  pathogen proteins: {len(train_path_seqs):,}")
    print(f"  Val   pathogens: {val_pathogens}")
    print(f"  Val   positives: {len(val_pos):,}  pathogen proteins: {len(val_path_seqs):,}")
    print(f"  Test  pathogens: {test_pathogens}")
    print(f"  Test  positives: {len(test_pos):,}  pathogen proteins: {len(test_path_seqs):,}")

    if not args.no_seq_filter:
        print("\nStep 3 — Sequence-level filter "
              f"(MMseqs2 @ {args.seq_id*100:.0f}%)")
        print("  Running cross-split contamination check …", end=" ")
        combined_seqs = {**train_path_seqs, **val_path_seqs, **test_path_seqs}
        combined_clusters = cluster_seqs(combined_seqs, args.seq_id,
                                         f"{split_name} pathogen seqs (combined)")

        cluster_members: dict[str, list[str]] = defaultdict(list)
        for uid, rep in combined_clusters.items():
            cluster_members[rep].append(uid)

        train_ids = set(train_path_seqs.keys())
        val_ids = set(val_path_seqs.keys())
        test_ids = set(test_path_seqs.keys())

        contaminated_train_reps: set[str] = set()
        contaminated_val_reps: set[str] = set()
        for rep, members in cluster_members.items():
            has_train = any(member in train_ids for member in members)
            has_val = any(member in val_ids for member in members)
            has_test = any(member in test_ids for member in members)
            if has_train and (has_val or has_test):
                contaminated_train_reps.add(rep)
            if has_val and has_test:
                contaminated_val_reps.add(rep)

        contaminated_train_prots = {
            member for member in train_ids
            if combined_clusters.get(member) in contaminated_train_reps
        }
        contaminated_val_prots = {
            member for member in val_ids
            if combined_clusters.get(member) in contaminated_val_reps
        }

        if contaminated_train_prots:
            print(f"\n  Removing {len(contaminated_train_prots)} train proteins "
                  f"from {len(contaminated_train_reps)} contaminated clusters")
            train_pos = train_pos[
                ~train_pos["protein_a"].isin(contaminated_train_prots)
            ].reset_index(drop=True)
            train_path_seqs = {
                uid: seq for uid, seq in train_path_seqs.items()
                if uid not in contaminated_train_prots
            }
        else:
            print("no train contamination found ✓")

        if contaminated_val_prots:
            print(f"  Removing {len(contaminated_val_prots)} val proteins "
                  f"from {len(contaminated_val_reps)} contaminated clusters")
            val_pos = val_pos[
                ~val_pos["protein_a"].isin(contaminated_val_prots)
            ].reset_index(drop=True)
            val_path_seqs = {
                uid: seq for uid, seq in val_path_seqs.items()
                if uid not in contaminated_val_prots
            }
        else:
            print("no val contamination found ✓")

        print(f"  Train positives after seq filter: {len(train_pos):,}")
        print(f"  Val   positives after seq filter: {len(val_pos):,}")
    else:
        print("\nStep 3 — Skipped (--no-seq-filter)")

    print("\nStep 4 — Hard negative sampling "
          f"(ratio {args.neg_ratio}:1)")

    def _sample_neg_per_pathogen(
        split_pathogens: list[str],
        split_pos: pd.DataFrame,
        *,
        seed_base: int,
    ) -> pd.DataFrame:
        parts = []
        for offset, pathogen in enumerate(split_pathogens):
            pos_pathogen = split_pos[split_pos["pathogen"] == pathogen]
            if pos_pathogen.empty:
                continue
            neg_df = _sample_negatives(
                pos_pathogen,
                pathogen_seqs_all[pathogen],
                host_pool,
                neg_ratio=args.neg_ratio,
                seed=seed_base + offset,
            )
            neg_df["pathogen"] = pathogen
            parts.append(neg_df)
        if not parts:
            return pd.DataFrame(
                columns=["protein_a", "protein_b", "neg_type", "label", "pathogen"]
            )
        return pd.concat(parts, ignore_index=True)

    present_train = [p for p in train_pathogens if p in train_pos["pathogen"].values]
    present_val = [p for p in val_pathogens if p in val_pos["pathogen"].values]
    present_test = [p for p in test_pathogens if p in test_pos["pathogen"].values]
    neg_label = "hard (bait+random)" if args.hard_negatives else "random cross-species"

    print("  Sampling train negatives (per pathogen) …")
    train_neg = _sample_neg_per_pathogen(present_train, train_pos, seed_base=args.seed)
    print(f"    → {len(train_neg):,} negatives ({neg_label})")

    print("  Sampling val negatives (per pathogen) …")
    val_neg = _sample_neg_per_pathogen(present_val, val_pos, seed_base=args.seed + 100)
    print(f"    → {len(val_neg):,} negatives ({neg_label})")

    print("  Sampling test negatives (per pathogen) …")
    test_neg = _sample_neg_per_pathogen(present_test, test_pos, seed_base=args.seed + 200)
    print(f"    → {len(test_neg):,} negatives ({neg_label})")

    train_pos["label"] = 1
    train_pos["neg_type"] = "positive"
    val_pos["label"] = 1
    val_pos["neg_type"] = "positive"
    test_pos["label"] = 1
    test_pos["neg_type"] = "positive"

    train_df = pd.concat([train_pos, train_neg], ignore_index=True)
    val_df = pd.concat([val_pos, val_neg], ignore_index=True)
    test_df = pd.concat([test_pos, test_neg], ignore_index=True)

    train_df = train_df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    val_df = val_df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    print("\nStep 5 — Summary")

    def split_summary(df: pd.DataFrame, name: str) -> None:
        n_pos = int((df["label"] == 1).sum())
        n_neg = int((df["label"] == 0).sum())
        rate = n_pos / len(df) if len(df) else 0.0
        print(f"  {name}: {len(df):,} pairs ({n_pos:,} pos / {n_neg:,} neg, rate {rate:.2%})")
        if "pathogen" in df.columns:
            for pathogen, grp in df[df["label"] == 1].groupby("pathogen"):
                print(f"    {pathogen}: {len(grp)} positives")

    split_summary(train_df, "Train")
    split_summary(val_df, "Val  ")
    split_summary(test_df, "Test ")

    def len_stats(seqs: dict[str, str], ids: list[str], label: str) -> None:
        lengths = [len(seqs.get(uid, "")) for uid in ids if uid in seqs]
        if not lengths:
            return
        q25, q50, q75 = np.percentile(lengths, [25, 50, 75])
        print(f"  {label} length: median={q50:.0f}  IQR=[{q25:.0f}, {q75:.0f}]  n={len(lengths)}")

    print()
    len_stats(train_path_seqs,
              train_df[train_df["label"] == 1]["protein_a"].tolist(),
              "Train pathogen protein")
    len_stats(val_path_seqs,
              val_df[val_df["label"] == 1]["protein_a"].tolist(),
              "Val   pathogen protein")
    len_stats(test_path_seqs,
              test_df[test_df["label"] == 1]["protein_a"].tolist(),
              "Test  pathogen protein")
    len_stats(host_pool,
              train_df[train_df["label"] == 1]["protein_b"].tolist(),
              "Train host protein    ")
    len_stats(host_pool,
              val_df[val_df["label"] == 1]["protein_b"].tolist(),
              "Val   host protein    ")
    len_stats(host_pool,
              test_df[test_df["label"] == 1]["protein_b"].tolist(),
              "Test  host protein    ")

    out_dir.mkdir(parents=True, exist_ok=True)
    train_out = out_dir / "train.tsv"
    val_out = out_dir / "val.tsv"
    test_out = out_dir / "test.tsv"
    train_df.to_csv(train_out, sep="\t", index=False)
    val_df.to_csv(val_out, sep="\t", index=False)
    test_df.to_csv(test_out, sep="\t", index=False)

    stats_lines = [
        f"Gold standard HPI benchmark — {split_name}",
        f"Train pathogens: {train_pathogens}",
        f"Val   pathogens: {val_pathogens}",
        f"Test  pathogens: {test_pathogens}",
        f"Sequence identity threshold: {args.seq_id}",
        f"Negative ratio: {args.neg_ratio}:1",
        "",
        f"Train: {len(train_df):,} pairs ({(train_df.label == 1).sum():,} pos / {(train_df.label == 0).sum():,} neg)",
        f"Val:   {len(val_df):,} pairs ({(val_df.label == 1).sum():,} pos / {(val_df.label == 0).sum():,} neg)",
        f"Test:  {len(test_df):,} pairs ({(test_df.label == 1).sum():,} pos / {(test_df.label == 0).sum():,} neg)",
    ]
    (out_dir / "stats.txt").write_text("\n".join(stats_lines))

    print(f"\n  Saved → {train_out}  ({len(train_df):,} rows)")
    print(f"  Saved → {val_out}    ({len(val_df):,} rows)")
    print(f"  Saved → {test_out}   ({len(test_df):,} rows)")
    print(f"  Saved → {out_dir / 'stats.txt'}")

    return {
        "name": split_name,
        "out_dir": out_dir,
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
    }

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--train-pathogens", nargs="+", default=None,
                        help="Pathogens to include in train split")
    parser.add_argument("--val-pathogens",   nargs="+", default=None,
                        help="Pathogens to include in validation split")
    parser.add_argument("--test-pathogens",  nargs="+", default=None,
                        help="Pathogens to include in test split")
    parser.add_argument("--seq-id",     type=float, default=0.40,
                        help="Max sequence identity between train/test pathogen proteins (default: 0.40)")
    parser.add_argument("--neg-ratio",  type=int,   default=10,
                        help="Number of negative pairs per positive (default: 10)")
    parser.add_argument("--hard-negatives", action="store_true",
                        help="Use hard (bait) negatives: half bait + half random. "
                             "Default: purely random cross-species negatives.")
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--out-dir",    type=Path,  default=OUT_DIR)
    parser.add_argument("--leave-one-out", action="store_true",
                        help="Build true leave-one-pathogen-out folds with empty validation splits")
    parser.add_argument("--no-seq-filter", action="store_true",
                        help="Skip MMseqs2 sequence identity filtering step")
    args = parser.parse_args()
    print(f"  Negative strategy: {'hard (bait + random)' if args.hard_negatives else 'random cross-species'}")

    using_single_split = any(
        split is not None for split in (args.train_pathogens, args.val_pathogens, args.test_pathogens)
    )
    if using_single_split and not all(
        split is not None for split in (args.train_pathogens, args.val_pathogens, args.test_pathogens)
    ):
        parser.error("Custom single-split mode requires --train-pathogens, --val-pathogens, and --test-pathogens")
    if using_single_split and args.leave_one_out:
        parser.error("--leave-one-out cannot be combined with custom train/val/test pathogen lists")

    if args.leave_one_out and args.out_dir == OUT_DIR:
        args.out_dir = OUT_DIR_LOO

    if using_single_split:
        fold_jobs = [("Custom split", {
            "train": args.train_pathogens,
            "val": args.val_pathogens,
            "test": args.test_pathogens,
        }, args.out_dir)]
    elif args.leave_one_out:
        all_pathogens_sorted = sorted(VIRUS_PATHOGENS | BACTERIA_PATHOGENS)
        fold_jobs = [
            (
                f"Leave-one-out {pathogen}",
                {
                    "train": [p for p in all_pathogens_sorted if p != pathogen],
                    "val": [],
                    "test": [pathogen],
                },
                args.out_dir / f"fold_{pathogen}",
            )
            for pathogen in all_pathogens_sorted
        ]
    else:
        try:
            _validate_default_fold_specs()
        except ValueError as exc:
            sys.exit(f"ERROR: {exc}")
        fold_jobs = [
            (f"Fold {fold_name}", split_spec, args.out_dir / f"fold_{fold_name.lower()}")
            for fold_name, split_spec in DEFAULT_FOLD_SPECS.items()
        ]

    all_pathogens = sorted({
        pathogen
        for _, split_spec, _ in fold_jobs
        for role in ("train", "val", "test")
        for pathogen in split_spec[role]
    })

    # -----------------------------------------------------------------------
    # 1. Load all data
    # -----------------------------------------------------------------------
    print("=" * 65)
    print("Step 1 — Loading benchmark data")
    print("=" * 65)

    pathogen_seqs_all:  dict[str, dict[str, str]] = {}
    host_seqs_all:      dict[str, dict[str, str]] = {}
    hpi_pairs_all:      dict[str, pd.DataFrame]   = {}

    for pathogen in all_pathogens:
        try:
            ps, hs, pairs = load_pathogen_data(pathogen)
        except FileNotFoundError as e:
            sys.exit(f"ERROR: {e}")
        pathogen_seqs_all[pathogen] = ps
        host_seqs_all[pathogen]     = hs
        hpi_pairs_all[pathogen]     = pairs
        print(f"  {pathogen:30s}  {len(pairs):>4} HPI pairs  "
              f"{len(ps):>4} pathogen seqs  {len(hs):>6} host seqs")

    # Shared host pool: union of all host seqs (they're all human)
    host_pool: dict[str, str] = {}
    for hs in host_seqs_all.values():
        host_pool.update(hs)
    print(f"\n  Shared host pool: {len(host_pool):,} unique human proteins")

    split_summaries = []
    for split_name, split_spec, out_dir in fold_jobs:
        role_sets = [(role, set(split_spec[role])) for role in ("train", "val", "test")]
        for (first_role, first_set), (second_role, second_set) in (
            (role_sets[0], role_sets[1]),
            (role_sets[0], role_sets[2]),
            (role_sets[1], role_sets[2]),
        ):
            overlap = first_set & second_set
            if overlap:
                sys.exit(
                    f"ERROR: {split_name} reuses pathogens across {first_role}/{second_role}: {sorted(overlap)}"
                )
        split_summaries.append(
            _build_split_dataset(
                split_name=split_name,
                split_spec=split_spec,
                out_dir=out_dir,
                pathogen_seqs_all=pathogen_seqs_all,
                hpi_pairs_all=hpi_pairs_all,
                host_pool=host_pool,
                args=args,
            )
        )

    if not using_single_split:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        benchmark_label = (
            "Gold standard HPI benchmark — leave-one-pathogen-out setup"
            if args.leave_one_out else
            "Gold standard HPI benchmark — balanced 3-fold setup"
        )
        summary_lines = [
            benchmark_label,
            f"Sequence identity threshold: {args.seq_id}",
            f"Negative ratio: {args.neg_ratio}:1",
            "",
        ]
        for summary in split_summaries:
            summary_lines.extend([
                f"{summary['name']}: {summary['out_dir']}",
                f"  train rows: {summary['train_rows']:,}",
                f"  val rows: {summary['val_rows']:,}",
                f"  test rows: {summary['test_rows']:,}",
                "",
            ])
        (args.out_dir / "stats.txt").write_text("\n".join(summary_lines).rstrip() + "\n")
        print(f"\nSaved fold overview → {args.out_dir / 'stats.txt'}")


if __name__ == "__main__":
    main()
