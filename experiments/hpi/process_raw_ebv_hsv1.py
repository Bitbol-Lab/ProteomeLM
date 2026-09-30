#!/usr/bin/env python3
"""
Process raw EBV and HSV1 interaction data into hpi_pairs.tsv format.

EBV  — Yiu et al. 2023 (Molecular Cell) AP-MS / CompPASS data
    raw/ebv/mmc3.xlsx
    Sheet: "CompPass output unfiltered" (Supplemental Table 2)
    Bait = EBV protein / ORF symbol, Prey = UniProt accession (Human or EBV)
    → keep pairs where Prey species == Human
    → keep stringent hits with ZScore >= 4.0 by default
    → resolve EBV bait symbols to UniProt via UniProt taxonomy search + aliases

HSV1 — Bogdanow et al. 2025 (Nature Communications) XL-MS data
       raw/hsv1/41467_2025_61618_MOESM12_ESM.xlsx
       Sheet: "Figure2a"  – gene-level PPI list (Gene_A, Gene_B, Species_A, Species_B)
       Sheet: "Figure1b"  – detected proteome with Fasta.headers (gene → UniProt)
       → keep pairs where species differ (virus × host)
       → map gene names to UniProt from Fasta.headers

Output (overwrites existing):
    data/benchmarks/processed/ebv/hpi_pairs.tsv
    data/benchmarks/processed/ebv/metadata.tsv
    data/benchmarks/processed/hsv1/hpi_pairs.tsv
    data/benchmarks/processed/hsv1/metadata.tsv

Usage:
    python process_raw_ebv_hsv1.py
    python process_raw_ebv_hsv1.py --dry-run   # print stats only, do not write
    python process_raw_ebv_hsv1.py --only ebv --ebv-z-threshold 4.0
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import pandas as pd
import requests

UNIPROT_API = "https://rest.uniprot.org"

# python-calamine is much faster than openpyxl for large xlsx files
try:
    from python_calamine import CalamineWorkbook as _CalamineWorkbook
    _HAS_CALAMINE = True
except ImportError:
    _HAS_CALAMINE = False


def _read_excel(path: Path, sheet_name: str, header: int = 0) -> pd.DataFrame:
    if _HAS_CALAMINE:
        wb = _CalamineWorkbook.from_path(str(path))
        rows = wb.get_sheet_by_name(sheet_name).to_python(skip_empty_area=False)
        if not rows:
            return pd.DataFrame()
        if header == 0:
            cols = [str(c) for c in rows[0]]
            data = rows[1:]
        else:
            # skip `header` rows, use row at index `header` as column names
            cols = [str(c) for c in rows[header]]
            data = rows[header + 1:]
        return pd.DataFrame(data, columns=cols)
    return pd.read_excel(path, sheet_name=sheet_name, header=header, engine="openpyxl")

SCRIPT_DIR = Path(__file__).resolve().parent
BENCH_DIR  = SCRIPT_DIR / "data" / "benchmarks"
RAW_DIR    = BENCH_DIR  / "raw"
PROC_DIR   = BENCH_DIR  / "processed"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _uniprot_from_fasta_header(header: str) -> list[str]:
    """Extract all sp|ACCESSION| tokens from a semicolon-joined FASTA header."""
    return re.findall(r"sp\|([A-Z0-9]+(?:-\d+)?)\|", str(header))


def _strip_isoform(acc: str) -> str:
    """Return base accession without isoform suffix (e.g. Q8WVV4-1 → Q8WVV4)."""
    return acc.split("-")[0]


def _normalize_gene_token(value: str) -> str:
    return re.sub(r"\s+", "", str(value).strip()).upper()


_EBV_BAIT_ALIASES = {
    "BALF1/0": "BALF1",
    "BBLF2/3": "BBLF2",
    "BGRF1/BDRF1": "BGRF1",
    "BSLF2/BMLF1(SM)": "BMLF1",
    "BPLF11-1000": "BPLF1",
    "BPLF1501-1500": "BPLF1",
    "BPLF11001-2000": "BPLF1",
    "BPLF11501-2500": "BPLF1",
    "BPLF12001-3000": "BPLF1",
    "BPLF13001-3149": "BPLF1",
    "LMP2A": "LMP2",
    "PEBNA3C": "EBNA3C",
}


def _candidate_ebv_bait_names(bait: str) -> list[str]:
    base = _normalize_gene_token(bait)
    alias = _EBV_BAIT_ALIASES.get(base, base)
    vals = [alias, base]
    vals += re.split(r"[/ ]+", alias)
    vals += [alias.split("(")[0].strip()]

    out: list[str] = []
    for val in vals:
        val = _normalize_gene_token(val)
        if val and val not in out:
            out.append(val)
    return out


def _load_ebv_uniprot_map() -> dict[str, str]:
    print("  Fetching EBV UniProt gene map (taxonomy_id:10376) ...")
    r = requests.get(
        f"{UNIPROT_API}/uniprotkb/search",
        params={
            "query": "taxonomy_id:10376",
            "format": "tsv",
            "fields": "accession,gene_names,protein_name",
            "size": 500,
        },
        timeout=120,
    )
    r.raise_for_status()

    gene_map: dict[str, str] = {}
    lines = [ln for ln in r.text.splitlines() if ln.strip()]
    for line in lines[1:]:
        parts = line.split("\t")
        acc = _strip_isoform(parts[0].strip()) if parts else ""
        if not acc:
            continue
        for field in parts[1:3]:
            for token in re.split(r"[\s;/(),-]+", str(field)):
                token = _normalize_gene_token(token)
                if token:
                    gene_map.setdefault(token, acc)
    return gene_map


def _query_ebv_uniprot_accession(term: str) -> str | None:
    r = requests.get(
        f"{UNIPROT_API}/uniprotkb/search",
        params={
            "query": f"(gene_exact:{term}) AND (taxonomy_id:10376)",
            "format": "tsv",
            "fields": "accession",
            "size": 1,
        },
        timeout=30,
    )
    r.raise_for_status()
    lines = [ln for ln in r.text.splitlines() if ln.strip()]
    if len(lines) > 1:
        return _strip_isoform(lines[1].strip())
    return None


def _resolve_ebv_bait_accessions(bait_symbols: list[str]) -> dict[str, str]:
    gene_map = _load_ebv_uniprot_map()
    resolved: dict[str, str] = {}
    for bait in bait_symbols:
        for cand in _candidate_ebv_bait_names(bait):
            acc = gene_map.get(cand)
            if acc is None:
                try:
                    acc = _query_ebv_uniprot_accession(cand)
                except Exception:
                    acc = None
            if acc:
                resolved[bait] = acc
                break
    return resolved


def _fetch_fastas(accessions: list[str], tag: str, batch: int = 50) -> dict[str, str]:
    """Fetch FASTA sequences from UniProt for a list of accessions.

    Returns {accession: fasta_block} where each block's header is
    ``>sp|ACCESSION|TAG`` (e.g. TAG='PATHOGEN' or 'HOST').
    """
    results: dict[str, str] = {}
    for i in range(0, len(accessions), batch):
        chunk = accessions[i : i + batch]
        query = " OR ".join(f"accession:{a}" for a in chunk)
        for attempt in range(5):
            try:
                r = requests.get(
                    f"{UNIPROT_API}/uniprotkb/stream",
                    params={"query": query, "format": "fasta"},
                    timeout=120,
                )
                r.raise_for_status()
                break
            except Exception as exc:
                if attempt < 4:
                    wait = 3 * (attempt + 1)
                    print(f"    [warn] UniProt fetch failed ({exc}), retry in {wait}s")
                    time.sleep(wait)
                else:
                    raise
        # Parse the response into per-accession blocks
        current_acc: str | None = None
        current_lines: list[str] = []
        for line in r.text.splitlines():
            if line.startswith(">"):
                if current_acc:
                    results[current_acc] = "\n".join(current_lines)
                parts = line[1:].split("|")
                current_acc = parts[1] if len(parts) > 1 else line[1:].split()[0]
                current_lines = [f">sp|{current_acc}|{tag}"]
            elif line.strip():
                current_lines.append(line)
        if current_acc:
            results[current_acc] = "\n".join(current_lines)
    return results


# Sibling datasets whose combined_proteome.fasta already contains the full
# reviewed human proteome (~20K proteins).  Used to copy host sequences so
# all datasets share the same host_HUMAN_esm.pt cache.
_HOST_FASTA_SOURCES = [
    RAW_DIR / "hiv1_jager2012" / "combined_proteome.fasta",
    RAW_DIR / "sars_cov2_zhou2022" / "combined_proteome.fasta",
    RAW_DIR / "influenza_a" / "combined_proteome.fasta",
]


def _load_host_blocks_from_source() -> tuple[list[str], list[str]]:
    """Return (lines_to_write, source_path_str) for the full human proteome."""
    for src in _HOST_FASTA_SOURCES:
        if src.exists():
            host_lines: list[str] = []
            in_host = False
            with open(src) as fh:
                for line in fh:
                    line = line.rstrip()
                    if line.startswith(">"):
                        in_host = "HOST" in line
                    if in_host:
                        host_lines.append(line)
            if host_lines:
                return host_lines, str(src)
    return [], ""


def build_fastas(pathogen: str, dry_run: bool = False) -> None:
    """Build pathogen_proteome.fasta and combined_proteome.fasta for a dataset.

    The combined_proteome.fasta always contains the FULL host (human) proteome
    copied from a sibling dataset so that the shared host_HUMAN_esm.pt cache
    is never invalidated by dataset-specific subsets.

    Reads UniProt IDs from processed/hpi_pairs.tsv (protein_a = virus) for
    the pathogen sequences, fetches them from UniProt, and writes:
      raw/{pathogen}/pathogen_proteome.fasta
      raw/{pathogen}/combined_proteome.fasta
    """
    pairs_tsv = PROC_DIR / pathogen / "hpi_pairs.tsv"
    if not pairs_tsv.exists():
        print(f"  [skip] {pairs_tsv} not found — run processing step first.")
        return

    df = pd.read_csv(pairs_tsv, sep="\t")
    path_accs = sorted(df["protein_a"].dropna().unique().tolist())

    host_lines, host_src = _load_host_blocks_from_source()
    if host_lines:
        n_host = sum(1 for l in host_lines if l.startswith(">"))
        print(f"\n  Host proteome: {n_host} proteins copied from {Path(host_src).parent.name}/")
    else:
        print("  [warn] No sibling host FASTA found — will fetch host subset from UniProt.")

    print(f"  Fetching {len(path_accs)} pathogen sequences from UniProt ...")

    if not dry_run:
        path_blocks = _fetch_fastas(path_accs, "PATHOGEN")

        raw_out = RAW_DIR / pathogen
        raw_out.mkdir(parents=True, exist_ok=True)

        path_fasta = raw_out / "pathogen_proteome.fasta"
        with open(path_fasta, "w") as fh:
            for acc in path_accs:
                if acc in path_blocks:
                    fh.write(path_blocks[acc] + "\n")
        print(f"  Wrote {len(path_blocks)}/{len(path_accs)} pathogen seqs → {path_fasta.name}")

        combined_fasta = raw_out / "combined_proteome.fasta"
        with open(combined_fasta, "w") as fh:
            # Pathogen first (PATHOGEN tag)
            for acc in path_accs:
                if acc in path_blocks:
                    fh.write(path_blocks[acc] + "\n")
            # Full host proteome
            if host_lines:
                fh.write("\n".join(host_lines) + "\n")
            else:
                # Fallback: fetch subset
                host_accs = sorted(df["protein_b"].dropna().unique().tolist())
                host_blocks = _fetch_fastas(host_accs, "HOST")
                for acc in host_accs:
                    if acc in host_blocks:
                        fh.write(host_blocks[acc] + "\n")
        n_combined = sum(1 for l in (host_lines or []) if l.startswith(">")) + len(path_blocks)
        print(f"  Wrote ~{n_combined} total seqs → {combined_fasta.name}")
    else:
        print(f"  [dry-run] Would write FASTAs for {pathogen} (full host proteome + {len(path_accs)} pathogen).")


# ---------------------------------------------------------------------------
# EBV processing
# ---------------------------------------------------------------------------

def process_ebv(dry_run: bool = False, z_threshold: float = 4.0) -> None:
    raw = RAW_DIR / "ebv"
    mmc3 = raw / "mmc3.xlsx"

    print("=" * 65)
    print("Processing EBV (Yiu et al. 2023 Molecular Cell AP-MS)")
    print("=" * 65)
    print(f"  Reading: {mmc3.name}")

    df = _read_excel(mmc3, "CompPass output unfiltered", header=0)
    df.columns = [str(c).strip() for c in df.columns]
    df["ZScore"] = pd.to_numeric(df["ZScore"], errors="coerce")

    print(f"  Raw rows: {len(df)}")
    print(f"  Prey species: {df['Prey species'].value_counts().to_dict()}")

    # ----------------------------------------------------------------
    # Extract virus-host pairs (bait=EBV, prey=Human)
    # ----------------------------------------------------------------
    human_rows = df[
        (df["Prey species"] == "Human")
        & (df["ZScore"] >= z_threshold)
    ].copy()
    print(f"\n  Human prey rows at ZScore >= {z_threshold:.1f}: {len(human_rows)}")

    bait_symbols = sorted(human_rows["Bait Symbol"].dropna().astype(str).unique())
    ebv_gene_to_uniprot = _resolve_ebv_bait_accessions(bait_symbols)
    print(f"  Resolved EBV bait symbols: {len(ebv_gene_to_uniprot)} / {len(bait_symbols)}")

    unmapped_baits = [b for b in bait_symbols if b not in ebv_gene_to_uniprot]
    if unmapped_baits:
        print(f"  Unmapped bait symbols ({len(unmapped_baits)}): {unmapped_baits[:10]}")

    pairs: list[tuple[str, str]] = []
    skipped_no_bait_uniprot = 0
    skipped_no_prey_uniprot = 0

    for _, row in human_rows.iterrows():
        bait_gene = str(row["Bait Symbol"]).strip()
        prey_acc  = str(row["Prey Uniprot or Trembl"]).strip()
        prey_acc  = _strip_isoform(prey_acc)

        if not prey_acc or prey_acc.lower() == "nan":
            skipped_no_prey_uniprot += 1
            continue

        bait_acc = ebv_gene_to_uniprot.get(bait_gene)
        if bait_acc is None:
            skipped_no_bait_uniprot += 1
            continue

        pairs.append((bait_acc, prey_acc))

    # Deduplicate
    pairs_df = pd.DataFrame(pairs, columns=["protein_a", "protein_b"]).drop_duplicates()
    print(f"\n  Pairs extracted:         {len(pairs)}")
    print(f"  After deduplication:     {len(pairs_df)}")
    print(f"  Skipped (no bait UP):    {skipped_no_bait_uniprot}")
    print(f"  Skipped (no prey UP):    {skipped_no_prey_uniprot}")
    print(f"\n  Bait UniProts (EBV):     {pairs_df['protein_a'].nunique()}")
    print(f"  Prey UniProts (Human):   {pairs_df['protein_b'].nunique()}")
    print(f"\n  Sample pairs:")
    print(pairs_df.head(5).to_string(index=False))

    if not dry_run:
        out_dir = PROC_DIR / "ebv"
        out_dir.mkdir(parents=True, exist_ok=True)
        pairs_df.to_csv(out_dir / "hpi_pairs.tsv", sep="\t", index=False)

        # Update metadata
        meta = {
            "pathogen":              "ebv",
            "study":                 "Epstein-Barr Virus × Human — Yiu et al. 2023 Molecular Cell AP-MS (Supplemental Table 2)",
            "pmids":                 "",
            "pathogen_taxid":        10376,
            "host_taxid":            9606,
            "n_pathogen_proteins":   int(pairs_df["protein_a"].nunique()),
            "n_host_proteins":       int(pairs_df["protein_b"].nunique()),
            "n_hpi_pairs":           len(pairs_df),
            "notes":                 (
                f"AP-MS CompPass output unfiltered; kept Human prey rows with ZScore >= {z_threshold:.1f}; "
                f"{len(bait_symbols)} EBV bait symbols considered; {skipped_no_bait_uniprot} rows skipped for unmapped bait symbol"
            ),
        }
        pd.Series(meta).to_csv(out_dir / "metadata.tsv", sep="\t")
        print(f"\n  → Saved {len(pairs_df)} pairs to {out_dir}/hpi_pairs.tsv")


# ---------------------------------------------------------------------------
# HSV1 processing
# ---------------------------------------------------------------------------

def process_hsv1(dry_run: bool = False, xl_threshold: int = 2) -> None:
    raw   = RAW_DIR / "hsv1"
    xlsx  = raw / "41467_2025_61618_MOESM12_ESM.xlsx"

    print("\n" + "=" * 65)
    print("Processing HSV1 (Bogdanow et al. 2025 Nature Communications XL-MS)")
    print("=" * 65)
    print(f"  Reading: {xlsx.name}")

    # ----------------------------------------------------------------
    # Build gene → UniProt map from Figure1b Fasta.headers
    # ----------------------------------------------------------------
    df1b = _read_excel(xlsx, "Figure1b", header=0)
    df1b.columns = [str(c).strip() for c in df1b.columns]

    gene_to_uniprot: dict[str, str] = {}
    for _, row in df1b.iterrows():
        headers_raw = str(row.get("Fasta.headers", ""))
        accs = _uniprot_from_fasta_header(headers_raw)
        if not accs:
            continue
        # Extract gene name from first FASTA entry: "sp|ACC|NAME GN=GENE"
        gn_match = re.search(r"GN=(\S+)", headers_raw)
        if gn_match:
            gene_to_uniprot[gn_match.group(1)] = accs[0]
        # Also index by the protein name part (sp|ACC|GENENAME_HUMAN)
        for header_part in headers_raw.split(";"):
            pipe_parts = header_part.strip().split("|")
            if len(pipe_parts) >= 3:
                acc       = pipe_parts[1]
                name_part = pipe_parts[2].split()[0]   # e.g. "GAL3A_HUMAN"
                gene_abbr = name_part.split("_")[0]    # e.g. "GAL3A"
                gene_to_uniprot.setdefault(gene_abbr, acc)

    print(f"\n  Gene→UniProt entries from Figure1b: {len(gene_to_uniprot)}")

    # ----------------------------------------------------------------
    # Extract virus-host pairs from Figure2a
    # ----------------------------------------------------------------
    df2a = _read_excel(xlsx, "Figure2a", header=2)
    df2a.columns = [str(c).strip() for c in df2a.columns]

    print(f"  Figure2a total pairs: {len(df2a)}")
    vh = df2a[df2a["Species_A"] != df2a["Species_B"]].copy()
    print(f"  Virus-host pairs:     {len(vh)}")
    if xl_threshold > 1:
        before = len(vh)
        vh = vh[pd.to_numeric(vh["XL_count"], errors="coerce").fillna(0) >= xl_threshold]
        print(f"  After XL_count >= {xl_threshold}:  {len(vh)}  (dropped {before - len(vh)})")

    pairs: list[tuple[str, str]] = []
    skipped = []

    for _, row in vh.iterrows():
        gene_a, gene_b = str(row["Gene_A"]).strip(), str(row["Gene_B"]).strip()
        sp_a,   sp_b   = str(row["Species_A"]).strip(), str(row["Species_B"]).strip()

        acc_a = gene_to_uniprot.get(gene_a)
        acc_b = gene_to_uniprot.get(gene_b)

        if acc_a is None or acc_b is None:
            skipped.append((gene_a, gene_b))
            continue

        # Convention: virus protein first
        if sp_a == "virus":
            pairs.append((acc_a, acc_b))
        else:
            pairs.append((acc_b, acc_a))

    pairs_df = pd.DataFrame(pairs, columns=["protein_a", "protein_b"]).drop_duplicates()

    print(f"\n  Pairs mapped:           {len(pairs)}")
    print(f"  After deduplication:    {len(pairs_df)}")
    print(f"  Pairs skipped (no UP):  {len(skipped)}")
    if skipped[:5]:
        print(f"  First skipped genes:    {skipped[:5]}")
    print(f"\n  Viral UniProts:   {pairs_df['protein_a'].nunique()}")
    print(f"  Host  UniProts:   {pairs_df['protein_b'].nunique()}")
    print(f"\n  Sample pairs:")
    print(pairs_df.head(5).to_string(index=False))

    if not dry_run:
        out_dir = PROC_DIR / "hsv1"
        out_dir.mkdir(parents=True, exist_ok=True)
        pairs_df.to_csv(out_dir / "hpi_pairs.tsv", sep="\t", index=False)

        meta = {
            "pathogen":              "hsv1",
            "study":                 "Herpes Simplex Virus 1 × Human — Bogdanow et al. 2025 Nature Communications XL-MS",
            "pmids":                 "40404699",
            "pathogen_taxid":        10299,
            "host_taxid":            9606,
            "n_pathogen_proteins":   int(pairs_df["protein_a"].nunique()),
            "n_host_proteins":       int(pairs_df["protein_b"].nunique()),
            "n_hpi_pairs":           len(pairs_df),
            "notes":                 (
                f"In-cell XL-MS; Figure2a gene-level PPIs; XL_count >= {xl_threshold}; "
                f"{len(skipped)} pairs dropped for missing UniProt mapping"
            ),
        }
        pd.Series(meta).to_csv(out_dir / "metadata.tsv", sep="\t")
        print(f"\n  → Saved {len(pairs_df)} pairs to {out_dir}/hpi_pairs.tsv")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print statistics without writing any files")
    parser.add_argument("--only", choices=["ebv", "hsv1"],
                        help="Process only one dataset")
    parser.add_argument("--no-fasta", action="store_true",
                        help="Skip building FASTA files (hpi_pairs.tsv only)")
    parser.add_argument("--ebv-z-threshold", type=float, default=4.0, metavar="Z",
                        help="Minimum EBV CompPASS ZScore on Table S2 unfiltered rows (default: 4.0)")
    parser.add_argument("--hsv1-xl-threshold", type=int, default=2, metavar="N",
                        help="Minimum XL_count for HSV1 Figure2a pairs (default: 2)")
    args = parser.parse_args()

    if args.only != "hsv1":
        process_ebv(dry_run=args.dry_run, z_threshold=args.ebv_z_threshold)
        if not args.no_fasta:
            build_fastas("ebv", dry_run=args.dry_run)

    if args.only != "ebv":
        process_hsv1(dry_run=args.dry_run, xl_threshold=args.hsv1_xl_threshold)
        if not args.no_fasta:
            build_fastas("hsv1", dry_run=args.dry_run)

    if args.dry_run:
        print("\n[dry-run] No files written.")
    else:
        print("\nDone. Re-run build_gold_benchmark.py to rebuild the gold splits.")


if __name__ == "__main__":
    main()
