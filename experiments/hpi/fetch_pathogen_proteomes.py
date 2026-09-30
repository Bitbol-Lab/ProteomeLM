#!/usr/bin/env python3
"""
fetch_pathogen_proteomes.py
===========================
Download the complete pathogen proteome for each benchmark from UniProt and
rebuild combined_proteome.fasta = full host proteome + full pathogen proteome.

Skips datasets whose pathogen_proteome.fasta already exists (use --force to
redo). EBV and HSV-1 are not handled here: their combined_proteome.fasta
(pathogen-first, interacting pathogen proteins only) is written by
process_raw_ebv_hsv1.py::build_fastas, and that is the version the paper used.

Usage:
    cd experiments/hpi
    python fetch_pathogen_proteomes.py [--datasets ds1 ds2 ...] [--force]

Requirements: requests
"""

import argparse
import time
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SCRIPT_DIR    = Path(__file__).resolve().parent
RAW_DIR       = SCRIPT_DIR / "data" / "benchmarks" / "raw"
PROCESSED_DIR = SCRIPT_DIR / "data" / "benchmarks" / "processed"

UNIPROT_API = "https://rest.uniprot.org"

# Host proteome source (reuse from any dataset that has 20k+ host proteins)
HOST_SOURCES = [
    RAW_DIR / "hiv1_jager2012" / "combined_proteome.fasta",
    RAW_DIR / "yersinia"       / "combined_proteome.fasta",
    RAW_DIR / "ebv"            / "combined_proteome.fasta",
]

# UniProt reference proteome IDs for each dataset (lstmphv_all excluded).
# Using proteome:UPID ensures the canonical, non-redundant reference set.
# ebv / hsv1 are deliberately absent: see the module docstring.
DATASET_PROTEOMES: dict[str, str] = {
    "sars_cov2_zhou2022": "UP000464024",  # SARS-CoV-2 Wuhan-Hu-1 (Zhou 2022 dataset)
    "hiv1_jager2012":     "UP000002241",  # HIV-1 HXB2
    "influenza_a":        "UP000009255",  # Influenza A H1N1 PR/8/34 (13 proteins)
    "hpv":                "UP000009251",  # HPV-16 (9 proteins)
    "sars_cov1":          "UP000000354",  # SARS-CoV-1 Tor2
    "yersinia":           "UP000000815",  # Yersinia pestis CO92
    "salmonella":         "UP000001014",  # Salmonella Typhimurium LT2
    "candida_albicans":   "UP000000559",  # C. albicans SC5314
    "denv2_shah2018":     "UP000002500",  # DENV-2 strain 16681
    "helicobacter_pylori":"UP000000429",  # H. pylori 26695
    "chlamydia":          "UP000000431",  # C. trachomatis D/UW-3/CX (taxid 272561)
    "tuberculosis":       "UP000001584",  # M. tuberculosis H37Rv
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_host_block(host_sources: list[Path]) -> dict[str, str]:
    """Load full host proteome as {acc: fasta_block} from first available source."""
    src = next((p for p in host_sources if p.exists()), None)
    if src is None:
        raise FileNotFoundError(
            "No host proteome source found. Download the benchmark data first (see README.md)."
        )
    print(f"  Host proteome source: {src}")
    blocks: dict[str, str] = {}
    current_acc: str | None = None
    current_lines: list[str] = []

    def _flush():
        if current_acc:
            blocks[current_acc] = "\n".join(current_lines)

    with open(src) as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                _flush()
                parts = line[1:].split("|")
                current_acc = parts[1] if len(parts) > 1 else line[1:]
                # Ensure header is >sp|ACC|HOST
                header = line if "HOST" in line else f">sp|{current_acc}|HOST"
                current_lines = [header]
            else:
                current_lines.append(line)
    _flush()
    # Only keep HOST entries
    return {acc: block for acc, block in blocks.items() if "HOST" in block.splitlines()[0]}


def download_uniprot_fasta(proteome_id: str, retries: int = 3) -> str:
    """Download all proteins for a UniProt reference proteome as raw FASTA."""
    url = f"{UNIPROT_API}/uniprotkb/stream"
    params = {
        "query":  f"proteome:{proteome_id}",
        "format": "fasta",
    }
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=120, stream=True)
            r.raise_for_status()
            return r.text
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(3 * (attempt + 1))
            else:
                raise RuntimeError(f"Download failed for proteome {proteome_id}: {e}") from e


def fasta_to_blocks(raw_fasta: str, tag: str) -> dict[str, str]:
    """Parse raw UniProt FASTA into {acc: '>sp|ACC|TAG\\nSEQ...'} dict."""
    blocks: dict[str, str] = {}
    current_acc: str | None = None
    current_lines: list[str] = []

    def _flush():
        if current_acc and current_lines:
            blocks[current_acc] = "\n".join(current_lines)

    for line in raw_fasta.splitlines():
        if line.startswith(">"):
            _flush()
            # UniProt header: >sp|ACC|NAME_ORGANISM description...
            parts = line[1:].split("|")
            current_acc = parts[1] if len(parts) > 1 else line[1:].split()[0]
            current_lines = [f">sp|{current_acc}|{tag}"]
        else:
            current_lines.append(line)
    _flush()
    return blocks


def write_combined(fasta_out: Path, host_blocks: dict[str, str],
                   pathogen_blocks: dict[str, str]) -> tuple[int, int]:
    """Write host + pathogen entries, return (n_host, n_pathogen)."""
    fasta_out.parent.mkdir(parents=True, exist_ok=True)
    with open(fasta_out, "w") as fh:
        for acc in sorted(host_blocks):
            fh.write(host_blocks[acc] + "\n")
        for acc in sorted(pathogen_blocks):
            fh.write(pathogen_blocks[acc] + "\n")
    return len(host_blocks), len(pathogen_blocks)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download full pathogen proteomes and rebuild combined_proteome.fasta",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--datasets", nargs="+", default=list(DATASET_PROTEOMES),
        help="Subset of dataset keys to process",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-download even if pathogen_proteome.fasta already exists",
    )
    args = parser.parse_args()

    print("Loading host proteome …")
    host_blocks = load_host_block(HOST_SOURCES)
    print(f"  {len(host_blocks)} host proteins\n")

    for ds in args.datasets:
        if ds not in DATASET_PROTEOMES:
            print(f"[skip] {ds}: not in DATASET_PROTEOMES")
            continue

        proteome_id = DATASET_PROTEOMES[ds]
        raw_ds_dir   = RAW_DIR / ds
        fasta_out    = raw_ds_dir / "combined_proteome.fasta"
        pathogen_out = raw_ds_dir / "pathogen_proteome.fasta"

        if not raw_ds_dir.exists():
            print(f"[skip] {ds}: raw dir missing ({raw_ds_dir})")
            continue

        print(f"{'='*60}")
        print(f"Dataset: {ds}  (proteome {proteome_id})")

        # Check if already complete
        if pathogen_out.exists() and not args.force:
            n = sum(1 for l in open(pathogen_out) if l.startswith(">"))
            print(f"  pathogen_proteome.fasta exists ({n} proteins) — skipping (use --force to redo)")
            continue

        print(f"  Downloading proteome from UniProt …")
        raw_fasta = download_uniprot_fasta(proteome_id)
        pathogen_blocks = fasta_to_blocks(raw_fasta, "PATHOGEN")
        print(f"  Downloaded {len(pathogen_blocks)} pathogen proteins")

        # Save separate pathogen_proteome.fasta
        with open(pathogen_out, "w") as fh:
            for acc in sorted(pathogen_blocks):
                fh.write(pathogen_blocks[acc] + "\n")
        print(f"  Saved → {pathogen_out}")

        # Rebuild combined
        n_host, n_path = write_combined(fasta_out, host_blocks, pathogen_blocks)
        print(f"  Rebuilt combined_proteome.fasta: {n_host} host + {n_path} pathogen = {n_host+n_path} total")
        print(f"  Saved → {fasta_out}\n")
        time.sleep(0.5)  # be polite to the API


if __name__ == "__main__":
    main()
