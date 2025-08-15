#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import logging
import multiprocessing as mp
import pathlib
import subprocess
import sys
import time
from datetime import timedelta
from typing import Iterable, Tuple

###############################################################################
# Logging helpers                                                             #
###############################################################################

def setup_logging(level: str = "INFO", logfile: pathlib.Path | None = None) -> None:
    log_fmt = "%(asctime)s [%(levelname)s] %(message)s"
    date_fmt = "%Y-%m-%d %H:%M:%S"
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if logfile is not None:
        handlers.append(logging.FileHandler(logfile, mode="w"))

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=log_fmt,
        datefmt=date_fmt,
        handlers=handlers,
        force=True,
    )

logger = logging.getLogger(__name__)

###############################################################################
# Database preparation                                                        #
###############################################################################

def tsv_to_fasta(input_file: str, output_file: str) -> None:
    """
    Converts a TSV file containing protein sequences into a FASTA file,
    using the OrthoDB ID as the header.

    Parameters:
        input_file (str): Path to the input TSV file.
        output_file (str): Path to the output FASTA file.
    """
    with open(input_file, "r") as tsv_in, open(output_file, "w") as fasta_out:
        reader = csv.DictReader(tsv_in, delimiter='\t')
        for row in reader:
            entry = row['Entry'].strip()
            orthodb = row['OrthoDB'].strip(';')  # remove trailing semicolon
            if len(orthodb) < 3:
                continue
            sequence = row['Sequence']
            fasta_out.write(f">{entry}\t{orthodb}\n{sequence}\n")

def build_diamond_database(fasta: pathlib.Path, db_prefix: pathlib.Path) -> None:
    """Index *fasta* into *db_prefix.dmnd* (runs once)."""
    logger.info("Building DIAMOND database %s.dmnd", db_prefix)
    subprocess.check_call(["diamond", "makedb", "--in", str(fasta), "-d", str(db_prefix)])

def ensure_diamond_db(database_fasta_file: pathlib.Path) -> pathlib.Path:
    """Return *.dmnd path, building it if missing."""
    db_prefix = database_fasta_file.with_suffix("")
    dmnd = db_prefix.with_suffix(".dmnd")
    if not dmnd.exists():
        build_diamond_database(database_fasta_file, db_prefix)
    else:
        logger.info("Using existing DIAMOND DB %s", dmnd.name)
    return dmnd

###############################################################################
# Worker: annotate a single proteome                                          #
###############################################################################


def annotate_one_proteome(args: Tuple[pathlib.Path, pathlib.Path, pathlib.Path, pathlib.Path, int]) -> pathlib.Path:
    (
        proteome_fasta,
        diamond_db,
        map_table,
        out_dir,
        threads,
    ) = args

    child_logger = logging.getLogger(__name__)
    out_tsv = out_dir / f"{proteome_fasta.stem}_og.tsv"
    if out_tsv.exists():
        child_logger.debug("Skip %s (already done)", proteome_fasta.name)
        return out_tsv

    child_logger.info("Annotating %s", proteome_fasta.name)

    # launch DIAMOND
    cmd = [
        "diamond", "blastp",
        "-d", str(diamond_db),
        "-q", str(proteome_fasta),
        "--very-sensitive", "-e", "1e-5", "-k", "1",
        "-f", "6", "qseqid", "sseqid",
        "--threads", str(threads),
    ]
    child_logger.debug("$ %s", " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)

    # load gene→OG map
    gene2og: dict[str, str] = {}
    opener = gzip.open if map_table.suffix == ".gz" else open
    with opener(map_table, "rt") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue  # skip empty lines and comments
            fields = line.split("\t")
            fields[0] = fields[0][1:]
            if len(fields) < 2:
                continue
            gene_id, og = fields[:2]
            gene2og[gene_id] = og

    processed = 0
    with open(out_tsv, "w", newline="") as fout:
        w = csv.writer(fout, delimiter="\t")
        for ln in proc.stdout:
            q, s = ln.rstrip().split("\t")
            w.writerow([gene2og.get(s, "NA"), q])
            processed += 1
            if processed % 10_000 == 0:
                child_logger.debug("%s: %d sequences", proteome_fasta.name, processed)

    proc.stdout.close()
    proc.wait()
    child_logger.info("Done %s (%d sequences)", proteome_fasta.name, processed)
    return out_tsv

###############################################################################
#                             CLI & driver                                    #
###############################################################################


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Annotate proteome FASTA files with OrthoDB OG identifiers (with logging)."
    )
    parser.add_argument("--proteome-dir", default=pathlib.Path("/data2/malbrank/proteomlm/human/human_genomes"), type=pathlib.Path, help="Directory with *.faa|*.fa|*.fasta files")
    parser.add_argument("--database-fasta-file", default=pathlib.Path("/data2/malbrank/proteomlm/human/uniprotkb_taxonomy_id_9606_2025_06_06.fasta"), type=pathlib.Path, help="OrthoDB FASTA file")
    parser.add_argument("--database-tsv-file", default=pathlib.Path("/data2/malbrank/proteomlm/human/uniprotkb_taxonomy_id_9606_2025_06_06.tsv"), type=pathlib.Path, help="OrthoDB TSV file")
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("./orthodb_og"), help="Output tables directory")
    parser.add_argument("--threads-per-search", type=int, default=8, help="CPUs per DIAMOND invocation")
    parser.add_argument("--parallel-proteomes", type=int, default=32, help="Concurrent proteomes (processes)")
    parser.add_argument("--log-level", default="INFO", help="DEBUG | INFO | WARNING | ERROR | CRITICAL")
    parser.add_argument("--log-file", type=pathlib.Path, help="Optional path to log file (overwrites)")
    return parser.parse_args(argv)


def fast_orthodb_matching(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    setup_logging(level=args.log_level, logfile=args.log_file)
    start_time = time.time()

    # Step 1. Ensure FASTA exists (convert from TSV if needed)
    if not args.database_fasta_file.exists():
        if not args.database_tsv_file.exists():
            logger.error("Neither FASTA nor TSV exists at %s", args.database_tsv_file)
            sys.exit(1)
        logger.info("Converting TSV to FASTA: %s -> %s", args.database_tsv_file, args.database_fasta_file)
        tsv_to_fasta(str(args.database_tsv_file), str(args.database_fasta_file))
    else:
        logger.info("Using existing FASTA file: %s", args.database_fasta_file)

    # Step 2. Ensure DIAMOND DB
    diamond_db_path = ensure_diamond_db(args.database_fasta_file)

    # Step 3. Ensure output directory exists
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Step 4. Collect FASTA files to annotate
    fasta_files = sorted(
        [f for f in args.proteome_dir.glob("*") if f.suffix.lower() in {".faa", ".fa", ".fasta"}]
    )
    if not fasta_files:
        logger.error("No FASTA files found in %s", args.proteome_dir)
        sys.exit(1)

    logger.info("Queueing %d proteomes for annotation", len(fasta_files))

    # Step 5. Build arguments for annotation workers
    worker_args = [
        (
            fasta_file,
            diamond_db_path,
            args.database_fasta_file,  # using FASTA file as mapping table
            args.output_dir,
            args.threads_per_search, 
        )
        for fasta_file in fasta_files
    ]

    # Step 6. Launch annotation workers
    completed = 0
    with mp.get_context("spawn").Pool(processes=args.parallel_proteomes) as pool:
        for completed_path in pool.imap_unordered(annotate_one_proteome, worker_args):
            completed += 1
            logger.info("[%d/%d] Completed %s", completed, len(fasta_files), completed_path.name)

    # Step 7. Merge all TSV outputs
    merged_tsv = args.output_dir / "ALL_proteomes_OG.tsv"
    logger.info("Merging %d TSV files into %s", len(fasta_files), merged_tsv.name)
    with open(merged_tsv, "w", newline="") as merged_file:
        first_header_written = False
        for individual_tsv in sorted(args.output_dir.glob("*_og.tsv")):
            with open(individual_tsv) as source:
                for line in source:
                    if line.startswith("#") and first_header_written:
                        continue
                    merged_file.write(line)
            first_header_written = True

    elapsed = timedelta(seconds=round(time.time() - start_time))
    logger.info("All done in %s. Merged annotations: %s", elapsed, merged_tsv)


if __name__ == "__main__":
    fast_orthodb_matching()
