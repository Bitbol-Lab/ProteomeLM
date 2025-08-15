import csv
import logging
import json
import os
import requests
import subprocess
import gzip
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys

sys.path.append("..")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set plot style
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['text.usetex'] = False
plt.rcParams['font.size'] = 14


def load_fasta(fasta_file: str) -> Tuple[List[str], List[str]]:
    """
    Load sequences from a FASTA file.

    Returns:
        Tuple of (labels, sequences)
    """
    labels = []
    sequences = []

    with open(fasta_file, 'r') as f:
        current_seq = ""
        current_label = None

        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_label is not None:
                    labels.append(current_label)
                    sequences.append(current_seq)
                current_label = line[1:]  # Remove '>'
                current_seq = ""
            else:
                current_seq += line

        # Add the last sequence
        if current_label is not None:
            labels.append(current_label)
            sequences.append(current_seq)

    return labels, sequences


def download_string_fasta(organism: str) -> Tuple[List[str], List[str]]:
    """
    Download the FASTA file for a given organism from the STRING database.
    """
    url = f"https://stringdb-downloads.org/download/protein.sequences.v12.0/{organism}.protein.sequences.v12.0.fa.gz"
    response = requests.get(url)
    if response.status_code == 200:
        with open("temp_string.fa.gz", "wb") as f:
            f.write(response.content)

        # Extract to fa
        with gzip.open("temp_string.fa.gz", "rt") as f:
            fasta_content = f.read()
        with open("temp_string.fa", "w") as f:
            f.write(fasta_content)
        os.remove("temp_string.fa.gz")

        labels, sequences = load_fasta("temp_string.fa")
        return labels, sequences
    else:
        raise Exception(f"Failed to download FASTA file for {organism}. Status code: {response.status_code}. Are you sure this is a valid STRING organism?")


def download_string_annotations(organism: str) -> str:
    """
    Download the annotations for a given organism from the STRING database.
    Returns the filename of the downloaded annotations.
    """
    url = f"https://stringdb-downloads.org/download/protein.links.detailed.v12.0/{organism}.protein.links.detailed.v12.0.txt.gz"
    response = requests.get(url)
    if response.status_code == 200:
        with open("temp_string_annotations.txt.gz", "wb") as f:
            f.write(response.content)

        # Extract to txt
        with gzip.open("temp_string_annotations.txt.gz", "rt") as f:
            annotations_content = f.read()
        with open("temp_string_annotations.txt", "w") as f:
            f.write(annotations_content)
        os.remove("temp_string_annotations.txt.gz")

        return "temp_string_annotations.txt"
    else:
        raise Exception(f"Failed to download annotations for {organism}. Status code: {response.status_code}. Are you sure this is a valid STRING organism?")


def fetch_uniprot_sequences_batch(
    ids: List[str],
    batch_size: int = 100
) -> Dict[str, str]:
    """
    Fetches UniProtKB sequences in FASTA format given a list of accessions.
    Returns a dictionary mapping UniProt accession to sequence.
    """
    url = "https://rest.uniprot.org/uniprotkb/stream"
    headers = {"Accept": "application/json"}
    sequences = {}
    ids = list(ids)

    for i in tqdm(range(0, len(ids), batch_size)):
        batch_ids = ids[i:i+batch_size]
        query = " OR ".join(f"accession:{pid}" for pid in batch_ids)
        params = {
            "query": query,
            "format": "json"
        }

        response = requests.get(url, params=params, headers=headers)

        if response.status_code == 200:
            data = response.json()
            for entry in data.get("results", []):
                accession = entry.get("primaryAccession")
                sequence = entry.get("sequence", {}).get("value", "")
                if accession and sequence:
                    sequences[accession] = sequence
        else:
            print(f"Failed batch {i//batch_size + 1}: {response.status_code} – {response.text}")

    return sequences


def map_string_to_uniprot(
    fasta_uniprot: str,
    fasta_string: str,
    db_name: str = "uniprot_db",
    blast_output: str = "string_vs_uniprot.tsv",
    identity_thresh: float = 95.0,
    cov_thresh: float = 90.0,
    threads: int = 4,
    json_output: Optional[str] = None
) -> Dict[str, str]:
    """
    Maps STRING protein IDs to UniProt accessions using BLASTP alignment.
    Returns a dictionary mapping STRING IDs to UniProt IDs.
    """
    # 1. Create BLAST database
    subprocess.run([
        "makeblastdb", "-in", fasta_string, "-dbtype", "prot", "-out", db_name
    ], check=True)

    # 2. Run BLASTP
    subprocess.run([
        "blastp",
        "-query", fasta_uniprot,
        "-db", db_name,
        "-outfmt", "6 qseqid sseqid pident qcovs evalue bitscore",
        "-qcov_hsp_perc", str(cov_thresh),
        "-num_threads", str(threads),
        "-out", blast_output
    ], check=True)

    # 3. Parse BLAST output
    best_hits: Dict[str, Dict[str, float]] = {}
    with open(blast_output) as f:
        reader = csv.DictReader(f, fieldnames=[
            "qseqid", "sseqid", "pident", "qcovs", "evalue", "bitscore"
        ], delimiter="\t")
        for row in reader:
            pident = float(row["pident"])
            qcov = float(row["qcovs"])
            bitscore = float(row["bitscore"])
            if pident < identity_thresh or qcov < cov_thresh:
                continue
            qid = row["qseqid"]
            if qid not in best_hits or bitscore > best_hits[qid]["bitscore"]:
                best_hits[qid] = {
                    "uniprot_id": row["sseqid"],
                    "bitscore": bitscore
                }

    mapping = {q: hit["uniprot_id"] for q, hit in best_hits.items()}

    if json_output:
        with open(json_output, "w") as f:
            json.dump(mapping, f, indent=2)

    return mapping
