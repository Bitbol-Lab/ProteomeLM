#!/usr/bin/env python3
"""
Proteome download and processing utilities for ProteomeLM validation examples.

This module provides shared functions for:
- Downloading proteomes from UniProt
- Downloading structures from PDB
- Finding specific proteins in proteomes
- Mapping proteins to proteome indices
- Loading OrthoDB group vectors and building functional embeddings
"""

import os
import pickle
import logging
from typing import Dict, List, Optional, Tuple

import torch
import requests

logger = logging.getLogger(__name__)


def download_proteome(
    proteome_id: str,
    output_dir: str,
    organism_name: str = "organism",
    reviewed_only: bool = True,
    include_isoforms: bool = False,
    max_retries: int = 3,
    download_orthodb: bool = False,
) -> str:
    """Download proteome from UniProt in FASTA format, optionally with OrthoDB mappings."""
    import time

    safe_name = organism_name.lower().replace(" ", "_")
    fasta_path = os.path.join(output_dir, f"{safe_name}_proteome.fasta")
    orthodb_path = os.path.join(output_dir, f"{safe_name}_orthodb.tsv")

    if os.path.exists(fasta_path) and os.path.getsize(fasta_path) > 10000:
        if not download_orthodb or os.path.exists(orthodb_path):
            print(f"  Using cached proteome: {fasta_path}")
            return fasta_path

    os.makedirs(output_dir, exist_ok=True)

    query = f"(proteome:{proteome_id})"
    if reviewed_only:
        query += " AND (reviewed:true)"

    base_url = "https://rest.uniprot.org/uniprotkb/stream"

    def _download(params, dest):
        for attempt in range(max_retries):
            try:
                resp = requests.get(base_url, params=params, timeout=600, stream=True)
                if resp.status_code == 429:
                    time.sleep(float(resp.headers.get("Retry-After", 10)))
                    continue
                resp.raise_for_status()
                tmp = dest + ".tmp"
                size = 0
                with open(tmp, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=65536):
                        f.write(chunk)
                        size += len(chunk)
                os.rename(tmp, dest)
                return size
            except (requests.RequestException, IOError) as e:
                print(f"  Attempt {attempt + 1} failed: {e}")
                time.sleep(2 ** attempt)
        raise RuntimeError(f"Failed to download after {max_retries} attempts")

    # Download FASTA
    print(f"  Downloading {organism_name} proteome ({proteome_id})...")
    size = _download(
        {"query": query, "format": "fasta", "includeIsoform": str(include_isoforms).lower()},
        fasta_path,
    )
    n = sum(1 for line in open(fasta_path) if line.startswith(">"))
    print(f"  Downloaded: {size / 1e6:.1f} MB, {n:,} proteins")

    # Download OrthoDB mappings
    if download_orthodb:
        print(f"  Fetching OrthoDB cross-references...")
        _download(
            {"query": query, "format": "tsv", "fields": "accession,xref_orthodb"},
            orthodb_path,
        )
        with open(orthodb_path) as f:
            next(f)
            mapped = sum(1 for line in f if line.strip().split("\t")[-1])
        print(f"  OrthoDB mappings: {mapped:,} proteins mapped")

    return fasta_path


def find_proteins_in_proteome(
    target_proteins: Dict[str, dict],
    proteome_sequences: Dict[str, str],
    uniprot_key: str = 'uniprot'
) -> Dict[str, dict]:
    """
    Find target proteins in proteome and return their indices.
    
    Args:
        target_proteins: Dict of protein_id -> {uniprot: 'P12345', ...}
        proteome_sequences: Dict of UniProt ID -> sequence
        uniprot_key: Key in target_proteins dict containing UniProt ID
        
    Returns:
        Dictionary of found proteins with proteome_idx added
    """
    # Create index mapping
    proteome_index = {uid: idx for idx, uid in enumerate(proteome_sequences.keys())}
    
    found_proteins = {}
    found_list = []
    missing_list = []
    
    for protein_id, protein_info in target_proteins.items():
        uniprot_id = protein_info.get(uniprot_key)
        
        if uniprot_id is None:
            missing_list.append(f"{protein_id} (no UniProt ID)")
            continue
        
        # Handle whitespace in UniProt IDs
        uniprot_id = uniprot_id.strip()
        
        if uniprot_id in proteome_index:
            found_proteins[protein_id] = {
                **protein_info,
                'proteome_idx': proteome_index[uniprot_id],
                'sequence': proteome_sequences[uniprot_id],
                'sequence_length': len(proteome_sequences[uniprot_id])
            }
            found_list.append(protein_id)
        else:
            missing_list.append(f"{protein_id} ({uniprot_id})")
    
    print(f"  Found {len(found_list)}/{len(target_proteins)} proteins in proteome")
    if missing_list:
        print(f"  Missing: {', '.join(missing_list)}")
    
    return found_proteins


# =============================================================================
# ORTHODB GROUP VECTOR LOADING & FUNCTIONAL EMBEDDINGS
# =============================================================================


def load_orthodb_group_vectors(
    db_path: str,
    min_group_size: int = 0,
) -> Dict[str, torch.Tensor]:
    """
    Load OrthoDB group mean embedding vectors from pickle files.

    The group_vectors pickle files live alongside the training shards
    (e.g. ``/data1/malbrank/proteomelm/training/group_vectors_*.pkl``).
    Each file is a dict mapping OrthoDB group IDs to
    ``(mean_embedding, group_size)`` tuples.  Only groups whose file
    threshold is >= *min_group_size* are loaded.

    Reuses the same file discovery logic as the training dataloader
    (see :func:`proteomelm.dataloaders._load_orthodb_data_once`).

    Args:
        db_path: Directory containing ``group_vectors_*.pkl`` files.
        min_group_size: Skip files whose size threshold is below this value.

    Returns:
        Dict mapping OrthoDB group ID -> mean embedding tensor (shape ``[1152]``).
    """
    group_vector_files = [
        "group_vectors_0.pkl",
        "group_vectors_10.pkl",
        "group_vectors_50.pkl",
        "group_vectors_200.pkl",
    ]

    group_means: Dict[str, torch.Tensor] = {}

    for filename in group_vector_files:
        # Parse threshold from filename (e.g. group_vectors_10.pkl -> 10)
        try:
            threshold = int(filename.split("_")[-1].split(".")[0])
        except (ValueError, IndexError):
            continue

        if threshold < min_group_size:
            continue

        file_path = os.path.join(db_path, filename)
        if not os.path.exists(file_path):
            logger.warning(f"OrthoDB file not found: {file_path}")
            continue

        logger.info(f"Loading OrthoDB group vectors from {file_path} ...")
        with open(file_path, "rb") as f:
            data = pickle.load(f)

        # value = (mean_embedding_tensor, group_size_float)
        for k, v in data.items():
            if len(v) > 0:
                group_means[k] = v[0]

        logger.info(f"  Loaded {len(data)} groups from {filename}")

    logger.info(f"Total OrthoDB groups loaded: {len(group_means)}")
    return group_means


def parse_orthodb_tsv(orthodb_tsv_path: str) -> Dict[str, List[str]]:
    """
    Parse the OrthoDB cross-reference TSV produced by :func:`download_proteome`.

    The TSV has header ``Entry\\tCross-reference (OrthoDB)`` and each row
    maps a UniProt accession to a semicolon-separated list of OrthoDB group
    IDs.

    Args:
        orthodb_tsv_path: Path to the TSV file.

    Returns:
        Dict mapping UniProt accession -> list of OrthoDB group IDs.
    """
    mapping: Dict[str, List[str]] = {}

    with open(orthodb_tsv_path) as f:
        try:
            header = next(f)  # skip header
        except StopIteration:
            logger.warning(f"OrthoDB TSV is empty: {orthodb_tsv_path}")
            return mapping
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            accession = parts[0].strip()
            og_field = parts[1].strip()
            if not og_field:
                continue
            # OrthoDB groups are separated by "; "
            og_ids = [g.strip().rstrip(";") for g in og_field.split(";") if g.strip()]
            if og_ids:
                mapping[accession] = og_ids

    return mapping


def build_group_embeddings_for_proteome(
    fasta_path: str,
    orthodb_tsv_path: str,
    orthodb_group_means: Dict[str, torch.Tensor],
    esm_embeddings: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build per-protein functional (group) embeddings for a full proteome.

    For each protein in the FASTA, look up its OrthoDB group ID(s) via the
    TSV mapping, then retrieve the corresponding mean embedding from
    *orthodb_group_means*.  If a protein maps to multiple groups, the first
    match found in the group vectors is used.  Proteins without a mapping
    (or whose group is absent from the vectors) fall back to their own ESM-C
    embedding — same behaviour as the training dataloader.

    Args:
        fasta_path: FASTA file (same one used for ESM encoding).
        orthodb_tsv_path: TSV produced by ``download_proteome(..., download_orthodb=True)``.
        orthodb_group_means: Output of :func:`load_orthodb_group_vectors`.
        esm_embeddings: ESM embeddings tensor ``(n_proteins, dim)``
            used as fallback for unmapped proteins.

    Returns:
        Tuple of ``(group_embeddings, mask)`` where:
        - *group_embeddings* has shape ``(n_proteins, dim)``
        - *mask* is a bool tensor of shape ``(n_proteins,)`` where True
          means the protein had a valid OrthoDB group embedding.
    """
    from .io import parse_fasta

    sequences = parse_fasta(fasta_path)
    protein_ids = list(sequences.keys())
    n_proteins = len(protein_ids)

    # Determine embedding dim from the first available group mean
    sample_embed = next(iter(orthodb_group_means.values()))
    dim = sample_embed.shape[0]

    # Parse UniProt -> OG mapping
    uniprot_to_ogs = parse_orthodb_tsv(orthodb_tsv_path)

    # Start from ESM embeddings (fallback for every protein by default)
    group_embeddings = esm_embeddings.clone().float()
    mask = torch.zeros(n_proteins, dtype=torch.bool)

    n_mapped = 0
    for i, pid in enumerate(protein_ids):
        og_ids = uniprot_to_ogs.get(pid, [])
        for og_id in og_ids:
            if og_id in orthodb_group_means:
                group_embeddings[i] = orthodb_group_means[og_id].float()
                mask[i] = True
                n_mapped += 1
                break

    n_fallback = n_proteins - n_mapped
    print(f"  OrthoDB group embeddings: {n_mapped}/{n_proteins} mapped, "
          f"{n_fallback} fallback to ESM-C")

    return group_embeddings, mask


def download_orthodb_tsv_for_accessions(
    accessions: List[str],
    output_path: str,
    max_per_query: int = 50,
) -> str:
    """
    Download OrthoDB cross-references for a list of UniProt accessions.

    Args:
        accessions: List of UniProt accessions.
        output_path: Path to the TSV file to write.
        max_per_query: Max number of accessions per query chunk.

    Returns:
        Path to the written TSV file.
    """
    if not accessions:
        raise ValueError("No accessions provided for OrthoDB TSV download")

    base_url = "https://rest.uniprot.org/uniprotkb/stream"
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    def _query_for_chunk(chunk: List[str]) -> str:
        query = "(" + " OR ".join([f"accession:{acc}" for acc in chunk]) + ")"
        params = {"query": query, "format": "tsv", "fields": "accession,xref_orthodb"}
        try:
            resp = requests.get(base_url, params=params, timeout=600)
            resp.raise_for_status()
            return resp.text
        except requests.HTTPError as e:
            # Fallback to POST for large queries
            if e.response is not None and e.response.status_code == 400:
                resp = requests.post(base_url, data=params, timeout=600)
                resp.raise_for_status()
                return resp.text
            raise

    written_header = False
    total_lines = 0
    with open(output_path, "w") as f:
        i = 0
        while i < len(accessions):
            chunk = accessions[i:i + max_per_query]
            try:
                text = _query_for_chunk(chunk)
            except requests.HTTPError as e:
                # Reduce chunk size if query is too large
                if e.response is not None and e.response.status_code == 400 and len(chunk) > 10:
                    max_per_query = max(10, max_per_query // 2)
                    continue
                raise

            lines = text.splitlines()
            if lines:
                if written_header:
                    lines = lines[1:]
                else:
                    written_header = True
                for line in lines:
                    f.write(line + "\n")
                    total_lines += 1

            i += len(chunk)

    if total_lines == 0:
        raise RuntimeError(f"No OrthoDB mappings returned for {len(accessions)} accessions")

    return output_path
