#!/usr/bin/env python3
"""
Small file I/O helpers shared across ProteomeLM.

This module provides shared utilities for:
- File I/O operations (FASTA parsing/writing, JSON handling)
- Directory management
- PDB structure download

Used by: proteomelm.ppi, proteomelm.utils.proteome and the experiments/examples scripts.
"""

import os
import json
from typing import Dict
import requests


def ensure_dir(path: str) -> str:
    """
    Create directory if it doesn't exist.
    
    Args:
        path: Directory path to create
        
    Returns:
        The path that was created
    """
    os.makedirs(path, exist_ok=True)
    return path


def parse_fasta(fasta_path: str) -> Dict[str, str]:
    """
    Parse FASTA file into dictionary.
    
    Handles both simple and complex headers (e.g., >sp|P12345|PROT_HUMAN).
    Extracts UniProt ID if present in header.
    
    Args:
        fasta_path: Path to FASTA file
        
    Returns:
        Dictionary mapping sequence ID to sequence
    """
    sequences = {}
    current_id = None
    current_seq = []
    
    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                # Save previous sequence
                if current_id is not None:
                    sequences[current_id] = ''.join(current_seq)
                
                # Parse header
                header = line[1:].strip()
                
                # Try to extract UniProt ID from header like >sp|P12345|NAME
                if '|' in header:
                    parts = header.split('|')
                    if len(parts) >= 2:
                        current_id = parts[1]
                    else:
                        current_id = parts[0]
                else:
                    current_id = header.split()[0]
                
                current_seq = []
            else:
                current_seq.append(line)
        
        # Save last sequence
        if current_id is not None:
            sequences[current_id] = ''.join(current_seq)
    
    return sequences


def write_fasta(sequences: Dict[str, str], output_path: str) -> str:
    """
    Write sequences to FASTA file.
    
    Args:
        sequences: Dictionary mapping sequence ID to sequence
        output_path: Path to output FASTA file
        
    Returns:
        Path to created file
    """
    with open(output_path, 'w') as f:
        for seq_id, seq in sequences.items():
            f.write(f">{seq_id}\n{seq}\n")
    return output_path


def save_json(data: dict, output_path: str, indent: int = 2) -> str:
    """
    Save data to JSON file.
    
    Args:
        data: Data to save
        output_path: Path to output JSON file
        indent: JSON indentation level
        
    Returns:
        Path to created file
    """
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=indent)
    return output_path


def download_pdb_structure(
    pdb_id: str,
    output_dir: str,
    file_format: str = 'cif'
) -> str:
    """
    Download structure from PDB.
    
    Args:
        pdb_id: PDB ID (e.g., '7K00')
        output_dir: Directory to save structure
        file_format: Format to download ('cif' or 'pdb')
        
    Returns:
        Path to downloaded structure file
    """
    pdb_id = pdb_id.upper()
    file_ext = file_format.lower()
    file_path = os.path.join(output_dir, f"{pdb_id}.{file_ext}")
    
    if os.path.exists(file_path):
        print(f"  Using cached structure: {file_path}")
        return file_path
    
    print(f"  Downloading PDB {pdb_id}...")
    url = f"https://files.rcsb.org/download/{pdb_id}.{file_ext}"
    
    response = requests.get(url, timeout=120)
    if response.status_code == 200:
        with open(file_path, 'w') as f:
            f.write(response.text)
        print(f"  Downloaded structure: {file_path}")
        return file_path
    else:
        raise RuntimeError(f"Failed to download PDB {pdb_id}: HTTP {response.status_code}")