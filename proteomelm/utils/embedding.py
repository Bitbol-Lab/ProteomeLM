#!/usr/bin/env python3
"""
Embedding computation utilities for ProteomeLM.

This module provides shared functions for:
- Computing ESM-C embeddings (``build_genome_esmc``, used by ``proteomelm.ppi``)
- Computing ProteomeLM contextualized embeddings
- Extracting attention weights
- Managing embedding caches (used by the experiments/examples scripts)
"""

import os
from typing import Optional, Tuple, Union, List, Dict
from pathlib import Path

import numpy as np
import torch
from esm.models.esmc import ESMC

from Bio import SeqIO
from tqdm import tqdm

ESMC_600M = "esmc_600m"


def average_representation(output, input_ids, pad_token_id: int):
    r"""
    Average the representation of a sequence by ignoring padding tokens.

    Args:
        output (torch.Tensor): The output tensor of shape (batch_size, sequence_length, hidden_size).
        input_ids (torch.LongTensor): The input tensor of shape (batch_size, sequence_length).
        pad_token_id (int): The padding token ID.

    Returns:
        torch.Tensor: The averaged representation of shape (batch_size, hidden_size).
    """
    mask = input_ids != pad_token_id
    output[~mask] = 0.0
    valid_counts = mask.sum(dim=1, keepdim=True).clamp(min=1)
    return output.sum(dim=1) / valid_counts


def check_embeddings(embeddings: torch.Tensor, labels: List[str], device: str) -> None:
    """Raise if any mean embedding is non-finite or exactly zero.

    Neither occurs for a real protein, but a faulty GPU has been seen to return
    all-zero (or NaN) ESM-C outputs without raising, which then flows silently
    into every downstream ProteomeLM feature.
    """
    bad = ~torch.isfinite(embeddings).all(dim=1) | (embeddings == 0).all(dim=1)
    if bad.any():
        examples = [labels[i] for i in bad.nonzero().flatten()[:5].tolist()]
        raise RuntimeError(
            f"ESM-C returned zero or non-finite embeddings for {int(bad.sum())}/{len(labels)} "
            f"sequences on device {device!r} (e.g. {examples}). This indicates a device/kernel "
            f"fault, not bad input; rerun on another device."
        )


def prepare_model_esm(checkpoint: str, device: str) -> ESMC:
    """
    Prepare a model for encoding sequences.

    Args:
        checkpoint (str): Path to the model checkpoint.
        device (str): Device to load the model on.

    Returns:
        ESMC: The loaded model.
    """
    model = ESMC.from_pretrained(checkpoint)
    model.eval().to(device, dtype=torch.bfloat16)
    return model


def build_genome_esmc(
        fasta_file: Union[Path, str],
        device: str = "cuda:0",
) -> Dict[str, np.array]:
    """
    Build genome-level embeddings by encoding sequences and normalizing them
    with OrthoDB group statistics.

    Args:
        fasta_file (Union[Path, str]): Path to the FASTA file containing protein sequences.
        device (str): Device to use for encoding (e.g., 'cuda:0' or 'cpu').

    Returns:
        Dict[str, np.array]: A dictionary containing `inputs_embeds`, `group_labels` and `group_embeds`.
    """
    # Prepare the model and tokenizer
    model = prepare_model_esm(ESMC_600M, device)
    # Encode dataset
    with torch.no_grad():
        output: Dict[str, Dict[str, np.array]] = encode_dataset_esmc(model, fasta_file, device=device)
    return output


@torch.no_grad()
def encode_dataset_esmc(
        model: ESMC,
        fasta_file: Optional[Union[Path, str]] = None,
        data: Optional[Tuple[List[str], List[str]]] = None,
        keep_hidden_layers: Optional[Tuple[int, ...]] = None,
        device: str = "cpu",
) -> Dict[str, np.array]:
    r"""
    Encode a dataset of protein sequences using a model and a tokenizer.

    Args:
        model (ESMC): The model to use for encoding.
        fasta_file (Union[Path, str]): Path to the FASTA file containing protein sequences.
        keep_hidden_layers (Tuple[int, ...], optional): Indices of ESM-C hidden layers whose mean-pooled
            representations are returned under ``hidden_states`` (``None``: skip them; the
            ``inputs_embeds``/``group_embeds`` outputs are unaffected).
        device (str): Device to use for encoding (e.g., 'cuda' or 'cpu').

    Returns:
         Dict[str, np.array]: A dictionary mapping sequence identifiers to their encoded representations.
    """

    # Step 1: Parse the FASTA file and tokenize sequences
    if data is not None:
        assert isinstance(data, tuple) and len(data) == 2, "Data must be a tuple of (labels, sequences)."
        labels, sequences = data
        assert isinstance(labels, list) and isinstance(sequences, list), "Labels and sequences must be lists."
        assert len(labels) == len(sequences), "Labels and sequences must have the same length."
    else:
        assert fasta_file is not None, "Either data or fasta_file must be provided."
        fasta_file = Path(fasta_file)
        labels: List[str] = []
        sequences: List[str] = []
        for record in SeqIO.parse(fasta_file, "fasta"):
            labels.append(record.id)
            sequences.append(str(record.seq)[:4096])

    # Step 2: Encode the dataset using the model
    all_hidden_states = None
    all_embeddings = []
    max_number_of_tokens = 16000
    sorted_indices = sorted(range(len(sequences)), key=lambda idx: len(sequences[idx]), reverse=True)
    sorted_sequences = [sequences[idx] for idx in sorted_indices]
    
    current_batch = []
    current_num_tokens = 0

    def run_batch(batch, all_hidden_states, all_embeddings):
        input_ids: torch.Tensor = model._tokenize(batch).long()

        output = model(input_ids.to(device))

        # Compute sequence-level representations
        embeddings = output.embeddings
        hiddens = output.hidden_states
        if all_hidden_states is None:
            all_hidden_states = [[] for _ in keep_hidden_layers] if keep_hidden_layers is not None else None
        if all_hidden_states is not None:
            for j, i in enumerate(keep_hidden_layers):
                all_hidden_states[j].append(
                    average_representation(hiddens[i], input_ids, model.tokenizer.pad_token_id).detach().cpu())
        all_embeddings.append(average_representation(embeddings, input_ids, model.tokenizer.pad_token_id).detach().cpu())
        return all_hidden_states, all_embeddings

    for i in tqdm(range(0, len(sorted_sequences)), desc="Encoding sequences"):
        # Prepare batch
        # empty cache
        torch.cuda.empty_cache()
        if current_num_tokens + len(sorted_sequences[i]) > max_number_of_tokens:
            all_hidden_states, all_embeddings = run_batch(current_batch, all_hidden_states, all_embeddings)
            current_batch = []
            current_num_tokens = 0
        current_batch.append(sorted_sequences[i])
        current_num_tokens += len(sorted_sequences[i])

    if len(current_batch) > 0:
        all_hidden_states, all_embeddings = run_batch(current_batch, all_hidden_states, all_embeddings)
    restore_order = torch.empty(len(sorted_indices), dtype=torch.long)
    restore_order[torch.tensor(sorted_indices, dtype=torch.long)] = torch.arange(len(sorted_indices), dtype=torch.long)
    all_hidden_states = [torch.cat(hiddens, 0)[restore_order] for hiddens in all_hidden_states] if all_hidden_states is not None else None
    # Restore original FASTA order after length-sorted batching.
    all_embeddings = torch.cat(all_embeddings, 0)[restore_order]
    check_embeddings(all_embeddings, labels, device)

    # Step 3: Map labels to representations
    return {"group_embeds": all_embeddings,  # to avoid relying on ODB. TODO: rely on odb on the fly!
            "hidden_states": all_hidden_states,
            "inputs_embeds": all_embeddings,
            "group_labels": labels}



def compute_esm_embeddings(
    fasta_path: str, 
    output_dir: str, 
    prefix: str = "proteome",
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    force_recompute: bool = False
) -> torch.Tensor:
    """
    Compute ESM-C embeddings for sequences in FASTA file.
    
    Uses the build_genome_esmc function from proteomelm.utils.
    Results are cached to avoid recomputation.
    
    Args:
        fasta_path: Path to FASTA file with sequences
        output_dir: Directory to save embeddings
        prefix: Prefix for output files (default: "proteome")
        device: Device to use for computation (default: auto-detect)
        force_recompute: Force recomputation even if cache exists
        
    Returns:
        ESM embeddings tensor of shape (n_proteins, hidden_dim)
    """    
    cache_path = os.path.join(output_dir, f"{prefix}_esm_embeddings.pt")
    
    # Load from cache if available
    if os.path.exists(cache_path) and not force_recompute:
        print(f"  Loading cached ESM embeddings: {cache_path}")
        return torch.load(cache_path, map_location='cpu')
    
    print(f"  Computing ESM-C embeddings from {fasta_path}...")
    print(f"    Device: {device}")
    
    # Compute embeddings using ProteomeLM's build_genome_esmc
    result = build_genome_esmc(fasta_path, device=device)
    embeddings = result['inputs_embeds']
    
    # Cache the results
    torch.save(embeddings, cache_path)
    print(f"  Saved ESM embeddings: {embeddings.shape} -> {cache_path}")
    
    return embeddings


def compute_proteomelm_embeddings(
    esm_embeddings: torch.Tensor,
    output_dir: str,
    model_path: str,
    prefix: str = "proteome",
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    force_recompute: bool = False,
    save_attentions: bool = True,
    group_embeds: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Compute ProteomeLM contextualized embeddings and optionally attention weights.
    
    Args:
        esm_embeddings: ESM embeddings tensor (n_proteins, hidden_dim)
        output_dir: Directory to save embeddings
        model_path: Path to ProteomeLM model checkpoint
        prefix: Prefix for output files (default: "proteome")
        device: Device to use for computation (default: auto-detect)
        force_recompute: Force recomputation even if cache exists
        save_attentions: Whether to extract and save attention weights (post-softmax)
        group_embeds: Optional functional/group embeddings (n_proteins, hidden_dim).
                      If provided, these will be used as the functional encoding for each protein.
                      Typically these are OrthoDB group mean embeddings.
                      If None, defaults to using ESM embeddings as group embeddings.
        
    Returns:
        Tuple of (contextualized_embeddings, attentions)
        - contextualized_embeddings: (n_proteins, hidden_dim)
        - attentions: (n_layers, n_heads, n_proteins, n_proteins) or None
    """
    from proteomelm.modeling_proteomelm import ProteomeLMForMaskedLM
    
    cache_path = os.path.join(output_dir, f"{prefix}_proteomelm_embeddings.pt")
    attn_cache_path = os.path.join(output_dir, f"{prefix}_proteomelm_attentions.pt")
    
    # Load from cache if available
    if os.path.exists(cache_path) and not force_recompute:
        print(f"  Loading cached ProteomeLM embeddings: {cache_path}")
        embeddings = torch.load(cache_path, map_location='cpu')
        
        attentions = None
        if save_attentions and os.path.exists(attn_cache_path):
            print(f"  Loading cached attentions: {attn_cache_path}")
            attentions = torch.load(attn_cache_path, map_location='cpu')
                
        return embeddings, attentions
    
    if model_path is None:
        raise ValueError("model_path is required to compute ProteomeLM embeddings")
    
    print(f"  Loading ProteomeLM model: {model_path}")
    device_obj = torch.device(device)
    model = ProteomeLMForMaskedLM.from_pretrained(model_path).to(device_obj).eval()
    
    print(f"  Computing ProteomeLM contextualized embeddings...")
    print(f"    Device: {device}")
    print(f"    Input shape: {esm_embeddings.shape}")
    

    with torch.no_grad():
        # Add batch dimension and move to device
        inputs_embeds = esm_embeddings.unsqueeze(0).to(device_obj, dtype=model.dtype)
        
        # Prepare group embeddings if provided
        group_embeds_tensor = None
        if group_embeds is not None:
            group_embeds_tensor = group_embeds.unsqueeze(0).to(device_obj, dtype=model.dtype)
            print(f"    Using provided group embeddings: {group_embeds.shape}")
                
        # Forward pass with hidden states and optionally attentions
        outputs = model(
            inputs_embeds=inputs_embeds,
            group_embeds=group_embeds_tensor,
            output_hidden_states=True,
            output_attentions=save_attentions
        )
        
        
        # Extract last hidden state (contextualized embeddings)
        contextualized = outputs.hidden_states[-1].squeeze(0).cpu()
        
        # Extract and save attention weights if requested
        attentions = None
        if save_attentions and outputs.attentions is not None:
            # Stack all layers: (n_layers, n_heads, n_proteins, n_proteins)
            attentions = torch.stack([attn.squeeze(0) for attn in outputs.attentions]).cpu()
            torch.save(attentions, attn_cache_path)
            print(f"  Saved attentions: {attentions.shape} -> {attn_cache_path}")
            
    # Save contextualized embeddings
    torch.save(contextualized, cache_path)
    print(f"  Saved ProteomeLM embeddings: {contextualized.shape} -> {cache_path}")
    
    return contextualized, attentions


def load_attentions(
    output_dir: str,
    prefix: str = "proteome"
) -> torch.Tensor:
    """
    Load cached attention weights.
    
    Args:
        output_dir: Directory containing attentions
        prefix: Prefix used when saving attentions
        
    Returns:
        Attention tensor (n_layers, n_heads, n_proteins, n_proteins)
        
    Raises:
        FileNotFoundError: If attentions file doesn't exist
    """
    cache_path = os.path.join(output_dir, f"{prefix}_proteomelm_attentions.pt")
    
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Attentions not found: {cache_path}")
    
    return torch.load(cache_path, map_location='cpu')


def compute_all_embeddings(
    fasta_path: str,
    output_dir: str,
    model_path: Optional[str] = None,
    prefix: str = "proteome",
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    force_recompute: bool = False,
    orthodb_tsv_path: Optional[str] = None,
    orthodb_db_path: Optional[str] = None,
    orthodb_min_group_size: int = 0,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Compute ESM, (optional) OrthoDB group, and ProteomeLM embeddings in one call.

    When *orthodb_tsv_path* and *orthodb_db_path* are provided, true OrthoDB
    functional group embeddings are built and passed to the ProteomeLM
    forward pass as ``group_embeds``.  Proteins without an OrthoDB mapping
    fall back to their ESM embedding (same behaviour as the training
    dataloader).

    Args:
        fasta_path: Path to FASTA file with sequences.
        output_dir: Directory to save embeddings.
        model_path: Path to ProteomeLM model (required for ProteomeLM embeddings).
        prefix: Prefix for output files.
        device: Device to use for computation.
        force_recompute: Force recomputation even if cache exists.
        orthodb_tsv_path: Path to the OrthoDB cross-reference TSV produced by
            ``download_proteome(..., download_orthodb=True)``.
        orthodb_db_path: Directory containing ``group_vectors_*.pkl`` files
            (e.g. ``data/training``).
        orthodb_min_group_size: Only load groups with at least this many members.

    Returns:
        Tuple of (esm_embeddings, proteomelm_embeddings, attentions)
    """
    from .proteome import load_orthodb_group_vectors, build_group_embeddings_for_proteome

    # Step 1: ESM embeddings
    esm_embeddings = compute_esm_embeddings(
        fasta_path=fasta_path,
        output_dir=output_dir,
        prefix=prefix,
        device=device,
        force_recompute=force_recompute,
    )

    # Step 2: OrthoDB group embeddings (optional)
    group_embeds = None
    if orthodb_tsv_path is not None and orthodb_db_path is not None:
        cache_path = os.path.join(output_dir, f"{prefix}_orthodb_group_embeddings.pt")
        if os.path.exists(cache_path) and not force_recompute:
            print(f"  Loading cached OrthoDB group embeddings: {cache_path}")
            group_embeds = torch.load(cache_path, map_location="cpu")
        else:
            orthodb_means = load_orthodb_group_vectors(orthodb_db_path, min_group_size=orthodb_min_group_size)
            group_embeds, mask = build_group_embeddings_for_proteome(
                fasta_path=fasta_path,
                orthodb_tsv_path=orthodb_tsv_path,
                orthodb_group_means=orthodb_means,
                esm_embeddings=esm_embeddings,
            )
            torch.save(group_embeds, cache_path)
            print(f"  Saved OrthoDB group embeddings: {group_embeds.shape} -> {cache_path}")

    # Step 3: ProteomeLM embeddings
    proteomelm_embeddings = None
    attentions = None

    if model_path is not None:
        proteomelm_embeddings, attentions = compute_proteomelm_embeddings(
            esm_embeddings=esm_embeddings,
            output_dir=output_dir,
            model_path=model_path,
            prefix=prefix,
            device=device,
            force_recompute=force_recompute,
            save_attentions=True,
            group_embeds=group_embeds,
        )

    return esm_embeddings, proteomelm_embeddings, attentions
