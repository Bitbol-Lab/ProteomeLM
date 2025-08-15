
"""
Utility functions for ProteomeLM.

This module provides various utility functions for model training,
evaluation, and data processing.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import re
import time
import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from Bio import SeqIO


# Try to import optional dependencies
import torch
import torch.nn as nn
import torch.nn.functional as F


from esm.models.esmc import ESMC
from .modeling_proteomelm import ProteomeLMForMaskedLM

from transformers import (
    get_cosine_schedule_with_warmup,
    get_constant_schedule_with_warmup
)
from .modeling_proteomelm import ProteomeLMConfig

logger = logging.getLogger(__name__)

ESMC_600M = "esmc_600m"


def setup_logging(level: Union[str, int] = logging.INFO,
                  format_string: Optional[str] = None,
                  include_timestamp: bool = True) -> None:
    """
    Set up logging configuration for ProteomeLM.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        format_string: Custom format string for log messages
        include_timestamp: Whether to include timestamp in log messages
    """
    if format_string is None:
        if include_timestamp:
            format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        else:
            format_string = "%(name)s - %(levelname)s - %(message)s"

    logging.basicConfig(
        level=level,
        format=format_string,
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Set specific logger levels
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)


def compute_metrics(eval_pred) -> Dict[str, float]:
    """
    Compute evaluation metrics for the model using cosine similarity over valid positions
    (i.e., ignoring positions where labels == -100).
    """

    (norm_loss, cosine_loss, predictions, _), labels = eval_pred
    # Flatten last dimension if needed, e.g. (batch, seq, dim)
    preds = predictions.reshape(-1, predictions.shape[-1])
    labs = labels.reshape(-1, labels.shape[-1])

    if len(preds) == 0:
        return {"cosine_similarity": 0.0}

    # Normalize and compute mean cosine similarity
    pnorm = np.linalg.norm(preds, axis=-1, keepdims=True)
    lnorm = np.linalg.norm(labs, axis=-1, keepdims=True)
    pnorm[pnorm == 0] = 1e-7
    lnorm[lnorm == 0] = 1e-7

    preds = preds / pnorm
    labs = labs / lnorm

    cosines = (preds * labs).sum(axis=-1)
    return {"cosine_similarity": float(np.mean(cosines)),
            "norm_loss": float(np.mean(norm_loss)),
            "cosine_loss": float(np.mean(cosine_loss))}


def print_number_of_parameters(model) -> None:
    """
    Print the number of trainable parameters in the model.

    Args:
        model (torch.nn.Module): The model instance.
    """
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params}")


def load_scheduler(optimizer, config: Dict[str, Any], len_train: int, eff_batch_size: int):
    """
    Load a scheduler for training.

    Args:
        optimizer (Optimizer): The optimizer used for training.
        config (Dict[str, Any]): Configuration dictionary.
        len_train (int): Length of the training dataset.
        eff_batch_size (int): Effective batch size.

    Returns:
        Scheduler instance.
    """
    total_steps = config["num_epochs"] * len_train // eff_batch_size

    if config["scheduler"] == "cosine":
        print("Using cosine scheduler")
        return get_cosine_schedule_with_warmup(optimizer, config["warmup_steps"], total_steps)
    elif config["scheduler"] == "constant":
        print("Using constant scheduler with warmup")
        return get_constant_schedule_with_warmup(optimizer, num_warmup_steps=config["warmup_steps"])
    else:
        raise ValueError("Invalid scheduler type. Must be 'cosine', 'cosine-restarts', or 'constant'.")


def load_model(
        model_path: str,
        device: str,
        model_class=nn.Module,
        dtype=torch.bfloat16,
        checkpoint_mixer: bool = False,
):
    """
    Load a model from a checkpoint.

    Args:
        model_path (str): Path to the model checkpoint.
        device (str): Device to load the model on.
        model_class: Class of the model to load.
        dtype: Precision for model weights.
        checkpoint_mixer (bool): Whether to use checkpointed mixer.

    Returns:
        The loaded model.
    """
    return model_class.from_pretrained(model_path, device=device, dtype=dtype, checkpoint_mixer=checkpoint_mixer)


def symmetrize(x):
    """Make layer symmetric in final two dimensions, used for contact prediction."""
    return x + x.transpose(-1, -2)


def average_product_correct(x):
    """Perform average product correct, used for contact prediction."""
    a1 = x.sum(-1, keepdims=True)
    a2 = x.sum(-2, keepdims=True)
    a12 = x.sum((-1, -2), keepdims=True) + 1e-5

    avg = a1 * a2
    avg.div_(a12)  # in-place to reduce memory
    normalized = x - avg
    return normalized


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


def pairwise_logsigmoid_loss_from_logits(logits, labels):
    """
    Computes a pairwise ranking loss using logsigmoid on model logits and binary labels.

    Arguments:
        logits: Tensor of shape (batch_size,) - model outputs (raw scores)
        labels: Tensor of shape (batch_size,) - binary labels (1 for positive, 0 for negative)

    Returns:
        Scalar tensor loss (mean over all pos-neg pairs)
    """
    # Ensure labels are boolean mask
    pos_mask = labels == 1
    neg_mask = labels == 0

    pos_scores = logits[pos_mask]
    neg_scores = logits[neg_mask]

    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)  # no valid pairs

    # Form all pairwise differences: pos_i - neg_j
    # Shape: (num_pos, num_neg)
    diff = pos_scores[:, None] - neg_scores[None, :]

    # Apply smooth ranking loss
    loss = -F.logsigmoid(diff).mean()

    return loss

# ----------------
# Data Utilities
# ----------------


def download_uniprot_tsv(query: str, output_file: Optional[str] = None, fasta_file: Optional[str] = None) -> None:
    """
    Download UniProt data in TSV format for a given organism and export sequences as FASTA.
    API documentation: https://www.uniprot.org/api-documentation/uniprotkb

    Parameters:
        query (str): query can be organism_id:{organism_id} or proteome:{proteome_id}
        output_file (Optional[str]): File name for the TSV output.
        fasta_file (Optional[str]): File name for the FASTA output.
    """

    id_ = query.split(":")[1]
    if output_file is None:
        output_file = f"uniprotkb_{id_}.tsv"
    if fasta_file is None:
        fasta_file = f"uniprotkb_{id_}.fasta"

    url = "https://rest.uniprot.org/uniprotkb/stream"
    remaining_fields = [
        "accession", "id", "protein_name", "gene_primary", "organism_name", "length", "sequence",
        "go", "go_p", "go_c", "go_f", "go_id", "ec",
        "cc_subcellular_location", "cc_tissue_specificity", "cc_developmental_stage", "cc_pathway", "cc_function",
        "xref_kegg", "xref_reactome", "xref_string", "xref_intact", "xref_corum", "xref_complexportal", "annotation_score"
    ]

    def make_request(fields: List[str], retries: int = 3, timeout: int = 30) -> Optional[requests.Response]:
        params = {
            "query": query,
            "format": "tsv",
            "fields": ",".join(fields),
            "includeIsoforms": "false",
        }
        attempt = 0
        while attempt < retries:
            try:
                response = requests.get(url, params=params, timeout=timeout)
                response.raise_for_status()  # Raise exception for 4xx/5xx responses
                return response
            except requests.exceptions.ChunkedEncodingError as e:
                attempt += 1
                logger.warning(f"Attempt {attempt} failed due to chunked encoding error: {e}. Retrying...")
                time.sleep(2 ** attempt)  # Exponential backoff
            except requests.exceptions.Timeout as e:
                attempt += 1
                logger.warning(f"Attempt {attempt} timed out: {e}. Retrying...")
                time.sleep(2 ** attempt)
            except requests.RequestException as e:
                logger.error(f"Request failed: {e}")
                break
        return None

    try:
        attempt = 1
        max_attempts = 5

        while attempt <= max_attempts:
            logger.info("Attempt %d: Requesting data with %d fields...", attempt, len(remaining_fields))
            response = make_request(remaining_fields)

            if response and response.status_code == 200:
                # Save raw response
                with open(output_file, "w") as f:
                    f.write(response.text)

                lines = response.text.strip().split("\n")
                if len(lines) < 2:
                    logger.error("No data returned from UniProt")
                    return

                logger.info("Success! TSV saved as '%s' with %d fields.", output_file, len(remaining_fields))

                # Process with pandas
                try:
                    df = pd.read_csv(output_file, sep="\t", low_memory=False)
                    if "Annotation" in df.columns and "Gene Names (primary)" in df.columns:
                        df = df.sort_values("Annotation", ascending=False).groupby("Gene Names (primary)", as_index=False).head(1)
                        df.to_csv(output_file, sep="\t", index=False)
                        logger.info("Filtered TSV saved as '%s'", output_file)
                except Exception as e:
                    logger.warning(f"Could not process with pandas: {e}")

                logger.info("Total entries: %d", len(lines) - 1)

                # Check headers and export FASTA if requested
                header = lines[0].strip().split("\t")
                if fasta_file and "Entry" in header and "Sequence" in header:
                    try:
                        acc_idx = header.index("Entry")
                        seq_idx = header.index("Sequence")
                        with open(fasta_file, "w") as f_fasta:
                            for line in lines[1:]:
                                cols = line.split("\t")
                                if len(cols) > max(acc_idx, seq_idx):
                                    accession = cols[acc_idx]
                                    sequence = cols[seq_idx]
                                    if accession and sequence:
                                        f_fasta.write(f">{accession}\n{sequence}\n")
                        logger.info("FASTA export complete: '%s'", fasta_file)
                    except Exception as e:
                        logger.error(f"Error exporting FASTA: {e}")
                elif fasta_file:
                    logger.warning("Cannot export FASTA: required fields 'Entry' or 'Sequence' missing.")

                logger.info("Data download completed successfully.")
                return

            elif response:
                # Handle invalid fields
                invalid_fields = re.findall(r"Invalid fields parameter value '([^']+)'", response.text)
                if invalid_fields:
                    logger.warning("Removing %d invalid fields.", len(invalid_fields))
                    for field in invalid_fields:
                        logger.warning(" - Removing field: %s", field)
                        if field in remaining_fields:
                            remaining_fields.remove(field)

                    if not remaining_fields:
                        logger.error("No valid fields remain. Cannot continue.")
                        return

                    attempt += 1
                    continue
                else:
                    logger.error("Request failed. Response:\n%s", response.text[:500])
                    return
            else:
                logger.error("Failed to get response from UniProt")
                return

        logger.error("Maximum attempts reached. Download failed.")

    except Exception as e:
        logger.error(f"Error downloading data: {e}")
        return


def check_model_compatibility(model_path: str) -> bool:
    """
    Check if a model is compatible with current ProteomeLM version.

    Args:
        model_path: Path to model or Hugging Face model ID

    Returns:
        True if compatible, False otherwise
    """
    try:
        # Try to load config first
        ProteomeLMConfig.from_pretrained(model_path)
        logger.info(f"Model config loaded successfully: {model_path}")
        return True

    except Exception as e:
        logger.error(f"Model compatibility check failed: {e}")
        return False


def get_model_info(model_path: str) -> Optional[Dict[str, Any]]:
    """
    Get information about a ProteomeLM model.

    Args:
        model_path: Path to model or Hugging Face model ID

    Returns:
        Dictionary with model information or None if failed
    """
    try:
        from .modeling_proteomelm import ProteomeLMConfig

        config = ProteomeLMConfig.from_pretrained(model_path)

        # Calculate approximate model size
        total_params = (
            config.dim * config.input_size +  # input projection
            config.n_layers * (
                4 * config.dim * config.dim +  # attention
                4 * config.dim * config.dim     # mlp
            ) +
            config.dim * config.input_size      # output projection
        )

        model_info = {
            "model_path": model_path,
            "parameters": total_params,
            "layers": config.n_layers,
            "hidden_size": config.dim,
            "attention_heads": config.n_heads,
            "input_size": config.input_size,
            "max_length": getattr(config, 'max_length', None),
            "vocab_size": getattr(config, 'vocab_size', None),
        }

        return model_info

    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        return None


def validate_training_config(config: Dict[str, Any]) -> List[str]:
    """
    Validate training configuration and return list of issues.

    Args:
        config: Training configuration dictionary

    Returns:
        List of validation error messages (empty if valid)
    """
    issues = []

    # Required fields
    required_fields = [
        "batch_size", "learning_rate", "num_epochs",
        "output_dir", "namedir", "dim", "n_layers", "n_heads"
    ]

    for field in required_fields:
        if field not in config:
            issues.append(f"Missing required field: {field}")

    # Value validation
    if "batch_size" in config and config["batch_size"] <= 0:
        issues.append("batch_size must be positive")

    if "learning_rate" in config and config["learning_rate"] <= 0:
        issues.append("learning_rate must be positive")

    if "num_epochs" in config and config["num_epochs"] <= 0:
        issues.append("num_epochs must be positive")

    if "n_layers" in config and config["n_layers"] <= 0:
        issues.append("n_layers must be positive")

    if "n_heads" in config and config["n_heads"] <= 0:
        issues.append("n_heads must be positive")

    if "dim" in config and config["dim"] <= 0:
        issues.append("dim must be positive")

    # Check if dim is divisible by n_heads
    if "dim" in config and "n_heads" in config:
        if config["dim"] % config["n_heads"] != 0:
            issues.append("dim must be divisible by n_heads")

    return issues

# ----------------
# ESM-C Utilities
# ----------------


@torch.no_grad()
def encode_dataset_esmc(
        model: ESMC,
        fasta_file: Optional[Union[Path, str]] = None,
        data: Optional[Tuple[List[str], List[str]]] = None,
        keep_hidden_layers: Optional[Tuple[int]] = (6, 12, 18, 24, 30),
        device: str = "cpu",
) -> Dict[str, np.array]:
    r"""
    Encode a dataset of protein sequences using a model and a tokenizer.

    Args:
        model (ESMC): The model to use for encoding.
        fasta_file (Union[Path, str]): Path to the FASTA file containing protein sequences.
        keep_hidden_layers (Tuple[int]): Indices of hidden layers to keep.
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

    for i in tqdm(range(0, len(sequences)), desc="Encoding sequences"):
        # Prepare batch
        # empty cache
        torch.cuda.empty_cache()
        if current_num_tokens + len(sequences[i]) > max_number_of_tokens:
            all_hidden_states, all_embeddings = run_batch(current_batch, all_hidden_states, all_embeddings)
            current_batch = []
            current_num_tokens = 0
        current_batch.append(sequences[i])
        current_num_tokens += len(sequences[i])

    if len(current_batch) > 0:
        all_hidden_states, all_embeddings = run_batch(current_batch, all_hidden_states, all_embeddings)
    all_hidden_states = [torch.cat(hiddens, 0) for hiddens in all_hidden_states] if all_hidden_states is not None else None
    all_embeddings = torch.cat(all_embeddings, 0)

    # Step 3: Map labels to representations
    return {"group_embeds": all_embeddings,  # to avoid relying on ODB. TODO: rely on odb on the fly!
            "hidden_states": all_hidden_states,
            "inputs_embeds": all_embeddings,
            "group_labels": labels}


def prepare_model(checkpoint: str, device: str) -> ESMC:
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
    model = prepare_model(ESMC_600M, device)
    # Encode dataset
    with torch.no_grad():
        output: Dict[str, Dict[str, np.array]] = encode_dataset_esmc(model, fasta_file, device=device)
    return output

# ----------------
# ProteomeLM Utilities
# ----------------


@torch.no_grad()
def encode_dataset_proteomelm(model: ProteomeLMForMaskedLM,
                              esmc_embeddings: torch.Tensor,
                              device: str = "cpu",
                              batch_size: int = 16) -> torch.Tensor:
    r"""
    Encode a dataset of protein sequences using a model and a tokenizer.
    Args:
        model:
        esmc_embeddings:
        device:

    """
    output = []
    for i in range(0, len(esmc_embeddings), batch_size):
        inputs_embeds = esmc_embeddings[i:i + batch_size].to(device, dtype=torch.bfloat16)
        output.append(model.forward(inputs_embeds=inputs_embeds).logits.cpu())
    return torch.cat(output, 0)
