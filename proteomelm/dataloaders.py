"""
Data loaders and dataset classes for ProteomeLM.

This module provides efficient data loading capabilities for ProteomeLM training,
including hierarchical protein embeddings and masking strategies.
"""

import logging
import os
import pickle
import random
import tarfile
from threading import Lock
from dataclasses import dataclass
from typing import Dict, List, Tuple, Sequence, Optional

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers.trainer_pt_utils import IterableDataset

logger = logging.getLogger(__name__)

# Global cache for OrthoDB data with thread safety
_GLOBAL_ORTHODB_CACHE = {}
_CACHE_LOCK = Lock()


def _load_orthodb_data_once(db_path: str, min_taxid_size: int) -> Tuple[set, Dict]:
    """
    Load OrthoDB data with caching and thread safety.

    Args:
        db_path: Path to the database directory
        min_taxid_size: Minimum taxonomic ID size filter

    Returns:
        Tuple of (orthodb_ids, orthodb_means)

    Raises:
        FileNotFoundError: If required OrthoDB files are missing
        ValueError: If no valid data is found
    """
    global _GLOBAL_ORTHODB_CACHE

    cache_key = (db_path, min_taxid_size)

    # Thread-safe cache check
    with _CACHE_LOCK:
        if cache_key in _GLOBAL_ORTHODB_CACHE:
            logger.debug(f"Using cached OrthoDB data for {cache_key}")
            return _GLOBAL_ORTHODB_CACHE[cache_key]

    # Define expected group vector files
    group_vector_files = [
        "group_vectors_0.pkl",
        "group_vectors_10.pkl",
        "group_vectors_50.pkl",
        "group_vectors_200.pkl",
    ]

    group_vectors = {}
    files_loaded = 0

    for filename in group_vector_files:
        file_path = os.path.join(db_path, filename)

        # Extract count from filename
        try:
            count = int(filename.split("_")[-1].split(".")[0])
        except (ValueError, IndexError) as e:
            logger.warning(f"Could not parse count from filename {filename}: {e}")
            continue

        if count < min_taxid_size:
            logger.debug(f"Skipping {filename} (count {count} < min_taxid_size {min_taxid_size})")
            continue

        if not os.path.exists(file_path):
            logger.warning(f"OrthoDB file not found: {file_path}")
            continue

        try:
            with open(file_path, "rb") as f:
                file_data = pickle.load(f)
                group_vectors.update(file_data)
                files_loaded += 1
                logger.debug(f"Loaded {len(file_data)} orthodb groups from {file_path}")
        except Exception as e:
            logger.error(f"Failed to load OrthoDB data from {file_path}: {e}")
            continue

    if not group_vectors:
        raise ValueError(f"No valid OrthoDB data found in {db_path} with min_taxid_size={min_taxid_size}")

    logger.info(f"Successfully loaded {len(group_vectors)} OrthoDB groups from {files_loaded} files")

    # Build final data structures
    orthodb_ids = set(group_vectors.keys())
    orthodb_means = {k: v[0] for k, v in group_vectors.items() if len(v) > 0}

    # Cache the results thread-safely
    with _CACHE_LOCK:
        _GLOBAL_ORTHODB_CACHE[cache_key] = (orthodb_ids, orthodb_means)

    return orthodb_ids, orthodb_means


class ProteomeLMDataset(IterableDataset):
    """
    Iterable dataset for ProteomeLM training data.

    This dataset loads protein sequences from tarfile shards and applies
    hierarchical masking based on OrthoDB groupings.

    Args:
        db_path: Path to the database directory containing shards
        dataset: Dataset split name ("train", "val", "test")
        max_length: Maximum number of sequences per sample
        min_taxid_size: Minimum taxonomic ID size for OrthoDB filtering
        mask_fraction: Fraction of sequences to mask during training
    """

    def __init__(self,
                 db_path: str,
                 dataset: str = "train",
                 max_length: int = 4096,
                 min_taxid_size: int = 100,
                 mask_fraction: float = 0.5,
                 shuffle_shards: bool = True,
                 *args, **kwargs):
        """Initialize the ProteomeLM dataset."""
        super().__init__()

        # Validate inputs
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Database path does not exist: {db_path}")

        if max_length <= 0:
            raise ValueError(f"max_length must be positive, got {max_length}")

        if not 0 <= mask_fraction <= 1:
            raise ValueError(f"mask_fraction must be between 0 and 1, got {mask_fraction}")

        # Store parameters
        self.db_path = db_path
        self.dataset = dataset
        self.max_length = max_length
        self.mask_fraction = mask_fraction
        self.min_taxid_size = min_taxid_size
        self.shuffle_shards = shuffle_shards

        # Build shard pattern and discover shards
        self.shard_pattern = os.path.join(self.db_path, dataset, "shard_{}.tar")
        self.shards, self.total_samples = self._discover_shards()

        # Load OrthoDB data once per process
        self.orthodb_ids, self.orthodb_means = _load_orthodb_data_once(
            self.db_path, self.min_taxid_size
        )

        logger.info(f"Initialized {self.__class__.__name__} with {len(self.shards)} shards, "
                    f"{self.total_samples} total samples, {len(self.orthodb_ids)} OrthoDB groups")

    def __len__(self) -> int:
        """Return the total number of samples across all shards."""
        return self.total_samples

    def _discover_shards(self) -> Tuple[List[str], int]:
        """
        Discover available shards and count total samples.

        Returns:
            Tuple of (shard_paths, total_sample_count)

        Raises:
            FileNotFoundError: If dataset directory or shards are missing
            ValueError: If no valid shards are found
        """
        dataset_dir = os.path.join(self.db_path, self.dataset)

        if not os.path.exists(dataset_dir):
            raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

        # Find all shard files
        try:
            shard_files = [f for f in os.listdir(dataset_dir) if f.startswith("shard_") and f.endswith(".tar")]
        except OSError as e:
            raise FileNotFoundError(f"Failed to list files in {dataset_dir}: {e}")

        if not shard_files:
            raise ValueError(f"No shard files found in {dataset_dir}")

        # Extract shard indices and sort
        shard_indices = []
        for filename in shard_files:
            try:
                # Extract number from "shard_X.tar"
                index = int(filename.split("_")[-1].split(".")[0])
                shard_indices.append(index)
            except (ValueError, IndexError) as e:
                logger.warning(f"Could not parse shard index from {filename}: {e}")
                continue

        if not shard_indices:
            raise ValueError(f"No valid shard files found in {dataset_dir}")

        shard_indices.sort()

        # Build shard paths
        shard_paths = [
            self.shard_pattern.format(idx) for idx in shard_indices
        ]

        # Verify shard files exist and count samples
        valid_shards = []
        total_samples = 0

        for shard_path in shard_paths:
            if not os.path.exists(shard_path):
                logger.warning(f"Shard file not found: {shard_path}")
                continue

            try:
                with tarfile.open(shard_path, 'r') as tar:
                    sample_count = len(tar.getmembers())
                    total_samples += sample_count
                    valid_shards.append(shard_path)
                    logger.debug(f"Shard {shard_path}: {sample_count} samples")
            except Exception as e:
                logger.error(f"Failed to read shard {shard_path}: {e}")
                continue

        if not valid_shards:
            raise ValueError(f"No readable shard files found in {dataset_dir}")

        # Shuffle shards if requested
        if self.shuffle_shards:
            random.seed(42)  # For reproducibility
            random.shuffle(valid_shards)

        logger.info(f"Discovered {len(valid_shards)} valid shards with {total_samples} total samples")
        return valid_shards, total_samples

    def __iter__(self):
        """
        Stream samples from the shards with proper error handling.

        Yields:
            Processed data samples from the shards
        """
        for shard_path in self.shards:
            logger.debug(f"Processing shard: {shard_path}")

            try:
                with tarfile.open(shard_path, 'r') as tar:
                    for member in tar:
                        if not member.isfile():
                            continue

                        try:
                            with tar.extractfile(member) as f:
                                if f is None:
                                    logger.warning(f"Could not extract member {member.name} from {shard_path}")
                                    continue

                                data = pickle.load(f)
                                yield self._process_sample(data)

                        except Exception as e:
                            logger.error(f"Failed to process member {member.name} in {shard_path}: {e}")
                            continue

            except Exception as e:
                logger.error(f"Failed to read shard {shard_path}: {e}")
                continue

    def _process_sample(self, data: Tuple[str, List[str], List[List[str]], List[torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Process a single data sample with hierarchical masking.

        Args:
            data: Tuple containing (tax_id, gene_names, orthodb_groups, embeddings)

        Returns:
            Dictionary containing processed embeddings and mask information

        Raises:
            ValueError: If data format is invalid
        """
        try:
            tax_id, gene_names, odb_groups, embeds = data
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid data format: expected 4-tuple, got {type(data)}: {e}")

        if len(odb_groups) != len(embeds):
            raise ValueError(f"Mismatch between orthodb groups ({len(odb_groups)}) and embeddings ({len(embeds)})")

        # Step 1: Subsample sequences to max_length
        members = list(zip(odb_groups, embeds))
        random.shuffle(members)
        subsampled_members = members[:self.max_length]

        if not subsampled_members:
            # Return empty tensors if no data
            empty_tensor = torch.empty(0, embeds[0].size(-1) if embeds else 512)
            return {
                "inputs_embeds": empty_tensor,
                "group_embeds": empty_tensor,
                "masked_tokens": torch.empty(0, dtype=torch.long),
            }

        # Step 2: Extract embeddings
        inputs_embeds = torch.stack([member[1] for member in subsampled_members])

        # Step 3: Get group embeddings with fallback to input embeddings
        group_embeds_list = []
        unmaskable_indices = []

        for i, (odb_group_list, input_embed) in enumerate(subsampled_members):
            # Find valid orthodb means for this sequence
            valid_means = [
                self.orthodb_means[k] for k in odb_group_list
                if k in self.orthodb_ids and k in self.orthodb_means
            ]

            if valid_means:
                # Randomly select one of the valid group means
                group_embeds_list.append(random.choice(valid_means))
            else:
                # Fallback to input embedding and mark as unmaskable
                group_embeds_list.append(input_embed)
                unmaskable_indices.append(i)

        group_embeds = torch.stack(group_embeds_list)

        # Step 4: Create masking pattern
        num_sequences = len(subsampled_members)
        num_to_mask = max(1, int(self.mask_fraction * num_sequences))

        # Create mask with random selection, avoiding unmaskable sequences
        maskable_indices = [i for i in range(num_sequences) if i not in unmaskable_indices]

        if len(maskable_indices) >= num_to_mask:
            masked_indices = random.sample(maskable_indices, num_to_mask)
        else:
            # If we don't have enough maskable sequences, mask what we can
            masked_indices = maskable_indices
            if masked_indices:
                logger.debug(
                    f"Only {len(masked_indices)} maskable sequences available, "
                    f"requested {num_to_mask}"
                )

        # Create mask tensor
        masked_tokens = torch.zeros(num_sequences, dtype=torch.long)
        if masked_indices:
            masked_tokens[masked_indices] = 1

        return {
            "inputs_embeds": inputs_embeds,
            "group_embeds": group_embeds,
            "masked_tokens": masked_tokens,
        }


@dataclass
class DataCollatorForProteomeLM:
    """
    Data collator for ProteomeLM that handles batching and padding.

    This collator pads sequences to the maximum length in the batch and
    applies masking based on the masked_tokens tensor.

    Args:
        return_tensors: Format for returned tensors ("pt" for PyTorch)
        pad_to_multiple_of: Pad sequence length to multiple of this value
    """

    return_tensors: str = "pt"
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate instances into a batch with proper padding and masking.

        Args:
            instances: List of instances to collate

        Returns:
            Dictionary containing the collated and padded batch data

        Raises:
            ValueError: If instances are empty or have inconsistent structure
        """
        if not instances:
            raise ValueError("Cannot collate empty list of instances")

        # Validate instance structure
        required_keys = {"inputs_embeds", "group_embeds", "masked_tokens"}
        for i, instance in enumerate(instances):
            if not isinstance(instance, dict):
                raise ValueError(f"Instance {i} is not a dictionary")
            missing_keys = required_keys - set(instance.keys())
            if missing_keys:
                raise ValueError(f"Instance {i} missing keys: {missing_keys}")

        # Extract components
        inputs_embeds = [instance["inputs_embeds"] for instance in instances]
        group_embeds = [instance["group_embeds"] for instance in instances]
        masked_tokens = [instance["masked_tokens"] for instance in instances]

        # Check for empty sequences
        valid_indices = []
        for i, (inp, grp, mask) in enumerate(zip(inputs_embeds, group_embeds, masked_tokens)):
            if inp.numel() > 0 and grp.numel() > 0 and mask.numel() > 0:
                valid_indices.append(i)
            else:
                logger.warning(f"Skipping empty instance {i}")

        if not valid_indices:
            # Return minimal batch if all instances are empty
            empty_tensor = torch.empty(0, 0, 512)  # Default embedding dim
            return {
                "inputs_embeds": empty_tensor,
                "group_embeds": empty_tensor,
                "masked_tokens": torch.empty(0, 0, dtype=torch.long),
                "labels": empty_tensor,
            }

        # Filter to valid instances
        inputs_embeds = [inputs_embeds[i] for i in valid_indices]
        group_embeds = [group_embeds[i] for i in valid_indices]
        masked_tokens = [masked_tokens[i] for i in valid_indices]

        # Pad sequences to maximum length in batch
        try:
            inputs_embeds_padded = pad_sequence(inputs_embeds, batch_first=True)
            group_embeds_padded = pad_sequence(group_embeds, batch_first=True)
            masked_tokens_padded = pad_sequence(masked_tokens, batch_first=True, padding_value=0)
        except Exception as e:
            raise ValueError(f"Failed to pad sequences: {e}")

        # Convert to appropriate data types
        inputs_embeds_padded = inputs_embeds_padded.to(torch.bfloat16)
        group_embeds_padded = group_embeds_padded.to(torch.bfloat16)
        masked_tokens_padded = masked_tokens_padded.to(torch.long)

        # Create labels (copy of original inputs before masking)
        labels = inputs_embeds_padded.clone()

        # Apply masking: replace masked positions with group embeddings
        mask_bool = (masked_tokens_padded == 1)
        inputs_embeds_padded[mask_bool] = group_embeds_padded[mask_bool]

        return {
            "inputs_embeds": inputs_embeds_padded,
            "group_embeds": group_embeds_padded,
            "masked_tokens": masked_tokens_padded,
            "labels": labels,
        }


# Utility functions

def get_shards_dataset(*args, **kwargs) -> ProteomeLMDataset:
    """
    Create a ProteomeLM dataset with lazy loading from shards.

    Args:
        *args: Positional arguments passed to ProteomeLMDataset
        **kwargs: Keyword arguments passed to ProteomeLMDataset

    Returns:
        Initialized ProteomeLMDataset instance
    """
    return ProteomeLMDataset(*args, **kwargs)


def create_dataloader(dataset: ProteomeLMDataset,
                      batch_size: int = 16,
                      num_workers: int = 0,
                      pin_memory: bool = True,
                      **kwargs) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader for ProteomeLM training.

    Args:
        dataset: ProteomeLM dataset instance
        batch_size: Number of samples per batch
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
        **kwargs: Additional arguments passed to DataLoader

    Returns:
        Configured DataLoader instance
    """
    collator = DataCollatorForProteomeLM()

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collator,
        **kwargs
    )


def clear_orthodb_cache():
    """Clear the global OrthoDB cache to free memory."""
    global _GLOBAL_ORTHODB_CACHE
    with _CACHE_LOCK:
        _GLOBAL_ORTHODB_CACHE.clear()
        logger.info("Cleared OrthoDB cache")


def get_dataset_info(db_path: str, dataset: str = "train") -> Dict[str, int]:
    """
    Get information about a dataset without loading it.

    Args:
        db_path: Path to the database directory
        dataset: Dataset split name

    Returns:
        Dictionary with dataset statistics
    """
    dataset_dir = os.path.join(db_path, dataset)

    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    # Count shards and samples
    shard_files = [f for f in os.listdir(dataset_dir) if f.startswith("shard_") and f.endswith(".tar")]

    total_samples = 0
    valid_shards = 0

    for shard_file in shard_files:
        shard_path = os.path.join(dataset_dir, shard_file)
        try:
            with tarfile.open(shard_path, 'r') as tar:
                total_samples += len(tar.getmembers())
                valid_shards += 1
        except Exception as e:
            logger.warning(f"Could not read shard {shard_file}: {e}")

    return {
        "total_shards": len(shard_files),
        "valid_shards": valid_shards,
        "total_samples": total_samples,
        "avg_samples_per_shard": total_samples / valid_shards if valid_shards > 0 else 0,
    }
