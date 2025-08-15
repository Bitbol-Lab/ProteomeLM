"""
Data processing utilities for protein-protein interaction analysis.
"""
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Union

import torch


logger = logging.getLogger(__name__)


class FastaProcessor:
    """Handles FASTA file processing operations."""

    @staticmethod
    def build_index_map(fasta_path: Union[str, Path]) -> Dict[str, int]:
        """
        Build an index map from protein identifiers to sequence indices.

        Args:
            fasta_path: Path to the FASTA file

        Returns:
            Dictionary mapping protein identifiers to sequence indices
        """
        index_map = {}
        index = 0

        with open(fasta_path, "r") as fasta_file:
            for line in fasta_file:
                if line.startswith(">"):
                    # Extract sequence identifiers (comma-separated)
                    identifiers = line.strip().split(",")
                    # Remove the ">" symbol from the first identifier
                    identifiers[0] = identifiers[0][1:]

                    # Map all identifiers to the current index
                    for identifier in identifiers:
                        index_map[identifier] = index

                    # Increment index for next sequence
                    index += 1

        logger.info(f"Built index map with {len(index_map)} identifiers for {index} sequences")
        return index_map

    @staticmethod
    def tsv_to_fasta(tsv_files: List[str], fasta_file: str) -> None:
        """
        Convert TSV files to a single FASTA file.

        Args:
            tsv_files: List of TSV file paths
            fasta_file: Output FASTA file path
        """
        with open(fasta_file, "w") as output_file:
            labels = set()
            for tsv_file in tsv_files:
                with open(tsv_file, "r") as input_file:
                    for line in input_file:
                        label, seq = line.strip().split("\t")[:2]
                        if label not in labels:
                            labels.add(label)
                            output_file.write(f">{label}\n{seq}\n")

        logger.info(f"Converted {len(tsv_files)} TSV files to FASTA with {len(labels)} unique sequences")


class InteractionExtractor:
    """Base class for extracting protein-protein interactions from different datasets."""

    def __init__(self, fasta_processor: FastaProcessor):
        self.fasta_processor = fasta_processor

    def extract(self, env_dir: Union[str, Path], fasta_file: str) -> Tuple[Dict[str, List[Tuple[int, int]]], Dict[str, torch.Tensor]]:
        """
        Extract interaction vectors from a dataset.

        Args:
            env_dir: Directory containing the dataset files
            fasta_file: Path to the FASTA file

        Returns:
            Tuple containing:
            - Dictionary with dataset names as keys and lists of index pairs as values
            - Dictionary with dataset names as keys and tensors of interaction labels as values
        """
        raise NotImplementedError("Subclasses must implement extract method")


class DScriptExtractor(InteractionExtractor):
    """Extracts interactions from DScript dataset format."""

    def extract(self, env_dir: Union[str, Path], fasta_file: str) -> Tuple[Dict[str, List[Tuple[int, int]]], Dict[str, torch.Tensor]]:
        """Extract interaction vectors from DScript dataset."""
        if isinstance(env_dir, str):
            env_dir = Path(env_dir)

        index_map = self.fasta_processor.build_index_map(env_dir / fasta_file)

        index_pairs_dict, y_dict = {}, {}

        for dataset in ["train", "val", "test"]:
            # Last part of the path is the species
            spec = env_dir.parts[-1]
            interaction_file = env_dir / f"{spec}_{dataset}_interaction.tsv"

            if not interaction_file.exists():
                logger.warning(f"Interaction file not found: {interaction_file}")
                continue
            index_pairs = []
            y = []  # Interaction labels
            with open(interaction_file, "r") as f:
                for line in f:
                    # Split the line into sequence1, sequence2, and label
                    seq1, seq2, label = line.strip().split("\t")
                    label = int(float(label))

                    # Convert sequences to indices using index_map
                    index1 = index_map.get(seq1)
                    index2 = index_map.get(seq2)

                    if index1 is not None and index2 is not None:
                        # Store index pairs and label
                        index_pairs.append((index1, index2))
                        y.append(label)
                    else:
                        logger.debug(f"Skipping pair ({seq1}, {seq2}) - not found in index map")

            index_pairs_dict[dataset] = index_pairs
            y_dict[dataset] = torch.tensor(y)
            logger.info(f"Extracted {len(index_pairs)} interactions for {dataset} dataset")

        return index_pairs_dict, y_dict


class BernettExtractor(InteractionExtractor):
    """Extracts interactions from Bernett et al. dataset."""

    def extract(self, env_dir: Union[str, Path], fasta_file: str) -> Tuple[Dict[str, List[Tuple[int, int]]], Dict[str, torch.Tensor]]:
        """Extract interaction vectors from Bernett et al. dataset."""
        if isinstance(env_dir, str):
            env_dir = Path(env_dir)

        index_map = self.fasta_processor.build_index_map(env_dir / fasta_file)

        index_pairs_dict, y_dict = {}, {}

        for i, dataset in enumerate(["train", "val", "test"]):
            interaction_file_pos = env_dir / f"Intra{i}_pos_rr.txt"
            interaction_file_neg = env_dir / f"Intra{i}_neg_rr.txt"

            if not interaction_file_pos.exists() or not interaction_file_neg.exists():
                logger.warning(f"Interaction files not found for {dataset} dataset")
                continue

            index_pairs = []
            y = []  # Interaction labels

            # Process positive interactions
            with open(interaction_file_pos, "r") as f:
                for line in f:
                    seq1, seq2 = line.strip().split()
                    index1 = index_map.get(seq1)
                    index2 = index_map.get(seq2)

                    if index1 is not None and index2 is not None:
                        index_pairs.append((index1, index2))
                        y.append(1)

            # Process negative interactions
            with open(interaction_file_neg, "r") as f:
                for line in f:
                    seq1, seq2 = line.strip().split()
                    index1 = index_map.get(seq1)
                    index2 = index_map.get(seq2)

                    if index1 is not None and index2 is not None:
                        index_pairs.append((index1, index2))
                        y.append(0)

            index_pairs_dict[dataset] = index_pairs
            y_dict[dataset] = torch.tensor(y)
            logger.info(f"Extracted {len(index_pairs)} interactions for {dataset} dataset")

        return index_pairs_dict, y_dict


# Factory function for creating extractors
def create_extractor(dataset_type: str) -> InteractionExtractor:
    """
    Create an appropriate interaction extractor for the given dataset type.

    Args:
        dataset_type: Type of dataset ('dscript' or 'goldstandard')

    Returns:
        Appropriate InteractionExtractor instance
    """
    fasta_processor = FastaProcessor()

    if dataset_type.lower() == "dscript":
        return DScriptExtractor(fasta_processor)
    elif dataset_type.lower() == "bernett":
        return BernettExtractor(fasta_processor)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
