"""
ProteomeLM Dataset Encoding Pipeline

This module provides functionality to download, process, and encode protein sequences
from OrthoDB using ESM-C embeddings, with hierarchical group vector computation.
"""

import gzip
import os
import tarfile
import time
import pickle
import shutil
import logging
import requests
from pathlib import Path
from random import shuffle
from typing import Union, List, Tuple, Set, Dict, Any, Optional
from collections import defaultdict
from dataclasses import dataclass
from contextlib import contextmanager

import torch
from Bio import SeqIO
from tqdm import tqdm
from esm.models.esmc import ESMC

from .utils import average_representation


def setup_logging(level: str = "INFO", log_file: Optional[str] = None):
    """Setup logging configuration."""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        format=log_format,
        level=getattr(logging, level.upper()),
        handlers=[
            logging.StreamHandler(),
            *([] if log_file is None else [logging.FileHandler(log_file)])
        ]
    )


logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Configuration class for the encoding pipeline."""
    # Data paths
    orthodb_version: str = "odb12v1"
    save_path: str = "data/orthodb12_raw"

    # URLs for OrthoDB data
    base_url: str = "https://data.orthodb.org/current/download"

    # Processing parameters
    num_fasta_parts: int = 64
    max_tokens_per_batch: int = 60000
    device: str = "cuda:0"
    batch_size_threshold: int = 100000
    intermediate_save_interval: int = 100

    # Shard parameters
    shard_size: int = 500
    min_file_size_kb: int = 750

    # Model parameters
    model_name: str = "esmc_600m"
    dtype: torch.dtype = torch.bfloat16

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not torch.cuda.is_available() and "cuda" in self.device:
            logger.warning("CUDA not available, falling back to CPU")
            self.device = "cpu"


@contextmanager
def error_context(operation: str):
    """Context manager for consistent error handling."""
    try:
        logger.info(f"Starting: {operation}")
        yield
        logger.info(f"Completed: {operation}")
    except Exception as e:
        logger.error(f"Failed during {operation}: {str(e)}")
        raise


class DownloadManager:
    """Manages downloading and extraction of OrthoDB data."""

    def __init__(self, config: Config):
        self.config = config
        self.session = requests.Session()
        # Add retry adapter for robustness
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def download_and_extract(self, filename: str, url: str) -> None:
        """
        Download a gzipped file from the given URL, extract it, and remove the .gz file.

        Args:
            filename: Expected final extracted filename (without .gz).
            url: Direct download link for the .gz file.

        Raises:
            RuntimeError: If download fails.
        """
        target_file = Path(self.config.save_path) / filename
        gz_file = Path(str(target_file) + ".gz")

        # Skip if extracted file already exists
        if target_file.exists():
            logger.info(f"Skipping {filename}: already exists.")
            return

        with error_context(f"downloading {filename}"):
            # Create directory if it doesn't exist
            target_file.parent.mkdir(parents=True, exist_ok=True)

            # Download file with streaming to avoid high memory usage
            logger.info(f"Downloading {url} → {gz_file}...")
            try:
                response = self.session.get(url, stream=True, timeout=30)
                response.raise_for_status()
            except requests.RequestException as e:
                raise RuntimeError(f"Download failed for {url}: {e}")

            # Write file to disk in chunks
            total_size = int(response.headers.get('content-length', 0))
            with gzip.open(gz_file, "wb") as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))

            logger.info("Download complete.")

        with error_context(f"extracting {filename}"):
            # Extract and remove the .gz archive
            logger.info(f"Extracting {gz_file} to {target_file}...")
            with gzip.open(gz_file, "rb") as f_in:
                with open(target_file, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)

            # Clean up
            gz_file.unlink()
            logger.info(f"Extracted {filename} and removed archive.")

    def download_orthodb_data(self) -> None:
        downloads = [
            ("odb12v0_OG2genes.tab", f"{self.config.base_url}/{self.config.orthodb_version}_OG2genes.tab.gz"),
            ("odb12v0_OG_pairs.tab", f"{self.config.base_url}/{self.config.orthodb_version}_OG_pairs.tab.gz"),
            ("odb12v0_aa.fasta", f"{self.config.base_url}/{self.config.orthodb_version}_aa_fasta.gz"),
        ]
        for filename, url in downloads:
            self.download_and_extract(filename, url)

    def __del__(self):
        """Cleanup session when object is destroyed."""
        if hasattr(self, 'session'):
            self.session.close()


def download_and_extract(save_path: str, filename: str, url: str):
    """
    Legacy function for backward compatibility.
    Use DownloadManager for new code.
    """
    config = Config(save_path=save_path)
    manager = DownloadManager(config)
    manager.download_and_extract(filename, url)


def download_data(save_path: str):
    """
    Legacy function for backward compatibility.
    Use DownloadManager.download_orthodb_data for new code.
    """
    config = Config(save_path=save_path)
    manager = DownloadManager(config)
    manager.download_orthodb_data()


class FastaSplitter:
    """Handles splitting of large FASTA files."""

    def __init__(self, config: Config):
        self.config = config

    def split_fasta(self, input_file: Union[str, Path], num_parts: Optional[int] = None) -> List[Path]:
        """
        Split a FASTA file into multiple smaller files with approximately equal numbers of sequences.
        Each part is sorted by sequence length in descending order.

        Args:
            input_file: Path to the input FASTA file.
            num_parts: Number of parts to split the file into. Uses config default if None.

        Returns:
            List of paths to the created part files.
        """
        input_file = Path(input_file)
        if num_parts is None:
            num_parts = self.config.num_fasta_parts

        with error_context(f"splitting {input_file} into {num_parts} parts"):
            try:
                sequences = list(SeqIO.parse(input_file, "fasta"))
            except Exception as e:
                raise RuntimeError(f"Failed to parse FASTA file {input_file}: {e}")

            total_sequences = len(sequences)
            if total_sequences == 0:
                raise ValueError(f"No sequences found in {input_file}")

            chunk_size = total_sequences // num_parts
            remainder = total_sequences % num_parts

            part_files = []
            start_idx = 0

            for part in tqdm(range(num_parts), desc="Splitting FASTA file"):
                end_idx = start_idx + chunk_size + (1 if part < remainder else 0)
                part_sequences = sequences[start_idx:end_idx]

                # Sort by sequence length in descending order
                part_sequences.sort(key=lambda seq: len(seq.seq), reverse=True)

                output_file = Path(f"{input_file}.part{part}.fasta")
                try:
                    with open(output_file, "w") as out_f:
                        SeqIO.write(part_sequences, out_f, "fasta")
                    part_files.append(output_file)
                except Exception as e:
                    raise RuntimeError(f"Failed to write part file {output_file}: {e}")

                start_idx = end_idx

            logger.info(f"Successfully split {input_file} into {num_parts} files, each sorted by sequence length.")
            return part_files


def split_fasta(input_file: str, num_parts: int = 64):
    """
    Legacy function for backward compatibility.
    Use FastaSplitter.split_fasta for new code.
    """
    config = Config(num_fasta_parts=num_parts)
    splitter = FastaSplitter(config)
    splitter.split_fasta(input_file, num_parts)


class SequenceEncoder:
    """Handles encoding of protein sequences using ESM-C model."""

    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self._device = torch.device(config.device)

    def load_model(self) -> None:
        """Load and prepare the ESM-C model."""
        if self.model is not None:
            return

        with error_context(f"loading model {self.config.model_name}"):
            try:
                self.model = ESMC.from_pretrained(self.config.model_name)
                self.model = self.model.to(self._device, dtype=self.config.dtype).eval()
                logger.info(f"Model loaded on {self._device} with dtype {self.config.dtype}")
            except Exception as e:
                raise RuntimeError(f"Failed to load model: {e}")

    def _run_batch(self, batch: List[str]) -> torch.Tensor:
        """Process a batch of sequences and return embeddings."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        try:
            input_ids = self.model._tokenize(batch).long().to(self._device)
            output = self.model(input_ids)
            embeddings = average_representation(
                output.embeddings, input_ids, self.model.tokenizer.pad_token_id
            ).cpu()
            return embeddings
        except torch.cuda.OutOfMemoryError:
            logger.error("CUDA out of memory. Try reducing max_tokens_per_batch.")
            raise
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            raise

    def _save_intermediate_results(self, labels: List[str], embeddings_list: List[torch.Tensor],
                                   output_pickle: Path, part_num: int, start_idx: int) -> int:
        """Save intermediate results and return the new start index."""
        if not embeddings_list:
            return start_idx

        embeddings_tensor = torch.cat(embeddings_list, 0)
        end_idx = start_idx + len(embeddings_tensor)

        output_file = Path(f"{output_pickle}.{part_num}")
        data = {
            "labels": labels[start_idx:end_idx],
            "embeddings": embeddings_tensor.to(dtype=self.config.dtype)
        }

        try:
            with open(output_file, "wb") as f:
                pickle.dump(data, f)
            logger.info(f"Saved intermediate embeddings to {output_file}")
        except Exception as e:
            logger.error(f"Failed to save intermediate results: {e}")
            raise

        return end_idx

    @torch.no_grad()
    def encode_dataset(self, fasta_file: Union[Path, str], output_pickle: Union[Path, str]) -> Tuple[List[str], torch.Tensor]:
        """
        Encode a dataset of protein sequences using the ESM-C model.

        Args:
            fasta_file: Path to the input FASTA file.
            output_pickle: Base path for output pickle files.

        Returns:
            Tuple of (labels, final_embeddings_tensor)
        """
        fasta_file = Path(fasta_file)
        output_pickle = Path(output_pickle)

        if not fasta_file.exists():
            raise FileNotFoundError(f"FASTA file not found: {fasta_file}")

        # Load model if not already loaded
        self.load_model()

        with error_context(f"encoding dataset from {fasta_file}"):
            # Parse sequences
            labels, sequences = [], []
            try:
                for record in SeqIO.parse(fasta_file, "fasta"):
                    labels.append(record.id)
                    # Truncate sequences to maximum length
                    sequences.append(str(record.seq)[:4096])
            except Exception as e:
                raise RuntimeError(f"Failed to parse FASTA file: {e}")

            if not sequences:
                raise ValueError(f"No sequences found in {fasta_file}")

            logger.info(f"Loaded {len(sequences)} sequences for encoding")

            # Initialize processing variables
            all_embeddings = []
            current_batch, current_num_tokens = [], 0

            start_time = time.time()
            last_time = time.time()
            cumsum = 0  # for intermediate saving

            # Process sequences
            for i, seq in enumerate(tqdm(sequences, desc="Encoding sequences")):
                # Progress reporting
                if i > 0 and i % 10000 == 0:
                    elapsed_time = (time.time() - start_time) / 3600
                    remaining_time = (time.time() - last_time) * (len(sequences) - i) / (10000 * 3600)
                    logger.info(
                        f"Encoding sequence {i}/{len(sequences)} "
                        f"[Elapsed: {elapsed_time:.2f}h, Remaining: {remaining_time:.2f}h]"
                    )
                    last_time = time.time()

                    # Intermediate saving
                    if i % self.config.batch_size_threshold == 0:
                        part_num = i // self.config.batch_size_threshold
                        cumsum = self._save_intermediate_results(
                            labels, all_embeddings, output_pickle, part_num, cumsum
                        )
                        all_embeddings = []

                # Check if we need to process current batch
                if current_num_tokens + len(seq) > self.config.max_tokens_per_batch:
                    if current_batch:  # Only process if batch is not empty
                        embeddings = self._run_batch(current_batch)
                        all_embeddings.append(embeddings)
                    current_batch, current_num_tokens = [], 0

                current_batch.append(seq)
                current_num_tokens += len(seq)

            # Process final batch
            if current_batch:
                embeddings = self._run_batch(current_batch)
                all_embeddings.append(embeddings)

            # Save final results
            if all_embeddings:
                final_part = len(sequences) // self.config.batch_size_threshold + 1
                self._save_intermediate_results(
                    labels, all_embeddings, output_pickle, final_part, cumsum
                )

                # Return final tensor for backward compatibility
                embeddings_tensor = torch.cat(all_embeddings, 0)
                return labels, embeddings_tensor
            else:
                logger.warning("No embeddings generated")
                return labels, torch.empty(0)


@torch.no_grad()
def encode_dataset(
        model: ESMC,
        fasta_file: Union[Path, str],
        output_pickle: Union[Path, str],
        max_tokens_per_batch: int = 60000,
        device: str = "cuda",
) -> Tuple[List[str], torch.Tensor]:
    """
    Legacy function for backward compatibility.
    Use SequenceEncoder.encode_dataset for new code.
    """
    # Create config and encoder for backward compatibility
    config = Config(
        max_tokens_per_batch=max_tokens_per_batch,
        device=device
    )
    encoder = SequenceEncoder(config)
    encoder.model = model  # Use provided model instead of loading

    return encoder.encode_dataset(fasta_file, output_pickle)


class OrthoDB_Processor:
    """Handles processing of OrthoDB hierarchy and gene mappings."""

    def __init__(self, config: Config):
        self.config = config

    def process_odb_graph(self, file_path: Union[str, Path]) -> Tuple[Dict[str, List[str]], List[str], Dict[str, int]]:
        """
        Process a tab-delimited file containing child-parent pairs.

        Args:
            file_path: Path to the input file.

        Returns:
            Tuple of:
            - parent_to_children: mapping from parent to list of its children
            - children_to_parents_ordered: ordering of nodes (children-to-parents order)
            - node_index_mapping: mapping from each node to its index in the ordering
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"OrthoDB pairs file not found: {file_path}")

        with error_context("processing OrthoDB graph"):
            child_to_parent: Dict[str, str] = {}
            parent_to_children: Dict[str, List[str]] = defaultdict(list)

            try:
                with open(file_path, "r") as f:
                    for line_number, line in enumerate(f, start=1):
                        parts = line.strip().split("\t")
                        if len(parts) != 2:
                            logger.warning(f"Skipping malformed line {line_number}: {line.strip()}")
                            continue
                        child, parent = parts
                        child_to_parent[child] = parent
                        parent_to_children[parent].append(child)
            except Exception as e:
                raise RuntimeError(f"Error reading OrthoDB pairs file: {e}")

            # Identify roots: nodes that appear as parents but never as children
            roots = [node for node in parent_to_children if node not in child_to_parent]
            logger.info(f"Found {len(roots)} root nodes")

            if not roots:
                raise ValueError("No root nodes found in OrthoDB hierarchy")

            # Traverse the graph ensuring each parent is processed before its children
            ordered_nodes = []
            stack = roots[:]  # shallow copy of roots
            visited = set()

            while stack:
                node = stack.pop()
                if node in visited:
                    continue
                visited.add(node)
                ordered_nodes.append(node)
                stack.extend(parent_to_children.get(node, []))

            logger.info(f"Processed {len(ordered_nodes)} nodes in hierarchy")

            # Reverse the order to get the children-to-parents ordering
            children_to_parents_ordered = ordered_nodes[::-1]
            node_index_mapping = {node: idx for idx, node in enumerate(children_to_parents_ordered)}

            logger.info("Graph processing complete.")
            return dict(parent_to_children), children_to_parents_ordered, node_index_mapping

    def process_odb_gene_to_og(self, file_path: Union[str, Path],
                               node_index: Dict[str, int]) -> Dict[str, str]:
        """
        Process OrthoDB group to gene mappings.

        Args:
            file_path: Path to the OrthoDB-to-gene mapping file.
            node_index: Mapping of nodes to their indices (used for priority).

        Returns:
            Mapping from gene to its assigned OrthoDB group.
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"OrthoDB gene mapping file not found: {file_path}")

        with error_context("processing OrthoDB gene mappings"):
            gene_to_og: Dict[str, str] = {}

            try:
                with open(file_path, "r") as f:
                    for line in tqdm(f, desc="Loading OrthoDB-to-gene mapping"):
                        parts = line.strip().split("\t")
                        if len(parts) != 2:
                            logger.warning(f"Skipping malformed line: {line.strip()}")
                            continue
                        og, gene = parts

                        # Update mapping only if the new group has higher priority (lower index)
                        if gene in gene_to_og:
                            current_priority = node_index.get(gene_to_og[gene], float('inf'))
                            new_priority = node_index.get(og, float('inf'))
                            if new_priority > current_priority:
                                continue

                        gene_to_og[gene] = og
            except Exception as e:
                raise RuntimeError(f"Error reading gene mapping file: {e}")

            logger.info(f"Loaded {len(gene_to_og)} gene-to-OrthoDB mappings.")
            return gene_to_og


# Legacy functions for backward compatibility
def process_odb_graph(file_path: str) -> Tuple[Dict[str, List[str]], List[str], Dict[str, int]]:
    """Legacy function. Use OrthoDB_Processor.process_odb_graph for new code."""
    processor = OrthoDB_Processor(Config())
    return processor.process_odb_graph(file_path)


def process_odb_gene_to_og(file_path: str, node_index: Dict[str, int]) -> Dict[str, str]:
    """Legacy function. Use OrthoDB_Processor.process_odb_gene_to_og for new code."""
    processor = OrthoDB_Processor(Config())
    return processor.process_odb_gene_to_og(file_path, node_index)


@torch.no_grad()
def process_group_vectors_and_count(
        folder: str,
        gene_to_og: Dict[str, str],
        children_to_parents_ordered: List[str],
        parent_to_children: Dict[str, List[str]]
) -> Dict[str, Tuple[torch.tensor, float]]:
    """
    Processes embedding files from a given folder and aggregates group vectors and counts
    for each OrthoDB group. Additionally, propagates vectors up the hierarchy defined by
    the parent_to_children mapping.

    Args:
        folder (str): Path to the folder containing embedding files.
        gene_to_og (Dict[str, str]): Mapping from gene to OrthoDB group.
        children_to_parents_ordered (List[str]): Ordered list of nodes (children-to-parents).
        parent_to_children (Dict[str, List[str]]): Mapping from parent to its children.

    Returns:
        Dict[str, Tuple[torch.tensor, float]]: Mapping from group (OrthoDB or internal node) to a tuple of:
                                             (aggregated vector, count)
    """
    group_vectors_and_count: Dict[str, Tuple[torch.tensor, float]] = {}
    files = os.listdir(folder)
    logging.info(f"Processing {len(files)} files in {folder}...")

    for i, file in tqdm(enumerate(files), desc="Processing embedding files"):
        if i % 100 == 0 and i > 0:
            logging.info(f"Processed {i} files. Saving intermediate results...")
            with open("intermediate_group_vectors.pkl", "wb") as f:
                pickle.dump(group_vectors_and_count, f)
        file_path = os.path.join(folder, file)
        with open(file_path, "rb") as f:
            data = pickle.load(f)
            embeddings = data["embeddings"]
            labels = data["labels"]
            for label, vec in zip(labels, embeddings):
                og = gene_to_og.get(label, "unclassified")
                if og not in group_vectors_and_count:
                    group_vectors_and_count[og] = (vec.clone(), 1.0)
                else:
                    group_vec, count = group_vectors_and_count[og]
                    new_count = count + 1
                    # Running average update
                    group_vec = (count / new_count) * group_vec + (1 / new_count) * vec
                    group_vectors_and_count[og] = (group_vec, new_count)
        # clean a bit
        del data
        del embeddings
        del labels
    logging.info(f"Processed {len(group_vectors_and_count)} groups from embedding files.")

    # Propagate the group vectors up the hierarchy.
    # (Nodes are processed in children-to-parents order so that children are computed first.)
    for node in tqdm(children_to_parents_ordered, desc="Propagating group vectors"):
        # Skip if the node has no children.
        if node not in parent_to_children:
            continue

        children = parent_to_children[node]
        parent_data = group_vectors_and_count.get(node, (None, 0))
        parent_vec, parent_count = parent_data

        # Gather data from children.
        valid_children = []
        total_count = parent_count  # start with parent's own count
        for child in children:
            child_vec, child_count = group_vectors_and_count.get(child, (None, 0))
            if child_count > 0 and child_vec is not None:
                valid_children.append((child_vec, child_count))
                total_count += child_count

        if total_count == 0:
            # Nothing to propagate if neither the node nor its children have a vector.
            continue

        # Compute weighted average from children vectors.
        if valid_children:
            # Here we weight the average by the number of vectors contributing from each child.
            sum_children = sum(child_vec for child_vec, _ in valid_children)
            avg_children = sum_children / len(valid_children)
        else:
            avg_children = None

        # Combine parent's vector and children's average.
        if parent_count > 0 and parent_vec is not None:
            if avg_children is not None:
                # Weighted by the counts (or simply combine counts).
                combined_vec = (parent_count * parent_vec + len(valid_children) * avg_children) / (
                        parent_count + len(valid_children))
            else:
                combined_vec = parent_vec
        else:
            combined_vec = avg_children

        if combined_vec is not None:
            group_vectors_and_count[node] = (combined_vec, total_count)

    return group_vectors_and_count


def calculate_group_vectors(og_pair_file: str, og_to_gene_file: str, input_folder: str, out_file: str):
    """
    Calculate group vectors from the embeddings of genes in each group and propagate these vectors
    up the OrthoDB hierarchy.

    Args:
        og_pair_file (str): Path to the file containing OrthoDB pair (child-parent) mappings.
        og_to_gene_file (str): Path to the file containing OrthoDB-to-gene mappings.
        input_folder (str): Path to the folder containing embedding files.
        out_file (str): Path to the output file where the group vectors will be saved.
    """
    parent_to_children, children_to_parents_ordered, node_index = process_odb_graph(og_pair_file)
    gene_to_og = process_odb_gene_to_og(og_to_gene_file, node_index)
    group_vectors_and_count = process_group_vectors_and_count(
        input_folder, gene_to_og, children_to_parents_ordered, parent_to_children
    )

    with open(out_file, "wb") as f:
        pickle.dump(group_vectors_and_count, f)
    logging.info(f"Saved group vectors to {out_file}.")


def split_group_vectors_by_count(file_path: str, counts: List[int]):
    """
    Split the group vectors by count into multiple files based on the given count

    Args:
        file_path (str): Path to the input file containing group vectors.
        counts (List[int]): List of counts to split the group vectors by.
    """
    with open(file_path, "rb") as f:
        data = pickle.load(f)

    data_by_counts = [dict() for _ in range(len(counts) - 1)]
    for key, (vector, n) in tqdm(data.items(), desc="Splitting by count"):
        for i, count in enumerate(counts[:-1]):
            if count <= n < counts[i + 1]:
                data_by_counts[i][key] = (vector, n)
                break

    for i, data_ in tqdm(enumerate(data_by_counts), desc="Saving"):
        with open(f"{file_path[:-4]}_{counts[i]}.pkl", "wb") as f:
            pickle.dump(data_, f)


def group_to_group_arborescence(file_path: str, save_path: str):
    """
    Processes a tab-delimited file containing child-parent pairs and builds an arborescence for each child.

    Args:
        file_path (str): Path to the input file.
        save_path (str): Path to save the arborescence mapping.
    """
    child_to_parent: Dict[str, str] = {}

    logging.info(f"Loading OrthoDB-to-OrthoDB mapping from {file_path}...")
    with open(file_path, "r") as f:
        for line_number, line in enumerate(f, start=1):
            parts = line.strip().split("\t")
            if len(parts) != 2:
                logging.warning(f"Skipping malformed line {line_number}: {line.strip()}")
                continue
            child, parent = parts
            child_to_parent[child] = parent

    child_to_arborescence: Dict[str, Set[str]] = {}
    logging.info(f"Building arborescence for {len(child_to_parent)} nodes...")
    for child, parent in tqdm(child_to_parent.items(), desc="Building arborescence"):
        child_to_arborescence[child] = set()
        current_cursor = child
        while current_cursor in child_to_parent:
            child_to_arborescence[child].add(current_cursor)
            current_cursor = child_to_parent[current_cursor]
    logging.info(f"Built arborescence for {len(child_to_arborescence)} nodes.")

    with open(save_path, "wb") as f:
        pickle.dump(child_to_arborescence, f)


def get_taxonomic_balance(file_path: str) -> Dict[str, float]:
    """
    Given a file path for a file describing the taxonomic relationships (as in odb12v0_level2species.tab),
    this function builds a tree (as a dictionary) where each key is a taxon (as an integer, except for the
    universal root which is labeled 'root') and its value is a tuple (children, species). "children" is a set
    of child taxonomic nodes and "species" is a set of species (terminal nodes) that occur directly at that
    taxonomic level.

    Then, it recursively computes a balanced weight for each species.

    The weighting works as follows:
      - At any taxon node, we consider its branches to be:
          * Each child taxon (each gets one branch)
          * The species branch (if any species are attached) – this branch is considered as one branch,
            but then its weight is divided equally among the species within it.
      - At each node, the parent's weight is split equally among its branches.
      - The final weight for a species is the product of the branch probabilities along the path from
        the root (which has weight 1) down to that species.

    Returns:
        A dict mapping species (str) to their final weight (float).
    """

    logging.info(f"Starting to build taxonomic tree from file: {file_path}")

    # --- Step 1: Build the tree ---
    # Our tree: key = taxon (int or 'root'), value = (set(child_taxa), set(species))
    tree: Dict[Any, Tuple[Set, Set]] = {'root': (set(), set())}
    line_count = 0

    try:
        with open(file_path, "r") as f:
            for line in f:
                line_count += 1
                parts = line.strip().split("\t")
                if len(parts) < 4:
                    logging.debug(f"Skipping malformed line {line_count}: {line.strip()}")
                    continue  # skip any malformed line

                # The second column contains the species ID.
                spec = parts[1].strip()

                # The fourth column contains the ordered lineage in the form: {2,1224,1236,2315472}
                lineage_str = parts[3].strip()
                if lineage_str.startswith("{") and lineage_str.endswith("}"):
                    lineage_str = lineage_str[1:-1]
                try:
                    lineage = list(map(int, lineage_str.split(",")))
                except Exception as e:
                    logging.error(f"Conversion error on line {line_count}: {line.strip()} - {e}")
                    continue  # skip lines with conversion errors

                # If the lineage is empty, assign species directly under the root.
                if not lineage:
                    tree['root'][1].add(spec)
                    logging.debug(f"Added species {spec} directly under root (empty lineage) at line {line_count}")
                    continue

                # Register the first taxon as a child of the root.
                tree['root'][0].add(lineage[0])
                logging.debug(f"Added taxon {lineage[0]} as child of root at line {line_count}")

                # Process the lineage: each adjacent pair defines a parent-child relationship.
                for j in range(len(lineage) - 1):
                    parent = lineage[j]
                    child = lineage[j + 1]
                    if parent not in tree:
                        tree[parent] = (set(), set())
                        logging.debug(f"Initialized taxon {parent} in tree at line {line_count}")
                    tree[parent][0].add(child)
                    logging.debug(f"Added taxon {child} as child of {parent} at line {line_count}")

                # The last element in the lineage is where the species attaches.
                last_level = lineage[-1]
                if last_level not in tree:
                    tree[last_level] = (set(), set())
                    logging.debug(f"Initialized taxon {last_level} in tree at line {line_count}")
                tree[last_level][1].add(spec)
                logging.debug(f"Added species {spec} to taxon {last_level} at line {line_count}")
    except FileNotFoundError as e:
        logging.error(f"File not found: {file_path}")
        raise e

    logging.info(f"Finished building tree with {line_count} lines processed.")

    # --- Step 2: Compute balanced taxonomic weights ---
    def _recurse(taxon: Any = "root", weight: float = 1.0) -> Dict[str, float]:
        """
        For a given taxon (integer or 'root') and an incoming weight,
        return a dictionary mapping species (str) to their computed weight.
        """
        species_weights: Dict[str, float] = {}
        children, species = tree[taxon]

        # Consider each child taxon as one branch and the species branch (if present) as one branch.
        n_branches = len(children) + (1 if species else 0)
        if n_branches == 0:
            logging.debug(f"Taxon {taxon} is a dead-end with no branches.")
            return species_weights

        branch_weight = weight / n_branches
        logging.debug(f"At taxon {taxon}: weight={weight}, branches={n_branches}, branch_weight={branch_weight}")

        # Process each child taxon branch.
        for child in children:
            child_species_weights = _recurse(child, branch_weight)
            for sp, sp_weight in child_species_weights.items():
                if sp in species_weights:
                    logging.warning(f"Species {sp} appears in multiple branches; summing weights.")
                species_weights[sp] = species_weights.get(sp, 0) + sp_weight

        # Process the species branch, if present.
        if species:
            n_species = len(species)
            species_branch_weight = branch_weight / n_species
            for sp in species:
                species_weights[sp] = species_branch_weight
                logging.debug(f"At taxon {taxon}: assigned weight {species_branch_weight} to species {sp}")

        return species_weights

    species_weights = _recurse()
    total_weight = sum(species_weights.values())
    logging.info(f"Sum of species weights: {total_weight}")
    return species_weights


def build_individual_embeddings_files(folder: str, save_folder: str, odb_file_path: str):
    # if folder exists ask the authorization to delete it if not given then exit
    if os.path.exists(save_folder):
        logging.warning(f"Folder {save_folder} already exists. Do you want to delete it? (y/n)")
        answer = input()
        if answer.lower() != "y":
            logging.error("Exiting...")
            return
        shutil.rmtree(save_folder)
    os.makedirs(save_folder)

    gene_to_og: Dict[str, List[str]] = {}
    logging.info(f"Loading OrthoDB-to-gene mapping from {odb_file_path}...")
    with open(odb_file_path, "r") as f:
        for line in tqdm(f, desc="Loading OrthoDB-to-gene mapping"):
            parts = line.strip().split("\t")
            if len(parts) != 2:
                logging.warning(f"Skipping malformed line: {line.strip()}")
                continue
            og, gene = parts
            if gene not in gene_to_og:
                gene_to_og[gene] = []
            gene_to_og[gene].append(og)
    logging.info(f"Loaded {len(gene_to_og)} gene-to-OrthoDB mappings.")

    for part in range(64):
        by_spec_embeddings: Dict[
            str, Tuple[
                List[str], List[List[str]], List[torch.Tensor]]] = {}  # key: taxid, value: (label, group, embedding)
        logging.info(f"Processing part {part}...")
        for subpart in range(1, 27):
            with open(f"{folder}/odb12v0_aa.fasta.part{part}.pkl.{subpart}", "rb") as f:
                data = pickle.load(f)
                labels = data["labels"]
                embeddings = data["embeddings"]
                for label, emb in zip(labels, embeddings):
                    taxid = label.split(":")[0]
                    if taxid not in by_spec_embeddings:
                        by_spec_embeddings[taxid] = ([], [], [])
                    by_spec_embeddings[taxid][0].append(label)
                    by_spec_embeddings[taxid][1].append(gene_to_og.get(label, []))
                    by_spec_embeddings[taxid][2].append(emb.clone())

        count_reloads = 0
        for taxid, embeddings in by_spec_embeddings.items():
            existing_embeddings = []
            existing_labels = []
            existing_groups = []
            if os.path.exists(f"{save_folder}/{taxid}.pkl"):
                with open(f"{save_folder}/{taxid}.pkl", "rb") as f:
                    data = pickle.load(f)
                    existing_labels = data[1]
                    existing_groups = data[2]
                    existing_embeddings = data[3]
                count_reloads += 1
            existing_labels.extend(embeddings[0])
            existing_groups.extend(embeddings[1])
            existing_embeddings.extend(embeddings[2])
            with open(f"{save_folder}/{taxid}.pkl", "wb") as f:
                pickle.dump((taxid, existing_labels, existing_groups, existing_embeddings), f)
        logging.info(f"Processed part {part} (with {count_reloads} reloads).")


def create_shard(shard_path: str, elements: List[Union[str, Path]]):
    """Create a tar file containing elements and remove original files."""
    try:
        with tarfile.open(shard_path, "w") as tar:
            for element in elements:
                file_path = str(element)
                tar.add(file_path.split('/')[-1], arcname=os.path.basename(file_path))
                os.remove(file_path)
    except Exception as e:
        print(f"Error creating shard at {shard_path}: {e}")
        raise


def convert_to_shards(folder: str, shardsize: int = 500, minsize_kb: int = 750):
    # List all files in the folder
    files = [f for f in os.listdir(folder) if f.endswith(".pkl")]
    shuffle(files)

    # create a folder dump file for the files that are too small
    dump_folder = os.path.join(folder, "dump")
    # make the dump folder if it does not exist
    if not os.path.exists(dump_folder):
        os.makedirs(dump_folder)

    # Load the data from each file and save it in the save folder
    next_shard = []
    shard_number = 1
    for file in files:
        size = os.path.getsize(os.path.join(folder, file))
        if size < minsize_kb * 1024:
            shutil.move(os.path.join(folder, file), os.path.join(dump_folder, file))
            continue
        next_shard.append(os.path.join(folder, file))
        if len(next_shard) >= shardsize:
            with tarfile.open(os.path.join(folder, f"shard_{shard_number}.tar"), "w") as tar:
                for element in next_shard:
                    tar.add(element, arcname=os.path.basename(element))
                    os.remove(element)
            logging.info(f"Created shard {shard_number}/{1+len(files)//shardsize}")
            next_shard = []
            shard_number += 1
    with tarfile.open(os.path.join(folder, f"shard_{shard_number}.tar"), "w") as tar:
        for element in next_shard:
            tar.add(element, arcname=os.path.basename(element))
            os.remove(element)
    logging.info(f"Created shard {shard_number}/{1+len(files)//shardsize}")


def filter_eukaryotes_from_shards(shards_folder: str, eukaryota_file: str, output_folder: str):
    """
    Filters shard tar files in a folder to keep only files corresponding to eukaryote genomes.

    This function reads eukaryote species from `eukaryota_file` (one per line in the format "taxid_extra",
    e.g. "9337_0"), extracts the taxid portion, and then for each shard (.tar archive) found in `shards_folder`,
    creates a new tar file (saved in `output_folder`) containing only those files whose basename (taxid before file
    extension) is in the eukaryote set.

    Args:
        shards_folder (str): Path to the folder containing shard .tar files.
        eukaryota_file (str): Path to the file with eukaryote species (one per line, e.g. "9337_0").
        output_folder (str): Path where the filtered shard files will be saved.
    """
    import os
    import tarfile
    from pathlib import Path

    # Load eukaryote taxids from eukaryota_file (extract taxid before the underscore)
    eukaryote_taxids = set()
    with open(eukaryota_file, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                taxid = line.strip()
                eukaryote_taxids.add(taxid)
    logging.info(f"Loaded {len(eukaryote_taxids)} eukaryote taxids from {eukaryota_file}.")

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # List all .tar shard files in the shards_folder
    shard_files = [f for f in os.listdir(shards_folder) if f.endswith(".tar")]
    logging.info(f"Found {len(shard_files)} shard files in {shards_folder}.")

    for shard_file in shard_files:
        shard_path = os.path.join(shards_folder, shard_file)
        output_shard_path = os.path.join(output_folder, shard_file.replace(".tar", "_euk.tar"))
        if Path(output_shard_path).exists():
            logging.info(f"Filtered shard already exists: {output_shard_path}; skipping.")
            continue
        kept_members = []

        # Open the shard tar file to scan its members.
        with tarfile.open(shard_path, "r") as tar:
            members = tar.getmembers()
            for member in members:
                # Expect member names like "9337.pkl"; extract the taxid before the extension.
                base_name = os.path.basename(member.name)
                taxid, _ = os.path.splitext(base_name)
                if taxid in eukaryote_taxids:
                    kept_members.append(member)

        if kept_members:
            # Create a new tar file with only eukaryote files.
            with tarfile.open(output_shard_path, "w") as out_tar:
                with tarfile.open(shard_path, "r") as tar:
                    for member in kept_members:
                        # Extract the file object from the original tar archive and add it to the new archive.
                        fileobj = tar.extractfile(member)
                        if fileobj is not None:
                            out_tar.addfile(member, fileobj)
            logging.info(f"Created filtered shard: {output_shard_path} with {len(kept_members)} files.")
        else:
            logging.info(f"No eukaryote files found in {shard_path}; skipping creation of filtered shard.")


class EncodingPipeline:
    """Main pipeline orchestrator for OrthoDB dataset encoding."""

    def __init__(self, config: Config):
        self.config = config
        self.download_manager = DownloadManager(config)
        self.fasta_splitter = FastaSplitter(config)
        self.encoder = SequenceEncoder(config)
        self.orthodb_processor = OrthoDB_Processor(config)

    def run_step(self, step_name: str, func, *args, **kwargs):
        """Run a pipeline step with consistent logging and error handling."""
        try:
            with error_context(f"Pipeline step: {step_name}"):
                result = func(*args, **kwargs)
                logger.info(f"✅ Completed step: {step_name}")
                return result
        except Exception as e:
            logger.error(f"❌ Failed step: {step_name} - {e}")
            raise

    def download_data(self):
        """Step 1: Download OrthoDB dataset."""
        self.run_step("Download OrthoDB data", self.download_manager.download_orthodb_data)

    def split_fasta(self, fasta_file: str):
        """Step 2: Split large FASTA file into smaller parts."""
        return self.run_step(
            "Split FASTA file",
            self.fasta_splitter.split_fasta,
            fasta_file
        )

    def encode_sequences(self, fasta_file: str, output_path: str):
        """Step 3: Encode sequences using ESM-C."""
        return self.run_step(
            "Encode sequences",
            self.encoder.encode_dataset,
            fasta_file, output_path
        )

    def calculate_group_vectors(self, og_pair_file: str, og_to_gene_file: str,
                                input_folder: str, output_file: str):
        """Step 4: Calculate group vectors and propagate up hierarchy."""
        def _calculate():
            parent_to_children, children_to_parents_ordered, node_index = \
                self.orthodb_processor.process_odb_graph(og_pair_file)
            gene_to_og = self.orthodb_processor.process_odb_gene_to_og(og_to_gene_file, node_index)

            # Use the existing function for group vector processing
            group_vectors_and_count = process_group_vectors_and_count(
                input_folder, gene_to_og, children_to_parents_ordered, parent_to_children
            )

            with open(output_file, "wb") as f:
                pickle.dump(group_vectors_and_count, f)
            logger.info(f"Saved group vectors to {output_file}")

        self.run_step("Calculate group vectors", _calculate)


def create_argument_parser():
    """Create command-line argument parser."""
    import argparse

    parser = argparse.ArgumentParser(
        description="ProteomeLM Dataset Encoding Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""

Examples:
  # Download OrthoDB data
  python encode_dataset.py download --save-path data/orthodb12_raw

  # Split FASTA file
  python encode_dataset.py split --input data/sequences.fasta --parts 64

  # Encode sequences
  python encode_dataset.py encode --input data/sequences.fasta --output embeddings.pt

  # Run full pipeline
  python encode_dataset.py pipeline --config config.yaml
        """
    )

    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging level")
    parser.add_argument("--log-file", help="Log file path")
    parser.add_argument("--device", default="cuda:0", help="Device for model inference")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Download command
    download_parser = subparsers.add_parser("download", help="Download OrthoDB data")
    download_parser.add_argument("--save-path", default="data/orthodb12_raw", help="Path to save downloaded data")

    # Split command
    split_parser = subparsers.add_parser("split", help="Split FASTA file")
    split_parser.add_argument("--input", required=True, help="Input FASTA file")
    split_parser.add_argument("--parts", type=int, default=64, help="Number of parts")

    # Encode command
    encode_parser = subparsers.add_parser("encode", help="Encode sequences")
    encode_parser.add_argument("--input", required=True, help="Input FASTA file")
    encode_parser.add_argument("--output", required=True, help="Output pickle file")
    encode_parser.add_argument("--max-tokens", type=int, default=60000, help="Maximum tokens per batch")

    # Pipeline command
    pipeline_parser = subparsers.add_parser("pipeline", help="Run full pipeline")
    pipeline_parser.add_argument("--config", help="Configuration file (optional)")
    pipeline_parser.add_argument("--steps", nargs="+",
                                 choices=["download", "split", "encode", "group_vectors"],
                                 help="Pipeline steps to run (default: all)")

    return parser


def main():
    """Main entry point with command-line interface."""
    parser = create_argument_parser()
    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level, args.log_file)

    # Create configuration
    config = Config(device=args.device)

    if args.command == "download":
        config.save_path = args.save_path
        pipeline = EncodingPipeline(config)
        pipeline.download_data()

    elif args.command == "split":
        config.num_fasta_parts = args.parts
        pipeline = EncodingPipeline(config)
        pipeline.split_fasta(args.input)

    elif args.command == "encode":
        config.max_tokens_per_batch = args.max_tokens
        pipeline = EncodingPipeline(config)
        pipeline.encode_sequences(args.input, args.output)

    elif args.command == "pipeline":
        pipeline = EncodingPipeline(config)
        steps = args.steps or ["download", "split", "encode", "group_vectors"]

        logger.info(f"Running pipeline steps: {steps}")

        if "download" in steps:
            pipeline.download_data()

        # Additional pipeline steps would be implemented here
        logger.info("Pipeline completed successfully!")
    else:
        parser.print_help()


if __name__ == "__main__":
    # Check if being run as script
    import sys
    if len(sys.argv) > 1:
        main()
    else:
        config = Config()
        pipeline = EncodingPipeline(config)
        # Original pipeline steps (commented out for safety)
        # Uncomment the steps you want to run:
        # 1. Download the OrthoDB dataset
        pipeline.download_data()

        # 2. Split the big fasta file into smaller parts for easy distribution
        big_fasta_file = "data/orthodb12_raw/odb12v0_aa.fasta"
        pipeline.split_fasta(big_fasta_file)

        # 3. Encode the dataset
        pipeline.encoder.load_model()
        fasta_file = "path/to/your/fasta/file"
        output_pickle = "path/to/your/output"
        pipeline.encode_sequences(fasta_file, output_pickle)

        # 4. Calculate group vectors
        og_pair_file = "data/orthodb12_raw/odb12v0_OG_pairs.tab"
        og_to_gene_file = "data/orthodb12_raw/odb12v0_OG2genes.tab"
        input_folder = "data/orthodb12_raw/esmc"
        out_file = "data/orthodb12_raw/group_vectors.pkl"
        pipeline.calculate_group_vectors(og_pair_file, og_to_gene_file, input_folder, out_file)
