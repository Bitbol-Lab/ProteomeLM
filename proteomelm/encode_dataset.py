"""
ProteomeLM Dataset Encoding Pipeline

This module provides functionality to download, process, and encode protein sequences
from OrthoDB using ESM-C embeddings, with hierarchical group vector computation.

Pipeline steps (steps 1-3 are exposed as CLI subcommands, see ``--help``):
  1. ``download``: OrthoDB OG2genes / OG_pairs / aa.fasta (``DownloadManager``)
  2. ``split``:    split the FASTA into parts sorted by length (``FastaSplitter``)
  3. ``encode``:   mean-pooled ESM-C embeddings per part (``SequenceEncoder``)
  4. group vectors from OG_pairs + OG2genes + the encoded parts
     (``EncodingPipeline.calculate_group_vectors``), then
     ``split_group_vectors_by_count`` into the group_vectors_{t}.pkl files (t in
     ``utils.proteome.ORTHODB_GROUP_SIZE_THRESHOLDS``) read by the training dataloader
  5. per-species files and tar shards (``build_individual_embeddings_files``,
     ``convert_to_shards``)
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
from typing import Union, List, Tuple, Dict, Optional
from collections import defaultdict
from dataclasses import dataclass
from contextlib import contextmanager

import torch
from Bio import SeqIO
from tqdm import tqdm
from esm.models.esmc import ESMC

from .utils import setup_logging
from .utils.embedding import average_representation


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
        # NOTE: local filenames keep the "odb12v0_" prefix that later steps hard-code
        # (e.g. build_individual_embeddings_files), whatever config.orthodb_version is
        # downloaded.
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
  python -m proteomelm.encode_dataset download --save-path data/orthodb12_raw

  # Split FASTA file
  python -m proteomelm.encode_dataset split --input data/sequences.fasta --parts 64

  # Encode sequences
  python -m proteomelm.encode_dataset encode --input data/sequences.fasta --output embeddings.pt
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

    return parser


def main():
    """Main entry point with command-line interface."""
    parser = create_argument_parser()
    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level, log_file=args.log_file)

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

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
