"""
Configuration settings for PPI extraction experiments.
"""
import os
from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path

# Single override point for where PPI benchmark/experiment data lives. Defaults
# to Cyril's cluster path (unchanged behavior for existing runs); set
# PROTEOMELM_DATA_ROOT to point these at a different machine's data layout.
DATA_ROOT = Path(os.environ.get("PROTEOMELM_DATA_ROOT", "data"))


@dataclass
class ExtractionConfig:
    """Configuration for PPI feature extraction."""
    checkpoint: str
    env_dir: Path
    fasta_file: str
    encoded_genome_file: Optional[Path] = None
    save_path: Optional[Path] = None
    include_attention: bool = True
    include_all_hidden_states: bool = True
    reload_if_possible: bool = True
    esm_device: str = "cuda:0"
    proteomelm_device: str = "cpu"
    orthodb_db_path: Optional[Path] = None
    orthodb_tsv_path: Optional[Path] = None
    orthodb_min_group_size: int = 10
    orthodb_fetch_online: bool = True


@dataclass
class ExperimentConfig:
    """Checkpoints to benchmark (experiments/ppi_benchmarks): ``base_path/checkpoint-<n>`` for each
    number, or ``base_path`` itself (a local directory or Hugging Face id) when the list is empty."""
    model_name: str
    base_path: Path
    checkpoint_numbers: List[int]
    reload_if_possible: bool = True


@dataclass
class DatasetConfig:
    """Configuration for dataset-specific settings."""
    name: str
    base_dir: Path
    fasta_file: str
    experiment_name: str
    orthodb_db_path: Optional[Path] = None
    orthodb_tsv_path: Optional[Path] = None
    orthodb_min_group_size: int = 10
    orthodb_fetch_online: bool = True

    @property
    def env_dir(self) -> Path:
        return self.base_dir

    @property
    def encoded_genome_file(self) -> Path:
        return self.env_dir / f"dump_dict_esm_{self.experiment_name}.pt"

    @property
    def save_path(self) -> Path:
        return self.env_dir / "dump_dict.pkl"

    @property
    def results_path(self) -> Path:
        return self.env_dir / "checkpoint_screening.csv"


# Predefined dataset configurations
BERNETT_CONFIG = DatasetConfig(
    name="bernett",
    base_dir=DATA_ROOT / "bernett",
    fasta_file="human_gold.faa",
    experiment_name="bernett"
)

DSCRIPT_SPECIES = ["human", "ecoli", "yeast", "fly", "worm", "mouse"]


def get_benchmark_config(species: str) -> DatasetConfig:
    """Get configuration for a specific DScript species."""
    return DatasetConfig(
        name=f"benchmark_{species}",
        base_dir=DATA_ROOT / "benchmark" / species,
        fasta_file=f"{species}.faa",
        experiment_name="benchmark"
    )


def get_dscript_config(species: str) -> DatasetConfig:
    """Get configuration for a specific DScript species."""
    return DatasetConfig(
        name=f"dscript_{species}",
        base_dir=DATA_ROOT / "dscript" / species,
        fasta_file=f"{species}.faa",
        experiment_name="dscript"
    )
