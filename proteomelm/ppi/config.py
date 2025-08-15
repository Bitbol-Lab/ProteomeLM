"""
Configuration settings for PPI extraction experiments.
"""
from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path


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


@dataclass
class ExperimentConfig:
    """Configuration for running experiments across multiple checkpoints."""
    model_name: str
    base_path: Path
    checkpoint_numbers: List[int]
    corrector: int = 0
    reload_if_possible: bool = True


@dataclass
class DatasetConfig:
    """Configuration for dataset-specific settings."""
    name: str
    base_dir: Path
    fasta_file: str
    experiment_name: str

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
    base_dir=Path("/data2/malbrank/proteomelm/bernett/"),
    fasta_file="human_goldstandard.faa",
    experiment_name="goldstandard"
)

DSCRIPT_SPECIES = ["human", "ecoli", "yeast", "fly", "worm", "mouse"]


def get_dscript_config(species: str) -> DatasetConfig:
    """Get configuration for a specific DScript species."""
    return DatasetConfig(
        name=f"dscript_{species}",
        base_dir=Path(f"/data2/malbrank/proteomelm/dscript/{species}/"),
        fasta_file=f"{species}.faa",
        experiment_name="dscript"
    )
