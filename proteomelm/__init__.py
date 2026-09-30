"""ProteomeLM: A proteome-scale language model for protein analysis."""

__version__ = "1.0.0"

from .dataloaders import DataCollatorForProteomeLM, ProteomeLMDataset, get_shards_dataset
from .modeling_proteomelm import (
    ProteomeLMConfig,
    ProteomeLMForMaskedLM,
    ProteomeLMMaskedLMOutput,
    ProteomeLMModel,
)

__all__ = [
    "ProteomeLMConfig",
    "ProteomeLMForMaskedLM",
    "ProteomeLMModel",
    "ProteomeLMMaskedLMOutput",
    "ProteomeLMTrainer",
    "DataCollatorForProteomeLM",
    "ProteomeLMDataset",
    "get_shards_dataset",
]


def __getattr__(name):
    """Lazy import of the trainer (pulls in psutil and the HF Trainer stack)."""
    if name == "ProteomeLMTrainer":
        from .trainer import ProteomeLMTrainer
        return ProteomeLMTrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
