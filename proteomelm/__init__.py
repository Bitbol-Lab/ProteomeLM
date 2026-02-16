"""ProteomeLM: A proteome-scale language model for protein analysis."""

__version__ = "1.0.0"

from .dataloaders import *
from .modeling_proteomelm import *
# Lazy import to avoid wandb dependency issues on module load
# from .train import *
try:
    from .encode_dataset import *
except ModuleNotFoundError:
    # ESM may not be available
    pass

__all__ = [
    "ProteomeLMConfig",
    "ProteomeLMForMaskedLM",
    "ProteomeLMTrainer",
    "DataCollatorForProteomeLM",
    "get_shards_dataset",
]

def __getattr__(name):
    """Lazy import for train module to avoid wandb dependency on module load."""
    if name == "ProteomeLMTrainer":
        from .train import ProteomeLMTrainer
        return ProteomeLMTrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
