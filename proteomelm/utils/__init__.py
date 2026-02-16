
"""
Utility functions for ProteomeLM.

This module provides various utility functions for model training,
evaluation, and data processing.
"""

import logging
from typing import Dict, Any, Optional, Union


from transformers import (
    get_cosine_schedule_with_warmup,
    get_constant_schedule_with_warmup
)

from .embedding import average_representation

logger = logging.getLogger(__name__)


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

