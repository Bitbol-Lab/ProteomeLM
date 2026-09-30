
"""
Utility functions for ProteomeLM.

This module provides various utility functions for model training,
evaluation, and data processing.
"""

import logging
import sys
from typing import Optional, Union

logger = logging.getLogger(__name__)


def setup_logging(level: Union[str, int] = logging.INFO,
                  format_string: Optional[str] = None,
                  include_timestamp: bool = True,
                  log_file: Optional[str] = None) -> None:
    """
    Set up logging configuration for ProteomeLM.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR), as a name or an int
        format_string: Custom format string for log messages
        include_timestamp: Whether to include timestamp in log messages
        log_file: Optional file to append log messages to (in addition to stdout)
    """
    if format_string is None:
        if include_timestamp:
            format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        else:
            format_string = "%(name)s - %(levelname)s - %(message)s"

    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        handlers.append(logging.FileHandler(log_file, mode="a"))

    logging.basicConfig(
        level=level.upper() if isinstance(level, str) else level,
        format=format_string,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
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
