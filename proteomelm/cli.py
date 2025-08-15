"""Command-line interface for ProteomeLM."""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any

import torch
import yaml

from .utils import setup_logging
from .train import run_training

logger = logging.getLogger(__name__)


def setup_distributed(use_one_gpu: str = "-1") -> int:
    """Setup distributed training configuration."""
    if int(use_one_gpu) >= 0:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = use_one_gpu
        rank = 0
    else:
        import torch.distributed as dist
        dist.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
        rank = int(os.environ['LOCAL_RANK'])
    return rank


def load_config(config_paths: List[str]) -> Dict[str, Any]:
    """Load and merge configuration files."""
    config = {}
    for path in config_paths:
        if not Path(path).exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path, "r", encoding="utf-8") as file:
            file_config = yaml.safe_load(file)
            config.update(file_config)
            logger.info(f"Loaded config from {path}")

    return config


def validate_config(config: Dict[str, Any]) -> None:
    """Validate configuration parameters."""
    required_keys = [
        "batch_size", "learning_rate", "num_epochs",
        "output_dir", "namedir", "dim", "n_layers", "n_heads"
    ]

    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise ValueError(f"Missing required configuration keys: {missing_keys}")

    # Validate ranges
    if config["batch_size"] <= 0:
        raise ValueError("batch_size must be positive")
    if config["learning_rate"] <= 0:
        raise ValueError("learning_rate must be positive")
    if config["num_epochs"] <= 0:
        raise ValueError("num_epochs must be positive")


def train_cli():
    """Command-line interface for training ProteomeLM."""
    parser = argparse.ArgumentParser(
        description="Train ProteomeLM models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--config",
        type=str,
        nargs='+',
        default=["configs/proteomelm.yaml"],
        help="Path(s) to configuration YAML file(s)"
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help="Path or Hugging Face model ID to fine-tune from"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the latest checkpoint"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate the configuration without training"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(level=getattr(logging, args.log_level))

    try:
        # Load and validate configuration
        config = load_config(args.config)
        validate_config(config)

        if args.validate_only:
            logger.info("Configuration validation successful!")
            return

        # Import training function and run
        trainer = run_training(config, args.pretrained, args.resume)

        logger.info("Training completed successfully!")
        return trainer

    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # This allows the CLI to be run as python -m proteomelm.cli
    if len(sys.argv) < 2:
        print("Usage: python -m proteomelm.cli [train|encode|predict] [options]")
        sys.exit(1)

    command = sys.argv[1]
    sys.argv = [sys.argv[0]] + sys.argv[2:]  # Remove the command from argv

    if command == "train":
        train_cli()
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
