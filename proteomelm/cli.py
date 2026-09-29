"""Command-line interface for ProteomeLM."""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any

import yaml

from .train import run_training, validate_config
from .utils import setup_logging

logger = logging.getLogger(__name__)


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
        default=["configs/pretraining/proteomelm.yaml"],
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

    # Setup logging (stdout + training.log in the working directory)
    setup_logging(level=getattr(logging, args.log_level), log_file="training.log")

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
        print("Usage: python -m proteomelm.cli train [options]")
        sys.exit(1)

    command = sys.argv[1]
    sys.argv = [sys.argv[0]] + sys.argv[2:]  # Remove the command from argv

    if command == "train":
        train_cli()
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
