"""
Configuration validation and management for ProteomeLM.

This module provides utilities for validating and managing training configurations.
"""

import yaml
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

logger = logging.getLogger(__name__)


class ConfigManager:
    """Manager for ProteomeLM configuration files."""

    DEFAULT_CONFIG = {
        # Model architecture
        "dim": 512,
        "n_layers": 6,
        "n_heads": 8,
        "input_size": 1152,
        "max_length": 16000,

        # Training settings
        "batch_size": 16,
        "learning_rate": 0.0003,
        "num_epochs": 100,
        "warmup_steps": 500,
        "max_grad_norm": 1.0,
        "gradient_accumulation_steps": 1,

        # Optimization
        "beta1": 0.9,
        "beta2": 0.999,
        "weight_decay": 0.01,
        "scheduler": "cosine",
        "loss_choice": "polar",

        # Data processing
        "min_taxid_size": 200,
        "mask_fraction": 0.5,
        "dataloader_num_workers": 0,

        # Output and logging
        "output_dir": "output/",
        "namedir": "proteomelm_experiment",
        "logging_steps": 10,
        "save_epochs": 10,
        "wandb_project": None,

        # Hardware
        "use_one_gpu": "-1",
    }

    REQUIRED_FIELDS = [
        "dim", "n_layers", "n_heads", "batch_size",
        "learning_rate", "num_epochs", "output_dir", "namedir"
    ]

    def __init__(self, config_path: Optional[Union[str, List[str]]] = None):
        """
        Initialize configuration manager.

        Args:
            config_path: Path(s) to configuration file(s)
        """
        self.config = self.DEFAULT_CONFIG.copy()

        if config_path:
            self.load_config(config_path)

    def load_config(self, config_paths: Union[str, List[str]]) -> None:
        """
        Load configuration from file(s).

        Args:
            config_paths: Path(s) to configuration file(s)
        """
        if isinstance(config_paths, str):
            config_paths = [config_paths]

        for path in config_paths:
            path_obj = Path(path)
            if not path_obj.exists():
                logger.warning(f"Configuration file not found: {path}")
                continue

            try:
                with open(path_obj, 'r', encoding='utf-8') as f:
                    file_config = yaml.safe_load(f)
                    if file_config:
                        self.config.update(file_config)
                        logger.info(f"Loaded configuration from: {path}")
            except Exception as e:
                logger.error(f"Failed to load config from {path}: {e}")
                raise

    def validate(self) -> List[str]:
        """
        Validate configuration.

        Returns:
            List of validation error messages
        """
        errors = []

        # Check required fields
        for field in self.REQUIRED_FIELDS:
            if field not in self.config:
                errors.append(f"Missing required field: {field}")

        # Validate value ranges
        if self.config.get("batch_size", 1) <= 0:
            errors.append("batch_size must be positive")

        if self.config.get("learning_rate", 1) <= 0:
            errors.append("learning_rate must be positive")

        if self.config.get("num_epochs", 1) <= 0:
            errors.append("num_epochs must be positive")

        if self.config.get("n_layers", 1) <= 0:
            errors.append("n_layers must be positive")

        if self.config.get("n_heads", 1) <= 0:
            errors.append("n_heads must be positive")

        if self.config.get("dim", 1) <= 0:
            errors.append("dim must be positive")

        # Check if dim is divisible by n_heads
        dim = self.config.get("dim", 1)
        n_heads = self.config.get("n_heads", 1)
        if dim % n_heads != 0:
            errors.append("dim must be divisible by n_heads")

        # Validate scheduler choice
        valid_schedulers = ["cosine", "constant", "linear"]
        scheduler = self.config.get("scheduler", "cosine")
        if scheduler not in valid_schedulers:
            errors.append(f"scheduler must be one of: {valid_schedulers}")

        # Validate loss choice
        valid_losses = ["mse", "cosine", "polar"]
        loss_choice = self.config.get("loss_choice", "mse")
        if loss_choice not in valid_losses:
            errors.append(f"loss_choice must be one of: {valid_losses}")

        return errors

    def save_config(self, output_path: str) -> None:
        """
        Save configuration to file.

        Args:
            output_path: Path to save configuration
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, indent=2)
            logger.info(f"Configuration saved to: {output_path}")
        except Exception as e:
            logger.error(f"Failed to save config to {output_path}: {e}")
            raise

    def get_config(self) -> Dict[str, Any]:
        """Get the current configuration."""
        return self.config.copy()

    def update_config(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration with new values.

        Args:
            updates: Dictionary of updates to apply
        """
        self.config.update(updates)
        logger.info(f"Configuration updated with {len(updates)} changes")

    def create_experiment_config(self,
                                 experiment_name: str,
                                 model_size: str = "S",
                                 **overrides) -> Dict[str, Any]:
        """
        Create configuration for a specific experiment.

        Args:
            experiment_name: Name of the experiment
            model_size: Model size (XS, S, M, L)
            **overrides: Additional configuration overrides

        Returns:
            Experiment configuration
        """
        # Model size presets
        size_configs = {
            "XS": {"dim": 256, "n_layers": 6, "n_heads": 8},
            "S": {"dim": 512, "n_layers": 6, "n_heads": 8},
            "M": {"dim": 768, "n_layers": 12, "n_heads": 12},
            "L": {"dim": 1024, "n_layers": 24, "n_heads": 16},
        }

        if model_size not in size_configs:
            raise ValueError(f"Invalid model size: {model_size}. Choose from: {list(size_configs.keys())}")

        # Start with current config
        exp_config = self.config.copy()

        # Apply model size config
        exp_config.update(size_configs[model_size])

        # Set experiment name
        exp_config["namedir"] = experiment_name

        # Apply any additional overrides
        exp_config.update(overrides)

        return exp_config


def validate_config_file(config_path: str) -> bool:
    """
    Validate a configuration file.

    Args:
        config_path: Path to configuration file

    Returns:
        True if valid, False otherwise
    """
    try:
        manager = ConfigManager(config_path)
        errors = manager.validate()

        if errors:
            logger.error("Configuration validation failed:")
            for error in errors:
                logger.error(f"  - {error}")
            return False

        logger.info("Configuration validation passed")
        return True

    except Exception as e:
        logger.error(f"Failed to validate config: {e}")
        return False


def create_default_config(output_path: str) -> None:
    """
    Create a default configuration file.

    Args:
        output_path: Path to save the default configuration
    """
    manager = ConfigManager()
    manager.save_config(output_path)
    logger.info(f"Default configuration created at: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ProteomeLM Configuration Manager")
    parser.add_argument("--validate", type=str, help="Validate a configuration file")
    parser.add_argument("--create-default", type=str, help="Create default configuration file")
    parser.add_argument("--create-experiment", type=str, help="Create experiment configuration")
    parser.add_argument("--model-size", type=str, default="S", choices=["XS", "S", "M", "L"])

    args = parser.parse_args()

    if args.validate:
        validate_config_file(args.validate)
    elif args.create_default:
        create_default_config(args.create_default)
    elif args.create_experiment:
        manager = ConfigManager()
        exp_config = manager.create_experiment_config(args.create_experiment, args.model_size)
        output_path = f"configs/{args.create_experiment}.yaml"
        Path(output_path).parent.mkdir(exist_ok=True)
        manager.config = exp_config
        manager.save_config(output_path)
    else:
        print("Please specify an action: --validate, --create-default, or --create-experiment")
