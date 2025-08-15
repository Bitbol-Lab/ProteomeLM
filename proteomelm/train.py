"""
ProteomeLM Training Script

This script provides a comprehensive training pipeline for ProteomeLM models
with support for distributed training, checkpointing, and monitoring.
"""

import logging
import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Optional, List, Union

import torch
import torch.distributed as dist
import yaml
import wandb
from torch.optim import AdamW
from transformers import TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

from .utils import load_scheduler, print_number_of_parameters
from .modeling_proteomelm import ProteomeLMForMaskedLM, ProteomeLMConfig
from .dataloaders import get_shards_dataset, DataCollatorForProteomeLM
from .trainer import (
    ProteomeLMTrainer,
    MemoryMonitorCallback,
    MinTaxidSchedulerCallback,
    SaveEveryNEpochsCallback
)

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("training.log", mode="a")
    ]
)
logger = logging.getLogger(__name__)


def setup_distributed(use_one_gpu: Union[str, int] = "-1") -> int:
    """
    Setup distributed training configuration.

    Args:
        use_one_gpu: GPU configuration. -1 for multi-GPU, >= 0 for specific GPU

    Returns:
        rank: Process rank for distributed training
    """
    if isinstance(use_one_gpu, int):
        use_one_gpu = str(use_one_gpu)

    if int(use_one_gpu) >= 0:
        # Single GPU setup
        os.environ["CUDA_VISIBLE_DEVICES"] = use_one_gpu
        rank = 0
        logger.info(f"Using single GPU: {use_one_gpu}")
    else:
        # Multi-GPU distributed setup
        try:
            dist.init_process_group(backend='nccl', init_method='env://')
            torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
            rank = int(os.environ['LOCAL_RANK'])
            logger.info(f"Distributed training setup complete. Rank: {rank}")
        except Exception as e:
            logger.error(f"Failed to setup distributed training: {e}")
            rank = 0

    return rank


def validate_config(config: Dict) -> None:
    """
    Validate configuration parameters.

    Args:
        config: Configuration dictionary

    Raises:
        ValueError: If configuration is invalid
    """
    required_keys = [
        "batch_size", "learning_rate", "num_epochs",
        "output_dir", "namedir", "dim", "n_layers", "n_heads"
    ]

    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise ValueError(f"Missing required configuration keys: {missing_keys}")

    # Validate parameter ranges
    if config["batch_size"] <= 0:
        raise ValueError("batch_size must be positive")
    if config["learning_rate"] <= 0:
        raise ValueError("learning_rate must be positive")
    if config["num_epochs"] <= 0:
        raise ValueError("num_epochs must be positive")

    logger.info("Configuration validation passed")


def setup_model(config: Dict, pretrained_path: Optional[str] = None) -> ProteomeLMForMaskedLM:
    """
    Setup ProteomeLM model from config or pretrained checkpoint.

    Args:
        config: Model configuration
        pretrained_path: Path to pretrained model or Hugging Face model ID

    Returns:
        ProteomeLM model instance
    """
    torch.set_default_dtype(torch.bfloat16)

    # Check for existing checkpoint
    output_path = Path(config["output_dir"]) / config["namedir"]
    last_checkpoint = None
    if output_path.exists():
        last_checkpoint = get_last_checkpoint(str(output_path))

    if last_checkpoint is not None:
        logger.info(f"Resuming from checkpoint: {last_checkpoint}")
        model = ProteomeLMForMaskedLM.from_pretrained(last_checkpoint)
    elif pretrained_path is not None:
        logger.info(f"Loading pretrained model: {pretrained_path}")
        model = ProteomeLMForMaskedLM.from_pretrained(pretrained_path)
    else:
        logger.info("Creating new model from config")
        proteomelm_config = ProteomeLMConfig(**config)
        model = ProteomeLMForMaskedLM(proteomelm_config)

    return model


def setup_datasets(config: Dict):
    """
    Setup training and evaluation datasets.

    Args:
        config: Configuration dictionary

    Returns:
        Tuple of (train_dataset, eval_dataset, data_collator)
    """
    logger.info("Loading datasets...")

    try:
        train_dataset = get_shards_dataset(dataset="train", **config)
        eval_dataset = get_shards_dataset(dataset="eval", **config)
        data_collator = DataCollatorForProteomeLM()

        logger.info(f"Training dataset size: {len(train_dataset)}")
        logger.info(f"Evaluation dataset size: {len(eval_dataset)}")

        return train_dataset, eval_dataset, data_collator

    except Exception as e:
        logger.error(f"Failed to load datasets: {e}")
        raise


def setup_training_components(model, config: Dict, dataset_size: int):
    """
    Setup optimizer, scheduler, and training arguments.

    Args:
        model: ProteomeLM model
        config: Configuration dictionary
        dataset_size: Size of training dataset

    Returns:
        Tuple of (optimizer, scheduler, training_args)
    """
    # Calculate effective batch size
    effective_batch_size = config['batch_size'] * config.get('gradient_accumulation_steps', 1)
    num_gpus = torch.cuda.device_count()
    effective_batch_size *= max(num_gpus, 1)

    logger.info(f"Effective batch size: {effective_batch_size}")
    logger.info(f"Steps per epoch: {dataset_size // effective_batch_size}")

    # Setup optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        betas=(config.get("beta1", 0.9), config.get("beta2", 0.999)),
        weight_decay=config.get("weight_decay", 0.01)
    )

    # Setup scheduler
    scheduler = load_scheduler(optimizer, config, dataset_size, effective_batch_size)

    # Setup training arguments
    output_path = Path(config["output_dir"]) / config["namedir"]
    training_args = TrainingArguments(
        learning_rate=config["learning_rate"],
        num_train_epochs=config["num_epochs"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        max_grad_norm=config.get("max_grad_norm", 1.0),
        dataloader_num_workers=config.get("dataloader_num_workers", 0),
        logging_steps=config.get("logging_steps", 10),
        eval_strategy="epoch",
        output_dir=str(output_path),
        logging_dir=str(output_path),
        label_names=["labels"],
        bf16=True,
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=2 if config.get("dataloader_num_workers", 0) > 0 else None,
        overwrite_output_dir=False,
        push_to_hub=config.get("push_to_hub", False),
        ddp_find_unused_parameters=False,
        save_strategy="epoch",
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 1),
        warmup_steps=config.get("warmup_steps", 500),
        save_total_limit=config.get("save_total_limit", 3),
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        load_best_model_at_end=True,
    )

    return optimizer, scheduler, training_args


def run_training(config: Dict, pretrained_path: Optional[str] = None, resume: bool = True) -> ProteomeLMTrainer:
    """
    Main training function for ProteomeLM.

    Args:
        config: Training configuration dictionary
        pretrained_path: Path to pretrained model or Hugging Face model ID
        resume: Whether to resume from checkpoint if available

    Returns:
        Trained ProteomeLM trainer instance
    """
    logger.info("Starting ProteomeLM training...")

    # Validate configuration
    validate_config(config)

    # Setup distributed training
    rank = setup_distributed(config.get("use_one_gpu", "-1"))

    # Setup datasets
    train_dataset, eval_dataset, data_collator = setup_datasets(config)

    # Setup model
    model = setup_model(config, pretrained_path if not resume else None)

    # Print model information
    print_number_of_parameters(model)

    # Setup training components
    optimizer, scheduler, training_args = setup_training_components(
        model, config, len(train_dataset)
    )

    # Setup Weights & Biases logging
    if rank == 0 and config.get("wandb_project"):
        wandb.init(
            project=config["wandb_project"],
            name=config["namedir"],
            config=config,
            resume="allow" if resume else False
        )
        logger.info(f"Weights & Biases initialized for project: {config['wandb_project']}")

    # Create trainer
    trainer = ProteomeLMTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        optimizers=(optimizer, scheduler),
        loss_choice=config.get("loss_choice", "mse"),
    )

    # Add callbacks
    if config.get("save_epochs"):
        trainer.add_callback(SaveEveryNEpochsCallback(
            save_every_n_epochs=config["save_epochs"]
        ))

    trainer.add_callback(MinTaxidSchedulerCallback(config, trainer))
    trainer.add_callback(MemoryMonitorCallback(log_interval=200))

    logger.info(f"Training setup complete. Using {torch.cuda.device_count()} GPUs")

    try:
        # Run evaluation before training
        logger.info("Running initial evaluation...")
        trainer.evaluate()

        # Start training
        logger.info("Starting training loop...")
        trainer.train(resume_from_checkpoint=resume)

        # Final evaluation
        logger.info("Running final evaluation...")
        final_metrics = trainer.evaluate()
        logger.info(f"Final metrics: {final_metrics}")

        # Save final model
        final_model_path = Path(training_args.output_dir) / "final_model"
        trainer.save_model(str(final_model_path))
        logger.info(f"Final model saved to: {final_model_path}")

        return trainer

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        # Cleanup
        if rank == 0 and config.get("wandb_project"):
            wandb.finish()


def load_config(config_paths: Union[str, List[str]]) -> Dict:
    """
    Load configuration from YAML file(s).

    Args:
        config_paths: Path(s) to configuration file(s)

    Returns:
        Merged configuration dictionary
    """
    if isinstance(config_paths, str):
        config_paths = [config_paths]

    if not config_paths:
        config_paths = ["configs/proteomelm.yaml"]

    config = {}
    for path in config_paths:
        if not Path(path).exists():
            logger.warning(f"Configuration file not found: {path}")
            continue

        try:
            with open(path, "r", encoding="utf-8") as file:
                file_config = yaml.safe_load(file)
                config.update(file_config)
                logger.info(f"Loaded configuration from: {path}")
        except Exception as e:
            logger.error(f"Failed to load config from {path}: {e}")
            raise

    if not config:
        raise ValueError("No valid configuration files found")

    return config


def main():
    """Main entry point for training script."""
    parser = argparse.ArgumentParser(
        description="Train ProteomeLM models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--config",
        type=str,
        nargs='+',
        default=["configs/proteomelm.yaml"],
        help="Path(s) to YAML configuration file(s)"
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help="Path or Hugging Face model ID to fine-tune from"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Don't resume from checkpoint even if available"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate configuration without training"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )

    args = parser.parse_args()

    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    try:
        # Load configuration
        config = load_config(args.config)

        if args.validate_only:
            validate_config(config)
            logger.info("Configuration validation completed successfully!")
            return

        # Run training
        run_training(
            config=config,
            pretrained_path=args.pretrained,
            resume=not args.no_resume
        )

        logger.info("Training completed successfully!")

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
