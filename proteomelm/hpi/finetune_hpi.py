import logging
import os
import random
from pathlib import Path
from typing import Dict, Optional

import torch
import argparse
from peft import LoraConfig, PeftModel, get_peft_model

from transformers import TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

from .dataloaders import DataCollatorForProteomeLMHP, get_shards_dataset
from .helpers import compute_metrics

from ..cli import load_config
from ..modeling_proteomelm import ProteomeLMForMaskedLM
from ..trainer import ProteomeLMTrainer
from ..utils import print_number_of_parameters
from ..ppi.model import _seed_everything

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)


def load_model(config: Dict) -> tuple[torch.nn.Module, Optional[str]]:
    """Load the model: resume from a LoRA checkpoint if one exists, otherwise
    load the pre-trained base weights and wrap with fresh LoRA adapters.

    Returns the model and the checkpoint path it was resumed from (or None for
    a fresh run) — the caller must pass this into `trainer.train()` as
    `resume_from_checkpoint`, otherwise the LR schedule, optimizer state, and
    global step counter silently restart from scratch even though the weights
    themselves were resumed.

    The architecture always comes from `pretrained_model_path` (its HF
    config.json); architecture keys in the YAML are not read.
    """
    output_path = Path(config["output_dir"]) / config["namedir"]
    checkpoint_path = config.get("checkpoint_path")
    if checkpoint_path is not None:
        last_checkpoint = checkpoint_path
    else:
        last_checkpoint = get_last_checkpoint(str(output_path)) if output_path.exists() else None

    if last_checkpoint is not None:
        base_model = ProteomeLMForMaskedLM.from_pretrained(config["pretrained_model_path"])
        model = PeftModel.from_pretrained(base_model, last_checkpoint, is_trainable=True)
        logging.info(f"Resuming from checkpoint: {last_checkpoint}")
    else:
        model = ProteomeLMForMaskedLM.from_pretrained(config["pretrained_model_path"])
        model = get_peft_model(model, LoraConfig(
            r=config["lora_r"],
            lora_alpha=config["lora_alpha"],
            lora_dropout=config["lora_dropout"],
            target_modules=config["lora_target_modules"],
            bias=config["lora_bias"],
        ))
        logging.info(f"Loaded pre-trained weights from {config['pretrained_model_path']}, LoRA r={config['lora_r']}")

    print_number_of_parameters(model)
    return model, last_checkpoint


def build_trainer(config: Dict, model: torch.nn.Module, train_dataset=None, eval_dataset=None) -> ProteomeLMTrainer:
    """Build a trainer for training or offline evaluation."""
    output_path = str(Path(config["output_dir"]) / config["namedir"])
    enable_eval_during_training = config.get("enable_eval_during_training", True)
    eval_only = config.get("eval_only", False)
    use_eval = eval_only or enable_eval_during_training
    compute_metrics_fn = compute_metrics if config.get("compute_eval_metrics", use_eval) else None

    training_args = TrainingArguments(
        output_dir=output_path,
        logging_dir=output_path,
        # Training schedule
        num_train_epochs=config["num_epochs"],
        max_steps=config["max_steps"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config.get("eval_batch_size", config["batch_size"]),
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        eval_accumulation_steps=config["eval_accumulation_steps"],
        # Optimizer
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        adam_beta1=config["beta1"],
        adam_beta2=config["beta2"],
        max_grad_norm=config["max_grad_norm"],
        lr_scheduler_type=config.get("scheduler", "cosine"),
        warmup_steps=config["warmup_steps"],
        # Logging & checkpointing
        logging_steps=config["logging_steps"],
        eval_steps=config["eval_steps"] if use_eval else None,
        save_steps=config["save_steps"],
        evaluation_strategy="steps" if use_eval else "no",
        eval_on_start=bool(config.get("eval_on_start", use_eval)),
        dataloader_num_workers=config["dataloader_num_workers"],
        dataloader_persistent_workers=config["dataloader_num_workers"] > 0,
        # Misc
        label_names=["labels", "source_ids"],
        bf16=True,
        overwrite_output_dir=False,
        push_to_hub=False,
        ddp_find_unused_parameters=False,
        report_to="wandb",
        run_name=config["namedir"],
    )

    return ProteomeLMTrainer(
        loss_choice=config.get("loss_choice", "polar"),
        block_pathogen_self_attention=config.get("block_pathogen_self_attention", False),
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForProteomeLMHP(),
        compute_metrics=compute_metrics_fn,
        args=training_args,
    )


def run(config: Dict):
    """Run the fine-tuning loop or a standalone evaluation pass."""
    # Restrict to a single GPU when requested (multi-GPU handled by torchrun)
    if str(config.get("use_one_gpu", "-1")) != "-1":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config["use_one_gpu"])

    if "wandb_project" in config:
        os.environ["WANDB_PROJECT"] = config["wandb_project"]

    torch.set_default_dtype(torch.bfloat16)

    eval_only = config.get("eval_only", False)
    if eval_only or config.get("deterministic_eval", False):
        # Masking (proteomelm.hpi.dataloaders.apply_masking) samples via the stdlib
        # `random` module even at eval time, so eval-only runs on the same checkpoint
        # would otherwise pick a different mask each invocation. `_seed_everything`
        # covers torch/numpy/cuda; `random.seed` covers the masking draw itself.
        seed = config.get("seed", 42)
        _seed_everything(seed)
        random.seed(seed)

    # Load model first so architecture/checkpoint errors surface before any disk I/O on the dataset.
    # Dataset objects are constructed cheaply here (no shard scanning or pkl loading yet);
    # the heavy disk reads are deferred to when trainer.train() starts iterating.
    model, last_checkpoint = load_model(config)
    train_dataset = None if eval_only else get_shards_dataset(split="train", **config)
    eval_dataset = get_shards_dataset(split="eval", **config)
    trainer = build_trainer(config, model, train_dataset=train_dataset, eval_dataset=eval_dataset)

    if eval_only:
        metrics = trainer.evaluate()
        logging.info("Standalone evaluation metrics: %s", metrics)
        return metrics

    trainer.train(resume_from_checkpoint=last_checkpoint)
    return trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, nargs="+", required=True,
                        help="Path(s) to YAML config file(s), merged in order (later files win).")
    parser.add_argument("--eval-only", action="store_true", help="Run a deterministic offline evaluation pass.")
    parser.add_argument("--checkpoint", type=str, help="Optional checkpoint path to load for offline evaluation.")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.eval_only:
        config["eval_only"] = True
        config["compute_eval_metrics"] = True
        config["eval_on_start"] = False
        if args.checkpoint:
            config["checkpoint_path"] = args.checkpoint
    run(config)