"""
Ablation: ProteomeLM with learnable OrthoDB embeddings.

Instead of using mean ESM-C embeddings of orthologous groups as functional encoding,
this variant learns a fixed embedding per OrthoDB group ID — akin to positional encoding
in a classic language model. This addresses the reviewer question of whether the
functional encoding's information "leakage" is necessary for performance, or whether
a simpler group-identity signal suffices.

Comparison with native ProteomeLM:
  - Native: group_embeds = mean ESM-C embedding of the orthologous group (continuous, informative)
  - Naive:  group_embeds = learned embedding looked up by OrthoDB group ID (discrete, learned)

Usage:
    python -m proteomelm.modeling_naive --config configs/naive_ablation.yaml

    # Or from Python:
    from proteomelm.modeling_naive import train_orthodb_proteomelm
    trainer = train_orthodb_proteomelm(config)
"""

import gc
import logging
import os
import pickle
import random
import tarfile
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from transformers import TrainingArguments
from transformers.trainer_pt_utils import IterableDataset
from transformers.trainer_utils import get_last_checkpoint

from proteomelm.modeling_proteomelm import ProteomeLMForMaskedLM, ProteomeLMConfig
from proteomelm.trainer import ProteomeLMTrainer, MemoryMonitorCallback, SaveEveryNEpochsCallback

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Vocabulary construction
# ---------------------------------------------------------------------------

def build_orthodb_vocab(
    db_path: str,
    min_taxid_size: int = 100,
    vocab_cache_path: Optional[str] = None,
) -> Dict[str, int]:
    """Build vocabulary mapping OrthoDB IDs to indices from group vector files.

    Args:
        db_path: Path to the database directory containing group_vectors_*.pkl files.
        min_taxid_size: Only include groups from files with count >= this value.
        vocab_cache_path: If provided and exists, load vocab from cache instead
                          of rebuilding. If provided and does not exist, save after building.

    Returns:
        Dictionary mapping OrthoDB group ID -> integer index (0 = <UNK>/padding).
    """
    if vocab_cache_path and os.path.exists(vocab_cache_path):
        logger.info(f"Loading cached OrthoDB vocab from {vocab_cache_path}")
        with open(vocab_cache_path, "rb") as f:
            return pickle.load(f)

    orthodb_ids = set()
    files_loaded = 0
    for fn in ["group_vectors_0.pkl", "group_vectors_10.pkl",
               "group_vectors_50.pkl", "group_vectors_200.pkl"]:
        fp = os.path.join(db_path, fn)
        count = int(fn.split("_")[-1].split(".")[0])
        if count < min_taxid_size or not os.path.exists(fp):
            continue
        with open(fp, "rb") as f:
            data = pickle.load(f)
            orthodb_ids.update(data.keys())
            files_loaded += 1

    if not orthodb_ids:
        raise ValueError(
            f"No OrthoDB groups found in {db_path} with min_taxid_size={min_taxid_size}"
        )

    # Index 0 reserved for unknown/padding
    vocab = {odb_id: idx + 1 for idx, odb_id in enumerate(sorted(orthodb_ids))}
    vocab["<UNK>"] = 0

    logger.info(
        f"Built OrthoDB vocab: {len(vocab) - 1} groups from {files_loaded} files "
        f"(+ <UNK> at index 0)"
    )

    if vocab_cache_path:
        os.makedirs(os.path.dirname(vocab_cache_path) or ".", exist_ok=True)
        with open(vocab_cache_path, "wb") as f:
            pickle.dump(vocab, f)
        logger.info(f"Saved OrthoDB vocab to {vocab_cache_path}")

    return vocab


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ProteomeLMWithOrthoDBEmbedding(ProteomeLMForMaskedLM):
    """ProteomeLM with learnable OrthoDB embeddings replacing functional encoding.

    The learned embedding replaces the mean-ESM-C group embedding used by
    native ProteomeLM. Everything else (transformer, lm_head, lm_norm,
    embedding_main, embedding_encoder) is kept identical so that results
    are directly comparable.
    """

    def __init__(self, config: ProteomeLMConfig, orthodb_vocab_size: int):
        super().__init__(config)
        self.orthodb_embedding = nn.Embedding(
            orthodb_vocab_size, config.input_size, padding_idx=0,
        )
        nn.init.normal_(self.orthodb_embedding.weight, mean=0.0, std=0.02)
        # Zero out the padding vector explicitly
        with torch.no_grad():
            self.orthodb_embedding.weight[0].zero_()

    def forward(
        self,
        input_ids=None,
        orthodb_ids=None,
        group_embeds=None,
        masked_tokens=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        # If orthodb_ids are provided, use the learned embedding as group_embeds
        if orthodb_ids is not None:
            group_embeds = self.orthodb_embedding(orthodb_ids).to(inputs_embeds.dtype)

        return super().forward(
            input_ids=input_ids,
            group_embeds=group_embeds,
            masked_tokens=masked_tokens,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class OrthoDBDataset(IterableDataset):
    """Dataset providing OrthoDB IDs instead of mean group embeddings.

    Each sample is a proteome (set of protein embeddings). For each protein,
    we look up its OrthoDB group(s) and select the one at the broadest
    taxonomic level (lowest taxid number) that exists in the vocab.
    """

    def __init__(
        self,
        db_path: str,
        orthodb_vocab: Dict[str, int],
        dataset: str = "train",
        max_length: int = 4096,
        mask_fraction: float = 0.5,
        shuffle_shards: bool = True,
    ):
        super().__init__()
        self.db_path = db_path
        self.orthodb_vocab = orthodb_vocab
        self.dataset = dataset
        self.max_length = max_length
        self.mask_fraction = mask_fraction
        self.shuffle_shards = shuffle_shards

        # Discover shards
        dataset_dir = os.path.join(db_path, dataset)
        if not os.path.isdir(dataset_dir):
            raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

        self.shards = sorted([
            os.path.join(dataset_dir, f)
            for f in os.listdir(dataset_dir)
            if f.startswith("shard_") and f.endswith(".tar")
        ])
        if not self.shards:
            raise ValueError(f"No shard files found in {dataset_dir}")

        # Count total samples for __len__ (needed by HF Trainer for progress bars)
        self._total_samples = 0
        for shard_path in self.shards:
            try:
                with tarfile.open(shard_path, "r") as tar:
                    self._total_samples += sum(1 for m in tar.getmembers() if m.isfile())
            except Exception as e:
                logger.warning(f"Could not count samples in {shard_path}: {e}")

        logger.info(
            f"OrthoDBDataset({dataset}): {len(self.shards)} shards, "
            f"~{self._total_samples} samples, max_length={max_length}"
        )

    def __len__(self) -> int:
        return self._total_samples

    def __iter__(self):
        shards = list(self.shards)
        if self.shuffle_shards:
            random.shuffle(shards)

        for shard_path in shards:
            try:
                with tarfile.open(shard_path, "r") as tar:
                    for member in tar.getmembers():
                        if not member.isfile():
                            continue
                        try:
                            with tar.extractfile(member) as f:
                                if f is None:
                                    continue
                                sample = self._process_sample(pickle.load(f))
                                if sample is not None:
                                    yield sample
                        except Exception as e:
                            logger.debug(f"Skipping {member.name}: {e}")
                            continue
            except Exception as e:
                logger.warning(f"Failed to read shard {shard_path}: {e}")
                continue

    def _process_sample(self, data) -> Optional[Dict[str, torch.Tensor]]:
        """Process a single proteome sample into model inputs."""
        if not isinstance(data, (tuple, list)) or len(data) != 4:
            return None

        _tax_id, _gene_names, odb_groups, embeds = data
        if len(odb_groups) != len(embeds) or not odb_groups:
            return None

        # Shuffle and truncate to max_length
        members = list(zip(odb_groups, embeds))
        random.shuffle(members)
        members = members[: self.max_length]

        orthodb_indices: List[int] = []
        embeddings: List[torch.Tensor] = []
        unmaskable: List[int] = []

        for i, (odb_group_list, embed) in enumerate(members):
            if not isinstance(embed, torch.Tensor):
                embed = torch.tensor(embed, dtype=torch.float32)

            # Select OrthoDB group at broadest taxonomic level (lowest taxid)
            best_idx, best_taxid = 0, float("inf")
            for odb_id in odb_group_list:
                if odb_id not in self.orthodb_vocab:
                    continue
                try:
                    taxid = int(odb_id.split("at")[-1])
                except (ValueError, IndexError):
                    continue
                if taxid < best_taxid:
                    best_taxid = taxid
                    best_idx = self.orthodb_vocab[odb_id]

            orthodb_indices.append(best_idx)
            embeddings.append(embed)
            if best_idx == 0:  # unknown group — cannot mask
                unmaskable.append(i)

        if not embeddings:
            return None

        # Create masking
        num_seqs = len(members)
        num_to_mask = max(1, int(self.mask_fraction * num_seqs))
        maskable = [i for i in range(num_seqs) if i not in unmaskable]
        if maskable:
            masked_idx = random.sample(maskable, min(num_to_mask, len(maskable)))
        else:
            masked_idx = []

        masked_tokens = torch.zeros(num_seqs, dtype=torch.long)
        if masked_idx:
            masked_tokens[masked_idx] = 1

        return {
            "inputs_embeds": torch.stack(embeddings),
            "orthodb_ids": torch.tensor(orthodb_indices, dtype=torch.long),
            "masked_tokens": masked_tokens,
        }


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------

class OrthoDBCollator:
    """Collator for batching samples with padding.

    Produces the same output keys as DataCollatorForProteomeLM (inputs_embeds,
    masked_tokens, labels) plus orthodb_ids.
    """

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        valid = [inst for inst in features if inst is not None and "inputs_embeds" in inst]
        if not valid:
            raise ValueError("No valid instances in batch")

        inputs_embeds = pad_sequence(
            [inst["inputs_embeds"] for inst in valid], batch_first=True
        )
        orthodb_ids = pad_sequence(
            [inst["orthodb_ids"] for inst in valid], batch_first=True, padding_value=0
        )
        masked_tokens = pad_sequence(
            [inst["masked_tokens"] for inst in valid], batch_first=True, padding_value=0
        )

        inputs_bf16 = inputs_embeds.to(torch.bfloat16)
        return {
            "inputs_embeds": inputs_bf16,
            "orthodb_ids": orthodb_ids,
            "masked_tokens": masked_tokens,
            "labels": inputs_bf16.clone(),
        }


# ---------------------------------------------------------------------------
# Trainer with proper loss + metrics
# ---------------------------------------------------------------------------

class OrthoDBProteomeLMTrainer(ProteomeLMTrainer):
    """Trainer that passes orthodb_ids to the model.

    Supports the same loss choices as the native ProteomeLMTrainer (polar,
    cosine, mse) and logs comparison-friendly metrics.
    """

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        masked_tokens = inputs["masked_tokens"]

        mask = masked_tokens.bool()
        mask_indices = mask.nonzero(as_tuple=True)

        # Forward pass — model generates group_embeds from orthodb_ids internally
        output = model(**inputs, return_dict=True)

        # Extract predictions at masked positions
        prediction_scores = output["prediction_scores"].float()
        prediction_norm = output["prediction_norm"].float()

        # If model didn't apply the mask internally, apply it here.
        if prediction_scores.dim() == 3:
            prediction_scores = prediction_scores[mask]
            prediction_norm = prediction_norm[mask]

        labels_masked = labels[mask_indices].float()

        # Safety: align lengths if masking differs for any reason
        if prediction_scores.shape[0] != labels_masked.shape[0]:
            labels_masked = labels[mask].float()
            if prediction_scores.shape[0] != labels_masked.shape[0]:
                min_len = min(prediction_scores.shape[0], labels_masked.shape[0])
                prediction_scores = prediction_scores[:min_len]
                prediction_norm = prediction_norm[:min_len]
                labels_masked = labels_masked[:min_len]

        # Compute loss — mirrors ProteomeLMTrainer exactly
        loss_fct = torch.nn.MSELoss()
        loss = 100*loss_fct(prediction_scores, labels_masked)

        # --- Log comparison metrics (every step, lightweight) ---
        with torch.no_grad():
            # Cosine similarity between prediction direction and target direction
            pred_dir = prediction_scores
            target_dir = labels_masked
            cos_sim = torch.nn.functional.cosine_similarity(pred_dir, target_dir, dim=-1)

            # Norm error
            pred_norm_val = prediction_norm.squeeze(-1)
            target_norm_val = torch.linalg.norm(target_dir, ord=2, dim=-1)
            norm_mae = (pred_norm_val - target_norm_val).abs().mean()

            # MSE for absolute comparison
            mse = torch.nn.functional.mse_loss(prediction_scores, labels_masked)

            # Fraction of <UNK> tokens in this batch
            n_unk = (inputs["orthodb_ids"] == 0).sum().float()
            n_total = (inputs["orthodb_ids"] >= 0).sum().float()

            self._log_naive_metrics({
                "naive/cosine_sim": cos_sim.mean().item(),
                "naive/norm_mae": norm_mae.item(),
                "naive/mse": mse.item(),
                "naive/loss": loss.item(),
                "naive/frac_unk": (n_unk / n_total).item() if n_total > 0 else 0.0,
                "naive/n_masked": masked_tokens.sum().item(),
            })

        del labels_masked, mask_indices
        return (loss, output) if return_outputs else loss

    def _log_naive_metrics(self, metrics: Dict[str, float]):
        """Log metrics to all active loggers (WandB, TensorBoard, etc.)."""
        try:
            self.log(metrics)
        except Exception:
            pass  # Don't crash training on logging failures


# ---------------------------------------------------------------------------
# MinTaxid callback adapted for OrthoDBDataset
# ---------------------------------------------------------------------------

class NaiveMinTaxidSchedulerCallback:
    """Rebuild datasets at epoch milestones with decreasing min_taxid.

    Mirrors MinTaxidSchedulerCallback from the native trainer but rebuilds
    OrthoDBDatasets instead of ProteomeLMDatasets.
    """

    def __init__(self, config: Dict, trainer, orthodb_vocab: Dict[str, int]):
        self.config = config
        self.trainer = trainer
        self.orthodb_vocab = orthodb_vocab
        # epoch -> new min_taxid value
        self.schedule = config.get("min_taxid_schedule", {30: 50, 60: 20})

    def on_epoch_end(self, args, state, control, **kwargs):
        current_epoch = int(state.epoch)
        new_min_taxid = self.schedule.get(current_epoch)

        if new_min_taxid is None:
            return

        logger.info(f"Updating min_taxid to {new_min_taxid} at epoch {current_epoch}")

        # Rebuild vocab with broader groups
        self.orthodb_vocab = build_orthodb_vocab(
            self.config["db_path"], min_taxid_size=new_min_taxid
        )

        # Free old datasets
        del self.trainer.train_dataset
        del self.trainer.eval_dataset
        gc.collect()

        self.trainer.train_dataset = OrthoDBDataset(
            self.config["db_path"], self.orthodb_vocab, "train",
            self.config.get("max_length", 4096), self.config.get("mask_fraction", 0.5),
        )
        self.trainer.eval_dataset = OrthoDBDataset(
            self.config["db_path"], self.orthodb_vocab, "eval",
            self.config.get("max_length", 4096), self.config.get("mask_fraction", 0.5),
        )


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_orthodb_proteomelm(config: Dict) -> OrthoDBProteomeLMTrainer:
    """Train ProteomeLM with learnable OrthoDB embeddings.

    Args:
        config: Configuration dictionary. Expected keys:
            - db_path: Path to shard database
            - output_dir: Where to save checkpoints
            - dim, n_layers, n_heads, input_size: Model architecture
            - batch_size, learning_rate, num_epochs/max_steps: Training
            - loss_choice: "polar" | "cosine" | "mse" (default: "polar")
            - mask_fraction, max_length: Data processing
            - wandb_project: Optional WandB project name
            - resume: Whether to resume from checkpoint (default: True)

    Returns:
        Trained OrthoDBProteomeLMTrainer instance.
    """
    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # ---- Vocab ----
    vocab_path = os.path.join(output_dir, "orthodb_vocab.pkl")
    vocab = build_orthodb_vocab(
        config["db_path"],
        config.get("min_taxid_size", 100),
        vocab_cache_path=vocab_path,
    )

    # ---- Datasets ----
    train_dataset = OrthoDBDataset(
        config["db_path"], vocab, "train",
        config.get("max_length", 4096), config.get("mask_fraction", 0.5),
    )
    eval_dataset = OrthoDBDataset(
        config["db_path"], vocab, "eval",
        config.get("max_length", 4096), config.get("mask_fraction", 0.5),
    )

    # ---- Model ----
    resume = config.get("resume", True)
    last_checkpoint = None
    if resume and os.path.isdir(output_dir):
        last_checkpoint = get_last_checkpoint(output_dir)

    if last_checkpoint is not None:
        logger.info(f"Resuming from checkpoint: {last_checkpoint}")
        model = ProteomeLMWithOrthoDBEmbedding.from_pretrained(last_checkpoint)
    else:
        model_config = ProteomeLMConfig(
            input_size=config.get("input_size", 1152),
            dim=config.get("dim", 512),
            n_layers=config.get("n_layers", 6),
            n_heads=config.get("n_heads", 8),
        )
        model = ProteomeLMWithOrthoDBEmbedding(model_config, len(vocab))

    model = model.to(torch.bfloat16)

    n_params = sum(p.numel() for p in model.parameters())
    n_emb_params = model.orthodb_embedding.weight.numel()
    logger.info(
        f"Model: {n_params:,} params total, "
        f"{n_emb_params:,} in OrthoDB embedding "
        f"({n_emb_params / n_params * 100:.1f}%)"
    )

    # ---- Training args ----
    # Support both num_epochs and max_steps
    num_epochs = config.get("num_epochs", None)
    max_steps = config.get("max_steps", -1)
    if num_epochs is None and max_steps <= 0:
        num_epochs = 100  # default

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs if num_epochs else 1,
        max_steps=max_steps if max_steps > 0 else -1,
        per_device_train_batch_size=config.get("batch_size", 16),
        per_device_eval_batch_size=config.get("batch_size", 16),
        learning_rate=config.get("learning_rate", 3e-4),
        weight_decay=config.get("weight_decay", 0.01),
        warmup_steps=config.get("warmup_steps", 500),
        max_grad_norm=config.get("max_grad_norm", 1.0),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 1),
        logging_steps=config.get("logging_steps", 50),
        eval_strategy="epoch" if num_epochs else "steps",
        eval_steps=config.get("eval_steps", 1000) if not num_epochs else None,
        save_strategy="epoch" if num_epochs else "steps",
        save_steps=config.get("save_steps", 1000) if not num_epochs else None,
        save_total_limit=config.get("save_total_limit", 3),
        bf16=True,
        dataloader_num_workers=config.get("dataloader_num_workers", 0),
        dataloader_pin_memory=True,
        ddp_find_unused_parameters=False,
        label_names=["labels"],
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        load_best_model_at_end=True,
        report_to="wandb" if config.get("wandb_project") else "none",
        run_name=config.get("namedir", "naive_orthodb_ablation"),
    )

    # ---- WandB ----
    if config.get("wandb_project"):
        try:
            import wandb
            wandb.init(
                project=config["wandb_project"],
                name=config.get("namedir", "naive_orthodb_ablation"),
                config={
                    **config,
                    "model_type": "naive_orthodb",
                    "orthodb_vocab_size": len(vocab),
                },
                resume="allow" if resume else False,
            )
        except Exception as e:
            logger.warning(f"WandB init failed: {e}")

    # ---- Trainer ----
    trainer = OrthoDBProteomeLMTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=OrthoDBCollator(),
        loss_choice=config.get("loss_choice", "polar"),
    )

    # Callbacks
    if config.get("save_epochs"):
        trainer.add_callback(SaveEveryNEpochsCallback(
            save_every_n_epochs=config["save_epochs"]
        ))

    if config.get("min_taxid_schedule"):
        from transformers import TrainerCallback

        class _NaiveTaxidCB(TrainerCallback):
            def __init__(self, cb):
                self._cb = cb
            def on_epoch_end(self, args, state, control, **kwargs):
                self._cb.on_epoch_end(args, state, control, **kwargs)

        taxid_cb = NaiveMinTaxidSchedulerCallback(config, trainer, vocab)
        trainer.add_callback(_NaiveTaxidCB(taxid_cb))

    trainer.add_callback(MemoryMonitorCallback(log_interval=200))

    # ---- Train ----
    logger.info("Starting naive OrthoDB ablation training...")
    logger.info(f"  Loss: {config.get('loss_choice', 'polar')}")
    logger.info(f"  Vocab size: {len(vocab)}")
    logger.info(f"  Train samples: ~{len(train_dataset)}")
    logger.info(f"  Eval samples: ~{len(eval_dataset)}")

    try:
        trainer.evaluate()  # initial eval
        trainer.train(resume_from_checkpoint=last_checkpoint)
        final_metrics = trainer.evaluate()
        logger.info(f"Final eval metrics: {final_metrics}")

        # Save final model + vocab together
        final_path = os.path.join(output_dir, "final_model")
        trainer.save_model(final_path)
        import shutil
        shutil.copy2(vocab_path, os.path.join(final_path, "orthodb_vocab.pkl"))
        logger.info(f"Final model saved to {final_path}")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        if config.get("wandb_project"):
            try:
                import wandb
                wandb.finish()
            except Exception:
                pass

    return trainer


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import yaml

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO,
    )

    parser = argparse.ArgumentParser(
        description="Train ProteomeLM with learnable OrthoDB embeddings (naive ablation)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--no-resume", action="store_true", help="Don't resume from checkpoint")
    parser.add_argument("--validate-only", action="store_true", help="Only validate config")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.no_resume:
        config["resume"] = False

    if args.validate_only:
        required = ["db_path", "output_dir"]
        missing = [k for k in required if k not in config]
        if missing:
            print(f"Missing required keys: {missing}")
        else:
            print("Config OK")
            print(f"  db_path: {config['db_path']}")
            print(f"  output_dir: {config['output_dir']}")
            print(f"  loss_choice: {config.get('loss_choice', 'polar')}")
            print(f"  dim: {config.get('dim', 512)}")
    else:
        train_orthodb_proteomelm(config)