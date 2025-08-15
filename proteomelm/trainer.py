import gc
import logging
import psutil
import torch
from transformers import Trainer, TrainerCallback

from .dataloaders import get_shards_dataset


class ProteomeLMTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        self.loss_choice = kwargs.pop("loss_choice", "polar")

        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        masked_tokens = inputs["masked_tokens"]
        group_embeds = inputs["group_embeds"]

        # Process only masked tokens to save memory
        mask = (masked_tokens == 1)

        # Instead of indexing the whole tensors first, get indices
        mask_indices = mask.nonzero(as_tuple=True)

        # Apply indexing only once
        root = group_embeds[mask_indices].float()
        labels_masked = labels[mask_indices].float()

        # Forward pass
        output = model(**inputs, return_dict=True)

        # Get only the masked token predictions
        prediction_scores = output["prediction_scores"].float()
        prediction_norm = output["prediction_norm"].float()

        if self.loss_choice == "polar":
            # Calculate losses
            loss_fct1 = torch.nn.CosineEmbeddingLoss()
            loss1 = loss_fct1(
                prediction_scores - root,
                labels_masked - root,
                torch.ones(root.size(0), device=prediction_scores.device)
            )

            # Norm loss
            loss_fct2 = torch.nn.MSELoss()
            loss2 = loss_fct2(
                prediction_norm,
                torch.linalg.norm(labels_masked - root, ord=2, dim=-1, keepdim=True)
            )
            loss = loss1 + loss2
        elif self.loss_choice == "cosine":
            # Calculate losses
            loss_fct = torch.nn.CosineEmbeddingLoss()
            loss = loss_fct(
                prediction_scores - root,
                labels_masked - root,
                torch.ones(root.size(0), device=prediction_scores.device)
            )
        elif self.loss_choice == "mse":
            # Calculate losses
            loss_fct = torch.nn.MSELoss()
            loss = loss_fct(
                prediction_scores,
                labels_masked
            )
        else:
            raise ValueError(f"Unknown loss choice: {self.loss_choice}")
        del root, labels_masked, mask_indices
        if return_outputs:
            return (loss, output)
        else:
            return loss


class MinTaxidSchedulerCallback(TrainerCallback):
    def __init__(self, config, trainer):
        self.config = config
        self.trainer = trainer

    def on_epoch_end(self, args, state, control, **kwargs):
        current_epoch = int(state.epoch)  # cast float to int
        new_min_taxid = None
        if current_epoch == 30:
            new_min_taxid = 50
        elif current_epoch == 60:
            new_min_taxid = 20

        if new_min_taxid is not None:
            logging.info(f"Updating min_taxid to {new_min_taxid} at epoch {current_epoch}")
            self.config["min_taxid"] = new_min_taxid

            # Force garbage collection before creating new datasets
            del self.trainer.train_dataset
            del self.trainer.eval_dataset
            gc.collect()  # Explicitly run garbage collection

            self.trainer.train_dataset = get_shards_dataset(dataset='train', **self.config)
            self.trainer.eval_dataset = get_shards_dataset(dataset='eval',  **self.config)


class MemoryMonitorCallback(TrainerCallback):
    """Monitor memory usage during training"""

    def __init__(self, log_interval=10):
        self.log_interval = log_interval
        self.step_count = 0

    def on_step_end(self, args, state, control, **kwargs):
        self.step_count += 1
        if self.step_count % self.log_interval == 0:
            process = psutil.Process()
            ram_usage = process.memory_info().rss / (1024 * 1024)  # Convert to MB
            logging.info(f"RAM Memory usage: {ram_usage:.2f} MB")


class SaveEveryNEpochsCallback(TrainerCallback):
    def __init__(self, save_every_n_epochs):
        self.save_every_n_epochs = save_every_n_epochs

    def on_epoch_end(self, args, state, control, **kwargs):
        if (state.epoch + 1) % self.save_every_n_epochs == 0:
            control.should_save = True
        else:
            control.should_save = False
        return control
