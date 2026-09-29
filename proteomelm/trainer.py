import logging
import psutil
import torch
from transformers import Trainer, TrainerCallback


class ProteomeLMTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        self.loss_choice = kwargs.pop("loss_choice", "polar")
        self.block_pathogen_self_attention = kwargs.pop("block_pathogen_self_attention", False)

        super().__init__(*args, **kwargs)

    @staticmethod
    def _build_asymmetric_mask(attention_mask, source_ids):
        """Build a [B, S, S] attention mask that blocks pathogen→pathogen attention.

        Host tokens (source_ids==0) can attend to everything (subject to padding).
        Pathogen tokens (source_ids==1) can only attend to host tokens.
        Values: 0.0 = attend, -inf = blocked.
        """
        B, S = attention_mask.shape
        is_pathogen = (source_ids == 1)                     # [B, S]
        # blocked[b, i, j] = True when query i is pathogen AND key j is pathogen
        blocked = is_pathogen.unsqueeze(2) & is_pathogen.unsqueeze(1)  # [B, S, S]
        # padding mask: key positions where attention_mask==0 are blocked for all queries
        padding_blocked = (attention_mask == 0).unsqueeze(1).expand(B, S, S)  # [B, S, S]
        blocked = blocked | padding_blocked
        mask_4d = torch.zeros(B, 1, S, S, device=attention_mask.device)
        mask_4d[:, 0][blocked] = torch.finfo(mask_4d.dtype).min
        return mask_4d

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        masked_tokens = inputs["masked_tokens"]
        group_embeds = inputs["group_embeds"]

        # Build asymmetric attention mask if enabled
        if self.block_pathogen_self_attention and "source_ids" in inputs:
            inputs["attention_mask"] = self._build_asymmetric_mask(
                inputs["attention_mask"], inputs["source_ids"]
            )

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
            residual_pred = prediction_scores - root
            residual_true = labels_masked - root

            # Numerically stable cosine: clamp norms from below to prevent
            # near-zero denominator gradients (~O(1/eps) without this).
            # eps=1e-3 keeps max raw gradient at ~1e3 (vs 1e4 at 1e-4),
            # eliminating the gradient spikes that spiked loss at ~step 3300.
            eps = 1e-3
            pred_norm = residual_pred.norm(dim=-1, keepdim=True).clamp(min=eps)
            true_norm_raw = residual_true.norm(dim=-1, keepdim=True)
            true_norm = true_norm_raw.clamp(min=eps)

            # Skip tokens whose true displacement is near-zero: the cosine
            # target is numerically undefined there and adds noisy gradient.
            valid = (true_norm_raw.squeeze(-1) >= eps)
            n_valid = valid.sum().clamp(min=1)

            cos_sim = (residual_pred / pred_norm * residual_true / true_norm).sum(dim=-1)
            loss1 = ((1 - cos_sim) * valid.float()).sum() / n_valid

            # Norm loss: predict the magnitude of the true displacement.
            # Only over valid (non-degenerate) tokens.
            loss_fct2 = torch.nn.MSELoss(reduction="none")
            loss2 = (loss_fct2(prediction_norm, true_norm) * valid.float().unsqueeze(-1)).sum() / n_valid
            loss = loss1 + loss2
        elif self.loss_choice == "cosine":
            residual_pred = prediction_scores - root
            residual_true = labels_masked - root

            eps = 1e-3
            pred_norm = residual_pred.norm(dim=-1, keepdim=True).clamp(min=eps)
            true_norm_raw = residual_true.norm(dim=-1, keepdim=True)
            true_norm = true_norm_raw.clamp(min=eps)

            valid = (true_norm_raw.squeeze(-1) >= eps)
            n_valid = valid.sum().clamp(min=1)

            cos_sim = (residual_pred / pred_norm * residual_true / true_norm).sum(dim=-1)
            loss = ((1 - cos_sim) * valid.float()).sum() / n_valid
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
