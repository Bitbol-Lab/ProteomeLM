import os, pickle, random, json, logging
from dataclasses import dataclass
from typing import Dict, List, Sequence

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import IterableDataset

from ..dataloaders import DataCollatorForProteomeLM
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def apply_masking(pair: dict, mask_fraction_H: float, mask_fraction_P: float) -> Dict[str, torch.Tensor]:
    """Build the masked training instance for one host-pathogen pair.

    For each segment, masks a random sample of that segment's *maskable*
    (`me=True`) proteins, sized to `round(mask_fraction * segment_length)`
    and capped by however many maskable proteins actually exist. Sampled
    fresh on every call (not a fixed window) so the same trailing proteins
    aren't masked every epoch for the whole training run, and the intended
    fraction isn't silently reduced by unmaskable proteins landing in a
    pre-selected window.
    """
    N_h: int = pair["hl"]
    N: int = pair["ie"].shape[0]
    N_p: int = N - N_h
    me: torch.Tensor = pair["me"]  # [N] bool: True = maskable

    masked = torch.zeros(N, dtype=torch.long)

    host_maskable = me[:N_h].nonzero(as_tuple=True)[0].tolist()
    n_mask_h = min(len(host_maskable), round(mask_fraction_H * N_h))
    for idx in random.sample(host_maskable, n_mask_h):
        masked[idx] = 1

    pathogen_maskable = (me[N_h:].nonzero(as_tuple=True)[0] + N_h).tolist()
    n_mask_p = min(len(pathogen_maskable), round(mask_fraction_P * N_p))
    for idx in random.sample(pathogen_maskable, n_mask_p):
        masked[idx] = 1

    return {
        "inputs_embeds": pair["ie"],
        "group_embeds": pair["ge"],
        "masked_tokens": masked,
        "source_ids": torch.cat([
            torch.zeros(N_h, dtype=torch.long),
            torch.ones(N_p, dtype=torch.long),
        ]),
    }


class ProteomeLMDatasetHP(IterableDataset):
    """Iterable dataset for host-pathogen pairs.

    Reads pre-processed pair shards (built on-cluster; the shard-building script
    isn't currently checked into this repository — see `pair_shards_dir` in
    configs/hpi_finetuning/ for where pre-built shards are expected to live).
    Each shard is a pickle list of pair dicts with bfloat16 tensors (protein
    selection, shuffling and group_embeds are all pre-computed). Masking is
    applied at yield time — 3 tensor ops — so mask fractions can differ across
    ablation conditions without rebuilding.

    Pair dict schema (stored on disk):
        'ie' : [N, 1152] bfloat16 — inputs_embeds (host ++ pathogen)
        'ge' : [N, 1152] bfloat16 — group_embeds  (host ++ pathogen)
        'me' : [N]       bool     — True where protein is maskable
        'hl' : int                — host segment length
    """

    def __init__(
        self,
        pair_shards_dir: str,
        split: str,
        mode: "int | str",
        mask_fraction_H: float = 0.2,
        mask_fraction_P: float = 0.8,
        **kwargs,  # absorb the rest of the training config
    ) -> None:
        super().__init__()
        self.split = split
        self.mask_fraction_H = mask_fraction_H
        self.mask_fraction_P = mask_fraction_P

        shard_dir = os.path.join(pair_shards_dir, str(mode), split)
        self.shard_files: List[str] = sorted(
            os.path.join(shard_dir, f)
            for f in os.listdir(shard_dir)
            if f.startswith("shard_") and f.endswith(".pkl")
        )
        with open(os.path.join(shard_dir, "metadata.json")) as f:
            meta = json.load(f)
        self._n_pairs: int = meta["n_pairs"]
        logging.info(f"{split}: {self._n_pairs} pairs across {len(self.shard_files)} shards")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._n_pairs

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        shards = list(self.shard_files)
        if worker_info is not None:
            if worker_info.num_workers > len(self.shard_files):
                logging.warning(
                    "dataloader_num_workers (%d) exceeds the number of shard files (%d) for "
                    "split=%r; worker %d will get an empty shard slice and yield nothing this epoch.",
                    worker_info.num_workers, len(self.shard_files), self.split, worker_info.id,
                )
            # Each worker gets a non-overlapping slice of shards.
            shards = shards[worker_info.id::worker_info.num_workers]

        if self.split == "train":
            random.shuffle(shards)

        for shard_path in shards:
            with open(shard_path, "rb") as f:
                pairs = pickle.load(f)
            if self.split == "train":
                random.shuffle(pairs)
            for pair in pairs:
                yield self._apply_masking(pair)

    # ------------------------------------------------------------------
    # Masking (applied at yield time so fractions can vary per ablation)
    # ------------------------------------------------------------------

    def _apply_masking(self, pair: dict) -> Dict[str, torch.Tensor]:
        return apply_masking(pair, self.mask_fraction_H, self.mask_fraction_P)


@dataclass
class DataCollatorForProteomeLMHP(DataCollatorForProteomeLM):
    return_tensors: str = "pt"

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        if not instances:
            raise ValueError("Cannot collate empty list of instances")

        required_keys = {"inputs_embeds", "group_embeds", "masked_tokens"}
        for i, instance in enumerate(instances):
            if not isinstance(instance, dict):
                raise ValueError(f"Instance {i} is not a dictionary")
            missing_keys = required_keys - set(instance.keys())
            if missing_keys:
                raise ValueError(f"Instance {i} missing keys: {missing_keys}")

        # Extract tensors
        inputs_embeds = [instance["inputs_embeds"] for instance in instances]
        group_embeds = [instance["group_embeds"] for instance in instances]
        masked_tokens = [instance["masked_tokens"] for instance in instances]
        has_source_ids = all("source_ids" in inst for inst in instances)
        if has_source_ids:
            source_ids_list = [instance["source_ids"] for instance in instances]

        # Filter out empty instances
        valid_indices = [
            i for i, (inp, grp, mask) in enumerate(zip(inputs_embeds, group_embeds, masked_tokens))
            if inp.numel() > 0 and grp.numel() > 0 and mask.numel() > 0
        ]
        if not valid_indices:
            empty_tensor = torch.empty(0, 0, 512)
            return {
                "inputs_embeds": empty_tensor,
                "group_embeds": empty_tensor,
                "masked_tokens": torch.empty(0, 0, dtype=torch.long),
                "attention_mask": torch.empty(0, 0, dtype=torch.long),
                "labels": empty_tensor,
                "source_ids": torch.empty(0, 0, dtype=torch.long),
            }

        # Keep only valid instances
        inputs_embeds = [inputs_embeds[i] for i in valid_indices]
        group_embeds = [group_embeds[i] for i in valid_indices]
        masked_tokens = [masked_tokens[i] for i in valid_indices]
        if has_source_ids:
            source_ids_list = [source_ids_list[i] for i in valid_indices]

        # Record real sequence lengths before padding (needed for attention mask)
        seq_lengths = [e.shape[0] for e in inputs_embeds]

        # Pad sequences
        inputs_embeds_padded = pad_sequence(inputs_embeds, batch_first=True)
        group_embeds_padded = pad_sequence(group_embeds, batch_first=True)
        masked_tokens_padded = pad_sequence(masked_tokens, batch_first=True, padding_value=0)

        # Convert types
        inputs_embeds_padded = inputs_embeds_padded.to(torch.bfloat16)
        group_embeds_padded = group_embeds_padded.to(torch.bfloat16)
        masked_tokens_padded = masked_tokens_padded.to(torch.long)

        # Build attention mask: 1 for real tokens, 0 for padding
        max_len = inputs_embeds_padded.shape[1]
        attention_mask = pad_sequence(
            [torch.ones(l, dtype=torch.long) for l in seq_lengths],
            batch_first=True, padding_value=0
        )  # [batch_size, max_len]

        # Create labels (original embeddings, before masking)
        labels = inputs_embeds_padded.clone()

        # Apply masking: replace masked positions with group embeddings
        mask_bool = masked_tokens_padded == 1
        inputs_embeds_padded[mask_bool] = group_embeds_padded[mask_bool]

        # Set labels to -100 for positions that are not masked (including padding)
        labels[~mask_bool] = -100

        batch = {
            "inputs_embeds": inputs_embeds_padded,
            "group_embeds": group_embeds_padded,
            "masked_tokens": masked_tokens_padded,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        if has_source_ids:
            # Pad source_ids with 0; masked positions retain their host(0)/pathogen(1) label
            source_ids_padded = pad_sequence(
                source_ids_list, batch_first=True, padding_value=0
            ).to(torch.long)
            batch["source_ids"] = source_ids_padded

        return batch


def get_shards_dataset(split: str, **config) -> ProteomeLMDatasetHP:
    """Build the HPI pair-shard dataset for *split* ("train" / "eval") from the training config."""
    return ProteomeLMDatasetHP(split=split, **config)
