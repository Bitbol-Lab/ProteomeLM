"""Per-layer essentiality classifiers on frozen ProteomeLM / ESM-C embeddings.

Pipeline for one run (``python -m experiments.essentiality.train``):

1. Embed every genome once (``embed_all_taxids``): ESM-C 600M mean embeddings,
   then a bf16 ProteomeLM forward pass keeping all hidden states. Cached per genome
   in ``embeds_folder/{ProteomeLM-{size}-210,ESMC}/{prefix}embeds_taxid{t}.pkl``.
2. For each layer, train a classifier (1, 2 or 3 linear layers) on the per-protein
   hidden state of that layer, with early stopping on the validation-fold AUPR, and
   save the best state dict to ``classifier_checkpoints_folder``.

Behaviour kept from the published runs:

* ``group_embeds`` is the protein's own ESM-C embedding (no OrthoDB lookup), for
  both the input and the group projection.
* ProteomeLM runs in bfloat16; ``plm_all_representations`` stacks
  ``hidden_states`` = (input projection, block 1, ..., block n). Layer ``i`` of a run
  is ``hidden_states[i]`` for ``i in range(n_layers)``, so index 0 is the input
  embedding and the output of the last block is never used.
* ESM-C runs use the mean-pooled output of block ``i`` (``i in range(36)``).
* Batches are whole genomes (padded, batch size 8, no shuffling); labels E=0, NE=1,
  unlabelled/out-of-split proteins are ignored (-100).
"""
import argparse
import copy
import os
import pickle
import random
import re
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from Bio import SeqIO
from sklearn.metrics import average_precision_score, balanced_accuracy_score
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from experiments.essentiality.common import (ESMC_DIM, ESMC_N_LAYERS, MODEL_IDS, PLM_SIZES, embeds_file_prefix,
                                             load_config, plm_checkpoint_path, plm_size_from_checkpoint, resolve,
                                             splits_filename, stored_embeds_folder)
from proteomelm.modeling_proteomelm import ProteomeLMConfig, ProteomeLMForMaskedLM

# --------------------------
# Classifier config and heads
# --------------------------


class ClassifierConfig(ProteomeLMConfig):
    """ProteomeLM config (n_layers, dim, hidden_dim) plus the classifier/run settings.

    Saved next to every classifier checkpoint; the evaluation reads the layer, seed,
    weights and embedding source back from it.
    """

    def __init__(self, **kwargs):
        self.dropout = kwargs.pop("dropout", 0)
        self.num_labels = kwargs.pop("num_labels", 2)  # E / NE
        self.fasta_folder = kwargs.pop("fasta_folder", None)
        self.label_folder = kwargs.pop("label_folder", None)
        self.splits_info_file = kwargs.pop("splits_info_file", None)
        self.esm_device = kwargs.pop("esm_device", "cuda:0")
        self.should_exclude_taxids = kwargs.pop("should_exclude_taxids", True)
        self.proteomeLM_checkpoint = kwargs.pop("proteomeLM_checkpoint", None)
        self.proteomeLM_checkpoint_folder = kwargs.pop("proteomeLM_checkpoint_folder", "")
        self.load_data_from_memory = kwargs.pop("load_data_from_memory", True)
        self.stored_embeds_folder = kwargs.pop("stored_embeds_folder", None)
        self.use_esmc_as_input = kwargs.pop("use_esmc_as_input", False)
        self.batch_size = kwargs.pop("batch_size", 4)
        self.classifier_device = kwargs.pop("classifier_device", "cuda:0")
        self.learning_rate = kwargs.pop("learning_rate", 0.001)
        self.n_epochs = kwargs.pop("n_epochs", 1000)
        self.patience = kwargs.pop("patience", 5)
        self.classifier_checkpoints_folder = kwargs.pop("classifier_checkpoints_folder", None)
        self.cached_dataset_folder = kwargs.pop("cached_dataset_folder", None)
        self.labels_padding_index = kwargs.pop("labels_padding_index", -100)
        self.model_id = kwargs.pop("model_id", "simpleclassifier")
        self.which_hidden_layer = kwargs.pop("which_hidden_layer", None)
        self.random_seed = kwargs.pop("random_seed", 42)
        self.optimizer = kwargs.pop("optimizer", "AdamW")
        self.weight_decay = kwargs.pop("weight_decay", 0.001)
        self.classifier_momentum = kwargs.pop("classifier_momentum", 0)
        self.wandb_project_name = kwargs.pop("wandb_project_name", None)
        self.classifier_hidden_dim = kwargs.pop("classifier_hidden_dim", 256)
        # Own field: config.max_length is a generation parameter in transformers.
        self.max_input_length = kwargs.pop("max_input_length", None)
        self.which_weights = kwargs.pop("which_weights", None)
        self.which_taxids_to_exclude = kwargs.pop("which_taxids_to_exclude", None)
        self.training_set_percentage = kwargs.pop("training_set_percentage", None)
        self.esmc_embeds_folder = kwargs.pop("esmc_embeds_folder", None)
        self.which_taxids_to_use = kwargs.pop("which_taxids_to_use", None)
        self.val_taxids_for_early_stopping = kwargs.pop("val_taxids_for_early_stopping", None)
        self.use_layernorm = kwargs.pop("use_layernorm", False)
        self.normalize_genome = kwargs.pop("normalize_genome", False)
        self.lr_gamma = kwargs.pop("lr_gamma", 1)
        self.classifier_betas = kwargs.pop("classifier_betas", [0.9, 0.999])
        self.use_lr_scheduler = kwargs.pop("use_lr_scheduler", True)

        super().__init__(**kwargs)


class ClassifierBaseModel(nn.Module):
    def __init__(self):
        super().__init__()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)


class SimpleClassifier(ClassifierBaseModel):
    """One linear layer (``--classifier-layers 1``)."""

    def __init__(self, config):
        super().__init__()
        self.layernorm = None
        if config.use_layernorm:
            self.layernorm = nn.LayerNorm(config.hidden_dim)
        # hidden_dim is the embedding width (ProteomeLM config; 1152 for ESM-C)
        self.linear = nn.Linear(config.hidden_dim, config.num_labels)
        self._init_weights()

    def forward(self, inputs, labels=None):
        x = self.layernorm(inputs) if self.layernorm is not None else inputs
        return self.linear(x)


class TwoLayerClassifier(ClassifierBaseModel):
    """Linear -> ReLU -> dropout -> linear (``--classifier-layers 2``, Fig. 5)."""

    def __init__(self, config):
        super().__init__()
        self.layernorm = None
        if config.use_layernorm:
            self.layernorm = nn.LayerNorm(config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.linear_in = nn.Linear(config.hidden_dim, config.classifier_hidden_dim)
        self.linear_out = nn.Linear(config.classifier_hidden_dim, config.num_labels)
        self.relu = nn.ReLU()
        self._init_weights()

    def forward(self, inputs, labels=None):
        x = self.layernorm(inputs) if self.layernorm is not None else inputs
        x = self.linear_in(x)
        x = self.relu(x)
        x = self.dropout(x)
        return self.linear_out(x)


class ThreeLayerClassifier(ClassifierBaseModel):
    """Two hidden layers (``--classifier-layers 3``; only used for the 1/2/3-layer comparison)."""

    def __init__(self, config):
        super().__init__()
        self.dropout = nn.Dropout(config.dropout)
        self.linear_in = nn.Linear(config.hidden_dim, config.classifier_hidden_dim)
        self.hidden = nn.Linear(config.classifier_hidden_dim, config.classifier_hidden_dim)
        self.linear_out = nn.Linear(config.classifier_hidden_dim, config.num_labels)
        self.relu = nn.ReLU()
        self._init_weights()

    def forward(self, inputs, labels=None):
        x = self.linear_in(inputs)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.hidden(x)
        x = self.relu(x)
        x = self.dropout(x)
        return self.linear_out(x)


MODEL_CLASSES = {"simpleclassifier": SimpleClassifier, "2layer": TwoLayerClassifier, "3layer": ThreeLayerClassifier}


def classifier_class(model_id: str):
    try:
        return MODEL_CLASSES[model_id]
    except KeyError:
        raise NotImplementedError("Only simpleclassifier, 2layer and 3layer are available") from None


# --------------------------
# Embeddings (ESM-C + ProteomeLM)
# --------------------------

_MODEL_CACHE: Dict[Tuple, Any] = {}


def _esmc_model(device: str):
    from proteomelm.utils.embedding import ESMC_600M, prepare_model_esm
    key = ("esmc", device)
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = prepare_model_esm(ESMC_600M, device)
    return _MODEL_CACHE[key]


def _proteomelm_model(checkpoint: str, device: str, dtype: torch.dtype):
    key = ("plm", str(checkpoint), device, dtype)
    if key not in _MODEL_CACHE:
        model = ProteomeLMForMaskedLM.from_pretrained(str(checkpoint))
        _MODEL_CACHE[key] = model.to(dtype=dtype, device=device).eval()
    return _MODEL_CACHE[key]


def prepare_ess_data(fasta_file: Optional[str] = None,
                     label_file: Optional[str] = None,
                     esm_device: str = "cuda:0",
                     proteomelm_device: str = "cuda:0",
                     proteomelm_checkpoint: Optional[str] = None,
                     only_compute_esmc_embeds: bool = False,
                     esmc_embeds_file: Optional[str] = None,
                     dtype: str = "bfloat16",
                     ) -> Tuple[Dict[str, Any], Optional[Dict[str, Dict[str, Any]]]]:
    r"""Embed one genome and load its labels.

    Proteins are sorted by decreasing length before ESM-C (so ``group_labels`` and
    every representation follow that order). ESM-C keeps the mean-pooled outputs of
    all 36 blocks for ESM-C runs and of blocks (6, 12, 18, 24, 30) otherwise; an
    existing ESM-C pickle (``esmc_embeds_file``) is reused instead of recomputing.
    Unless ``only_compute_esmc_embeds``, the ProteomeLM hidden states are added
    (``compute_proteomelm_embeds``).
    """
    if esmc_embeds_file is None:
        fasta_path = Path(fasta_file)
        assert fasta_path.exists(), f"FASTA file {fasta_path} does not exist."

        # Sort by length (less padding in the ESM-C batches)
        list_of_records = list(SeqIO.parse(fasta_path, "fasta"))
        if not list_of_records:
            raise Exception("No records present in this fasta_file.")
        list_of_records.sort(key=lambda x: len(x.seq), reverse=True)

        with tempfile.NamedTemporaryFile(suffix=".fasta", delete=True) as temp_file:
            SeqIO.write(list_of_records, temp_file.name, "fasta")
            temp_file.flush()
            if only_compute_esmc_embeds:
                which_esmc_hidden_layers_to_keep = tuple(range(ESMC_N_LAYERS))
            else:
                which_esmc_hidden_layers_to_keep = (6, 12, 18, 24, 30)
            from proteomelm.utils.embedding import encode_dataset_esmc
            with torch.no_grad():
                repr_data = encode_dataset_esmc(_esmc_model(esm_device), temp_file.name,
                                                keep_hidden_layers=which_esmc_hidden_layers_to_keep,
                                                device=esm_device)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        assert os.path.exists(esmc_embeds_file), f"File with embeddings {esmc_embeds_file} does not exist."
        with open(esmc_embeds_file, "rb") as f:
            repr_data, _ = pickle.load(f)

    assert "inputs_embeds" in repr_data and "group_embeds" in repr_data, "Generated data missing required keys."
    assert isinstance(repr_data["inputs_embeds"], torch.Tensor) and isinstance(repr_data["group_embeds"], torch.Tensor), \
        "Inputs must be torch.Tensor arrays."

    if not only_compute_esmc_embeds:
        repr_data = compute_proteomelm_embeds(data=repr_data,
                                              proteomelm_checkpoint=proteomelm_checkpoint,
                                              proteomelm_device=proteomelm_device,
                                              dtype=dtype)

    label = None
    if label_file is not None:
        with open(label_file, "rb") as f:
            label = pickle.load(f)
    return repr_data, label


def compute_proteomelm_embeds(data, proteomelm_checkpoint, proteomelm_device, dtype="bfloat16"):
    """One bf16 ProteomeLM pass over the whole genome (group_embeds = ESM-C embedding).

    Adds ``plm_all_representations`` ([n_layers + 1, N, dim]; index 0 = input
    projection), ``plm_representations`` (last hidden state) and ``plm_logits``
    (contextualized embeddings) to ``data``.
    """
    dtype = getattr(torch, dtype)
    model = _proteomelm_model(proteomelm_checkpoint, proteomelm_device, dtype)
    with torch.no_grad():
        inputs_embeds = data["inputs_embeds"][None].to(proteomelm_device, dtype=dtype)
        group_embeds = data["group_embeds"][None].to(proteomelm_device, dtype=dtype)
        output = model(inputs_embeds=inputs_embeds,
                       group_embeds=group_embeds,
                       output_attentions=False,
                       output_hidden_states=True)
        representations = output.last_hidden_states.cpu()
        logits = output.logits.cpu()
        all_representations = torch.cat([x.cpu() for x in output.hidden_states], 0)
    data["plm_attentions"] = None
    data["plm_representations"] = representations
    data["plm_logits"] = logits
    data["plm_all_representations"] = all_representations
    return data


def embed_all_taxids(fasta_folder,
                     label_folder,
                     output_folder,
                     esm_device="cuda:0",
                     proteomelm_device="cuda:0",
                     verbose=False,
                     proteomelm_checkpoint=None,
                     only_compute_esmc_embeds=False,
                     esmc_embeds_folder=None,
                     taxids_to_exclude=(),
                     taxids_to_include=None,
                     file_prefix="trained_"):
    """Write ``{output_folder}/{file_prefix}embeds_taxid{t}.pkl`` = (repr_data, labels)
    for every ``*_data_taxid{t}.fasta`` (skipping existing files). A genome that fails
    to embed is stored as ``None`` (with a warning), as in the published runs."""
    os.makedirs(output_folder, exist_ok=True)
    pattern = re.compile(r'^(.*?)_data_taxid(\d+)\.fasta$')

    for file in tqdm(sorted(os.listdir(fasta_folder)), ncols=80, desc="TaxIDs"):
        match = pattern.match(file)
        if not match:
            continue
        taxid = int(match.group(2))
        if taxid in taxids_to_exclude:
            continue
        if (taxids_to_include is not None) and (taxid not in taxids_to_include):
            continue

        fasta_file = os.path.join(fasta_folder, file)
        output_file = os.path.join(output_folder, f"{file_prefix}embeds_taxid{taxid}.pkl")

        esmc_embeds_file = None
        if (esmc_embeds_folder is not None) and os.path.exists(esmc_embeds_folder):
            esmc_embeds_file = os.path.join(esmc_embeds_folder, f"trained_embeds_taxid{taxid}.pkl")
            if not os.path.exists(esmc_embeds_file):
                esmc_embeds_file = None

        if os.path.exists(output_file):
            if verbose:
                print(f"Taxid {taxid} embeds already present at {output_file}")
            continue
        if verbose:
            print(f"Embedding taxid {taxid}...")
        label_file = os.path.join(label_folder, f"labeled_essentiality_taxid{taxid}.pkl")
        data = None
        try:
            repr_data, labels = prepare_ess_data(fasta_file,
                                                 label_file,
                                                 esm_device=esm_device,
                                                 proteomelm_device=proteomelm_device,
                                                 proteomelm_checkpoint=proteomelm_checkpoint,
                                                 only_compute_esmc_embeds=only_compute_esmc_embeds,
                                                 esmc_embeds_file=esmc_embeds_file)
            for key, value in repr_data.items():
                if isinstance(value, torch.Tensor):
                    repr_data[key] = value.to("cpu")
            data = (repr_data, labels)
        except Exception as e:
            print(f"Warning: got an exception while embedding taxid {taxid}")
            print(e)
        with open(output_file, "wb") as f:
            pickle.dump(data, f)


# --------------------------
# Baseline weights (random / statistics)
# --------------------------

def resample_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Resample the entries of ``tensor`` with replacement (keeps its value distribution)."""
    flat = tensor.view(-1)
    numel = flat.numel()
    resampled = flat[torch.randint(0, numel, (numel,))]
    return resampled.view_as(tensor)


def create_resampled_model(original_model: nn.Module) -> nn.Module:
    """'statistics' baseline: every trainable tensor resampled from its own trained values."""
    resampled_model = copy.deepcopy(original_model)
    for (name, param), (orig_name, orig_param) in zip(resampled_model.named_parameters(),
                                                      original_model.named_parameters()):
        assert name == orig_name
        if param.requires_grad:
            with torch.no_grad():
                param.copy_(resample_tensor(orig_param.data))
    return resampled_model


def make_baseline_models(sizes, seeds, baseline_dir, checkpoint_dir=None, overwrite=False):
    """Write the 'random' (fresh init) and 'statistics' (resampled) ProteomeLM baselines
    to ``{baseline_dir}/ProteomeLM-{size}-{random,statistics}-seed{seed}``."""
    for size in sizes:
        checkpoint_path = plm_checkpoint_path(size, "trained", checkpoint_dir=checkpoint_dir)
        for seed in seeds:
            random_dir = plm_checkpoint_path(size, "random", seed, baseline_dir=baseline_dir)
            stat_dir = plm_checkpoint_path(size, "statistics", seed, baseline_dir=baseline_dir)
            if overwrite or not os.path.exists(random_dir):
                set_random_seed(seed)
                config = ProteomeLMConfig.from_pretrained(str(checkpoint_path))
                ProteomeLMForMaskedLM(config).save_pretrained(random_dir)
                print(f"Saved {random_dir}")
            if overwrite or not os.path.exists(stat_dir):
                set_random_seed(seed)
                stat_model = create_resampled_model(ProteomeLMForMaskedLM.from_pretrained(str(checkpoint_path)))
                stat_model.save_pretrained(stat_dir)
                print(f"Saved {stat_dir}")


# --------------------------
# Dataset
# --------------------------

class ProteomeLMDatasetForEssentiality(Dataset):
    """One item = one genome: the chosen layer's per-protein representations and labels.

    ``split`` selects which genes keep their label (from the fold pickle: 0 test,
    1 val, 2-4 train); the others get -100. ``val-ood`` uses the genomes in
    ``val_taxids_for_early_stopping`` with all their labels.
    """

    def __init__(self,
                 config: ClassifierConfig,
                 split: str,  # train, val, test, all, val-ood
                 which_hidden_layer: Optional[int] = -1,
                 which_esmc_hidden_layer: Optional[int] = None,
                 *args, **kwargs):
        super().__init__()
        self.max_input_length = config.max_input_length
        self.fasta_folder = config.fasta_folder
        self.label_folder = config.label_folder
        self.stored_embeds_folder = config.stored_embeds_folder
        self.which_hidden_layer = which_hidden_layer
        self.which_esmc_hidden_layer = which_esmc_hidden_layer
        self.normalize_genome = config.normalize_genome
        self.labels_padding_index = config.labels_padding_index
        self.which_weights = config.which_weights
        self.embeds_prefix = embeds_file_prefix(config.which_weights, config.random_seed)

        assert split in ["train", "val", "test", "all", "val-ood"]
        self.which_split = split
        if self.which_split in ["train", "val", "test"]:
            with open(config.splits_info_file, "rb") as f:
                self.split_info = pickle.load(f)
            self.split_labels = self._get_split(self.which_split, percentage=config.training_set_percentage)

        if self.which_split == "val-ood":
            assert config.val_taxids_for_early_stopping is not None
            self.taxid_list = config.val_taxids_for_early_stopping
        elif config.which_taxids_to_use is None:
            if config.which_taxids_to_exclude is not None:
                taxids_to_exclude = config.which_taxids_to_exclude
            else:
                taxids_to_exclude = self._find_problematic_taxids_from_labels() if config.should_exclude_taxids else None
            self.taxid_list = self._get_taxid_list(config.fasta_folder, taxids_to_exclude)
        else:
            self.taxid_list = config.which_taxids_to_use

        self.vocab = {"UNK": self.labels_padding_index, "E": 0, "NE": 1}

    def _get_split(self, split: str, percentage: Optional[int] = None):
        """Gene -> 1 if its label is used in ``split``. ``percentage`` (per mille) keeps
        only that fraction of the split's genes."""
        if split == "all":
            split_labels = {key: 1 for key in self.split_info.keys()}
        elif split == "train":  # folds 2, 3, 4
            split_labels = {key: int(value >= 1.5) for key, value in self.split_info.items()}
        elif split == "val":  # fold 1
            split_labels = {key: int(value == 1) for key, value in self.split_info.items()}
        elif split == "test":  # fold 0
            split_labels = {key: int(value == 0) for key, value in self.split_info.items()}
        else:
            raise KeyError("split can only be 'all', 'train', 'val' or 'test'")

        if percentage is None:
            return split_labels
        genes = [key for key, value in split_labels.items() if value == 1]
        genes = random.sample(genes, k=int(len(genes) * (1 - percentage / 1000)))
        for gene in genes:
            split_labels[gene] = 0
        return split_labels

    def _find_problematic_taxids_from_labels(self, verbose=False):
        problematic_taxids = []
        for label_file in os.listdir(self.label_folder):
            taxid = int(re.search(r'taxid(.*?).pkl', label_file).group(1))
            with open(os.path.join(self.label_folder, label_file), "rb") as f:
                label = pickle.load(f)
            num_of_labels = sum(1 for v in label.values() if v["Essentiality"])
            if (len(label) < 10) or (num_of_labels < 10):
                problematic_taxids.append(taxid)
            if verbose:
                print(f"For taxid {taxid} ---- labelled {num_of_labels} out of {len(label)}")
        print(f"Excluding problematic taxids: {problematic_taxids}")
        return problematic_taxids

    def _get_ess(self, label_dict: Dict[str, Any]):
        r"""Last E/NE call among the OGEE calls of a gene (other calls, e.g. conditional, are ignored)."""
        ess = [item for item in label_dict["Essentiality"] if item in ("E", "NE")]
        return ess.pop() if ess else "UNK"

    def _encode_ess(self, ess_list: List[str]):
        return torch.tensor([self.vocab.get(s, self.labels_padding_index) for s in ess_list])

    def _get_taxid_list(self, files_folder: str, taxids_to_exclude: List[int] = None):
        taxid_list = [int(re.search(r'taxid(\d+)', filename).group(1)) for filename in os.listdir(files_folder)]
        if taxids_to_exclude is not None:
            for id in taxids_to_exclude:
                if id in taxid_list:
                    taxid_list.remove(id)
        return taxid_list

    def _normalize_tensor(self, tensor, eps=1e-9):
        # Per-protein standardization over the feature dimension (off in the final config).
        mean = tensor.mean(dim=-1, keepdim=True)
        std = tensor.std(dim=-1, unbiased=False, keepdim=True)
        return (tensor - mean) / (std + eps)

    def __len__(self):
        return len(self.taxid_list)

    def __getitem__(self, i: int, taxid: int = None) -> dict:
        taxid = self.taxid_list[i] if taxid is None else taxid
        embed_file = os.path.join(self.stored_embeds_folder, f"{self.embeds_prefix}embeds_taxid{taxid}.pkl")
        with open(embed_file, "rb") as f:
            data = pickle.load(f)
        if data is None:
            raise RuntimeError(f"{embed_file} is empty: embedding taxid {taxid} failed (see the embedding log)")
        repr_data, labels = data

        gene_names = repr_data["group_labels"]
        ess_labels = []
        for gene_id in gene_names:
            if (self.which_split == "all") or (self.which_split == "val-ood"):
                ess_labels.append(self._get_ess(labels[gene_id]))
            elif self.split_labels[gene_id]:
                ess_labels.append(self._get_ess(labels[gene_id]))
            else:
                ess_labels.append("UNK")

        output = {
            "plm_representations": (repr_data["plm_all_representations"][self.which_hidden_layer][:self.max_input_length]
                                    if self.which_hidden_layer is not None else None),
            "ess_labels": self._encode_ess(ess_labels)[:self.max_input_length],
            "gene_names": gene_names[:self.max_input_length],
            "esmc_representations": (repr_data["hidden_states"][self.which_esmc_hidden_layer][:self.max_input_length]
                                     if self.which_esmc_hidden_layer is not None else None),
        }
        if self.normalize_genome:
            for key in ("plm_representations", "esmc_representations"):
                if output[key] is not None:
                    output[key] = self._normalize_tensor(output[key])
        return output

    @staticmethod
    def get_collator(config, padding_index=None):
        if padding_index is None:
            padding_index = config.labels_padding_index
        return DataCollatorForProteomeLMForEssentiality(padding_index=padding_index)


class DataCollatorForProteomeLMForEssentiality:
    """Pads genomes to the longest one in the batch (features 0, labels ``padding_index``)."""

    def __init__(self, padding_index=-1):
        self.padding_index = padding_index

    def __call__(self, examples):
        examples = [ex for ex in examples if ex is not None]
        inputs_reps_list, inputs_reps_padded = None, None
        if examples[0]["plm_representations"] is not None:
            inputs_reps_list = [ex["plm_representations"].squeeze(dim=0) for ex in examples]
            inputs_reps_padded = pad_sequence(inputs_reps_list, batch_first=True, padding_value=0)
        esmc_reps_padded = None
        if examples[0]["esmc_representations"] is not None:
            esmc_reps_list = [ex["esmc_representations"].squeeze(dim=0) for ex in examples]
            esmc_reps_padded = pad_sequence(esmc_reps_list, batch_first=True, padding_value=0)
            if inputs_reps_list is None:
                inputs_reps_list = esmc_reps_list

        labels_padded = pad_sequence([ex["ess_labels"] for ex in examples], batch_first=True,
                                     padding_value=self.padding_index)
        batch_size, max_length, _ = (inputs_reps_padded if inputs_reps_padded is not None else esmc_reps_padded).shape
        attention_mask = torch.zeros(batch_size, max_length, dtype=torch.long)
        for i, seq in enumerate(inputs_reps_list):
            attention_mask[i, :seq.size(0)] = 1
        gene_names = [ex["gene_names"] + ["pad"] * (max_length - len(ex["gene_names"])) for ex in examples]
        return {
            "plm_representations": inputs_reps_padded,
            "attention_mask": attention_mask,
            "ess_labels": labels_padded.contiguous(),
            "gene_names": gene_names,
            "esmc_representations": esmc_reps_padded,
        }


class CachedBatchedDataset(Dataset):
    def __init__(self, batched_dataset_file):
        self.batches = torch.load(batched_dataset_file, weights_only=False)

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        return self.batches[idx]


# --------------------------
# Training
# --------------------------

def set_random_seed(random_seed):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)


def checkpoint_dirname(config, run_tag: str, wandb_run=None) -> str:
    """``{yymmdd}-{esmc-}{model_id}-{run_tag}`` (the published runs used the wandb run
    id in place of ``run_tag``; any name containing the model id is picked up by the
    evaluation, which selects checkpoints by their saved config)."""
    date = datetime.now().strftime("%y%m%d")
    prefix = "esmc-" if config.use_esmc_as_input else ""
    tag = wandb_run.id if wandb_run is not None else run_tag
    return f"{date}-{prefix}{config.model_id}-{tag}"


def save_model_state_dict(state_dict, config, run_tag: str, wandb_run=None) -> str:
    os.makedirs(config.classifier_checkpoints_folder, exist_ok=True)
    base = os.path.join(config.classifier_checkpoints_folder, checkpoint_dirname(config, run_tag, wandb_run))
    weights_foldername, n = base, 1
    while os.path.exists(weights_foldername):
        weights_foldername = f"{base}-v{n}"
        n += 1
    os.mkdir(weights_foldername)
    torch.save(state_dict, os.path.join(weights_foldername, "pytorch_model.bin"))
    config.save_pretrained(weights_foldername)
    if wandb_run is not None:
        wandb_run.config.weightsfile = os.path.join(weights_foldername, "pytorch_model.bin")
    print(f"Saved classifier to {weights_foldername}")
    return weights_foldername


def compute_and_save_datasets_to_cache(config, which_hidden_layer=-1, which_esmc_hidden_layer=None,
                                       overwrite_data=True, filename_suffix='', should_output_ood_val=False):
    """Pre-collate the train/val(/val-ood) batches of one layer into ``cached_dataset_folder``."""
    if config.use_esmc_as_input:
        layer_identifier = f"_esmc_layer-{which_esmc_hidden_layer}" if which_esmc_hidden_layer != -1 else "_layer-last"
    else:
        layer_identifier = f"_proteomelm_layer-{which_hidden_layer}" if which_hidden_layer != -1 else "_layer-last"
    os.makedirs(config.cached_dataset_folder, exist_ok=True)

    output_filenames = []
    sets_to_compute = ["train", "val", "val-ood"] if should_output_ood_val else ["train", "val"]
    for train_or_val in sets_to_compute:
        dataset_filename = os.path.join(config.cached_dataset_folder,
                                        f"cached_{train_or_val}_dataset{layer_identifier}{filename_suffix}.pt")
        if (not os.path.exists(dataset_filename)) or overwrite_data:
            dataset = ProteomeLMDatasetForEssentiality(config=config, split=train_or_val,
                                                       which_hidden_layer=which_hidden_layer,
                                                       which_esmc_hidden_layer=which_esmc_hidden_layer)
            dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False,
                                    collate_fn=ProteomeLMDatasetForEssentiality.get_collator(config))
            cached_batches = [batch for batch in tqdm(dataloader, desc=f"{train_or_val} dataset caching", ncols=80)]
            torch.save(cached_batches, dataset_filename)
        else:
            print(f"Using already cached file at {dataset_filename}")
        output_filenames.append(dataset_filename)
    return output_filenames


def _loader(cache_filename):
    if not os.path.exists(cache_filename):
        raise FileNotFoundError(f"Attempted to load cached dataset but {cache_filename} does not exist.")
    return DataLoader(CachedBatchedDataset(batched_dataset_file=cache_filename), batch_size=None, shuffle=False)


def _evaluate_loader(model, dataloader, loss_fct, config):
    """(mean loss, AUPR, balanced accuracy at 0.5) of p(NE) over the labelled proteins."""
    eval_loss = 0.0
    y_pred_list, y_true_list = [], []
    with torch.inference_mode():
        for data in dataloader:
            inputs = data["plm_representations"] if not config.use_esmc_as_input else data["esmc_representations"]
            inputs = inputs.to(device=config.classifier_device, dtype=getattr(torch, config.dtype))
            outputs = model(inputs)
            loss = loss_fct(outputs.view(-1, config.num_labels), data["ess_labels"].to(config.classifier_device).view(-1))
            eval_loss += loss.item()
            y_pred_list.extend(torch.softmax(outputs.reshape((-1, 2)), dim=-1, dtype=torch.float).cpu().numpy())
            y_true_list.extend(data["ess_labels"].flatten().to(torch.int).cpu().numpy())
    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)[:, 1]
    keep = y_true != config.labels_padding_index
    aupr = average_precision_score(y_true[keep], y_pred[keep])
    balanced_acc = balanced_accuracy_score(y_true[keep], y_pred[keep] > 0.5)
    return eval_loss / len(dataloader), aupr, balanced_acc


def train_model(ModelClass, config, run_tag: str, log_on_wandb: bool = False,
                train_dataset_cache_filename: str = None, val_dataset_cache_filename: str = None,
                ood_val_dataset_cache_filename: Optional[str] = None):
    """Adam on whole-genome batches; keep the epoch with the best validation AUPR
    (patience ``config.patience``). OOD validation (held-out genomes) is only logged."""
    wandb_run = None
    if log_on_wandb:
        import wandb
        which_layer = f"layer {np.arange(config.n_layers)[config.which_hidden_layer]}"
        which_layer = "esmc-" + which_layer if config.use_esmc_as_input else which_layer
        wandb_run_name = f"hidden-state-layer{which_layer.split(' ')[-1]}"
        wandb_run_name = "esmc-" + wandb_run_name if config.use_esmc_as_input else wandb_run_name
        wandb_run = wandb.init(project=config.wandb_project_name, name=wandb_run_name, reinit=True,
                               config=config.__dict__)
        wandb_run.config.which_layer = which_layer

    train_dataloader = _loader(train_dataset_cache_filename)
    val_dataloader = _loader(val_dataset_cache_filename)
    ood_val_dataloader = _loader(ood_val_dataset_cache_filename) if ood_val_dataset_cache_filename else None

    # Published behaviour: the classifier is created (and initialized) in config.dtype.
    torch.set_default_dtype(getattr(torch, config.dtype))
    model = ModelClass(config)
    model = model.to(device=config.classifier_device, dtype=getattr(torch, config.dtype))
    loss_fct = nn.CrossEntropyLoss(ignore_index=config.labels_padding_index)

    optimizer_class = getattr(torch.optim, config.optimizer, None)
    if optimizer_class is None:
        raise ValueError(f"Unsupported optimizer: {config.optimizer}")
    optimizer_args = {"lr": config.learning_rate, "weight_decay": config.weight_decay}
    if config.optimizer == "SGD":
        optimizer_args["momentum"] = config.classifier_momentum
    elif "Adam" in config.optimizer:
        optimizer_args["betas"] = tuple(config.classifier_betas)
    optimizer = optimizer_class(model.parameters(), **optimizer_args)
    scheduler = None
    if config.use_lr_scheduler:
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=10, num_training_steps=config.n_epochs)
    if wandb_run is not None:
        wandb_run.watch(model, log='all')

    best_auc = 0
    epochs_no_improve = 0
    best_model = None
    print("Starting training loop...")
    for epoch in range(config.n_epochs):
        model.train()
        running_loss = 0.0
        for data in train_dataloader:
            optimizer.zero_grad()
            inputs = data["plm_representations"] if not config.use_esmc_as_input else data["esmc_representations"]
            inputs = inputs.to(device=config.classifier_device, dtype=getattr(torch, config.dtype))
            outputs = model(inputs)
            loss = loss_fct(outputs.view(-1, config.num_labels), data["ess_labels"].to(config.classifier_device).view(-1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        if scheduler is not None:
            scheduler.step()
        train_loss = running_loss / len(train_dataloader)

        model.eval()
        val_loss, auc, balanced_acc = _evaluate_loader(model, val_dataloader, loss_fct, config)
        ood_val_loss, ood_auc, ood_balanced_acc = 0.0, 0.0, None
        if ood_val_dataloader is not None:
            ood_val_loss, ood_auc, ood_balanced_acc = _evaluate_loader(model, ood_val_dataloader, loss_fct, config)

        print(f"Epoch {epoch + 1}/{config.n_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"AUPR: {auc:.4f}, OOD Val Loss: {ood_val_loss:.4f}, OOD AUPR: {ood_auc:.4f}")
        if wandb_run is not None:
            log = {"learning_rate": optimizer.param_groups[0]['lr'],
                   "training loss": train_loss,
                   "validation loss": val_loss,
                   "validation AUPR": auc,
                   "balanced accuracy": balanced_acc}
            if ood_val_dataloader is not None:
                log.update({"out of distribution validation loss": ood_val_loss,
                            "out of distribution AUPR": ood_auc,
                            "out of distribution balanced accuracy": ood_balanced_acc})
            wandb_run.log(log)

        # Early stopping on the in-distribution validation AUPR
        if auc > best_auc:
            best_auc = auc
            best_model = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= config.patience:
                print("Early stopping triggered!")
                if wandb_run is not None:
                    wandb_run.config.early_stopping_epoch = epoch + 1
                break

    folder = save_model_state_dict(best_model, config, run_tag, wandb_run)
    model.load_state_dict(best_model)
    if wandb_run is not None:
        wandb_run.finish()
    return folder


# --------------------------
# Run configuration
# --------------------------

def model_id_for_layers(classifier_layers: int) -> str:
    return MODEL_IDS[classifier_layers]


def run_identifier(model_id, checkpoint, seed, clustering, weights, train_percentage=None) -> str:
    """Suffix of the per-run cached dataset files (unique per concurrent job)."""
    return "".join([f"_{model_id}", f"_ckpt{checkpoint}", f"_seed{seed}", f"_clust{clustering}",
                    f"_weights_{weights}", f"_train{train_percentage}" if train_percentage is not None else ""])


def build_classifier_config(cfg, checkpoint: str, seed: int, classifier_layers: int, weights: str = "trained",
                            which_hidden_layer: Optional[int] = None, device: str = "cuda:0",
                            splits_file: Optional[str] = None, checkpoint_dir=None, classifier_dir=None,
                            train_percentage=None) -> ClassifierConfig:
    """ClassifierConfig for one run: ``cfg['classifier']`` + resolved paths + run args."""
    p = cfg["paths"]
    use_esmc = checkpoint == "ESMC"
    if use_esmc and weights != "trained":
        raise ValueError("ESMC input only supports trained weights")
    if splits_file is None:
        s = cfg["split"]
        splits_file = os.path.join(p["splits_folder"], splits_filename(s["prefix"], s["threshold"], s["seed"]))
    plm_path = None if use_esmc else plm_checkpoint_path(checkpoint, weights, seed, checkpoint_dir=checkpoint_dir,
                                                         baseline_dir=p["baseline_weights_folder"])
    config_dict = dict(cfg["classifier"])
    config_dict.update({
        "fasta_folder": p["fasta_folder"],
        "label_folder": p["label_folder"],
        "classifier_checkpoints_folder": classifier_dir or p["classifier_checkpoints_folder"],
        "cached_dataset_folder": p["cached_dataset_folder"],
        "esmc_embeds_folder": p["esmc_embeds_folder"],
        "use_esmc_as_input": use_esmc,
        "which_hidden_layer": which_hidden_layer,
        "classifier_device": device,
        "esm_device": device,
        "splits_info_file": splits_file,
        "random_seed": seed,
        "model_id": model_id_for_layers(classifier_layers),
        "which_weights": weights,
        "proteomeLM_checkpoint": plm_path,
        "proteomeLM_checkpoint_folder": "",
        "stored_embeds_folder": stored_embeds_folder(p["embeds_folder"], checkpoint),
        "training_set_percentage": train_percentage,
    })
    if plm_path is not None:
        return ClassifierConfig.from_pretrained(plm_path, **config_dict)
    config = ClassifierConfig(**config_dict)
    config.n_layers = ESMC_N_LAYERS
    config.hidden_dim = ESMC_DIM
    return config


def embed_for_config(config, verbose=False, taxids_to_include=None):
    """Embed every genome used by ``config`` (train/val/test genomes + OOD validation
    genomes), or only ``taxids_to_include``."""
    taxids_to_exclude = config.which_taxids_to_exclude or []
    if config.val_taxids_for_early_stopping is not None:
        taxids_to_exclude = list(set(taxids_to_exclude) - set(config.val_taxids_for_early_stopping))
    embed_all_taxids(config.fasta_folder, config.label_folder, config.stored_embeds_folder,
                     esm_device=config.esm_device, proteomelm_device=config.esm_device,
                     proteomelm_checkpoint=config.proteomeLM_checkpoint, verbose=verbose,
                     only_compute_esmc_embeds=config.use_esmc_as_input,
                     esmc_embeds_folder=config.esmc_embeds_folder,
                     taxids_to_exclude=taxids_to_exclude,
                     taxids_to_include=taxids_to_include,
                     file_prefix=embeds_file_prefix(config.which_weights, config.random_seed))


def run_experiment(config, run_id: str, run_tag: str, log_on_wandb: bool = False, embed_only: bool = False):
    """Embed (if needed), then train one classifier per layer (all layers unless
    ``config.which_hidden_layer`` is set)."""
    set_random_seed(config.random_seed)
    embed_for_config(config)
    if embed_only:
        return []

    layers = [config.which_hidden_layer] if config.which_hidden_layer is not None else range(0, config.n_layers)
    folders = []
    for layer_to_use in layers:
        print(f"Working on layer {layer_to_use}...")
        config.which_hidden_layer = layer_to_use
        should_output_ood = config.val_taxids_for_early_stopping is not None
        cached = compute_and_save_datasets_to_cache(
            config,
            which_hidden_layer=None if config.use_esmc_as_input else layer_to_use,
            which_esmc_hidden_layer=layer_to_use if config.use_esmc_as_input else None,
            overwrite_data=True, filename_suffix=run_id, should_output_ood_val=should_output_ood)
        train_cache, val_cache = cached[0], cached[1]
        ood_cache = cached[2] if should_output_ood else None

        # Same initialization for every layer
        set_random_seed(config.random_seed)
        folders.append(train_model(classifier_class(config.model_id), config, run_tag=f"{run_tag}-layer{layer_to_use}",
                                   log_on_wandb=log_on_wandb, train_dataset_cache_filename=train_cache,
                                   val_dataset_cache_filename=val_cache, ood_val_dataset_cache_filename=ood_cache))
        for f in (train_cache, val_cache, ood_cache):
            if f is not None and os.path.exists(f):
                os.remove(f)
    return folders


def add_common_args(parser):
    parser.add_argument("--data-dir", default=None, help="Root of all data paths (default DATA_ROOT/essentiality)")
    parser.add_argument("--config", default=None, help="Config YAML (default: config.yaml next to this file)")
    parser.add_argument("--checkpoint-dir", default=None,
                        help="Local ProteomeLM weights ({dir}/ProteomeLM-{size}/checkpoint-210); "
                             "default: Hugging Face Bitbol-Lab/ProteomeLM-{size}")
    parser.add_argument("--legacy-split-pkl", default=None,
                        help="Use this fold pickle (e.g. the published all_sequences2_labelled_splits_40.pkl, "
                             "relative to --data-dir) instead of the seeded split")
    parser.add_argument("--split-seed", type=int, default=None, help="Split seed (default: config)")
    parser.add_argument("--classifier-dir", default=None,
                        help="Classifier checkpoint folder, relative to --data-dir "
                             "(default: config paths.classifier_checkpoints_folder)")
    return parser


def resolve_splits_file(cfg, legacy_split_pkl=None, split_seed=None) -> str:
    if legacy_split_pkl is not None:
        return resolve(cfg["data_dir"], legacy_split_pkl)
    s = cfg["split"]
    seed = s["seed"] if split_seed is None else split_seed
    return os.path.join(cfg["paths"]["splits_folder"], splits_filename(s["prefix"], s["threshold"], seed))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train per-layer essentiality classifiers (one run).")
    sub = parser.add_subparsers(dest="command")
    tr = add_common_args(sub.add_parser("run", help="Embed genomes and train (default command)"))
    tr.add_argument("--checkpoint", "-c", default="M", choices=("ESMC",) + PLM_SIZES)
    tr.add_argument("--seed", "-s", type=int, default=42)
    tr.add_argument("--classifier-layers", type=int, default=1, choices=sorted(MODEL_IDS))
    tr.add_argument("--weights", default="trained", choices=("trained", "random", "statistics"))
    tr.add_argument("--which-hidden-layer", type=int, default=None, help="Train only this layer (default: all)")
    tr.add_argument("--train-percentage", type=int, default=None, help="Keep this per-mille of the training genes")
    tr.add_argument("--gpu", type=int, default=0)
    tr.add_argument("--wandb", action="store_true", help="Log to Weights & Biases (off by default)")
    tr.add_argument("--embed-only", action="store_true", help="Only compute the embeddings")
    mb = add_common_args(sub.add_parser("make-baselines", help="Write random/statistics ProteomeLM weights"))
    mb.add_argument("--sizes", nargs="+", default=list(PLM_SIZES), choices=PLM_SIZES)
    mb.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45])
    raw = list(sys.argv[1:] if argv is None else argv)
    if not raw or raw[0] not in ("run", "make-baselines", "-h", "--help"):
        raw = ["run"] + raw
    args = parser.parse_args(raw)

    cfg = load_config(args.config, args.data_dir)
    args.classifier_dir = resolve(cfg["data_dir"], args.classifier_dir)
    if args.command == "make-baselines":
        make_baseline_models(args.sizes, args.seeds, cfg["paths"]["baseline_weights_folder"],
                             checkpoint_dir=args.checkpoint_dir)
        return

    splits_file = resolve_splits_file(cfg, args.legacy_split_pkl, args.split_seed)
    if not os.path.exists(splits_file) and not args.embed_only:
        raise FileNotFoundError(f"{splits_file} not found: run `python -m experiments.essentiality.data split` first")
    config = build_classifier_config(cfg, args.checkpoint, args.seed, args.classifier_layers, weights=args.weights,
                                     which_hidden_layer=args.which_hidden_layer, device=f"cuda:{args.gpu}",
                                     splits_file=splits_file, checkpoint_dir=args.checkpoint_dir,
                                     classifier_dir=args.classifier_dir, train_percentage=args.train_percentage)
    thr = cfg["split"]["threshold"]
    run_id = run_identifier(config.model_id, args.checkpoint, args.seed, thr, args.weights, args.train_percentage)
    run_tag = f"{args.checkpoint}-{args.weights}-seed{args.seed}"
    run_experiment(config, run_id, run_tag, log_on_wandb=args.wandb, embed_only=args.embed_only)


def plm_path_for_classifier(config, checkpoint_dir=None, baseline_dir=None) -> Optional[str]:
    """ProteomeLM weights behind a saved classifier config (published or new), re-resolved
    for this machine: the size/weights/seed are parsed from ``proteomeLM_checkpoint``."""
    if config.use_esmc_as_input:
        return None
    size = plm_size_from_checkpoint(config.proteomeLM_checkpoint)
    if size is None:
        raise ValueError(f"cannot parse the ProteomeLM size from {config.proteomeLM_checkpoint!r}")
    return plm_checkpoint_path(size, config.which_weights or "trained", config.random_seed,
                               checkpoint_dir=checkpoint_dir, baseline_dir=baseline_dir)


if __name__ == "__main__":
    main()
