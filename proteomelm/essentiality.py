"""Gene-essentiality prediction with ProteomeLM-ess (inference only).

ProteomeLM-ess is the essentiality classifier of the ProteomeLM paper (PNAS 2026,
Fig. 5): a two-layer head (linear -> ReLU -> dropout -> linear) on one hidden state
of a frozen ProteomeLM backbone. Scoring a proteome takes three steps:

1. ESM-C 600M embeds every protein (mean over residues, first 4096 residues).
2. ProteomeLM reads the whole proteome at once, in bfloat16, with each protein's
   own ESM-C embedding as both input and group embedding, and the head's layer
   ``hidden_states[layer]`` is kept (index 0 is the input projection).
3. The head (float32) gives p(essential) for every protein. Calls are made either
   for the top N proteins, the top fraction, or above a probability threshold.

The head is a folder (or Hugging Face repo, default ``Bitbol-Lab/ProteomeLM-ess``)
with ``config.json`` (:class:`EssentialityHeadConfig`) and ``model.safetensors``.

Example::

    from proteomelm.essentiality import EssentialityPredictor, read_fasta
    ids, seqs = read_fasta("proteome.fasta")
    table = EssentialityPredictor.from_pretrained(device="cuda").predict(ids, seqs, top_n=300)
"""
from __future__ import annotations

import json
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch import nn

DEFAULT_HEAD = "Bitbol-Lab/ProteomeLM-ess"
CONFIG_NAME = "config.json"
WEIGHTS_NAME = "model.safetensors"
MODEL_TYPE = "proteomelm-essentiality-head"
ESSENTIAL_LABEL = 0      # class index of "essential" in the head's output
ESMC_MAX_LENGTH = 4096   # residues embedded per protein (longer sequences are truncated)
# Largest proteome the head saw in training (Arabidopsis thaliana, 27,386 proteins).
LARGEST_TRAINING_PROTEOME = 27386
TSV_COLUMNS = ("protein_id", "p_essential", "rank", "predicted_essential")


@dataclass
class EssentialityHeadConfig:
    """``config.json`` of an essentiality head."""

    backbone: str = "Bitbol-Lab/ProteomeLM-L"   # ProteomeLM checkpoint (Hugging Face id or folder)
    layer: int = 8                              # index into ProteomeLM ``hidden_states`` (0 = input projection)
    input_dim: int = 1152                       # width of that hidden state
    hidden_dim: int = 2048
    dropout: float = 0.5
    num_labels: int = 2
    id2label: Dict[str, str] = field(default_factory=lambda: {"0": "essential", "1": "non-essential"})
    esm_model: str = "esmc_600m"
    backbone_dtype: str = "bfloat16"
    model_type: str = MODEL_TYPE

    @property
    def essential_index(self) -> int:
        for key, value in self.id2label.items():
            if str(value).lower() == "essential":
                return int(key)
        raise ValueError(f"id2label has no 'essential' class: {self.id2label}")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "EssentialityHeadConfig":
        known = {f for f in cls.__dataclass_fields__}
        unknown = sorted(set(data) - known)
        if unknown:
            warnings.warn(f"Ignoring unknown essentiality head config fields: {unknown}")
        values = {k: v for k, v in data.items() if k in known}
        if "id2label" in values:
            values["id2label"] = {str(k): v for k, v in values["id2label"].items()}
        return cls(**values)

    def save(self, folder: Union[str, Path]) -> Path:
        path = Path(folder) / CONFIG_NAME
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")
        return path

    @classmethod
    def load(cls, folder: Union[str, Path]) -> "EssentialityHeadConfig":
        return cls.from_dict(json.loads((Path(folder) / CONFIG_NAME).read_text()))


class EssentialityHead(nn.Module):
    """Linear -> ReLU -> dropout -> linear (same parameter names as the training code's 2-layer classifier)."""

    def __init__(self, config: EssentialityHeadConfig):
        super().__init__()
        self.linear_in = nn.Linear(config.input_dim, config.hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(config.dropout)
        self.linear_out = nn.Linear(config.hidden_dim, config.num_labels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear_out(self.dropout(self.relu(self.linear_in(inputs))))


# ---------------------------------------------------------------------------
# Loading and saving heads
# ---------------------------------------------------------------------------

def resolve_head_folder(source: Union[str, Path] = DEFAULT_HEAD, revision: Optional[str] = None) -> Path:
    """Local folder of a head: ``source`` itself if it is a folder, else a Hugging Face download."""
    if Path(source).expanduser().is_dir():
        return Path(source).expanduser()
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - huggingface_hub ships with transformers
        raise ImportError("Loading a head from the Hugging Face Hub needs `pip install huggingface_hub`.") from exc
    try:
        return Path(snapshot_download(repo_id=str(source), revision=revision, allow_patterns=[CONFIG_NAME, WEIGHTS_NAME]))
    except Exception as exc:
        raise FileNotFoundError(
            f"Could not load the essentiality head '{source}': it is neither a local folder nor a downloadable "
            f"Hugging Face repository ({type(exc).__name__}: {exc}). Pass a folder containing {CONFIG_NAME} and "
            f"{WEIGHTS_NAME}, or a Hugging Face repository id."
        ) from exc


def load_head(source: Union[str, Path] = DEFAULT_HEAD, device: Union[str, torch.device] = "cpu",
              revision: Optional[str] = None) -> Tuple[EssentialityHeadConfig, EssentialityHead]:
    """Load a head (folder or Hugging Face repo id) in float32, in eval mode."""
    from safetensors.torch import load_file

    folder = resolve_head_folder(source, revision)
    config = EssentialityHeadConfig.load(folder)
    if config.model_type != MODEL_TYPE:
        raise ValueError(f"{folder}/{CONFIG_NAME} is not an essentiality head (model_type={config.model_type!r}).")
    head = EssentialityHead(config)
    state_dict = load_file(str(folder / WEIGHTS_NAME))
    head.load_state_dict(state_dict)  # stored weights are cast to the float32 parameters
    return config, head.to(device=device, dtype=torch.float32).eval()


def save_head(folder: Union[str, Path], config: EssentialityHeadConfig, state_dict: Dict[str, torch.Tensor]) -> Path:
    """Write ``config.json`` and ``model.safetensors`` (tensors keep their dtype)."""
    from safetensors.torch import save_file

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    EssentialityHead(config).load_state_dict(state_dict)  # fails early on a shape/key mismatch
    config.save(folder)
    save_file({k: v.detach().cpu().contiguous() for k, v in state_dict.items()}, str(folder / WEIGHTS_NAME))
    return folder


# ---------------------------------------------------------------------------
# Proteome input
# ---------------------------------------------------------------------------

def read_fasta(path: Union[str, Path]) -> Tuple[List[str], List[str]]:
    """(record ids, sequences) of a FASTA file (Biopython ids: first word of each header)."""
    from Bio import SeqIO

    ids, sequences = [], []
    for record in SeqIO.parse(str(path), "fasta"):
        ids.append(record.id)
        sequences.append(str(record.seq))
    if not ids:
        raise ValueError(f"No sequences found in {path}: is it a protein FASTA file (headers starting with '>')?")
    return ids, sequences


def check_proteome(ids: Sequence[str], sequences: Sequence[str]) -> None:
    """Raise on empty input, duplicate ids or empty sequences; warn on unusually large proteomes."""
    if len(ids) != len(sequences):
        raise ValueError(f"{len(ids)} ids but {len(sequences)} sequences.")
    if not ids:
        raise ValueError("The proteome is empty.")
    seen, duplicates = set(), []
    for protein_id in ids:
        if protein_id in seen:
            duplicates.append(protein_id)
        seen.add(protein_id)
    if duplicates:
        raise ValueError(f"{len(duplicates)} duplicate protein ids (e.g. {duplicates[:5]}); ids must be unique.")
    empty = [i for i, s in zip(ids, sequences) if not s]
    if empty:
        raise ValueError(f"{len(empty)} empty sequences (e.g. {empty[:5]}).")
    if len(ids) > LARGEST_TRAINING_PROTEOME:
        warnings.warn(f"{len(ids):,} proteins is more than the largest proteome seen in training "
                      f"({LARGEST_TRAINING_PROTEOME:,}); predictions are less reliable.")


def length_sorted_order(sequences: Sequence[str]) -> List[int]:
    """Indices by decreasing sequence length (stable): the protein order used to embed a proteome."""
    return sorted(range(len(sequences)), key=lambda i: len(sequences[i]), reverse=True)


# ---------------------------------------------------------------------------
# Embedding and scoring
# ---------------------------------------------------------------------------

def load_esmc(device: Union[str, torch.device] = "cuda", model_name: str = "esmc_600m"):
    """ESM-C in bfloat16, eval mode."""
    from proteomelm.utils.embedding import prepare_model_esm

    return prepare_model_esm(model_name, str(device))


def load_backbone(checkpoint: str = "Bitbol-Lab/ProteomeLM-L", device: Union[str, torch.device] = "cuda",
                  dtype: torch.dtype = torch.bfloat16):
    """Frozen ProteomeLM backbone in ``dtype`` (bfloat16 for ProteomeLM-ess), eval mode."""
    from proteomelm.modeling_proteomelm import ProteomeLMForMaskedLM

    return ProteomeLMForMaskedLM.from_pretrained(str(checkpoint)).to(dtype=dtype, device=device).eval()


@torch.no_grad()
def esmc_embeddings(esm_model, sequences: Sequence[str], device: Union[str, torch.device] = "cuda",
                    labels: Optional[Sequence[str]] = None) -> torch.Tensor:
    """Mean ESM-C embeddings [N, 1152] (bfloat16, CPU) of ``sequences``, in the given order.

    Pass the sequences sorted by decreasing length (:func:`length_sorted_order`):
    ESM-C batches up to 16k residues, and the batch composition changes the
    bfloat16 numerics slightly, so this order reproduces the published pipeline.
    """
    from proteomelm.utils.embedding import encode_dataset_esmc

    sequences = [s[:ESMC_MAX_LENGTH] for s in sequences]
    labels = list(labels) if labels is not None else [str(i) for i in range(len(sequences))]
    out = encode_dataset_esmc(esm_model, data=(labels, sequences), keep_hidden_layers=None, device=str(device))
    return out["inputs_embeds"]


@torch.no_grad()
def backbone_hidden_state(backbone, esmc: torch.Tensor, layer: int,
                          device: Union[str, torch.device] = "cuda") -> torch.Tensor:
    """ProteomeLM ``hidden_states[layer]`` [N, dim] for a whole proteome (one forward pass).

    Each protein's own ESM-C embedding is both its input and its group embedding.
    """
    dtype = next(backbone.parameters()).dtype
    x = esmc[None].to(device=device, dtype=dtype)
    output = backbone(inputs_embeds=x, group_embeds=x, output_attentions=False, output_hidden_states=True)
    return output.hidden_states[layer][0].cpu()


@torch.no_grad()
def head_probabilities(head: EssentialityHead, hidden: torch.Tensor) -> np.ndarray:
    """Softmax class probabilities [N, num_labels] (float32) of the head on ``hidden``."""
    device = next(head.parameters()).device
    logits = head(hidden.to(device=device, dtype=torch.float32))
    return torch.softmax(logits.reshape(-1, logits.shape[-1]).to(torch.float), dim=-1, dtype=torch.float).cpu().numpy()


def essentiality_ranks(p_nonessential: np.ndarray) -> np.ndarray:
    """1-based rank of each protein, most essential first (increasing p(non-essential))."""
    order = np.argsort(np.asarray(p_nonessential))
    ranks = np.empty(len(order), dtype=np.int64)
    ranks[order] = np.arange(1, len(order) + 1)
    return ranks


def resolve_top_n(n_proteins: int, top_n: Optional[int] = None, top_fraction: Optional[float] = None) -> Optional[int]:
    """Number of proteins to call essential: ``top_n`` (capped at N) or round(``top_fraction`` x N);
    None when neither is given."""
    if top_n is not None and top_fraction is not None:
        raise ValueError("Use either top_n or top_fraction, not both.")
    if top_n is not None:
        if top_n < 0:
            raise ValueError(f"top_n must be >= 0, got {top_n}.")
        return min(int(top_n), n_proteins)
    if top_fraction is not None:
        if not 0 <= top_fraction <= 1:
            raise ValueError(f"top_fraction must be in [0, 1], got {top_fraction}.")
        return int(round(top_fraction * n_proteins))
    return None


def essentiality_table(protein_ids: Sequence[str], probabilities: np.ndarray, top_n: Optional[int] = None,
                       threshold: Optional[float] = None, essential_index: int = ESSENTIAL_LABEL) -> pd.DataFrame:
    """Result table sorted by rank: ``protein_id``, ``p_essential``, ``rank`` and, with a
    calling rule, ``predicted_essential``.

    Ranks (and so the top-N calls) order proteins by increasing p(non-essential), as
    in the paper's whole-genome predictions; ``threshold`` calls p(essential) >= threshold.
    """
    if top_n is not None and threshold is not None:
        raise ValueError("Use either top_n or threshold, not both.")
    probabilities = np.asarray(probabilities)
    if probabilities.ndim != 2 or probabilities.shape[1] != 2:
        raise ValueError(f"expected [N, 2] class probabilities, got shape {probabilities.shape}")
    p_essential = probabilities[:, essential_index]
    p_nonessential = probabilities[:, 1 - essential_index]
    table = pd.DataFrame({"protein_id": list(protein_ids), "p_essential": p_essential.astype(np.float64),
                          "rank": essentiality_ranks(p_nonessential)})
    if top_n is not None:
        table["predicted_essential"] = table["rank"] <= top_n
    elif threshold is not None:
        table["predicted_essential"] = table["p_essential"] >= threshold
    return table.sort_values("rank", kind="stable").reset_index(drop=True)


def write_table(table: pd.DataFrame, path: Union[str, Path]) -> Path:
    """Write a result table as TSV (probabilities with 9 significant digits: exact for float32)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, sep="\t", index=False, float_format="%.9g")
    return path


def _oom_message(n_proteins: int, what: str) -> str:
    return (f"Out of GPU memory while running {what} on {n_proteins:,} proteins. Free GPU memory, run on a larger "
            f"GPU, or run on CPU (slower).")


class EssentialityPredictor:
    """Head + ESM-C + ProteomeLM backbone, loaded once, to score whole proteomes."""

    def __init__(self, config: EssentialityHeadConfig, head: EssentialityHead, device: Union[str, torch.device] = "cuda",
                 esm_device: Optional[Union[str, torch.device]] = None, backbone=None, esm_model=None):
        self.config = config
        self.head = head
        self.device = str(device)
        self.esm_device = str(esm_device or device)
        self._backbone = backbone
        self._esm_model = esm_model
        self.timings: Dict[str, float] = {}

    @classmethod
    def from_pretrained(cls, head: Union[str, Path] = DEFAULT_HEAD, device: Union[str, torch.device] = "cuda",
                        esm_device: Optional[Union[str, torch.device]] = None, revision: Optional[str] = None
                        ) -> "EssentialityPredictor":
        config, module = load_head(head, device=device, revision=revision)
        return cls(config, module, device=device, esm_device=esm_device)

    @property
    def backbone(self):
        if self._backbone is None:
            self._backbone = load_backbone(self.config.backbone, self.device, getattr(torch, self.config.backbone_dtype))
        return self._backbone

    @property
    def esm_model(self):
        if self._esm_model is None:
            self._esm_model = load_esmc(self.esm_device, self.config.esm_model)
        return self._esm_model

    def unload_esmc(self) -> None:
        """Free ESM-C (e.g. once the proteome is embedded, to leave the GPU to ProteomeLM)."""
        self._esm_model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def embed(self, sequences_sorted: Sequence[str], ids_sorted: Optional[Sequence[str]] = None) -> torch.Tensor:
        """ESM-C embeddings of length-sorted sequences (see :func:`esmc_embeddings`)."""
        start = time.time()
        try:
            out = esmc_embeddings(self.esm_model, sequences_sorted, self.esm_device, ids_sorted)
        except torch.cuda.OutOfMemoryError as exc:
            raise RuntimeError(_oom_message(len(sequences_sorted), "ESM-C")) from exc
        self.timings["esmc_s"] = time.time() - start
        return out

    def probabilities(self, esmc_sorted: torch.Tensor) -> np.ndarray:
        """Class probabilities [N, 2] from length-sorted ESM-C embeddings."""
        start = time.time()
        try:
            hidden = backbone_hidden_state(self.backbone, esmc_sorted, self.config.layer, self.device)
        except torch.cuda.OutOfMemoryError as exc:
            raise RuntimeError(_oom_message(len(esmc_sorted), "ProteomeLM")) from exc
        probs = head_probabilities(self.head, hidden)
        self.timings["proteomelm_s"] = time.time() - start
        return probs

    def predict(self, protein_ids: Sequence[str], sequences: Sequence[str], top_n: Optional[int] = None,
                top_fraction: Optional[float] = None, threshold: Optional[float] = None,
                esmc: Optional[torch.Tensor] = None) -> pd.DataFrame:
        """Score a whole proteome and return :func:`essentiality_table` (sorted by rank).

        ``esmc``: precomputed ESM-C embeddings in the input order (skips ESM-C).
        """
        check_proteome(protein_ids, sequences)
        order = length_sorted_order(sequences)
        ids_sorted = [protein_ids[i] for i in order]
        if esmc is None:
            esmc_sorted = self.embed([sequences[i] for i in order], ids_sorted)
        else:
            esmc_sorted = esmc[torch.as_tensor(order)]
        probs = self.probabilities(esmc_sorted)
        n = resolve_top_n(len(ids_sorted), top_n, top_fraction)
        return essentiality_table(ids_sorted, probs, top_n=n, threshold=threshold,
                                  essential_index=self.config.essential_index)


def estimate_memory_gb(n_proteins: int, dim: int = 1152, n_layers: int = 18, n_heads: int = 12,
                       on_gpu: bool = True, backbone_params: float = 328e6) -> Dict[str, float]:
    """Rough peak memory (GB) of the two model steps for a proteome of ``n_proteins``.

    ESM-C 600M (bf16 weights + batches of up to 16k residues) needs ~3 GB whatever the
    proteome size. ProteomeLM keeps its bf16 weights and every hidden state
    ([n_layers + 1, N, dim]); on GPU the attention is computed without materializing
    the N x N maps, on CPU the ``n_heads`` x N x N scores (bf16, ~2 copies) dominate.
    """
    weights = 2 * backbone_params / 1e9
    hidden = 2 * (n_layers + 1) * n_proteins * dim / 1e9 + 4 * 8 * n_proteins * dim / 1e9  # + FFN activations
    attention = 0.0 if on_gpu else 4 * n_heads * n_proteins * n_proteins / 1e9
    return {"esmc_gb": 3.0, "proteomelm_gb": weights + hidden + attention + 0.3}


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def build_arg_parser(prog: Optional[str] = None):
    import argparse

    parser = argparse.ArgumentParser(
        prog=prog, description="Predict which proteins of a whole proteome are essential (ProteomeLM-ess).")
    parser.add_argument("--fasta", required=True, help="Protein FASTA of one whole proteome")
    parser.add_argument("--head", default=DEFAULT_HEAD,
                        help=f"Head folder (config.json + model.safetensors) or Hugging Face repo id (default {DEFAULT_HEAD})")
    parser.add_argument("--revision", default=None, help="Hugging Face revision of the head")
    parser.add_argument("--out", default=None, help="Output TSV (default: <fasta name>.essentiality.tsv)")
    rule = parser.add_mutually_exclusive_group()
    rule.add_argument("--top-n", type=int, default=None, help="Call the N highest-ranked proteins essential")
    rule.add_argument("--top-fraction", type=float, default=None, help="Call this fraction of the proteome essential")
    rule.add_argument("--threshold", type=float, default=None, help="Call p_essential >= threshold essential")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index; -1 runs on CPU (slow)")
    return parser


def run_cli(args, load_head_fn=load_head) -> pd.DataFrame:
    """Run the command line (``load_head_fn(source, device, revision)`` loads the head)."""
    device = "cpu" if args.gpu < 0 else f"cuda:{args.gpu}"
    if device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU is visible; use --gpu -1 to run on CPU.")
    out = Path(args.out) if args.out else Path(Path(args.fasta).name).with_suffix(".essentiality.tsv")
    ids, sequences = read_fasta(args.fasta)
    config, head = load_head_fn(args.head, device, args.revision)
    predictor = EssentialityPredictor(config, head, device=device)
    print(f"{len(ids):,} proteins; head {args.head} ({config.backbone}, hidden_states[{config.layer}]) on {device}")
    table = predictor.predict(ids, sequences, top_n=args.top_n, top_fraction=args.top_fraction,
                              threshold=args.threshold)
    write_table(table, out)
    timing = ", ".join(f"{k[:-2]} {v:.0f} s" for k, v in predictor.timings.items())
    called = f"; {int(table['predicted_essential'].sum()):,} predicted essential" \
        if "predicted_essential" in table else ""
    print(f"Wrote {out} ({timing}{called})")
    return table


def main(argv=None) -> None:
    run_cli(build_arg_parser("python -m proteomelm.essentiality").parse_args(argv))


__all__ = [
    "DEFAULT_HEAD", "EssentialityHeadConfig", "EssentialityHead", "EssentialityPredictor", "load_head", "save_head",
    "read_fasta", "check_proteome", "length_sorted_order", "load_esmc", "load_backbone", "esmc_embeddings",
    "backbone_hidden_state", "head_probabilities", "essentiality_ranks", "resolve_top_n", "essentiality_table",
    "write_table", "estimate_memory_gb", "build_arg_parser", "run_cli",
]


if __name__ == "__main__":
    main()
