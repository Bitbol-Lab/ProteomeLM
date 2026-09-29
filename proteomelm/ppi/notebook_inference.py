from __future__ import annotations

import difflib
import logging
import math
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Literal, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

from proteomelm.modeling_proteomelm import ProteomeLMForMaskedLM

from .model import EnhancedPPIModel, prepare_ppi

logger = logging.getLogger(__name__)

QueryMode = Literal[
    "all_pairs",
    "sampled_all_pairs",
    "query_proteins",
    "explicit_pairs",
]

# Peak memory of one ProteomeLM forward with attention output, in bytes per
# (head x protein x protein) entry: the eager DistilBERT attention holds the
# masked scores and the softmax output (both bf16) at the same time. With the
# 0.5 GB overhead below this matches measured peaks within ~1-3% (ProteomeLM-S:
# 1.0 GB for 4,140 proteins, 12.8 GB measured vs 12.9 GB estimated for 19,699).
ATTENTION_BYTES_PER_ENTRY = 4.0
# Rough host-memory cost of one scored pair: pair features (float32 per head)
# are counted separately; this covers the result table row, the STRING lookup
# set entry and pandas overhead.
RESULT_BYTES_PER_PAIR = 250.0
# Fraction of free memory a run may use. The GPU estimate is tight (see above);
# host RAM also holds Python objects that are harder to predict.
GPU_MEMORY_FRACTION = 0.95
HOST_MEMORY_FRACTION = 0.85


@dataclass(frozen=True)
class PreparedProteome:
    """An encoded proteome: ESM-C inputs for attention passes, ProteomeLM logits as protein embeddings."""

    labels: List[str]
    inputs_embeds: torch.Tensor
    group_embeds: torch.Tensor
    protein_embeddings: torch.Tensor

    @property
    def n_proteins(self) -> int:
        return len(self.labels)


@dataclass(frozen=True)
class ResourceEstimate:
    """Memory plan for one notebook run (see estimate_notebook_resources)."""

    mode: QueryMode
    n_proteins: int
    candidate_pairs: int
    proteomelm_device: str
    attention_gb: float
    pair_feature_gb: float
    results_gb: float
    host_peak_gb: float
    device_peak_gb: float
    available_device_gb: Optional[float]
    available_host_gb: Optional[float]
    fits: bool
    recommendation: str
    warnings: Tuple[str, ...]
    errors: Tuple[str, ...] = ()

    def summary(self) -> str:
        """Human-readable multi-line summary for printing in the notebook."""
        def _avail(value: Optional[float]) -> str:
            return f"{value:.1f} GB available" if value is not None else "availability unknown"

        lines = [
            f"Proteins: {self.n_proteins:,} | pairs to score: {self.candidate_pairs:,}",
            f"ProteomeLM on {self.proteomelm_device}: ~{self.device_peak_gb:.1f} GB peak "
            f"({_avail(self.available_device_gb)})",
            f"Host RAM ({'ProteomeLM, ' if not self.proteomelm_device.startswith('cuda') else ''}pair features, results): "
            f"~{self.host_peak_gb:.1f} GB ({_avail(self.available_host_gb)})",
        ]
        lines.extend(f"WARNING: {message}" for message in self.warnings)
        lines.extend(f"PROBLEM: {message}" for message in self.errors)
        lines.append(self.recommendation)
        return "\n".join(lines)


def prepare_notebook_proteome(
    checkpoint: Union[Path, str],
    fasta_file: Union[Path, str],
    encoded_genome_file: Optional[Union[Path, str]] = None,
    esm_device: str = "cuda:0",
    proteomelm_device: str = "cpu",
    reload_if_possible: bool = True,
    model: Optional[ProteomeLMForMaskedLM] = None,
) -> PreparedProteome:
    """ESM-C-encode ``fasta_file`` (cached in ``encoded_genome_file``) and run ProteomeLM once.

    ``model``: ``checkpoint`` already loaded by :func:`load_proteomelm_backbone`,
    reused instead of loading it a second time (not a LoRA adapter: protein
    embeddings come from the base model).
    """
    output = prepare_ppi(
        checkpoint=checkpoint,
        fasta_file=fasta_file,
        encoded_genome_file=encoded_genome_file,
        esm_device=esm_device,
        proteomelm_device=proteomelm_device,
        include_attention=False,
        reload_if_possible=reload_if_possible,
        model=model,
    )
    return PreparedProteome(
        labels=list(output["group_labels"]),
        inputs_embeds=output["inputs_embeds"].float().contiguous(),
        group_embeds=output["group_embeds"].float().contiguous(),
        protein_embeddings=output["plm_logits"][0].float().contiguous(),
    )


def load_proteomelm_backbone(
    checkpoint: Union[Path, str],
    base_model_path: Optional[Union[Path, str]] = None,
    device: str = "cpu",
) -> ProteomeLMForMaskedLM:
    checkpoint_str = str(checkpoint)
    checkpoint_path = Path(checkpoint_str)

    if checkpoint_path.exists() and (checkpoint_path / "adapter_config.json").exists():
        if base_model_path is None:
            raise ValueError("base_model_path is required when loading a LoRA checkpoint.")
        # peft is only needed for LoRA adapters, so it is not a hard dependency.
        from peft import PeftModel

        base = ProteomeLMForMaskedLM.from_pretrained(str(base_model_path))
        model = PeftModel.from_pretrained(base, str(checkpoint_path.resolve()))
    else:
        model = ProteomeLMForMaskedLM.from_pretrained(checkpoint_str)

    return model.to(dtype=torch.bfloat16, device=device).eval()


def resolve_device(device: Optional[str]) -> str:
    """Map ``"auto"``/empty to ``"cuda"`` when a GPU is visible, else ``"cpu"``."""
    value = (device or "auto").strip()
    if value.lower() == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if value.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"Device '{value}' was requested but no CUDA GPU is visible. "
            "Use 'auto' or 'cpu' (on Colab: Runtime > Change runtime type > GPU)."
        )
    return value


def backbone_attention_shape(
    checkpoint: Union[Path, str],
    base_model_path: Optional[Union[Path, str]] = None,
) -> Tuple[int, int]:
    """Return ``(n_layers, n_heads)`` of a ProteomeLM checkpoint from its config only.

    The pair-feature dimension used by the scoring heads is ``n_layers * n_heads``.
    For a LoRA adapter directory the base model's config is read.
    """
    source = str(checkpoint)
    if Path(source).exists() and (Path(source) / "adapter_config.json").exists() and base_model_path is not None:
        source = str(base_model_path)
    config = ProteomeLMForMaskedLM.config_class.from_pretrained(source)
    return int(config.n_layers), int(config.n_heads)


def available_memory_gb(device: str) -> Optional[float]:
    """Free memory on ``device`` in GB (GPU: free CUDA memory; CPU: available RAM)."""
    try:
        if str(device).startswith("cuda"):
            if not torch.cuda.is_available():
                return None
            free_bytes, _ = torch.cuda.mem_get_info(torch.device(device))
            return free_bytes / 1e9
        import psutil

        return psutil.virtual_memory().available / 1e9
    except Exception:  # pragma: no cover - best effort only
        return None


def estimate_notebook_resources(
    n_proteins: int,
    mode: QueryMode,
    candidate_pairs: Optional[int] = None,
    include_string_annotations: bool = False,
    include_supervised_ppi: bool = False,
    pair_feature_dim: int = 48,
    dtype_bytes: int = 4,
    n_heads: int = 8,
    chunk_size: int = 1_000_000,
    proteomelm_device: str = "cpu",
    available_device_gb: Optional[float] = None,
    available_host_gb: Optional[float] = None,
) -> ResourceEstimate:
    """Estimate whether a notebook run fits in memory, before doing any heavy work.

    Two budgets matter:

    * **ProteomeLM device** (GPU or CPU RAM): one forward pass over the whole
      proteome materialises ``n_heads x N x N`` attention maps per layer, so
      memory grows quadratically with the number of proteins ``N``.
    * **Host RAM**: pair features for one chunk plus the result table for all
      scored pairs (and the STRING lookup when comparing with STRING).

    ``available_*_gb`` default to "unknown" (no fit check) so the function stays
    pure; the notebook passes :func:`available_memory_gb` values.
    """
    if candidate_pairs is None:
        if mode in {"all_pairs", "sampled_all_pairs"}:
            candidate_pairs = n_proteins * (n_proteins - 1) // 2
        else:
            candidate_pairs = 0

    attention_gb = ATTENTION_BYTES_PER_ENTRY * n_heads * n_proteins * n_proteins / 1e9
    pairs_per_chunk = min(candidate_pairs, max(chunk_size, 1))
    pair_feature_gb = pairs_per_chunk * pair_feature_dim * dtype_bytes / 1e9
    per_pair_bytes = RESULT_BYTES_PER_PAIR + (RESULT_BYTES_PER_PAIR if include_string_annotations else 0.0)
    results_gb = candidate_pairs * per_pair_bytes / 1e9
    # Supervised scoring runs in batches of <= 50k pairs; its activations are small.
    host_peak_gb = pair_feature_gb * (2.0 if include_supervised_ppi else 1.0) + results_gb
    on_gpu = str(proteomelm_device).startswith("cuda")
    # ProteomeLM weights + embeddings are small next to the attention maps; 0.5 GB covers them.
    device_peak_gb = attention_gb + 0.5
    if not on_gpu:
        host_peak_gb += device_peak_gb

    warning_messages: List[str] = []
    errors: List[str] = []
    if on_gpu and available_device_gb is not None and device_peak_gb > GPU_MEMORY_FRACTION * available_device_gb:
        errors.append(
            f"ProteomeLM needs ~{device_peak_gb:.1f} GB of GPU memory for {n_proteins:,} proteins "
            f"(attention maps grow with N^2) but only {available_device_gb:.1f} GB is free on {proteomelm_device}."
        )
    if available_host_gb is not None and host_peak_gb > HOST_MEMORY_FRACTION * available_host_gb:
        errors.append(
            f"This run needs ~{host_peak_gb:.1f} GB of RAM but only {available_host_gb:.1f} GB is available."
        )
    if candidate_pairs > 20_000_000:
        warning_messages.append(
            f"{candidate_pairs:,} pairs is a lot: expect a large result table (~{results_gb:.1f} GB) and a big CSV."
        )
    if candidate_pairs == 0:
        warning_messages.append("No protein pairs to score with the current settings.")

    if errors:
        advice = ["Options:"]
        if on_gpu:
            advice.append("run ProteomeLM on CPU (proteomelm_device='cpu', slower, uses RAM)")
        if mode in {"all_pairs", "sampled_all_pairs"}:
            advice.append("score fewer pairs (scoring_mode='query_proteins' or a smaller sampled_pairs)")
        advice.append("use a smaller proteome, or a machine/Colab runtime with more memory")
        recommendation = " ".join(advice[:1]) + " " + "; ".join(advice[1:]) + "."
    elif mode == "all_pairs":
        recommendation = "Plan OK: scoring every pair of the proteome in chunks."
    elif mode == "sampled_all_pairs":
        recommendation = "Plan OK: scoring a random sample of pairs."
    elif mode == "query_proteins":
        recommendation = "Plan OK: scoring the query proteins against the whole proteome."
    else:
        recommendation = "Plan OK: scoring only the requested pairs."

    return ResourceEstimate(
        mode=mode,
        n_proteins=n_proteins,
        candidate_pairs=candidate_pairs,
        proteomelm_device=str(proteomelm_device),
        attention_gb=attention_gb,
        pair_feature_gb=pair_feature_gb,
        results_gb=results_gb,
        host_peak_gb=host_peak_gb,
        device_peak_gb=device_peak_gb,
        available_device_gb=available_device_gb if on_gpu else available_host_gb,
        available_host_gb=available_host_gb,
        fits=not errors,
        recommendation=recommendation,
        warnings=tuple(warning_messages),
        errors=tuple(errors),
    )


def plan_notebook_run(
    n_proteins: int,
    mode: QueryMode,
    candidate_pairs: int,
    n_layers: int,
    n_heads: int,
    proteomelm_device: str = "auto",
    chunk_size: int = 1_000_000,
    include_string_annotations: bool = False,
    include_supervised_ppi: bool = False,
    auto_adjust: bool = True,
    available_gpu_gb: Optional[float] = None,
    available_host_gb: Optional[float] = None,
) -> Tuple[str, ResourceEstimate, List[str]]:
    """Pick the ProteomeLM device and check the run fits, before any heavy work.

    ``proteomelm_device="auto"`` prefers the GPU and, when ``auto_adjust`` is on,
    falls back to CPU if the attention maps would not fit in free GPU memory.
    Returns ``(device, estimate, notes)``; ``notes`` explains any automatic change.
    Pass ``available_*_gb`` explicitly for a pure (testable) call; ``None`` queries the system.
    """
    notes: List[str] = []
    requested = (proteomelm_device or "auto").strip()
    device = resolve_device(requested)

    def _estimate(dev: str) -> ResourceEstimate:
        gpu_gb = available_gpu_gb
        host_gb = available_host_gb
        if dev.startswith("cuda") and gpu_gb is None:
            gpu_gb = available_memory_gb(dev)
        if host_gb is None:
            host_gb = available_memory_gb("cpu")
        return estimate_notebook_resources(
            n_proteins=n_proteins,
            mode=mode,
            candidate_pairs=candidate_pairs,
            include_string_annotations=include_string_annotations,
            include_supervised_ppi=include_supervised_ppi,
            pair_feature_dim=n_layers * n_heads,
            n_heads=n_heads,
            chunk_size=chunk_size,
            proteomelm_device=dev,
            available_device_gb=gpu_gb,
            available_host_gb=host_gb,
        )

    estimate = _estimate(device)
    gpu_is_the_problem = device.startswith("cuda") and any("GPU memory" in err for err in estimate.errors)
    if gpu_is_the_problem and auto_adjust and requested.lower() == "auto":
        cpu_estimate = _estimate("cpu")
        if cpu_estimate.fits:
            notes.append(
                f"The proteome is too large for free GPU memory ({estimate.device_peak_gb:.1f} GB needed); "
                "running ProteomeLM on CPU instead (slower). ESM-C encoding still uses the GPU."
            )
            device, estimate = "cpu", cpu_estimate
    return device, estimate, notes


def query_candidate_pairs(n_proteins: int, query_count: int) -> int:
    return query_count * (n_proteins - query_count) + (query_count * (query_count - 1) // 2)


def compute_candidate_pairs(
    mode: QueryMode,
    n_proteins: int,
    query_indices: Optional[Sequence[int]] = None,
    explicit_pair_labels: Optional[Sequence[Tuple[str, str]]] = None,
    sampled_pairs: Optional[int] = None,
) -> int:
    if mode == "all_pairs":
        return n_proteins * (n_proteins - 1) // 2
    if mode == "sampled_all_pairs":
        if sampled_pairs is None:
            raise ValueError("sampled_pairs is required for sampled_all_pairs mode.")
        return min(sampled_pairs, n_proteins * (n_proteins - 1) // 2)
    if mode == "query_proteins":
        if not query_indices:
            raise ValueError("Provide one or more query proteins for query_proteins mode.")
        return query_candidate_pairs(n_proteins, len(set(query_indices)))
    return len(explicit_pair_labels or [])


def parse_protein_list(text: str) -> List[str]:
    """Split a free-form list of protein names (commas, semicolons, whitespace or newlines)."""
    tokens = re.split(r"[,;\s]+", text or "")
    return [token for token in (t.strip() for t in tokens) if token and not token.startswith("#")]


def _protein_name_keys(label: str, alias: Optional[str] = None) -> List[str]:
    """Exact-match keys (lower-cased) under which a protein can be looked up.

    Covers the full label, its first whitespace token (FASTA record id), the
    parts of UniProt-style ids (``sp|P0A8V2|RPOB_ECOLI``), the locus after a
    STRING taxon prefix (``511145.b3987`` -> ``b3987``), the UniProt accession
    without isoform suffix, and the gene name of the display alias
    (``"rpoB | DNA-directed RNA polymerase subunit beta"`` -> ``rpob``).
    """
    keys = [label]
    record_id = label.split(maxsplit=1)[0] if label.strip() else label
    keys.append(record_id)
    if "|" in record_id:
        keys.extend(part for part in record_id.split("|")[1:] if part)
    string_match = re.match(r"^\d+\.(.+)$", record_id)
    if string_match:
        keys.append(string_match.group(1))
    for key in list(keys):
        if re.match(r"^[A-Z0-9]+-\d+$", key):
            keys.append(key.split("-", 1)[0])
    if alias:
        keys.append(alias)
        gene = alias.split(" | ", 1)[0].strip()
        if gene:
            keys.append(gene)
    seen = set()
    result = []
    for key in keys:
        lowered = key.strip().lower()
        if lowered and lowered not in seen:
            seen.add(lowered)
            result.append(lowered)
    return result


def _describe_protein(labels: Sequence[str], aliases: Optional[Mapping[str, str]], idx: int) -> str:
    label = labels[idx]
    alias = aliases.get(label) if aliases else None
    if alias and alias != label:
        return f"{label.split(maxsplit=1)[0]} ({alias[:60]})"
    return label[:80]


def resolve_protein_queries(
    labels: Sequence[str],
    queries: Sequence[str],
    aliases: Optional[Mapping[str, str]] = None,
    max_suggestions: int = 5,
) -> List[int]:
    """Map user-typed protein names to proteome indices.

    Each query is matched, case-insensitively and in this order, against: the
    exact protein id / FASTA header and the id variants listed in
    :func:`_protein_name_keys` (UniProt accession, STRING locus, gene name from
    ``aliases``), then a unique substring of the header or alias. ``aliases``
    maps a label to a readable name such as ``"rpoB | DNA-directed RNA
    polymerase subunit beta"`` (the notebook's display labels).

    All unresolved or ambiguous queries are reported together in one
    ``ValueError`` with close-match suggestions.
    """
    exact: Dict[str, List[int]] = {}
    for idx, label in enumerate(labels):
        alias = aliases.get(label) if aliases else None
        for key in _protein_name_keys(label, alias):
            exact.setdefault(key, []).append(idx)

    searchable = [
        (label + " " + (aliases.get(label, "") if aliases else "")).lower() for label in labels
    ]
    resolved: List[int] = []
    problems: List[str] = []

    for query in queries:
        lowered = query.strip().lower()
        if not lowered:
            problems.append("empty protein name")
            continue
        hits = sorted(set(exact.get(lowered, [])))
        if not hits:
            hits = [idx for idx, text in enumerate(searchable) if lowered in text]
        if len(hits) == 1:
            resolved.append(hits[0])
            continue
        if not hits:
            suggestions = difflib.get_close_matches(lowered, list(exact.keys()), n=max_suggestions, cutoff=0.75)
            hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
            problems.append(f"'{query}' was not found in the proteome.{hint}")
            continue
        shown = "; ".join(_describe_protein(labels, aliases, idx) for idx in hits[:max_suggestions])
        more = f" (+{len(hits) - max_suggestions} more)" if len(hits) > max_suggestions else ""
        problems.append(
            f"'{query}' matches {len(hits)} proteins: {shown}{more}. Use one of these ids instead."
        )

    if problems:
        example = labels[0].split(maxsplit=1)[0] if labels else "an id"
        raise ValueError(
            "Could not resolve some query proteins:\n  - "
            + "\n  - ".join(problems)
            + f"\nUse protein ids as in the FASTA (e.g. '{example}'), UniProt accessions or gene names."
        )
    return resolved


def resolve_pair_queries(
    labels: Sequence[str],
    pair_queries: Sequence[Tuple[str, str]],
    aliases: Optional[Mapping[str, str]] = None,
) -> List[Tuple[int, int]]:
    """Resolve ``(name_a, name_b)`` pairs with :func:`resolve_protein_queries` rules."""
    flat = [name for pair in pair_queries for name in pair]
    indices = resolve_protein_queries(labels, flat, aliases=aliases)
    return [(indices[2 * i], indices[2 * i + 1]) for i in range(len(pair_queries))]


def iter_pair_chunks(
    n_proteins: int,
    chunk_size: int,
    explicit_pairs: Optional[Sequence[Tuple[int, int]]] = None,
    query_indices: Optional[Sequence[int]] = None,
    sampled_pairs: Optional[int] = None,
    seed: int = 42,
    show_progress: bool = False,
    progress_desc: str = "Scoring pair chunks",
) -> Iterator[np.ndarray]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    total_chunks = _estimate_chunk_count(
        n_proteins=n_proteins,
        chunk_size=chunk_size,
        explicit_pairs=explicit_pairs,
        query_indices=query_indices,
        sampled_pairs=sampled_pairs,
    )
    progress = _create_progress_bar(
        total=total_chunks,
        show_progress=show_progress,
        desc=progress_desc,
        unit="chunk",
    )

    try:
        if explicit_pairs is not None:
            clean_pairs: List[Tuple[int, int]] = []
            seen = set()
            for a, b in explicit_pairs:
                if a == b:
                    continue
                ia, ib = (a, b) if a < b else (b, a)
                if not (0 <= ia < n_proteins and 0 <= ib < n_proteins):
                    raise IndexError(f"Pair ({a}, {b}) is out of range for {n_proteins} proteins.")
                if (ia, ib) not in seen:
                    seen.add((ia, ib))
                    clean_pairs.append((ia, ib))
            for start in range(0, len(clean_pairs), chunk_size):
                if progress is not None:
                    progress.update(1)
                yield np.asarray(clean_pairs[start:start + chunk_size], dtype=np.int64)
            return

        if query_indices is not None:
            query_set = sorted(set(int(idx) for idx in query_indices))
            query_lookup = set(query_set)
            buffer: List[Tuple[int, int]] = []
            for query_idx in query_set:
                if not 0 <= query_idx < n_proteins:
                    raise IndexError(f"Query index {query_idx} is out of range for {n_proteins} proteins.")
                for partner_idx in range(n_proteins):
                    if partner_idx == query_idx:
                        continue
                    if partner_idx in query_lookup and partner_idx < query_idx:
                        continue
                    pair = (query_idx, partner_idx) if query_idx < partner_idx else (partner_idx, query_idx)
                    buffer.append(pair)
                    if len(buffer) >= chunk_size:
                        if progress is not None:
                            progress.update(1)
                        yield np.asarray(buffer, dtype=np.int64)
                        buffer = []
            if buffer:
                if progress is not None:
                    progress.update(1)
                yield np.asarray(buffer, dtype=np.int64)
            return

        if sampled_pairs is not None:
            max_pairs = n_proteins * (n_proteins - 1) // 2
            n_samples = min(sampled_pairs, max_pairs)
            rng = np.random.RandomState(seed)
            seen = set()
            sampled: List[Tuple[int, int]] = []
            while len(seen) < n_samples:
                ia = int(rng.randint(0, n_proteins))
                ib = int(rng.randint(0, n_proteins))
                if ia == ib:
                    continue
                pair = (ia, ib) if ia < ib else (ib, ia)
                if pair in seen:
                    continue
                seen.add(pair)
                sampled.append(pair)
                if len(sampled) >= chunk_size:
                    if progress is not None:
                        progress.update(1)
                    yield np.asarray(sampled, dtype=np.int64)
                    sampled = []
            if sampled:
                if progress is not None:
                    progress.update(1)
                yield np.asarray(sampled, dtype=np.int64)
            return

        buffer: List[Tuple[int, int]] = []
        for left_idx in range(n_proteins - 1):
            for right_idx in range(left_idx + 1, n_proteins):
                buffer.append((left_idx, right_idx))
                if len(buffer) >= chunk_size:
                    if progress is not None:
                        progress.update(1)
                    yield np.asarray(buffer, dtype=np.int64)
                    buffer = []
        if buffer:
            if progress is not None:
                progress.update(1)
            yield np.asarray(buffer, dtype=np.int64)
    finally:
        if progress is not None:
            progress.close()


def extract_attention_pair_features(
    model: ProteomeLMForMaskedLM,
    proteome: PreparedProteome,
    pair_indices: np.ndarray,
    device: str = "cpu",
) -> torch.Tensor:
    if pair_indices.ndim != 2 or pair_indices.shape[1] != 2:
        raise ValueError("pair_indices must have shape (n_pairs, 2).")
    if len(pair_indices) == 0:
        return torch.empty((0, 0), dtype=torch.float32)

    attention_modules = _collect_attention_modules(model)
    n_layers = len(attention_modules)
    n_heads = attention_modules[0].n_heads
    n_pairs = pair_indices.shape[0]

    idx_a = torch.from_numpy(pair_indices[:, 0].astype(np.int64)).to(device)
    idx_b = torch.from_numpy(pair_indices[:, 1].astype(np.int64)).to(device)
    features = np.zeros((n_pairs, n_layers, n_heads), dtype=np.float32)

    hooks = []
    for layer_idx, attention_module in enumerate(attention_modules):
        hooks.append(attention_module.register_forward_hook(_make_attention_hook(layer_idx, idx_a, idx_b, features)))

    try:
        with torch.no_grad():
            model(
                inputs_embeds=proteome.inputs_embeds[None].to(device=device, dtype=torch.bfloat16),
                group_embeds=proteome.group_embeds[None].to(device=device, dtype=torch.bfloat16),
                output_attentions=True,
            )
    finally:
        for hook in hooks:
            hook.remove()

    flattened = features.reshape(n_pairs, n_layers * n_heads)
    return torch.from_numpy(flattened)


def score_pairs_unsupervised(
    pair_features: Union[np.ndarray, torch.Tensor],
    logistic_model=None,
) -> np.ndarray:
    features_np = _to_numpy(pair_features)
    if features_np.size == 0:
        return np.empty((0,), dtype=np.float32)
    if logistic_model is None:
        return features_np.sum(axis=1, dtype=np.float32)
    expected_dim = _infer_logistic_pair_feature_dim(logistic_model)
    features_np = _match_pair_feature_dimension(features_np, expected_dim)
    return logistic_model.predict_proba(features_np)[:, 1].astype(np.float32)


def load_supervised_ppi_model(
    checkpoint_path: Union[Path, str],
    protein_embed_dim: int,
    pair_feature_dim: int,
    device: Union[str, torch.device],
    expected_backbone: Optional[str] = None,
) -> EnhancedPPIModel:
    """Load an ``EnhancedPPIModel`` checkpoint (dims are inferred from its weights).

    ``expected_backbone`` (e.g. ``"Bitbol-Lab/ProteomeLM-S"``) is compared with the
    checkpoint's ``backbone`` metadata, when present, and a mismatch is logged:
    a supervised head only makes sense on features from the backbone it was trained on.
    """
    # The bundled checkpoints store metrics/metadata next to the weights, which
    # the weights_only unpickler rejects; only load checkpoints you trust.
    checkpoint_data = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint_data["state_dict"] if isinstance(checkpoint_data, dict) and "state_dict" in checkpoint_data else checkpoint_data
    architecture = checkpoint_data.get("model_architecture") if isinstance(checkpoint_data, dict) else None
    backbone = checkpoint_data.get("backbone") if isinstance(checkpoint_data, dict) else None

    if expected_backbone and backbone and str(backbone).lower() != str(expected_backbone).lower():
        logger.warning(
            "Supervised checkpoint %s was trained on %s features, but the notebook uses %s; "
            "its scores are not meaningful for this backbone.",
            checkpoint_path, backbone, expected_backbone,
        )

    inferred_protein_dim = protein_embed_dim
    inferred_pair_dim = pair_feature_dim
    if isinstance(architecture, dict):
        inferred_protein_dim = int(architecture.get("protein_embed_dim", inferred_protein_dim))
        inferred_pair_dim = int(architecture.get("pair_feature_dim", inferred_pair_dim))

    try:
        inferred_protein_dim = int(state_dict["protein_branch.0.weight"].shape[1])
        inferred_pair_dim = int(state_dict["pair_branch.0.weight"].shape[1])
        model = EnhancedPPIModel(
            protein_embed_dim=inferred_protein_dim,
            pair_feature_dim=inferred_pair_dim,
        )
        model.load_state_dict(state_dict)
    except (KeyError, RuntimeError) as exc:
        raise RuntimeError(
            f"Could not load the supervised PPI checkpoint {checkpoint_path}: its weights do not match "
            "the current EnhancedPPIModel architecture (it was probably saved by an older version). "
            "Use an up-to-date checkpoint, or set supervised_model='none' to use the attention-based "
            f"score only.\nDetails: {str(exc).splitlines()[0]}"
        ) from exc
    return model.to(device).eval()


def score_pairs_supervised(
    model: EnhancedPPIModel,
    protein_embeddings: Union[np.ndarray, torch.Tensor],
    pair_features: Union[np.ndarray, torch.Tensor],
    pair_indices: np.ndarray,
    batch_size: int = 100000,
    device: Optional[Union[str, torch.device]] = None,
    show_progress: bool = False,
    progress_desc: str = "Supervised scoring",
) -> np.ndarray:
    if pair_indices.ndim != 2 or pair_indices.shape[1] != 2:
        raise ValueError("pair_indices must have shape (n_pairs, 2).")
    if len(pair_indices) == 0:
        return np.empty((0,), dtype=np.float64)

    features = _to_tensor(pair_features, dtype=torch.float32)
    embeddings = _to_tensor(protein_embeddings, dtype=torch.float32)
    device_obj = torch.device(device) if device is not None else next(model.parameters()).device

    features = _match_pair_feature_dimension(features, model.pair_feature_dim)
    # extract_attention_pair_features averages the two attention directions,
    # 0.5 * (a_ij + a_ji), the convention of the bundled logistic-regression
    # models. EnhancedPPIModel checkpoints are trained on the pipeline's sum
    # a_ij + a_ji (PPIFeatureExtractor._process_attention), and their BatchNorm
    # running statistics are not scale-invariant, so convert back to the sum.
    features = 2.0 * features
    embeddings = embeddings.to(device_obj)
    scores: List[np.ndarray] = []
    progress = _create_progress_bar(
        total=math.ceil(len(pair_indices) / batch_size),
        show_progress=show_progress,
        desc=progress_desc,
        unit="batch",
    )

    try:
        with torch.inference_mode():
            for start in range(0, len(pair_indices), batch_size):
                stop = min(start + batch_size, len(pair_indices))
                batch_pairs = pair_indices[start:stop]
                batch_features = features[start:stop].to(device_obj)
                idx_a = torch.from_numpy(batch_pairs[:, 0]).to(device_obj)
                idx_b = torch.from_numpy(batch_pairs[:, 1]).to(device_obj)
                logits = model(
                    F=batch_features,
                    E1=embeddings[idx_a],
                    E2=embeddings[idx_b],
                ).squeeze(-1)
                # float64: in float32 the most confident pairs (hundreds per proteome) all
                # round to exactly 1.0 and tie in the ranking.
                scores.append(torch.sigmoid(2 * logits.double()).cpu().numpy())
                if progress is not None:
                    progress.update(1)
    finally:
        if progress is not None:
            progress.close()

    return np.concatenate(scores)


RESULT_COLUMNS = [
    "protein_a", "protein_b", "protein_a_label", "protein_b_label", "idx_a", "idx_b", "unsupervised_score",
]


def primary_score_column(results_df: pd.DataFrame) -> str:
    """The score the notebook ranks by: supervised when available, else unsupervised."""
    return "supervised_score" if "supervised_score" in results_df.columns else "unsupervised_score"


def score_pair_chunks(
    backbone: ProteomeLMForMaskedLM,
    proteome: PreparedProteome,
    pair_chunks: Iterable[np.ndarray],
    device: str,
    logistic_model=None,
    supervised_model: Optional[EnhancedPPIModel] = None,
    supervised_device: Optional[str] = None,
    display_labels: Optional[Mapping[str, str]] = None,
    supervised_batch_size: int = 50_000,
) -> pd.DataFrame:
    """Score every pair in ``pair_chunks`` and return one table sorted by the primary score.

    Each chunk costs one ProteomeLM forward pass (attention features for the
    chunk's pairs), then the logistic-regression score (``unsupervised_score``;
    the raw summed attention when ``logistic_model`` is None) and, optionally,
    the supervised head (``supervised_score``).
    """
    labels = proteome.labels
    frames: List[pd.DataFrame] = []
    for pair_chunk in pair_chunks:
        if len(pair_chunk) == 0:
            continue
        pair_features = extract_attention_pair_features(backbone, proteome, pair_chunk, device=device)
        idx_a = pair_chunk[:, 0]
        idx_b = pair_chunk[:, 1]
        names_a = [labels[i] for i in idx_a]
        names_b = [labels[i] for i in idx_b]
        frame = pd.DataFrame({
            "protein_a": names_a,
            "protein_b": names_b,
            "protein_a_label": [display_labels.get(n, n) for n in names_a] if display_labels else names_a,
            "protein_b_label": [display_labels.get(n, n) for n in names_b] if display_labels else names_b,
            "idx_a": idx_a,
            "idx_b": idx_b,
            "unsupervised_score": score_pairs_unsupervised(pair_features, logistic_model=logistic_model),
        })
        if supervised_model is not None:
            frame["supervised_score"] = score_pairs_supervised(
                model=supervised_model,
                protein_embeddings=proteome.protein_embeddings,
                pair_features=pair_features,
                pair_indices=pair_chunk,
                batch_size=supervised_batch_size,
                device=supervised_device or device,
            )
        frames.append(frame)
        del pair_features

    if not frames:
        columns = RESULT_COLUMNS + (["supervised_score"] if supervised_model is not None else [])
        return pd.DataFrame(columns=columns)
    results = pd.concat(frames, ignore_index=True)
    return results.sort_values(primary_score_column(results), ascending=False, kind="stable").reset_index(drop=True)


def summarize_top_partners(
    results_df: pd.DataFrame,
    query_indices: Sequence[int],
    top_k: int,
    score_column: Optional[str] = None,
) -> pd.DataFrame:
    """Top-``top_k`` partners of each query protein as one tidy table.

    Columns: ``query``, ``rank``, ``partner``, ``partner_id``, every score column
    present in ``results_df`` (``unsupervised_score``, ``supervised_score``,
    ``string_score``) and ``query_id``. Queries keep the order they were given in.
    """
    score_column = score_column or primary_score_column(results_df)
    extra_scores = [c for c in ("unsupervised_score", "supervised_score", "string_score") if c in results_df.columns]
    columns = ["query", "rank", "partner", "partner_id", *extra_scores, "query_id"]
    if results_df.empty or top_k <= 0:
        return pd.DataFrame(columns=columns)

    idx_a = results_df["idx_a"].to_numpy()
    idx_b = results_df["idx_b"].to_numpy()
    rows: List[pd.DataFrame] = []
    for query_idx in dict.fromkeys(int(q) for q in query_indices):
        is_a = idx_a == query_idx
        mask = is_a | (idx_b == query_idx)
        if not mask.any():
            continue
        sub = results_df.loc[mask]
        sub_is_a = is_a[mask]
        top = sub.assign(_a=sub_is_a).nlargest(top_k, score_column, keep="first")
        query_first = top["_a"].to_numpy()
        table = pd.DataFrame({
            "query": np.where(query_first, top["protein_a_label"], top["protein_b_label"]),
            "rank": np.arange(1, len(top) + 1),
            "partner": np.where(query_first, top["protein_b_label"], top["protein_a_label"]),
            "partner_id": np.where(query_first, top["protein_b"], top["protein_a"]),
        })
        for column in extra_scores:
            table[column] = top[column].to_numpy()
        table["query_id"] = np.where(query_first, top["protein_a"], top["protein_b"])
        rows.append(table)
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.concat(rows, ignore_index=True)[columns]


def _collect_attention_modules(model: ProteomeLMForMaskedLM) -> List[torch.nn.Module]:
    """Self-attention modules of the layers ProteomeLM's forward actually runs, in layer order.

    ``ProteomeLMForMaskedLM`` inherits an unused ``distilbert.transformer`` stack
    from ``DistilBertForMaskedLM`` next to the ``transformer.transformer`` stack its
    forward uses; hooking both would add all-zero features for the unused layers.
    """
    candidates = [
        (name, module)
        for name, module in model.named_modules()
        if hasattr(module, "q_lin")
        and hasattr(module, "n_heads")
        and re.search(r"(^|\.)transformer(\.transformer)?\.layer\.\d+\.attention$", name)
    ]
    stacks: Dict[str, List[Tuple[str, torch.nn.Module]]] = {}
    for name, module in candidates:
        stacks.setdefault(re.sub(r"\.layer\.\d+\.attention$", "", name), []).append((name, module))
    if not stacks:
        raise RuntimeError("Could not find any ProteomeLM self-attention modules.")

    preferred = [p for p in stacks if p == "transformer.transformer" or p.endswith(".transformer.transformer")]
    if not preferred:
        preferred = [p for p in stacks if "distilbert" not in p.split(".")] or list(stacks)
    if len(preferred) > 1:
        raise RuntimeError(f"Ambiguous ProteomeLM attention stacks: {sorted(preferred)}")
    named_attention = sorted(stacks[preferred[0]], key=_layer_sort_key)
    return [module for _, module in named_attention]


def _layer_sort_key(named_module: Tuple[str, torch.nn.Module]) -> int:
    match = re.search(r"\.layer\.(\d+)\.", named_module[0])
    if match:
        return int(match.group(1))
    return math.inf


def _make_attention_hook(
    layer_idx: int,
    idx_a: torch.Tensor,
    idx_b: torch.Tensor,
    features: np.ndarray,
):
    def _hook(_module, _inputs, output):
        if not isinstance(output, (tuple, list)) or len(output) < 2:
            return output
        weights = output[1]
        if weights is None or not isinstance(weights, torch.Tensor):
            return output

        # Gather the requested entries before casting to float32: casting the
        # full (heads, N, N) map first would double the peak memory.
        attn = weights[0]
        attn_ab = attn[:, idx_a, idx_b].float().T.cpu().numpy()
        attn_ba = attn[:, idx_b, idx_a].float().T.cpu().numpy()
        features[:, layer_idx, :] = 0.5 * (attn_ab + attn_ba)
        return (output[0], None) + output[2:]

    return _hook


def _to_numpy(array: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    if isinstance(array, np.ndarray):
        return array
    return array.detach().cpu().numpy()


def _to_tensor(array: Union[np.ndarray, torch.Tensor], dtype: torch.dtype) -> torch.Tensor:
    if isinstance(array, torch.Tensor):
        return array.to(dtype=dtype)
    return torch.from_numpy(np.asarray(array)).to(dtype=dtype)


def _infer_logistic_pair_feature_dim(logistic_model) -> int:
    coef = getattr(logistic_model, "coef_", None)
    if coef is None:
        raise ValueError("The logistic model does not expose coef_, so its feature dimension cannot be inferred.")
    return int(coef.shape[1])


def _match_pair_feature_dimension(
    features: Union[np.ndarray, torch.Tensor],
    expected_dim: int,
) -> Union[np.ndarray, torch.Tensor]:
    current_dim = int(features.shape[1])
    if current_dim == expected_dim:
        return features
    if current_dim < expected_dim:
        raise ValueError(
            f"Pair features have dimension {current_dim}, but the model expects {expected_dim}."
        )

    # The bundled downstream models were trained on 48 attention features
    # (ProteomeLM-S: 6 layers x 8 heads). When a broader attention extractor is
    # used, keep the final contiguous block so the scoring still runs, but the
    # scores are not calibrated for that backbone.
    warnings.warn(
        f"Pair features have {current_dim} dimensions but the scoring model expects {expected_dim}; "
        f"using the last {expected_dim}. The bundled scoring models were fit on ProteomeLM-S, so scores "
        "from other backbones are not calibrated.",
        stacklevel=3,
    )
    return features[:, current_dim - expected_dim:]


def _estimate_chunk_count(
    n_proteins: int,
    chunk_size: int,
    explicit_pairs: Optional[Sequence[Tuple[int, int]]] = None,
    query_indices: Optional[Sequence[int]] = None,
    sampled_pairs: Optional[int] = None,
) -> int:
    """Number of chunks :func:`iter_pair_chunks` yields (for its progress bar)."""
    if explicit_pairs is not None:
        candidate_pairs = len({(a, b) if a < b else (b, a) for a, b in explicit_pairs if a != b})
    elif query_indices is not None:
        candidate_pairs = query_candidate_pairs(n_proteins, len(set(int(idx) for idx in query_indices)))
    elif sampled_pairs is not None:
        candidate_pairs = compute_candidate_pairs("sampled_all_pairs", n_proteins, sampled_pairs=sampled_pairs)
    else:
        candidate_pairs = compute_candidate_pairs("all_pairs", n_proteins)
    return math.ceil(candidate_pairs / chunk_size)


def _create_progress_bar(
    total: int,
    show_progress: bool,
    desc: str,
    unit: str,
):
    if not show_progress or tqdm is None or total <= 0:
        return None
    return tqdm(total=total, desc=desc, unit=unit)
