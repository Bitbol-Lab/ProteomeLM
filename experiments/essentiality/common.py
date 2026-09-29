"""Shared configuration and path helpers for the essentiality experiment.

Every data path in ``config.yaml`` is relative to a single ``--data-dir``
(default ``DATA_ROOT/essentiality``); absolute paths are kept as they are. The
filenames are the ones used on the original server, so pointing ``--data-dir``
at the original data folder reuses the downloaded/processed files as-is.
"""
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "config.yaml"

PLM_SIZES = ("XS", "S", "M", "L")
CHECKPOINTS = ("ESMC",) + PLM_SIZES
WEIGHTS = ("trained", "random", "statistics")
MODEL_IDS = {1: "simpleclassifier", 2: "2layer", 3: "3layer"}
HF_REPO = "Bitbol-Lab/ProteomeLM-{size}"
# Layout of local checkpoints (--checkpoint-dir), as on the original server
LOCAL_CHECKPOINT = "ProteomeLM-{size}/checkpoint-210"
BASELINE_CHECKPOINT = "ProteomeLM-{size}-{weights}-seed{seed}"
# ESM-C 600M (ESMC input): 36 transformer blocks, 1152-d
ESMC_N_LAYERS = 36
ESMC_DIM = 1152


def default_data_dir() -> Path:
    from proteomelm.ppi.config import DATA_ROOT
    return Path(DATA_ROOT) / "essentiality"


def resolve(data_dir, path) -> Optional[str]:
    """``path`` relative to ``data_dir`` (absolute paths and None pass through)."""
    if path is None:
        return None
    path = os.path.expanduser(str(path))
    return os.path.normpath(path if os.path.isabs(path) else os.path.join(str(data_dir), path))


def load_config(config_path=None, data_dir=None) -> Dict[str, Any]:
    """Read ``config.yaml`` and resolve its ``paths`` block against ``data_dir``.

    Returns the raw YAML dict with ``paths`` replaced by absolute paths and
    ``data_dir`` added.
    """
    config_path = Path(config_path) if config_path is not None else DEFAULT_CONFIG
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    for key in ("paths", "classifier", "data_download"):
        if key not in cfg:
            raise KeyError(f"{config_path}: missing top-level section '{key}'")
    data_dir = Path(data_dir) if data_dir is not None else default_data_dir()
    cfg["data_dir"] = str(data_dir)
    cfg["paths"] = {k: resolve(data_dir, v) for k, v in cfg["paths"].items()}
    return cfg


def splits_filename(prefix: str, threshold: int, split_seed: Optional[int] = None) -> str:
    """Fold-assignment pickle name. ``split_seed=None`` is the legacy (published) name,
    ``{prefix}_{threshold}.pkl``; a seeded split adds ``_seed{S}``."""
    if split_seed is None:
        return f"{prefix}_{threshold}.pkl"
    return f"{prefix}_{threshold}_seed{split_seed}.pkl"


def plm_checkpoint_path(size: str, weights: str = "trained", seed: Optional[int] = None,
                        checkpoint_dir=None, baseline_dir=None) -> str:
    """Where the ProteomeLM weights for one run come from.

    trained: ``{checkpoint_dir}/ProteomeLM-{size}/checkpoint-210`` if a local
    checkpoint dir is given, else the Hugging Face repo ``Bitbol-Lab/ProteomeLM-{size}``
    (identical weights). random/statistics: the baseline models written by
    ``train.py make-baselines`` under ``baseline_dir``.
    """
    if size not in PLM_SIZES:
        raise ValueError(f"unknown ProteomeLM size {size!r}")
    if weights == "trained":
        if checkpoint_dir:
            return os.path.join(os.path.expanduser(str(checkpoint_dir)), LOCAL_CHECKPOINT.format(size=size))
        return HF_REPO.format(size=size)
    if weights in ("random", "statistics"):
        if seed is None or baseline_dir is None:
            raise ValueError("random/statistics weights need a seed and baseline_dir")
        return os.path.join(str(baseline_dir), BASELINE_CHECKPOINT.format(size=size, weights=weights, seed=seed))
    raise ValueError(f"unknown weights {weights!r}")


_PLM_RE = re.compile(r"ProteomeLM-(XS|S|M|L)(?:/|$|-(?:random|statistics)-seed\d+)")


def plm_size_from_checkpoint(proteomelm_checkpoint: Optional[str]) -> Optional[str]:
    """Parse the ProteomeLM size from a classifier config's ``proteomeLM_checkpoint``.

    Handles the published names (``ProteomeLM-L/checkpoint-210``,
    ``ProteomeLM-baseline/ProteomeLM-L-random-seed42``) and the new ones
    (``Bitbol-Lab/ProteomeLM-L`` or a local path). ``None`` means ESM-C input.
    """
    if proteomelm_checkpoint is None:
        return None
    m = _PLM_RE.search(str(proteomelm_checkpoint))
    return m.group(1) if m else None


def stored_embeds_folder(embeds_root: str, checkpoint: str) -> str:
    """Per-model embedding cache folder (published names: ``ProteomeLM-{size}-210``, ``ESMC``)."""
    return os.path.join(embeds_root, "ESMC" if checkpoint == "ESMC" else f"ProteomeLM-{checkpoint}-210")


def embeds_file_prefix(weights: str, seed: Optional[int] = None) -> str:
    """Prefix of the per-genome embedding pickles.

    ``trained_`` matches the published files. Baseline (random/statistics)
    embeddings include the weight seed: the original code wrote them without it
    and relied on the driver deleting them between seeds.
    """
    if weights == "trained":
        return "trained_"
    return f"{weights}-seed{seed}_"
