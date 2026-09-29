"""Data, configuration and I/O helpers for ``notebooks/ppi_prediction_efficient.ipynb``.

Everything here is plain Python so the notebook stays short and the logic is
testable. Model-side helpers (encoding, attention features, scoring) live in
``notebook_inference.py``.

Where files go
--------------
All notebook files live under one *work directory* (:func:`get_cache_dir`):

* in a clone of the repository: ``<repo>/notebooks/cache/`` (gitignored);
* otherwise (pip install, Google Colab): ``./proteomelm_ppi/`` in the current
  directory (``/content/proteomelm_ppi`` on Colab);
* or wherever ``PROTEOMELM_NOTEBOOK_DIR`` / :func:`set_cache_dir` points.

It contains ``downloads/`` (STRING / UniProt files, reused across runs),
``embeddings/`` (ESM-C encoded proteomes), ``assets/`` (bundled scoring models
fetched from GitHub when not running from a clone) and ``results/``.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import io
import logging
import os
import pickle
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import pandas as pd
import requests
import torch

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

logger = logging.getLogger(__name__)

# Source of the bundled scoring models (data/interactomes/*) when the notebook
# does not run from a clone. The notebook passes its own PROTEOMELM_REPO_URL /
# PROTEOMELM_REF, so this is only the library default.
PUBLIC_REPO_URL = "https://github.com/Bitbol-Lab/ProteomeLM"
DEFAULT_REPO_REF = "main"
BUNDLED_ASSET_DIR = "data/interactomes"
BUNDLED_MODEL_FILES: Dict[str, Dict[str, str]] = {
    "logreg": {
        "human": "logistic_regression_model_human.pkl",
        "pathogens": "logistic_regression_model_pathogens.pkl",
    },
    # multispecies first: on 18 bacteria unseen in training it beats dscript (human only) on all 18,
    # and it matches or beats it on animals; bernett does not transfer across species.
    "supervised": {
        "multispecies": "enhanced_ppi_model_multispecies.pt",
        "dscript": "enhanced_ppi_model_dscript.pt",
        "bernett": "enhanced_ppi_model_bernett.pt",
    },
}
# Shown as "<name> (recommended)" in the notebook's model menus; the suffix is dropped when reading the choice.
RECOMMENDED_MODELS = {"supervised": "multispecies"}
# The bundled logistic-regression and supervised models were fit on these features.
BUNDLED_MODELS_BACKBONE = "Bitbol-Lab/ProteomeLM-S"

STRING_VERSION = "v12.0"
STRING_DOWNLOAD_URL = "https://stringdb-downloads.org/download"
STRING_API_URL = "https://version-12-0.string-db.org/api"
UNIPROT_STREAM_URL = "https://rest.uniprot.org/uniprotkb/stream"

DEFAULT_ENCODED_GENOME_FILE = "temp_proteome_encoded.pt"


@dataclass(frozen=True)
class NotebookRuntimeData:
    labels: List[str]
    sequences: List[str]
    fasta_for_inference: str
    annotation_organism_id: Optional[str]
    mapping_required: bool
    organism_name: str
    encoded_genome_file: str
    protein_annotations: Dict[str, Dict[str, Optional[str]]]
    display_labels: Dict[str, str]

    @property
    def record_ids(self) -> List[str]:
        """FASTA record ids (first header token), the protein ids ProteomeLM reports."""
        return [fasta_record_id(label) for label in self.labels]


# ---------------------------------------------------------------------------
# Environment and paths
# ---------------------------------------------------------------------------

def in_colab() -> bool:
    """True inside a Google Colab kernel."""
    return "google.colab" in sys.modules


def find_repo_root() -> Optional[Path]:
    """Root of the ProteomeLM source checkout this package was imported from, if any.

    Returns None for a regular (non-editable) pip install, e.g. on Colab.
    """
    root = Path(__file__).resolve().parents[2]
    has_project_file = (root / "pyproject.toml").is_file() or (root / "setup.py").is_file()
    if has_project_file and (root / "proteomelm" / "__init__.py").is_file():
        return root
    return None


def default_cache_dir() -> Path:
    """Work directory used when :func:`set_cache_dir` was not called (see module docstring)."""
    env_dir = os.environ.get("PROTEOMELM_NOTEBOOK_DIR", "").strip()
    if env_dir:
        return Path(env_dir).expanduser().resolve()
    repo_root = find_repo_root()
    if repo_root is not None:
        return repo_root / "notebooks" / "cache"
    return (Path.cwd() / "proteomelm_ppi").resolve()


_CACHE_DIR_OVERRIDE: Optional[Path] = None


def set_cache_dir(path: Optional[Union[Path, str]]) -> Path:
    """Use ``path`` as the notebook work directory (``None`` restores the default)."""
    global _CACHE_DIR_OVERRIDE
    _CACHE_DIR_OVERRIDE = Path(path).expanduser().resolve() if path else None
    return get_cache_dir()


def get_cache_dir() -> Path:
    """Absolute notebook work directory (created on demand by the helpers that write to it)."""
    return _CACHE_DIR_OVERRIDE if _CACHE_DIR_OVERRIDE is not None else default_cache_dir()


def _cache_subdir(name: str) -> Path:
    path = get_cache_dir() / name
    path.mkdir(parents=True, exist_ok=True)
    return path


class _NotebookLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        message = record.getMessage()
        return message if record.levelno < logging.WARNING else f"{record.levelname.capitalize()}: {message}"


def configure_notebook_logging(verbose: bool = False) -> None:
    """Quiet third-party warnings/log spam; keep ProteomeLM progress messages."""
    import warnings

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s", force=True)
    # ProteomeLM progress messages go to stdout (plain cell output), warnings keep a prefix.
    package_logger = logging.getLogger("proteomelm")
    package_logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    package_logger.handlers = [h for h in package_logger.handlers if not getattr(h, "_proteomelm_notebook", False)]
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_NotebookLogFormatter())
    handler._proteomelm_notebook = True
    package_logger.addHandler(handler)
    package_logger.propagate = False
    for name in ("transformers", "huggingface_hub", "esm", "urllib3", "httpx", "filelock", "fsspec"):
        logging.getLogger(name).setLevel(logging.ERROR)
    try:
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
    except Exception:  # pragma: no cover - transformers is a hard dependency
        pass
    if not verbose:
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", message=".*pynvml.*")
        warnings.filterwarnings("ignore", message=".*TypedStorage is deprecated.*")
        warnings.filterwarnings("ignore", message=".*IProgress not found.*")


# ---------------------------------------------------------------------------
# Downloads
# ---------------------------------------------------------------------------

def _download_file(
    url: str,
    dest: Union[Path, str],
    desc: Optional[str] = None,
    not_found_message: Optional[str] = None,
    params: Optional[Mapping[str, str]] = None,
    timeout: int = 120,
    force: bool = False,
) -> Path:
    """Stream ``url`` to ``dest`` (atomic, with a progress bar); reuse ``dest`` if present."""
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0 and not force:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    try:
        with requests.get(url, params=params, stream=True, timeout=timeout) as response:
            if response.status_code == 404:
                raise FileNotFoundError(not_found_message or f"Not found (HTTP 404): {response.url}")
            response.raise_for_status()
            total = int(response.headers.get("content-length") or 0)
            show = tqdm is not None and (total == 0 or total > 2_000_000)
            bar = tqdm(total=total or None, unit="B", unit_scale=True, desc=desc or dest.name, leave=False) if show else None
            with open(tmp, "wb") as handle:
                for chunk in response.iter_content(chunk_size=1 << 20):
                    handle.write(chunk)
                    if bar is not None:
                        bar.update(len(chunk))
            if bar is not None:
                bar.close()
        tmp.replace(dest)
    except requests.RequestException as exc:
        raise ConnectionError(f"Download failed for {url}: {exc}. Check your internet connection.") from exc
    finally:
        tmp.unlink(missing_ok=True)
    return dest


def _open_text(path: Union[Path, str]):
    """Open a text file, transparently decompressing ``.gz``."""
    path = Path(path)
    if path.suffix == ".gz":
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _check_taxon_id(value: Union[str, int], what: str) -> str:
    text = str(value).strip()
    if not re.fullmatch(r"\d+", text):
        raise ValueError(
            f"{what} must be a numeric NCBI taxon id (e.g. 511145 for E. coli K-12, 9606 for human), got '{value}'."
        )
    return text


def string_file(organism: Union[str, int], kind: str) -> Path:
    """Download (once) a STRING per-organism file and return its local ``.gz`` path.

    ``kind`` is a STRING download family, e.g. ``protein.sequences``,
    ``protein.links.detailed`` or ``protein.info``.
    """
    organism = _check_taxon_id(organism, "STRING organism id")
    extension = "fa.gz" if kind == "protein.sequences" else "txt.gz"
    filename = f"{organism}.{kind}.{STRING_VERSION}.{extension}"
    url = f"{STRING_DOWNLOAD_URL}/{kind}.{STRING_VERSION}/{filename}"
    return _download_file(
        url,
        _cache_subdir("downloads") / filename,
        desc=f"STRING {kind} ({organism})",
        not_found_message=(
            f"STRING {STRING_VERSION} has no organism with id {organism}. Look up the id of your organism at "
            "https://string-db.org (Search > Organisms) or use another protein source."
        ),
    )


def string_fasta_path(organism: Union[str, int]) -> Path:
    """Decompressed STRING proteome FASTA for ``organism`` (downloaded once)."""
    gz_path = string_file(organism, "protein.sequences")
    fasta_path = gz_path.with_suffix("")  # strip .gz
    if not fasta_path.exists() or fasta_path.stat().st_size == 0:
        tmp = fasta_path.with_name(fasta_path.name + ".part")
        with gzip.open(gz_path, "rb") as src, open(tmp, "wb") as dst:
            shutil.copyfileobj(src, dst)
        tmp.replace(fasta_path)
    return fasta_path


def download_string_annotations(organism: str) -> str:
    """Path of the (gzipped) STRING ``protein.links.detailed`` file of ``organism``."""
    return str(string_file(organism, "protein.links.detailed"))


def download_string_protein_info(organism: str) -> str:
    """Path of the (gzipped) STRING ``protein.info`` file of ``organism``."""
    return str(string_file(organism, "protein.info"))


def fetch_string_organism_name(organism: Union[str, int], example_protein: Optional[str] = None) -> Optional[str]:
    """Species name STRING uses for ``organism`` (one tiny API call), or None."""
    if not example_protein:
        return None
    try:
        response = requests.get(
            f"{STRING_API_URL}/json/get_string_ids",
            params={"identifiers": example_protein, "species": str(organism), "limit": 1},
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        if payload:
            return payload[0].get("taxonName")
    except Exception as exc:
        logger.debug("Could not fetch the STRING organism name for %s: %s", organism, exc)
    return None


def fetch_uniprot_sequences_batch(
    ids: Sequence[str],
    batch_size: int = 100,
) -> Dict[str, str]:
    url = UNIPROT_STREAM_URL
    headers = {"Accept": "application/json"}
    sequences: Dict[str, str] = {}
    ids = list(ids)

    iterator = range(0, len(ids), batch_size)
    if tqdm is not None and len(ids) > batch_size:
        iterator = tqdm(iterator, desc="Fetching UniProt sequences", unit="batch")

    for start in iterator:
        batch_ids = ids[start:start + batch_size]
        query = " OR ".join(f"accession:{protein_id}" for protein_id in batch_ids)
        params = {"query": query, "format": "json"}
        response = requests.get(url, params=params, headers=headers, timeout=120)

        if response.status_code != 200:
            logger.warning(
                "Failed UniProt sequence batch %d with HTTP %s.",
                start // batch_size + 1,
                response.status_code,
            )
            continue

        payload = response.json()
        for entry in payload.get("results", []):
            accession = entry.get("primaryAccession")
            sequence = entry.get("sequence", {}).get("value", "")
            if accession and sequence:
                sequences[accession] = sequence

    missing = [protein_id for protein_id in ids if protein_id not in sequences]
    if missing:
        logger.warning(
            "UniProt sequence fetch returned %d/%d requested identifiers; %d missing "
            "(e.g. %s) due to failed batches or unresolved accessions.",
            len(sequences), len(ids), len(missing), missing[:5],
        )

    return sequences


# ---------------------------------------------------------------------------
# FASTA and parsing
# ---------------------------------------------------------------------------

def load_fasta(fasta_file: Union[Path, str]) -> Tuple[List[str], List[str]]:
    labels: List[str] = []
    sequences: List[str] = []

    with _open_text(fasta_file) as handle:
        current_label: Optional[str] = None
        current_sequence: List[str] = []

        for raw_line in handle:
            line = raw_line.strip()
            if line.startswith(">"):
                if current_label is not None:
                    labels.append(current_label)
                    sequences.append("".join(current_sequence))
                current_label = line[1:]
                current_sequence = []
            else:
                current_sequence.append(line)

        if current_label is not None:
            labels.append(current_label)
            sequences.append("".join(current_sequence))

    return labels, sequences


def fasta_record_id(label: str) -> str:
    """Record id of a FASTA header, as Biopython (and so ProteomeLM's encoder) reports it."""
    parts = label.split(maxsplit=1)
    return parts[0] if parts else ""


def parse_uniprot_ids(text: str) -> List[str]:
    return [token for token in re.split(r"[,;\s]+", text or "") if token]


def parse_explicit_pair_labels(text: str) -> List[Tuple[str, str]]:
    """Parse ``"a,b"`` pairs, one per line or separated by ``;``.

    Within a pair the two names can be separated by a comma, tab or spaces.
    Lines starting with ``#`` are ignored.
    """
    pairs: List[Tuple[str, str]] = []
    for raw_line in re.split(r"[;\n]", text or ""):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [piece for piece in re.split(r"[,\t ]+", line) if piece]
        if len(parts) != 2:
            raise ValueError(
                f"Invalid protein pair '{raw_line.strip()}': write two protein names per pair, e.g. 'rpoB,rpoC'."
            )
        pairs.append((parts[0], parts[1]))
    return pairs


# ---------------------------------------------------------------------------
# Notebook configuration
# ---------------------------------------------------------------------------

DATA_SOURCE_CHOICES = {
    "string": "STRING organism",
    "local_path": "Local FASTA",
    "taxon": "UniProt taxon",
    "uniprot_ids": "UniProt accessions",
}
QUERY_MODE_CHOICES = {
    "query_proteins": "Query proteins vs proteome",
    "explicit_pairs": "Explicit protein pairs",
    "sampled_all_pairs": "Random sample of pairs",
    "all_pairs": "All pairs",
}
_DATA_SOURCE_ALIASES = {
    **{key: key for key in DATA_SOURCE_CHOICES},
    **{label.lower(): key for key, label in DATA_SOURCE_CHOICES.items()},
    "local_fasta": "local_path", "fasta": "local_path", "local": "local_path",
    "uniprot": "uniprot_ids", "uniprot_taxon": "taxon",
}
_QUERY_MODE_ALIASES = {
    **{key: key for key in QUERY_MODE_CHOICES},
    **{label.lower(): key for key, label in QUERY_MODE_CHOICES.items()},
    "query": "query_proteins", "queries": "query_proteins", "query proteins": "query_proteins",
    "pairs": "explicit_pairs", "explicit pairs": "explicit_pairs",
    "sampled": "sampled_all_pairs", "sample": "sampled_all_pairs", "random pairs": "sampled_all_pairs",
    "all": "all_pairs",
}

DEFAULT_NOTEBOOK_CONFIG: Dict[str, Any] = {
    "data_source_mode": "string",
    "string_id": "511145",
    "local_fasta_path": "",
    "taxon_id": "",
    "uniprot_ids_text": "",
    "organism_name": "",
    "query_mode": "query_proteins",
    "query_proteins_text": "rpoB, ftsZ",
    "explicit_pairs_text": "",
    "sampled_pairs": 100_000,
    "top_k": 20,
    "checkpoint": "Bitbol-Lab/ProteomeLM-S",
    "base_model_path": "",
    "logreg_model": "human",
    "supervised_model": "none",
    "compare_with_string": True,
    "esm_device": "auto",
    "proteomelm_device": "auto",
    "pair_chunk_size": 1_000_000,
    "auto_adjust": True,
    "output_dir": "",
    "encoded_genome_file": DEFAULT_ENCODED_GENOME_FILE,
}

def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _as_int(value: Any, key: str, problems: List[str], minimum: int = 1) -> int:
    try:
        number = int(str(value).replace("_", "").replace(",", "").strip())
    except (TypeError, ValueError):
        problems.append(f"{key} must be an integer, got '{value}'.")
        return minimum
    if number < minimum:
        problems.append(f"{key} must be >= {minimum}, got {number}.")
    return number


def _canonical_choice(value: Any, aliases: Mapping[str, str], key: str, problems: List[str]) -> str:
    text = str(value or "").strip()
    canonical = aliases.get(text) or aliases.get(text.lower()) or aliases.get(text.lower().replace("-", "_"))
    if canonical is None:
        options = sorted(set(aliases.values()))
        problems.append(f"{key}='{value}' is not one of {options}.")
        return text
    return canonical


def canonical_checkpoint(value: str) -> str:
    """``"ProteomeLM-S"`` / ``"s"`` -> ``"Bitbol-Lab/ProteomeLM-S"``; anything else unchanged."""
    text = str(value or "").strip()
    match = re.fullmatch(r"(?i)(?:proteomelm-)?(xs|s|m|l)", text)
    if match:
        return f"Bitbol-Lab/ProteomeLM-{match.group(1).upper()}"
    return text or DEFAULT_NOTEBOOK_CONFIG["checkpoint"]


def normalize_notebook_config(
    config: Mapping[str, Any],
    require_local_fasta: bool = True,
) -> Dict[str, Any]:
    """Validate the notebook settings and return a canonical copy.

    Accepts the friendly labels shown in the notebook forms (e.g. ``"STRING
    organism"``, ``"Query proteins vs proteome"``), model shorthands
    (``"ProteomeLM-S"``); unknown keys are ignored. Adds parsed ``query_proteins``
    (list) and ``explicit_pairs`` (list of tuples). All problems are reported
    together in one ``ValueError``.
    """
    merged: Dict[str, Any] = dict(DEFAULT_NOTEBOOK_CONFIG)
    merged.update({key: value for key, value in config.items() if key in DEFAULT_NOTEBOOK_CONFIG})

    problems: List[str] = []
    out = dict(merged)
    out["data_source_mode"] = _canonical_choice(merged["data_source_mode"], _DATA_SOURCE_ALIASES, "data_source", problems)
    out["query_mode"] = _canonical_choice(merged["query_mode"], _QUERY_MODE_ALIASES, "scoring_mode", problems)
    for key in ("string_id", "local_fasta_path", "taxon_id", "uniprot_ids_text", "organism_name",
                "query_proteins_text", "explicit_pairs_text", "base_model_path", "output_dir"):
        out[key] = str(merged.get(key) or "").strip()
    out["checkpoint"] = canonical_checkpoint(merged.get("checkpoint", ""))
    out["logreg_model"] = str(merged.get("logreg_model") or "none").strip()
    out["supervised_model"] = str(merged.get("supervised_model") or "none").strip()
    out["esm_device"] = str(merged.get("esm_device") or "auto").strip()
    out["proteomelm_device"] = str(merged.get("proteomelm_device") or "auto").strip()
    out["compare_with_string"] = _as_bool(merged.get("compare_with_string"))
    out["auto_adjust"] = _as_bool(merged.get("auto_adjust"))
    out["top_k"] = _as_int(merged.get("top_k"), "top_k", problems)
    out["pair_chunk_size"] = _as_int(merged.get("pair_chunk_size"), "pair_chunk_size", problems)
    out["sampled_pairs"] = _as_int(merged.get("sampled_pairs"), "sampled_pairs", problems)
    out["encoded_genome_file"] = str(merged.get("encoded_genome_file") or DEFAULT_ENCODED_GENOME_FILE)

    source = out["data_source_mode"]
    if source == "string" and not re.fullmatch(r"\d+", out["string_id"]):
        problems.append(
            f"string_id must be a numeric STRING/NCBI taxon id (e.g. 511145 = E. coli K-12, 9606 = human), "
            f"got '{out['string_id']}'."
        )
    if source == "taxon" and not re.fullmatch(r"\d+", out["taxon_id"]):
        problems.append(f"taxon_id must be a numeric NCBI taxon id (e.g. 83333), got '{out['taxon_id']}'.")
    if source == "uniprot_ids" and not parse_uniprot_ids(out["uniprot_ids_text"]):
        problems.append("uniprot_ids is empty: list at least two UniProt accessions (e.g. 'P0A8V2, P0A8T7').")
    if source == "local_path" and require_local_fasta and not out["local_fasta_path"]:
        problems.append("local_fasta_path is empty: give the path of a protein FASTA file.")

    mode = out["query_mode"]
    out["query_proteins"] = [t for t in re.split(r"[,;\s]+", out["query_proteins_text"]) if t]
    try:
        out["explicit_pairs"] = parse_explicit_pair_labels(out["explicit_pairs_text"])
    except ValueError as exc:
        out["explicit_pairs"] = []
        if mode == "explicit_pairs":
            problems.append(str(exc))
    if mode == "query_proteins" and not out["query_proteins"]:
        problems.append("query_proteins is empty: list the proteins to find partners for (e.g. 'rpoB, ftsZ').")
    if mode == "explicit_pairs" and not out["explicit_pairs"] and not problems:
        problems.append("explicit_pairs is empty: list pairs such as 'rpoB,rpoC; ftsZ,ftsA'.")

    for key, kind in (("logreg_model", "logreg"), ("supervised_model", "supervised")):
        choice = out[key] = re.sub(r"\s*\(recommended\)$", "", out[key], flags=re.IGNORECASE)
        if choice.lower() in BUNDLED_MODEL_FILES[kind] or choice.lower() in {"", "none"}:
            out[key] = choice.lower() or "none"
        elif not Path(choice).expanduser().exists():
            options = ", ".join([*BUNDLED_MODEL_FILES[kind], "none"])
            problems.append(f"{key}='{choice}' is neither one of ({options}) nor an existing file.")

    if problems:
        raise ValueError("Please fix these settings:\n  - " + "\n  - ".join(problems))
    return out


def raw_github_url(repo_url: str, ref: str, relative_path: str) -> str:
    """``https://github.com/org/repo`` + ref + path -> raw.githubusercontent.com URL."""
    repo = repo_url.strip().rstrip("/")
    if repo.endswith(".git"):
        repo = repo[:-4]
    match = re.search(r"github\.com[/:]([^/]+/[^/]+)$", repo)
    if not match:
        raise ValueError(f"Not a GitHub repository URL: {repo_url}")
    return f"https://raw.githubusercontent.com/{match.group(1)}/{ref}/{relative_path.lstrip('/')}"


def resolve_model_file(
    choice: Optional[str],
    kind: str,
    repo_url: str = PUBLIC_REPO_URL,
    ref: str = DEFAULT_REPO_REF,
) -> Optional[str]:
    """Local path of a scoring model given its bundled name (or a path).

    ``kind`` is ``"logreg"`` or ``"supervised"``. Bundled names (see
    :data:`BUNDLED_MODEL_FILES`) are read from ``data/interactomes/`` of the
    repository clone when available, otherwise downloaded once from
    ``repo_url``@``ref`` into the work directory. ``"none"``/empty -> None.
    """
    text = str(choice or "").strip()
    if text.lower() in {"", "none"}:
        return None
    filename = BUNDLED_MODEL_FILES[kind].get(text.lower())
    if filename is None:
        path = Path(text).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Scoring model file not found: {path}")
        return str(path.resolve())

    relative = f"{BUNDLED_ASSET_DIR}/{filename}"
    repo_root = find_repo_root()
    if repo_root is not None and (repo_root / relative).is_file():
        return str(repo_root / relative)
    local_repo = Path(re.sub(r"^file://", "", repo_url)).expanduser()
    if "://" not in repo_url or repo_url.startswith("file://"):
        # A local checkout (or file:// URL) given as the repository: read the file from it.
        if (local_repo / relative).is_file():
            return str((local_repo / relative).resolve())
        raise FileNotFoundError(f"Bundled model '{text}' not found at {local_repo / relative}.")
    url = raw_github_url(repo_url, ref, relative)
    safe_ref = re.sub(r"[^A-Za-z0-9_.-]+", "_", ref)
    dest = _cache_subdir("assets") / safe_ref / filename
    _download_file(
        url,
        dest,
        desc=filename,
        not_found_message=(
            f"Could not download the bundled model '{text}' from {url}. Check PROTEOMELM_REPO_URL / "
            "PROTEOMELM_REF in the setup cell, or pass the path of a local copy."
        ),
    )
    return str(dest)


def scoring_model_warnings(
    checkpoint: str,
    n_pair_features: int,
    logistic_model=None,
    logreg_choice: str = "none",
    supervised_choice: str = "none",
) -> List[str]:
    """Explain when the chosen scoring models do not match the ProteomeLM backbone."""
    messages: List[str] = []
    uses_bundled = (logreg_choice in BUNDLED_MODEL_FILES["logreg"]) or (supervised_choice in BUNDLED_MODEL_FILES["supervised"])
    if uses_bundled and checkpoint.lower() != BUNDLED_MODELS_BACKBONE.lower():
        messages.append(
            f"The bundled scoring models were fit on {BUNDLED_MODELS_BACKBONE} attention; with {checkpoint} "
            "their scores are not calibrated (rankings may still be informative). Set logreg_model='none' "
            "to rank by raw attention instead."
        )
    coef = getattr(logistic_model, "coef_", None)
    if coef is not None and int(coef.shape[1]) != n_pair_features:
        messages.append(
            f"The logistic-regression model expects {int(coef.shape[1])} attention features but {checkpoint} "
            f"has {n_pair_features} (layers x heads); only the last {int(coef.shape[1])} will be used."
        )
    return messages


def load_logistic_model(path: Optional[Union[Path, str]]):
    """Load a pickled scikit-learn classifier (the bundled files hold ``(model, coef, intercept)``)."""
    if not path:
        return None
    import warnings

    with warnings.catch_warnings():
        # The bundled models were pickled with scikit-learn 1.6.0; LogisticRegression
        # unpickles fine across nearby versions, so the version-mismatch notice is noise.
        warnings.filterwarnings("ignore", message=".*Trying to unpickle estimator.*")
        with Path(path).open("rb") as handle:
            payload = pickle.load(handle)
    return payload[0] if isinstance(payload, tuple) else payload


# ---------------------------------------------------------------------------
# Loading proteins
# ---------------------------------------------------------------------------

def upload_fasta_in_colab() -> str:
    """Ask for a FASTA upload in Colab and return its saved path."""
    from google.colab import files  # type: ignore[import-not-found]

    print("Choose a protein FASTA file to upload...")
    uploaded = files.upload()
    if not uploaded:
        raise ValueError("No file was uploaded.")
    name, content = next(iter(uploaded.items()))
    dest = _cache_subdir("uploads") / Path(name).name
    dest.write_bytes(content)
    return str(dest)


def prepare_notebook_runtime_data(
    config: Dict[str, Any],
    default_encoded_genome_file: str = DEFAULT_ENCODED_GENOME_FILE,
    uniprot_fasta_file: Optional[str] = None,
) -> NotebookRuntimeData:
    labels, sequences, fasta_for_inference, annotation_organism_id, mapping_required = _load_sequences_from_runtime(
        config=config,
        uniprot_fasta_file=uniprot_fasta_file,
    )
    organism_name = _infer_organism_name(labels, str(config.get("organism_name", "") or ""))
    encoded_genome_file = _resolve_encoded_genome_cache_path(config, labels, default_encoded_genome_file, sequences)
    _invalidate_stale_encoded_cache(encoded_genome_file, labels)
    protein_annotations, display_labels = build_display_annotations(labels)

    if config.get("data_source_mode") == "string" and annotation_organism_id is not None:
        if not organism_name:
            organism_name = fetch_string_organism_name(annotation_organism_id, fasta_record_id(labels[0])) or ""
        try:
            protein_info_file = download_string_protein_info(str(annotation_organism_id))
            string_overrides = _load_string_display_overrides(protein_info_file)
            for label in labels:
                display = string_overrides.get(label)
                if not display:
                    continue
                updated = {
                    **protein_annotations[label],
                    "gene_name": display.split(" | ", 1)[0] if " | " in display else protein_annotations[label].get("gene_name"),
                    "protein_name": display.split(" | ", 1)[1] if " | " in display else protein_annotations[label].get("protein_name"),
                    "display": display,
                }
                protein_annotations[label] = updated
                display_labels[label] = display
                label_token = label.split()[0].strip()
                if label_token and label_token in protein_annotations:
                    protein_annotations[label_token] = {**protein_annotations[label_token], **updated}
                if label_token:
                    display_labels[label_token] = display
        except Exception as exc:
            logger.warning("Failed to load STRING protein info for %s: %s", annotation_organism_id, exc)

    if not organism_name:
        organism_name = _default_organism_name(config)

    return NotebookRuntimeData(
        labels=labels,
        sequences=sequences,
        fasta_for_inference=fasta_for_inference,
        annotation_organism_id=annotation_organism_id,
        mapping_required=mapping_required,
        organism_name=organism_name,
        encoded_genome_file=encoded_genome_file,
        protein_annotations=protein_annotations,
        display_labels=display_labels,
    )


def _default_organism_name(config: Mapping[str, Any]) -> str:
    mode = config.get("data_source_mode")
    if mode == "string":
        return f"STRING taxon {config.get('string_id')}"
    if mode == "taxon":
        return f"taxon {config.get('taxon_id')}"
    if mode == "local_path":
        return Path(str(config.get("local_fasta_path", "proteome"))).stem
    return "custom protein set"


def _load_sequences_from_runtime(
    config: Dict[str, Any],
    uniprot_fasta_file: Optional[str],
) -> Tuple[List[str], List[str], str, Optional[str], bool]:
    mode = config["data_source_mode"]

    if mode == "local_path":
        fasta_path = Path(str(config["local_fasta_path"])).expanduser()
        if not fasta_path.is_file():
            raise FileNotFoundError(
                f"FASTA file not found: {fasta_path} (relative paths are resolved from {Path.cwd()})."
            )
        labels, sequences = load_fasta(str(fasta_path))
        if not labels:
            raise ValueError(f"No sequences found in {fasta_path}: is it a FASTA file (headers starting with '>')?")
        return labels, sequences, str(fasta_path.resolve()), None, False

    if mode == "string":
        organism = _check_taxon_id(config["string_id"], "string_id")
        fasta_path = string_fasta_path(organism)
        labels, sequences = load_fasta(fasta_path)
        return labels, sequences, str(fasta_path), organism, False

    if mode == "taxon":
        taxon = _check_taxon_id(config["taxon_id"], "taxon_id")
        target = Path(uniprot_fasta_file) if uniprot_fasta_file else _cache_subdir("downloads") / f"uniprot_taxon_{taxon}_reviewed.fasta"
        _download_file(
            UNIPROT_STREAM_URL,
            target,
            desc=f"UniProt taxon {taxon}",
            params={"format": "fasta", "query": f"(organism_id:{taxon}) AND reviewed:true"},
        )
        labels, sequences = load_fasta(target)
        if not labels:
            target.unlink(missing_ok=True)
            raise ValueError(
                f"UniProt has no reviewed (Swiss-Prot) proteins for taxon {taxon}. Check the id, or use the "
                "STRING source or a local FASTA for organisms without a reviewed proteome."
            )
        return labels, sequences, str(target), taxon, True

    if mode == "uniprot_ids":
        accessions = parse_uniprot_ids(str(config.get("uniprot_ids_text", "")))
        if not accessions:
            raise ValueError("Provide at least one UniProt accession ID.")
        sequence_map = fetch_uniprot_sequences_batch(accessions)
        missing = [acc for acc in accessions if acc not in sequence_map]
        if not sequence_map:
            raise ValueError(f"UniProt returned no sequence for any of {accessions[:5]}: check the accessions.")
        if missing:
            print(f"Note: {len(missing)} accession(s) not found in UniProt and skipped: {', '.join(missing[:10])}")
        labels = list(sequence_map.keys())
        sequences = list(sequence_map.values())
        digest = hashlib.sha1("\n".join(labels).encode("utf-8")).hexdigest()[:10]
        target = Path(uniprot_fasta_file) if uniprot_fasta_file else _cache_subdir("downloads") / f"uniprot_ids_{digest}.fasta"
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w") as handle:
            for label, sequence in zip(labels, sequences):
                handle.write(f">{label}\n{sequence}\n")
        return labels, sequences, str(target), None, False

    raise ValueError(f"Unsupported data source mode: {mode}")


# ---------------------------------------------------------------------------
# STRING comparison
# ---------------------------------------------------------------------------

def map_string_to_uniprot(
    fasta_uniprot: str,
    fasta_string: str,
    identity_thresh: float = 95.0,
    cov_thresh: float = 90.0,
    threads: int = 4,
) -> Dict[str, str]:
    """BLAST the query FASTA against the STRING proteome: ``{query id: STRING id}``."""
    makeblastdb_path = shutil.which("makeblastdb")
    blastp_path = shutil.which("blastp")
    if makeblastdb_path is None or blastp_path is None:
        raise RuntimeError(
            "BLAST+ tools 'makeblastdb' and 'blastp' are required for UniProt-to-STRING mapping "
            "(conda install -c bioconda blast, or apt-get install ncbi-blast+ on Colab)."
        )
    blast_dir = _cache_subdir("blast")
    db_name = str(blast_dir / f"string_{Path(fasta_string).stem}")
    blast_output = str(blast_dir / f"{Path(fasta_uniprot).stem}_vs_string.tsv")

    subprocess.run([
        makeblastdb_path, "-in", fasta_string, "-dbtype", "prot", "-out", db_name
    ], check=True, stdout=subprocess.DEVNULL)

    subprocess.run([
        blastp_path,
        "-query", fasta_uniprot,
        "-db", db_name,
        "-outfmt", "6 qseqid sseqid pident qcovs evalue bitscore",
        "-qcov_hsp_perc", str(cov_thresh),
        "-num_threads", str(threads),
        "-out", blast_output,
    ], check=True)

    best_hits: Dict[str, Dict[str, float]] = {}
    with open(blast_output) as handle:
        reader = csv.DictReader(
            handle,
            fieldnames=["qseqid", "sseqid", "pident", "qcovs", "evalue", "bitscore"],
            delimiter="\t",
        )
        for row in reader:
            pident = float(row["pident"])
            qcov = float(row["qcovs"])
            bitscore = float(row["bitscore"])
            if pident < identity_thresh or qcov < cov_thresh:
                continue
            query_id = row["qseqid"]
            if query_id not in best_hits or bitscore > best_hits[query_id]["bitscore"]:
                best_hits[query_id] = {
                    "uniprot_id": row["sseqid"],
                    "bitscore": bitscore,
                }

    return {query_id: hit["uniprot_id"] for query_id, hit in best_hits.items()}


def string_ids_to_result_ids(
    query_to_string: Mapping[str, str],
    result_ids: Sequence[str],
) -> Dict[str, str]:
    """Invert a ``{query id: STRING id}`` BLAST mapping onto the ids used in the results.

    BLAST may report a UniProt query as ``sp|P0A8V2|RPOB_ECOLI`` or just
    ``P0A8V2``; both are matched to the result id via the accession.
    """
    by_accession: Dict[str, str] = {}
    for result_id in result_ids:
        by_accession.setdefault(_extract_accession(result_id), result_id)
        by_accession.setdefault(result_id, result_id)
    inverted: Dict[str, str] = {}
    for query_id, string_id in query_to_string.items():
        result_id = by_accession.get(query_id) or by_accession.get(_extract_accession(query_id))
        if result_id is not None:
            inverted.setdefault(string_id, result_id)
    return inverted


def attach_sparse_string_scores(
    results_df: pd.DataFrame,
    input_fasta_path: str,
    organism_id: Optional[Union[str, int]],
    mapping_required: bool,
) -> pd.DataFrame:
    """Add a ``string_score`` column (STRING combined score 0-1000; 0 = no STRING link)."""
    if results_df.empty or organism_id is None:
        return results_df

    if mapping_required and (shutil.which("makeblastdb") is None or shutil.which("blastp") is None):
        logger.warning(
            "Skipping the STRING comparison: mapping UniProt ids to STRING needs BLAST+ "
            "('makeblastdb'/'blastp' not found; on Colab: !apt-get -qq install ncbi-blast+)."
        )
        return results_df

    annotation_file = download_string_annotations(str(organism_id))

    string_to_result: Optional[Dict[str, str]] = None
    if mapping_required:
        string_fa_path = str(string_fasta_path(str(organism_id)))
        try:
            query_to_string = map_string_to_uniprot(input_fasta_path, string_fa_path)
        except (RuntimeError, subprocess.CalledProcessError) as exc:
            logger.warning("Skipping the STRING comparison: %s", exc)
            return results_df
        result_ids = pd.unique(pd.concat([results_df["protein_a"], results_df["protein_b"]], ignore_index=True))
        string_to_result = string_ids_to_result_ids(query_to_string, list(result_ids))

    wanted_pairs = {tuple(sorted((a, b))) for a, b in zip(results_df["protein_a"], results_df["protein_b"])}
    score_lookup: Dict[Tuple[str, str], int] = {}
    with _open_text(annotation_file) as handle:
        next(handle)
        iterator = tqdm(handle, desc="Scanning STRING links", unit=" links", unit_scale=True, leave=False) if tqdm is not None else handle
        for line in iterator:
            parts = line.split()
            if len(parts) < 2:
                continue
            protein1, protein2 = parts[0], parts[1]
            if string_to_result is not None:
                protein1 = string_to_result.get(protein1)
                protein2 = string_to_result.get(protein2)
                if protein1 is None or protein2 is None:
                    continue
            key = (protein1, protein2) if protein1 <= protein2 else (protein2, protein1)
            if key in wanted_pairs:
                score_lookup[key] = int(parts[-1])

    results_df = results_df.copy()
    results_df["string_score"] = [
        score_lookup.get(tuple(sorted((a, b))), 0)
        for a, b in zip(results_df["protein_a"], results_df["protein_b"])
    ]
    return results_df


# ---------------------------------------------------------------------------
# Display labels
# ---------------------------------------------------------------------------

def _extract_accession(label: str) -> str:
    if "|" in label:
        parts = label.split("|")
        if len(parts) > 1 and parts[1].strip():
            return parts[1].strip()
    return label.split()[0].strip() if label.strip() else label


def _canonical_uniprot_accession(value: str) -> str:
    token = value.strip()
    if "-" in token:
        token = token.split("-", 1)[0]
    return token


def _looks_like_uniprot_accession(value: str) -> bool:
    token = _canonical_uniprot_accession(value)
    if not token or "." in token or ":" in token:
        return False
    return bool(
        re.fullmatch(r"[OPQ][0-9][A-Z0-9]{3}[0-9]", token)
        or re.fullmatch(r"[A-NR-Z][0-9][A-Z][A-Z0-9]{2}[0-9]", token)
        or re.fullmatch(r"[A-NR-Z][0-9](?:[A-Z0-9]{3}[0-9]){2}", token)
    )


def _parse_header_annotation(label: str) -> Dict[str, Optional[str]]:
    accession = _extract_accession(label)
    annotation: Dict[str, Optional[str]] = {
        "accession": accession,
        "canonical_accession": _canonical_uniprot_accession(accession),
        "display": label,
        "protein_name": None,
        "gene_name": None,
    }
    header = label.strip()
    if "|" in header:
        parts = header.split("|", 2)
        if len(parts) == 3:
            tail = parts[2].strip()
            if " OS=" in tail:
                protein_name = tail.split(" OS=", 1)[0].strip()
                if " " in protein_name:
                    protein_name = protein_name.split(" ", 1)[1].strip()
                annotation["protein_name"] = protein_name or None
            if " GN=" in tail:
                annotation["gene_name"] = tail.split(" GN=")[1].split()[0].strip()
    else:
        parts = header.split(None, 1)
        if len(parts) > 1:
            annotation["protein_name"] = parts[1].strip() or None

    if annotation["gene_name"] and annotation["protein_name"]:
        annotation["display"] = f"{annotation['gene_name']} | {annotation['protein_name']}"
    elif annotation["protein_name"]:
        annotation["display"] = annotation["protein_name"]
    elif annotation["gene_name"]:
        annotation["display"] = annotation["gene_name"]
    return annotation


def _fetch_uniprot_annotations_batch(accessions: Sequence[str], batch_size: int = 100) -> Dict[str, Dict[str, Optional[str]]]:
    annotations: Dict[str, Dict[str, Optional[str]]] = {}
    deduplicated = [accession for accession in dict.fromkeys(accessions) if accession]
    for start in range(0, len(deduplicated), batch_size):
        batch = deduplicated[start:start + batch_size]
        query = " OR ".join(f"accession:{accession}" for accession in batch)
        params = {
            "query": query,
            "format": "json",
            "fields": "accession,protein_name,gene_names",
            "size": str(batch_size),
        }
        try:
            response = requests.get("https://rest.uniprot.org/uniprotkb/search", params=params, timeout=120)
        except requests.RequestException as exc:
            logger.warning("Skipping UniProt annotation batch (%s).", exc)
            continue
        if response.status_code != 200:
            logger.warning(
                "Skipping UniProt annotation batch after HTTP %s for %d identifiers.",
                response.status_code,
                len(batch),
            )
            continue
        payload = response.json()
        for entry in payload.get("results", []):
            accession = entry.get("primaryAccession")
            if not accession:
                continue
            protein_name = None
            protein_desc = entry.get("proteinDescription", {})
            rec_name = protein_desc.get("recommendedName", {})
            full_name = rec_name.get("fullName", {})
            if isinstance(full_name, dict):
                protein_name = full_name.get("value")
            elif isinstance(full_name, str):
                protein_name = full_name
            if protein_name is None:
                submission_names = protein_desc.get("submissionNames", [])
                if submission_names:
                    protein_name = submission_names[0].get("fullName", {}).get("value")
            genes = entry.get("genes", [])
            gene_name = None
            if genes:
                gene_name = genes[0].get("geneName", {}).get("value")
            display = protein_name or gene_name or accession
            if gene_name and protein_name:
                display = f"{gene_name} | {protein_name}"
            annotations[accession] = {
                "accession": accession,
                "protein_name": protein_name,
                "gene_name": gene_name,
                "display": display,
            }

    missing = [accession for accession in deduplicated if accession not in annotations]
    if missing:
        logger.warning(
            "UniProt annotation fetch returned %d/%d requested accessions; %d missing "
            "(e.g. %s) due to failed batches or unresolved accessions.",
            len(annotations), len(deduplicated), len(missing), missing[:5],
        )
    return annotations


def build_display_annotations(labels: Sequence[str]) -> Tuple[Dict[str, Dict[str, Optional[str]]], Dict[str, str]]:
    annotations = {label: _parse_header_annotation(label) for label in labels}
    unresolved = [
        meta["canonical_accession"]
        for meta in annotations.values()
        if meta["display"] == meta["accession"] and _looks_like_uniprot_accession(str(meta["canonical_accession"]))
    ]
    if unresolved:
        fetched = _fetch_uniprot_annotations_batch(unresolved)
        for label, meta in annotations.items():
            fetched_meta = fetched.get(str(meta["canonical_accession"]))
            if fetched_meta is not None:
                annotations[label] = {**meta, **fetched_meta}
    display_labels = {label: str(annotations[label]["display"]) for label in labels}
    for label in labels:
        label_token = label.split()[0].strip() if label.strip() else label
        if label_token and label_token not in annotations:
            annotations[label_token] = dict(annotations[label])
        if label_token and label_token not in display_labels:
            display_labels[label_token] = display_labels[label]
    return annotations, display_labels


def _load_string_display_overrides(info_file: Union[Path, str]) -> Dict[str, str]:
    overrides: Dict[str, str] = {}
    with _open_text(info_file) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            protein_id = (
                row.get("#string_protein_id")
                or row.get("string_protein_id")
                or row.get("protein_external_id")
            )
            if not protein_id:
                continue
            preferred_name = (row.get("preferred_name") or "").strip()
            annotation = (row.get("annotation") or "").strip()
            if annotation.lower() == "n/a":
                annotation = ""
            if annotation:
                annotation = annotation.split(";", 1)[0].strip()

            if preferred_name and annotation:
                display = f"{preferred_name} | {annotation}"
            else:
                display = preferred_name or annotation or protein_id
            overrides[protein_id] = display
    return overrides


def _infer_organism_name(labels: Sequence[str], fallback: str) -> str:
    for label in labels[:10]:
        if " OS=" not in label:
            continue
        organism_name = label.split(" OS=", 1)[1]
        for marker in (" OX=", " GN=", " PE=", " SV="):
            if marker in organism_name:
                organism_name = organism_name.split(marker, 1)[0]
        organism_name = organism_name.strip()
        if organism_name:
            return organism_name
    return fallback


# ---------------------------------------------------------------------------
# Encoded-proteome cache
# ---------------------------------------------------------------------------

def _resolve_encoded_genome_cache_path(
    config: Dict[str, Any],
    labels: Sequence[str],
    default_encoded_genome_file: str,
    sequences: Optional[Sequence[str]] = None,
) -> str:
    configured_path = Path(str(config.get("encoded_genome_file") or default_encoded_genome_file))
    if configured_path.name != default_encoded_genome_file:
        # User explicitly overrode the path — respect it verbatim, don't redirect.
        return str(configured_path)
    signature_parts = [
        str(config.get("checkpoint", "")),
        str(config.get("data_source_mode", "")),
        str(config.get("local_fasta_path", "")),
        str(config.get("string_id", "")),
        str(config.get("taxon_id", "")),
        str(config.get("uniprot_ids_text", "")),
        *labels,
        # Sequence content, not just labels/paths — a label/path can stay the same
        # while the underlying sequence changes (edited local FASTA, re-downloaded
        # STRING/UniProt data), which would otherwise silently reuse a stale cache.
        *(sequences or []),
    ]
    digest = hashlib.sha1("\n".join(signature_parts).encode("utf-8")).hexdigest()[:16]
    # Rooted in the work directory regardless of configured_path's own directory:
    # this is the auto-derived (default, non-overridden) branch, so the location is ours to pick.
    return str(_cache_subdir("embeddings") / f"{configured_path.stem}_{digest}.pt")


def _invalidate_stale_encoded_cache(cache_path: Union[Path, str], expected_labels: Sequence[str]) -> bool:
    cache_path = Path(cache_path)
    if not cache_path.exists():
        return False
    try:
        payload = torch.load(cache_path, map_location="cpu")
    except Exception as exc:
        logger.warning("Removing unreadable encoded proteome cache %s: %s", cache_path, exc)
        cache_path.unlink(missing_ok=True)
        return True

    cached_labels = payload.get("group_labels") if isinstance(payload, dict) else None
    # The encoder stores FASTA record ids (first header token), the notebook keeps full headers.
    expected_ids = [fasta_record_id(label) for label in expected_labels]
    if cached_labels is None or [fasta_record_id(str(x)) for x in cached_labels] != expected_ids:
        logger.info("Removing stale encoded proteome cache %s.", cache_path)
        cache_path.unlink(missing_ok=True)
        return True
    return False


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

def _slug(text: str, max_length: int = 60) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")
    return slug[:max_length].rstrip("_") or "proteome"


_MODE_SUFFIX = {
    "query_proteins": "query",
    "explicit_pairs": "pairs",
    "sampled_all_pairs": "sampled",
    "all_pairs": "allpairs",
}


def results_basename(config: Mapping[str, Any], organism_name: str = "") -> str:
    """File stem for the result files, e.g. ``ppi_escherichia_coli_k12_string511145_query``."""
    source = config.get("data_source_mode")
    organism_name = organism_name.split(" (", 1)[0]  # drop strain details, e.g. "(strain ATCC 33530 / ...)"
    parts = ["ppi"]
    if source == "string":
        parts += [_slug(organism_name)] if organism_name and not organism_name.startswith("STRING taxon") else []
        parts.append(f"string{config.get('string_id')}")
    elif source == "taxon":
        parts += [_slug(organism_name)] if organism_name and not organism_name.startswith("taxon ") else []
        parts.append(f"taxon{config.get('taxon_id')}")
    elif source == "local_path":
        parts.append(_slug(Path(str(config.get("local_fasta_path", "proteome"))).stem))
    else:
        parts.append("uniprot_selection")
    parts.append(_MODE_SUFFIX.get(str(config.get("query_mode")), "results"))
    return "_".join(part for part in parts if part)


def save_notebook_results(
    results_df: pd.DataFrame,
    config: Mapping[str, Any],
    organism_name: str = "",
    top_df: Optional[pd.DataFrame] = None,
    labels: Optional[Sequence[str]] = None,
    output_dir: Optional[Union[Path, str]] = None,
    max_rows_with_names: int = 1_000_000,
) -> Dict[str, Path]:
    """Write ``<stem>.csv`` (all scored pairs), ``<stem>_top.csv`` and ``<stem>.pkl``.

    Tables longer than ``max_rows_with_names`` keep only protein ids in
    ``<stem>.csv`` (the readable names would roughly triple its size) and the
    names go to ``<stem>_proteins.csv`` instead. Returns the written paths keyed
    by ``"all_pairs"``, ``"top"``, ``"proteins"`` and ``"pickle"``.
    ``output_dir`` defaults to ``<work dir>/results``.
    """
    out_dir = Path(output_dir).expanduser() if output_dir else _cache_subdir("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = results_basename(config, organism_name)
    paths: Dict[str, Path] = {}
    paths["all_pairs"] = out_dir / f"{stem}.csv"
    drop = ["idx_a", "idx_b"]
    if len(results_df) > max_rows_with_names and "protein_a_label" in results_df.columns:
        drop += ["protein_a_label", "protein_b_label"]
        names = pd.concat([
            results_df[["protein_a", "protein_a_label"]].set_axis(["protein", "name"], axis=1),
            results_df[["protein_b", "protein_b_label"]].set_axis(["protein", "name"], axis=1),
        ]).drop_duplicates("protein")
        paths["proteins"] = out_dir / f"{stem}_proteins.csv"
        names.to_csv(paths["proteins"], index=False)
    results_df.drop(columns=drop, errors="ignore").to_csv(paths["all_pairs"], index=False, float_format="%.6g")
    if top_df is not None and not top_df.empty:
        paths["top"] = out_dir / f"{stem}_top.csv"
        top_df.to_csv(paths["top"], index=False)
    paths["pickle"] = out_dir / f"{stem}.pkl"
    with open(paths["pickle"], "wb") as handle:
        pickle.dump({"config": dict(config), "labels": list(labels or []), "results": results_df}, handle)
    return paths


def offer_result_downloads(
    paths: Mapping[str, Path],
    figure_paths: Optional[Sequence[Union[Path, str]]] = None,
    max_browser_mb: float = 100.0,
) -> None:
    """On Colab, download the CSV results (and the figures, as one zip) to the browser.

    Elsewhere, print where the files are.
    """
    csv_paths = [Path(p) for key, p in paths.items() if str(p).endswith(".csv")]
    results_dir = Path(next(iter(paths.values()))).parent
    if figure_paths:
        from .notebook_plots import zip_figures

        stem = Path(paths["all_pairs"]).stem if "all_pairs" in paths else "ppi"
        figures_zip = zip_figures(figure_paths, results_dir / f"{stem}_figures.zip")
        if figures_zip is not None:
            csv_paths.append(figures_zip)
    if not in_colab():
        print("Results and figures are saved in:", results_dir)
        return
    from google.colab import files  # type: ignore[import-not-found]

    for path in csv_paths:
        size_mb = path.stat().st_size / 1e6
        if size_mb <= max_browser_mb:
            files.download(str(path))
        else:
            print(
                f"{path.name} is {size_mb:.0f} MB, too large for a browser download. Copy it to Google Drive instead:\n"
                "  from google.colab import drive; drive.mount('/content/drive')\n"
                f"  import shutil; shutil.copy('{path}', '/content/drive/MyDrive/')"
            )


# ---------------------------------------------------------------------------
# Optional ipywidgets form
# ---------------------------------------------------------------------------

def build_config_form(config: MutableMapping[str, Any]):
    """Interactive form that edits ``config`` in place; None when ipywidgets is missing.

    Every change is written straight into ``config`` (canonical values), so the
    notebook reads the same dict whether or not the form is used. Only the
    fields relevant to the chosen protein source and scoring mode are shown.
    """
    try:
        import ipywidgets as widgets
    except ImportError:
        return None

    problems: List[str] = []
    config["data_source_mode"] = _canonical_choice(config.get("data_source_mode", "string"), _DATA_SOURCE_ALIASES, "data_source", problems)
    config["query_mode"] = _canonical_choice(config.get("query_mode", "query_proteins"), _QUERY_MODE_ALIASES, "scoring_mode", problems)
    if problems:
        config["data_source_mode"] = config["data_source_mode"] if config["data_source_mode"] in DATA_SOURCE_CHOICES else "string"
        config["query_mode"] = config["query_mode"] if config["query_mode"] in QUERY_MODE_CHOICES else "query_proteins"

    style = {"description_width": "170px"}

    def full():  # one Layout per widget: fields are shown/hidden through their own layout
        return widgets.Layout(width="100%")
    boxes: Dict[str, Any] = {}

    def _bind(key: str, widget, cast=None):
        def _on_change(change):
            value = change["new"]
            config[key] = cast(value) if cast else value
        widget.observe(_on_change, names="value")
        boxes[key] = widget
        return widget

    def _text(key, description, placeholder="", area=False):
        cls = widgets.Textarea if area else widgets.Text
        layout = widgets.Layout(width="100%", min_height="70px") if area else full()
        value = config.get(key)
        return _bind(key, cls(value="" if value is None else str(value), description=description,
                              placeholder=placeholder, style=style, layout=layout), lambda v: v.strip())

    def _combo(key, description, options):
        return _bind(key, widgets.Combobox(value=str(config.get(key) or ""), options=list(options),
                                           description=description, ensure_option=False, style=style, layout=full()),
                     lambda v: v.strip())

    source = _bind("data_source_mode", widgets.Dropdown(
        options=[(label, key) for key, label in DATA_SOURCE_CHOICES.items()],
        value=config["data_source_mode"], description="Protein source:", style=style, layout=full()))
    source_fields = {
        "string": _text("string_id", "STRING taxon id:", "511145 (E. coli K-12), 9606 (human)"),
        "local_path": _text("local_fasta_path", "FASTA path:", "/path/to/proteome.fasta"),
        "taxon": _text("taxon_id", "NCBI taxon id:", "83333"),
        "uniprot_ids": _text("uniprot_ids_text", "UniProt accessions:", "P0A8V2, P0A8T7, ...", area=True),
    }
    mode = _bind("query_mode", widgets.Dropdown(
        options=[(label, key) for key, label in QUERY_MODE_CHOICES.items()],
        value=config["query_mode"], description="Scoring mode:", style=style, layout=full()))
    mode_fields = {
        "query_proteins": _text("query_proteins_text", "Query proteins:", "rpoB, ftsZ (gene names or ids)", area=True),
        "explicit_pairs": _text("explicit_pairs_text", "Protein pairs:", "rpoB,rpoC; ftsZ,ftsA", area=True),
        "sampled_all_pairs": _bind("sampled_pairs", widgets.IntText(
            value=int(config.get("sampled_pairs") or 100_000), description="Pairs to sample:", style=style, layout=full())),
        "all_pairs": None,
    }
    top_k = _bind("top_k", widgets.IntSlider(value=int(config.get("top_k") or 20), min=1, max=100,
                                             description="Top k:", style=style, layout=full()))
    checkpoint = _combo("checkpoint", "ProteomeLM model:", [f"Bitbol-Lab/ProteomeLM-{s}" for s in ("S", "XS", "M", "L")])
    logreg = _combo("logreg_model", "Attention scorer:", [*BUNDLED_MODEL_FILES["logreg"], "none"])
    supervised_options = [("none", "none")] + [
        (f"{name} (recommended)" if name == RECOMMENDED_MODELS["supervised"] else name, name)
        for name in BUNDLED_MODEL_FILES["supervised"]]
    current = re.sub(r"\s*\(recommended\)$", "", str(config.get("supervised_model") or "none"), flags=re.IGNORECASE)
    if current not in {value for _, value in supervised_options}:  # a checkpoint path set in CONFIG
        supervised_options.append((current, current))
    supervised = _bind("supervised_model", widgets.Dropdown(
        options=supervised_options, value=current, description="Supervised model:", style=style, layout=full()))
    compare = _bind("compare_with_string", widgets.Checkbox(
        value=_as_bool(config.get("compare_with_string", True)), description="Compare with STRING scores", indent=False))
    advanced = widgets.VBox([
        _text("organism_name", "Organism name:", "optional, inferred when empty"),
        _text("esm_device", "ESM-C device:", "auto, cuda, cuda:1, cpu"),
        _text("proteomelm_device", "ProteomeLM device:", "auto, cuda, cpu"),
        _bind("pair_chunk_size", widgets.IntText(value=int(config.get("pair_chunk_size") or 1_000_000),
                                                 description="Pairs per pass:", style=style, layout=full())),
        _bind("auto_adjust", widgets.Checkbox(value=_as_bool(config.get("auto_adjust", True)),
                                              description="Fall back to CPU when the GPU is too small", indent=False)),
        _text("base_model_path", "LoRA base model:", "only for LoRA adapter checkpoints"),
        _text("output_dir", "Output folder:", "empty = <work dir>/results"),
    ])
    accordion = widgets.Accordion(children=[advanced])
    accordion.set_title(0, "Advanced")
    accordion.selected_index = None

    def _refresh(*_):
        for key, widget in source_fields.items():
            widget.layout.display = "" if source.value == key else "none"
        for key, widget in mode_fields.items():
            if widget is not None:
                widget.layout.display = "" if mode.value == key else "none"

    source.observe(_refresh, names="value")
    mode.observe(_refresh, names="value")
    _refresh()

    def _card(title, children):
        return widgets.VBox([widgets.HTML(f"<b>{title}</b>"), *children],
                            layout=widgets.Layout(width="100%", padding="6px 10px", border="1px solid #d0d0d0", margin="0 0 8px 0"))

    return widgets.VBox([
        widgets.HTML("<div style='color:#555'>This form edits <code>CONFIG</code> directly; re-running the "
                     "cell resets it to the values written in the code.</div>"),
        _card("Proteins", [source, *source_fields.values()]),
        _card("What to score", [mode, *[w for w in mode_fields.values() if w is not None], top_k]),
        _card("Models", [checkpoint, logreg, supervised, compare]),
        accordion,
    ], layout=full())
