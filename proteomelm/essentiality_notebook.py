"""Helpers for ``notebooks/essentiality_prediction.ipynb`` (proteome loading, labels, figures, files).

Inference itself is in :mod:`proteomelm.essentiality`. Downloads, embeddings and
results go to the work directory of the PPI notebook helpers
(:func:`proteomelm.ppi.notebook_runtime.get_cache_dir`), so STRING / UniProt
downloads are shared between the two notebooks.
"""
from __future__ import annotations

import gzip
import hashlib
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import requests
import torch

from proteomelm.ppi.notebook_inference import _protein_name_keys, parse_protein_list
from proteomelm.ppi.notebook_plots import AXIS, GRID, INK, INK_SECONDARY, MUTED, SERIES, STRING_MEDIUM, SURFACE, \
    _rc, _style

UNIPROT_PROTEOMES_URL = "https://rest.uniprot.org/proteomes"
UNIPROT_FTP_REFERENCE = ("https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/"
                         "reference_proteomes/{kingdom}/{upid}/{upid}_{taxid}.fasta.gz")
UNIPROT_STREAM_URL = "https://rest.uniprot.org/uniprotkb/stream"

SOURCES = {
    "UniProt reference proteome (taxon id)": "uniprot_taxon",
    "UniProt proteome id (UP...)": "uniprot_proteome",
    "STRING organism (taxon id)": "string",
    "FASTA file": "fasta",
}
RULES = {"Top fraction": "top_fraction", "Top N": "top_n", "Probability threshold": "threshold"}

# Figure colors: essential = series blue, non-essential = gray, quasi-essential = light blue.
ESSENTIAL_COLOR = SERIES[0]
NONESSENTIAL_COLOR = AXIS
QE_COLOR = STRING_MEDIUM
OTHER_COLOR = GRID


@dataclass
class LoadedProteome:
    ids: List[str]                  # FASTA record ids (first word of each header)
    sequences: List[str]
    gene: Dict[str, str]            # id -> gene name ("" if unknown)
    description: Dict[str, str]     # id -> protein name / annotation ("" if unknown)
    organism: str
    fasta: str

    @property
    def n_proteins(self) -> int:
        return len(self.ids)


# ---------------------------------------------------------------------------
# Proteome input
# ---------------------------------------------------------------------------

def _get_json(url: str, params: Optional[Mapping[str, str]] = None, timeout: int = 60) -> dict:
    try:
        response = requests.get(url, params=params, timeout=timeout)
    except requests.RequestException as exc:
        raise ConnectionError(f"Request to {url} failed ({exc}). Check your internet connection.") from exc
    if response.status_code == 404:
        raise FileNotFoundError(f"Not found: {response.url}")
    response.raise_for_status()
    return response.json()


def find_uniprot_proteome(taxon_id: Union[str, int]) -> dict:
    """UniProt proteome entry of a taxon: its reference proteome, else its first proteome listed."""
    taxon = str(taxon_id).strip()
    if not re.fullmatch(r"\d+", taxon):
        raise ValueError(f"taxon_id must be a numeric NCBI taxon id (e.g. 83333 for E. coli K-12), got '{taxon_id}'.")
    results = _get_json(f"{UNIPROT_PROTEOMES_URL}/search",
                        {"query": f"organism_id:{taxon}", "format": "json", "size": "50"}).get("results", [])
    results = [r for r in results if "excluded" not in str(r.get("proteomeType", "")).lower()]
    if not results:
        raise ValueError(f"UniProt has no proteome for taxon {taxon}. Check the id at https://www.uniprot.org/proteomes "
                         "or use another source (STRING organism, FASTA file).")
    reference = [r for r in results if "reference" in str(r.get("proteomeType", "")).lower()
                 and "non reference" not in str(r.get("proteomeType", "")).lower()]
    return (reference or results)[0]


def download_uniprot_proteome(upid: str, folder: Union[str, Path], entry: Optional[dict] = None) -> Tuple[Path, dict]:
    """FASTA of UniProt proteome ``upid`` (canonical sequences), downloaded once into ``folder``.

    Reference proteomes come from the UniProt FTP site (fast), others from the REST API.
    """
    upid = str(upid).strip().upper()
    if not re.fullmatch(r"UP\d{9,}", upid):
        raise ValueError(f"'{upid}' is not a UniProt proteome id (UP followed by digits, e.g. UP000000625).")
    entry = entry or _get_json(f"{UNIPROT_PROTEOMES_URL}/{upid}", {"format": "json"})
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    fasta = folder / f"uniprot_{upid}.fasta"
    if fasta.exists() and fasta.stat().st_size > 0:
        return fasta, entry
    gz = fasta.with_suffix(".fasta.gz")
    taxid = entry.get("taxonomy", {}).get("taxonId")
    kingdom = str(entry.get("superkingdom", "")).capitalize()
    urls = []
    if "reference" in str(entry.get("proteomeType", "")).lower() and taxid and kingdom:
        urls.append((UNIPROT_FTP_REFERENCE.format(kingdom=kingdom, upid=upid, taxid=taxid), None))
    urls.append((UNIPROT_STREAM_URL, {"format": "fasta", "compressed": "true", "query": f"(proteome:{upid})"}))
    errors = []
    for url, params in urls:
        try:
            with requests.get(url, params=params, stream=True, timeout=300) as response:
                response.raise_for_status()
                with open(gz, "wb") as handle:
                    shutil.copyfileobj(response.raw, handle)
            with gzip.open(gz, "rb") as src, open(fasta, "wb") as dst:
                shutil.copyfileobj(src, dst)
            gz.unlink(missing_ok=True)
            return fasta, entry
        except Exception as exc:  # try the next source
            errors.append(f"{url}: {exc}")
            gz.unlink(missing_ok=True)
            fasta.unlink(missing_ok=True)
    raise ConnectionError("Could not download proteome " + upid + ":\n  " + "\n  ".join(errors))


def parse_header(header: str) -> Tuple[str, str]:
    """(gene name, description) from a UniProt-style FASTA header (``... desc OS=... GN=gene ...``)."""
    rest = header.split(maxsplit=1)[1] if len(header.split(maxsplit=1)) > 1 else ""
    gene = re.search(r"\bGN=(\S+)", rest) or re.search(r"\[gene=([^\]]+)\]", rest)
    description = rest.split(" OS=", 1)[0].strip()
    return (gene.group(1) if gene else ""), description


def read_proteome_fasta(path: Union[str, Path], organism: str = "") -> LoadedProteome:
    """Read a FASTA file (Biopython ids and sequences, as :func:`proteomelm.essentiality.read_fasta`)."""
    from Bio import SeqIO

    ids, sequences, gene, description = [], [], {}, {}
    for record in SeqIO.parse(str(path), "fasta"):
        ids.append(record.id)
        sequences.append(str(record.seq))
        gene[record.id], description[record.id] = parse_header(record.description)
    if not ids:
        raise ValueError(f"No sequences found in {path}: is it a protein FASTA file (headers starting with '>')?")
    if not organism:
        with open(path) as handle:
            match = re.search(r" OS=(.+?)(?: OX=| GN=| PE=| SV=|$)", handle.readline())
        organism = match.group(1).strip() if match else Path(path).stem
    return LoadedProteome(ids, sequences, gene, description, organism, str(path))


def _string_names(taxon: str, ids: Sequence[str]) -> Tuple[Dict[str, str], Dict[str, str]]:
    from proteomelm.ppi.notebook_runtime import _open_text, download_string_protein_info

    gene, description = {}, {}
    with _open_text(download_string_protein_info(taxon)) as handle:
        header = handle.readline().rstrip("\n").lstrip("#").split("\t")
        for line in handle:
            row = dict(zip(header, line.rstrip("\n").split("\t")))
            protein = row.get("string_protein_id")
            if protein:
                gene[protein] = row.get("preferred_name", "")
                description[protein] = row.get("annotation", "").split(";")[0].strip()
    return {i: gene.get(i, "") for i in ids}, {i: description.get(i, "") for i in ids}


def load_proteome(source: str, value: str = "", work_dir: Optional[Union[str, Path]] = None) -> LoadedProteome:
    """Load a whole proteome from one of :data:`SOURCES` (menu label or its short key).

    ``value``: taxon id, UniProt proteome id, STRING taxon id or FASTA path.
    """
    from proteomelm.ppi.notebook_runtime import fetch_string_organism_name, get_cache_dir, string_fasta_path

    key = SOURCES.get(source, source)
    downloads = Path(work_dir or get_cache_dir()) / "downloads"
    value = str(value or "").strip()
    if key == "fasta":
        path = Path(value).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"FASTA file not found: '{value}' (relative paths start from {Path.cwd()}).")
        return read_proteome_fasta(path)
    if key == "string":
        fasta = string_fasta_path(value)
        proteome = read_proteome_fasta(fasta, organism=f"STRING organism {value}")
        name = fetch_string_organism_name(value, proteome.ids[0])
        if name:
            proteome.organism = f"{name} (STRING {value})"
        try:
            proteome.gene, proteome.description = _string_names(value, proteome.ids)
        except Exception as exc:  # names are optional
            print(f"Note: could not load STRING protein names ({exc}).")
        return proteome
    if key in ("uniprot_taxon", "uniprot_proteome"):
        entry = find_uniprot_proteome(value) if key == "uniprot_taxon" else None
        upid = entry["id"] if entry else value
        fasta, entry = download_uniprot_proteome(upid, downloads, entry)
        organism = entry.get("taxonomy", {}).get("scientificName", "")
        proteome = read_proteome_fasta(fasta, organism=f"{organism} ({upid})" if organism else upid)
        if key == "uniprot_taxon" and "reference" not in str(entry.get("proteomeType", "")).lower():
            print(f"Note: taxon {value} has no reference proteome; using {upid} ({entry.get('proteomeType')}).")
        return proteome
    raise ValueError(f"Unknown proteome source '{source}'; choose one of {list(SOURCES)}.")


# ---------------------------------------------------------------------------
# Embeddings cache, device plan
# ---------------------------------------------------------------------------

def proteome_digest(ids: Sequence[str], sequences: Sequence[str]) -> str:
    return hashlib.sha1("\n".join(f"{i}\t{s}" for i, s in zip(ids, sequences)).encode()).hexdigest()[:16]


def cached_esmc(predictor, proteome: LoadedProteome, work_dir: Optional[Union[str, Path]] = None) -> torch.Tensor:
    """ESM-C embeddings of the proteome in input order, cached in ``<work dir>/embeddings``."""
    from proteomelm.essentiality import length_sorted_order
    from proteomelm.ppi.notebook_runtime import get_cache_dir

    path = Path(work_dir or get_cache_dir()) / "embeddings" / \
        f"esmc_{predictor.config.esm_model}_{proteome_digest(proteome.ids, proteome.sequences)}.pt"
    if path.exists():
        return torch.load(path, map_location="cpu")
    order = length_sorted_order(proteome.sequences)
    sorted_embeddings = predictor.embed([proteome.sequences[i] for i in order], [proteome.ids[i] for i in order])
    embeddings = torch.empty_like(sorted_embeddings)
    embeddings[torch.as_tensor(order)] = sorted_embeddings
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings, path)
    return embeddings


def plan_devices(n_proteins: int, backbone: str = "Bitbol-Lab/ProteomeLM-L", device: str = "auto",
                 esm_device: str = "auto") -> Tuple[str, str, List[str]]:
    """(ProteomeLM device, ESM-C device, notes): GPU when available and large enough, else CPU."""
    from proteomelm.essentiality import estimate_memory_gb
    from proteomelm.modeling_proteomelm import ProteomeLMConfig
    from proteomelm.ppi.notebook_inference import available_memory_gb, resolve_device

    config = ProteomeLMConfig.from_pretrained(backbone)
    kwargs = dict(dim=config.dim, n_layers=config.n_layers, n_heads=config.n_heads)
    notes = []
    plm_device, esm_dev = resolve_device(device), resolve_device(esm_device)
    need = estimate_memory_gb(n_proteins, on_gpu=True, **kwargs)
    free = available_memory_gb(plm_device) if plm_device.startswith("cuda") else None
    notes.append(f"{n_proteins:,} proteins: ESM-C needs ~{need['esmc_gb']:.0f} GB, ProteomeLM ~{need['proteomelm_gb']:.1f} GB "
                 f"on GPU" + (f" ({free:.1f} GB free)." if free is not None else "."))
    if free is not None and device == "auto" and need["proteomelm_gb"] + need["esmc_gb"] > 0.95 * free:
        cpu = estimate_memory_gb(n_proteins, on_gpu=False, **kwargs)["proteomelm_gb"]
        notes.append(f"Not enough free GPU memory: running ProteomeLM on CPU (~{cpu:.1f} GB of RAM, slower).")
        plm_device = "cpu"
    if esm_dev == "cpu" and n_proteins > 500:
        notes.append(f"ESM-C will embed {n_proteins:,} proteins on CPU: this can take hours. A GPU is strongly recommended.")
    return plm_device, esm_dev, notes


# ---------------------------------------------------------------------------
# Results table and known labels
# ---------------------------------------------------------------------------

def calling_rule(rule: str, top_fraction: float = 0.1, top_n: int = 300, threshold: float = 0.5) -> Dict[str, float]:
    """Keyword argument of :meth:`EssentialityPredictor.predict` for a :data:`RULES` menu entry."""
    key = RULES.get(rule, rule)
    value = {"top_fraction": top_fraction, "top_n": top_n, "threshold": threshold}.get(key)
    if value is None:
        raise ValueError(f"Unknown calling rule '{rule}'; choose one of {list(RULES)}.")
    return {key: value}


def annotate(table: pd.DataFrame, proteome: LoadedProteome) -> pd.DataFrame:
    """Add ``gene`` and ``description`` columns after ``protein_id``."""
    out = table.copy()
    out.insert(1, "gene", [proteome.gene.get(i, "") for i in out["protein_id"]])
    out.insert(2, "description", [proteome.description.get(i, "") for i in out["protein_id"]])
    return out


def match_proteins(table: pd.DataFrame, names: Union[str, Sequence[str]]) -> Tuple[List[int], List[str]]:
    """Row positions of ``table`` matching ``names`` (protein ids, UniProt accessions, STRING loci or
    gene names, case-insensitive), and the names that matched nothing. A name matching several
    proteins selects all of them."""
    names = parse_protein_list(names) if isinstance(names, str) else list(names)
    index: Dict[str, List[int]] = {}
    genes = table["gene"] if "gene" in table else [""] * len(table)
    for pos, (protein, gene) in enumerate(zip(table["protein_id"], genes)):
        for key in _protein_name_keys(str(protein), str(gene) if gene else None):
            index.setdefault(key, []).append(pos)
    found, missing = [], []
    for name in names:
        hits = index.get(name.strip().lower())
        if hits:
            found.extend(h for h in hits if h not in found)
        else:
            missing.append(name)
    return found, missing


def read_name_list(text: str) -> List[str]:
    """Protein names from a text box: a list (commas, spaces, newlines) or the path of a text file."""
    text = (text or "").strip()
    if text and "\n" not in text and Path(text).expanduser().is_file():
        text = Path(text).expanduser().read_text()
    return parse_protein_list(text)


def known_labels(table: pd.DataFrame, essential: str, nonessential: str = "") -> Tuple[Optional[np.ndarray], List[str]]:
    """Per-row labels (1 essential, 0 non-essential, -1 unlabelled) from the pasted names, and unmatched names.

    Without a non-essential list, every protein not listed as essential counts as non-essential.
    None when no essential gene was given or matched.
    """
    ess_names = read_name_list(essential)
    if not ess_names:
        return None, []
    ess_rows, missing = match_proteins(table, ess_names)
    if not ess_rows:
        return None, missing
    ne_names = read_name_list(nonessential)
    labels = np.full(len(table), 0 if not ne_names else -1)
    if ne_names:
        ne_rows, ne_missing = match_proteins(table, ne_names)
        labels[ne_rows] = 0
        missing += ne_missing
    labels[ess_rows] = 1
    return labels, missing


def label_metrics(table: pd.DataFrame, labels: np.ndarray) -> dict:
    """AUROC / AUPR of ``p_essential`` on the labelled rows, and the essential genes in the top N (N = # essential)."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    keep = labels >= 0
    y, s = labels[keep], table["p_essential"].to_numpy()[keep]
    n_ess = int((labels == 1).sum())
    top = table["rank"].to_numpy() <= n_ess
    result = {"n_essential": n_ess, "n_nonessential": int((labels == 0).sum()),
              "essential_in_top_n": int((top & (labels == 1)).sum())}
    if len(np.unique(y)) == 2:
        result.update(auroc=float(roc_auc_score(y, s)), aupr=float(average_precision_score(y, s)),
                      aupr_baseline=float(y.mean()))
    return result


def save_table(table: pd.DataFrame, stem: str, output_dir: Optional[Union[str, Path]] = None) -> Path:
    from proteomelm.essentiality import write_table
    from proteomelm.ppi.notebook_runtime import get_cache_dir

    folder = Path(output_dir).expanduser() if output_dir else get_cache_dir() / "results"
    return write_table(table, folder / f"{stem}.tsv")


def results_stem(organism: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", organism.lower()).strip("_")[:60] or "proteome"
    return f"essentiality_{slug}"


def offer_downloads(paths: Sequence[Union[str, Path]], figures: Sequence[Union[str, Path]] = ()) -> None:
    """On Colab, download the results and a zip of the figures; elsewhere print where they are."""
    from proteomelm.ppi.notebook_plots import zip_figures
    from proteomelm.ppi.notebook_runtime import in_colab

    paths = [Path(p) for p in paths]
    if figures:
        archive = zip_figures(figures, paths[0].with_name(paths[0].stem + "_figures.zip"))
        if archive is not None:
            paths.append(archive)
    if not in_colab():
        print("Results and figures are saved in:", paths[0].parent)
        return
    from google.colab import files  # type: ignore[import-not-found]

    for path in paths:
        files.download(str(path))


def build_form(config: MutableMapping[str, object], choices: Mapping[str, Sequence[str]]):
    """Minimal ipywidgets form editing ``config`` in place (None without ipywidgets)."""
    try:
        import ipywidgets as widgets
    except ImportError:
        return None
    rows = []
    for key, value in config.items():
        if key in choices:
            widget = widgets.Dropdown(options=list(choices[key]), value=value)
        elif isinstance(value, bool):
            widget = widgets.Checkbox(value=value)
        elif isinstance(value, int):
            widget = widgets.IntText(value=value)
        elif isinstance(value, float):
            widget = widgets.FloatText(value=value)
        else:
            widget = widgets.Text(value=str(value))
        widget.description = key
        widget.style = {"description_width": "140px"}
        widget.layout = widgets.Layout(width="520px")
        widget.observe(lambda change, k=key: config.__setitem__(k, change["new"]), names="value")
        rows.append(widget)
    return widgets.VBox(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _plt():
    import matplotlib.pyplot as plt

    return plt


def plot_score_distribution(table: pd.DataFrame, title: str = ""):
    """Histogram of p_essential (log count axis), called-essential proteins in blue."""
    plt = _plt()
    p = table["p_essential"].to_numpy()
    called = table["predicted_essential"].to_numpy() if "predicted_essential" in table else np.zeros(len(p), bool)
    bins = np.linspace(0, 1, 41)
    with plt.rc_context(_rc()):
        fig, ax = plt.subplots(figsize=(6.4, 3.4))
        ax.hist(p[~called], bins=bins, color=NONESSENTIAL_COLOR, edgecolor=SURFACE, linewidth=1,
                label=f"not called ({(~called).sum():,})")
        if called.any():
            ax.hist(p[called], bins=bins, color=ESSENTIAL_COLOR, edgecolor=SURFACE, linewidth=1,
                    label=f"predicted essential ({called.sum():,})")
            cutoff = p[called].min()
            ax.axvline(cutoff, color=INK_SECONDARY, linewidth=1, linestyle="--")
            ax.annotate(f"cutoff {cutoff:.3g}", (cutoff, 1), xycoords=("data", "axes fraction"), xytext=(-4, -4),
                        textcoords="offset points", ha="right", va="top", color=INK_SECONDARY, fontsize=8)
        ax.set_yscale("log")
        ax.set_xlim(0, 1)
        ax.set_xlabel("p_essential")
        ax.set_ylabel("proteins")
        ax.set_title(title or f"Essentiality scores of {len(p):,} proteins", loc="left")
        ax.legend(loc="upper center")
        _style(ax, grid_axis="y")
        fig.tight_layout()
    return fig


def _donut(ax, outer: Optional[Sequence[str]], inner: Sequence[str], title: str) -> None:
    colors = {"E": ESSENTIAL_COLOR, "NE": NONESSENTIAL_COLOR, "QE": QE_COLOR, "Other": OTHER_COLOR}
    order = {"E": 0, "QE": 1, "NE": 2, "Other": 3}
    idx = sorted(range(len(inner)), key=lambda i: (order[outer[i]] if outer is not None else 0, order[inner[i]]))
    rings = [(outer, 1.0), (inner, 0.7)] if outer is not None else [(inner, 1.0)]
    for values, radius in rings:
        # one wedge per run of consecutive proteins with the same class (proteins in the same order on both rings)
        runs: List[List] = []
        for i in idx:
            if runs and runs[-1][0] == values[i]:
                runs[-1][1] += 1
            else:
                runs.append([values[i], 1])
        ax.pie([n for _, n in runs], colors=[colors[c] for c, _ in runs], radius=radius, startangle=90,
               counterclock=False, wedgeprops=dict(width=0.25, edgecolor=SURFACE, linewidth=1))
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10, color=INK)


def plot_call_donut(table: pd.DataFrame, labels: Optional[np.ndarray] = None, title: str = ""):
    """Donut of the calls; with ``labels`` (1/0/-1), the outer ring shows the labels (as in Fig. 5B)."""
    plt = _plt()
    if "predicted_essential" not in table:
        print("Figure skipped: no calling rule (set top N, top fraction or a threshold).")
        return None
    calls = np.where(table["predicted_essential"].to_numpy(), "E", "NE")
    outer = None if labels is None else np.array(["E" if v == 1 else "NE" if v == 0 else "Other" for v in labels])
    with plt.rc_context(_rc()):
        fig, ax = plt.subplots(figsize=(5.2, 4.2))
        _donut(ax, outer, calls, title or ("Labelled (outer) vs predicted (inner)" if outer is not None
                                           else "Predicted essential proteins"))
        n_called = int((calls == "E").sum())
        text = f"{n_called:,} / {len(calls):,}\npredicted essential"
        if outer is not None:
            n_e = int((outer == "E").sum())
            hit = int(((outer == "E") & (calls == "E")).sum())
            ne = int((outer == "NE").sum())
            fp = int(((outer == "NE") & (calls == "E")).sum())
            text += f"\n\nlabelled E called E: {hit}/{n_e} ({hit / max(n_e, 1):.0%})" \
                    f"\nlabelled NE called E: {fp}/{ne} ({fp / max(ne, 1):.0%})"
        ax.text(0, 0, text, ha="center", va="center", fontsize=8, color=INK_SECONDARY)
        from matplotlib.patches import Patch
        handles = [Patch(color=ESSENTIAL_COLOR, label="essential"), Patch(color=NONESSENTIAL_COLOR, label="non-essential")]
        if outer is not None and (outer == "Other").any():
            handles.append(Patch(color=OTHER_COLOR, label="unlabelled"))
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.0, 1.0))
        fig.tight_layout()
    return fig


def plot_label_curves(table: pd.DataFrame, labels: Optional[np.ndarray]):
    """ROC and precision-recall curves of p_essential against known labels."""
    plt = _plt()
    if labels is None:
        print("Figure skipped: no known essential genes given.")
        return None
    from sklearn.metrics import precision_recall_curve, roc_curve

    keep = labels >= 0
    y, s = labels[keep], table["p_essential"].to_numpy()[keep]
    if len(np.unique(y)) < 2:
        print("Figure skipped: the labels need both essential and non-essential proteins.")
        return None
    metrics = label_metrics(table, labels)
    fpr, tpr, _ = roc_curve(y, s)
    precision, recall, _ = precision_recall_curve(y, s)
    with plt.rc_context(_rc()):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.8))
        ax1.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle=":")
        ax1.plot(fpr, tpr, color=SERIES[0], linewidth=2)
        ax1.set(xlabel="false positive rate", ylabel="true positive rate", xlim=(0, 1), ylim=(0, 1.01))
        ax1.set_title(f"ROC (AUROC {metrics['auroc']:.3f})", loc="left")
        ax2.axhline(metrics["aupr_baseline"], color=MUTED, linewidth=1, linestyle=":")
        ax2.plot(recall, precision, color=SERIES[0], linewidth=2)
        ax2.set(xlabel="recall", ylabel="precision", xlim=(0, 1), ylim=(0, 1.01))
        ax2.set_title(f"Precision-recall (AUPR {metrics['aupr']:.3f})", loc="left")
        ax2.annotate(f"random: {metrics['aupr_baseline']:.3f}", (0, metrics["aupr_baseline"]), xytext=(4, 4),
                     textcoords="offset points", ha="left", fontsize=8, color=MUTED)
        for ax in (ax1, ax2):
            _style(ax)
        fig.tight_layout()
    return fig
