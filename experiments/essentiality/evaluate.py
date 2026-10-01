"""Evaluation of the essentiality classifiers.

* ``metrics``: for one (model, classifier depth, seed, weights), evaluate every
  per-layer classifier on the test fold and write four pickles to the plots folder
  (``get_filename``): ``roc-auc-scores.pkl``, ``precision-recall-curves.pkl``,
  ``otherscores.pkl`` (incl. the bootstrapped AUROC used by Fig. 5A) and
  ``metrics-distribution.pkl`` (per-genome metrics). Each is a dict keyed by the
  classifier checkpoint folder.
* Whole-genome predictions for Fig. 5B (yeast, E. coli, JCVI-Syn1.0/Syn3A): a genome
  is embedded, scored, and the N highest-scoring proteins are called essential,
  with N = the number of proteins labelled essential in that genome (``top_n_essential``).
"""
import argparse
import json
import os
import pickle
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests
import torch
from Bio import SeqIO
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, matthews_corrcoef,
                             precision_recall_curve, precision_recall_fscore_support, roc_auc_score, roc_curve)
from torch.utils.data import DataLoader

from experiments.essentiality.common import CHECKPOINTS, MODEL_IDS, load_config, plm_size_from_checkpoint, \
    resolve, stored_embeds_folder

warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')

# --------------------------
# Output file naming
# --------------------------


@dataclass
class PlotArguments:
    checkpoint: str = None          # XS, S, M, L or ESMC
    modeltype: str = None           # simpleclassifier, 2layer, 3layer
    random_seed: object = None      # int, or a list of seeds for the plotting functions
    splits_thr: int = None
    holdout: bool = None
    plots_data_directory: str = None
    which_weights: str = "trained"
    exp_prefix: str = None
    train_percentage: int = None


def get_filename(args: PlotArguments, file: str) -> str:
    """Published naming scheme of the metric pickles, e.g.
    ``{plots}/ProteomeLM-L/2layer-trained-seed42-cluster40-otherscores.pkl``."""
    filename = [args.plots_data_directory, "/",
                f"ProteomeLM-{args.checkpoint}/" if args.checkpoint != "ESMC" else "",
                f"{args.exp_prefix}-" if args.exp_prefix is not None else "",
                "ESMC-" if args.checkpoint == "ESMC" else "",
                "holdout-" if args.holdout else "",
                "" if args.modeltype == "simpleclassifier" else f"{args.modeltype}-",
                f"{args.which_weights}-",
                f"seed{args.random_seed}-" if args.random_seed is not None else "",
                f"cluster{args.splits_thr}-" if args.splits_thr is not None else "",
                f"train{args.train_percentage}-" if args.train_percentage is not None else "",
                file]
    return "".join(filename)


# --------------------------
# Classifier checkpoints
# --------------------------

def read_classifier_config(checkpoint_folder: str, classifier_dir: Optional[str] = None) -> dict:
    """``config.json`` of a classifier checkpoint. Metric pickles are keyed by the
    absolute folder of the machine that wrote them; if that folder is missing here,
    the same basename is looked up in ``classifier_dir``."""
    path = checkpoint_folder
    if not os.path.isdir(path) and classifier_dir is not None:
        path = os.path.join(classifier_dir, os.path.basename(os.path.normpath(checkpoint_folder)))
    with open(os.path.join(path, "config.json")) as f:
        return json.load(f)


def checkpoint_matches(cfg: dict, checkpoint: str, weights: str, seed: Optional[int], splits_file: Optional[str],
                       wandb_project_name: Optional[str] = None) -> bool:
    """Does a saved classifier config belong to this (model, weights, seed, split)?

    Works for published configs (``ProteomeLM-L/checkpoint-210``, absolute split path)
    and new ones (Hugging Face id / local path): the split is compared by basename.
    """
    if checkpoint == "ESMC":
        if not cfg.get("use_esmc_as_input", False):
            return False
    elif cfg.get("use_esmc_as_input", False) or plm_size_from_checkpoint(cfg.get("proteomeLM_checkpoint")) != checkpoint:
        return False
    if (cfg.get("which_weights") or "trained") != weights:
        return False
    if seed is not None and cfg.get("random_seed") != seed:
        return False
    if splits_file is not None and os.path.basename(str(cfg.get("splits_info_file"))) != os.path.basename(splits_file):
        return False
    if wandb_project_name is not None and cfg.get("wandb_project_name") != wandb_project_name:
        return False
    return True


def get_relevant_checkpoints(checkpoint_directory: str, type_of_model: str, checkpoint: str, weights: str,
                             seed: Optional[int], splits_file: Optional[str],
                             wandb_project_name: Optional[str] = None) -> List[str]:
    """Classifier folders of one run, one per layer. When a layer has several
    checkpoints (re-runs), the most recently written one is kept, with a warning."""
    assert os.path.exists(checkpoint_directory), checkpoint_directory
    by_layer: Dict[int, List[str]] = {}
    for element in sorted(os.listdir(checkpoint_directory)):
        if type_of_model not in element:
            continue
        folder = os.path.join(checkpoint_directory, element)
        if not os.path.isdir(folder):
            continue
        try:
            cfg = read_classifier_config(folder)
        except Exception:
            print(f"Could not read config.json file for checkpoint {element}")
            continue
        if cfg.get("model_id") != type_of_model:
            continue
        if checkpoint_matches(cfg, checkpoint, weights, seed, splits_file, wandb_project_name):
            by_layer.setdefault(cfg["which_hidden_layer"], []).append(folder)
    selected = []
    for layer in sorted(by_layer):
        folders = by_layer[layer]
        if len(folders) > 1:
            folders = sorted(folders, key=lambda f: os.path.getmtime(os.path.join(f, "pytorch_model.bin")))
            print(f"Warning: {len(folders)} checkpoints for layer {layer}; using the newest {folders[-1]}")
        selected.append(folders[-1])
    return selected


def load_classifier(checkpoint_folder: str, device: str, dtype: torch.dtype = torch.float32):
    from experiments.essentiality.train import ClassifierConfig, classifier_class
    config = ClassifierConfig.from_pretrained(checkpoint_folder, max_length=None)
    model = classifier_class(config.model_id)(config)
    state_dict = torch.load(os.path.join(checkpoint_folder, "pytorch_model.bin"), map_location=device)
    model.load_state_dict(state_dict)
    return config, model.to(device=device, dtype=dtype).eval()


def localize_config(config, cfg, checkpoint: str, splits_file: str, device: str, checkpoint_dir=None):
    """Point a (possibly published) classifier config at this machine's data and weights."""
    from experiments.essentiality.train import plm_path_for_classifier
    p = cfg["paths"]
    config.fasta_folder = p["fasta_folder"]
    config.label_folder = p["label_folder"]
    config.esmc_embeds_folder = p["esmc_embeds_folder"]
    config.stored_embeds_folder = stored_embeds_folder(p["embeds_folder"], checkpoint)
    config.splits_info_file = splits_file
    config.proteomeLM_checkpoint = plm_path_for_classifier(config, checkpoint_dir, p["baseline_weights_folder"])
    config.proteomeLM_checkpoint_folder = ""
    config.classifier_device = device
    config.esm_device = device
    return config


# --------------------------
# Test-fold metrics
# --------------------------

def bootstrapped_auroc(y_true, y_pred, n_bootstraps=1000):
    """(mean, standard error) of the AUROC over proteins resampled with replacement (seed 42)."""
    rng = np.random.RandomState(42)
    bootstrapped_scores = []
    for _ in range(n_bootstraps):
        indices = rng.randint(0, len(y_pred), len(y_pred))
        if len(np.unique(y_true[indices])) < 2:
            continue  # AUROC undefined without both classes
        bootstrapped_scores.append(roc_auc_score(y_true[indices], y_pred[indices]))
    return np.average(bootstrapped_scores), np.std(bootstrapped_scores) / np.sqrt(len(bootstrapped_scores))


def prepare_test_dataset(config, which_hidden_layer, which_esmc_hidden_layer, use_holdout_dataset=False):
    from experiments.essentiality.train import ProteomeLMDatasetForEssentiality, embed_for_config
    embed_for_config(config, taxids_to_include=config.which_taxids_to_use if use_holdout_dataset else None)
    test_dataset = ProteomeLMDatasetForEssentiality(config=config, split="test",
                                                    which_hidden_layer=which_hidden_layer,
                                                    which_esmc_hidden_layer=which_esmc_hidden_layer)
    print(f"Test dataset with {len(test_dataset)} genomes")
    return test_dataset


def _genome_metrics(y_true, y_pred, decision_threshold):
    # Warnings are errors here (as in the published runs): a genome whose metrics
    # would be ill-defined (e.g. a single class) gets None instead.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        precision, recall, f1score, support = precision_recall_fscore_support(y_true, y_pred[:, 1] > decision_threshold)
        balanced_accuracy = balanced_accuracy_score(y_true, y_pred[:, 1] > decision_threshold)
        matthews = matthews_corrcoef(y_true, y_pred[:, 1] > decision_threshold)
        roc = roc_curve(y_true, y_pred[:, 1])
        pr = precision_recall_curve(y_true, y_pred[:, 1])
    return {"precision": precision, "recall": recall, "f1score": f1score, "balanced_accuracy": balanced_accuracy,
            "support": support, "matthews_corrcoef": matthews, "roc_auc_curve": roc, "precision_recall_curve": pr}


def compute_metrics(config, model, testing_dtype=torch.float32, return_distribution=False,
                    use_holdout_dataset=False, decision_threshold=0.5):
    """Test-fold metrics of one per-layer classifier, pooled over genomes and (optionally) per genome.

    Scores are p(NE) (label 1); AUROC is symmetric in the class choice.
    """
    from experiments.essentiality.train import ProteomeLMDatasetForEssentiality
    if config.use_esmc_as_input:
        which_hidden_layer, which_esmc_hidden_layer = None, config.which_hidden_layer
    else:
        which_hidden_layer, which_esmc_hidden_layer = config.which_hidden_layer, None
    test_dataset = prepare_test_dataset(config, which_hidden_layer, which_esmc_hidden_layer, use_holdout_dataset)
    # One genome per batch, no shuffling: batch index == test_dataset.taxid_list index
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                                 collate_fn=ProteomeLMDatasetForEssentiality.get_collator(config))
    model.to(device=config.classifier_device, dtype=testing_dtype).eval()

    y_pred, y_true = [], []
    metrics_distribution, taxids_with_exceptions = {}, {}
    with torch.inference_mode():
        for idx, data in enumerate(test_dataloader):
            inputs = data["plm_representations"] if which_esmc_hidden_layer is None else data["esmc_representations"]
            outputs = model(inputs.to(device=config.classifier_device, dtype=testing_dtype))
            one_genome_y_pred = torch.softmax(outputs.reshape((-1, 2)), dim=-1, dtype=torch.float).cpu().numpy()
            one_genome_y_true = data["ess_labels"].to(torch.int).cpu().flatten().numpy()
            y_pred.extend(one_genome_y_pred)
            y_true.extend(one_genome_y_true)
            if return_distribution:
                keep = one_genome_y_true != config.labels_padding_index
                try:
                    m = _genome_metrics(one_genome_y_true[keep], one_genome_y_pred[keep], decision_threshold)
                except Exception as e:
                    taxids_with_exceptions[test_dataset.taxid_list[idx]] = e
                    m = {"precision": None, "recall": None, "f1score": None, "balanced_accuracy": None,
                         "support": None, "matthews_corrcoef": None, "roc_auc_curve": (None, None, None),
                         "precision_recall_curve": (None, None, None)}
                metrics_distribution[test_dataset.taxid_list[idx]] = m
    if taxids_with_exceptions:
        print(f"Per-genome metrics undefined for: {taxids_with_exceptions}")

    y_pred, y_true = np.array(y_pred), np.array(y_true)
    y_pred = y_pred[y_true != config.labels_padding_index]
    y_true = y_true[y_true != config.labels_padding_index]

    fpr, tpr, _ = roc_curve(y_true, y_pred[:, 1])
    prec, rec, thr = precision_recall_curve(y_true, y_pred[:, 1])
    auc = average_precision_score(y_true, y_pred[:, 1])
    auc_with_std = bootstrapped_auroc(y_true, y_pred[:, 1])
    try:
        precision, recall, f1score, support = precision_recall_fscore_support(y_true, y_pred[:, 1] > decision_threshold)
    except Exception as e:
        print(f"Exception for overall metrics --> {e}")
        precision, recall, f1score, support = (None, None, None, None)
    try:
        balanced_accuracy = balanced_accuracy_score(y_true, y_pred[:, 1] > decision_threshold)
    except Exception as e:
        print(f"Exception for balanced_accuracy --> {e}")
        balanced_accuracy = None
    try:
        matthews = matthews_corrcoef(y_true, y_pred[:, 1] > decision_threshold)
    except Exception as e:
        print(f"Exception for matthews correlation coefficient --> {e}")
        matthews = None

    overall_metrics = {
        "roc_auc_curve": (auc, (fpr, tpr)),  # note: `auc` here is the average precision
        "precision_recall_curve": (prec, rec, thr),
        "precision": precision,
        "recall": recall,
        "f1score": f1score,
        "balanced_accuracy": balanced_accuracy,
        "support": support,
        "matthews_corrcoef": matthews,
        "auroc_with_std": auc_with_std,
    }
    return overall_metrics, metrics_distribution


def evaluate_run(cfg, checkpoint: str, modeltype: str, seed: int, weights: str, splits_file: str,
                 classifier_dir: Optional[str] = None, plots_dir: Optional[str] = None, device: str = "cuda:0",
                 holdout: bool = False, exp_prefix: Optional[str] = None, checkpoint_dir=None) -> Dict[str, str]:
    """Evaluate every per-layer classifier of one run and write the four metric pickles."""
    classifier_dir = classifier_dir or cfg["paths"]["classifier_checkpoints_folder"]
    plots_dir = plots_dir or cfg["paths"]["plots_folder"]
    list_of_checkpoints = get_relevant_checkpoints(classifier_dir, modeltype, checkpoint, weights, seed, splits_file,
                                                   cfg["classifier"].get("wandb_project_name"))
    print(f"matching checkpoints = {list_of_checkpoints}")
    if not list_of_checkpoints:
        raise FileNotFoundError(f"No {modeltype} classifier for {checkpoint}/{weights}/seed {seed} "
                                f"({os.path.basename(splits_file)}) in {classifier_dir}")

    roc_auc_curves, precision_recall_curves, othermetrics, metrics_distributions = {}, {}, {}, {}
    for ckpt in list_of_checkpoints:
        print(f"==== {ckpt}")
        config, model = load_classifier(ckpt, device)
        config.training_set_percentage = None
        localize_config(config, cfg, checkpoint, splits_file, device, checkpoint_dir)
        if holdout:
            config.which_taxids_to_exclude = None
            config.which_taxids_to_use = cfg["data_download"]["holdout_testset"]
        metrics, metrics_dist = compute_metrics(config, model, return_distribution=True, use_holdout_dataset=holdout)
        roc_auc_curves[ckpt] = metrics.pop("roc_auc_curve")
        precision_recall_curves[ckpt] = metrics.pop("precision_recall_curve")
        othermetrics[ckpt] = metrics
        metrics_distributions[ckpt] = metrics_dist

    args = PlotArguments(plots_data_directory=plots_dir, checkpoint=checkpoint, exp_prefix=exp_prefix,
                         holdout=holdout, modeltype=modeltype, which_weights=weights, random_seed=seed,
                         splits_thr=cfg["split"]["threshold"])
    outputs = {}
    for name, obj in [("roc-auc-scores.pkl", roc_auc_curves), ("precision-recall-curves.pkl", precision_recall_curves),
                      ("otherscores.pkl", othermetrics), ("metrics-distribution.pkl", metrics_distributions)]:
        path = get_filename(args, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        print(f"Saved {path}")
        outputs[name] = path
    return outputs


# --------------------------
# Whole-genome predictions (Fig. 5B)
# --------------------------

def top_n_essential(p_nonessential: Sequence[float], n: int) -> np.ndarray:
    """Hard calls for a whole genome: the ``n`` proteins with the lowest p(NE) are
    essential (0), all others non-essential (1). Ties follow ``np.argsort`` order."""
    p = np.asarray(p_nonessential)
    calls = np.ones_like(p)
    calls[np.argsort(p)[:n]] = 0
    return calls


def compute_model_predictions(fasta_file, classifier_checkpoint, classification_threshold=0.5, device="cuda:0",
                              return_score=False, top_n: Optional[int] = None, embeds_path=None,
                              checkpoint_dir=None, baseline_dir=None):
    """Embed a genome and score it with one per-layer classifier (in float32).

    Returns ``{gene: "E"/"NE"}`` (and ``{gene: p(NE)}`` if ``return_score``). With
    ``top_n`` the calls follow ``top_n_essential`` instead of the threshold.
    """
    from experiments.essentiality.train import ClassifierConfig, classifier_class, plm_path_for_classifier, \
        prepare_ess_data
    config = ClassifierConfig.from_pretrained(classifier_checkpoint)
    config.dtype = "float32"
    proteomelm_path = plm_path_for_classifier(config, checkpoint_dir, baseline_dir)

    if embeds_path is not None and os.path.exists(embeds_path):
        print(f"Loading stored embeds at {embeds_path}")
        with open(embeds_path, "rb") as f:
            repr_data = pickle.load(f)
    else:
        repr_data, _ = prepare_ess_data(fasta_file=fasta_file, esm_device=device, proteomelm_device=device,
                                        proteomelm_checkpoint=proteomelm_path,
                                        only_compute_esmc_embeds=config.use_esmc_as_input)
        if embeds_path is not None:
            with open(embeds_path, "wb") as f:
                pickle.dump(repr_data, f)

    torch.set_default_dtype(getattr(torch, config.dtype))
    model = classifier_class(config.model_id)(config).to(device=device)
    model.load_state_dict(torch.load(os.path.join(classifier_checkpoint, "pytorch_model.bin"), map_location=device))
    model.eval()
    with torch.inference_mode():
        if config.use_esmc_as_input:
            inputs = repr_data["hidden_states"][config.which_hidden_layer]
        else:
            inputs = repr_data["plm_all_representations"][config.which_hidden_layer]
        outputs = model(inputs.to(device=device, dtype=getattr(torch, config.dtype)))
        y_pred = torch.softmax(outputs.reshape((-1, 2)).to(torch.float), dim=-1, dtype=torch.float).cpu().numpy()

    y_pred_soft = y_pred[:, 1]
    y_pred_hard = y_pred_soft > classification_threshold if top_n is None else top_n_essential(y_pred_soft, top_n)
    pred_translate = ["E", "NE"]
    predictions = {gene: pred_translate[int(y_pred_hard[i])] for i, gene in enumerate(repr_data["group_labels"])}
    prediction_scores = {gene: y_pred_soft[i] for i, gene in enumerate(repr_data["group_labels"])}
    return (predictions, prediction_scores) if return_score else predictions


def group_label(label, new_labels=("E", "NE", "QE", "Other")):
    """Coarse label used by the donuts: E, NE, QE (quasi/conditionally essential) or Other."""
    return label if label in new_labels else "Other"


def sort_items_by_label(data_true: dict, data_pred: dict) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """Genes ordered by coarse true label (E, NE, Other, QE); predictions in the same order."""
    sorted_true = sorted(data_true.items(), key=lambda x: group_label(x[1]))
    sorted_pred = [(key, data_pred[key]) for key, _ in sorted_true]
    return sorted_true, sorted_pred


def get_sorted_fraction(true_sorted, pred_sorted) -> Dict[str, float]:
    """Fraction of E-, NE- and QE-labelled genes that are predicted E (the donut tables)."""
    stats_dict = {"E in E": 0, "E in NE": 0, "E in QE": 0}
    total = {"E": 0, "NE": 0, "QE": 0}
    for (key_true, label), (key_pred, pred) in zip(true_sorted, pred_sorted):
        assert key_true == key_pred
        coarse = group_label(label)
        if coarse in total:
            total[coarse] += 1
            if pred == "E":
                stats_dict[f"E in {coarse}"] += 1
    for key in total.keys():
        stats_dict[f"E in {key}"] = stats_dict[f"E in {key}"] / total[key] if total[key] else float("nan")
    return stats_dict


def auroc_on_labelled(labels: dict, scores: dict) -> float:
    """AUROC of p(NE) on the genes labelled E or NE (E=0, NE=1)."""
    pairs = np.array([[0 if labels[g] == "E" else 1, scores[g]] for g in labels if labels[g] in ("E", "NE")])
    return roc_auc_score(pairs[:, 0], pairs[:, 1])


class GetMinimalCellLabels:
    """Essentiality labels of the minimal cells: JCVI-Syn3A (SynWiki) and JCVI-Syn1.0
    (Hutchison et al. 2016, Science, Database S1, mirrored on Google Drive). When
    ``minimalcell_taxid{t}_labels.tsv`` exists (written by ``data fetch``), its labels
    are used instead of the source tables."""

    def __init__(self):
        self.minimalcells_info = {
            2144189: {
                "Name": "JCVI-Syn3A",
                "Url": 'https://synwiki.uni-goettingen.de/v1/gene?query=essential&__accept=CSV',
                "EssDataFormat": ".csv",
                "PossibleLabels": {'Possibly essential': "E?", 'yes': "E", 'Not a real gene': "Not_a_Gene",
                                   'Essential': "E", 'Not a Mycoplasma gene': "Not_a_Gene", 'Quasi essential': "QE",
                                   'essential': "E", 'Non essential': "NE", 'Possibly non essential': "NE?",
                                   'No label': "No_label"},
                "LocusColumn": " locus",
                "EssentialColumn": " essential",
                "DownloadFunction": self.download_using_requests,
                "CreateDataFrame": pd.read_csv},
            766747: {
                "Name": "JCVI-Syn1.0",
                # Original: https://www.science.org/doi/suppl/10.1126/science.aad6253/suppl_file/aad6253-hutchison-sm-database-s1.xlsx
                "Url": "https://drive.google.com/file/d/1ji2jrFX5V4P12SIIBZZ2QehZ-7_fgpwN/view",
                "EssDataFormat": ".xlsx",
                "PossibleLabels": {'e': "E", 'n': "NE", 'i': "QE", 'ie': "QE", 'in': "QE", 'No label': "No_label",
                                   "n?": "NE?", "e?": "E?", "x": "???"},
                "LocusColumn": "Locus tag (accession CP002027). ",
                "EssentialColumn": "Essential? Date=130821",
                "DownloadFunction": self.download_using_gdown,
                "CreateDataFrame": pd.read_excel},
        }

    def download_using_requests(self, url, output_path):
        response = requests.get(url)
        with open(output_path, 'wb') as f:
            f.write(response.content)

    def download_using_gdown(self, url, output_path):
        from experiments.essentiality.data import _import_gdown
        _import_gdown().download(url, output_path, fuzzy=True)

    def get_ess_df(self, taxid, folder_path):
        info = self.minimalcells_info[taxid]
        path = os.path.join(folder_path, f'minimalcell_taxid{taxid}_essential_genes{info["EssDataFormat"]}')
        if not os.path.exists(path):
            info["DownloadFunction"](url=info["Url"], output_path=path)
        essentiality_df = info["CreateDataFrame"](path)
        return essentiality_df.map(lambda x: x.strip() if isinstance(x, str) else x)

    def __call__(self, taxid, folder_path, fasta_file, return_all_counts=False, return_gene_to_labels=False):
        assert taxid in self.minimalcells_info, f"Taxid {taxid} not implemented yet"
        info = self.minimalcells_info[taxid]
        labels_tsv = os.path.join(folder_path, f"minimalcell_taxid{taxid}_labels.tsv")
        if os.path.exists(labels_tsv):
            stored = pd.read_csv(labels_tsv, sep="\t", dtype=str).set_index("protein_id")["label"].to_dict()
        else:
            stored, essentiality_df = None, self.get_ess_df(taxid, folder_path)
        labels = {}
        stats = {key: 0 for key in set(info["PossibleLabels"].values())}
        for record in SeqIO.parse(fasta_file, format="fasta"):
            if stored is not None:
                ess = stored.get(record.id, "No_label")
            else:
                try:
                    ess = essentiality_df.loc[essentiality_df[info["LocusColumn"]] == record.id,
                                              info["EssentialColumn"]].values[0]
                    ess = info["PossibleLabels"][ess]
                except Exception:
                    ess = "No_label"
            labels[record.id] = ess
            stats[ess] += 1
        coarse_stats = {"E": 0, "NE": 0, "QE": 0, "Other": 0}
        for key, value in stats.items():
            coarse_stats[group_label(key, coarse_stats.keys())] += value
        return_value = [coarse_stats]
        if return_all_counts:
            return_value.append(stats)
        if return_gene_to_labels:
            return_value.append(labels)
        return return_value


def download_minimalcell_proteome(taxid, folder_path, fasta_filename, accession=None):
    """NCBI proteome (locus tags from the CDS) with exact duplicates removed."""
    from experiments.essentiality.data import download_data_from_ncbi, get_records_without_duplicates
    fasta_file_path = os.path.join(folder_path, fasta_filename)
    if not os.path.exists(fasta_file_path):
        os.makedirs(folder_path, exist_ok=True)
        download_data_from_ncbi(folder_path, taxid, fasta_filename, get_gene_name_from_cds=True, accession=accession)
        _, records = get_records_without_duplicates(fasta_file=fasta_file_path, max_esmc_length=4096)
        SeqIO.write(records, fasta_file_path, "fasta")


def get_minimalcell_true_and_predict(taxid: int, folder_path: str, classifier_checkpoint: str, device="cuda:0",
                                     decision_threshold=0.5, checkpoint_dir=None, baseline_dir=None):
    """Labels and top-N predictions (N = number of E labels) for a minimal cell."""
    fasta_file_path = os.path.join(folder_path, f"ncbi_data_taxid{taxid}.fasta")
    download_minimalcell_proteome(taxid, folder_path, f"ncbi_data_taxid{taxid}.fasta")
    coarse_stats, gene_to_labels = GetMinimalCellLabels()(taxid=taxid, folder_path=folder_path,
                                                          fasta_file=fasta_file_path, return_gene_to_labels=True)
    number_of_E = sum(1 for label in gene_to_labels.values() if label == "E")
    assert os.path.exists(classifier_checkpoint), classifier_checkpoint
    predictions, prediction_scores = compute_model_predictions(
        fasta_file=fasta_file_path, classifier_checkpoint=classifier_checkpoint, device=device,
        classification_threshold=decision_threshold, return_score=True, top_n=number_of_E,
        checkpoint_dir=checkpoint_dir, baseline_dir=baseline_dir)
    return coarse_stats, gene_to_labels, predictions, prediction_scores


def yeast_labels(fasta_path: str, ogee_essentiality_file: str, taxid: int = 580240) -> Dict[str, str]:
    """S. cerevisiae labels from OGEE by SGD locus (header ``... SGDID:S000...,``); C -> QE."""
    essentiality_df = pd.read_csv(ogee_essentiality_file, sep="\t", encoding_errors="replace",
                                  usecols=["taxaID", "locus", "gene", "essentiality"], low_memory=False)
    essentiality_df = essentiality_df.loc[essentiality_df["taxaID"] == taxid]
    labels = {}
    for record in SeqIO.parse(fasta_path, format="fasta"):
        locus_name = record.description.split(" ")[2].split(":")[1].strip(",")
        try:
            ess = essentiality_df.loc[essentiality_df['locus'] == locus_name, 'essentiality'].item()
            if ess == "C":
                ess = "QE"
        except Exception:
            ess = "UNK"
        labels[record.id] = ess
    return labels


def labels_from_pickle(label_path: str) -> Dict[str, str]:
    """Labels of a genome from ``labeled_essentiality_taxid{t}.pkl`` (last OGEE call; C -> QE)."""
    with open(label_path, "rb") as f:
        stored_labels_data = pickle.load(f)
    labels = {}
    for gene, entry in stored_labels_data.items():
        ess = list(entry["Essentiality"])
        ess = ess.pop() if ess != [] else "UNK"
        labels[gene] = "QE" if ess == "C" else ess
    return labels


def heldout_genome_predictions(genome: str, cfg, classifier_checkpoint: str, device="cuda:0", checkpoint_dir=None):
    """(labels, predictions, scores) for the held-out genomes of Fig. 5B:
    ``yeast`` (taxid 580240, S288C) or ``ecoli`` (taxid 83333, K-12 MG1655)."""
    p = cfg["paths"]
    if genome == "yeast":
        fasta_path = os.path.join(p["fasta_folder"], "othersource_data_taxid580240.fasta")
        labels = yeast_labels(fasta_path, os.path.join(p["ogee_data_dir"], "gene_essentiality.txt"))
    elif genome == "ecoli":
        fasta_path = os.path.join(p["fasta_folder"], "uniprotkb_data_taxid83333.fasta")
        labels = labels_from_pickle(os.path.join(p["label_folder"], "labeled_essentiality_taxid83333.pkl"))
    else:
        raise ValueError(genome)
    number_of_E = sum(1 for v in labels.values() if v == "E")
    predictions, scores = compute_model_predictions(fasta_file=fasta_path, classifier_checkpoint=classifier_checkpoint,
                                                    device=device, return_score=True, top_n=number_of_E,
                                                    checkpoint_dir=checkpoint_dir,
                                                    baseline_dir=p["baseline_weights_folder"])
    return labels, predictions, scores


def find_classifier(classifier_dir: str, checkpoint: str = "L", modeltype: str = "2layer", seed: int = 42,
                    layer: int = 8, weights: str = "trained", splits_file: Optional[str] = None) -> str:
    """One per-layer classifier; the defaults are the published Fig. 5B model
    (``250801-2layer-082ev5el``: ProteomeLM-L, seed 42, 2 layers, layer 8)."""
    for folder in get_relevant_checkpoints(classifier_dir, modeltype, checkpoint, weights, seed, splits_file):
        if read_classifier_config(folder)["which_hidden_layer"] == layer:
            return folder
    raise FileNotFoundError(f"No {modeltype} {checkpoint} seed {seed} layer {layer} classifier in {classifier_dir}")


# --------------------------
# Command line
# --------------------------

def main(argv=None):
    from experiments.essentiality.train import add_common_args, resolve_splits_file
    parser = add_common_args(argparse.ArgumentParser(description="Test-fold metrics of one classifier run."))
    parser.add_argument("--checkpoint", default="S", choices=CHECKPOINTS)
    parser.add_argument("--classifier-layers", type=int, default=1, choices=sorted(MODEL_IDS))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--weights", default="trained", choices=("trained", "random", "statistics"))
    parser.add_argument("--plots-dir", default=None, help="Output folder (default: config paths.plots_folder)")
    parser.add_argument("--holdout", action="store_true",
                        help="Evaluate on the test fold of the held-out genomes (data_download.holdout_testset)")
    parser.add_argument("--exp-prefix", default=None)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args(argv)
    cfg = load_config(args.config, args.data_dir)
    args.classifier_dir, args.plots_dir = (resolve(cfg["data_dir"], d) for d in (args.classifier_dir, args.plots_dir))
    splits_file = resolve_splits_file(cfg, args.legacy_split_pkl, args.split_seed)
    evaluate_run(cfg, args.checkpoint, MODEL_IDS[args.classifier_layers], args.seed, args.weights, splits_file,
                 classifier_dir=args.classifier_dir, plots_dir=args.plots_dir, device=f"cuda:{args.gpu}",
                 holdout=args.holdout, exp_prefix=args.exp_prefix, checkpoint_dir=args.checkpoint_dir)


if __name__ == "__main__":
    main()
