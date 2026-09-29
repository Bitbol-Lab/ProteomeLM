"""Figures of the essentiality experiment (ProteomeLM PNAS paper, Fig. 5 and SI).

* ``fig5a``: test-fold AUROC vs relative layer depth for ESM-C and ProteomeLM-XS/S/M/L
  (bootstrap-weighted mean over seeds; 2-layer classifier by default).
* ``baselines``: same for trained vs random vs resampled ("statistics") ProteomeLM weights.
* ``fig5b``: donuts of labelled (outer ring) vs predicted (inner ring, top-N rule)
  essential genes for S. cerevisiae, E. coli K-12 and JCVI-Syn1.0 / Syn3A.
* ``depth``: 1- vs 2- vs 3-layer classifier ROC/PR curves (ProteomeLM-L, one layer).
* ``interpretability``: Two-NN intrinsic dimension, PCA ID and matrix entropy per layer.
"""
import argparse
import ast
import json
import os
import pickle
import subprocess
import warnings
from dataclasses import replace
from typing import List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, to_hex
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.metrics import auc as compute_auc

from experiments.essentiality.common import MODEL_IDS, PLM_SIZES, load_config, resolve
from experiments.essentiality.evaluate import (PlotArguments, auroc_on_labelled, find_classifier,
                                               get_minimalcell_true_and_predict, get_filename, get_sorted_fraction,
                                               group_label, heldout_genome_predictions, read_classifier_config,
                                               sort_items_by_label)

matplotlib.use("Agg")


def _paper_style():
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['text.usetex'] = False
    plt.rcParams['font.size'] = 14


class EssentialityColors:
    def __init__(self, gradient=False):
        if gradient:
            cmap = plt.get_cmap('plasma')
            colors = [to_hex(cmap(val)) for val in np.linspace(0, 0.9, 4)]
        else:
            colors = ["#fb49b0", "#00beff", "#b51d14", "#ddb310"]
        self.model_to_color = {"L": {"color": colors[0], "alpha": 1},
                               "M": {"color": colors[1], "alpha": 1},
                               "S": {"color": colors[2], "alpha": 1},
                               "XS": {"color": colors[3], "alpha": 1},
                               "ESMC": {"color": "#cacaca", "alpha": 1}}


# --------------------------
# Metric vs layer
# --------------------------

def _seeds(args: PlotArguments) -> List[int]:
    return args.random_seed if isinstance(args.random_seed, list) else [args.random_seed]


def _per_layer(curves: dict, value_fn, classifier_dir: Optional[str] = None):
    """Array over layers of ``value_fn(curve)``; the layer of each entry is read from the
    classifier checkpoint it is keyed by. Returns (values, n_layers)."""
    infos = [(read_classifier_config(ckpt, classifier_dir), curve) for ckpt, curve in curves.items()]
    n_layers = infos[0][0]["n_layers"]
    if len(infos) != n_layers:
        warnings.warn(f"{len(infos)} checkpoints for {n_layers} layers (missing or duplicate layers); "
                      "missing layers are 0 and later duplicates win")
    values = np.zeros(shape=(n_layers,))
    for cfg, curve in infos:
        values[cfg["which_hidden_layer"]] = value_fn(curve)
    return values, n_layers


def plot_auc_vs_layer(args: PlotArguments, ax_auc=None, which_plot="precision-recall", classifier_dir=None, **kwargs):
    """Area under the pooled ROC or PR curve per layer, mean +- s.e.m. over seeds."""
    assert which_plot in ["precision-recall", "roc"]
    filename = "precision-recall-curves.pkl" if which_plot == "precision-recall" else "roc-auc-scores.pkl"

    def area(curve):
        if which_plot == "precision-recall":
            y, x, _ = curve  # precision, recall
        else:
            _, (x, y) = curve  # fpr, tpr
        return compute_auc(x, y)

    all_auc = []
    for seed in _seeds(args):
        with open(get_filename(replace(args, random_seed=seed), filename), "rb") as f:
            values, n_layers = _per_layer(pickle.load(f), area, classifier_dir)
        all_auc.append(values)
    all_auc = np.array(all_auc)
    if ax_auc:
        marker = kwargs.pop("marker", "o")
        markersize = kwargs.pop("markersize", 5)
        x = np.arange(len(all_auc[0])) / (n_layers - 1)
        ax_auc.errorbar(x, np.average(all_auc, axis=0), yerr=np.std(all_auc, axis=0) / np.sqrt(all_auc.shape[0]),
                        linestyle="", marker=marker, markersize=markersize, **kwargs)
        ax_auc.errorbar(x, np.average(all_auc, axis=0), linestyle="--", **kwargs)
    return all_auc


def plot_bootstrap_auroc_vs_layer(args: PlotArguments, ax_auc=None, classifier_dir=None, **kwargs):
    """Bootstrapped test AUROC per layer, combined over seeds.

    Each seed's AUROC is weighted by 1/(s.e. * sqrt(1000)) (the bootstrap standard
    deviation); the error bar is 1/sqrt(mean(weight^2)), as in the published figure.
    """
    all_auc, all_auc_std = [], []
    for seed in _seeds(args):
        with open(get_filename(replace(args, random_seed=seed), "otherscores.pkl"), "rb") as f:
            curves = pickle.load(f)
        auc_vs_layer, n_layers = _per_layer(curves, lambda c: c["auroc_with_std"][0], classifier_dir)
        auc_std_vs_layer, _ = _per_layer(curves, lambda c: 1 / (c["auroc_with_std"][1] * np.sqrt(1000)), classifier_dir)
        all_auc.append(auc_vs_layer)
        all_auc_std.append(auc_std_vs_layer)
    all_auc = np.array(all_auc)
    all_auc_std = np.array(all_auc_std)
    mean_auc = np.average(all_auc, axis=0, weights=all_auc_std)
    mean_std = 1 / np.sqrt(np.average(all_auc_std ** 2, axis=0))
    if ax_auc:
        marker = kwargs.pop("marker", "o")
        markersize = kwargs.pop("markersize", 5)
        x = np.arange(len(all_auc[0])) / (n_layers - 1)
        ax_auc.errorbar(x, mean_auc, yerr=mean_std, linestyle="", marker=marker, markersize=markersize, **kwargs)
        ax_auc.errorbar(x, mean_auc, linestyle="--", **kwargs)
    return mean_auc, mean_std


def plot_essentiality_figure_for_paper(args: PlotArguments, output_dir: str, gradient=True, which_plot="roc",
                                       extension="pdf", classifier_dir=None) -> str:
    """Fig. 5A: AUROC vs layer depth for ESM-C and the four ProteomeLM sizes."""
    _paper_style()
    model_to_color = EssentialityColors(gradient=gradient).model_to_color
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    rows = []
    for checkpoint in ["ESMC", "XS", "S", "M", "L"]:
        a = replace(args, checkpoint=checkpoint)
        try:
            if which_plot == "roc":
                avg_auc, std_auc = plot_bootstrap_auroc_vs_layer(a, ax_auc=ax, classifier_dir=classifier_dir,
                                                                 label=checkpoint, **model_to_color[checkpoint])
            else:
                all_auc = plot_auc_vs_layer(a, ax_auc=ax, which_plot=which_plot, classifier_dir=classifier_dir,
                                            label=checkpoint, **model_to_color[checkpoint])
                avg_auc, std_auc = np.average(all_auc, axis=0), np.std(all_auc, axis=0)
            print(f"{checkpoint} --> {np.max(avg_auc):.4f} <-- layer {np.argmax(avg_auc)}")
            rows += [{"model": checkpoint, "layer": i, "depth": i / (len(avg_auc) - 1), "auc": m, "err": s}
                     for i, (m, s) in enumerate(zip(avg_auc, std_auc))]
        except FileNotFoundError as e:
            print(f"model {checkpoint} --> {e}")

    ax.set_ylabel("AUC")
    ax.set_xlabel("Layer depth")
    custom_legend = [Line2D([0], [0], marker='o', linestyle='None', label=model, **model_to_color[model])
                     for model in model_to_color]
    fig.legend(custom_legend, model_to_color.keys(), loc="lower right", bbox_to_anchor=(1, 0.12))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    os.makedirs(output_dir, exist_ok=True)
    stem = f"fig5a-essentiality-AU{which_plot}-{args.modeltype}{'-holdout' if args.holdout else ''}"
    output_filename = os.path.join(output_dir, f"{stem}.{extension}")
    fig.savefig(output_filename, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(os.path.join(output_dir, f"{stem}.csv"), index=False)
    print(f"saved at {output_filename}")
    return output_filename


def plot_baseline_figure(args: PlotArguments, output_dir: str, classifier_dir=None, extension="pdf") -> str:
    """SI: bootstrapped AUROC vs depth for trained, random and resampled ("statistics") weights."""
    _paper_style()
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 7), sharey=False, sharex=True, constrained_layout=True)
    axes = ax.flatten()
    model_to_color = EssentialityColors(gradient=True).model_to_color
    weights_to_fig_kwargs = {"trained": {"alpha": 1, "marker": "o", "markersize": 5},
                             "statistics": {"alpha": 1, "marker": "v", "markersize": 6},
                             "random": {"alpha": 1, "marker": "*", "markersize": 8}}
    for j, checkpoint in enumerate(PLM_SIZES):
        cmap = LinearSegmentedColormap.from_list("mycmap", ["black", model_to_color[checkpoint]["color"], "white"])
        weight_to_colors = {w: cmap(i) for w, i in zip(["trained", "statistics", "random"], np.linspace(0.35, 0.8, 3))}
        for which_weights in ["trained", "random", "statistics"]:
            try:
                plot_bootstrap_auroc_vs_layer(replace(args, which_weights=which_weights, checkpoint=checkpoint),
                                              ax_auc=axes[j], classifier_dir=classifier_dir,
                                              color=weight_to_colors[which_weights], label=which_weights,
                                              **dict(weights_to_fig_kwargs[which_weights]))
            except FileNotFoundError as e:
                print(f"model {checkpoint} weights {which_weights} --> {e}")
        axes[j].set_title(checkpoint)
        axes[j].spines['top'].set_visible(False)
        axes[j].spines['right'].set_visible(False)
    ax[1, 0].set_xlabel("Layer depth")
    ax[1, 1].set_xlabel("Layer depth")
    ax[0, 0].set_ylabel("AUC")
    ax[1, 0].set_ylabel("AUC")
    custom_legend = [Line2D([0], [0], **kw, linestyle='None', label=w, color="black")
                     for w, kw in weights_to_fig_kwargs.items()]
    fig.legend(custom_legend, weights_to_fig_kwargs.keys(), loc="center right", bbox_to_anchor=(1.2, 0.5))
    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, f"baseline-bootstrap-{args.modeltype}.{extension}")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"saved at {out}")
    return out


def plot_classifier_depth_comparison(args: PlotArguments, output_dir: str, layer: int = 9, classifier_dir=None,
                                     extension="pdf") -> str:
    """SI: ROC and PR curves of the 1-, 2- and 3-layer classifiers on one ProteomeLM layer."""
    _paper_style()
    colors = ["#00beff", "#00b25d", "#fb49b0"]
    model_to_label = {"simpleclassifier": "One layer", "2layer": "Two layers", "3layer": "Three layers"}
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))
    for color, model_id in zip(colors, ["simpleclassifier", "2layer", "3layer"]):
        a = replace(args, modeltype=model_id)
        try:
            with open(get_filename(a, "roc-auc-scores.pkl"), "rb") as f:
                roc_auc_curves = pickle.load(f)
            with open(get_filename(a, "precision-recall-curves.pkl"), "rb") as f:
                pr_curves = pickle.load(f)
        except FileNotFoundError as e:
            print(f"{model_id} --> {e}")
            continue
        for checkpoint, (_, (fpr, tpr)) in roc_auc_curves.items():
            if read_classifier_config(checkpoint, classifier_dir)["which_hidden_layer"] != layer:
                continue
            precision, recall, _ = pr_curves[checkpoint]
            ax[0].plot(fpr, tpr, label=model_to_label[model_id], color=color)
            ax[1].plot(recall, precision, label=model_to_label[model_id], color=color)
            print(f"{model_to_label[model_id]} --> AUC {compute_auc(fpr, tpr):.3f}")
    for a_, (xl, yl) in zip(ax, [("False Positive Rate", "True Positive Rate"), ("Recall", "Precision")]):
        a_.spines['top'].set_visible(False)
        a_.spines['right'].set_visible(False)
        a_.set_xlabel(xl)
        a_.set_ylabel(yl)
    plt.tight_layout()
    handles, labels = ax[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(unique.values(), unique.keys(), loc="center right", bbox_to_anchor=(1.2, 0.5))
    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, f"deeper-is-better.{extension}")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# --------------------------
# Fig. 5B donuts
# --------------------------

DONUT_COLORS = {"E": "#4053d3", 'QE': '#00beff', "NE": "#ddb310", 'Other': '#bdbdbd'}


def donut_plot(fig, ax, true_data, prediction):
    """Outer ring: labels; inner ring: predictions; one equal slice per gene, genes
    grouped by label. The table gives the fraction of E/NE/QE genes predicted E."""
    sorted_items1, sorted_items2 = sort_items_by_label(true_data, prediction)
    stats_dict = get_sorted_fraction(sorted_items1, sorted_items2)
    values = [1] * len(sorted_items1)
    colors_outer = [DONUT_COLORS.get(group_label(v), '#000000') for _, v in sorted_items1]
    colors_inner = [DONUT_COLORS.get(group_label(v), "#000000") for _, v in sorted_items2]
    ax.pie(values, colors=colors_outer, radius=1.0, startangle=90, counterclock=False,
           wedgeprops=dict(width=0.25, edgecolor='none'))
    ax.pie(values, colors=colors_inner, radius=0.7, startangle=90, counterclock=False,
           wedgeprops=dict(width=0.25, edgecolor='none'))
    ax.set_aspect('equal')
    ax.axis('off')
    legend_labels = {v: k for k, v in DONUT_COLORS.items()}
    handles = [Patch(facecolor=color, label=legend_labels[color])
               for color in sorted(set(colors_outer + colors_inner), key=lambda c: list(DONUT_COLORS.values()).index(c))]
    ax.legend(handles=handles, loc='center')
    ax.set_title('Predicted (inner ring) vs Labelled (outer ring)', fontsize=10)
    cell_text = [["E", f"{100 * stats_dict['E in E']:.1f} %"],
                 ["NE", f"{100 * stats_dict['E in NE']:.1f} %"],
                 ["QE", f"{100 * stats_dict['E in QE']:.1f} %"]]
    print(cell_text)
    table = ax.table(cellText=cell_text, colLabels=['True', 'Predicted E'], cellLoc='center', loc='bottom',
                     bbox=[0.3, -0.25, 0.4, 0.15], edges="vertical")
    table.scale(1, 1.2)
    for _, cell in table.get_celld().items():
        cell.set_linewidth(0.5)
    return fig, ax


def plot_heldout_donut(genome: str, cfg, classifier_checkpoint: str, output_dir: str, device="cuda:0",
                       checkpoint_dir=None, extension="svg") -> str:
    """Fig. 5B donut for the held-out ``yeast`` or ``ecoli`` genome."""
    labels, predictions, scores = heldout_genome_predictions(genome, cfg, classifier_checkpoint, device=device,
                                                             checkpoint_dir=checkpoint_dir)
    print(f"{genome} --> AUC {auroc_on_labelled(labels, scores)}")
    fig, ax = plt.subplots(figsize=(10, 6))
    donut_plot(fig, ax, labels, predictions)
    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, f"{genome}-donut-final.{extension}")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_donuts_side_by_side(cfg, classifier_checkpoint: str, output_dir: str, device="cuda:0", checkpoint_dir=None,
                             extension="svg") -> str:
    """Fig. 5B donuts for the minimal cells JCVI-Syn1.0 (766747) and JCVI-Syn3A (2144189)."""
    folder_path = cfg["paths"]["minimalcell_folder"]
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    handles = None
    for i, (cellname, taxid) in enumerate([("JCVI-Syn1.0", 766747), ("JCVI-Syn3A", 2144189)]):
        _, gene_to_labels, predictions, pred_scores = get_minimalcell_true_and_predict(
            taxid, folder_path, classifier_checkpoint, device=device, checkpoint_dir=checkpoint_dir,
            baseline_dir=cfg["paths"]["baseline_weights_folder"])
        print(f"{cellname} --> AUC {auroc_on_labelled(gene_to_labels, pred_scores)}")
        donut_plot(fig, ax[i], gene_to_labels, predictions)
        legend = ax[i].get_legend()
        handles = legend.legend_handles
        legend.set_visible(False)
        ax[i].set_title(cellname, y=0.95)
    fig.subplots_adjust(wspace=0, bottom=0)
    fig.legend(handles=handles, loc="center right", bbox_to_anchor=(1, 0.5), bbox_transform=fig.transFigure,
               borderaxespad=0.0)
    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, f"minimalcells-donuts-final.{extension}")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# --------------------------
# Interpretability (SI)
# --------------------------

def mask_superkingdom(taxid_list: List[int]):
    """(is_eukaryote, is_prokaryote) masks, via the `ncbi-taxonomist` CLI (pip install ncbi-taxonomist)."""
    def get_domain(taxid):
        result = subprocess.run(f"ncbi-taxonomist collect -t {taxid} | grep domain", shell=True, check=True,
                                capture_output=True, text=True)
        return json.loads(result.stdout.strip())['name']
    is_eukaryote = np.array([get_domain(taxid) == "Eukaryota" for taxid in taxid_list])
    return is_eukaryote, ~is_eukaryote


def superkingdoms(taxids) -> dict:
    euk_mask, _ = mask_superkingdom(taxids)
    return {taxid: ("Eukaryote" if is_euk else "Prokaryote") for taxid, is_euk in zip(taxids, euk_mask)}


def bootstrap_correlation(v1, v2, n_bootstraps=10000):
    """(mean, std) of the Pearson correlation over resampled genomes (seed 42)."""
    from scipy.stats import pearsonr
    rng = np.random.RandomState(42)
    scores = []
    for _ in range(n_bootstraps):
        indices = rng.randint(0, len(v1), len(v1))
        scores.append(pearsonr(v1[indices], v2[indices]).statistic)
    return np.average(scores), np.std(scores)


def pca_id(pca_variance, threshold: float) -> int:
    """Number of principal components needed to explain more than ``threshold`` of the variance
    (0-based index of the first cumulative ratio above it, as in the published figure)."""
    if isinstance(pca_variance, str):
        pca_variance = ast.literal_eval(pca_variance)
    cumulative_var = np.cumsum([var for _, var in pca_variance])
    return int(np.min(np.argwhere(cumulative_var > threshold)))


def load_interpretability(folder: str, checkpoint: str, weights="trained", pca_threshold=0.95) -> pd.DataFrame:
    data_df = pd.read_csv(os.path.join(folder, f"{weights}-interpretability-{checkpoint}.csv"))
    data_df["PCA ID"] = [pca_id(v, pca_threshold) for v in data_df["PCA Variance"]]
    return data_df


def plot_interpretability(interp_folder: str, output_dir: str, label_folder: str, checkpoint="M", extension="pdf"):
    """Layer-wise Two-NN ID, PCA ID (95%) and matrix entropy (prokaryotes vs eukaryotes),
    plus their correlations across genomes and with genome size."""
    import seaborn as sns
    _paper_style()
    os.makedirs(output_dir, exist_ok=True)
    outputs = []
    model_to_color = EssentialityColors(gradient=True).model_to_color

    # Metrics per layer, one model
    data_df = load_interpretability(interp_folder, checkpoint).loc[:, ["TaxID", "Layer", "ID", "Entropy Std", "PCA ID"]]
    kingdom = superkingdoms(sorted(set(data_df["TaxID"].tolist())))
    data_df["Superkingdom"] = data_df["TaxID"].map(kingdom)
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 3), constrained_layout=True)
    ylabel = {"ID": "Two-NN ID", "Entropy Std": "Matrix Entropy", "PCA ID": "PCA ID"}
    for i, metric in enumerate(["ID", "PCA ID", "Entropy Std"]):
        for alpha, superking in [(1, "Prokaryote"), (0.3, "Eukaryote")]:
            sns.pointplot(data=data_df.loc[data_df["Superkingdom"] == superking], x="Layer", y=metric,
                          color=model_to_color[checkpoint]["color"], alpha=alpha, ax=ax[i], orient="v",
                          errorbar="se", label=superking)
        if ax[i].legend_ is not None:
            ax[i].legend_.remove()
        ax[i].set_ylabel(ylabel[metric])
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
    handles, labels = ax[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(unique.values(), unique.keys(), loc="center right", bbox_to_anchor=(1.15, 0.5))
    outputs.append(os.path.join(output_dir, f"layer-interpretability-{checkpoint}.{extension}"))
    fig.savefig(outputs[-1], bbox_inches="tight")
    plt.close(fig)

    # Correlations between the three measures, all models
    for name, pairs, titles in [
        ("PCA-ID-correlation-95", [("Entropy Std", "ID"), ("PCA ID", "ID")], ["Two-NN ID vs Entropy", "Two-NN ID vs PCA ID"]),
        ("entropy-ID-correlation", [("Entropy Std", "ID"), ("Entropy Std", "PCA ID")], ["Two-NN ID vs Entropy", "PCA ID vs Entropy"]),
    ]:
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4), sharey=True)
        for size in PLM_SIZES:
            try:
                df = load_interpretability(interp_folder, size)
            except FileNotFoundError:
                continue
            layers = np.sort(df["Layer"].unique()).astype(int)
            for k, (a, b) in enumerate(pairs):
                stats = [bootstrap_correlation(df.loc[df["Layer"] == layer, a].to_numpy().astype(float),
                                               df.loc[df["Layer"] == layer, b].to_numpy().astype(float))
                         for layer in layers]
                ax[k].errorbar(layers / (len(layers) - 1), [s[0] for s in stats], yerr=[s[1] for s in stats],
                               marker="o", linestyle="--", label=size, **model_to_color[size])
        for k, title in enumerate(titles):
            ax[k].set_title(title)
            ax[k].set_xlabel("Layer depth")
            ax[k].spines['top'].set_visible(False)
            ax[k].spines['right'].set_visible(False)
        ax[0].set_ylabel("Pearson correlation")
        handles, labels = ax[0].get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        fig.legend(unique.values(), unique.keys(), loc="center right", bbox_to_anchor=(1.15, 0.5))
        fig.tight_layout()
        outputs.append(os.path.join(output_dir, f"{name}.{extension}"))
        fig.savefig(outputs[-1], bbox_inches="tight")
        plt.close(fig)

    # Two-NN ID vs genome size (number of proteins), per superkingdom
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
    for size in PLM_SIZES:
        try:
            df = load_interpretability(interp_folder, size)
        except FileNotFoundError:
            continue
        df["Superkingdom"] = df["TaxID"].map(kingdom)
        genome_size = {}
        for taxid in set(df["TaxID"].tolist()):
            with open(os.path.join(label_folder, f"labeled_essentiality_taxid{taxid}.pkl"), "rb") as f:
                genome_size[taxid] = len(pickle.load(f))
        df["Genome Size"] = df["TaxID"].map(genome_size)
        layers = np.sort(df["Layer"].unique()).astype(int)
        for i, superking in enumerate(["Eukaryote", "Prokaryote"]):
            rel = df.loc[df["Superkingdom"] == superking]
            stats = []
            for layer in layers:
                v = rel.loc[rel["Layer"] == layer, ["Genome Size", "ID"]].to_numpy().astype(float)
                stats.append(bootstrap_correlation(v[:, 0], v[:, 1]))
            ax[i].errorbar(layers / (len(layers) - 1), [s[0] for s in stats], yerr=[s[1] for s in stats],
                           marker="o", linestyle="--", label=size, **model_to_color[size])
            ax[i].set_title(superking)
            ax[i].set_xlabel("Layer depth")
            ax[i].spines['top'].set_visible(False)
            ax[i].spines['right'].set_visible(False)
    ax[0].set_ylabel("Pearson correlation")
    handles, labels = ax[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(unique.values(), unique.keys(), loc="center right", bbox_to_anchor=(1.15, 0.5))
    fig.tight_layout()
    outputs.append(os.path.join(output_dir, f"twoNN-ID-and-genome-size-correlation.{extension}"))
    fig.savefig(outputs[-1], bbox_inches="tight")
    plt.close(fig)

    # PCA ID at several variance thresholds, one model
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
    base = pd.read_csv(os.path.join(interp_folder, f"trained-interpretability-{checkpoint}.csv"))
    for alpha, thr in zip(sorted(np.linspace(0.1, 1, 4), reverse=True), [0.99, 0.95, 0.8, 0.5]):
        df = base.assign(**{"PCA ID": [pca_id(v, thr) for v in base["PCA Variance"]]})
        sns.pointplot(data=df, x="Layer", y="PCA ID", color=model_to_color[checkpoint]["color"], alpha=alpha, ax=ax,
                      orient="v", errorbar="se", label=thr)
    if ax.legend_ is not None:
        ax.legend_.remove()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(unique.values(), unique.keys(), loc="center right", bbox_to_anchor=(1.17, 0.5))
    outputs.append(os.path.join(output_dir, f"PCA-vs-threshold.{extension}"))
    fig.savefig(outputs[-1], bbox_inches="tight")
    plt.close(fig)
    return outputs


# --------------------------
# Command line
# --------------------------

def main(argv=None):
    from experiments.essentiality.train import add_common_args, resolve_splits_file
    parser = add_common_args(argparse.ArgumentParser(description=__doc__.split("\n")[0]))
    parser.add_argument("figure", choices=["fig5a", "baselines", "fig5b", "depth", "interpretability"])
    parser.add_argument("--plots-dir", default=None, help="Metric pickles (default: config paths.plots_folder)")
    parser.add_argument("--figures-dir", default=None, help="Output folder (default: config paths.figures_folder)")
    parser.add_argument("--classifier-layers", type=int, default=2, choices=sorted(MODEL_IDS),
                        help="Classifier depth plotted in fig5a/baselines (published Fig. 5A: 2)")
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="Seeds to combine (default: 42-46 for fig5a, 42-45 for baselines)")
    parser.add_argument("--which-plot", default="roc", choices=["roc", "precision-recall"])
    parser.add_argument("--holdout", action="store_true")
    parser.add_argument("--extension", default=None)
    parser.add_argument("--donut-checkpoint", default=None,
                        help="Classifier folder for fig5b (default: ProteomeLM-L, 2 layers, seed 42, layer 8)")
    parser.add_argument("--donut-layer", type=int, default=8)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args(argv)

    cfg = load_config(args.config, args.data_dir)
    p = cfg["paths"]
    for key in ("classifier_dir", "plots_dir", "figures_dir", "donut_checkpoint"):
        setattr(args, key, resolve(cfg["data_dir"], getattr(args, key)))
    classifier_dir = args.classifier_dir or p["classifier_checkpoints_folder"]
    figures_dir = args.figures_dir or p["figures_folder"]
    plot_args = PlotArguments(plots_data_directory=args.plots_dir or p["plots_folder"], checkpoint=None,
                              modeltype=MODEL_IDS[args.classifier_layers], splits_thr=cfg["split"]["threshold"],
                              holdout=args.holdout, which_weights="trained")
    if args.figure == "fig5a":
        seeds = args.seeds or [42, 43, 44, 45, 46]
        plot_essentiality_figure_for_paper(replace(plot_args, random_seed=seeds), figures_dir,
                                           which_plot=args.which_plot, extension=args.extension or "pdf",
                                           classifier_dir=classifier_dir)
    elif args.figure == "baselines":
        seeds = args.seeds or [42, 43, 44, 45]
        plot_baseline_figure(replace(plot_args, random_seed=seeds), figures_dir, classifier_dir=classifier_dir,
                             extension=args.extension or "pdf")
    elif args.figure == "depth":
        plot_classifier_depth_comparison(replace(plot_args, checkpoint="L", random_seed=(args.seeds or [42])[0]),
                                         figures_dir, classifier_dir=classifier_dir, extension=args.extension or "pdf")
    elif args.figure == "fig5b":
        splits_file = resolve_splits_file(cfg, args.legacy_split_pkl, args.split_seed)
        ckpt = args.donut_checkpoint or find_classifier(classifier_dir, "L", "2layer", 42, args.donut_layer,
                                                        splits_file=splits_file)
        print(f"Fig. 5B classifier: {ckpt}")
        device = f"cuda:{args.gpu}"
        ext = args.extension or "svg"
        plot_heldout_donut("yeast", cfg, ckpt, figures_dir, device, args.checkpoint_dir, ext)
        plot_donuts_side_by_side(cfg, ckpt, figures_dir, device, args.checkpoint_dir, ext)
        plot_heldout_donut("ecoli", cfg, ckpt, figures_dir, device, args.checkpoint_dir, ext)
    else:
        plot_interpretability(p["interpretability_folder"], figures_dir, p["label_folder"],
                              extension=args.extension or "pdf")


if __name__ == "__main__":
    main()
