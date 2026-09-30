"""Figures for ``notebooks/ppi_prediction_efficient.ipynb``.

Each ``plot_*`` function takes the notebook's result table (see
``notebook_inference.score_pair_chunks``) and returns a matplotlib ``Figure``,
or prints a one-line note and returns ``None`` when the figure does not apply
(no STRING scores, no supervised score, too few pairs, networkx missing...).
Large tables (all pairs of a proteome, millions of rows) are subsampled where
plotting every point would be slow, so every figure stays fast.

Colors follow the data-viz reference palette (validated for color-vision
deficiencies): categorical slots 1-2 for score series, a two-step blue ordinal
ramp for STRING confidence levels, gray for context.
"""
from __future__ import annotations

import math
import zipfile
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np
import pandas as pd

# --- palette (light surface) ---------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ("#2a78d6", "#eb6834")        # categorical slots 1-2
STRING_MEDIUM = "#86b6ef"              # ordinal blue, STRING 400-699
STRING_HIGH = "#256abf"                # ordinal blue, STRING >= 700
NOT_IN_STRING = "#c3c2b7"              # de-emphasis gray
DIVERGING = ("#e34948", "#f0efec", "#2a78d6")  # red <- gray midpoint -> blue

STRING_MEDIUM_THRESHOLD = 400
STRING_HIGH_THRESHOLD = 700
SCORE_NAMES = {
    "unsupervised_score": "Attention score",
    "supervised_score": "Supervised score",
}


def _plt():
    import matplotlib.pyplot as plt

    return plt


def _skip(message: str) -> None:
    print(f"Figure skipped: {message}")
    return None


def _rc():
    return {
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "semibold",
        "axes.labelsize": 9,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_SECONDARY,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_SECONDARY,
        "ytick.labelcolor": INK_SECONDARY,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "svg.fonttype": "none",
    }


def _style(ax, grid_axis: Optional[str] = "both") -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(length=3, width=0.8)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)


def _gene(label: str) -> str:
    return str(label).split(" | ", 1)[0].strip()


def _short_label(label: str, width: int = 34) -> str:
    """``"rpoC | RNA polymerase, beta prime subunit"`` -> ``"rpoC · RNA polymerase, beta prime su…"``."""
    text = str(label)
    if " | " in text:
        gene, desc = text.split(" | ", 1)
        desc = desc.rstrip(".")
        if len(desc) > width:
            desc = desc[: width - 1].rstrip() + "…"
        return f"{gene} · {desc}"
    return text if len(text) <= width + 8 else text[: width + 7] + "…"


def _string_colors(string_scores: np.ndarray) -> List[str]:
    return [
        STRING_HIGH if s >= STRING_HIGH_THRESHOLD else STRING_MEDIUM if s >= STRING_MEDIUM_THRESHOLD else NOT_IN_STRING
        for s in string_scores
    ]


def _string_legend_handles():
    from matplotlib.lines import Line2D

    def handle(color, label):
        return Line2D([0], [0], color=color, lw=2, marker="o", markersize=6, markeredgecolor=SURFACE, label=label)

    return [
        handle(STRING_HIGH, f"STRING ≥ {STRING_HIGH_THRESHOLD} (high confidence)"),
        handle(STRING_MEDIUM, f"STRING {STRING_MEDIUM_THRESHOLD}–{STRING_HIGH_THRESHOLD - 1}"),
        handle(NOT_IN_STRING, f"STRING < {STRING_MEDIUM_THRESHOLD} or absent"),
    ]


def _primary_score(results: pd.DataFrame, score_column: Optional[str]) -> str:
    from .notebook_inference import primary_score_column

    return score_column or primary_score_column(results)


def _sample_rows(frame: pd.DataFrame, max_rows: int, seed: int = 0) -> pd.DataFrame:
    if len(frame) <= max_rows:
        return frame
    return frame.sample(n=max_rows, random_state=seed)


# ---------------------------------------------------------------------------
# 1. Top partners
# ---------------------------------------------------------------------------

MIN_PAIRS_TO_STANDARDIZE = 50


def _panel_values(population: np.ndarray, order: np.ndarray, population_name: str, score_column: str) -> dict:
    """x values of a top-partner panel: SDs above the population mean, or raw scores for tiny populations."""
    if len(population) >= MIN_PAIRS_TO_STANDARDIZE:
        sample = population
        if len(sample) > 2_000_000:
            sample = np.random.default_rng(0).choice(sample, 2_000_000, replace=False)
        mean, std = sample.mean(), sample.std() or 1.0
        return {"x": (population[order] - mean) / std, "standardized": True,
                "x_label": f"Score, SD above the mean of {population_name}"}
    return {"x": population[order], "standardized": False, "x_label": SCORE_NAMES.get(score_column, score_column)}

def plot_top_partners(
    results: pd.DataFrame,
    query_indices: Optional[Sequence[int]] = None,
    top_k: int = 15,
    score_column: Optional[str] = None,
    max_queries: int = 6,
):
    """Ranked lollipop chart of the best predictions, one panel per query protein.

    The x axis is the score in standard deviations above the mean score of the
    same query's pairs (all scored pairs outside query mode), which makes the
    attention score (whose raw values sit just above 0.5), raw attention and
    supervised scores comparable; a panel with fewer than 50 scored pairs shows
    the raw score instead (dots only). With STRING scores, marks are colored by
    STRING confidence.
    """
    if results is None or results.empty:
        return _skip("no scored pairs.")
    score_column = _primary_score(results, score_column)
    has_string = "string_score" in results.columns
    top_k = max(1, int(top_k))
    idx_a = results["idx_a"].to_numpy()
    idx_b = results["idx_b"].to_numpy()
    scores = results[score_column].to_numpy(dtype=np.float64)

    panels = []
    queries = list(dict.fromkeys(int(q) for q in (query_indices or [])))
    if queries:
        for query in queries[:max_queries]:
            is_a = idx_a == query
            mask = is_a | (idx_b == query)
            if not mask.any():
                continue
            sub = results.loc[mask]
            s = scores[mask]
            order = np.argsort(-s, kind="stable")[:top_k]
            query_first = is_a[mask][order]
            partner_labels = np.where(query_first, sub["protein_b_label"].to_numpy()[order],
                                      sub["protein_a_label"].to_numpy()[order])
            query_label = sub["protein_a_label"].to_numpy()[order[0]] if query_first[0] else sub["protein_b_label"].to_numpy()[order[0]]
            panels.append({
                "title": _short_label(query_label, 36),
                "subtitle": f"top {len(order)} of {mask.sum():,} partners",
                "labels": [_short_label(x) for x in partner_labels],
                **_panel_values(s, order, "the query's pairs", score_column),
                "string": sub["string_score"].to_numpy()[order] if has_string else None,
            })
        if len(queries) > max_queries:
            print(f"Showing the first {max_queries} of {len(queries)} query proteins.")
    else:
        order = np.argsort(-scores, kind="stable")[:top_k]
        top = results.iloc[order]
        panels.append({
            "title": "Highest-scoring pairs",
            "subtitle": f"top {len(order)} of {len(results):,} scored pairs",
            "labels": [f"{_gene(a)} – {_gene(b)}" for a, b in zip(top["protein_a_label"], top["protein_b_label"])],
            **_panel_values(scores, order, "all scored pairs", score_column),
            "string": top["string_score"].to_numpy() if has_string else None,
        })
    if not panels:
        return _skip("none of the query proteins has a scored pair.")

    plt = _plt()
    ncols = 1 if len(panels) == 1 else 2
    nrows = math.ceil(len(panels) / ncols)
    rows = max(len(p["labels"]) for p in panels)
    with plt.rc_context(_rc()):
        fig, axes = plt.subplots(nrows, ncols, figsize=(6.4 * ncols, 0.27 * rows * nrows + 1.2 * nrows + 0.9),
                                 squeeze=False, constrained_layout=True)
        for ax, panel in zip(axes.flat, panels):
            y = np.arange(len(panel["labels"]))
            colors = _string_colors(panel["string"]) if panel["string"] is not None else [SERIES[0]] * len(y)
            x = panel["x"]
            if panel["standardized"]:
                ax.hlines(y, 0, x, colors=colors, linewidth=2, capstyle="round")
                ax.set_xlim(left=min(0.0, float(np.min(x))) * 1.05, right=float(np.max(x)) * 1.08 + 0.1)
            else:  # raw scores of a handful of pairs: dots only, no implied zero baseline
                pad = (float(np.max(x)) - float(np.min(x))) * 0.15 or abs(float(np.max(x))) * 1e-4 or 0.01
                ax.set_xlim(float(np.min(x)) - pad, float(np.max(x)) + pad)
            ax.scatter(x, y, s=40, c=colors, edgecolors=SURFACE, linewidths=1.5, zorder=3)
            ax.set_yticks(y, panel["labels"])
            ax.set_ylim(len(y) - 0.4, -0.6)
            ax.set_title(f"{panel['title']}\n", loc="left")
            ax.text(0, 1.0, panel["subtitle"], transform=ax.transAxes, fontsize=8.5, color=INK_SECONDARY, va="bottom")
            ax.set_xlabel(panel["x_label"])
            _style(ax, grid_axis="x")
            ax.tick_params(axis="y", length=0)
        for ax in list(axes.flat)[len(panels):]:
            ax.axis("off")
        fig.suptitle(f"Top predicted partners, ranked by {SCORE_NAMES.get(score_column, score_column).lower()}",
                     x=0.0, ha="left", fontsize=12, fontweight="semibold")
        if has_string:
            fig.legend(handles=_string_legend_handles(), loc="outside lower center", ncols=3)
    return fig


# ---------------------------------------------------------------------------
# 2. Agreement with STRING
# ---------------------------------------------------------------------------

def _roc_points(labels: np.ndarray, scores: np.ndarray, n_points: int = 400):
    from sklearn.metrics import roc_auc_score, roc_curve

    fpr, tpr, _ = roc_curve(labels, scores)
    grid = np.linspace(0, 1, n_points)
    return grid, np.interp(grid, fpr, tpr), roc_auc_score(labels, scores)


def plot_string_agreement(
    results: pd.DataFrame,
    score_column: Optional[str] = None,
    max_curve_pairs: int = 1_000_000,
    seed: int = 0,
):
    """How well the ranking agrees with STRING, in two panels.

    1. Fraction of the top-N predictions with a STRING score >= 400 / >= 700, against
       the same fraction among all scored pairs (the background a random ranking gets).
    2. ROC curve for STRING >= 400, for each score present (attention, supervised).
    """
    if results is None or results.empty or "string_score" not in results.columns:
        return _skip("no STRING scores (turn on compare_with_string with a STRING or UniProt-taxon source).")
    if len(results) < 3:
        return _skip("too few scored pairs to compare with STRING.")
    score_column = _primary_score(results, score_column)
    string = results["string_score"].to_numpy()
    scores = results[score_column].to_numpy(dtype=np.float64)
    n = len(scores)
    thresholds = (STRING_MEDIUM_THRESHOLD, STRING_HIGH_THRESHOLD)
    colors = (STRING_MEDIUM, STRING_HIGH)

    plt = _plt()
    from matplotlib.ticker import PercentFormatter

    with plt.rc_context(_rc()):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True,
                                       gridspec_kw={"width_ratios": [1.35, 1]})

        # Panel 1: enrichment of STRING pairs among the top-N predictions.
        order = np.argsort(-scores, kind="stable")
        n_min = 5 if n >= 50 else 1  # the first few ranks are too noisy to read as a fraction
        ns = np.unique(np.logspace(math.log10(n_min), math.log10(n), num=min(250, n)).astype(np.int64))
        top_fraction = 0.0
        for threshold, color in zip(thresholds, colors):
            hits = np.cumsum(string[order] >= threshold)
            fraction = hits[ns - 1] / ns
            background = float((string >= threshold).mean())
            top_fraction = max(top_fraction, float(fraction.max()), background)
            ax1.plot(ns, fraction, color=color, linewidth=2, solid_capstyle="round",
                     label=f"STRING ≥ {threshold}  (all pairs: {background:.1%})")
            ax1.axhline(background, color=color, linewidth=1)
        ax1.set_xscale("log")
        ax1.set_xlim(n_min, max(n, n_min + 1))
        ax1.set_ylim(0, min(1.0, top_fraction * 1.15 + 0.02))
        ax1.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax1.set_xlabel(f"Top N predictions, ranked by {SCORE_NAMES.get(score_column, score_column).lower()}")
        ax1.set_ylabel("Fraction with a STRING link")
        ax1.set_title("Enrichment among the top predictions", loc="left")
        ax1.legend(loc="upper right")
        _style(ax1)

        # The ROC uses a random subsample for very large tables.
        sample = _sample_rows(results, max_curve_pairs, seed)
        labels = (sample["string_score"].to_numpy() >= STRING_MEDIUM_THRESHOLD).astype(int)
        ax2.plot([0, 1], [0, 1], color=MUTED, linewidth=1, label="Chance (AUROC 0.50)")
        if 0 < labels.sum() < len(labels):
            for column, color in zip(("unsupervised_score", "supervised_score"), SERIES):
                if column in sample.columns:
                    fpr, tpr, auc = _roc_points(labels, sample[column].to_numpy(dtype=np.float64))
                    ax2.plot(fpr, tpr, color=color, linewidth=2, label=f"{SCORE_NAMES[column]} (AUROC {auc:.2f})")
            ax2.legend(loc="lower right")
        else:
            ax2.text(0.5, 0.5, "Needs pairs both in and out of STRING", ha="center", color=INK_SECONDARY,
                     transform=ax2.transAxes)
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1.01)
        ax2.set_aspect("equal")
        ax2.set_xlabel("False positive rate")
        ax2.set_ylabel(f"True positive rate (STRING ≥ {STRING_MEDIUM_THRESHOLD})")
        ax2.set_title("ROC against STRING", loc="left")
        _style(ax2)


        subtitle = f"{n:,} scored pairs"
        if len(sample) < n:
            subtitle += f" (ROC on a random sample of {len(sample):,})"
        fig.suptitle(f"Agreement with STRING  ·  {subtitle}", x=0.0, ha="left", fontsize=12,
                     fontweight="semibold")
    return fig


# ---------------------------------------------------------------------------
# 3. Per-head AUROC
# ---------------------------------------------------------------------------

def head_auroc_matrix(features: np.ndarray, labels: np.ndarray, n_layers: int) -> np.ndarray:
    """AUROC of each attention head's feature for ``labels``: shape ``(n_layers, n_heads)``."""
    from sklearn.metrics import roc_auc_score

    features = np.asarray(features, dtype=np.float64)
    n_heads = features.shape[1] // n_layers
    aucs = np.array([roc_auc_score(labels, features[:, j]) for j in range(features.shape[1])])
    return aucs.reshape(n_layers, n_heads)


def plot_head_auroc(
    results: pd.DataFrame,
    backbone,
    proteome,
    device: str = "cpu",
    threshold: int = STRING_MEDIUM_THRESHOLD,
    max_pairs: int = 100_000,
    min_per_class: int = 10,
    seed: int = 0,
):
    """Heatmap of how well each attention head alone separates STRING pairs (AUROC).

    Re-extracts the per-head attention features for (a sample of) the scored
    pairs with one ProteomeLM pass. Blue heads rank STRING-supported pairs
    higher than chance, red heads lower, gray is chance (0.5).
    """
    from .notebook_inference import _collect_attention_modules, extract_attention_pair_features

    if results is None or results.empty or "string_score" not in results.columns:
        return _skip("no STRING scores to evaluate the attention heads against.")
    sample = _sample_rows(results, max_pairs, seed)
    labels = (sample["string_score"].to_numpy() >= threshold).astype(int)
    n_pos = int(labels.sum())
    if n_pos < min_per_class or len(labels) - n_pos < min_per_class:
        return _skip(f"needs at least {min_per_class} pairs in and out of STRING (≥ {threshold}); got {n_pos} in.")
    pairs = sample[["idx_a", "idx_b"]].to_numpy(dtype=np.int64)
    features = extract_attention_pair_features(backbone, proteome, pairs, device=device).numpy()
    n_layers = len(_collect_attention_modules(backbone))
    aucs = head_auroc_matrix(features, labels, n_layers)
    n_heads = aucs.shape[1]

    plt = _plt()
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    spread = max(0.02, float(np.max(np.abs(aucs - 0.5))))
    norm = TwoSlopeNorm(vmin=0.5 - spread, vcenter=0.5, vmax=0.5 + spread)
    cmap = LinearSegmentedColormap.from_list("auroc", list(DIVERGING))
    best = np.argsort(-aucs.ravel())[:3]
    with plt.rc_context(_rc()):
        fig, ax = plt.subplots(figsize=(max(5.5, 0.62 * n_heads + 2.2), 0.55 * n_layers + 1.9), constrained_layout=True)
        mesh = ax.pcolormesh(np.arange(n_heads + 1), np.arange(n_layers + 1), aucs, cmap=cmap, norm=norm,
                             edgecolors=SURFACE, linewidth=2)
        for flat in best:
            layer, head = divmod(int(flat), n_heads)
            value = aucs[layer, head]
            ink = "#ffffff" if abs(value - 0.5) > 0.6 * spread else INK
            ax.text(head + 0.5, layer + 0.5, f"{value:.2f}", ha="center", va="center", fontsize=8, color=ink)
        ax.set_xticks(np.arange(n_heads) + 0.5, [str(h + 1) for h in range(n_heads)])
        ax.set_yticks(np.arange(n_layers) + 0.5, [str(layer + 1) for layer in range(n_layers)])
        ax.invert_yaxis()
        ax.set_xlabel("Head")
        ax.set_ylabel("Layer")
        for side in ax.spines.values():
            side.set_visible(False)
        ax.tick_params(length=0)
        bar = fig.colorbar(mesh, ax=ax, shrink=0.9, pad=0.02)
        bar.set_label("AUROC (0.5 = chance)")
        bar.outline.set_visible(False)
        bar.ax.tick_params(length=0)
        ax.set_title(f"Which attention heads separate STRING pairs (≥ {threshold})\n", loc="left")
        ax.text(0, 1.0, f"AUROC of each head alone; {len(labels):,} pairs, {n_pos:,} in STRING; best three labeled",
                transform=ax.transAxes, fontsize=8.5, color=INK_SECONDARY, va="bottom")
    return fig


# ---------------------------------------------------------------------------
# 4. Network view
# ---------------------------------------------------------------------------

def plot_partner_network(
    results: pd.DataFrame,
    query_indices: Optional[Sequence[int]],
    top_k: int = 10,
    score_column: Optional[str] = None,
    max_queries: int = 6,
):
    """Hub-and-spoke view of each query protein and its top partners.

    Partners sit on a ring around their query, clockwise from the top by rank.
    Partners in the top ``top_k`` of several queries sit between those queries,
    linked to each. Edge color shows STRING support; thicker edges are
    higher-ranked partners. Plain matplotlib (deterministic layout, no graph library).
    """
    queries = list(dict.fromkeys(int(q) for q in (query_indices or [])))[:max_queries]
    if not queries:
        return _skip("the network view needs query proteins (scoring_mode = query proteins).")
    if results is None or results.empty:
        return _skip("no scored pairs.")
    from .notebook_inference import summarize_top_partners

    score_column = _primary_score(results, score_column)
    top = summarize_top_partners(results, queries, top_k, score_column=score_column)
    if top.empty:
        return _skip("none of the query proteins has a scored pair.")
    has_string = "string_score" in top.columns
    query_ids = list(dict.fromkeys(top["query_id"]))
    query_names = {row.query_id: _gene(row.query) for row in top.itertuples(index=False)}
    partner_queries = top.groupby("partner_id")["query_id"].apply(lambda ids: [q for q in ids if q in query_ids])
    shared = {pid for pid, qs in partner_queries.items() if len(set(qs)) > 1 or pid in query_ids}

    per_row = 3
    spacing, radius = 4.0, 1.25
    centers = {q: ((i % per_row) * spacing, -(i // per_row) * spacing) for i, q in enumerate(query_ids)}
    positions = {q: centers[q] for q in query_ids}
    label_align = {}
    for q in query_ids:
        rows = top[(top["query_id"] == q) & ~top["partner_id"].isin(shared)]
        count = max(len(top[top["query_id"] == q]), 1)
        for row in rows.itertuples(index=False):
            angle = math.pi / 2 - 2 * math.pi * (row.rank - 1) / count
            cx, cy = centers[q]
            positions[row.partner_id] = (cx + radius * math.cos(angle), cy + radius * math.sin(angle))
            label_align[row.partner_id] = angle
    shared_slots: dict = {}
    for pid in sorted(shared - set(query_ids)):
        qs = sorted(set(partner_queries[pid]), key=query_ids.index)
        key = tuple(qs)
        slot = shared_slots.get(key, 0)
        shared_slots[key] = slot + 1
        mx = float(np.mean([centers[q][0] for q in qs]))
        my = float(np.mean([centers[q][1] for q in qs]))
        positions[pid] = (mx, my + radius + 0.55 + 0.4 * slot)  # above the gap between the hubs
        label_align[pid] = None

    plt = _plt()
    n_cols = min(len(query_ids), per_row)
    n_rows = math.ceil(len(query_ids) / per_row)
    with plt.rc_context(_rc()):
        fig, ax = plt.subplots(figsize=(3.6 * n_cols + 1.2, 3.6 * n_rows + 1.2), constrained_layout=True)
        for row in top.itertuples(index=False):
            if row.partner_id == row.query_id:
                continue
            color = _string_colors(np.array([row.string_score]))[0] if has_string else SERIES[0]
            width = 2.8 - 1.8 * (row.rank - 1) / max(top_k - 1, 1)
            (x0, y0), (x1, y1) = positions[row.query_id], positions[row.partner_id]
            ax.plot([x0, x1], [y0, y1], color=color, linewidth=width, solid_capstyle="round", zorder=1)
        partner_ids = [p for p in positions if p not in query_ids]
        ax.scatter([positions[p][0] for p in partner_ids], [positions[p][1] for p in partner_ids], s=60,
                   color=INK_SECONDARY, edgecolors=SURFACE, linewidths=2, zorder=2)
        ax.scatter([positions[q][0] for q in query_ids], [positions[q][1] for q in query_ids], s=1100,
                   color=SERIES[0], edgecolors=SURFACE, linewidths=2, zorder=3)
        names = dict(zip(top["partner_id"], top["partner"].map(_gene)))
        for pid in partner_ids:
            x, y = positions[pid]
            angle = label_align.get(pid)
            if angle is None:
                ax.annotate(names.get(pid, pid), (x, y), xytext=(8, 0), textcoords="offset points",
                            va="center", ha="left", fontsize=8, color=INK_SECONDARY, fontweight="semibold")
                continue
            dx, dy = math.cos(angle), math.sin(angle)
            ax.annotate(names.get(pid, pid), (x, y), xytext=(9 * dx, 9 * dy), textcoords="offset points",
                        va="center", ha="left" if dx > 0.2 else "right" if dx < -0.2 else "center",
                        fontsize=8, color=INK_SECONDARY)
        for q in query_ids:
            x, y = positions[q]
            ax.text(x, y, query_names[q][:7], ha="center", va="center", fontsize=8.5, fontweight="semibold",
                    color="#ffffff", zorder=4)
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.margins(0.18)
        subtitle = "Clockwise from the top by rank; thicker edge = higher rank"
        if shared - set(query_ids):
            subtitle += "; shared partners (bold) sit between hubs"
        if has_string:
            fig.legend(handles=_string_legend_handles(), loc="outside lower center", ncols=3)
        ax.set_title(f"Top {top_k} predicted partners of each query\n", loc="left")
        ax.text(0, 1.0, subtitle, transform=ax.transAxes, fontsize=8.5, color=INK_SECONDARY, va="bottom")
    return fig


# ---------------------------------------------------------------------------
# 5. Attention vs supervised scores
# ---------------------------------------------------------------------------

def plot_score_comparison(results: pd.DataFrame, max_points: int = 20_000, seed: int = 0):
    """Percentile ranks of the attention score vs the supervised score (one dot per pair).

    Dots in the top-right corner are ranked high by both; STRING-supported pairs are
    highlighted when available.
    """
    if results is None or results.empty or "supervised_score" not in results.columns:
        return _skip("no supervised score (supervised_model is 'none').")
    if len(results) < 3:
        return _skip("too few scored pairs.")
    ranks = pd.DataFrame({
        "attention": results["unsupervised_score"].rank(pct=True).to_numpy() * 100,
        "supervised": results["supervised_score"].rank(pct=True).to_numpy() * 100,
    })
    if "string_score" in results.columns:
        ranks["string"] = results["string_score"].to_numpy() >= STRING_MEDIUM_THRESHOLD
    rho = float(np.corrcoef(ranks["attention"], ranks["supervised"])[0, 1])
    sample = _sample_rows(ranks, max_points, seed)

    plt = _plt()
    with plt.rc_context(_rc()):
        fig, ax = plt.subplots(figsize=(6.2, 6.3), constrained_layout=True)
        if "string" in sample.columns:
            context, supported = sample[~sample["string"]], sample[sample["string"]]
            ax.scatter(context["attention"], context["supervised"], s=8, color=NOT_IN_STRING, alpha=0.6,
                       linewidths=0, label=f"STRING < {STRING_MEDIUM_THRESHOLD} or absent")
            ax.scatter(supported["attention"], supported["supervised"], s=16, color=SERIES[0], edgecolors=SURFACE,
                       linewidths=0.8, label=f"STRING ≥ {STRING_MEDIUM_THRESHOLD}")
            ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.13), ncols=2, markerscale=2.2)
        else:
            ax.scatter(sample["attention"], sample["supervised"], s=8, color=SERIES[0], alpha=0.5, linewidths=0)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_aspect("equal")
        ax.set_xlabel("Attention score, percentile rank")
        ax.set_ylabel("Supervised score, percentile rank")
        shown = f"; {len(sample):,} of {len(ranks):,} pairs shown" if len(sample) < len(ranks) else ""
        ax.set_title("Attention vs supervised scores\n", loc="left")
        ax.text(0, 1.0, f"Spearman ρ = {rho:.2f}{shown}", transform=ax.transAxes, fontsize=8.5,
                color=INK_SECONDARY, va="bottom")
        _style(ax)
    return fig


# ---------------------------------------------------------------------------
# Saving and showing
# ---------------------------------------------------------------------------

def save_figure(fig, name: str, prefix: Union[Path, str], dpi: int = 160) -> List[Path]:
    """Save ``fig`` as ``<prefix>_<name>.png`` and ``.svg``; returns the paths."""
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for extension in ("png", "svg"):
        path = prefix.parent / f"{prefix.name}_{name}.{extension}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        paths.append(path)
    return paths


def show_figure(fig, name: str, prefix: Union[Path, str]) -> List[Path]:
    """Display ``fig`` in the notebook, save it next to the results and close it.

    Returns the saved paths (empty when ``fig`` is None, i.e. the figure was skipped).
    """
    if fig is None:
        return []
    paths = save_figure(fig, name, prefix)
    try:
        from IPython.display import display

        display(fig)
    except ImportError:  # pragma: no cover - IPython ships with Jupyter/Colab
        pass
    _plt().close(fig)
    return paths


def zip_figures(paths: Sequence[Union[Path, str]], zip_path: Union[Path, str]) -> Optional[Path]:
    """Bundle figure files into one zip (one browser download on Colab)."""
    paths = [Path(p) for p in paths if Path(p).exists()]
    if not paths:
        return None
    zip_path = Path(zip_path)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, arcname=path.name)
    return zip_path

