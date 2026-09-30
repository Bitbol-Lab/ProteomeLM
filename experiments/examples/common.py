"""Shared helpers for the validation examples (cct.py, paris.py, ribosomes.py).

Plot styling, the symmetrized pair-attention read-out, and OrthoDB group
embeddings for a downloaded proteome.
"""

import colorsys
import os

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import torch

from proteomelm.utils.proteome import (
    build_group_embeddings_for_proteome,
    download_proteome,
    load_orthodb_group_vectors,
)

# =============================================================================
# PLOTTING STYLE
# =============================================================================

colorspal6 = [
    (0.25098039215686274, 0.3254901960784314, 0.8274509803921568),
    (0.8666666666666667, 0.7019607843137254, 0.06274509803921569),
    (0.7098039215686275, 0.11372549019607843, 0.0784313725490196),
    (0.0, 0.7450980392156863, 1.0),
    (0.984313725490196, 0.28627450980392155, 0.6901960784313725),
    (0.0, 0.6980392156862745, 0.36470588235294116),
    (0.792156862745098, 0.792156862745098, 0.792156862745098),
]

COLOR_BLUE = colorspal6[0]
COLOR_YELLOW = colorspal6[1]
COLOR_RED = colorspal6[2]
COLOR_CYAN = colorspal6[3]
COLOR_MAGENTA = colorspal6[4]
COLOR_GREEN = colorspal6[5]
COLOR_GRAY = colorspal6[6]


def to_colormap(base_color, dark=0.2, light=0.9, name="custom_colormap"):
    """Light-to-dark colormap through ``base_color`` (same hue and saturation)."""
    rgb = mcolors.to_rgb(base_color)
    h, l, s = colorsys.rgb_to_hls(*rgb)
    colors = [
        colorsys.hls_to_rgb(h, light, s),
        colorsys.hls_to_rgb(h, l, s),
        colorsys.hls_to_rgb(h, dark, s),
    ]
    rgb_colors = [mcolors.to_rgb(c) for c in colors]
    return mcolors.LinearSegmentedColormap.from_list(name, rgb_colors)


def apply_plot_style(base_size=16):
    """Global rcParams: titles at ``base_size + 2``, legend and ticks at ``base_size - 2``."""
    plt.rcParams.update({
        "font.family": "Arial",
        "text.usetex": False,
        "font.size": base_size,
        "axes.titlesize": base_size + 2,
        "axes.labelsize": base_size,
        "legend.fontsize": base_size - 2,
        "xtick.labelsize": base_size - 2,
        "ytick.labelsize": base_size - 2,
        "svg.fonttype": "none",
    })


def style_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def save_figure(fig, output_dir, name, also_png=False):
    """Save ``fig`` as PDF + SVG (optionally PNG) under ``output_dir`` and close it."""
    pdf_path = os.path.join(output_dir, f"{name}.pdf")
    svg_path = os.path.join(output_dir, f"{name}.svg")
    fig.savefig(pdf_path, dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
    if also_png:
        png_path = os.path.join(output_dir, f"{name}.png")
        fig.savefig(png_path, dpi=400, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# =============================================================================
# ATTENTION READ-OUT
# =============================================================================

def pair_attn(attn, i, j):
    """Symmetrized attention between proteins ``i`` and ``j``: (mean a_ij + mean a_ji) / 2.

    ``attn`` is ``[..., N, N]``; the mean runs over the leading axes
    (all layers and heads for ``[L, H, N, N]``, nothing for one head's ``[N, N]``).
    """
    return (attn[..., i, j].mean().item() + attn[..., j, i].mean().item()) / 2


# =============================================================================
# ORTHODB GROUP EMBEDDINGS
# =============================================================================

def build_orthodb_group_embeds(
    fasta_path,
    esm_embeddings,
    output_dir,
    organism,
    proteome_id,
    db_path,
    tsv_path=None,
    min_group_size=0,
    cache_path=None,
):
    """Per-protein OrthoDB group embeddings for a proteome, or None if inputs are missing.

    ``db_path`` is the directory with ``group_vectors_*.pkl`` (a path to one of
    those files is accepted too). Without ``tsv_path``, the UniProt OrthoDB
    cross-reference TSV is downloaded to ``{output_dir}/{organism}_orthodb.tsv``.
    Proteins without an OrthoDB group keep their ESM-C embedding. When None is
    returned, ProteomeLM falls back to the ESM-C embeddings as group embeddings.
    """
    if cache_path and os.path.exists(cache_path):
        print(f"  Loading cached OrthoDB group embeddings: {cache_path}")
        return torch.load(cache_path, map_location='cpu')

    if db_path and os.path.isfile(db_path):
        db_path = os.path.dirname(db_path.split(',')[0])

    if tsv_path is None:
        tsv_path = os.path.join(output_dir, f"{organism}_orthodb.tsv")
        if not os.path.exists(tsv_path):
            download_proteome(
                proteome_id=proteome_id,
                output_dir=output_dir,
                organism_name=organism,
                reviewed_only=True,
                include_isoforms=False,
                download_orthodb=True,
            )

    if not db_path or not os.path.exists(db_path) or not os.path.exists(tsv_path):
        print(f"  Warning: OrthoDB TSV ({tsv_path}) or DB path ({db_path}) missing, "
              "using ESM group embeddings")
        return None

    orthodb_means = load_orthodb_group_vectors(db_path, min_group_size=min_group_size)
    group_embeds, mask = build_group_embeddings_for_proteome(
        fasta_path=fasta_path,
        orthodb_tsv_path=tsv_path,
        orthodb_group_means=orthodb_means,
        esm_embeddings=esm_embeddings,
    )
    print(f"  OrthoDB functional embeddings: {mask.sum().item()}/{mask.shape[0]} mapped")

    if cache_path:
        torch.save(group_embeds, cache_path)
        print(f"  Saved OrthoDB group embeddings: {group_embeds.shape} -> {cache_path}")
    return group_embeds
