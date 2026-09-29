"""proteomelm.ppi.notebook_plots: every figure renders (Agg backend) on synthetic results,
skips cleanly when it does not apply, and stays fast on large tables. CPU-only, no network."""
import time

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

import proteomelm.ppi.notebook_inference as nbi  # noqa: E402
import proteomelm.ppi.notebook_plots as nbp  # noqa: E402
from proteomelm.modeling_proteomelm import ProteomeLMConfig, ProteomeLMForMaskedLM  # noqa: E402

from matplotlib.figure import Figure  # noqa: E402


def _results(n_proteins=60, queries=(0, 1), string=True, supervised=False, all_pairs=False, seed=0):
    """Synthetic notebook result table; STRING-supported pairs get higher scores."""
    rng = np.random.default_rng(seed)
    if all_pairs:
        a, b = np.triu_indices(n_proteins, k=1)
    else:
        pairs = sorted({tuple(sorted((q, p))) for q in queries for p in range(n_proteins) if p != q})
        a, b = np.array(pairs).T
    labels = [f"g{i} | Protein number {i} with a rather long descriptive name" for i in range(n_proteins)]
    in_string = rng.random(len(a)) < 0.15
    frame = pd.DataFrame({
        "protein_a": [f"p{i}" for i in a], "protein_b": [f"p{i}" for i in b],
        "protein_a_label": [labels[i] for i in a], "protein_b_label": [labels[i] for i in b],
        "idx_a": a, "idx_b": b,
        "unsupervised_score": 0.5 + 1e-4 * (rng.normal(size=len(a)) + 2.0 * in_string),
    })
    if supervised:
        frame["supervised_score"] = 1 / (1 + np.exp(-(rng.normal(size=len(a)) + in_string)))
    if string:
        frame["string_score"] = np.where(in_string, rng.integers(400, 1000, len(a)), rng.integers(0, 400, len(a)))
    return frame.sort_values("unsupervised_score", ascending=False, ignore_index=True)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    matplotlib.pyplot.close("all")


def test_top_partners_query_and_overall_modes():
    results = _results(queries=(0, 1, 2))
    fig = nbp.plot_top_partners(results, query_indices=[0, 1, 2], top_k=8)
    assert isinstance(fig, Figure) and len([ax for ax in fig.axes if ax.has_data()]) == 3
    assert isinstance(nbp.plot_top_partners(_results(string=False, all_pairs=True, n_proteins=20), top_k=5), Figure)


def test_string_figures_skip_without_string(capsys):
    results = _results(string=False)
    assert nbp.plot_string_agreement(results) is None
    assert nbp.plot_head_auroc(results, backbone=None, proteome=None) is None
    assert "Figure skipped" in capsys.readouterr().out


def test_string_agreement_and_score_comparison_render():
    results = _results(supervised=True)
    fig = nbp.plot_string_agreement(results)
    assert isinstance(fig, Figure)
    legend_text = " ".join(t.get_text() for ax in fig.axes if ax.get_legend() for t in ax.get_legend().get_texts())
    assert "Attention score (AUROC" in legend_text and "Supervised score (AUROC" in legend_text
    assert isinstance(nbp.plot_score_comparison(results), Figure)
    assert nbp.plot_score_comparison(_results(supervised=False)) is None


def test_tiny_and_single_class_tables_do_not_crash():
    tiny = _results(n_proteins=3, queries=(0,))
    for fig in (nbp.plot_top_partners(tiny, [0], top_k=20), nbp.plot_string_agreement(tiny),
                nbp.plot_partner_network(tiny, [0], top_k=20)):
        assert fig is None or isinstance(fig, Figure)
    one_class = _results(n_proteins=10, queries=(0,))
    one_class["string_score"] = 0
    assert isinstance(nbp.plot_string_agreement(one_class), Figure)  # ROC panel shows a note instead
    assert nbp.plot_top_partners(pd.DataFrame(columns=tiny.columns)) is None
    # A handful of pairs is shown on the raw score scale, not standardized against itself.
    fig = nbp.plot_top_partners(tiny.head(3), top_k=5)
    assert fig.axes[0].get_xlabel() == "Attention score"
    big = nbp.plot_top_partners(_results(n_proteins=80, queries=(0,)), [0], top_k=5)
    assert big.axes[0].get_xlabel().startswith("Score, SD above the mean")


def test_network_places_shared_partners_and_skips_outside_query_mode():
    results = _results(queries=(0, 1))
    assert isinstance(nbp.plot_partner_network(results, [0, 1], top_k=15), Figure)
    assert nbp.plot_partner_network(results, [], top_k=10) is None


def test_large_tables_stay_fast():
    rng = np.random.default_rng(0)
    n = 1_500_000
    large = pd.DataFrame({
        "protein_a": "a", "protein_b": "b", "protein_a_label": "x | y", "protein_b_label": "x | z",
        "idx_a": rng.integers(0, 3000, n), "idx_b": rng.integers(0, 3000, n),
        "unsupervised_score": rng.random(n), "supervised_score": rng.random(n),
        "string_score": rng.integers(0, 1000, n),
    })
    start = time.time()
    for fig in (nbp.plot_top_partners(large, top_k=10), nbp.plot_string_agreement(large, max_curve_pairs=200_000),
                nbp.plot_score_comparison(large)):
        assert isinstance(fig, Figure)
    assert time.time() - start < 30


def test_head_auroc_matrix_and_heatmap_on_tiny_model():
    features = np.zeros((40, 6))
    labels = np.array([0, 1] * 20)
    features[:, 4] = labels  # layer 2, head 2 is perfect
    features[:, 1] = -labels  # layer 1, head 2 is inverted
    aucs = nbp.head_auroc_matrix(features, labels, n_layers=2)
    assert aucs.shape == (2, 3) and aucs[1, 1] == 1.0 and aucs[0, 1] == 0.0

    torch.manual_seed(0)
    config = ProteomeLMConfig(input_size=12, dim=8, hidden_dim=16, n_layers=2, n_heads=2, vocab_size=4,
                              max_position_embeddings=8)
    model = ProteomeLMForMaskedLM(config).to(torch.bfloat16).eval()
    emb = torch.randn(30, 12)
    proteome = nbi.PreparedProteome(labels=[f"p{i}" for i in range(30)], inputs_embeds=emb, group_embeds=emb,
                                    protein_embeddings=emb)
    results = _results(n_proteins=30, all_pairs=True)
    fig = nbp.plot_head_auroc(results, model, proteome, device="cpu", min_per_class=5)
    assert isinstance(fig, Figure)


def test_show_figure_saves_png_and_svg_and_zip(tmp_path):
    fig = nbp.plot_top_partners(_results(), [0], top_k=5)
    paths = nbp.show_figure(fig, "top", tmp_path / "run")
    assert [p.name for p in paths] == ["run_top.png", "run_top.svg"] and all(p.stat().st_size > 0 for p in paths)
    assert nbp.show_figure(None, "skipped", tmp_path / "run") == []
    archive = nbp.zip_figures(paths, tmp_path / "figures.zip")
    import zipfile
    assert sorted(zipfile.ZipFile(archive).namelist()) == ["run_top.png", "run_top.svg"]
