"""proteomelm.essentiality_notebook: proteome parsing, known-label matching, figures on synthetic
results (Agg backend), ESM-C cache order. CPU-only, no network, no weights."""
import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

import proteomelm.essentiality as ess  # noqa: E402
import proteomelm.essentiality_notebook as nb  # noqa: E402


def _table(n=200, n_essential=30, called=True, seed=0):
    """Synthetic annotated result table; the first ``n_essential`` genes are essential and score higher."""
    rng = np.random.default_rng(seed)
    labels = np.zeros(n, dtype=int)
    labels[:n_essential] = 1
    p_e = np.clip(rng.normal(0.3 + 0.5 * labels, 0.2), 0, 1).astype(np.float32)
    ids = [f"sp|P{i:05d}|G{i}_ECOLI" for i in range(n)]
    table = ess.essentiality_table(ids, np.stack([p_e, 1 - p_e], 1), top_n=n_essential if called else None)
    proteome = nb.LoadedProteome(ids, ["M"] * n, {i: f"gen{k}" for k, i in enumerate(ids)},
                                 {i: "protein" for i in ids}, "Testus", "x.fasta")
    table = nb.annotate(table, proteome)
    by_id = dict(zip(ids, labels))
    return table, np.array([by_id[i] for i in table["protein_id"]])


def test_parse_header_and_read_fasta(tmp_path):
    assert nb.parse_header("sp|P0A8V2|RPOB_ECOLI DNA-directed RNA polymerase subunit beta OS=Escherichia coli "
                           "(strain K12) OX=83333 GN=rpoB PE=1 SV=1") == ("rpoB", "DNA-directed RNA polymerase subunit beta")
    assert nb.parse_header("lcl|x [gene=dnaA] [protein=initiator]")[0] == "dnaA"
    assert nb.parse_header("plain_id") == ("", "")
    fasta = tmp_path / "up.fasta"
    fasta.write_text(">sp|P1|A_ECO Protein A OS=Escherichia coli (strain K12) OX=83333 GN=aaa PE=1 SV=1\nMKV\n"
                     ">tr|Q2|Q2_ECO Uncharacterized OS=Escherichia coli (strain K12) OX=83333 PE=4 SV=1\nMA\n")
    proteome = nb.read_proteome_fasta(fasta)
    assert proteome.ids == ["sp|P1|A_ECO", "tr|Q2|Q2_ECO"] and proteome.sequences == ["MKV", "MA"]
    assert proteome.organism == "Escherichia coli (strain K12)" and proteome.n_proteins == 2
    assert proteome.gene == {"sp|P1|A_ECO": "aaa", "tr|Q2|Q2_ECO": ""}
    assert (proteome.ids, proteome.sequences) == ess.read_fasta(fasta)


def test_load_proteome_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        nb.load_proteome("FASTA file", str(tmp_path / "missing.fasta"))
    with pytest.raises(ValueError, match="Unknown proteome source"):
        nb.load_proteome("somewhere", "1")
    with pytest.raises(ValueError, match="numeric"):
        nb.find_uniprot_proteome("E. coli")
    with pytest.raises(ValueError, match="proteome id"):
        nb.download_uniprot_proteome("P12345", tmp_path)


def test_calling_rule():
    assert nb.calling_rule("Top fraction", 0.2, 10, 0.5) == {"top_fraction": 0.2}
    assert nb.calling_rule("Top N", 0.2, 10, 0.5) == {"top_n": 10}
    assert nb.calling_rule("Probability threshold", 0.2, 10, 0.7) == {"threshold": 0.7}
    with pytest.raises(ValueError):
        nb.calling_rule("best guess")


def test_match_proteins_and_known_labels(tmp_path):
    table, truth = _table()
    rows, missing = nb.match_proteins(table, "gen3, P00004, G5_ECOLI, sp|P00006|G6_ECOLI, nothere")
    assert sorted(table["protein_id"].iloc[rows]) == [f"sp|P{i:05d}|G{i}_ECOLI" for i in (3, 4, 5, 6)]
    assert missing == ["nothere"]
    essential = " ".join(f"gen{i}" for i in range(30))
    labels, missing = nb.known_labels(table, essential)
    np.testing.assert_array_equal(labels, truth)
    assert missing == []
    listfile = tmp_path / "ess.txt"
    listfile.write_text("\n".join(f"gen{i}" for i in range(30)) + "\nunknown\n")
    labels2, missing = nb.known_labels(table, str(listfile), "gen100 gen101")
    assert (labels2 == 1).sum() == 30 and (labels2 == 0).sum() == 2 and (labels2 == -1).sum() == 168
    assert missing == ["unknown"]
    assert nb.known_labels(table, "")[0] is None
    assert nb.known_labels(table, "nothere")[0] is None


def test_label_metrics():
    table, truth = _table()
    m = nb.label_metrics(table, truth)
    assert m["n_essential"] == 30 and m["n_nonessential"] == 170
    assert 0.8 < m["auroc"] <= 1 and m["aupr_baseline"] == pytest.approx(0.15)
    assert 0 < m["essential_in_top_n"] <= 30


def test_figures_render():
    table, truth = _table()
    for fig in (nb.plot_score_distribution(table), nb.plot_call_donut(table), nb.plot_call_donut(table, truth),
                nb.plot_label_curves(table, truth)):
        assert isinstance(fig, Figure)
    partial = truth.copy()
    partial[100:] = -1
    assert isinstance(nb.plot_call_donut(table, partial), Figure)
    assert isinstance(nb.plot_score_distribution(_table(called=False)[0]), Figure)


def test_figures_skip_cleanly(capsys):
    table, truth = _table(called=False)
    assert nb.plot_call_donut(table) is None
    assert nb.plot_label_curves(table, None) is None
    assert nb.plot_label_curves(table, np.ones(len(table), dtype=int)) is None
    assert "skipped" in capsys.readouterr().out


def test_cached_esmc_restores_input_order(tmp_path):
    class FakePredictor:
        config = ess.EssentialityHeadConfig()
        calls = 0

        def embed(self, sequences_sorted, ids_sorted=None):
            FakePredictor.calls += 1
            return torch.tensor([[float(len(s))] for s in sequences_sorted])

    proteome = nb.LoadedProteome(["a", "b", "c"], ["MK", "MKLLV", "M"], {}, {}, "x", "x.fasta")
    out = nb.cached_esmc(FakePredictor(), proteome, work_dir=tmp_path)
    assert out.flatten().tolist() == [2.0, 5.0, 1.0]
    again = nb.cached_esmc(FakePredictor(), proteome, work_dir=tmp_path)
    assert torch.equal(out, again) and FakePredictor.calls == 1


def test_save_table_and_stem(tmp_path):
    table, _ = _table(n=10, n_essential=2)
    assert nb.results_stem("Escherichia coli (strain K12)") == "essentiality_escherichia_coli_strain_k12"
    path = nb.save_table(table, "run", tmp_path)
    back = pd.read_csv(path, sep="\t")
    assert list(back.columns) == ["protein_id", "gene", "description", "p_essential", "rank", "predicted_essential"]


def test_build_form_edits_config():
    pytest.importorskip("ipywidgets")
    config = {"proteome_source": "FASTA file", "proteome": "x", "top_n": 3, "top_fraction": 0.1, "flag": True}
    form = nb.build_form(config, {"proteome_source": list(nb.SOURCES)})
    form.children[1].value = "y.fasta"
    form.children[2].value = 7
    assert config["proteome"] == "y.fasta" and config["top_n"] == 7
