"""proteomelm.essentiality (ProteomeLM-ess inference), experiments.essentiality.predict / package_head.

CPU-only, no pretrained weights, no network: tiny random heads and backbones.
"""
import json

import numpy as np
import pandas as pd
import pytest
import torch

import proteomelm.essentiality as ess
from experiments.essentiality.evaluate import top_n_essential
from experiments.essentiality.package_head import head_config_from_classifier, is_classifier_checkpoint
from experiments.essentiality.train import ClassifierConfig, TwoLayerClassifier
from proteomelm.modeling_proteomelm import ProteomeLMConfig, ProteomeLMForMaskedLM

ESM_DIM, DIM = 12, 8


def _tiny_config(**kw):
    values = dict(backbone="unused", layer=1, input_dim=DIM, hidden_dim=16, dropout=0.5)
    values.update(kw)
    return ess.EssentialityHeadConfig(**values)


def _state_dict(config, dtype=torch.bfloat16, seed=0):
    torch.manual_seed(seed)
    return {k: v.to(dtype) for k, v in ess.EssentialityHead(config).state_dict().items()}


def _tiny_backbone_dir(tmp_path):
    config = ProteomeLMConfig(input_size=ESM_DIM, dim=DIM, hidden_dim=16, n_layers=2, n_heads=2, vocab_size=4,
                              max_position_embeddings=8)
    torch.manual_seed(1)
    folder = tmp_path / "backbone"
    ProteomeLMForMaskedLM(config).save_pretrained(folder)
    return str(folder)


def test_config_round_trip(tmp_path):
    config = _tiny_config(layer=3, id2label={0: "essential", 1: "non-essential"})
    config.save(tmp_path)
    loaded = ess.EssentialityHeadConfig.load(tmp_path)
    assert loaded == ess.EssentialityHeadConfig.from_dict(json.loads((tmp_path / "config.json").read_text()))
    assert loaded.layer == 3 and loaded.id2label == {"0": "essential", "1": "non-essential"}
    assert loaded.essential_index == 0
    assert ess.EssentialityHeadConfig(id2label={"0": "non-essential", "1": "essential"}).essential_index == 1
    with pytest.warns(UserWarning, match="unknown"):
        ess.EssentialityHeadConfig.from_dict({**config.to_dict(), "extra_field": 1})


def test_save_and_load_head_from_folder(tmp_path):
    config = _tiny_config()
    state_dict = _state_dict(config)
    ess.save_head(tmp_path / "head", config, state_dict)
    assert sorted(p.name for p in (tmp_path / "head").iterdir()) == ["config.json", "model.safetensors"]
    loaded_config, head = ess.load_head(tmp_path / "head")
    assert loaded_config == config and not head.training
    for name, tensor in head.state_dict().items():
        assert tensor.dtype == torch.float32
        assert torch.equal(tensor, state_dict[name].float())  # bf16 -> fp32 is exact


def test_load_head_rejects_other_models(tmp_path):
    config = _tiny_config()
    ess.save_head(tmp_path, config, _state_dict(config))
    data = json.loads((tmp_path / "config.json").read_text())
    (tmp_path / "config.json").write_text(json.dumps({**data, "model_type": "something-else"}))
    with pytest.raises(ValueError, match="not an essentiality head"):
        ess.load_head(tmp_path)
    with pytest.raises(RuntimeError):  # wrong shapes fail before anything is written
        ess.save_head(tmp_path / "bad", _tiny_config(hidden_dim=4), _state_dict(config))


def test_head_matches_training_classifier():
    """Same parameter names and outputs as the training code's 2-layer classifier."""
    config = _tiny_config()
    train_config = ClassifierConfig(hidden_dim=DIM, classifier_hidden_dim=16, dropout=0.5, num_labels=2)
    reference = TwoLayerClassifier(train_config).eval()
    head = ess.EssentialityHead(config).eval()
    head.load_state_dict(reference.state_dict())
    x = torch.randn(7, DIM)
    probs = ess.head_probabilities(head, x)
    expected = torch.softmax(reference(x).float(), dim=-1).detach().numpy()
    np.testing.assert_array_equal(probs, expected)
    assert probs.shape == (7, 2) and np.allclose(probs.sum(1), 1)


def test_head_config_from_classifier():
    cfg = {"model_id": "2layer", "use_esmc_as_input": False, "which_weights": "trained", "use_layernorm": False,
           "normalize_genome": False, "proteomeLM_checkpoint": "ProteomeLM-L/checkpoint-210", "which_hidden_layer": 8,
           "hidden_dim": 1152, "classifier_hidden_dim": 2048, "dropout": 0.5, "num_labels": 2}
    config = head_config_from_classifier(cfg)
    assert (config.backbone, config.layer, config.input_dim, config.hidden_dim) == ("Bitbol-Lab/ProteomeLM-L", 8, 1152, 2048)
    assert config.id2label == {"0": "essential", "1": "non-essential"}
    for bad in ({"model_id": "simpleclassifier"}, {"use_esmc_as_input": True}, {"which_weights": "random"},
                {"normalize_genome": True}):
        with pytest.raises(ValueError, match="Cannot package"):
            head_config_from_classifier({**cfg, **bad})


def test_is_classifier_checkpoint(tmp_path):
    assert not is_classifier_checkpoint(tmp_path)
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "pytorch_model.bin").write_bytes(b"")
    assert is_classifier_checkpoint(tmp_path)


def test_ranks_and_top_n_match_the_paper_rule():
    rng = np.random.default_rng(0)
    p_ne = rng.random(50).astype(np.float32)
    probs = np.stack([1 - p_ne, p_ne], axis=1)
    for n in (0, 1, 7, 50):
        table = ess.essentiality_table([f"p{i}" for i in range(50)], probs, top_n=n)
        called = set(table.loc[table["predicted_essential"], "protein_id"])
        expected = {f"p{i}" for i in np.flatnonzero(top_n_essential(p_ne, n) == 0)}
        assert called == expected
    assert table["rank"].tolist() == list(range(1, 51))
    assert table["p_essential"].is_monotonic_decreasing


def test_threshold_rule_and_table_errors():
    probs = np.array([[0.9, 0.1], [0.2, 0.8], [0.5, 0.5]], dtype=np.float32)
    table = ess.essentiality_table(["a", "b", "c"], probs, threshold=0.5)
    assert table["protein_id"].tolist() == ["a", "c", "b"]
    assert table["predicted_essential"].tolist() == [True, True, False]
    assert "predicted_essential" not in ess.essentiality_table(["a", "b", "c"], probs)
    with pytest.raises(ValueError):
        ess.essentiality_table(["a", "b", "c"], probs, top_n=1, threshold=0.5)
    with pytest.raises(ValueError, match="shape"):
        ess.essentiality_table(["a"], np.ones(3))


def test_resolve_top_n():
    assert ess.resolve_top_n(100) is None
    assert ess.resolve_top_n(100, top_n=30) == 30
    assert ess.resolve_top_n(100, top_n=300) == 100
    assert ess.resolve_top_n(455, top_fraction=0.1) == 46
    for kwargs in ({"top_n": -1}, {"top_fraction": 1.5}, {"top_n": 1, "top_fraction": 0.1}):
        with pytest.raises(ValueError):
            ess.resolve_top_n(100, **kwargs)


def test_tsv_output_format(tmp_path):
    probs = np.array([[0.123456789, 0.876543211], [0.9999, 0.0001]], dtype=np.float32)
    path = ess.write_table(ess.essentiality_table(["x|1", "y"], probs, top_n=1), tmp_path / "sub" / "out.tsv")
    lines = path.read_text().splitlines()
    assert lines[0].split("\t") == list(ess.TSV_COLUMNS)
    assert lines[1].split("\t")[0] == "y" and lines[1].endswith("\t1\tTrue")
    back = pd.read_csv(path, sep="\t")
    np.testing.assert_array_equal(back["p_essential"].to_numpy(np.float32), probs[[1, 0], 0])  # 9 digits: exact


def test_check_proteome_and_sort_order():
    with pytest.raises(ValueError, match="duplicate"):
        ess.check_proteome(["a", "a"], ["M", "MK"])
    with pytest.raises(ValueError, match="empty"):
        ess.check_proteome(["a", "b"], ["M", ""])
    with pytest.raises(ValueError, match="empty"):
        ess.check_proteome([], [])
    assert ess.length_sorted_order(["MK", "MKLL", "MA", "M", "MKLL"]) == [1, 4, 0, 2, 3]  # stable for ties


def test_read_fasta(tmp_path):
    fasta = tmp_path / "p.fasta"
    fasta.write_text(">sp|P1|A_ECOLI desc GN=a\nMKV\nLL\n>b\nMA\n")
    assert ess.read_fasta(fasta) == (["sp|P1|A_ECOLI", "b"], ["MKVLL", "MA"])
    (tmp_path / "empty.fasta").write_text("")
    with pytest.raises(ValueError, match="No sequences"):
        ess.read_fasta(tmp_path / "empty.fasta")


def test_predictor_scores_in_length_order(tmp_path):
    """predict() embeds in decreasing-length order and returns every protein once, ranked."""
    config = _tiny_config(backbone=_tiny_backbone_dir(tmp_path), layer=1)
    head = ess.EssentialityHead(config).eval()
    predictor = ess.EssentialityPredictor(config, head, device="cpu")
    ids = ["short", "longest", "mid"]
    seqs = ["MA", "MKLLVVAA", "MKLL"]
    esmc = torch.randn(3, ESM_DIM).to(torch.bfloat16)
    table = predictor.predict(ids, seqs, esmc=esmc, top_n=1)
    assert sorted(table["protein_id"]) == sorted(ids) and table["predicted_essential"].sum() == 1
    order = [1, 2, 0]
    hidden = ess.backbone_hidden_state(predictor.backbone, esmc[order], 1, "cpu")
    expected = ess.head_probabilities(head, hidden)[:, 0]
    got = table.set_index("protein_id").loc[[ids[i] for i in order], "p_essential"].to_numpy()
    np.testing.assert_allclose(got, expected, rtol=0, atol=0)


def test_cli_end_to_end_with_tiny_models(tmp_path, monkeypatch):
    config = _tiny_config(backbone=_tiny_backbone_dir(tmp_path), layer=2)
    ess.save_head(tmp_path / "head", config, _state_dict(config))
    fasta = tmp_path / "proteome.fasta"
    fasta.write_text("".join(f">prot{i} some protein\n{'M' + 'K' * i}\n" for i in range(12)))
    monkeypatch.setattr(ess, "load_esmc", lambda *a, **k: object())
    monkeypatch.setattr(ess, "esmc_embeddings", lambda model, seqs, device, labels=None: torch.randn(len(seqs), ESM_DIM))
    from experiments.essentiality.predict import main
    out = tmp_path / "scores.tsv"
    table = main(["--fasta", str(fasta), "--head", str(tmp_path / "head"), "--out", str(out), "--top-fraction", "0.25",
                  "--gpu", "-1"])
    back = pd.read_csv(out, sep="\t")
    assert list(back.columns) == list(ess.TSV_COLUMNS) and len(back) == 12
    assert back["predicted_essential"].sum() == 3 and back["rank"].tolist() == list(range(1, 13))
    assert back["protein_id"].tolist() == table["protein_id"].tolist()


def test_estimate_memory_grows_with_proteome():
    small, large = ess.estimate_memory_gb(1000), ess.estimate_memory_gb(20000)
    assert large["proteomelm_gb"] > small["proteomelm_gb"] and small["esmc_gb"] == large["esmc_gb"]
    assert ess.estimate_memory_gb(20000, on_gpu=False)["proteomelm_gb"] > 5 * large["proteomelm_gb"]
