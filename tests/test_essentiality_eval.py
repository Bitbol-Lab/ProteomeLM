"""experiments.essentiality.evaluate: top-N calls, donut fractions and checkpoint matching."""
import json
import math
import os

import numpy as np
import pytest

from experiments.essentiality.evaluate import (PlotArguments, auroc_on_labelled, checkpoint_matches, get_filename,
                                               get_relevant_checkpoints, get_sorted_fraction, group_label,
                                               read_classifier_config, sort_items_by_label, top_n_essential)


def test_top_n_marks_lowest_p_nonessential_as_essential():
    p_ne = np.array([0.9, 0.1, 0.5, 0.05, 0.7])
    calls = top_n_essential(p_ne, 2)
    assert calls.tolist() == [1, 0, 1, 0, 1]  # 0 = E, 1 = NE
    assert top_n_essential(p_ne, 0).tolist() == [1] * 5
    assert top_n_essential(p_ne, 5).tolist() == [0] * 5


def test_group_label():
    assert [group_label(x) for x in ["E", "NE", "QE", "E?", "No_label", "UNK"]] == \
        ["E", "NE", "QE", "Other", "Other", "Other"]


def test_sorted_fraction_counts_predicted_e_per_true_class():
    true = {"a": "E", "b": "E", "c": "NE", "d": "NE", "e": "NE", "f": "QE", "g": "No_label"}
    pred = {"a": "E", "b": "NE", "c": "E", "d": "NE", "e": "NE", "f": "E", "g": "E"}
    sorted_true, sorted_pred = sort_items_by_label(true, pred)
    assert [k for k, _ in sorted_true] == [k for k, _ in sorted_pred]
    assert [group_label(v) for _, v in sorted_true] == ["E", "E", "NE", "NE", "NE", "Other", "QE"]
    stats = get_sorted_fraction(sorted_true, sorted_pred)
    assert stats == {"E in E": 0.5, "E in NE": pytest.approx(1 / 3), "E in QE": 1.0}


def test_sorted_fraction_without_qe_is_nan():
    true, pred = {"a": "E", "b": "NE"}, {"a": "E", "b": "E"}
    stats = get_sorted_fraction(*sort_items_by_label(true, pred))
    assert stats["E in E"] == 1.0 and stats["E in NE"] == 1.0 and math.isnan(stats["E in QE"])


def test_auroc_on_labelled_ignores_other_labels():
    labels = {"a": "E", "b": "E", "c": "NE", "d": "QE"}
    scores = {"a": 0.1, "b": 0.2, "c": 0.9, "d": 0.0}
    assert auroc_on_labelled(labels, scores) == 1.0


def test_get_filename_matches_published_scheme():
    args = PlotArguments(plots_data_directory="/p", checkpoint="L", modeltype="2layer", random_seed=42,
                         splits_thr=40, holdout=False, which_weights="trained")
    assert get_filename(args, "otherscores.pkl") == "/p/ProteomeLM-L/2layer-trained-seed42-cluster40-otherscores.pkl"
    esmc = PlotArguments(plots_data_directory="/p", checkpoint="ESMC", modeltype="simpleclassifier", random_seed=43,
                         splits_thr=40, which_weights="trained")
    assert get_filename(esmc, "x.pkl") == "/p/ESMC-trained-seed43-cluster40-x.pkl"


def _published_cfg(**kw):
    cfg = {"use_esmc_as_input": False, "proteomeLM_checkpoint": "ProteomeLM-L/checkpoint-210",
           "which_weights": "trained", "random_seed": 42, "model_id": "2layer", "which_hidden_layer": 3,
           "splits_info_file": "/old/server/all_sequences2_labelled_splits_40.pkl", "n_layers": 18,
           "wandb_project_name": "P"}
    cfg.update(kw)
    return cfg


def test_checkpoint_matches_published_and_new_configs():
    legacy_split = "/elsewhere/all_sequences2_labelled_splits_40.pkl"
    assert checkpoint_matches(_published_cfg(), "L", "trained", 42, legacy_split, "P")
    assert not checkpoint_matches(_published_cfg(), "M", "trained", 42, legacy_split)
    assert not checkpoint_matches(_published_cfg(), "L", "trained", 43, legacy_split)
    assert not checkpoint_matches(_published_cfg(), "L", "trained", 42, "/x/all_sequences2_labelled_splits_40_seed0.pkl")
    assert not checkpoint_matches(_published_cfg(), "L", "trained", 42, legacy_split, "other-project")
    new = _published_cfg(proteomeLM_checkpoint="Bitbol-Lab/ProteomeLM-L",
                         splits_info_file="/d/all_sequences2_labelled_splits_40_seed0.pkl")
    assert checkpoint_matches(new, "L", "trained", 42, "/d/all_sequences2_labelled_splits_40_seed0.pkl")
    baseline = _published_cfg(proteomeLM_checkpoint="ProteomeLM-baseline/ProteomeLM-S-random-seed44",
                              which_weights="random", random_seed=44)
    assert checkpoint_matches(baseline, "S", "random", 44, legacy_split)
    esmc = _published_cfg(use_esmc_as_input=True, proteomeLM_checkpoint=None)
    assert checkpoint_matches(esmc, "ESMC", "trained", 42, legacy_split)
    assert not checkpoint_matches(esmc, "L", "trained", 42, legacy_split)


def _write_ckpt(root, name, cfg):
    d = os.path.join(root, name)
    os.makedirs(d)
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump(cfg, f)
    open(os.path.join(d, "pytorch_model.bin"), "wb").close()
    return d


def test_get_relevant_checkpoints_one_per_layer(tmp_path):
    root = str(tmp_path)
    split = "all_sequences2_labelled_splits_40.pkl"
    a = _write_ckpt(root, "250801-2layer-aaaa", _published_cfg(which_hidden_layer=0))
    b = _write_ckpt(root, "250801-2layer-bbbb", _published_cfg(which_hidden_layer=1))
    old = _write_ckpt(root, "250801-2layer-cccc", _published_cfg(which_hidden_layer=1))
    os.utime(os.path.join(old, "pytorch_model.bin"), (0, 0))  # duplicate layer 1, older
    _write_ckpt(root, "250801-simpleclassifier-dddd", _published_cfg(which_hidden_layer=2, model_id="simpleclassifier"))
    _write_ckpt(root, "250801-2layer-eeee", _published_cfg(which_hidden_layer=2, random_seed=43))
    found = get_relevant_checkpoints(root, "2layer", "L", "trained", 42, split)
    assert found == [a, b]
    # Metric pickles keyed by another machine's path resolve through classifier_dir
    assert read_classifier_config("/old/server/checkpoints/x/250801-2layer-aaaa", root)["which_hidden_layer"] == 0
