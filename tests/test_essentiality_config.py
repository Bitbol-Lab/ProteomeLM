"""experiments.essentiality.common: config loading and path resolution."""
import os

import pytest

from experiments.essentiality.common import (embeds_file_prefix, load_config, plm_checkpoint_path,
                                             plm_size_from_checkpoint, resolve, splits_filename,
                                             stored_embeds_folder)


def test_bundled_config_is_the_published_classifier_setup(tmp_path):
    cfg = load_config(data_dir=tmp_path)
    c = cfg["classifier"]
    assert (c["optimizer"], c["learning_rate"], c["weight_decay"], c["dropout"]) == ("Adam", 0.001, 0, 0.5)
    assert (c["classifier_hidden_dim"], c["batch_size"], c["patience"], c["dtype"]) == (2048, 8, 30, "bfloat16")
    assert c["use_lr_scheduler"] is False and c["normalize_genome"] is False
    assert sorted(c["which_taxids_to_exclude"]) == sorted([580240, 83333, 199310, 679895, 941322, 1380365, 1124478])
    assert cfg["split"]["threshold"] == 40 and cfg["split"]["n_splits"] == 5


def test_paths_resolve_against_data_dir(tmp_path):
    cfg = load_config(data_dir=tmp_path)
    p = cfg["paths"]
    assert p["fasta_folder"] == os.path.join(str(tmp_path), "all_fasta_noduplicates2")
    assert p["label_folder"] == os.path.join(str(tmp_path), "label_to_ess7")
    assert p["splits_folder"] == str(tmp_path)
    assert all(os.path.isabs(v) for v in p.values())


def test_absolute_paths_are_kept(tmp_path, write_yaml):
    cfg_path = write_yaml("c.yaml", {"paths": {"a": "/abs/x", "b": "rel/y", "c": None},
                                     "classifier": {}, "data_download": {}, "split": {}})
    p = load_config(cfg_path, data_dir="/root")["paths"]
    assert p == {"a": "/abs/x", "b": "/root/rel/y", "c": None}
    assert resolve("/root", "~/z") == os.path.expanduser("~/z")


def test_missing_section_raises(write_yaml):
    with pytest.raises(KeyError, match="classifier"):
        load_config(write_yaml("c.yaml", {"paths": {}, "data_download": {}}), data_dir="/r")


def test_default_data_dir_follows_proteomelm_data_root(monkeypatch):
    import proteomelm.ppi.config as ppi_config
    monkeypatch.setattr(ppi_config, "DATA_ROOT", __import__("pathlib").Path("/somewhere"))
    assert load_config()["data_dir"] == "/somewhere/essentiality"


def test_splits_filename_never_overwrites_published():
    assert splits_filename("all_sequences2_labelled_splits", 40) == "all_sequences2_labelled_splits_40.pkl"
    assert splits_filename("all_sequences2_labelled_splits", 40, 0) == "all_sequences2_labelled_splits_40_seed0.pkl"


def test_plm_checkpoint_path():
    assert plm_checkpoint_path("S") == "Bitbol-Lab/ProteomeLM-S"
    assert plm_checkpoint_path("L", checkpoint_dir="/srv/common/proteomelm") == \
        "/srv/common/proteomelm/ProteomeLM-L/checkpoint-210"
    assert plm_checkpoint_path("M", "statistics", 43, baseline_dir="/d/ProteomeLM-baseline") == \
        "/d/ProteomeLM-baseline/ProteomeLM-M-statistics-seed43"
    with pytest.raises(ValueError):
        plm_checkpoint_path("M", "random")
    with pytest.raises(ValueError):
        plm_checkpoint_path("XL")


@pytest.mark.parametrize("name, size", [
    ("ProteomeLM-L/checkpoint-210", "L"),
    ("ProteomeLM-XS/checkpoint-210", "XS"),
    ("ProteomeLM-baseline/ProteomeLM-S-random-seed42", "S"),
    ("ProteomeLM-baseline/ProteomeLM-M-statistics-seed45", "M"),
    ("Bitbol-Lab/ProteomeLM-S", "S"),
    ("/srv/common/proteomelm/ProteomeLM-XS/checkpoint-210", "XS"),
    ("ProteomeLM-S-Cosine/checkpoint-210", None),
    (None, None),
])
def test_plm_size_from_checkpoint(name, size):
    assert plm_size_from_checkpoint(name) == size


def test_embedding_names_match_published_layout():
    assert stored_embeds_folder("/e", "L") == "/e/ProteomeLM-L-210"
    assert stored_embeds_folder("/e", "ESMC") == "/e/ESMC"
    assert embeds_file_prefix("trained") == "trained_"
    assert embeds_file_prefix("random", 42) == "random-seed42_"
