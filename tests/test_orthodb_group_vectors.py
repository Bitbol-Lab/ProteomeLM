"""The three OrthoDB group-vector loaders share one file/threshold rule
(proteomelm.utils.proteome.orthodb_group_vector_files)."""
import pickle

import pytest
import torch

from proteomelm.alternate.modeling_naive import build_orthodb_vocab
from proteomelm.dataloaders import _load_orthodb_data_once
from proteomelm.utils.proteome import (
    ORTHODB_GROUP_SIZE_THRESHOLDS,
    load_orthodb_group_vectors,
    orthodb_group_vector_files,
)


@pytest.fixture
def db_path(tmp_path):
    """group_vectors_{0,10,50}.pkl present, group_vectors_200.pkl missing."""
    for t in (0, 10, 50):
        groups = {f"{t}_{i}at2": (torch.full((4,), float(t + i)), float(t + i)) for i in range(3)}
        with open(tmp_path / f"group_vectors_{t}.pkl", "wb") as f:
            pickle.dump(groups, f)
    return str(tmp_path)


def test_thresholds_match_published_files():
    assert ORTHODB_GROUP_SIZE_THRESHOLDS == (0, 10, 50, 200)


@pytest.mark.parametrize("min_size,expected", [(0, [0, 10, 50]), (10, [10, 50]), (11, [50]), (200, [])])
def test_file_selection_skips_small_thresholds_and_missing_files(db_path, min_size, expected):
    paths = orthodb_group_vector_files(db_path, min_size)
    assert [int(p.rsplit("_", 1)[-1].split(".")[0]) for p in paths] == expected


@pytest.mark.parametrize("min_size", [0, 10, 50])
def test_loaders_agree(db_path, min_size):
    means = load_orthodb_group_vectors(db_path, min_group_size=min_size)
    ids, dl_means = _load_orthodb_data_once(db_path, min_size)
    vocab = build_orthodb_vocab(db_path, min_taxid_size=min_size)

    assert set(means) == ids == set(dl_means) == set(vocab) - {"<UNK>"}
    assert all(torch.equal(means[k], dl_means[k]) for k in means)
    assert vocab["<UNK>"] == 0 and sorted(vocab.values()) == list(range(len(vocab)))


def test_dataloader_raises_when_nothing_loads(db_path):
    with pytest.raises(ValueError, match="No valid OrthoDB data"):
        _load_orthodb_data_once(db_path, 200)
