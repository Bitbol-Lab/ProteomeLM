"""Shared fixtures for the ProteomeLM test suite.

Scope is intentionally limited to pure-Python logic that needs no GPU, no
pretrained weights, and no real proteome data — see tests/README.md (and the
README's Testing section) for what's deliberately out of scope.
"""
import yaml
import torch
import pytest


@pytest.fixture
def write_yaml(tmp_path):
    """Write a dict to a YAML file under tmp_path and return its path as str."""
    def _write(name: str, content: dict) -> str:
        path = tmp_path / name
        with open(path, "w") as f:
            yaml.safe_dump(content, f)
        return str(path)
    return _write


@pytest.fixture
def base_pretraining_config() -> dict:
    """Minimal config satisfying proteomelm.cli.validate_config's required keys."""
    return {
        "batch_size": 16,
        "learning_rate": 3e-4,
        "num_epochs": 10,
        "output_dir": "output/",
        "namedir": "test-run",
        "dim": 512,
        "n_layers": 6,
        "n_heads": 8,
    }


def make_pair(host_len: int, pathogen_len: int, embed_dim: int = 8, maskable=None) -> dict:
    """Build a synthetic host-pathogen pair dict matching the on-disk shard schema
    documented in proteomelm/hpi/dataloaders.py (keys 'ie'/'ge'/'me'/'hl')."""
    n = host_len + pathogen_len
    if maskable is None:
        me = torch.ones(n, dtype=torch.bool)
    else:
        me = torch.tensor(maskable, dtype=torch.bool)
        assert me.shape[0] == n
    return {
        "ie": torch.randn(n, embed_dim, dtype=torch.float32),
        "ge": torch.randn(n, embed_dim, dtype=torch.float32),
        "me": me,
        "hl": host_len,
    }


@pytest.fixture
def make_pair_fixture():
    return make_pair
