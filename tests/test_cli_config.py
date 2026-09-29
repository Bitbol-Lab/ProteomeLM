"""proteomelm.cli.load_config / validate_config."""
import pytest

from proteomelm.cli import load_config, validate_config


def test_load_config_single_file(write_yaml):
    path = write_yaml("a.yaml", {"dim": 512, "batch_size": 16})
    config = load_config([path])
    assert config == {"dim": 512, "batch_size": 16}


def test_load_config_merges_in_order_later_file_wins(write_yaml):
    base = write_yaml("base.yaml", {"dim": 512, "batch_size": 16, "namedir": "base"})
    override = write_yaml("override.yaml", {"batch_size": 32})
    config = load_config([base, override])
    assert config["batch_size"] == 32  # override wins
    assert config["dim"] == 512        # untouched key survives from base
    assert config["namedir"] == "base"


def test_load_config_missing_file_raises(tmp_path):
    missing = str(tmp_path / "does_not_exist.yaml")
    with pytest.raises(FileNotFoundError):
        load_config([missing])


def test_validate_config_accepts_complete_config(base_pretraining_config):
    validate_config(base_pretraining_config)  # should not raise


@pytest.mark.parametrize("missing_key", [
    "batch_size", "learning_rate", "num_epochs", "output_dir", "namedir", "dim", "n_layers", "n_heads",
])
def test_validate_config_rejects_missing_required_key(base_pretraining_config, missing_key):
    del base_pretraining_config[missing_key]
    with pytest.raises(ValueError, match="Missing required configuration keys"):
        validate_config(base_pretraining_config)


@pytest.mark.parametrize("key,bad_value", [
    ("batch_size", 0),
    ("batch_size", -1),
    ("learning_rate", 0),
    ("learning_rate", -1e-4),
    ("num_epochs", 0),
    ("num_epochs", -5),
])
def test_validate_config_rejects_non_positive_values(base_pretraining_config, key, bad_value):
    base_pretraining_config[key] = bad_value
    with pytest.raises(ValueError):
        validate_config(base_pretraining_config)
