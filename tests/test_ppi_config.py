"""proteomelm.ppi.config: DatasetConfig path derivation and PROTEOMELM_DATA_ROOT."""
import importlib
from pathlib import Path

import proteomelm.ppi.config as ppi_config


def test_dataset_config_derived_paths():
    ds = ppi_config.DatasetConfig(
        name="demo",
        base_dir=Path("/tmp/demo_base"),
        fasta_file="demo.faa",
        experiment_name="demo_exp",
    )
    assert ds.env_dir == Path("/tmp/demo_base")
    assert ds.encoded_genome_file == Path("/tmp/demo_base/dump_dict_esm_demo_exp.pt")
    assert ds.save_path == Path("/tmp/demo_base/dump_dict.pkl")
    assert ds.results_path == Path("/tmp/demo_base/checkpoint_screening.csv")


def test_get_benchmark_config_and_get_dscript_config_use_data_root():
    benchmark = ppi_config.get_benchmark_config("human")
    dscript = ppi_config.get_dscript_config("yeast")
    assert benchmark.base_dir == ppi_config.DATA_ROOT / "benchmark" / "human"
    assert dscript.base_dir == ppi_config.DATA_ROOT / "dscript" / "yeast"
    assert benchmark.fasta_file == "human.faa"
    assert dscript.fasta_file == "yeast.faa"


def test_bernett_config_uses_data_root():
    assert ppi_config.BERNETT_CONFIG.base_dir == ppi_config.DATA_ROOT / "bernett"


def test_data_root_defaults_to_cluster_path_when_env_unset(monkeypatch):
    monkeypatch.delenv("PROTEOMELM_DATA_ROOT", raising=False)
    reloaded = importlib.reload(ppi_config)
    assert reloaded.DATA_ROOT == Path("data")
    importlib.reload(ppi_config)  # restore module state for subsequent tests


def test_data_root_respects_env_override(monkeypatch):
    monkeypatch.setenv("PROTEOMELM_DATA_ROOT", "/tmp/custom_data_root")
    reloaded = importlib.reload(ppi_config)
    try:
        assert reloaded.DATA_ROOT == Path("/tmp/custom_data_root")
        assert reloaded.get_benchmark_config("human").base_dir == Path("/tmp/custom_data_root/benchmark/human")
    finally:
        monkeypatch.delenv("PROTEOMELM_DATA_ROOT", raising=False)
        importlib.reload(ppi_config)  # restore module state for subsequent tests
