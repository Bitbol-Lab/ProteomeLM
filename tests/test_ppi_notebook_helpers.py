"""proteomelm.ppi.notebook_{runtime,inference}: the pure logic behind the PPI notebook.

CPU-only, no network, no pretrained weights: settings parsing/validation, protein
name resolution, memory planning, result tables, file locations, and attention
feature extraction on a tiny randomly initialised ProteomeLM.
"""
import gzip

import numpy as np
import pandas as pd
import pytest
import torch

import proteomelm.ppi.notebook_inference as nbi
import proteomelm.ppi.notebook_runtime as nbr
from proteomelm.modeling_proteomelm import ProteomeLMConfig, ProteomeLMForMaskedLM


# --------------------------------------------------------------------------- parsing

def test_parse_explicit_pairs_accepts_lines_semicolons_commas_tabs_and_spaces():
    text = "rpoB,rpoC; ftsZ ftsA\n# a comment\nP1\tP2\n\n"
    assert nbr.parse_explicit_pair_labels(text) == [("rpoB", "rpoC"), ("ftsZ", "ftsA"), ("P1", "P2")]


def test_parse_explicit_pairs_rejects_incomplete_pair():
    with pytest.raises(ValueError, match="two protein names"):
        nbr.parse_explicit_pair_labels("rpoB,rpoC; ftsZ")


def test_parse_protein_and_uniprot_lists():
    assert nbi.parse_protein_list("rpoB, ftsZ;dnaK\nmreB  ") == ["rpoB", "ftsZ", "dnaK", "mreB"]
    assert nbr.parse_uniprot_ids("P0A8V2,\nP0A8T7 Q9XYZ1") == ["P0A8V2", "P0A8T7", "Q9XYZ1"]


# --------------------------------------------------------------------------- query resolution

STRING_IDS = ["511145.b3987", "511145.b3988", "511145.b0095", "511145.b0094"]
STRING_ALIASES = {
    "511145.b3987": "rpoB | DNA-directed RNA polymerase subunit beta",
    "511145.b3988": "rpoC | DNA-directed RNA polymerase subunit beta'",
    "511145.b0095": "ftsZ | Cell division protein FtsZ",
    "511145.b0094": "ftsA | Cell division protein FtsA",
}


def test_resolve_queries_by_id_locus_and_gene_name_case_insensitive():
    queries = ["511145.b3987", "b3988", "FTSZ", "ftsA"]
    assert nbi.resolve_protein_queries(STRING_IDS, queries, aliases=STRING_ALIASES) == [0, 1, 2, 3]


def test_resolve_queries_uniprot_headers_by_accession_and_entry_name():
    labels = ["sp|P0A8V2|RPOB_ECOLI", "sp|P0A9A6|FTSZ_ECOLI"]
    assert nbi.resolve_protein_queries(labels, ["P0A9A6", "rpob_ecoli"]) == [1, 0]


def test_resolve_queries_unique_substring_of_description():
    assert nbi.resolve_protein_queries(STRING_IDS, ["subunit beta'"], aliases=STRING_ALIASES) == [1]


def test_resolve_queries_reports_all_problems_with_candidates_and_suggestions():
    with pytest.raises(ValueError) as excinfo:
        nbi.resolve_protein_queries(STRING_IDS, ["rpoX", "polymerase", "ftsZ"], aliases=STRING_ALIASES)
    message = str(excinfo.value)
    assert "'rpoX' was not found" in message and "Did you mean" in message
    assert "'polymerase' matches 2 proteins" in message and "511145.b3987" in message
    assert "ftsZ" not in message.split("Could not resolve")[1].split("\n  - ")[0]


def test_resolve_pair_queries():
    pairs = [("rpoB", "rpoC"), ("ftsZ", "511145.b0094")]
    assert nbi.resolve_pair_queries(STRING_IDS, pairs, aliases=STRING_ALIASES) == [(0, 1), (2, 3)]


# --------------------------------------------------------------------------- settings

def test_normalize_config_maps_form_labels_and_shorthands():
    run = nbr.normalize_notebook_config({
        "data_source_mode": "STRING organism",
        "string_id": " 511145 ",
        "query_mode": "Query proteins vs proteome",
        "query_proteins_text": "rpoB, ftsZ",
        "checkpoint": "ProteomeLM-M",
        "logreg_model": "Human",
        "top_k": "25",
    })
    assert run["data_source_mode"] == "string" and run["string_id"] == "511145"
    assert run["query_mode"] == "query_proteins" and run["query_proteins"] == ["rpoB", "ftsZ"]
    assert run["checkpoint"] == "Bitbol-Lab/ProteomeLM-M"
    assert run["logreg_model"] == "human" and run["supervised_model"] == "none"
    assert run["top_k"] == 25 and run["esm_device"] == "auto"


def test_normalize_config_reports_every_problem_at_once():
    with pytest.raises(ValueError) as excinfo:
        nbr.normalize_notebook_config({
            "data_source_mode": "STRING organism",
            "string_id": "E. coli",
            "query_mode": "query_proteins",
            "query_proteins_text": "",
            "supervised_model": "/does/not/exist.pt",
            "top_k": 0,
        })
    message = str(excinfo.value)
    for fragment in ("string_id must be a numeric", "query_proteins is empty", "supervised_model=", "top_k must be >= 1"):
        assert fragment in message


def test_normalize_config_empty_models_and_local_fasta_requirement():
    run = nbr.normalize_notebook_config({
        "data_source_mode": "local_path",
        "local_fasta_path": "",
        "query_mode": "all_pairs",
        "compare_with_string": "false",
        "supervised_model": "",
        "logreg_model": "",
    }, require_local_fasta=False)
    assert run["compare_with_string"] is False and run["supervised_model"] == "none" and run["logreg_model"] == "none"
    with pytest.raises(ValueError, match="local_fasta_path is empty"):
        nbr.normalize_notebook_config({"data_source_mode": "Local FASTA", "query_mode": "all_pairs"})


def test_normalize_config_rejects_unknown_choice():
    with pytest.raises(ValueError, match="data_source="):
        nbr.normalize_notebook_config({"data_source_mode": "PDB", "query_mode": "all_pairs"})


# --------------------------------------------------------------------------- memory planning

def test_estimate_attention_memory_scales_quadratically():
    small = nbi.estimate_notebook_resources(1000, "all_pairs", n_heads=8)
    large = nbi.estimate_notebook_resources(2000, "all_pairs", n_heads=8)
    assert large.attention_gb == pytest.approx(4 * small.attention_gb)
    assert small.candidate_pairs == 1000 * 999 // 2
    assert small.fits  # availability unknown -> no fit check


def test_plan_falls_back_to_cpu_when_gpu_too_small(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    # 20k proteins x 8 heads -> ~13 GB of attention: too big for 8 GB free on the GPU.
    device, estimate, notes = nbi.plan_notebook_run(
        n_proteins=20_000, mode="query_proteins", candidate_pairs=40_000, n_layers=6, n_heads=8,
        proteomelm_device="auto", available_gpu_gb=8.0, available_host_gb=64.0,
    )
    assert device == "cpu" and estimate.fits and notes
    # Same run on a 40 GB GPU stays on the GPU.
    device, estimate, notes = nbi.plan_notebook_run(
        n_proteins=20_000, mode="query_proteins", candidate_pairs=40_000, n_layers=6, n_heads=8,
        proteomelm_device="auto", available_gpu_gb=40.0, available_host_gb=64.0,
    )
    assert device == "cuda" and estimate.fits and not notes
    # Human-sized proteome on a 16 GB T4 (~14.7 GB free) stays on the GPU.
    device, estimate, _ = nbi.plan_notebook_run(
        n_proteins=19_699, mode="query_proteins", candidate_pairs=39_395, n_layers=6, n_heads=8,
        proteomelm_device="auto", available_gpu_gb=14.7, available_host_gb=12.0,
    )
    assert device == "cuda" and estimate.fits and 12.5 < estimate.device_peak_gb < 13.5


def test_plan_keeps_explicit_device_and_reports_problem(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    device, estimate, notes = nbi.plan_notebook_run(
        n_proteins=20_000, mode="query_proteins", candidate_pairs=40_000, n_layers=6, n_heads=8,
        proteomelm_device="cuda", available_gpu_gb=8.0, available_host_gb=64.0,
    )
    assert device == "cuda" and not estimate.fits and not notes
    assert "GPU memory" in estimate.errors[0] and "PROBLEM" in estimate.summary()


def test_plan_reports_host_memory_problem_for_huge_all_pairs():
    device, estimate, _ = nbi.plan_notebook_run(
        n_proteins=20_000, mode="all_pairs", candidate_pairs=20_000 * 19_999 // 2, n_layers=6, n_heads=8,
        proteomelm_device="cpu", available_host_gb=12.0,
    )
    assert device == "cpu" and not estimate.fits
    assert any("RAM" in err for err in estimate.errors) and "query_proteins" in estimate.recommendation


def test_resolve_device_auto_and_missing_gpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert nbi.resolve_device("auto") == "cpu" and nbi.resolve_device("") == "cpu"
    with pytest.raises(RuntimeError, match="no CUDA GPU"):
        nbi.resolve_device("cuda:0")


# --------------------------------------------------------------------------- results

def _results_frame():
    return pd.DataFrame({
        "protein_a": ["q", "a", "q", "b", "a"],
        "protein_b": ["a", "q2", "b", "q", "b"],
        "protein_a_label": ["Q", "A", "Q", "B", "A"],
        "protein_b_label": ["A", "Q2", "B", "Q", "B"],
        "idx_a": [0, 1, 0, 2, 1],
        "idx_b": [1, 3, 2, 0, 2],
        "unsupervised_score": [0.9, 0.8, 0.3, 0.7, 0.99],
    })


def test_summarize_top_partners_orders_by_score_and_names_partner():
    top = nbi.summarize_top_partners(_results_frame(), query_indices=[0, 3], top_k=2)
    assert list(top.columns) == ["query", "rank", "partner", "partner_id", "unsupervised_score", "query_id"]
    q = top[top["query_id"] == "q"]
    assert q["partner"].tolist() == ["A", "B"] and q["rank"].tolist() == [1, 2]
    assert q["unsupervised_score"].tolist() == [0.9, 0.7]  # (b, q) counts as a partner of q
    assert top[top["query_id"] == "q2"]["partner_id"].tolist() == ["a"]


def test_primary_score_prefers_supervised():
    frame = _results_frame()
    assert nbi.primary_score_column(frame) == "unsupervised_score"
    frame["supervised_score"] = 0.5
    assert nbi.primary_score_column(frame) == "supervised_score"


def test_results_basename_and_save(tmp_path):
    run = {"data_source_mode": "string", "string_id": "511145", "query_mode": "query_proteins"}
    assert nbr.results_basename(run, "Escherichia coli K12") == "ppi_escherichia_coli_k12_string511145_query"
    assert nbr.results_basename(run, "STRING taxon 511145") == "ppi_string511145_query"
    taxon = {"data_source_mode": "taxon", "taxon_id": "243273", "query_mode": "explicit_pairs"}
    assert nbr.results_basename(taxon, "Mycoplasmoides genitalium (strain ATCC 33530 / G37)") == \
        "ppi_mycoplasmoides_genitalium_taxon243273_pairs"
    local = {"data_source_mode": "local_path", "local_fasta_path": "/x/My Proteome.fasta", "query_mode": "all_pairs"}
    assert nbr.results_basename(local) == "ppi_my_proteome_allpairs"

    paths = nbr.save_notebook_results(_results_frame(), run, "E. coli", top_df=_results_frame().head(2), output_dir=tmp_path)
    assert set(paths) == {"all_pairs", "top", "pickle"} and all(p.exists() for p in paths.values())
    assert "idx_a" not in pd.read_csv(paths["all_pairs"]).columns

    # Large tables: ids only in the main CSV, names in a separate protein table.
    paths = nbr.save_notebook_results(_results_frame(), run, "E. coli", output_dir=tmp_path, max_rows_with_names=3)
    assert "protein_a_label" not in pd.read_csv(paths["all_pairs"]).columns
    names = pd.read_csv(paths["proteins"])
    assert dict(zip(names["protein"], names["name"]))["q2"] == "Q2" and names["protein"].is_unique


def test_string_mapping_is_inverted_onto_result_ids():
    # BLAST maps each query (UniProt) protein to a STRING id; STRING links must go the other way.
    query_to_string = {"P0A8V2": "511145.b3987", "sp|P0A9A6|FTSZ_ECOLI": "511145.b0095"}
    result_ids = ["sp|P0A8V2|RPOB_ECOLI", "sp|P0A9A6|FTSZ_ECOLI"]
    assert nbr.string_ids_to_result_ids(query_to_string, result_ids) == {
        "511145.b3987": "sp|P0A8V2|RPOB_ECOLI",
        "511145.b0095": "sp|P0A9A6|FTSZ_ECOLI",
    }


# --------------------------------------------------------------------------- files and paths

def test_cache_dir_env_override_and_setter(tmp_path, monkeypatch):
    monkeypatch.setenv("PROTEOMELM_NOTEBOOK_DIR", str(tmp_path / "env_dir"))
    nbr.set_cache_dir(None)
    assert nbr.get_cache_dir() == (tmp_path / "env_dir").resolve()
    try:
        assert nbr.set_cache_dir(tmp_path / "explicit") == (tmp_path / "explicit").resolve()
    finally:
        nbr.set_cache_dir(None)


def test_cache_dir_outside_a_clone_is_under_cwd(tmp_path, monkeypatch):
    monkeypatch.delenv("PROTEOMELM_NOTEBOOK_DIR", raising=False)
    monkeypatch.setattr(nbr, "find_repo_root", lambda: None)
    monkeypatch.chdir(tmp_path)
    assert nbr.default_cache_dir() == (tmp_path / "proteomelm_ppi").resolve()


def test_resolve_model_file_prefers_clone_then_download(tmp_path, monkeypatch):
    assert nbr.resolve_model_file("none", "logreg") is None
    repo = tmp_path / "repo"
    (repo / "data" / "interactomes").mkdir(parents=True)
    bundled = repo / "data" / "interactomes" / "logistic_regression_model_human.pkl"
    bundled.write_bytes(b"x")
    monkeypatch.setattr(nbr, "find_repo_root", lambda: repo)
    assert nbr.resolve_model_file("human", "logreg") == str(bundled)

    calls = []
    monkeypatch.setattr(nbr, "find_repo_root", lambda: None)
    monkeypatch.setattr(nbr, "_download_file", lambda url, dest, **kw: calls.append((url, dest)) or dest)
    nbr.set_cache_dir(tmp_path / "work")
    try:
        path = nbr.resolve_model_file("bernett", "supervised", "https://github.com/me/fork.git", "dev")
    finally:
        nbr.set_cache_dir(None)
    assert calls[0][0] == "https://raw.githubusercontent.com/me/fork/dev/data/interactomes/enhanced_ppi_model_bernett.pt"
    assert path.endswith("assets/dev/enhanced_ppi_model_bernett.pt")
    with pytest.raises(FileNotFoundError):
        nbr.resolve_model_file(str(tmp_path / "missing.pkl"), "logreg")


def test_load_fasta_reads_gzip(tmp_path):
    path = tmp_path / "p.fa.gz"
    with gzip.open(path, "wt") as handle:
        handle.write(">a desc\nMKV\nLL\n>b\nMA\n")
    assert nbr.load_fasta(path) == (["a desc", "b"], ["MKVLL", "MA"])


def test_stale_cache_check_compares_record_ids_not_full_headers(tmp_path):
    cache = tmp_path / "enc.pt"
    torch.save({"group_labels": ["sp|P1|A_ECOLI", "sp|P2|B_ECOLI"]}, cache)
    headers = ["sp|P1|A_ECOLI Protein A OS=E. coli", "sp|P2|B_ECOLI Protein B OS=E. coli"]
    assert nbr._invalidate_stale_encoded_cache(cache, headers) is False and cache.exists()
    assert nbr._invalidate_stale_encoded_cache(cache, headers[:1]) is True and not cache.exists()


# --------------------------------------------------------------------------- attention features

def _tiny_proteomelm(n_layers=2, n_heads=2):
    torch.manual_seed(0)
    config = ProteomeLMConfig(input_size=12, dim=8, hidden_dim=16, n_layers=n_layers, n_heads=n_heads,
                              vocab_size=4, max_position_embeddings=8)
    return ProteomeLMForMaskedLM(config).to(torch.bfloat16).eval()  # as load_proteomelm_backbone


def _tiny_proteome(n=7, dim=12):
    emb = torch.randn(n, dim)
    labels = [f"p{i}" for i in range(n)]
    return nbi.PreparedProteome(labels=labels, inputs_embeds=emb, group_embeds=emb, protein_embeddings=emb)


def test_attention_features_use_only_the_layers_the_forward_runs():
    # ProteomeLMForMaskedLM also holds an unused `distilbert` stack; its layers must not be hooked.
    model = _tiny_proteomelm(n_layers=2, n_heads=2)
    proteome = _tiny_proteome()
    pairs = np.array([[0, 1], [2, 5], [3, 6]])
    features = nbi.extract_attention_pair_features(model, proteome, pairs).numpy()
    assert features.shape == (3, 4)

    with torch.no_grad():
        out = model(inputs_embeds=proteome.inputs_embeds[None].bfloat16(),
                    group_embeds=proteome.group_embeds[None].bfloat16(), output_attentions=True)
    expected = np.stack([
        0.5 * (att[0, :, pairs[:, 0], pairs[:, 1]] + att[0, :, pairs[:, 1], pairs[:, 0]]).T.numpy()
        for att in (a.float() for a in out.attentions)
    ], axis=1).reshape(3, -1)
    np.testing.assert_allclose(features, expected, rtol=0, atol=1e-6)
    assert (np.abs(features) > 0).all()


def test_score_pair_chunks_builds_sorted_table():
    model = _tiny_proteomelm()
    proteome = _tiny_proteome()
    chunks = nbi.iter_pair_chunks(proteome.n_proteins, chunk_size=4, query_indices=[0])
    results = nbi.score_pair_chunks(model, proteome, chunks, device="cpu",
                                    display_labels={"p0": "Query zero"})
    assert len(results) == proteome.n_proteins - 1
    assert results["unsupervised_score"].is_monotonic_decreasing
    assert set(results.columns) == set(nbi.RESULT_COLUMNS)
    assert (results["protein_a_label"] == "Query zero").all()


# --------------------------------------------------------------------------- optional widget form

def _walk(widget):
    yield widget
    for child in getattr(widget, "children", ()):
        yield from _walk(child)


def test_config_form_edits_config_in_place_and_shows_relevant_fields():
    pytest.importorskip("ipywidgets")
    config = dict(nbr.DEFAULT_NOTEBOOK_CONFIG, data_source_mode="STRING organism", query_mode="All pairs")
    form = nbr.build_config_form(config)
    assert config["data_source_mode"] == "string" and config["query_mode"] == "all_pairs"
    by_description = {getattr(w, "description", None): w for w in _walk(form)}

    by_description["Protein source:"].value = "local_path"
    by_description["FASTA path:"].value = "  /data/my.fasta "
    by_description["Top k:"].value = 7
    assert config["data_source_mode"] == "local_path"
    assert config["local_fasta_path"] == "/data/my.fasta" and config["top_k"] == 7
    assert by_description["FASTA path:"].layout.display == ""
    assert by_description["STRING taxon id:"].layout.display == "none"
    run = nbr.normalize_notebook_config(config)
    assert run["data_source_mode"] == "local_path" and run["query_mode"] == "all_pairs"


def test_normalize_config_accepts_recommended_label_of_supervised_model():
    run = nbr.normalize_notebook_config({
        "data_source_mode": "STRING organism", "string_id": "511145", "query_mode": "all_pairs",
        "supervised_model": "multispecies (recommended)",
    })
    assert run["supervised_model"] == "multispecies"


def test_config_form_supervised_model_is_a_dropdown_with_recommended_label():
    widgets = pytest.importorskip("ipywidgets")
    config = dict(nbr.DEFAULT_NOTEBOOK_CONFIG, data_source_mode="STRING organism", query_mode="All pairs")
    menu = {getattr(w, "description", None): w for w in _walk(nbr.build_config_form(config))}["Supervised model:"]
    assert isinstance(menu, widgets.Dropdown)
    assert list(menu.options) == [("none", "none"), ("multispecies (recommended)", "multispecies"),
                                  ("dscript", "dscript"), ("bernett", "bernett")]
    menu.value = "multispecies"
    assert nbr.normalize_notebook_config(config)["supervised_model"] == "multispecies"
