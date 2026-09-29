"""Batch PPI benchmarks on the Bernett and D-SCRIPT datasets (see README.md).

For each ProteomeLM checkpoint: extract pair features over a dataset's proteome
(``PPIFeatureExtractor``), then

* ``--mode unsupervised``: score every attention head alone on the test split
  (``AttentionAnalyzer``: per-head, per-layer, summed and PCA AUC/AUPR);
* ``--mode supervised``: the same, plus ``EnhancedPPIModel`` heads trained on
  several feature combinations (``PerformanceEvaluator``);
* ``--dataset cross-species``: train supervised heads on D-SCRIPT human and test
  them on every species (``--mode supervised``), or score attention heads per
  species (``--mode unsupervised``).

Run from the repository root::

    python -m experiments.ppi_benchmarks.run_benchmarks --dataset bernett --mode supervised
"""
import argparse
import logging
import pickle
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd
import torch

from experiments.ppi_benchmarks.evaluation import PerformanceEvaluator
from proteomelm.ppi.config import (
    BERNETT_CONFIG,
    DATA_ROOT,
    DSCRIPT_SPECIES,
    DatasetConfig,
    ExperimentConfig,
    ExtractionConfig,
    get_benchmark_config,
    get_dscript_config,
)
from proteomelm.ppi.data_processing import create_extractor
from proteomelm.ppi.feature_extraction import PPIFeatureExtractor
from proteomelm.ppi.model import test_model_cv, train_model_cv
from proteomelm.utils.io import ensure_dir


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result tables
# ---------------------------------------------------------------------------

def _parse_results(results: dict, model_name: str, checkpoint):
    """Parse a flat results dict into two tidy DataFrames.

    Returns
    -------
    supervised_rows : list[dict]
        One row per (model, checkpoint, feature_combo, replica).
    unsupervised_rows : list[dict]
        One row per (model, checkpoint, layer, head) plus per-layer and summary rows.
    """
    base = {"model_name": model_name, "checkpoint": checkpoint}
    supervised_rows = []
    unsupervised_rows = []

    # --- supervised ---
    seen_combos: set = set()
    for key, val in results.items():
        if not key.startswith("AUPR, Supervised, "):
            continue
        tail = key[len("AUPR, Supervised, "):]
        parts = tail.rsplit("_", 1)
        if len(parts) != 2:
            continue
        combo, rep = parts[0], parts[1]
        auc_key = f"AUC, Supervised, {tail}"
        if (combo, rep) in seen_combos:
            continue
        seen_combos.add((combo, rep))
        supervised_rows.append({
            **base,
            "feature_combo": combo,
            "replica": int(rep),
            "auc": results.get(auc_key, float("nan")),
            "aupr": val,
        })

    # --- unsupervised: per-head ---
    for key, val in results.items():
        if not key.startswith("AUC, Layer ") or "Head" not in key:
            continue
        # "AUC, Layer L, Head H"
        parts = key.split(",")
        layer = int(parts[1].strip().split()[1])
        head = int(parts[2].strip().split()[1])
        aupr_key = f"AUPR, Layer {layer}, Head {head}"
        unsupervised_rows.append({
            **base,
            "layer": layer,
            "head": head,
            "auc": val,
            "aupr": results.get(aupr_key, float("nan")),
        })

    # summary rows
    for agg in ("Sum", "Mean", "Max"):
        auc_key = f"AUC, {agg}"
        aupr_key = f"AUPR, {agg}"
        if auc_key in results:
            unsupervised_rows.append({
                **base,
                "layer": agg.lower(),
                "head": "all",
                "auc": results[auc_key],
                "aupr": results.get(aupr_key, float("nan")),
            })

    return supervised_rows, unsupervised_rows


def _print_results_summary(model_name: str, checkpoint, sup_rows: list, unsup_rows: list) -> None:
    """Pretty-print a results summary to the logger."""
    lines = [f"\n{'='*62}",
             f"  {model_name}  checkpoint={checkpoint}",
             f"{'='*62}"]

    if sup_rows:
        lines.append("  Supervised (test) — sorted by AUPR")
        lines.append(f"  {'Feature combo':<30}  {'AUC':>6}  {'AUPR':>6}")
        lines.append(f"  {'-'*30}  {'------':>6}  {'------':>6}")
        for r in sorted(sup_rows, key=lambda x: -x["aupr"]):
            lines.append(f"  {r['feature_combo']:<30}  {r['auc']:6.4f}  {r['aupr']:6.4f}")

    per_head = [r for r in unsup_rows if isinstance(r["head"], int)]
    if per_head:
        lines.append("")
        lines.append("  Unsupervised — best head per layer")
        for layer in sorted({r["layer"] for r in per_head}):
            layer_rows = [r for r in per_head if r["layer"] == layer]
            best = max(layer_rows, key=lambda x: x["auc"])
            all_aucs = "  ".join(f"{r['auc']:.3f}" for r in
                                 sorted(layer_rows, key=lambda x: x["head"]))
            lines.append(f"  L{layer}  [{all_aucs}]  best=h{best['head']}  AUC={best['auc']:.4f}")

    for agg in ("max", "mean", "sum"):
        row = next((r for r in unsup_rows if r["layer"] == agg), None)
        if row:
            lines.append(f"  {agg.capitalize():<6}  AUC={row['auc']:.4f}  AUPR={row['aupr']:.4f}")

    lines.append("=" * 62)
    logger.info("\n".join(lines))


def _upsert_csv(path: Path, rows: list, model_name: str, checkpoint) -> None:
    """Append ``rows`` to the CSV at ``path``, replacing earlier rows of the same model and checkpoint."""
    df = pd.DataFrame(rows)
    if path.exists() and path.stat().st_size > 0:
        old = pd.read_csv(path)
        mask = (old["model_name"] == model_name) & (old["checkpoint"].astype(str) == str(checkpoint))
        df = pd.concat([old[~mask].reset_index(drop=True), df], ignore_index=True)
    df.to_csv(path, index=False)


def save_results(results: dict, model_name: str, checkpoint, results_dir: Path) -> None:
    """Write results to two tidy long-format CSVs in ``results_dir`` and print a summary.

    supervised_results.csv   — one row per (model, checkpoint, feature_combo, replica)
    unsupervised_results.csv — one row per (model, checkpoint, layer, head)
    """
    ensure_dir(str(results_dir))
    sup_rows, unsup_rows = _parse_results(results, model_name, checkpoint)
    if sup_rows:
        _upsert_csv(results_dir / "supervised_results.csv", sup_rows, model_name, checkpoint)
    if unsup_rows:
        _upsert_csv(results_dir / "unsupervised_results.csv", unsup_rows, model_name, checkpoint)
    _print_results_summary(model_name, checkpoint, sup_rows, unsup_rows)


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

def _checkpoints(experiment_config: ExperimentConfig):
    """``(label, path)`` per checkpoint: ``<base_path>/checkpoint-<n>``, or ``base_path`` itself
    (label ``"final"``) when no checkpoint numbers are given (e.g. a Hugging Face model id)."""
    if not experiment_config.checkpoint_numbers:
        return [("final", str(experiment_config.base_path))]
    return [(n, str(experiment_config.base_path / f"checkpoint-{n}")) for n in experiment_config.checkpoint_numbers]


def _extract(checkpoint: str, env_dir: Path, fasta_file: str, encoded_genome_file: Path, save_path: Path,
             experiment_config: ExperimentConfig, include_all_hidden_states: bool, interaction_extractor,
             dataset_config: Optional[DatasetConfig] = None) -> dict:
    """Run ``PPIFeatureExtractor`` (pickled to ``save_path``) with the dataset's OrthoDB settings, if any."""
    orthodb = {} if dataset_config is None else dict(
        orthodb_db_path=dataset_config.orthodb_db_path,
        orthodb_tsv_path=dataset_config.orthodb_tsv_path,
        orthodb_min_group_size=dataset_config.orthodb_min_group_size,
        orthodb_fetch_online=dataset_config.orthodb_fetch_online,
    )
    config = ExtractionConfig(
        checkpoint=checkpoint,
        env_dir=env_dir,
        fasta_file=fasta_file,
        encoded_genome_file=encoded_genome_file,
        save_path=save_path,
        reload_if_possible=experiment_config.reload_if_possible,
        include_all_hidden_states=include_all_hidden_states,  # per-layer features, supervised only
        **orthodb,
    )
    return PPIFeatureExtractor(config, interaction_extractor).extract_features()


def run_dataset(
    experiment_config: ExperimentConfig,
    dataset_config: DatasetConfig,
    results_dir: Path,
    supervised: bool = False,
    n_replicas: int = 5,
    save_models: bool = True,
    models_save_dir: Union[str, Path, None] = None,
) -> None:
    """Attention-head (and, if ``supervised``, supervised-head) benchmark of every checkpoint on one dataset.

    Trained models go to ``<models_save_dir or env_dir/trained_models>/checkpoint_<n>/``.
    """
    logger.info("Starting %s experiment for %s on %s",
                "supervised" if supervised else "unsupervised", experiment_config.model_name, dataset_config.name)
    extractor_type = "bernett" if "bernett" in dataset_config.name and "benchmark" not in dataset_config.name \
        else "dscript"
    interaction_extractor = create_extractor(extractor_type)
    evaluator = PerformanceEvaluator()

    for checkpoint, checkpoint_path in _checkpoints(experiment_config):
        logger.info("Processing checkpoint %s", checkpoint)
        checkpoint_models_dir = None
        if supervised and save_models:
            checkpoint_models_dir = Path(models_save_dir or dataset_config.env_dir / "trained_models") \
                / f"checkpoint_{checkpoint}"

        _extract(checkpoint_path, dataset_config.env_dir, dataset_config.fasta_file,
                 dataset_config.encoded_genome_file, dataset_config.save_path, experiment_config,
                 include_all_hidden_states=supervised, interaction_extractor=interaction_extractor,
                 dataset_config=dataset_config)
        results = evaluator.evaluate_unsupervised_learning(
            dataset_config.save_path,
            include_supervised=supervised,
            n_replicas=n_replicas,
            save_models=save_models,
            models_save_dir=checkpoint_models_dir,
        )
        save_results(results, experiment_config.model_name, checkpoint, results_dir)

    logger.info("Experiment completed. Results saved to %s", results_dir)


def _train_cross_species(evaluator: PerformanceEvaluator, species_data: dict, species_list: List[str],
                         n_replicas: int, models_dir: Optional[Path]) -> dict:
    """Train on D-SCRIPT human train/val, test on every species' test split (val if it has none)."""
    human_data = species_data["human"]
    d = human_data["train"]["repr_proteomelm"].shape[-1] // 2
    X_human = {k: evaluator._prepare_feature_combinations(human_data[k], d, evaluator.d_esm) for k in ["train", "val"]}
    y_human = {k: human_data[k]["y"] for k in ["train", "val"]}

    X_test_species, y_test_species = {}, {}
    for species in species_list:
        test_split = species_data[species]["test"] if "test" in species_data[species] else species_data[species]["val"]
        X_test_species[species] = evaluator._prepare_feature_combinations(test_split, d, evaluator.d_esm)
        y_test_species[species] = test_split["y"]

    results = {}
    for feature_combo in X_human["train"].keys():
        logger.info("Training cross-species model for %s", feature_combo)
        for replica in range(n_replicas):
            model, train_metrics = train_model_cv(
                X_human["train"][feature_combo], X_human["val"][feature_combo],
                y_human["train"], y_human["val"],
                n_epochs=200, patience=20, verbose=False, replica_seed=replica,
            )
            for species in species_list:
                _, _, _, test_metrics = test_model_cv(model, X_test_species[species][feature_combo],
                                                      y_test_species[species])
                results[f"AUC, CrossSpecies, {feature_combo}_{replica}, {species}"] = test_metrics["auc"]
                results[f"AUPR, CrossSpecies, {feature_combo}_{replica}, {species}"] = test_metrics["aupr"]

            if models_dir is not None:
                ensure_dir(str(models_dir))
                model_path = models_dir / f"cross_species_{feature_combo}_replica_{replica}.pt"
                torch.save({
                    'state_dict': model.state_dict(),
                    'feature_combination': feature_combo,
                    'replica': replica,
                    'train_metrics': train_metrics,
                    'training_species': 'human',
                    'experiment_type': 'cross_species',
                    'model_architecture': {
                        'protein_embed_dim': model.protein_embed_dim,
                        'pair_feature_dim': model.pair_feature_dim
                    },
                    'test_results_by_species': {
                        species: {
                            'auc': results[f"AUC, CrossSpecies, {feature_combo}_{replica}, {species}"],
                            'aupr': results[f"AUPR, CrossSpecies, {feature_combo}_{replica}, {species}"]
                        } for species in species_list
                    }
                }, model_path)
                logger.debug("Saved cross-species model: %s", model_path)

    for feature_combo in X_human["train"].keys():
        logger.info("  %s:", feature_combo)
        for species in species_list:
            aupr_values = [v for k, v in results.items()
                           if f"AUPR, CrossSpecies, {feature_combo}" in k and species in k]
            if aupr_values:
                logger.info("    %s: AUPR %.3f", species, sum(aupr_values) / len(aupr_values))

    if models_dir is not None:
        trained_models = {f.stem: f for f in models_dir.glob("cross_species_*.pt")}
        if trained_models:
            registry_path = models_dir / "cross_species_model_registry.pkl"
            with open(registry_path, 'wb') as f:
                pickle.dump(trained_models, f)
            logger.info("Saved cross-species model registry to: %s", registry_path)
    return results


def run_cross_species(
    experiment_config: ExperimentConfig,
    base_data_path: Path,
    species_list: List[str],
    results_dir: Path,
    supervised: bool = False,
    n_replicas: int = 5,
    save_models: bool = True,
    models_save_dir: Union[str, Path, None] = None,
) -> None:
    """D-SCRIPT species under ``base_data_path/<species>/``.

    Unsupervised: attention-head metrics per species. Supervised: heads trained
    on human (``species_list`` must include it) and tested on every species;
    models go to ``<models_save_dir or base_data_path/trained_models>/checkpoint_<n>/``.
    Cross-species metrics are only logged: ``save_results`` has no species column,
    so the CSVs do not keep them apart (see README.md).
    """
    logger.info("Starting %s cross-species experiment for %s",
                "supervised" if supervised else "unsupervised", experiment_config.model_name)
    if supervised and "human" not in species_list:
        raise ValueError("Human data is required for cross-species training")
    interaction_extractor = create_extractor("dscript")
    evaluator = PerformanceEvaluator()

    for checkpoint, checkpoint_path in _checkpoints(experiment_config):
        logger.info("Processing checkpoint %s", checkpoint)
        results, species_data = {}, {}
        for species in species_list:
            logger.info("  Processing species: %s", species)
            species_dir = base_data_path / species
            dump_dict = _extract(checkpoint_path, species_dir, f"{species}.faa", species_dir / "dump_dict_esm_dscript.pt",
                                 species_dir / "dump_dict.pkl", experiment_config,
                                 include_all_hidden_states=supervised, interaction_extractor=interaction_extractor)
            if supervised:
                for split in dump_dict.values():
                    split["repr_proteomelm"] = split["logits_proteomelm"].float().numpy()
                    split["repr_esm"] = split["repr_esm"].float().numpy()
                species_data[species] = dump_dict
            else:
                species_results = evaluator.evaluate_unsupervised_learning(species_dir / "dump_dict.pkl",
                                                                           include_supervised=False)
                results.update({f"{key}, {species}": value for key, value in species_results.items()})
                logger.info("  %s: Mean AUPR %.3f, Max AUPR %.3f, Sum AUPR %.3f", species,
                            species_results["AUPR, Mean"], species_results["AUPR, Max"], species_results["AUPR, Sum"])

        if supervised:
            models_dir = None
            if save_models:
                models_dir = Path(models_save_dir or base_data_path / "trained_models") / f"checkpoint_{checkpoint}"
            results = _train_cross_species(evaluator, species_data, species_list, n_replicas, models_dir)
        save_results(results, experiment_config.model_name, checkpoint, results_dir)

    logger.info("Cross-species experiment completed. Results saved to %s", results_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _dataset_config(name: str) -> DatasetConfig:
    if name == "bernett":
        return BERNETT_CONFIG
    kind, _, species = name.partition(":")
    if kind == "dscript" and species:
        return get_dscript_config(species)
    if kind == "benchmark" and species:
        return get_benchmark_config(species)
    raise ValueError(f"Unknown dataset '{name}': use bernett, dscript:<species>, benchmark:<species> or cross-species")


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="bernett",
                    help="bernett, dscript:<species>, benchmark:<species>, or cross-species (default: bernett)")
    ap.add_argument("--mode", choices=["unsupervised", "supervised"], default="unsupervised")
    ap.add_argument("--checkpoint", default="Bitbol-Lab/ProteomeLM-S",
                    help="Hugging Face model id or local checkpoint directory")
    ap.add_argument("--checkpoint-numbers", type=int, nargs="*", default=[],
                    help="evaluate <checkpoint>/checkpoint-<n> for each n (a training-run directory)")
    ap.add_argument("--model-name", help="name in the result tables (default: last part of --checkpoint)")
    ap.add_argument("--species", nargs="+", default=DSCRIPT_SPECIES, help="cross-species: D-SCRIPT species")
    ap.add_argument("--n-replicas", type=int, default=5, help="supervised: training replicas per feature combination")
    ap.add_argument("--no-save-models", action="store_true", help="supervised: do not save the trained heads")
    ap.add_argument("--results-dir", type=Path, help="default: the dataset directory under DATA_ROOT")
    args = ap.parse_args(argv)
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)

    experiment_config = ExperimentConfig(
        model_name=args.model_name or Path(args.checkpoint).name,
        base_path=Path(args.checkpoint),
        checkpoint_numbers=args.checkpoint_numbers,
    )
    supervised = args.mode == "supervised"
    save_models = not args.no_save_models
    if args.dataset == "cross-species":
        base_data_path = DATA_ROOT / "dscript"
        run_cross_species(
            experiment_config, base_data_path, args.species, args.results_dir or base_data_path,
            supervised=supervised, n_replicas=args.n_replicas, save_models=save_models,
            models_save_dir=base_data_path / "cross_species_models" / experiment_config.model_name,
        )
    else:
        dataset_config = _dataset_config(args.dataset)
        run_dataset(
            experiment_config, dataset_config, args.results_dir or dataset_config.env_dir,
            supervised=supervised, n_replicas=args.n_replicas, save_models=save_models,
            models_save_dir=dataset_config.env_dir / "trained_models" / experiment_config.model_name,
        )


if __name__ == "__main__":
    main()
