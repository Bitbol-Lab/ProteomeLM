"""
Experiment management and execution for PPI analysis.
"""
import logging
import pandas as pd
from pathlib import Path
from typing import List, Union

from .config import ExperimentConfig, ExtractionConfig, DatasetConfig
from .data_processing import create_extractor
from .evaluation import PerformanceEvaluator
from .feature_extraction import PPIFeatureExtractor
from .model import train_model_cv, test_model_cv
import pickle


logger = logging.getLogger(__name__)


class BaseExperimentRunner:
    """Base class for running PPI experiments with common functionality."""

    def __init__(self):
        self.evaluator = PerformanceEvaluator()

    def _extract_features(self, extraction_config: ExtractionConfig, interaction_extractor) -> None:
        """Extract features for a given configuration."""
        feature_extractor = PPIFeatureExtractor(extraction_config, interaction_extractor)
        feature_extractor.extract_features()

    def _load_or_create_results_df(self, save_results_path: Path) -> pd.DataFrame:
        """Load existing results or create empty DataFrame."""
        if save_results_path.exists():
            df = pd.read_csv(save_results_path)
            logger.info(f"Loaded existing results with {len(df)} rows")
        else:
            df = pd.DataFrame()
            logger.info("Starting with empty results dataframe")
        return df

    def _save_results(self, df: pd.DataFrame, results: dict, experiment_config: ExperimentConfig,
                      checkpoint_number: int, save_results_path: Path) -> pd.DataFrame:
        """Save results to CSV."""
        results["model_name"] = experiment_config.model_name
        results["checkpoint"] = checkpoint_number + experiment_config.corrector

        df = pd.concat([df, pd.DataFrame([results])], ignore_index=True)
        df.to_csv(save_results_path, index=False)
        return df


class UnsupervisedExperimentRunner(BaseExperimentRunner):
    """Runs unsupervised learning experiments (attention-based analysis)."""

    def run_single_dataset_experiment(
        self,
        experiment_config: ExperimentConfig,
        dataset_config: DatasetConfig,
        save_results_path: Path
    ) -> None:
        """
        Run unsupervised experiments on a single dataset across multiple checkpoints.

        Args:
            experiment_config: Configuration for the experiment
            dataset_config: Configuration for the dataset
            save_results_path: Path to save results CSV
        """
        logger.info(f"Starting unsupervised experiment for {experiment_config.model_name} on {dataset_config.name}")

        df = self._load_or_create_results_df(save_results_path)

        # Create interaction extractor based on dataset type
        extractor_type = "bernett" if "goldstandard" in dataset_config.name else "dscript"
        interaction_extractor = create_extractor(extractor_type)

        for checkpoint_number in experiment_config.checkpoint_numbers:
            logger.info(f"Processing checkpoint {checkpoint_number}")

            # Configure extraction
            checkpoint_path = experiment_config.base_path / f"checkpoint-{checkpoint_number}"
            extraction_config = ExtractionConfig(
                checkpoint=str(checkpoint_path),
                env_dir=dataset_config.env_dir,
                fasta_file=dataset_config.fasta_file,
                encoded_genome_file=dataset_config.encoded_genome_file,
                save_path=dataset_config.save_path,
                reload_if_possible=experiment_config.reload_if_possible,
                include_all_hidden_states=False  # Not needed for unsupervised
            )

            # Extract features
            self._extract_features(extraction_config, interaction_extractor)

            # Evaluate results (unsupervised only)
            results = self.evaluator.evaluate_unsupervised_learning(
                dataset_config.save_path,
                include_supervised=False
            )

            # Log key metrics
            logger.info(f"Unsupervised results for checkpoint {checkpoint_number + experiment_config.corrector}:")
            logger.info(f"  Mean AUPR: {results['AUPR, Mean']:.3f}")
            logger.info(f"  Max AUPR: {results['AUPR, Max']:.3f}")
            logger.info(f"  Sum AUPR: {results['AUPR, Sum']:.3f}")

            # Save results
            df = self._save_results(df, results, experiment_config, checkpoint_number, save_results_path)

        logger.info(f"Unsupervised experiment completed. Results saved to {save_results_path}")

    def run_cross_species_experiment(
        self,
        experiment_config: ExperimentConfig,
        base_data_path: Path,
        species_list: List[str],
        save_results_path: Path
    ) -> None:
        """
        Run unsupervised cross-species experiments.

        Args:
            experiment_config: Configuration for the experiment
            base_data_path: Base path containing species data
            species_list: List of species to process
            save_results_path: Path to save results CSV
        """
        logger.info(f"Starting unsupervised cross-species experiment for {experiment_config.model_name}")

        df = self._load_or_create_results_df(save_results_path)
        interaction_extractor = create_extractor("dscript")

        for checkpoint_number in experiment_config.checkpoint_numbers:
            logger.info(f"Processing checkpoint {checkpoint_number}")

            checkpoint_path = experiment_config.base_path / f"checkpoint-{checkpoint_number}"

            # Extract features for each species
            species_results = {}

            for species in species_list:
                logger.info(f"  Processing species: {species}")

                species_dir = base_data_path / species
                extraction_config = ExtractionConfig(
                    checkpoint=str(checkpoint_path),
                    env_dir=species_dir,
                    fasta_file=f"{species}.faa",
                    encoded_genome_file=species_dir / "dump_dict_esm_dscript.pt",
                    save_path=species_dir / "dump_dict.pkl",
                    reload_if_possible=experiment_config.reload_if_possible,
                    include_all_hidden_states=False  # Not needed for unsupervised
                )

                self._extract_features(extraction_config, interaction_extractor)

                # Evaluate unsupervised performance for this species
                species_results_raw = self.evaluator.evaluate_unsupervised_learning(
                    species_dir / "dump_dict.pkl",
                    include_supervised=False
                )

                # Add species prefix to results
                for key, value in species_results_raw.items():
                    species_results[f"{key}, {species}"] = value

            # Log key metrics per species
            logger.info(f"Unsupervised cross-species results for checkpoint {checkpoint_number + experiment_config.corrector}:")
            for species in species_list:
                mean_aupr = species_results.get(f"AUPR, Mean, {species}", 0.0)
                max_aupr = species_results.get(f"AUPR, Max, {species}", 0.0)
                sum_aupr = species_results.get(f"AUPR, Sum, {species}", 0.0)
                logger.info(f"  {species}: Mean AUPR {mean_aupr:.3f}, Max AUPR {max_aupr:.3f}, Sum AUPR {sum_aupr:.3f}")

            # Save results
            df = self._save_results(df, species_results, experiment_config, checkpoint_number, save_results_path)

        logger.info(f"Unsupervised cross-species experiment completed. Results saved to {save_results_path}")


class SupervisedExperimentRunner(BaseExperimentRunner):
    """Runs supervised learning experiments (neural network training)."""

    def run_single_dataset_experiment(
        self,
        experiment_config: ExperimentConfig,
        dataset_config: DatasetConfig,
        save_results_path: Path,
        n_replicas: int = 5,
        save_models: bool = True,
        models_save_dir: Union[str, Path, None] = None
    ) -> None:
        """
        Run supervised experiments on a single dataset across multiple checkpoints.

        Args:
            experiment_config: Configuration for the experiment
            dataset_config: Configuration for the dataset
            save_results_path: Path to save results CSV
            n_replicas: Number of training replicas for each feature combination
            save_models: Whether to save trained models
            models_save_dir: Directory to save models (if None, uses dataset-specific directory)
        """
        logger.info(f"Starting supervised experiment for {experiment_config.model_name} on {dataset_config.name}")

        df = self._load_or_create_results_df(save_results_path)

        # Create interaction extractor based on dataset type
        extractor_type = "bernett" if "goldstandard" in dataset_config.name else "dscript"
        interaction_extractor = create_extractor(extractor_type)

        for checkpoint_number in experiment_config.checkpoint_numbers:
            logger.info(f"Processing checkpoint {checkpoint_number}")

            # Set up model save directory for this checkpoint if saving models
            checkpoint_models_dir = None
            if save_models:
                if models_save_dir is None:
                    checkpoint_models_dir = dataset_config.env_dir / "trained_models" / f"checkpoint_{checkpoint_number}"
                else:
                    checkpoint_models_dir = Path(models_save_dir) / f"checkpoint_{checkpoint_number}"

            # Configure extraction
            checkpoint_path = experiment_config.base_path / f"checkpoint-{checkpoint_number}"
            extraction_config = ExtractionConfig(
                checkpoint=str(checkpoint_path),
                env_dir=dataset_config.env_dir,
                fasta_file=dataset_config.fasta_file,
                encoded_genome_file=dataset_config.encoded_genome_file,
                save_path=dataset_config.save_path,
                reload_if_possible=experiment_config.reload_if_possible,
                include_all_hidden_states=True  # Needed for supervised
            )

            # Extract features
            self._extract_features(extraction_config, interaction_extractor)

            # Evaluate results (supervised only)
            results = self.evaluator.evaluate_unsupervised_learning(
                dataset_config.save_path,
                include_supervised=True,
                n_replicas=n_replicas,
                save_models=save_models,
                models_save_dir=checkpoint_models_dir
            )

            # Log key metrics (just the supervised ones)
            supervised_keys = [k for k in results.keys() if "Supervised" in k]
            if supervised_keys:
                logger.info(f"Supervised results for checkpoint {checkpoint_number + experiment_config.corrector}:")
                # Group by feature combination and show mean performance
                feature_combinations = set(k.split('_')[2] for k in supervised_keys if 'AUPR' in k)
                for combo in feature_combinations:
                    aupr_values = [results[k] for k in supervised_keys if f"AUPR, Supervised, {combo}" in k]
                    if aupr_values:
                        mean_aupr = sum(aupr_values) / len(aupr_values)
                        logger.info(f"  {combo} Mean AUPR: {mean_aupr:.3f}")

            # Save results
            df = self._save_results(df, results, experiment_config, checkpoint_number, save_results_path)

        logger.info(f"Supervised experiment completed. Results saved to {save_results_path}")

    def run_cross_species_experiment(
        self,
        experiment_config: ExperimentConfig,
        base_data_path: Path,
        species_list: List[str],
        save_results_path: Path,
        n_replicas: int = 5,
        save_models: bool = True,
        models_save_dir: Union[str, Path, None] = None
    ) -> None:
        """
        Run cross-species generalization experiments.

        Trains supervised models on human data and tests generalization on other species.

        Args:
            experiment_config: Configuration for the experiment
            base_data_path: Base path containing species data
            species_list: List of species to process (should include 'human')
            save_results_path: Path to save results CSV
            n_replicas: Number of training replicas for each feature combination
            save_models: Whether to save trained models
            models_save_dir: Directory to save models (if None, uses base_data_path)
        """

        logger.info(f"Starting cross-species supervised experiment for {experiment_config.model_name}")

        if "human" not in species_list:
            raise ValueError("Human data is required for cross-species training")

        df = self._load_or_create_results_df(save_results_path)
        interaction_extractor = create_extractor("dscript")

        for checkpoint_number in experiment_config.checkpoint_numbers:
            logger.info(f"Processing checkpoint {checkpoint_number}")

            checkpoint_path = experiment_config.base_path / f"checkpoint-{checkpoint_number}"

            # Set up model save directory for this checkpoint if saving models
            checkpoint_models_dir = None
            if save_models:
                if models_save_dir is None:
                    checkpoint_models_dir = base_data_path / "trained_models" / f"checkpoint_{checkpoint_number}"
                else:
                    checkpoint_models_dir = Path(models_save_dir) / f"checkpoint_{checkpoint_number}"

            # First, extract features for all species
            species_data = {}
            for species in species_list:
                logger.info(f"  Extracting features for species: {species}")

                species_dir = base_data_path / species
                extraction_config = ExtractionConfig(
                    checkpoint=str(checkpoint_path),
                    env_dir=species_dir,
                    fasta_file=f"{species}.faa",
                    encoded_genome_file=species_dir / "dump_dict_esm_dscript.pt",
                    save_path=species_dir / "dump_dict.pkl",
                    reload_if_possible=experiment_config.reload_if_possible,
                    include_all_hidden_states=True  # Needed for supervised
                )

                self._extract_features(extraction_config, interaction_extractor)

                # Load the extracted data
                with open(species_dir / "dump_dict.pkl", "rb") as f:
                    dump_dict = pickle.load(f)

                # Convert tensors to numpy arrays
                for k in dump_dict.keys():
                    dump_dict[k]["repr_proteomelm"] = dump_dict[k]["logits_proteomelm"].float().numpy()
                    dump_dict[k]["repr_esm"] = dump_dict[k]["repr_esm"].float().numpy()

                species_data[species] = dump_dict

            # Prepare human training data
            human_data = species_data["human"]
            d = human_data["train"]["repr_proteomelm"].shape[-1] // 2

            # Prepare feature combinations for human training/validation
            X_human = {}
            for k in ["train", "val"]:
                X_human[k] = self.evaluator._prepare_feature_combinations(human_data[k], d, self.evaluator.d_esm)

            # Add per-layer ProteomeLM features for human data
            """
            D = human_data["train"]["all_representations"].size(-1) // 2
            n_layers = len(human_data["train"]["all_representations"])
            for k in ["train", "val"]:
                reprs = human_data[k]["all_representations"].view(-1, n_layers + 1, 2 * D).float().numpy()
                for i in range(n_layers):
                    X_human[k][f"ProteomeLM-Layer{i + 1}"] = {
                        "edges": human_data[k]["A"],
                        "x1": reprs[:, i, :D],
                        "x2": reprs[:, i, D:]
                    }"""

            # Prepare human labels
            y_human = {k: human_data[k]["y"] for k in ["train", "val"]}

            # Prepare test data for all species
            X_test_species = {}
            y_test_species = {}

            for species in species_list:
                spec_data = species_data[species]
                X_test_species[species] = self.evaluator._prepare_feature_combinations(
                    spec_data["test"] if "test" in spec_data else spec_data["val"], d, self.evaluator.d_esm
                )

                # Add per-layer features for test data
                """test_reprs = spec_data["test"]["all_representations"].view(-1, n_layers + 1, 2 * D).float().numpy()
                for i in range(n_layers):
                    X_test_species[species][f"ProteomeLM-Layer{i + 1}"] = {
                        "edges": spec_data["test"]["A"],
                        "x1": test_reprs[:, i, :D],
                        "x2": test_reprs[:, i, D:]
                    }"""

                y_test_species[species] = spec_data["test"]["y"] if "test" in spec_data else spec_data["val"]["y"]

            # Train models on human data and test on all species
            results = {}

            for feature_combo in X_human["train"].keys():
                logger.info(f"Training cross-species model for {feature_combo}")

                for replica in range(n_replicas):
                    logger.debug(f"Training replica {replica} for {feature_combo}")

                    # Train model on human data
                    model, train_metrics = train_model_cv(
                        X_human["train"][feature_combo],
                        X_human["val"][feature_combo],
                        y_human["train"],
                        y_human["val"],
                        n_epochs=200,
                        patience=20,
                        verbose=False,
                        replica_seed=replica
                    )

                    # Test on all species
                    for species in species_list:
                        _, _, _, test_metrics = test_model_cv(
                            model, X_test_species[species][feature_combo], y_test_species[species]
                        )

                        # Store results with species information
                        results[f"AUC, CrossSpecies, {feature_combo}_{replica}, {species}"] = test_metrics["auc"]
                        results[f"AUPR, CrossSpecies, {feature_combo}_{replica}, {species}"] = test_metrics["aupr"]

                    # Save model if requested
                    if save_models and checkpoint_models_dir is not None:
                        checkpoint_models_dir.mkdir(parents=True, exist_ok=True)
                        model_name = f"cross_species_{feature_combo}_replica_{replica}"
                        model_path = checkpoint_models_dir / f"{model_name}.pt"

                        # Save model with cross-species metadata
                        import torch
                        model_data = {
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
                        }

                        torch.save(model_data, model_path)
                        logger.debug(f"Saved cross-species model: {model_path}")

            # Log summary results
            logger.info(f"Cross-species results for checkpoint {checkpoint_number + experiment_config.corrector}:")

            # Calculate mean performance per species for each feature combination
            for feature_combo in X_human["train"].keys():
                logger.info(f"  {feature_combo}:")
                for species in species_list:
                    aupr_values = [results[k] for k in results.keys() if f"AUPR, CrossSpecies, {feature_combo}" in k and species in k]
                    if aupr_values:
                        mean_aupr = sum(aupr_values) / len(aupr_values)
                        logger.info(f"    {species}: AUPR {mean_aupr:.3f}")

            # Save results
            df = self._save_results(df, results, experiment_config, checkpoint_number, save_results_path)

            # Save model registry if models were saved
            if save_models and checkpoint_models_dir is not None:
                trained_models = {}
                for model_file in checkpoint_models_dir.glob("cross_species_*.pt"):
                    model_name = model_file.stem
                    trained_models[model_name] = model_file

                if trained_models:
                    registry_path = checkpoint_models_dir / "cross_species_model_registry.pkl"
                    with open(registry_path, 'wb') as f:
                        pickle.dump(trained_models, f)
                    logger.info(f"Saved cross-species model registry to: {registry_path}")

        logger.info(f"Cross-species supervised experiment completed. Results saved to {save_results_path}")


class CombinedExperimentRunner(BaseExperimentRunner):
    """Runs both unsupervised and supervised experiments."""

    def __init__(self):
        super().__init__()
        self.unsupervised_runner = UnsupervisedExperimentRunner()
        self.supervised_runner = SupervisedExperimentRunner()

    def run_single_dataset_experiment(
        self,
        experiment_config: ExperimentConfig,
        dataset_config: DatasetConfig,
        save_results_path: Path,
        n_replicas: int = 5,
        save_models: bool = True,
        models_save_dir: Union[str, Path, None] = None
    ) -> None:
        """
        Run both unsupervised and supervised experiments on a single dataset.

        Args:
            experiment_config: Configuration for the experiment
            dataset_config: Configuration for the dataset
            save_results_path: Path to save results CSV
            n_replicas: Number of training replicas for supervised experiments
            save_models: Whether to save trained models
            models_save_dir: Directory to save models (if None, uses dataset-specific directory)
        """
        logger.info(f"Starting combined experiment for {experiment_config.model_name} on {dataset_config.name}")

        df = self._load_or_create_results_df(save_results_path)

        # Create interaction extractor based on dataset type
        extractor_type = "bernett" if "goldstandard" in dataset_config.name else "dscript"
        interaction_extractor = create_extractor(extractor_type)

        for checkpoint_number in experiment_config.checkpoint_numbers:
            logger.info(f"Processing checkpoint {checkpoint_number}")

            # Set up model save directory for this checkpoint if saving models
            checkpoint_models_dir = None
            if save_models:
                if models_save_dir is None:
                    checkpoint_models_dir = dataset_config.env_dir / "trained_models" / f"checkpoint_{checkpoint_number}"
                else:
                    checkpoint_models_dir = Path(models_save_dir) / f"checkpoint_{checkpoint_number}"

            # Configure extraction
            checkpoint_path = experiment_config.base_path / f"checkpoint-{checkpoint_number}"
            extraction_config = ExtractionConfig(
                checkpoint=str(checkpoint_path),
                env_dir=dataset_config.env_dir,
                fasta_file=dataset_config.fasta_file,
                encoded_genome_file=dataset_config.encoded_genome_file,
                save_path=dataset_config.save_path,
                reload_if_possible=experiment_config.reload_if_possible,
                include_all_hidden_states=True  # Needed for supervised
            )

            # Extract features
            self._extract_features(extraction_config, interaction_extractor)

            # Evaluate both unsupervised and supervised
            results = self.evaluator.evaluate_unsupervised_learning(
                dataset_config.save_path,
                include_supervised=True,
                n_replicas=n_replicas,
                save_models=save_models,
                models_save_dir=checkpoint_models_dir
            )

            # Log key metrics
            logger.info(f"Results for checkpoint {checkpoint_number + experiment_config.corrector}:")
            logger.info(f"  Unsupervised Mean AUPR: {results['AUPR, Mean']:.3f}")
            logger.info(f"  Unsupervised Max AUPR: {results['AUPR, Max']:.3f}")
            logger.info(f"  Unsupervised Sum AUPR: {results['AUPR, Sum']:.3f}")

            # Log supervised results
            supervised_keys = [k for k in results.keys() if "Supervised" in k and "AUPR" in k]
            if supervised_keys:
                feature_combinations = set(k.split('_')[2] for k in supervised_keys)
                for combo in feature_combinations:
                    aupr_values = [results[k] for k in supervised_keys if f"AUPR, Supervised, {combo}" in k]
                    if aupr_values:
                        mean_aupr = sum(aupr_values) / len(aupr_values)
                        logger.info(f"  Supervised {combo} Mean AUPR: {mean_aupr:.3f}")

            # Save results
            df = self._save_results(df, results, experiment_config, checkpoint_number, save_results_path)

        logger.info(f"Combined experiment completed. Results saved to {save_results_path}")


class BatchExperimentRunner:
    """Runs experiments across multiple models and configurations."""

    def __init__(self):
        self.unsupervised_runner = UnsupervisedExperimentRunner()
        self.supervised_runner = SupervisedExperimentRunner()
        self.combined_runner = CombinedExperimentRunner()

    def run_model_comparison(
        self,
        model_configs: List[ExperimentConfig],
        dataset_config: DatasetConfig,
        save_results_path: Path,
        experiment_type: str = "combined",
        n_replicas: int = 5,
        save_models: bool = True,
        models_save_dir: Union[str, Path, None] = None
    ) -> None:
        """
        Run experiments comparing multiple models on the same dataset.

        Args:
            model_configs: List of experiment configurations for different models
            dataset_config: Configuration for the dataset
            save_results_path: Path to save combined results
            experiment_type: Type of experiment ('unsupervised', 'supervised', or 'combined')
            n_replicas: Number of replicas for supervised experiments
            save_models: Whether to save trained models (only applies to supervised/combined)
            models_save_dir: Directory to save models (if None, uses dataset-specific directory)
        """
        logger.info(f"Starting {experiment_type} model comparison with {len(model_configs)} models")

        # Select appropriate runner
        if experiment_type == "unsupervised":
            runner = self.unsupervised_runner
        elif experiment_type == "supervised":
            runner = self.supervised_runner
        elif experiment_type == "combined":
            runner = self.combined_runner
        else:
            raise ValueError(f"Unknown experiment type: {experiment_type}")

        for model_config in model_configs:
            logger.info(f"Running {experiment_type} experiments for model: {model_config.model_name}")

            # Set up model save directory for this model if saving models
            model_models_dir = None
            if save_models and experiment_type in ["supervised", "combined"]:
                if models_save_dir is None:
                    model_models_dir = dataset_config.env_dir / "trained_models" / model_config.model_name
                else:
                    model_models_dir = Path(models_save_dir) / model_config.model_name

            if experiment_type in ["supervised", "combined"]:
                runner.run_single_dataset_experiment(
                    model_config, dataset_config, save_results_path, n_replicas, save_models, model_models_dir
                )
            else:
                runner.run_single_dataset_experiment(
                    model_config, dataset_config, save_results_path
                )

        logger.info(f"{experiment_type.capitalize()} model comparison completed")

    def run_species_comparison(
        self,
        experiment_configs: List[ExperimentConfig],
        base_data_path: Path,
        species_list: List[str],
        save_results_path: Path,
        experiment_type: str = "unsupervised",
        n_replicas: int = 5,
        save_models: bool = True,
        models_save_dir: Union[str, Path, None] = None
    ) -> None:
        """
        Run cross-species experiments for multiple models.

        Args:
            experiment_configs: List of experiment configurations
            base_data_path: Base path containing species data
            species_list: List of species to process
            save_results_path: Path to save results
            experiment_type: Type of experiment ('unsupervised' or 'supervised')
            n_replicas: Number of replicas for supervised experiments
            save_models: Whether to save trained models (only applies to supervised)
            models_save_dir: Directory to save models (if None, uses base_data_path)
        """
        logger.info(f"Starting {experiment_type} species comparison with {len(experiment_configs)} models")

        for experiment_config in experiment_configs:
            logger.info(f"Running {experiment_type} cross-species experiments for: {experiment_config.model_name}")

            if experiment_type == "supervised":
                # Set up model save directory if saving models
                cross_species_models_dir = None
                if save_models:
                    if models_save_dir is None:
                        cross_species_models_dir = base_data_path / "cross_species_models" / experiment_config.model_name
                    else:
                        cross_species_models_dir = Path(models_save_dir) / experiment_config.model_name

                self.supervised_runner.run_cross_species_experiment(
                    experiment_config, base_data_path, species_list, save_results_path,
                    n_replicas, save_models, cross_species_models_dir
                )
            else:
                logging.info("Running cross-species experiments is only supported for supervised experiments.")
        logger.info(f"{experiment_type.capitalize()} species comparison completed")
