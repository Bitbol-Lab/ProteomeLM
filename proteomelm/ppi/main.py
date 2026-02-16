"""
Main entry point for running PPI experiments.
"""
import logging
from pathlib import Path

from .config import (
    DSCRIPT_SPECIES,
    BERNETT_CONFIG,
    ExperimentConfig,
    get_dscript_config,
    get_benchmark_config
)
from .experiment_runner import BatchExperimentRunner


# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)


def create_model_configs() -> dict:
    """Create experiment configurations for different models."""

    base_configs = {
        "ProteomeLM-XS": ExperimentConfig(
            model_name="ProteomeLM-XS",
            base_path=Path("/data1/malbrank/proteomelm/weights/ProteomeLM-XS"),
            checkpoint_numbers=[15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 195, 210]
        ),
        "ProteomeLM-S": ExperimentConfig(
            model_name="ProteomeLM-S",
            base_path=Path("/data1/malbrank/proteomelm/weights/ProteomeLM-S"),
            checkpoint_numbers=[210]
        ),
        "ProteomeLM-M": ExperimentConfig(
            model_name="ProteomeLM-M",
            base_path=Path("/data1/malbrank/proteomelm/weights/ProteomeLM-M"),
            checkpoint_numbers=[210]
        ),
        "ProteomeLM-L": ExperimentConfig(
            model_name="ProteomeLM-L",
            base_path=Path("/data1/malbrank/proteomelm/weights/ProteomeLM-L"),
            checkpoint_numbers=[15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 195, 210]
        ),
        "ProteomeLM-Mini-Cosine": ExperimentConfig(
            model_name="ProteomeLM-Mini-Cosine",
            base_path=Path("/data1/malbrank/proteomelm/weights/ProteomeLM-Mini-Kuma-Cosine"),
            checkpoint_numbers=[15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 195, 210]
        ),
        "ProteomeLM-Mini-MSE": ExperimentConfig(
            model_name="ProteomeLM-Mini-MSE",
            base_path=Path("/data1/malbrank/proteomelm/weights/ProteomeLM-Mini-Kuma-MSE"),
            checkpoint_numbers=[15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 195, 210]
        ),

        "ProteomeLM-Mini-Eukaryotes-Pretrain": ExperimentConfig(
            model_name="ProteomeLM-Mini-Eukaryotes-Pretrain",
            base_path=Path("/data1/malbrank/proteomelm/weights/ProteomeLM-Mini-Kuma-Eukaryotes-Pretrain"),
            checkpoint_numbers=[15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 195, 210, 225, 240, 255, 270, 285, 300, 315, 330, 345]
        ),
        "ProteomeLM-Mini-Eukaryotes-Scratch": ExperimentConfig(
            model_name="ProteomeLM-Mini-Eukaryotes-Scratch",
            base_path=Path("/data1/malbrank/proteomelm/weights/ProteomeLM-Mini-Kuma-Eukaryotes-Scratch"),
            checkpoint_numbers=[15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 195, 210, 225, 240, 255, 270, 285, 300, 315, 330, 345]
        ),
        "ProteomeLM-Mini-LongContext": ExperimentConfig(
            model_name="ProteomeLM-Mini-Kuma-LongContext",
            base_path=Path("/data1/malbrank/proteomelm/weights/ProteomeLM-Mini-Kuma-LongContext"),
            checkpoint_numbers=[15, 30, 45, 60, 75, 90, 105, 120]
        )
    }

    return base_configs


def run_bernett_experiments():
    """Run experiments on the Bernett dataset."""
    logger.info("Starting Bernett experiments")

    model_configs = create_model_configs()
    runner = BatchExperimentRunner()

    # Select which models to run (customize as needed)
    selected_models = ["ProteomeLM-S"]  # Add more models as needed

    configs_to_run = [model_configs[name] for name in selected_models if name in model_configs]

    # You can choose experiment type: "unsupervised", "supervised", or "combined"
    runner.run_model_comparison(
        model_configs=configs_to_run,
        dataset_config=BERNETT_CONFIG,
        save_results_path=BERNETT_CONFIG.results_path,
        experiment_type="combined",  # Run both unsupervised and supervised
        n_replicas=1
    )

    logger.info("Bernettexperiments completed")


def run_dscript_experiments():
    """Run experiments on DScript datasets."""
    logger.info("Starting DScript experiments")

    model_configs = create_model_configs()
    runner = BatchExperimentRunner()

    # Select models for DScript experiments
    selected_models = ["ProteomeLM-S"]  # Customize as needed

    configs_to_run = [model_configs[name] for name in selected_models if name in model_configs]

    base_data_path = Path("/data1/malbrank/proteomelm/dscript/")
    save_results_path = base_data_path / "checkpoint_screening.csv"

    # Cross-species experiments can now be either unsupervised or supervised
    # Unsupervised: Attention-based analysis across species
    # Supervised: Train on human, test generalization on other species

    experiment_type = "supervised"  # or "unsupervised"

    runner.run_species_comparison(
        experiment_configs=configs_to_run,
        base_data_path=base_data_path,
        species_list=DSCRIPT_SPECIES,
        save_results_path=save_results_path,
        experiment_type=experiment_type,
        n_replicas=1,  # Fewer replicas for cross-species
        save_models=True,
        models_save_dir=base_data_path / "cross_species_models"
    )

    logger.info("DScript experiments completed")


def run_single_species_experiments():
    """Run experiments on individual species datasets."""
    logger.info("Starting single species experiments")

    model_configs = create_model_configs()
    runner = BatchExperimentRunner()

    # Run experiments for each species individually
    for species in DSCRIPT_SPECIES:
        logger.info(f"Running experiments for species: {species}")

        species_config = get_dscript_config(species)
        selected_models = ["ProteomeLM-M"]  # Customize as needed

        configs_to_run = [model_configs[name] for name in selected_models if name in model_configs]

        # You can choose the experiment type per species
        runner.run_model_comparison(
            model_configs=configs_to_run,
            dataset_config=species_config,
            save_results_path=species_config.results_path,
            experiment_type="unsupervised",  # Or "supervised" or "combined"
            n_replicas=3  # Fewer replicas for individual species
        )

    logger.info("Single species experiments completed")


def run_benchmark_experiments():
    """
    Run experiments on benchmark datasets created with build_four_way_benchmark.py.
    
    Benchmark datasets use high-confidence IntAct interactions with sequence-similarity-based
    train/val/test splits (max 40% identity) in DScript-compatible format.
    """
    logger.info("Starting benchmark experiments")

    model_configs = create_model_configs()
    runner = BatchExperimentRunner()

    # Select which models to run (customize as needed)
    selected_models = ["ProteomeLM-S","ProteomeLM-M"]  # Add more models as needed

    configs_to_run = [model_configs[name] for name in selected_models if name in model_configs]

    # You can choose experiment type: "unsupervised", "supervised", or "combined"
    runner.run_model_comparison(
    model_configs=configs_to_run,
    dataset_config=get_dscript_config("ecoli"),
    save_results_path=get_dscript_config("ecoli").results_path,
    experiment_type="unsupervised",
    compare_orthodb=True,
    comparison_output_dir=get_dscript_config("ecoli").env_dir / "orthodb_compare"
    )   
    runner.run_model_comparison(
    model_configs=configs_to_run,
    dataset_config=get_dscript_config("yeast"),
    save_results_path=get_dscript_config("yeast").results_path,
    experiment_type="unsupervised",
    compare_orthodb=True,
    comparison_output_dir=get_dscript_config("yeast").env_dir / "orthodb_compare"
    )   
    runner.run_model_comparison(
    model_configs=configs_to_run,
    dataset_config=get_dscript_config("human"),
    save_results_path=get_dscript_config("human").results_path,
    experiment_type="unsupervised",
    compare_orthodb=True,
    comparison_output_dir=get_dscript_config("human").env_dir / "orthodb_compare"
    )   


    """runner.run_model_comparison(
        model_configs=configs_to_run,
        dataset_config=get_benchmark_config("yeast"),
        save_results_path=get_benchmark_config("yeast").results_path,
        experiment_type="supervised",  # Run both unsupervised and supervised
        n_replicas=5
    )
    runner.run_model_comparison(
        model_configs=configs_to_run,
        dataset_config=get_benchmark_config("human"),
        save_results_path=get_benchmark_config("human").results_path,
        experiment_type="supervised",  # Run both unsupervised and supervised
        n_replicas=5
    )

    runner.run_model_comparison(
        model_configs=configs_to_run,
        dataset_config=get_benchmark_config("ecoli"),
        save_results_path=get_benchmark_config("ecoli").results_path,
        experiment_type="supervised",  # Run both unsupervised and supervised
        n_replicas=5
    )"""



    logger.info("Benchmark experiments completed")




def main():
    """Main entry point for running experiments."""
    logger.info("Starting PPI experiments")

    # Uncomment the experiments you want to run
    # run_bernett_experiments()
    # run_dscript_experiments()
    # run_single_species_experiments()
    run_benchmark_experiments()

    logger.info("All experiments completed")


if __name__ == "__main__":
    main()
