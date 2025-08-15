"""
Package initialization for PPI experiments.
"""
from .config import (
    DatasetConfig,
    ExperimentConfig,
    ExtractionConfig,
    BERNETT_CONFIG,
    DSCRIPT_SPECIES,
    get_dscript_config
)
from .data_processing import (
    FastaProcessor,
    InteractionExtractor,
    DScriptExtractor,
    BernettExtractor,
    create_extractor
)
from .feature_extraction import (
    PPIFeatureExtractor,
    FullProteomeExtractor
)
from .evaluation import (
    AttentionAnalyzer,
    PerformanceEvaluator
)
from .experiment_runner import (
    UnsupervisedExperimentRunner,
    SupervisedExperimentRunner,
    BatchExperimentRunner
)

__all__ = [
    # Config classes
    "DatasetConfig",
    "ExperimentConfig",
    "ExtractionConfig",
    "BERNETT_CONFIG",
    "DSCRIPT_SPECIES",
    "get_dscript_config",

    # Data processing
    "FastaProcessor",
    "InteractionExtractor",
    "DScriptExtractor",
    "BernettExtractor",
    "create_extractor",

    # Feature extraction
    "PPIFeatureExtractor",
    "FullProteomeExtractor",

    # Evaluation
    "AttentionAnalyzer",
    "PerformanceEvaluator",

    # Experiment runners
    "UnsupervisedExperimentRunner",
    "SupervisedExperimentRunner",
    "BatchExperimentRunner"
]
