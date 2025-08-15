"""
Evaluation and analysis utilities for PPI experiments.
"""
import logging
import os
import pickle

from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, roc_auc_score
from proteomelm.ppi.model import train_model_cv, test_model_cv


logger = logging.getLogger(__name__)


class AttentionAnalyzer:
    """Analyzes attention patterns for PPI prediction performance."""

    @staticmethod
    def analyze_attention_performance(attention_matrix: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        """
        Analyze the predictive performance of attention patterns.

        Args:
            attention_matrix: Attention matrix with shape (num_pairs, num_layers, num_heads)
            labels: Binary interaction labels

        Returns:
            Dictionary containing various performance metrics
        """
        if attention_matrix is None:
            raise ValueError("Attention matrix is not available.")

        A = attention_matrix.numpy()
        y = labels.numpy()

        _, n_layers, n_heads = A.shape

        # Compute different aggregations of attention
        per_layer_sums = A.sum(axis=-1).reshape(-1, n_layers)
        total_sum = per_layer_sums.sum(axis=-1).reshape(-1)
        matrix = A.reshape(-1, n_layers * n_heads)

        # PCA analysis
        pca_A = PCA(n_components=10).fit_transform(matrix)

        results = {}

        # Individual head/layer performance
        aucs = [roc_auc_score(y, matrix[:, i]) for i in range(matrix.shape[1])]
        auprs = [average_precision_score(y, matrix[:, i]) for i in range(matrix.shape[1])]

        # PCA component performance
        pca_aucs = [roc_auc_score(y, pca_A[:, i]) for i in range(pca_A.shape[1])]
        pca_auprs = [average_precision_score(y, pca_A[:, i]) for i in range(pca_A.shape[1])]

        # Fix PCA components if they're inverted
        for i, pca_auc in enumerate(pca_aucs):
            if pca_auc < 0.5:
                pca_aucs[i] = 1 - pca_auc
                pca_A[:, i] = -pca_A[:, i]
                pca_auprs[i] = average_precision_score(y, pca_A[:, i])

        # Per-layer aggregated performance
        per_layer_aucs = [roc_auc_score(y, per_layer_sums[:, i]) for i in range(per_layer_sums.shape[1])]
        per_layer_auprs = [average_precision_score(y, per_layer_sums[:, i]) for i in range(per_layer_sums.shape[1])]

        # Overall aggregated performance
        sum_auc = roc_auc_score(y, total_sum)
        sum_aupr = average_precision_score(y, total_sum)

        # Summary statistics
        mean_auc = np.mean(aucs)
        max_auc = np.max(aucs)
        mean_aupr = np.mean(auprs)
        max_aupr = np.max(auprs)

        # Store detailed results
        for i, (auc, aupr) in enumerate(zip(aucs, auprs)):
            layer_idx = i // n_heads
            head_idx = i % n_heads
            results[f"AUC, Layer {layer_idx}, Head {head_idx}"] = auc
            results[f"AUPR, Layer {layer_idx}, Head {head_idx}"] = aupr

        for i, (auc, aupr) in enumerate(zip(pca_aucs, pca_auprs)):
            results[f"AUC, PCA {i}"] = auc
            results[f"AUPR, PCA {i}"] = aupr

        for i, (auc, aupr) in enumerate(zip(per_layer_aucs, per_layer_auprs)):
            results[f"AUC, Layer {i}"] = auc
            results[f"AUPR, Layer {i}"] = aupr

        # Summary metrics
        results.update({
            "AUC, Sum": sum_auc,
            "AUC, Mean": mean_auc,
            "AUC, Max": max_auc,
            "AUPR, Sum": sum_aupr,
            "AUPR, Mean": mean_aupr,
            "AUPR, Max": max_aupr
        })

        return results


class PerformanceEvaluator:
    """Evaluates model performance on PPI prediction tasks."""

    def __init__(self, d_esm: int = 1152):
        self.attention_analyzer = AttentionAnalyzer()
        self.d_esm = d_esm  # Dimension of ESM features

    def evaluate_unsupervised_learning(self, dump_file: Union[str, Path],
                                       include_supervised: bool = True,
                                       n_replicas: int = 5,
                                       save_models: bool = True,
                                       models_save_dir: Union[str, Path, None] = None) -> Dict[str, Any]:
        """
        Evaluate performance on unsupervised learning and optionally supervised learning.

        Args:
            dump_file: Path to the dump file containing PPI features
            include_supervised: Whether to include supervised learning evaluation
            n_replicas: Number of replicas to train for supervised learning
            save_models: Whether to save trained models (only applies to supervised learning)
            models_save_dir: Directory to save models (if None, creates 'trained_models' in current dir)

        Returns:
            Dictionary containing evaluation results
        """

        logger.info(f"Loading evaluation data from {dump_file}")

        with open(dump_file, "rb") as f:
            dump_dict = pickle.load(f)

        # Extract test set data for unsupervised evaluation
        test_data = dump_dict["test"]
        y = test_data["y"]
        A = test_data["A"]

        # Analyze attention patterns (unsupervised)
        results = self.attention_analyzer.analyze_attention_performance(A, y)

        # Add supervised learning evaluation if requested
        if include_supervised:
            supervised_results = self._evaluate_supervised_learning(
                dump_dict, n_replicas, save_models, models_save_dir
            )
            results.update(supervised_results)

        logger.info(f"Evaluation completed - Mean AUPR: {results['AUPR, Mean']:.3f}")
        return results

    def _evaluate_supervised_learning(self, dump_dict: Dict[str, Any], n_replicas: int = 5,
                                      save_models: bool = True, models_save_dir: Union[str, Path, None] = None) -> Dict[str, Any]:
        """
        Evaluate supervised learning performance using multiple feature combinations.

        Args:
            dump_dict: Dictionary containing train/val/test data
            n_replicas: Number of replicas to train for each feature combination
            save_models: Whether to save trained models
            models_save_dir: Directory to save models (if None, creates 'trained_models' in current dir)

        Returns:
            Dictionary containing supervised learning results
        """

        logger.info("Starting supervised learning evaluation")

        # Set up model saving directory
        if save_models:
            if models_save_dir is None:
                models_save_dir = Path("trained_models")
            else:
                models_save_dir = Path(models_save_dir)
            models_save_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Will save trained models to: {models_save_dir}")

        # Convert tensors to numpy arrays
        for k in dump_dict.keys():
            dump_dict[k]["repr_proteomelm"] = dump_dict[k]["logits_proteomelm"].float().numpy()
            dump_dict[k]["repr_esm"] = dump_dict[k]["repr_esm"].float().numpy()

        d = dump_dict["test"]["repr_proteomelm"].shape[-1] // 2

        # Prepare feature combinations for all splits
        X = {}
        for k in ["train", "val", "test"]:
            X[k] = self._prepare_feature_combinations(dump_dict[k], d, self.d_esm)

        # Add per-layer ProteomeLM features
        """n_layers = len(dump_dict["test"]["all_representations"])
        D = dump_dict["test"]["all_representations"].size(-1) // 2
        for k in ["train", "val", "test"]:
            reprs = dump_dict[k]["all_representations"].view(-1, n_layers + 1, 2 * D).float().numpy()
            for i in range(n_layers):
                X[k][f"ProteomeLM-Layer{i + 1}"] = {
                    "edges": dump_dict[k]["A"],
                    "x1": reprs[:, i, :D],
                    "x2": reprs[:, i, D:]
                }"""

        # Prepare labels
        y = {}
        for k in ["train", "val", "test"]:
            y[k] = dump_dict[k]["y"]

        # Train and evaluate models
        results = {}
        trained_models = {}  # Store references to trained models

        for key in X["train"].keys():
            logger.info(f"Training model for {key}")

            for i in range(n_replicas):
                logger.debug(f"Training replica {i} for {key}")

                # Train model
                model, metrics = train_model_cv(
                    X["train"][key], X["val"][key], y["train"], y["val"],
                    n_epochs=200, patience=20, verbose=False, replica_seed=i
                )

                # Test model
                _, _, _, metrics_test = test_model_cv(model, X["test"][key], y["test"])

                # Store results
                results[f"AUC, Supervised, {key}_{i}"] = metrics_test["auc"]
                results[f"AUPR, Supervised, {key}_{i}"] = metrics_test["aupr"]

                # Save model weights if requested
                if save_models:
                    model_name = f"{key}_replica_{i}"
                    model_path = models_save_dir / f"{model_name}.pt"

                    # Save model state dict and metadata
                    model_data = {
                        'state_dict': model.state_dict(),
                        'feature_combination': key,
                        'replica': i,
                        'train_metrics': metrics,
                        'test_metrics': metrics_test,
                        'model_architecture': {
                            'protein_embed_dim': model.protein_embed_dim,
                            'pair_feature_dim': model.pair_feature_dim
                        }
                    }

                    torch.save(model_data, model_path)
                    trained_models[model_name] = model_path
                    logger.debug(f"Saved model: {model_path}")

        # Save model registry if any models were saved
        if save_models and trained_models:
            registry_path = models_save_dir / "model_registry.pkl"
            with open(registry_path, 'wb') as f:
                pickle.dump(trained_models, f)
            logger.info(f"Saved model registry to: {registry_path}")
            results["saved_models"] = trained_models

        logger.info("Supervised learning evaluation completed")
        return results

    def evaluate_cross_species(self, data_path: Union[str, Path], species: List[str]) -> Dict[str, Any]:
        """
        Evaluate cross-species generalization performance.

        Args:
            data_path: Base path containing species-specific data
            species: List of species to evaluate

        Returns:
            Dictionary containing cross-species evaluation results
        """
        logger.info(f"Starting cross-species evaluation for {len(species)} species")

        # Load data for all species
        y_dict = {}
        X_test_dict = {}

        for spec in species:
            dump_file = os.path.join(data_path, spec, "dump_dict.pkl")

            with open(dump_file, "rb") as f:
                dump_dict = pickle.load(f)

            # Convert tensors to numpy arrays
            for k in dump_dict.keys():
                dump_dict[k]["repr_proteomelm"] = dump_dict[k]["logits_proteomelm"].float().numpy()
                dump_dict[k]["repr_esm"] = dump_dict[k]["repr_esm"].float().numpy()

            d = dump_dict["test"]["repr_proteomelm"].shape[-1] // 2

            # Prepare test data for all species
            test_data = dump_dict["test"]
            y_dict[spec] = test_data["y"]
            X_test_dict[spec] = self._prepare_feature_combinations(test_data, d, self.d_esm)
        # Train and evaluate models (simplified version without actual training)
        results = {}

        # For now, return empty results since training is commented out in original
        logger.info("Cross-species evaluation completed")
        return results

    def _prepare_feature_combinations(self, data: Dict[str, Any], d: int, d_esm: int) -> Dict[str, Dict[str, Any]]:
        """
        Prepare different feature combinations for evaluation.

        Args:
            data: Data dictionary for a split
            d: Dimension of ProteomeLM features (per protein)
            d_esm: Dimension of ESM features (per protein)

        Returns:
            Dictionary containing different feature combinations
        """
        combinations = {
            "ProteomeLM+Att": {
                "edges": data["A"],
                "x1": data["repr_proteomelm"][:, :d],
                "x2": data["repr_proteomelm"][:, d:]
            },
            "ProteomeLM": {
                "edges": None,
                "x1": data["repr_proteomelm"][:, :d],
                "x2": data["repr_proteomelm"][:, d:]
            },
            "ESM": {
                "edges": None,
                "x1": data["repr_esm"][:, :d_esm],
                "x2": data["repr_esm"][:, d_esm:]
            },
            "Att": {
                "edges": data["A"],
                "x1": None,
                "x2": None
            },
            "ProteomeLM+ESM+Att": {
                "edges": data["A"],
                "x1": np.concatenate([data["repr_proteomelm"][:, :d], data["repr_esm"][:, :d_esm]], -1),
                "x2": np.concatenate([data["repr_proteomelm"][:, d:], data["repr_esm"][:, d_esm:]], -1)
            },
            "ProteomeLM+ESM": {
                "edges": None,
                "x1": np.concatenate([data["repr_proteomelm"][:, :d], data["repr_esm"][:, :d_esm]], -1),
                "x2": np.concatenate([data["repr_proteomelm"][:, d:], data["repr_esm"][:, d_esm:]], -1)
            },
            "ESM+Att": {
                "edges": data["A"],
                "x1": data["repr_esm"][:, :d_esm],
                "x2": data["repr_esm"][:, d_esm:]
            }
        }

        return combinations
