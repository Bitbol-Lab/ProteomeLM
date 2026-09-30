"""Layer-wise geometry of ProteomeLM embeddings, per genome (SI of the PNAS paper).

For every labelled genome (>= 10 proteins and >= 10 labelled) and every layer
(``range(n_layers)``, same indexing as the classifiers: 0 = input projection), compute
on the per-protein hidden states (at most 10,000 proteins):

* the Two-NN intrinsic dimension (dadapy; ``pip install dadapy``),
* the matrix entropy of the Gram matrix (raw and after centring + RMS scaling),
* the PCA explained-variance ratios (first 1000 components).

Writes ``{interpretability_folder}/{weights}-interpretability-{size}.{csv,pkl}``;
``figures.py interpretability`` plots them.
"""
import argparse
import concurrent.futures
import os
import pickle
import re

import numpy as np
import pandas as pd
import torch
from sklearn import decomposition
from threadpoolctl import threadpool_limits

from experiments.essentiality.common import PLM_SIZES, embeds_file_prefix, load_config, plm_checkpoint_path, \
    stored_embeds_folder

N_LAYERS = {"XS": 6, "S": 6, "M": 12, "L": 18}


def _import_dadapy_data():
    try:
        from dadapy.data import Data
    except ImportError as e:
        raise ImportError("The Two-NN intrinsic dimension needs dadapy: pip install dadapy") from e
    return Data


class ComputeMetrics:
    def __init__(self, repr_data, max_length, checkpoint, taxid,
                 which_metrics=("ID", "Entropy", "Entropy Std", "PCA Variance")):
        self.plm_representations = repr_data["plm_all_representations"]
        self.n_layers, _, self.dim = repr_data["plm_all_representations"].shape
        self.max_length = max_length
        self.checkpoint = checkpoint
        self.taxid = taxid
        self.which_metrics = which_metrics

    def __call__(self, which_layer):
        plm_representations = self.plm_representations[which_layer]
        id = entropy = entropy_std = pca_variance = None
        if "ID" in self.which_metrics:
            id, _, _ = self.compute_intrinsic_dimension(plm_representations, self.dim, max_length=self.max_length)
        if "Entropy" in self.which_metrics:
            entropy = self.compute_matrix_entropy(plm_representations, self.dim, max_length=self.max_length)
        if "Entropy Std" in self.which_metrics:
            entropy_std = self.compute_matrix_entropy(plm_representations, self.dim, max_length=self.max_length,
                                                      standardize=True)
        if "PCA Variance" in self.which_metrics:
            pca_variance = self.compute_PCA_variance(plm_representations, self.dim, max_length=self.max_length)
        return {"TaxID": self.taxid, "Model": self.checkpoint, "Layer": which_layer, "ID": id, "Entropy": entropy,
                "Entropy Std": entropy_std, "PCA Variance": pca_variance}

    @staticmethod
    def compute_intrinsic_dimension(data, dim, max_length=None):
        """Two-NN estimator on (N, dim) coordinates; returns (id, error, distance scale)."""
        Data = _import_dadapy_data()
        coordinates = data.reshape((-1, dim)).to(device="cpu", dtype=torch.float).numpy()[:max_length]
        return Data(coordinates=coordinates).compute_id_2NN()

    @staticmethod
    def compute_matrix_entropy(data, dim, cutoff=1e-9, max_length=None, transpose=False, standardize=False):
        """Von Neumann entropy of the normalized Gram-matrix spectrum."""
        data = data.reshape((-1, dim)).to(device="cpu", dtype=torch.float).numpy()[:max_length]
        if standardize:
            data = data - np.mean(data, axis=0)
            rms = np.sqrt(np.mean(np.sum(data ** 2, axis=1)))
            data = data / rms
        gram_matrix = data.T.dot(data) if transpose else data.dot(data.T)
        eigenvalues, _ = np.linalg.eigh(gram_matrix)
        eigenvalues = eigenvalues[eigenvalues >= cutoff]
        eigenvalues = eigenvalues / np.sum(eigenvalues)
        return -np.sum(eigenvalues * np.log(eigenvalues))

    @staticmethod
    def compute_PCA_variance(data, dim, max_length=None, pca_components=1000):
        """[(component, explained variance ratio)] padded with zeros to ``pca_components``."""
        data = data.reshape((-1, dim)).to(device="cpu", dtype=torch.float).numpy()[:max_length]
        n_comps = min(data.shape[0], dim, pca_components)
        pca = decomposition.PCA(n_components=n_comps)
        pca.fit(data)
        explained_variance = np.array(pca.explained_variance_ratio_)
        if n_comps < pca_components:
            explained_variance.resize(pca_components)
        return [(i, var) for i, var in enumerate(explained_variance)]


def baseline_seed(baseline_dir: str, size: str, weights: str) -> int:
    """Smallest seed among the existing ``ProteomeLM-{size}-{weights}-seed{S}`` baselines."""
    prefix = f"ProteomeLM-{size}-{weights}-seed"
    seeds = sorted(int(d[len(prefix):]) for d in os.listdir(baseline_dir) if d.startswith(prefix))
    if not seeds:
        raise FileNotFoundError(f"no {prefix}* in {baseline_dir}: run `train make-baselines` first")
    return seeds[0]


def run(cfg, checkpoint: str, weights: str = "trained", which_hidden_layer=None, device="cuda:0",
        num_workers=None, checkpoint_dir=None, max_length=10000):
    from experiments.essentiality.train import embed_all_taxids
    _import_dadapy_data()  # fail before embedding if dadapy is missing
    p = cfg["paths"]
    seed = baseline_seed(p["baseline_weights_folder"], checkpoint, weights) if weights != "trained" else None
    plm_path = plm_checkpoint_path(checkpoint, weights, seed, checkpoint_dir=checkpoint_dir,
                                   baseline_dir=p["baseline_weights_folder"])
    embeds_dir = stored_embeds_folder(p["embeds_folder"], checkpoint)
    prefix = embeds_file_prefix(weights, seed)
    n_layers = N_LAYERS[checkpoint]
    layers = np.arange(n_layers) if which_hidden_layer is None else [which_hidden_layer]
    num_workers = min(num_workers if num_workers is not None else n_layers, os.cpu_count())

    embed_all_taxids(p["fasta_folder"], p["label_folder"], embeds_dir, esm_device=device, proteomelm_device=device,
                     proteomelm_checkpoint=plm_path, taxids_to_exclude=cfg["classifier"]["which_taxids_to_exclude"],
                     esmc_embeds_folder=p["esmc_embeds_folder"], file_prefix=prefix)

    all_results = []
    with threadpool_limits(limits=num_workers):
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for embed_file in sorted(os.listdir(embeds_dir)):
                match = re.fullmatch(rf"{re.escape(prefix)}embeds_taxid(\d+)\.pkl", embed_file)
                if not match:
                    continue
                with open(os.path.join(embeds_dir, embed_file), "rb") as f:
                    data = pickle.load(f)
                if data is None:
                    continue
                repr_data, labels = data
                num_of_labels = sum(1 for v in labels.values() if v["Essentiality"])
                if (len(labels) < 10) or (num_of_labels < 10):
                    continue
                compute_metrics = ComputeMetrics(repr_data=repr_data, max_length=max_length, checkpoint=checkpoint,
                                                 taxid=match.group(1))
                futures.extend(executor.submit(compute_metrics, layer) for layer in layers)
            for future in concurrent.futures.as_completed(futures):
                all_results.append(future.result())

    out_dir = p["interpretability_folder"]
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, f"{weights}-interpretability-{checkpoint}")
    pd.DataFrame(data=all_results).to_csv(f"{stem}.csv")
    with open(f"{stem}.pkl", "wb") as f:
        pickle.dump(all_results, f)
    print(f"Saved {stem}.csv")
    return f"{stem}.csv"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Layer-wise intrinsic dimension / entropy / PCA of ProteomeLM embeddings")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--checkpoint", "-c", default="M", choices=PLM_SIZES)
    parser.add_argument("--which-weights", default="trained", choices=("trained", "random", "statistics"))
    parser.add_argument("--which-hidden-layer", type=int, default=None)
    parser.add_argument("--gpu", "-g", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=None, help="CPU processes (default: number of layers)")
    args = parser.parse_args(argv)
    cfg = load_config(args.config, args.data_dir)
    run(cfg, args.checkpoint, args.which_weights, args.which_hidden_layer, f"cuda:{args.gpu}", args.num_workers,
        args.checkpoint_dir)


if __name__ == "__main__":
    main()
