"""proteomelm.ppi.model: EnhancedPPIModel construction/symmetry and train_model_cv batching.

Small CPU-only MLPs on random data, plus a strict load of the bundled supervised checkpoints.
"""
from pathlib import Path

import numpy as np
import pytest
import torch

import proteomelm.ppi.model as ppi_model
from proteomelm.ppi.model import EnhancedPPIModel, train_model_cv
from proteomelm.ppi.notebook_runtime import BUNDLED_ASSET_DIR, BUNDLED_MODEL_FILES, BUNDLED_MODELS_BACKBONE

PROTEIN_DIM, PAIR_DIM = 16, 8
NO_DROPOUT = {k: 0.0 for k in (
    "dropout_protein1", "dropout_protein2", "dropout_pair", "dropout_interaction",
    "dropout_classifier1", "dropout_classifier2",
)}


def _inputs(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (
        torch.randn(n, PAIR_DIM, generator=g),
        torch.randn(n, PROTEIN_DIM, generator=g),
        torch.randn(n, PROTEIN_DIM, generator=g),
    )


def _randomize_film_and_skip(model):
    # These are zero-initialised (inert); randomise them so the tests exercise them.
    torch.manual_seed(0)
    for layer in (model.protein_skip, model.film_scale, model.film_shift):
        torch.nn.init.normal_(layer.weight, std=0.1)


def test_forward_output_shape_in_train_and_eval():
    model = EnhancedPPIModel(protein_embed_dim=PROTEIN_DIM, pair_feature_dim=PAIR_DIM)
    F, E1, E2 = _inputs(10)
    assert model.train()(F, E1, E2).shape == (10, 1)
    assert model.eval()(F, E1, E2).shape == (10, 1)


def test_symmetric_in_eval_mode():
    model = EnhancedPPIModel(protein_embed_dim=PROTEIN_DIM, pair_feature_dim=PAIR_DIM)
    _randomize_film_and_skip(model)
    F, E1, E2 = _inputs(10)
    model.train()(F, E1, E2)  # populate BatchNorm running stats
    model.eval()
    with torch.no_grad():
        torch.testing.assert_close(model(F, E1, E2), model(F, E2, E1))


def test_per_pair_swap_invariant_in_train_mode():
    # E1 and E2 are encoded as one pooled batch, so swapping the two proteins of
    # *some* pairs leaves the BatchNorm statistics, and every pair's output,
    # unchanged in training mode. With separate E1/E2 passes the swap would
    # change each side's batch statistics and shift all outputs.
    model = EnhancedPPIModel(protein_embed_dim=PROTEIN_DIM, pair_feature_dim=PAIR_DIM, **NO_DROPOUT)
    _randomize_film_and_skip(model)
    F, E1, E2 = _inputs(10)
    E2 = E2 * 3 + 5  # different population on each side, e.g. host vs. pathogen
    swap = torch.arange(10) < 5
    E1_swapped = torch.where(swap[:, None], E2, E1)
    E2_swapped = torch.where(swap[:, None], E1, E2)
    model.train()
    with torch.no_grad():
        torch.testing.assert_close(model(F, E1, E2), model(F, E1_swapped, E2_swapped))


def test_score_pairs_supervised_rescales_mean_attention_to_training_sum():
    # Notebook pair features average both attention directions; the supervised
    # model is trained on their sum, so scoring must see 2x the notebook features.
    from proteomelm.ppi.notebook_inference import score_pairs_supervised

    model = EnhancedPPIModel(protein_embed_dim=PROTEIN_DIM, pair_feature_dim=PAIR_DIM)
    _randomize_film_and_skip(model)
    F, E1, E2 = _inputs(20)
    model.train()(F, E1, E2)  # populate BatchNorm running stats
    model.eval()

    embeddings = torch.randn(6, PROTEIN_DIM)
    pairs = np.array([[0, 1], [2, 3], [4, 5], [1, 4]])
    mean_features = torch.rand(len(pairs), PAIR_DIM)
    scores = score_pairs_supervised(model, embeddings, mean_features, pairs)
    with torch.no_grad():
        logits = model(2 * mean_features, embeddings[pairs[:, 0]], embeddings[pairs[:, 1]]).squeeze(-1)
    np.testing.assert_allclose(scores, torch.sigmoid(2 * logits).numpy(), rtol=1e-5)


@pytest.mark.parametrize("model_type", ["enhancedppi", "simplemlp"])
@pytest.mark.parametrize("n_train", [65, 64])  # 65 % 32 == 1 leaves a single-sample batch
def test_train_model_cv_handles_single_sample_trailing_batch(model_type, n_train):
    rng = np.random.default_rng(0)

    def split(n):
        X = {
            "edges": rng.normal(size=(n, PAIR_DIM)).astype(np.float32),
            "x1": rng.normal(size=(n, PROTEIN_DIM)).astype(np.float32),
            "x2": rng.normal(size=(n, PROTEIN_DIM)).astype(np.float32),
        }
        y = np.array([0, 1] * (n // 2) + [0] * (n % 2))
        return X, y

    X_train, y_train = split(n_train)
    X_test, y_test = split(20)
    cpu = torch.device("cpu")
    model, metrics = train_model_cv(
        X_train, X_test, y_train, y_test,
        n_epochs=2, patience=2, model_type=model_type, batch_size=32,
        warmup_epochs=1, device=cpu,
    )
    assert 0.0 <= metrics["auc"] <= 1.0
    # Accessed via the module: a bare `test_model_cv` name would be collected by pytest.
    _, _, _, test_metrics = ppi_model.test_model_cv(model, X_test, y_test, device=cpu)
    assert set(test_metrics) == {"auc", "aupr", "f1", "mcc"}



def test_average_state_dicts_averages_floats_and_keeps_last_integer_buffers():
    a = EnhancedPPIModel(protein_embed_dim=PROTEIN_DIM, pair_feature_dim=PAIR_DIM).state_dict()
    b = EnhancedPPIModel(protein_embed_dim=PROTEIN_DIM, pair_feature_dim=PAIR_DIM).state_dict()
    b["pair_branch.1.num_batches_tracked"] += 7
    avg = ppi_model.average_state_dicts([a, b])
    torch.testing.assert_close(avg["pair_branch.0.weight"], (a["pair_branch.0.weight"] + b["pair_branch.0.weight"]) / 2)
    assert avg["pair_branch.1.num_batches_tracked"] == b["pair_branch.1.num_batches_tracked"]
    assert avg["pair_branch.1.num_batches_tracked"].dtype == torch.long


def test_recompute_batchnorm_stats_matches_full_data_statistics():
    model = EnhancedPPIModel(protein_embed_dim=PROTEIN_DIM, pair_feature_dim=PAIR_DIM, **NO_DROPOUT).eval()
    F, E1, E2 = _inputs(64)
    ppi_model.recompute_batchnorm_stats(model, [(F[i:i + 16], E1[i:i + 16], E2[i:i + 16]) for i in range(0, 64, 16)])
    bn = model.pair_branch[1]
    with torch.no_grad():
        pre = model.pair_branch[0](F)
    torch.testing.assert_close(bn.running_mean, pre.mean(0), rtol=1e-4, atol=1e-5)
    assert bn.momentum == 0.1 and not model.training  # restored


@pytest.mark.parametrize("name", sorted(BUNDLED_MODEL_FILES["supervised"]))
def test_bundled_supervised_checkpoint_loads_into_current_architecture(name):
    # The notebook loads these strictly: changing EnhancedPPIModel means retraining them
    # (experiments/ppi_bundled_models/train_bundled.py).
    path = Path(__file__).resolve().parents[1] / BUNDLED_ASSET_DIR / BUNDLED_MODEL_FILES["supervised"][name]
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = EnhancedPPIModel(**checkpoint["model_architecture"])
    model.load_state_dict(checkpoint["state_dict"])
    assert checkpoint["backbone"] == BUNDLED_MODELS_BACKBONE


def test_score_pairs_supervised_keeps_confident_pairs_distinct():
    # In float32, sigmoid(2 * logit) is exactly 1.0 for every logit above ~8.3, tying the top of the ranking.
    from proteomelm.ppi.notebook_inference import score_pairs_supervised

    class PairFeatureLogit(torch.nn.Module):
        pair_feature_dim = PAIR_DIM

        def forward(self, F, E1, E2):
            return F[:, :1]

    features = torch.zeros(6, PAIR_DIM)
    features[:, 0] = torch.arange(9, 15) / 2  # logits 9..14 after the 2x rescale
    pairs = np.array([[0, 1]] * 6)
    scores = score_pairs_supervised(PairFeatureLogit(), torch.zeros(2, PROTEIN_DIM), features, pairs, device="cpu")
    assert (np.diff(scores) > 0).all() and (scores < 1).all()
