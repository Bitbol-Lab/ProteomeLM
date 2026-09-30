"""proteomelm.utils.embedding.check_embeddings: fail loudly on degenerate ESM-C output."""
import pytest
import torch

from proteomelm.utils.embedding import check_embeddings


def test_accepts_regular_embeddings():
    check_embeddings(torch.randn(4, 8), ["a", "b", "c", "d"], "cpu")


@pytest.mark.parametrize("bad_value", [0.0, float("nan"), float("inf")])
def test_rejects_zero_or_non_finite_rows(bad_value):
    emb = torch.randn(4, 8)
    emb[2] = bad_value
    with pytest.raises(RuntimeError, match=r"1/4 sequences on device 'cuda:1'.*\['c'\]"):
        check_embeddings(emb, ["a", "b", "c", "d"], "cuda:1")
