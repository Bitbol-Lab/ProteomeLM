"""proteomelm.hpi.dataloaders.DataCollatorForProteomeLMHP.__call__"""
import pytest
import torch

from proteomelm.hpi.dataloaders import DataCollatorForProteomeLMHP


def _instance(seq_len: int, dim: int = 4, masked_at=(), source_ids=None) -> dict:
    masked = torch.zeros(seq_len, dtype=torch.long)
    for idx in masked_at:
        masked[idx] = 1
    inst = {
        "inputs_embeds": torch.randn(seq_len, dim),
        "group_embeds": torch.randn(seq_len, dim),
        "masked_tokens": masked,
    }
    if source_ids is not None:
        inst["source_ids"] = torch.tensor(source_ids, dtype=torch.long)
    return inst


@pytest.fixture
def collator():
    return DataCollatorForProteomeLMHP()


def test_raises_on_empty_instance_list(collator):
    with pytest.raises(ValueError, match="Cannot collate empty list"):
        collator([])


def test_raises_on_missing_required_key(collator):
    bad = {"inputs_embeds": torch.randn(3, 4), "group_embeds": torch.randn(3, 4)}  # no masked_tokens
    with pytest.raises(ValueError, match="missing keys"):
        collator([bad])


def test_all_empty_instances_returns_placeholder_batch(collator):
    empty = {
        "inputs_embeds": torch.empty(0, 4),
        "group_embeds": torch.empty(0, 4),
        "masked_tokens": torch.empty(0, dtype=torch.long),
    }
    batch = collator([empty, empty])
    assert batch["inputs_embeds"].shape == (0, 0, 512)
    assert batch["masked_tokens"].shape == (0, 0)


def test_pads_variable_length_instances(collator):
    a = _instance(seq_len=3, masked_at=[1])
    b = _instance(seq_len=5, masked_at=[0, 4])
    batch = collator([a, b])

    assert batch["inputs_embeds"].shape == (2, 5, 4)
    assert batch["attention_mask"].tolist() == [[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]]


def test_masked_positions_replaced_with_group_embeds(collator):
    a = _instance(seq_len=3, masked_at=[1])
    batch = collator([a])
    masked_pos = batch["masked_tokens"][0] == 1
    assert torch.allclose(batch["inputs_embeds"][0][masked_pos], batch["group_embeds"][0][masked_pos])


def test_labels_are_ignored_index_except_at_masked_positions(collator):
    a = _instance(seq_len=4, masked_at=[0, 2])
    b = _instance(seq_len=2, masked_at=[1])
    batch = collator([a, b])

    labels = batch["labels"]
    # a: positions 0,2 masked (kept as real labels); 1,3 -> -100
    assert not torch.equal(labels[0, 0], torch.full((4,), -100.0))
    assert torch.equal(labels[0, 1], torch.full((4,), -100.0))
    assert torch.equal(labels[0, 3], torch.full((4,), -100.0))
    # b is padded from length 2 to 4 -> positions 2,3 are padding -> -100
    assert torch.equal(labels[1, 2], torch.full((4,), -100.0))
    assert torch.equal(labels[1, 3], torch.full((4,), -100.0))


def test_source_ids_padded_when_present_on_all_instances(collator):
    a = _instance(seq_len=2, source_ids=[0, 0])
    b = _instance(seq_len=4, source_ids=[0, 0, 1, 1])
    batch = collator([a, b])
    assert batch["source_ids"].shape == (2, 4)
    assert batch["source_ids"][0].tolist() == [0, 0, 0, 0]  # padded with 0
    assert batch["source_ids"][1].tolist() == [0, 0, 1, 1]


def test_source_ids_omitted_when_not_all_instances_have_it(collator):
    a = _instance(seq_len=2, source_ids=[0, 1])
    b = _instance(seq_len=2)  # no source_ids
    batch = collator([a, b])
    assert "source_ids" not in batch


def test_mixed_valid_and_fully_empty_instances_keeps_only_valid(collator):
    valid = _instance(seq_len=3, masked_at=[0])
    empty = {
        "inputs_embeds": torch.empty(0, 4),
        "group_embeds": torch.empty(0, 4),
        "masked_tokens": torch.empty(0, dtype=torch.long),
    }
    batch = collator([valid, empty])
    assert batch["inputs_embeds"].shape[0] == 1  # only the valid instance survives
