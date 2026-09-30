"""proteomelm.hpi.dataloaders.apply_masking — the free-function masking core.

Regression coverage for the Phase-0 fix: masking used to be a fixed trailing
window (same proteins masked every epoch, truncated fraction counts, and the
maskable ('me') filter applied *after* window selection so it could silently
reduce the achieved fraction). It's now random sampling restricted to the
maskable pool, resampled on every call.
"""
import random

import torch

from proteomelm.hpi.dataloaders import apply_masking


def test_masked_counts_match_rounded_fraction(make_pair_fixture):
    pair = make_pair_fixture(host_len=10, pathogen_len=10)
    out = apply_masking(pair, mask_fraction_H=0.3, mask_fraction_P=0.7)
    assert out["masked_tokens"][:10].sum().item() == round(0.3 * 10)
    assert out["masked_tokens"][10:].sum().item() == round(0.7 * 10)


def test_mask_fraction_zero_masks_nothing(make_pair_fixture):
    pair = make_pair_fixture(host_len=5, pathogen_len=5)
    out = apply_masking(pair, mask_fraction_H=0.0, mask_fraction_P=0.0)
    assert out["masked_tokens"].sum().item() == 0


def test_mask_fraction_one_masks_entire_segment(make_pair_fixture):
    pair = make_pair_fixture(host_len=4, pathogen_len=6)
    out = apply_masking(pair, mask_fraction_H=1.0, mask_fraction_P=1.0)
    assert out["masked_tokens"][:4].sum().item() == 4
    assert out["masked_tokens"][4:].sum().item() == 6


def test_masking_respects_maskable_filter(make_pair_fixture):
    # Pathogen segment: positions 2 and 4 (0-indexed within the 5-protein
    # pathogen segment) are not maskable; total masking must skip them even
    # though mask_fraction_P=1.0 nominally means "mask everything".
    maskable = [True, True, True] + [True, True, False, True, False, True]
    pair = make_pair_fixture(host_len=3, pathogen_len=6, maskable=maskable)
    out = apply_masking(pair, mask_fraction_H=0.0, mask_fraction_P=1.0)
    masked = out["masked_tokens"]
    assert masked[3:].tolist() == [1, 1, 0, 1, 0, 1]  # non-maskable positions never set


def test_masking_never_exceeds_maskable_count(make_pair_fixture):
    # Only 1 of 8 pathogen proteins is maskable; a fraction that would nominally
    # ask for more than that must be capped, not raise (random.sample requirement).
    maskable = [True, True] + [False] * 7 + [True]
    pair = make_pair_fixture(host_len=2, pathogen_len=8, maskable=maskable)
    out = apply_masking(pair, mask_fraction_H=0.0, mask_fraction_P=1.0)
    assert out["masked_tokens"][2:].sum().item() == 1


def test_degenerate_single_protein_segment_does_not_crash(make_pair_fixture):
    pair = make_pair_fixture(host_len=1, pathogen_len=3)
    out = apply_masking(pair, mask_fraction_H=0.2, mask_fraction_P=0.5)
    assert out["masked_tokens"].shape[0] == 4
    # round(0.2 * 1) == 0: a lone host protein is never masked at this fraction —
    # an inherent quantization limit at N=1, not a bug.
    assert out["masked_tokens"][0].item() == 0


def test_zero_length_pathogen_segment_does_not_crash(make_pair_fixture):
    pair = make_pair_fixture(host_len=5, pathogen_len=0)
    out = apply_masking(pair, mask_fraction_H=0.5, mask_fraction_P=0.9)
    assert out["masked_tokens"].shape[0] == 5
    assert out["masked_tokens"].sum().item() == round(0.5 * 5)


def test_source_ids_mark_host_and_pathogen_segments(make_pair_fixture):
    pair = make_pair_fixture(host_len=3, pathogen_len=2)
    out = apply_masking(pair, mask_fraction_H=0.0, mask_fraction_P=0.0)
    assert out["source_ids"].tolist() == [0, 0, 0, 1, 1]


def test_masking_is_resampled_not_a_fixed_window(make_pair_fixture):
    # With a fixed window (the pre-fix behavior) two calls on the same pair
    # would always mask the exact same positions. With random resampling,
    # repeated calls should disagree at least once over many trials.
    pair = make_pair_fixture(host_len=20, pathogen_len=0)
    random.seed(0)
    draws = {tuple(apply_masking(pair, 0.3, 0.0)["masked_tokens"].tolist()) for _ in range(20)}
    assert len(draws) > 1
