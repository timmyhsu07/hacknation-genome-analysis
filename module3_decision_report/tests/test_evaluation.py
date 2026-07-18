"""Tests for the held-out evaluation panel (evaluation.run_evaluation)."""

from __future__ import annotations

from decision_report.evaluation import run_evaluation
from decision_report.mock_pipeline import MOCK_SPECIES, COVERED_DRUGS, MockPredictor, build_held_out_set


def test_evaluation_produces_expected_shapes(config):
    held_out = build_held_out_set(n=40, seed=7)
    result = run_evaluation(held_out, MockPredictor(), config, species=MOCK_SPECIES)

    assert set(result.per_drug["drug"]) == set(COVERED_DRUGS)
    for col in ("auroc", "pr_auc", "brier", "no_call_rate", "accuracy_on_called"):
        assert col in result.per_drug.columns

    assert "seen_in_training" in result.per_group.columns
    assert set(result.per_group["seen_in_training"]) <= {True, False}

    for key in ("n_predictions", "balanced_accuracy", "no_call_rate", "accuracy_on_called"):
        assert key in result.overall
    assert result.overall["n_predictions"] == len(held_out) * len(COVERED_DRUGS)


def test_evaluation_is_deterministic_for_a_fixed_seed(config):
    held_out_a = build_held_out_set(n=20, seed=99)
    held_out_b = build_held_out_set(n=20, seed=99)
    result_a = run_evaluation(held_out_a, MockPredictor(), config, species=MOCK_SPECIES)
    result_b = run_evaluation(held_out_b, MockPredictor(), config, species=MOCK_SPECIES)
    assert result_a.overall == result_b.overall


def test_evaluation_handles_empty_held_out_set(config):
    result = run_evaluation([], MockPredictor(), config, species=MOCK_SPECIES)
    assert result.records.empty
    assert result.overall == {}
    assert result.reliability == []
