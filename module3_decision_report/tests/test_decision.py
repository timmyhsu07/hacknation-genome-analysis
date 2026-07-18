"""Unit tests for the decision rule engine (decision.decide), rule by rule.

Each test isolates one branch by constructing the DrugPrediction/EvidenceResult
directly, independent of evidence categorization (covered separately in
test_evidence.py).
"""

from __future__ import annotations

import math

from decision_report.contracts import DecisionLabel, DrugPrediction, EvidenceCategory, NoCallReason
from decision_report.decision import decide, no_call_uncovered
from decision_report.evidence import EvidenceResult


def _pred(prob, target_present=True, ood=0.1) -> DrugPrediction:
    return DrugPrediction(
        drug="Ciprofloxacin",
        calibrated_prob_resistant=prob,
        target_present=target_present,
        top_features=[],
        ood_score=ood,
    )


def _evidence(category: EvidenceCategory) -> EvidenceResult:
    return EvidenceResult(category=category, curated_hits=[], associated_hits=[], driver_features=[])


def test_rule2_invalid_probability_is_no_call(config):
    for bad in (None, math.nan, 1.5, -0.1):
        dec = decide(_pred(bad), _evidence(EvidenceCategory.NO_SIGNAL), config)
        assert dec.label is DecisionLabel.NO_CALL
        assert dec.no_call_reason is NoCallReason.INVALID_INPUT


def test_rule3_target_absent_overrides_low_probability(config):
    dec = decide(_pred(0.05, target_present=False), _evidence(EvidenceCategory.NO_SIGNAL), config)
    assert dec.label is DecisionLabel.LIKELY_TO_FAIL
    assert dec.intrinsic_resistance is True
    assert dec.no_call_reason is None


def test_rule4_out_of_distribution_is_no_call(config):
    dec = decide(_pred(0.9, ood=config.ood_threshold), _evidence(EvidenceCategory.ASSOCIATION_ONLY), config)
    assert dec.label is DecisionLabel.NO_CALL
    assert dec.no_call_reason is NoCallReason.OUT_OF_DISTRIBUTION


def test_rule5_uncertainty_band_is_no_call(config):
    mid = (config.uncertainty_band_low + config.uncertainty_band_high) / 2
    dec = decide(_pred(mid), _evidence(EvidenceCategory.NO_SIGNAL), config)
    assert dec.label is DecisionLabel.NO_CALL
    assert dec.no_call_reason is NoCallReason.UNCERTAINTY_BAND


def test_rule6_known_mechanism_but_model_susceptible_conflicts(config):
    dec = decide(_pred(0.1), _evidence(EvidenceCategory.KNOWN_MECHANISM), config)
    assert dec.label is DecisionLabel.NO_CALL
    assert dec.no_call_reason is NoCallReason.CONFLICTING_EVIDENCE


def test_rule7_no_signal_but_model_resistant_conflicts(config):
    dec = decide(_pred(0.9), _evidence(EvidenceCategory.NO_SIGNAL), config)
    assert dec.label is DecisionLabel.NO_CALL
    assert dec.no_call_reason is NoCallReason.CONFLICTING_EVIDENCE


def test_rule8_confident_resistant_call(config):
    dec = decide(_pred(0.9), _evidence(EvidenceCategory.ASSOCIATION_ONLY), config)
    assert dec.label is DecisionLabel.LIKELY_TO_FAIL
    assert dec.no_call_reason is None


def test_rule9_confident_susceptible_call(config):
    dec = decide(_pred(0.1), _evidence(EvidenceCategory.ASSOCIATION_ONLY), config)
    assert dec.label is DecisionLabel.LIKELY_TO_WORK
    assert dec.no_call_reason is None


def test_rule1_uncovered_drug_is_no_call():
    dec = no_call_uncovered("Meropenem")
    assert dec.label is DecisionLabel.NO_CALL
    assert dec.no_call_reason is NoCallReason.DRUG_NOT_COVERED
