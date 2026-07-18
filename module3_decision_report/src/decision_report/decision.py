"""Decision logic (Module 3 owns this).

Maps (calibrated probability, target-present gate, evidence category, OOD score)
to one of {likely to fail, likely to work, no-call} using a documented,
first-match-wins rule order. Returning a no-call for weak, conflicting, or
out-of-distribution evidence is intended behavior, not a failure.

Rule order (first match wins):
  1. drug not covered            -> NO_CALL(DRUG_NOT_COVERED)   [see report.py]
  2. probability missing/invalid -> NO_CALL(INVALID_INPUT)
  3. target_present is False      -> LIKELY_TO_FAIL (intrinsic; overrides prob)
  4. ood_score >= ood_threshold  -> NO_CALL(OUT_OF_DISTRIBUTION)
  5. prob in [band_low, band_high] -> NO_CALL(UNCERTAINTY_BAND)
  6. known mechanism + model susceptible -> NO_CALL(CONFLICTING_EVIDENCE)
  7. no resistance signal + model resistant -> NO_CALL(CONFLICTING_EVIDENCE)
  8. prob > band_high            -> LIKELY_TO_FAIL
  9. prob < band_low            -> LIKELY_TO_WORK
"""

from __future__ import annotations

import math

from .config import DecisionConfig
from .contracts import (
    DecisionLabel,
    DrugDecision,
    DrugPrediction,
    EvidenceCategory,
    NoCallReason,
)
from .evidence import EvidenceResult

IMPORTANCE_CAVEAT = (
    "Feature importance/SHAP reflects statistical association with the label, "
    "NOT biological causation."
)
INTRINSIC_NOTE = (
    "No molecular target present: the drug is intrinsically ineffective. This is "
    "a deterministic call and never 'likely to work' on an absent-marker basis."
)


def _supporting_hits(evidence: EvidenceResult) -> list:
    if evidence.category is EvidenceCategory.KNOWN_MECHANISM:
        return list(evidence.curated_hits)
    if evidence.category is EvidenceCategory.ASSOCIATION_ONLY:
        return list(evidence.associated_hits)
    return []


def _caveats(evidence: EvidenceResult) -> list[str]:
    caveats: list[str] = []
    if evidence.category is EvidenceCategory.ASSOCIATION_ONLY:
        caveats.append(
            "Association-only evidence: drivers are not curated resistance "
            "mechanisms; treat as correlation, not proven cause."
        )
    if evidence.driver_features:
        caveats.append(IMPORTANCE_CAVEAT)
    return caveats


def _no_call(
    prediction: DrugPrediction,
    evidence: EvidenceResult,
    reason: NoCallReason,
    rationale: str,
) -> DrugDecision:
    return DrugDecision(
        drug=prediction.drug,
        label=DecisionLabel.NO_CALL,
        evidence_category=evidence.category,
        calibrated_prob_resistant=prediction.calibrated_prob_resistant,
        target_present=prediction.target_present,
        intrinsic_resistance=False,
        no_call_reason=reason,
        ood_score=prediction.ood_score,
        supporting_hits=_supporting_hits(evidence),
        top_features=list(prediction.top_features),
        rationale=rationale,
        caveats=_caveats(evidence),
    )


def decide(
    prediction: DrugPrediction,
    evidence: EvidenceResult,
    config: DecisionConfig,
) -> DrugDecision:
    """Apply rules 2-9 for a covered drug. Rule 1 is handled by the report layer."""
    p = prediction.calibrated_prob_resistant

    # Rule 2: invalid / missing probability -> no silent guess.
    if p is None or (isinstance(p, float) and math.isnan(p)) or not (0.0 <= p <= 1.0):
        return _no_call(
            prediction,
            evidence,
            NoCallReason.INVALID_INPUT,
            "Calibrated probability is missing or out of [0,1]; refusing to guess.",
        )

    # Rule 3: deterministic target-absent override.
    if prediction.target_present is False:
        return DrugDecision(
            drug=prediction.drug,
            label=DecisionLabel.LIKELY_TO_FAIL,
            evidence_category=EvidenceCategory.KNOWN_MECHANISM,
            calibrated_prob_resistant=p,
            target_present=False,
            intrinsic_resistance=True,
            no_call_reason=None,
            ood_score=prediction.ood_score,
            supporting_hits=_supporting_hits(evidence),
            top_features=list(prediction.top_features),
            rationale=INTRINSIC_NOTE,
            caveats=[],
        )

    # Rule 4: out-of-distribution genome.
    if prediction.ood_score >= config.ood_threshold:
        return _no_call(
            prediction,
            evidence,
            NoCallReason.OUT_OF_DISTRIBUTION,
            f"OOD score {prediction.ood_score:.2f} >= threshold "
            f"{config.ood_threshold:.2f}; genome is unlike the training data.",
        )

    # Rule 5: uncertainty band around 0.5.
    if config.uncertainty_band_low <= p <= config.uncertainty_band_high:
        return _no_call(
            prediction,
            evidence,
            NoCallReason.UNCERTAINTY_BAND,
            f"Calibrated probability {p:.2f} is inside the uncertainty band "
            f"[{config.uncertainty_band_low:.2f}, {config.uncertainty_band_high:.2f}].",
        )

    resistant_side = p > config.uncertainty_band_high
    susceptible_side = p < config.uncertainty_band_low

    # Rule 6: known mechanism present but model leans susceptible.
    if evidence.category is EvidenceCategory.KNOWN_MECHANISM and susceptible_side:
        return _no_call(
            prediction,
            evidence,
            NoCallReason.CONFLICTING_EVIDENCE,
            "A curated known resistance mechanism was detected, but the model "
            f"leans susceptible (prob {p:.2f}). Deferring to human review.",
        )

    # Rule 7: model leans resistant but there is no resistance signal at all.
    if evidence.category is EvidenceCategory.NO_SIGNAL and resistant_side:
        return _no_call(
            prediction,
            evidence,
            NoCallReason.CONFLICTING_EVIDENCE,
            f"The model leans resistant (prob {p:.2f}) but no known resistance "
            "signal was found; the call is unsupported by mechanism.",
        )

    # Rules 8/9: confident call.
    if resistant_side:
        label = DecisionLabel.LIKELY_TO_FAIL
        rationale = (
            f"Calibrated probability of resistance {p:.2f} exceeds "
            f"{config.uncertainty_band_high:.2f}."
        )
    else:
        label = DecisionLabel.LIKELY_TO_WORK
        rationale = (
            f"Calibrated probability of resistance {p:.2f} is below "
            f"{config.uncertainty_band_low:.2f}."
        )

    return DrugDecision(
        drug=prediction.drug,
        label=label,
        evidence_category=evidence.category,
        calibrated_prob_resistant=p,
        target_present=prediction.target_present,
        intrinsic_resistance=False,
        no_call_reason=None,
        ood_score=prediction.ood_score,
        supporting_hits=_supporting_hits(evidence),
        top_features=list(prediction.top_features),
        rationale=rationale,
        caveats=_caveats(evidence),
    )


def no_call_uncovered(drug: str) -> DrugDecision:
    """Rule 1: a requested drug the predictor does not cover."""
    return DrugDecision(
        drug=drug,
        label=DecisionLabel.NO_CALL,
        evidence_category=EvidenceCategory.NO_SIGNAL,
        calibrated_prob_resistant=None,
        target_present=None,
        intrinsic_resistance=False,
        no_call_reason=NoCallReason.DRUG_NOT_COVERED,
        ood_score=None,
        supporting_hits=[],
        top_features=[],
        rationale=f"'{drug}' is not covered by the current predictor.",
        caveats=[],
    )
