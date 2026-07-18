"""Typed interface contract for Module 3 (The Decision Report).

This module defines:

1. The **consumed** interfaces Module 3 depends on but does NOT implement:
   * ``FeatureExtractor`` / ``FeatureBundle`` (Module 1 output),
   * ``Predictor`` / ``DrugPrediction`` (Module 2 output),
   * ``HeldOutGenome`` (evaluation loader).
   Module 3 ships a *mock* satisfying each so it runs standalone (see
   :mod:`decision_report.mock_pipeline`).

2. The types Module 3 **owns**: the decision/evidence/no-call enums and the
   per-drug ``DrugDecision`` / ``GenomeReport`` structures.

Semantics note on labels: a drug is "likely to fail" when the organism is
resistant to it, and "likely to work" when susceptible. Module 3 never asserts a
treatment decision -- every output requires human confirmation by lab testing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable

# --------------------------------------------------------------------------- #
# Consumed: Module 1 (feature extraction)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class LongHit:
    """One AMRFinderPlus-style hit from Module 1's long/provenance table."""

    element_symbol: str
    element_type: str  # "AMR" | "VIRULENCE" | "STRESS" | ...
    element_subtype: str  # "acquired_gene" | "point_mutation" | ...
    method: str  # "EXACTX" | "POINTX" | "BLASTX" | "PARTIALX" | "HMM" | ...
    drug_class: str | None  # e.g. "QUINOLONE", "BETA-LACTAM"
    pct_identity: float | None = None
    pct_coverage: float | None = None


@dataclass(frozen=True)
class FeatureBundle:
    """Module 1 output for one genome: the binary row + the rich hit table."""

    genome_id: str
    binary_row: dict[str, int]  # feature symbol -> 0/1 presence
    long_hits: list[LongHit] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        """True when no features were extracted at all (nothing to reason on)."""
        return not self.binary_row and not self.long_hits


@runtime_checkable
class FeatureExtractor(Protocol):
    """Module 1 interface: assembled FASTA -> feature bundle."""

    def __call__(self, fasta_path: str) -> FeatureBundle: ...


# --------------------------------------------------------------------------- #
# Consumed: Module 2 (prediction + deterministic target gate)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TopFeature:
    """A model driver for one drug prediction (SHAP/importance).

    ``contribution`` is signed: positive pushes toward *resistant*. It is a
    statistical attribution, NOT a claim of biological causation.
    """

    name: str
    contribution: float
    is_known_mechanism: bool = False


@dataclass(frozen=True)
class DrugPrediction:
    """Module 2 output for one drug."""

    drug: str
    calibrated_prob_resistant: float  # in [0, 1]
    target_present: bool  # deterministic drug-target gate (Module 2)
    top_features: list[TopFeature] = field(default_factory=list)
    ood_score: float = 0.0  # novelty / distance-from-training; higher = stranger


@runtime_checkable
class Predictor(Protocol):
    """Module 2 interface."""

    def predict(self, features: FeatureBundle) -> list[DrugPrediction]: ...

    def covered_drugs(self) -> list[str]: ...

    def covered_species(self) -> list[str]: ...


# --------------------------------------------------------------------------- #
# Consumed: held-out evaluation loader
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class HeldOutGenome:
    """One labeled held-out genome for the evaluation panel.

    ``seen_in_training`` refers to the genetic GROUP: False marks a group unseen
    in training, used for the generalization breakdown. Test labels are for
    REPORTING only -- never for tuning thresholds.
    """

    genome_id: str
    genetic_group: str
    seen_in_training: bool
    features: FeatureBundle
    true_labels: dict[str, bool]  # drug -> resistant?(True) / susceptible?(False)


# --------------------------------------------------------------------------- #
# Owned by Module 3: decision + report types
# --------------------------------------------------------------------------- #


class DecisionLabel(str, Enum):
    LIKELY_TO_FAIL = "likely to fail"  # organism resistant -> drug fails
    LIKELY_TO_WORK = "likely to work"  # organism susceptible -> drug works
    NO_CALL = "no-call"


class EvidenceCategory(str, Enum):
    KNOWN_MECHANISM = "known_resistance_mechanism"  # (i)
    ASSOCIATION_ONLY = "statistical_association_only"  # (ii)
    NO_SIGNAL = "no_known_resistance_signal"  # (iii)


class NoCallReason(str, Enum):
    UNCERTAINTY_BAND = "calibrated probability inside the uncertainty band"
    CONFLICTING_EVIDENCE = "model and mechanistic evidence disagree"
    OUT_OF_DISTRIBUTION = "genome is unlike the training data (high OOD score)"
    DRUG_NOT_COVERED = "drug is not covered by the predictor"
    INVALID_INPUT = "prediction input was missing or invalid"


@dataclass(frozen=True)
class DrugDecision:
    """The full per-drug decision surfaced on a report card."""

    drug: str
    label: DecisionLabel
    evidence_category: EvidenceCategory
    calibrated_prob_resistant: float | None
    target_present: bool | None
    intrinsic_resistance: bool  # True when the target is absent (drug can't act)
    no_call_reason: NoCallReason | None
    ood_score: float | None
    supporting_hits: list[LongHit] = field(default_factory=list)
    top_features: list[TopFeature] = field(default_factory=list)
    rationale: str = ""
    caveats: list[str] = field(default_factory=list)

    @property
    def is_no_call(self) -> bool:
        return self.label is DecisionLabel.NO_CALL


@dataclass(frozen=True)
class GenomeReport:
    """The whole per-genome report Module 3 hands to the UI."""

    genome_id: str
    species: str
    species_supported: bool
    decisions: list[DrugDecision]
    covered_drugs: list[str]
    uncovered_drugs_requested: list[str] = field(default_factory=list)
    disclaimer: str = ""
    errors: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Exceptions (graceful-failure surface)
# --------------------------------------------------------------------------- #


class DecisionReportError(Exception):
    """Base class for Module 3 errors."""


class PredictorUnavailable(DecisionReportError):
    """Raised when the predictor artifact/service cannot be loaded."""


class FeatureExtractionError(DecisionReportError):
    """Raised when features cannot be produced from an input."""
