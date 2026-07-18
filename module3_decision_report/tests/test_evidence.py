"""Unit tests for evidence categorization (evidence.categorize_evidence)."""

from __future__ import annotations

from decision_report.contracts import DrugPrediction, EvidenceCategory, FeatureBundle, LongHit, TopFeature
from decision_report.evidence import categorize_evidence

DRUG_CLASSES = {"QUINOLONE"}


def _pred(top_features=()) -> DrugPrediction:
    return DrugPrediction(
        drug="Ciprofloxacin",
        calibrated_prob_resistant=0.9,
        target_present=True,
        top_features=list(top_features),
        ood_score=0.1,
    )


def test_curated_hit_is_known_mechanism(config):
    hit = LongHit("gyrA_S83L", "AMR", "point_mutation", "POINTX", "QUINOLONE")
    features = FeatureBundle("g1", {"gyrA_S83L": 1}, [hit])
    result = categorize_evidence(_pred(), features, DRUG_CLASSES, config)
    assert result.category is EvidenceCategory.KNOWN_MECHANISM
    assert result.curated_hits == [hit]


def test_weak_method_hit_is_association_only(config):
    hit = LongHit("qnrS1", "AMR", "acquired_gene", "BLASTX", "QUINOLONE")
    features = FeatureBundle("g2", {"qnrS1": 1}, [hit])
    result = categorize_evidence(_pred(), features, DRUG_CLASSES, config)
    assert result.category is EvidenceCategory.ASSOCIATION_ONLY
    assert result.associated_hits == [hit]


def test_hit_for_unrelated_drug_class_is_ignored(config):
    hit = LongHit("blaTEM-1", "AMR", "acquired_gene", "EXACTX", "BETA-LACTAM")
    features = FeatureBundle("g3", {"blaTEM-1": 1}, [hit])
    result = categorize_evidence(_pred(), features, DRUG_CLASSES, config)
    assert result.category is EvidenceCategory.NO_SIGNAL


def test_non_amr_hit_is_ignored(config):
    hit = LongHit("someVirulenceGene", "VIRULENCE", "acquired_gene", "EXACTX", "QUINOLONE")
    features = FeatureBundle("g4", {}, [hit])
    result = categorize_evidence(_pred(), features, DRUG_CLASSES, config)
    assert result.category is EvidenceCategory.NO_SIGNAL


def test_known_mechanism_model_feature_without_any_hit(config):
    pred = _pred([TopFeature("gyrA_S83L", 0.5, is_known_mechanism=True)])
    features = FeatureBundle("g5", {}, [])
    result = categorize_evidence(pred, features, DRUG_CLASSES, config)
    assert result.category is EvidenceCategory.KNOWN_MECHANISM


def test_material_non_curated_feature_is_association_only(config):
    pred = _pred([TopFeature("kmer_1", 0.35, is_known_mechanism=False)])
    features = FeatureBundle("g6", {}, [])
    result = categorize_evidence(pred, features, DRUG_CLASSES, config)
    assert result.category is EvidenceCategory.ASSOCIATION_ONLY
    assert result.driver_features


def test_weak_non_curated_feature_below_threshold_is_no_signal(config):
    pred = _pred([TopFeature("kmer_1", 0.01, is_known_mechanism=False)])
    features = FeatureBundle("g7", {}, [])
    result = categorize_evidence(pred, features, DRUG_CLASSES, config)
    assert result.category is EvidenceCategory.NO_SIGNAL


def test_nothing_present_is_no_signal(config):
    features = FeatureBundle("g8", {}, [])
    result = categorize_evidence(_pred(), features, DRUG_CLASSES, config)
    assert result.category is EvidenceCategory.NO_SIGNAL
    assert result.curated_hits == []
    assert result.associated_hits == []
    assert result.driver_features == []
