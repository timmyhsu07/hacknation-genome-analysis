"""Mock implementations of the consumed interfaces (Modules 1 & 2) + held-out
loader, so Module 3 runs standalone with zero real artifacts.

Contains:
* ``MockFeatureExtractor``  - FASTA path -> deterministic FeatureBundle.
* ``MockPredictor``         - feature-derived, plausible per-drug predictions.
* ``ScriptedPredictor``     - returns hand-authored predictions (demo scenarios).
* ``build_held_out_set``    - labeled genomes across genetic groups (some unseen)
                              for the evaluation panel.
* ``demo_cases``            - crafted (features, predictions) pairs illustrating
                              every decision branch.

Everything is seeded and deterministic. These are MOCKS: values are plausible,
not real, and clearly labeled as such.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path

from .catalog import GENE_POOL, TRAINING_VOCAB, classes_for
from .contracts import (
    DrugPrediction,
    FeatureBundle,
    FeatureExtractionError,
    HeldOutGenome,
    LongHit,
    TopFeature,
)

MOCK_SPECIES = "Escherichia coli"
COVERED_DRUGS = [
    "Ciprofloxacin",
    "Ampicillin",
    "Gentamicin",
    "Trimethoprim-sulfamethoxazole",
    "Colistin",
]

# Reserved mock-only binary_row key: encodes "the drug-target gate says the
# molecular target is absent" (Module 2's deterministic gate). Real Module 2
# supplies target_present directly; the mock reads this sentinel.
_NO_TARGET_KEY = "mock_no_target__{drug}"

_TIER_BASE_PROB = {"exact": 0.93, "blast": 0.70, "partial": 0.50, "none": 0.10}


def _seed_int(*parts: str) -> int:
    h = hashlib.sha256("|".join(parts).encode()).hexdigest()
    return int(h[:8], 16)


def _jitter(seed: int, spread: float = 0.06) -> float:
    return (_seed_int(str(seed)) % 1000 / 1000.0) * 2 * spread - spread


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _make_hit(symbol: str, drug_class: str, subtype: str, method: str) -> LongHit:
    return LongHit(
        element_symbol=symbol,
        element_type="AMR",
        element_subtype=subtype,
        method=method,
        drug_class=drug_class,
        pct_identity=99.5 if method.startswith(("EXACT", "POINT", "ALLELE")) else 88.0,
        pct_coverage=100.0 if method.startswith(("EXACT", "POINT", "ALLELE")) else 75.0,
    )


# --------------------------------------------------------------------------- #
# Mock feature extractor (Module 1 stand-in)
# --------------------------------------------------------------------------- #
class MockFeatureExtractor:
    """Deterministically fabricate a FeatureBundle from a FASTA path.

    We cannot really annotate here, so features are a seeded pseudo-sample drawn
    from the gene pool. Fails loudly on an empty/missing file.
    """

    def __call__(self, fasta_path: str) -> FeatureBundle:
        p = Path(fasta_path)
        if not p.exists():
            raise FeatureExtractionError(f"input FASTA not found: {fasta_path}")
        content = p.read_bytes()
        if not content.strip():
            raise FeatureExtractionError(f"input FASTA is empty: {fasta_path}")

        genome_id = p.name
        for ext in (".fasta", ".fa", ".fna", ".gz"):
            if genome_id.lower().endswith(ext):
                genome_id = genome_id[: -len(ext)]
        rng = random.Random(int(hashlib.sha256(content).hexdigest()[:8], 16))

        long_hits: list[LongHit] = []
        binary_row: dict[str, int] = {}
        for cls, entries in GENE_POOL.items():
            if rng.random() < 0.45:  # class present in this pseudo-genome
                sym, sub, method = rng.choice(entries)
                long_hits.append(_make_hit(sym, cls, sub, method))
                binary_row[sym] = 1
        return FeatureBundle(genome_id=genome_id, binary_row=binary_row, long_hits=long_hits)


# --------------------------------------------------------------------------- #
# Mock predictor (Module 2 stand-in)
# --------------------------------------------------------------------------- #
def _strongest_tier(features: FeatureBundle, classes: set[str]) -> tuple[str, list[LongHit]]:
    classes_u = {c.upper() for c in classes}
    relevant = [
        h
        for h in features.long_hits
        if (h.element_type or "").upper() == "AMR"
        and h.drug_class
        and h.drug_class.upper() in classes_u
    ]
    if not relevant:
        return "none", []
    methods = [(h.method or "").upper() for h in relevant]
    if any(m.startswith(("EXACT", "POINT", "ALLELE")) for m in methods):
        return "exact", relevant
    if any(m.startswith(("BLAST", "HMM")) for m in methods):
        return "blast", relevant
    if any(m.startswith("PARTIAL") for m in methods):
        return "partial", relevant
    return "blast", relevant


class MockPredictor:
    """Feature-derived predictions. Probability rises with mechanism strength;
    OOD rises with unfamiliar features; target gate read from a mock sentinel."""

    def covered_drugs(self) -> list[str]:
        return list(COVERED_DRUGS)

    def covered_species(self) -> list[str]:
        return [MOCK_SPECIES]

    def _ood(self, features: FeatureBundle) -> float:
        genes = [k for k in features.binary_row if not k.startswith("mock_")]
        if not genes:
            return 0.1
        unknown = sum(1 for g in genes if g not in TRAINING_VOCAB)
        return _clip01(0.1 + 0.9 * (unknown / len(genes)))

    def _top_features(self, tier: str, hits: list[LongHit], seed: int) -> list[TopFeature]:
        feats: list[TopFeature] = []
        contrib = {"exact": 0.45, "blast": 0.25, "partial": 0.15, "none": 0.0}[tier]
        for h in hits:
            feats.append(
                TopFeature(
                    name=h.element_symbol,
                    contribution=contrib,
                    is_known_mechanism=(tier == "exact"),
                )
            )
        if tier in ("blast", "partial"):
            feats.append(TopFeature(name=f"kmer_{seed % 9973}", contribution=0.12, is_known_mechanism=False))
        if tier == "none":
            # A known gene the model looked for but did NOT find -> negative,
            # absence-driven contribution (does not qualify as category i or ii).
            feats.append(TopFeature(name="absent_marker", contribution=-0.2, is_known_mechanism=True))
        return feats

    def predict(self, features: FeatureBundle) -> list[DrugPrediction]:
        preds: list[DrugPrediction] = []
        ood = self._ood(features)
        for drug in self.covered_drugs():
            classes = classes_for(drug)
            tier, hits = _strongest_tier(features, classes)
            seed = _seed_int(features.genome_id, drug)
            prob = _clip01(_TIER_BASE_PROB[tier] + _jitter(seed))
            target_present = not bool(features.binary_row.get(_NO_TARGET_KEY.format(drug=drug), 0))
            preds.append(
                DrugPrediction(
                    drug=drug,
                    calibrated_prob_resistant=prob,
                    target_present=target_present,
                    top_features=self._top_features(tier, hits, seed),
                    ood_score=ood,
                )
            )
        return preds


# --------------------------------------------------------------------------- #
# Scripted predictor (for hand-authored demo scenarios)
# --------------------------------------------------------------------------- #
class ScriptedPredictor:
    """Returns pre-authored predictions per genome_id (used by demo cases)."""

    def __init__(
        self,
        predictions_by_genome: dict[str, list[DrugPrediction]],
        covered_drugs: list[str],
        covered_species: list[str],
    ):
        self._by_genome = predictions_by_genome
        self._drugs = covered_drugs
        self._species = covered_species

    def covered_drugs(self) -> list[str]:
        return list(self._drugs)

    def covered_species(self) -> list[str]:
        return list(self._species)

    def predict(self, features: FeatureBundle) -> list[DrugPrediction]:
        if features.genome_id not in self._by_genome:
            raise KeyError(f"no scripted predictions for genome '{features.genome_id}'")
        return list(self._by_genome[features.genome_id])


# --------------------------------------------------------------------------- #
# Held-out set for the evaluation panel
# --------------------------------------------------------------------------- #
_SEEN_GROUPS = ["ST131", "ST95", "ST69"]
_UNSEEN_GROUPS = ["ST_novelA", "ST_novelB"]
_PREVALENCE = {
    "Ciprofloxacin": 0.45,
    "Ampicillin": 0.6,
    "Gentamicin": 0.3,
    "Trimethoprim-sulfamethoxazole": 0.4,
    "Colistin": 0.12,
}


def _features_for_labels(
    genome_id: str, labels: dict[str, bool], seen: bool, rng: random.Random
) -> FeatureBundle:
    long_hits: list[LongHit] = []
    binary_row: dict[str, int] = {}
    for drug, resistant in labels.items():
        classes = list(classes_for(drug))
        if not classes:
            continue
        cls = rng.choice(classes)
        pool = GENE_POOL.get(cls, [])
        if not pool:
            continue
        sym, sub, method = rng.choice(pool)
        if resistant:
            # Usually the causal gene is present (curated); sometimes missed
            # (label noise), sometimes only a weak partial hit.
            r = rng.random()
            if r < 0.75:
                long_hits.append(_make_hit(sym, cls, sub, method))
                binary_row[sym] = 1
            elif r < 0.9:
                long_hits.append(_make_hit(sym, cls, sub, "PARTIALX"))
                binary_row[sym] = 1
            # else: resistant but no detectable gene (model will likely miss)
        else:
            # Usually clean; occasionally a spurious partial hit.
            if rng.random() < 0.1:
                long_hits.append(_make_hit(sym, cls, sub, "PARTIALX"))
                binary_row[sym] = 1

    # Colistin intrinsic-resistance demonstration: a few genomes lose the target.
    if labels.get("Colistin") and rng.random() < 0.5:
        binary_row[_NO_TARGET_KEY.format(drug="Colistin")] = 1

    # Unseen genetic groups carry novel accessory genes -> higher OOD.
    if not seen:
        for j in range(rng.randint(2, 4)):
            binary_row[f"novel_gene_{genome_id}_{j}"] = 1

    return FeatureBundle(genome_id=genome_id, binary_row=binary_row, long_hits=long_hits)


def build_held_out_set(n: int = 60, seed: int = 20260718) -> list[HeldOutGenome]:
    """Deterministic labeled held-out set spanning seen and unseen genetic groups."""
    rng = random.Random(seed)
    genomes: list[HeldOutGenome] = []
    for i in range(n):
        seen = rng.random() < 0.8
        group = rng.choice(_SEEN_GROUPS if seen else _UNSEEN_GROUPS)
        gid = f"HO_{i:03d}_{group}"
        labels = {drug: (rng.random() < _PREVALENCE[drug]) for drug in COVERED_DRUGS}
        features = _features_for_labels(gid, labels, seen, rng)
        genomes.append(
            HeldOutGenome(
                genome_id=gid,
                genetic_group=group,
                seen_in_training=seen,
                features=features,
                true_labels=labels,
            )
        )
    return genomes


# --------------------------------------------------------------------------- #
# Crafted demo cases (one per decision branch)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DemoCase:
    name: str
    description: str
    species: str
    features: FeatureBundle
    predictions: list[DrugPrediction]


def _low_preds(exclude: set[str] | None = None) -> dict[str, DrugPrediction]:
    """Susceptible/clean predictions for all covered drugs (label 'likely work')."""
    exclude = exclude or set()
    out = {}
    for d in COVERED_DRUGS:
        if d in exclude:
            continue
        out[d] = DrugPrediction(
            drug=d,
            calibrated_prob_resistant=0.08,
            target_present=True,
            top_features=[TopFeature("absent_marker", -0.2, is_known_mechanism=True)],
            ood_score=0.1,
        )
    return out


def demo_cases() -> list[DemoCase]:
    cases: list[DemoCase] = []

    # 1. Known mechanism -> likely to fail (evidence i) + susceptible others.
    feats = FeatureBundle(
        "demo_known_mechanism",
        {"gyrA_S83L": 1, "blaTEM-1": 1},
        [
            _make_hit("gyrA_S83L", "QUINOLONE", "point_mutation", "POINTX"),
            _make_hit("blaTEM-1", "BETA-LACTAM", "acquired_gene", "EXACTX"),
        ],
    )
    preds = _low_preds({"Ciprofloxacin", "Ampicillin"})
    preds["Ciprofloxacin"] = DrugPrediction("Ciprofloxacin", 0.95, True, [TopFeature("gyrA_S83L", 0.5, True)], 0.12)
    preds["Ampicillin"] = DrugPrediction("Ampicillin", 0.93, True, [TopFeature("blaTEM-1", 0.48, True)], 0.12)
    cases.append(DemoCase("Known mechanism (resistant)", "Exact gene/point-mutation hits drive a confident 'likely to fail' with category (i) evidence.", MOCK_SPECIES, feats, list(preds.values())))

    # 2. Clean susceptible -> likely to work (evidence iii).
    feats = FeatureBundle("demo_susceptible", {}, [])
    cases.append(DemoCase("Susceptible (clean)", "No resistance markers; every drug is 'likely to work', evidence (iii) driven by absence.", MOCK_SPECIES, feats, list(_low_preds().values())))

    # 3. Uncertainty band -> no-call.
    feats = FeatureBundle("demo_uncertainty", {}, [])
    preds = _low_preds({"Gentamicin"})
    preds["Gentamicin"] = DrugPrediction("Gentamicin", 0.52, True, [TopFeature("kmer_42", 0.11, False)], 0.15)
    cases.append(DemoCase("Uncertainty band (no-call)", "Gentamicin probability 0.52 sits in the uncertainty band around 0.5 -> no-call.", MOCK_SPECIES, feats, list(preds.values())))

    # 4. Out-of-distribution -> no-call.
    feats = FeatureBundle("demo_ood", {"novel_gene_1": 1, "novel_gene_2": 1}, [])
    preds = {d: DrugPrediction(d, 0.85, True, [TopFeature("kmer_7", 0.2, False)], 0.92) for d in COVERED_DRUGS}
    cases.append(DemoCase("Out-of-distribution (no-call)", "High OOD score (0.92) -> every drug is a no-call regardless of probability.", MOCK_SPECIES, feats, list(preds.values())))

    # 5. Conflict A: known mechanism present but model leans susceptible.
    feats = FeatureBundle("demo_conflict_a", {"gyrA_S83L": 1}, [_make_hit("gyrA_S83L", "QUINOLONE", "point_mutation", "POINTX")])
    preds = _low_preds({"Ciprofloxacin"})
    preds["Ciprofloxacin"] = DrugPrediction("Ciprofloxacin", 0.2, True, [TopFeature("gyrA_S83L", 0.4, True)], 0.15)
    cases.append(DemoCase("Conflict: mechanism vs model (no-call)", "A curated gyrA_S83L mechanism is present but the model leans susceptible (0.20) -> conflicting-evidence no-call.", MOCK_SPECIES, feats, list(preds.values())))

    # 6. Conflict B: model resistant but no resistance signal at all.
    feats = FeatureBundle("demo_conflict_b", {}, [])
    preds = _low_preds({"Trimethoprim-sulfamethoxazole"})
    preds["Trimethoprim-sulfamethoxazole"] = DrugPrediction("Trimethoprim-sulfamethoxazole", 0.85, True, [TopFeature("absent_marker", -0.1, True)], 0.15)
    cases.append(DemoCase("Conflict: model vs no-signal (no-call)", "The model leans resistant (0.85) yet no known resistance signal exists -> conflicting-evidence no-call.", MOCK_SPECIES, feats, list(preds.values())))

    # 7. Intrinsic resistance: target absent.
    feats = FeatureBundle("demo_intrinsic", {"mock_no_target__Colistin": 1}, [])
    preds = _low_preds({"Colistin"})
    preds["Colistin"] = DrugPrediction("Colistin", 0.05, False, [], 0.12)
    cases.append(DemoCase("Intrinsic (no molecular target)", "Colistin's molecular target is absent -> deterministic 'likely to fail'; never 'likely to work' on absent-marker basis.", MOCK_SPECIES, feats, list(preds.values())))

    # 8. Association-only -> likely to fail (evidence ii) with caveat.
    feats = FeatureBundle("demo_association", {"kmer_marker": 1}, [])
    preds = _low_preds({"Gentamicin"})
    preds["Gentamicin"] = DrugPrediction("Gentamicin", 0.82, True, [TopFeature("kmer_555", 0.35, False), TopFeature("kmer_777", 0.2, False)], 0.2)
    cases.append(DemoCase("Association-only (resistant)", "A resistant call driven by non-curated statistical features -> evidence (ii), labeled association not causation.", MOCK_SPECIES, feats, list(preds.values())))

    return cases


def scripted_predictor_for(cases: list[DemoCase]) -> ScriptedPredictor:
    return ScriptedPredictor(
        {c.features.genome_id: c.predictions for c in cases},
        covered_drugs=COVERED_DRUGS,
        covered_species=[MOCK_SPECIES],
    )
