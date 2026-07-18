"""Drug -> AMR-class reference data (Module 3 domain reference, not a model).

Used to match Module 1 hits (which carry a ``drug_class``) to the drug a card is
about. This is static curation data, not anything trained or fitted.

Class names follow the AMRFinderPlus / Module 1 vocabulary.
"""

from __future__ import annotations

# drug -> the AMR class(es) whose determinants confer resistance to it.
DRUG_CATALOG: dict[str, frozenset[str]] = {
    "Ciprofloxacin": frozenset({"QUINOLONE"}),
    "Ampicillin": frozenset({"BETA-LACTAM"}),
    "Gentamicin": frozenset({"AMINOGLYCOSIDE"}),
    "Trimethoprim-sulfamethoxazole": frozenset({"TRIMETHOPRIM", "SULFONAMIDE"}),
    "Colistin": frozenset({"POLYMYXIN"}),
    # In the catalog but intentionally NOT covered by the mock predictor, to
    # demonstrate the drug-not-covered no-call.
    "Meropenem": frozenset({"BETA-LACTAM"}),
}


def classes_for(drug: str) -> set[str]:
    """AMR classes for a drug (empty set if unknown)."""
    return set(DRUG_CATALOG.get(drug, frozenset()))


# Representative determinants per class, for building mock features. Each entry:
# (element_symbol, element_subtype, default_method).
GENE_POOL: dict[str, list[tuple[str, str, str]]] = {
    "QUINOLONE": [
        ("gyrA_S83L", "point_mutation", "POINTX"),
        ("parC_S80I", "point_mutation", "POINTX"),
        ("qnrS1", "acquired_gene", "EXACTX"),
    ],
    "BETA-LACTAM": [
        ("blaTEM-1", "acquired_gene", "EXACTX"),
        ("blaCTX-M-15", "acquired_gene", "EXACTX"),
    ],
    "AMINOGLYCOSIDE": [
        ("aac(3)-IIa", "acquired_gene", "EXACTX"),
        ("aph(3')-Ia", "acquired_gene", "EXACTX"),
    ],
    "TRIMETHOPRIM": [("dfrA1", "acquired_gene", "EXACTX")],
    "SULFONAMIDE": [("sul1", "acquired_gene", "EXACTX"), ("sul2", "acquired_gene", "EXACTX")],
    "POLYMYXIN": [("mcr-1.1", "acquired_gene", "EXACTX")],
}

# The set of determinant symbols the mock predictor considers "familiar" from
# training. Genomes carrying many symbols outside this set look novel (high OOD).
TRAINING_VOCAB: frozenset[str] = frozenset(
    sym for entries in GENE_POOL.values() for (sym, _sub, _m) in entries
)
