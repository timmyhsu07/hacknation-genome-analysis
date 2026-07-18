"""End-to-end test: every crafted demo case (mock_pipeline.demo_cases) resolves
to the decision branch its name/description claims, run through the real
build_report orchestration (evidence categorization + decision rules together).
"""

from __future__ import annotations

import pytest

from decision_report.contracts import DecisionLabel, NoCallReason
from decision_report.mock_pipeline import demo_cases, scripted_predictor_for
from decision_report.report import build_report

# (case name) -> (drug the case is about, expected label, expected no-call reason or None)
EXPECTED = {
    "Known mechanism (resistant)": [
        ("Ciprofloxacin", DecisionLabel.LIKELY_TO_FAIL, None),
        ("Ampicillin", DecisionLabel.LIKELY_TO_FAIL, None),
    ],
    "Susceptible (clean)": [
        ("Ciprofloxacin", DecisionLabel.LIKELY_TO_WORK, None),
    ],
    "Uncertainty band (no-call)": [
        ("Gentamicin", DecisionLabel.NO_CALL, NoCallReason.UNCERTAINTY_BAND),
    ],
    "Out-of-distribution (no-call)": [
        ("Ciprofloxacin", DecisionLabel.NO_CALL, NoCallReason.OUT_OF_DISTRIBUTION),
    ],
    "Conflict: mechanism vs model (no-call)": [
        ("Ciprofloxacin", DecisionLabel.NO_CALL, NoCallReason.CONFLICTING_EVIDENCE),
    ],
    "Conflict: model vs no-signal (no-call)": [
        ("Trimethoprim-sulfamethoxazole", DecisionLabel.NO_CALL, NoCallReason.CONFLICTING_EVIDENCE),
    ],
    "Intrinsic (no molecular target)": [
        ("Colistin", DecisionLabel.LIKELY_TO_FAIL, None),
    ],
    "Association-only (resistant)": [
        ("Gentamicin", DecisionLabel.LIKELY_TO_FAIL, None),
    ],
}


@pytest.fixture
def reports_by_case_name(config):
    cases = demo_cases()
    predictor = scripted_predictor_for(cases)
    return {
        case.name: build_report(case.features, predictor, config, case.species) for case in cases
    }


def test_all_demo_cases_are_covered_by_the_expectation_table():
    assert {c.name for c in demo_cases()} == set(EXPECTED)


@pytest.mark.parametrize("case_name", list(EXPECTED))
def test_demo_case_resolves_to_expected_branch(reports_by_case_name, case_name):
    report = reports_by_case_name[case_name]
    for drug, label, reason in EXPECTED[case_name]:
        dec = next(d for d in report.decisions if d.drug == drug)
        assert dec.label is label, f"{case_name}/{drug}: expected {label}, got {dec.label}"
        assert dec.no_call_reason is reason


def test_intrinsic_case_is_flagged_as_intrinsic(reports_by_case_name):
    dec = next(d for d in reports_by_case_name["Intrinsic (no molecular target)"].decisions if d.drug == "Colistin")
    assert dec.intrinsic_resistance is True


def test_every_demo_report_carries_the_mandatory_disclaimer(reports_by_case_name):
    for report in reports_by_case_name.values():
        assert "DECISION SUPPORT ONLY" in report.disclaimer
