from __future__ import annotations

import pytest

from decision_report.config import DecisionConfig
from decision_report.mock_pipeline import MOCK_SPECIES


@pytest.fixture
def config() -> DecisionConfig:
    return DecisionConfig(covered_species=(MOCK_SPECIES,))
