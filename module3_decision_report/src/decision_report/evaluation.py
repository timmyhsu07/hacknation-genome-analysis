"""Held-out evaluation panel (Module 3).

Runs the full decision pipeline over a labeled held-out set and REPORTS (never
tunes) performance: per-drug AUROC / PR-AUC / Brier / F1, balanced accuracy with
resistant and susceptible recall reported SEPARATELY, the no-call rate together
with accuracy on the remaining (called) predictions, a reliability curve, and a
breakdown by genetic group including groups unseen in training.

Probability-based metrics (AUROC/PR-AUC/Brier/reliability) use the calibrated
model probability and EXCLUDE deterministic intrinsic-resistance calls (target
absent), where the probability is not the operative signal. Discrete-label
metrics are computed over non-no-call decisions only.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from . import metrics as M
from .config import DecisionConfig
from .contracts import DecisionLabel, HeldOutGenome, Predictor
from .report import build_report


@dataclass(frozen=True)
class EvaluationResult:
    records: pd.DataFrame
    per_drug: pd.DataFrame
    per_group: pd.DataFrame
    overall: dict
    reliability: list[M.ReliabilityBin]


def _predicted_class(label: DecisionLabel) -> bool | None:
    if label is DecisionLabel.LIKELY_TO_FAIL:
        return True  # predicted resistant
    if label is DecisionLabel.LIKELY_TO_WORK:
        return False  # predicted susceptible
    return None  # no-call


def _build_records(
    held_out: list[HeldOutGenome], predictor: Predictor, config: DecisionConfig, species: str
) -> pd.DataFrame:
    rows = []
    for genome in held_out:
        report = build_report(genome.features, predictor, config, species)
        for dec in report.decisions:
            if dec.drug not in genome.true_labels:
                continue
            rows.append(
                {
                    "genome_id": genome.genome_id,
                    "genetic_group": genome.genetic_group,
                    "seen_in_training": genome.seen_in_training,
                    "drug": dec.drug,
                    "y_true": bool(genome.true_labels[dec.drug]),
                    "prob": dec.calibrated_prob_resistant,
                    "label": dec.label.value,
                    "predicted_class": _predicted_class(dec.label),
                    "is_no_call": dec.is_no_call,
                    "intrinsic": dec.intrinsic_resistance,
                }
            )
    return pd.DataFrame(rows)


def _prob_slice(df: pd.DataFrame) -> pd.DataFrame:
    """Rows usable for probability metrics: valid prob, not intrinsic override."""
    return df[(~df["intrinsic"]) & df["prob"].notna()]


def _called_slice(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["predicted_class"].notna()]


def _per_drug(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for drug, g in df.groupby("drug", sort=True):
        prob_g = _prob_slice(g)
        called = _called_slice(g)
        cm = M.classification_metrics(called["y_true"], called["predicted_class"]) if len(called) else M.classification_metrics([], [])
        acc_called = cm.accuracy
        rows.append(
            {
                "drug": drug,
                "n": len(g),
                "n_resistant": int(g["y_true"].sum()),
                "n_susceptible": int((~g["y_true"]).sum()),
                "auroc": M.auroc(prob_g["prob"], prob_g["y_true"]) if len(prob_g) else None,
                "pr_auc": M.average_precision(prob_g["prob"], prob_g["y_true"]) if len(prob_g) else None,
                "brier": M.brier_score(prob_g["prob"], prob_g["y_true"]) if len(prob_g) else None,
                "balanced_accuracy": cm.balanced_accuracy,
                "recall_resistant": cm.recall_resistant,
                "recall_susceptible": cm.recall_susceptible,
                "f1_resistant": cm.f1_resistant,
                "no_call_rate": float((g["is_no_call"]).mean()),
                "accuracy_on_called": acc_called,
                "n_called": len(called),
            }
        )
    return pd.DataFrame(rows)


def _per_group(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group, g in df.groupby("genetic_group", sort=True):
        called = _called_slice(g)
        cm = M.classification_metrics(called["y_true"], called["predicted_class"]) if len(called) else M.classification_metrics([], [])
        rows.append(
            {
                "genetic_group": group,
                "seen_in_training": bool(g["seen_in_training"].iloc[0]),
                "n": len(g),
                "no_call_rate": float(g["is_no_call"].mean()),
                "accuracy_on_called": cm.accuracy,
                "balanced_accuracy": cm.balanced_accuracy,
            }
        )
    out = pd.DataFrame(rows)
    # Show unseen groups last, flagged, so generalization gaps stand out.
    return out.sort_values(["seen_in_training", "genetic_group"], ascending=[False, True]).reset_index(drop=True)


def _overall(df: pd.DataFrame, config: DecisionConfig) -> tuple[dict, list[M.ReliabilityBin]]:
    called = _called_slice(df)
    prob_g = _prob_slice(df)
    cm = M.classification_metrics(called["y_true"], called["predicted_class"]) if len(called) else M.classification_metrics([], [])
    reliability = M.reliability_curve(prob_g["prob"], prob_g["y_true"], config.reliability_bins) if len(prob_g) else []
    overall = {
        "n_predictions": len(df),
        "balanced_accuracy": cm.balanced_accuracy,
        "recall_resistant": cm.recall_resistant,
        "recall_susceptible": cm.recall_susceptible,
        "f1_resistant": cm.f1_resistant,
        "brier": M.brier_score(prob_g["prob"], prob_g["y_true"]) if len(prob_g) else None,
        "no_call_rate": float(df["is_no_call"].mean()) if len(df) else None,
        "accuracy_on_called": cm.accuracy,
        "n_called": len(called),
    }
    return overall, reliability


def run_evaluation(
    held_out: list[HeldOutGenome],
    predictor: Predictor,
    config: DecisionConfig,
    species: str,
) -> EvaluationResult:
    """Run the pipeline over the held-out set and assemble the evaluation panel."""
    df = _build_records(held_out, predictor, config, species)
    if df.empty:
        empty = pd.DataFrame()
        return EvaluationResult(df, empty, empty, {}, [])
    per_drug = _per_drug(df)
    per_group = _per_group(df)
    overall, reliability = _overall(df, config)
    return EvaluationResult(df, per_drug, per_group, overall, reliability)
