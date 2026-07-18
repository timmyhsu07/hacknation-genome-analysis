"""Command-line entrypoint for Module 3 (The Decision Report).

This module ships mock implementations of Module 1 (feature extraction) and
Module 2 (prediction) so it runs standalone with zero real artifacts -- see
:mod:`decision_report.mock_pipeline`. Point a real deployment's own Module 1/2
adapters at :func:`decision_report.report.build_report` instead of the mocks
used here.

Subcommands:
  demo      run the crafted demo cases (one per decision branch) and print/
            optionally save each genome's report.
  evaluate  build the synthetic held-out panel, run the full pipeline over it,
            and print/optionally save the evaluation metrics.
  report    build a single report from a FASTA file using the mock feature
            extractor + mock predictor.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from enum import Enum
from pathlib import Path
from typing import Any

from .config import ConfigError, DecisionConfig, load_config
from .contracts import DecisionReportError, GenomeReport
from .evaluation import run_evaluation
from .mock_pipeline import (
    MOCK_SPECIES,
    MockFeatureExtractor,
    MockPredictor,
    build_held_out_set,
    demo_cases,
    scripted_predictor_for,
)
from .report import build_report, report_from_fasta


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Enum):
        return obj.value
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    raise TypeError(f"not JSON serializable: {type(obj)!r}")


def _dump(obj: Any) -> str:
    return json.dumps(obj, default=_json_default, indent=2)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _report_summary(report: GenomeReport) -> str:
    lines = [f"{report.genome_id}  ({report.species}, supported={report.species_supported})"]
    for err in report.errors:
        lines.append(f"  ! {err}")
    for dec in report.decisions:
        reason = f" [{dec.no_call_reason.value}]" if dec.no_call_reason else ""
        lines.append(f"  - {dec.drug}: {dec.label.value}{reason}  (evidence: {dec.evidence_category.value})")
    return "\n".join(lines)


def _cmd_demo(args: argparse.Namespace) -> int:
    config = load_config(args.config) if args.config else DecisionConfig()
    cases = demo_cases()
    predictor = scripted_predictor_for(cases)
    out_dir = Path(args.output_dir) if args.output_dir else None
    for case in cases:
        report = build_report(case.features, predictor, config, case.species)
        print(f"=== {case.name} ===\n{case.description}")
        print(_report_summary(report))
        print()
        if out_dir is not None:
            _write(out_dir / f"{report.genome_id}.json", _dump(report))
    if out_dir is not None:
        print(f"Wrote {len(cases)} report(s) to {out_dir}")
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    config = load_config(args.config) if args.config else DecisionConfig()
    held_out = build_held_out_set(n=args.n, seed=args.seed)
    predictor = MockPredictor()
    result = run_evaluation(held_out, predictor, config, species=MOCK_SPECIES)
    print("Overall:")
    print(_dump(result.overall))
    print("\nPer-drug:")
    print(result.per_drug.to_string(index=False))
    print("\nPer-group:")
    print(result.per_group.to_string(index=False))
    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        _write(out_dir / "overall.json", _dump(result.overall))
        result.per_drug.to_csv(out_dir / "per_drug.csv", index=False)
        result.per_group.to_csv(out_dir / "per_group.csv", index=False)
        _write(out_dir / "reliability.json", _dump(result.reliability))
        print(f"\nWrote evaluation artifacts to {out_dir}")
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    config = load_config(args.config) if args.config else DecisionConfig()
    report = report_from_fasta(
        args.fasta,
        MockFeatureExtractor(),
        MockPredictor(),
        config,
        species=args.species,
    )
    text = _dump(report)
    if args.output:
        _write(Path(args.output), text)
        print(f"Wrote report to {args.output}")
    else:
        print(text)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="decision-report",
        description="Module 3 - The Decision Report (mock-pipeline demo CLI).",
    )
    p.add_argument("--config", default=None, help="Path to a YAML/JSON DecisionConfig file.")
    sub = p.add_subparsers(dest="command", required=True)

    demo_p = sub.add_parser("demo", help="Run the crafted demo cases (one per decision branch).")
    demo_p.add_argument("--output-dir", default=None, help="Write each report as JSON here.")
    demo_p.set_defaults(func=_cmd_demo)

    eval_p = sub.add_parser("evaluate", help="Run the held-out evaluation panel.")
    eval_p.add_argument("--n", type=int, default=60, help="Number of held-out genomes.")
    eval_p.add_argument("--seed", type=int, default=20260718, help="RNG seed for the held-out set.")
    eval_p.add_argument("--output-dir", default=None, help="Write metrics artifacts here.")
    eval_p.set_defaults(func=_cmd_evaluate)

    report_p = sub.add_parser("report", help="Build a report for one FASTA file (mock features + predictor).")
    report_p.add_argument("--fasta", required=True, help="Path to a genome FASTA file.")
    report_p.add_argument("--species", default=MOCK_SPECIES, help="Species to report against.")
    report_p.add_argument("--output", default=None, help="Write the report JSON here instead of stdout.")
    report_p.set_defaults(func=_cmd_report)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except ConfigError as exc:
        print(f"ERROR (config): {exc}", file=sys.stderr)
        return 2
    except DecisionReportError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
