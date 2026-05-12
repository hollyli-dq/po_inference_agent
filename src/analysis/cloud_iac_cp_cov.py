#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.cloud_iac_coverage import CloudIacCriticalPairCoverageAnalyzer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze Cloud-IaC IP-Cov over observed traces.")
    parser.add_argument(
        "--trace-source",
        choices=("expert", "model", "combined"),
        default="combined",
        help="Which trace source to include.",
    )
    parser.add_argument(
        "--include-failed",
        action="store_true",
        help="Include failed traces instead of filtering to success-only traces.",
    )
    parser.add_argument(
        "--keep-unknown-actions",
        action="store_true",
        help="Keep actions that are not part of the scenario task set.",
    )
    parser.add_argument(
        "--keep-duplicates",
        action="store_true",
        help="Keep repeated actions within a trace.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    analyzer = CloudIacCriticalPairCoverageAnalyzer(
        project_root=PROJECT_ROOT,
        trace_source=args.trace_source,
        only_success=not args.include_failed,
        drop_unknown_actions=not args.keep_unknown_actions,
        dedup_actions=not args.keep_duplicates,
    )
    results = analyzer.analyze()
    analyzer.print_report(results)


if __name__ == "__main__":
    main()
