#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np


def _parse_num_tasks(values: List[str]) -> List[int]:
    tasks: List[int] = []
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if part:
                tasks.append(int(part))
    if not tasks:
        raise ValueError("No valid --num-tasks provided.")
    return tasks


def _build_recipe(
    num_tasks: int,
    *,
    runtime_factor: Optional[float],
    input_file_size_factor: Optional[float],
    output_file_size_factor: Optional[float],
):
    try:
        from wfcommons.wfchef.recipes import SeismologyRecipe
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: wfcommons. Install with `pip install wfcommons`."
        ) from exc

    kwargs = {}
    if runtime_factor is not None:
        kwargs["runtime_factor"] = runtime_factor
    if input_file_size_factor is not None:
        kwargs["input_file_size_factor"] = input_file_size_factor
    if output_file_size_factor is not None:
        kwargs["output_file_size_factor"] = output_file_size_factor
    return SeismologyRecipe.from_num_tasks(num_tasks=num_tasks, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate WFCommons Seismology workflows with WfGen.",
    )
    parser.add_argument(
        "--num-tasks",
        nargs="+",
        default=["250"],
        help="Task counts (e.g. 250 or 100,250 500).",
    )
    parser.add_argument(
        "--num-workflows",
        type=int,
        default=1,
        help="Number of workflows to generate per task count.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/wfcommons_wfgen/seismology",
        help="Directory to write WfFormat JSON files.",
    )
    parser.add_argument(
        "--runtime-factor",
        type=float,
        default=None,
        help="Multiply task runtimes by this factor.",
    )
    parser.add_argument(
        "--input-file-size-factor",
        type=float,
        default=None,
        help="Multiply input file sizes by this factor.",
    )
    parser.add_argument(
        "--output-file-size-factor",
        type=float,
        default=None,
        help="Multiply output file sizes by this factor.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--write-dot",
        action="store_true",
        help="Also write .dot graph files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files if present.",
    )
    args = parser.parse_args()

    if args.num_workflows < 1:
        raise SystemExit("--num-workflows must be >= 1.")

    num_tasks_list = _parse_num_tasks(args.num_tasks)

    try:
        from wfcommons import WorkflowGenerator
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: wfcommons. Install with `pip install wfcommons`."
        ) from exc

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for num_tasks in num_tasks_list:
        recipe = _build_recipe(
            num_tasks,
            runtime_factor=args.runtime_factor,
            input_file_size_factor=args.input_file_size_factor,
            output_file_size_factor=args.output_file_size_factor,
        )
        generator = WorkflowGenerator(recipe)
        if args.num_workflows == 1:
            workflows = [generator.build_workflow()]
        else:
            workflows = generator.build_workflows(args.num_workflows)

        for idx, workflow in enumerate(workflows):
            out_path = output_dir / f"seismology-workflow-{num_tasks}-{idx}.json"
            if out_path.exists() and not args.overwrite:
                print(f"Skip existing: {out_path}")
                continue
            workflow.write_json(out_path)
            print(f"Wrote {out_path}")
            if args.write_dot:
                dot_path = out_path.with_suffix(".dot")
                workflow.write_dot(dot_path)
                print(f"Wrote {dot_path}")


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(1)
