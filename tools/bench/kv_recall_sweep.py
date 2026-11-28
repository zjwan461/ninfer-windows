#!/usr/bin/env python3
"""Measure which retention ratio keeps a fact that sits at a known depth.

The gate (`run_kv_recall_gate.py`) answers one question: at this budget, is the
answer byte-identical to the full view? That is not enough to choose a budget,
because the answer depends on where in the history the evidence sits — the same
67 % retention kept a five-marker layout and lost a four-marker one. This script
fills the matrix: for each budget, plant one marker per depth and record which
depths still come back verbatim.

It shells out to the gate twice per cell (the gate already serves the baseline
and the candidate back to back), so a full matrix costs
``budgets × depths × 2`` server runs. Nothing here parses model output itself;
the gate owns that, and this script only aggregates.

Usage::

    python tools/bench/kv_recall_sweep.py \
        --exe build-4080s/apps/ninfer-serve.exe \
        --artifact E:/models/qwen3_8_27b_a8.ninfer/qwen3_8_27b_a8.ninfer \
        --corpus eval/corpora/perplexity-1m/data/ninfer \
        --budgets 65536,131072,196608 --chars 700000

Writes ``matrix.json`` (raw verdicts) and prints a markdown table: rows are
budgets, columns are depths, a cell says ``ok`` when the marker at that depth
came back with its planted value and the whole answer matched.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
GATE = HERE / "run_kv_recall_gate.py"


def run_cell(exe: pathlib.Path, artifact: pathlib.Path, corpus: pathlib.Path,
             chars: int, budget: int, depths: list[float], extra: list[str]) -> dict:
    """Plant one marker per depth at this budget and report every marker's verdict."""
    verdicts = {}
    for depth in depths:
        command = [sys.executable, str(GATE),
                   "--exe", str(exe), "--artifact", str(artifact),
                   "--corpus", str(corpus), "--chars", str(chars),
                   "--budget", str(budget), "--depths", f"{depth}",
                   "--expect-codes", "1", *extra]
        completed = subprocess.run(command, capture_output=True, text=True, timeout=7200)
        verdicts[f"{depth:.2f}"] = {
            "returncode": completed.returncode,
            "passed": completed.returncode == 0,
            "tail": completed.stdout.strip().splitlines()[-1] if completed.stdout.strip() else "",
        }
        print(f"budget {budget} depth {depth:.2f}: "
              f"{'ok' if completed.returncode == 0 else 'FAIL'} — {verdicts[f'{depth:.2f}']['tail']}",
              file=sys.stderr)
    return verdicts


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exe", type=pathlib.Path, required=True)
    parser.add_argument("--artifact", type=pathlib.Path, required=True)
    parser.add_argument("--corpus", type=pathlib.Path, required=True)
    parser.add_argument("--chars", type=int, default=700000)
    parser.add_argument("--budgets", default="65536,131072,196608")
    parser.add_argument("--depths", default="0.1,0.3,0.5,0.7,0.9")
    parser.add_argument("--extra", default="--max-context 262144 --kv-dtype int8",
                        help="remaining gate flags, space separated")
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("matrix.json"))
    args = parser.parse_args(argv)

    budgets = [int(value) for value in args.budgets.split(",") if value.strip()]
    depths = [float(value) for value in args.depths.split(",") if value.strip()]
    extra = args.extra.split()

    matrix = {}
    for budget in budgets:
        matrix[str(budget)] = run_cell(args.exe, args.artifact, args.corpus, args.chars,
                                       budget, depths, extra)

    args.out.write_text(json.dumps(matrix, indent=2), encoding="utf-8")
    header = "| budget | " + " | ".join(f"{depth:.2f}" for depth in depths) + " |"
    rule = "|" + "---|" * (len(depths) + 1)
    print(header)
    print(rule)
    for budget in budgets:
        cells = []
        for depth in depths:
            cells.append("ok" if matrix[str(budget)][f"{depth:.2f}"]["passed"] else "**wrong**")
        print(f"| {budget} | " + " | ".join(cells) + " |")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())