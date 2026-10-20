"""Reproduce the GARCH analysis and save numerical results."""

import argparse
import importlib.metadata
import json
from pathlib import Path
import sys

from src.tutorial1 import analyze, load_tutorial_data

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", default="NLNPE_TD_data.xlsx")
    parser.add_argument("--start", default="1981-01-01")
    parser.add_argument("--end", default="2026-10-09")
    parser.add_argument("--initialization", choices=["sample", "long-run"], default="sample")
    args = parser.parse_args()
    returns, provenance = load_tutorial_data(ROOT/args.workbook, args.start, args.end)
    result, _, _ = analyze(returns, provenance, args.initialization)
    result["environment"] = {p: importlib.metadata.version(p) for p in ["numpy", "pandas", "scipy", "arch", "statsmodels"]}
    result["environment"]["python"] = sys.version.split()[0]
    (ROOT/"results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Tutorial 1 A/B: {result['n']} weekly returns; arch benchmark matches = {result['benchmark_pass']}")
    print(f"GJR gamma = {result['gjr']['estimates']['gamma']:.6f}; robust Wald = {result['gjr']['gamma_t']:.3f}")
    print(f"Numerical results: {ROOT/'results.json'}")


if __name__ == "__main__":
    main()
