#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import sys
from pathlib import Path


def main() -> int:
  if len(sys.argv) != 2:
    print("usage: check_coverage.py <coverage-json-path>", file=sys.stderr)
    return 2

  coverage_path = Path(sys.argv[1])
  payload = json.loads(coverage_path.read_text())
  totals = payload["totals"]

  line_coverage = float(totals.get("percent_covered", 0.0))
  branch_total = int(totals.get("num_branches", 0))
  branch_covered = int(totals.get("covered_branches", 0))
  branch_coverage = 100.0 if branch_total == 0 else (branch_covered / branch_total) * 100.0
  missing_lines = int(totals.get("missing_lines", 0))
  missing_branches = int(totals.get("missing_branches", 0))

  if (
    missing_lines
    or missing_branches
    or not math.isclose(line_coverage, 100.0)
    or not math.isclose(branch_coverage, 100.0)
  ):
    print(
      "coverage check failed: "
      f"line_coverage={line_coverage:.2f}, branch_coverage={branch_coverage:.2f}, "
      f"missing_lines={missing_lines}, missing_branches={missing_branches}",
      file=sys.stderr,
    )
    return 1

  print("coverage check passed: line and branch coverage are 100%")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
