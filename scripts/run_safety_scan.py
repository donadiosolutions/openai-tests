#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import sys


def main() -> int:
  api_key = os.environ.get("SAFETY_API_KEY", "").strip()
  if not api_key:
    print(
      "SAFETY_API_KEY is required to run the Safety CLI scan.",
      file=sys.stderr,
    )
    return 2

  completed = subprocess.run(
    ["safety", "--key", api_key, "scan", "--target", "."],
    check=False,
  )
  return completed.returncode


if __name__ == "__main__":
  raise SystemExit(main())
