#!/usr/bin/env python3
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTEGRATION_DIR = PROJECT_ROOT / "tests" / "integration"
COMPOSE_CANDIDATES = (
  PROJECT_ROOT / "compose.yaml",
  PROJECT_ROOT / "compose.yml",
  INTEGRATION_DIR / "compose.yaml",
  INTEGRATION_DIR / "compose.yml",
)


def run(cmd: list[str]) -> int:
  completed = subprocess.run(cmd, cwd=PROJECT_ROOT)
  return completed.returncode


def find_compose_file() -> Path | None:
  for candidate in COMPOSE_CANDIDATES:
    if candidate.exists():
      return candidate
  return None


def has_integration_tests() -> bool:
  return any(path.name.startswith("test_") and path.suffix == ".py" for path in INTEGRATION_DIR.glob("*.py"))


def podman_compose_prefix() -> list[str]:
  if shutil.which("podman") is None:
    raise RuntimeError("podman is required for container-backed integration tests")
  return ["podman", "compose"]


def main() -> int:
  compose_file = find_compose_file()
  tests_present = has_integration_tests()

  if not compose_file and not tests_present:
    print("No integration suite is configured; skipping integration tests.")
    return 0

  compose_prefix = podman_compose_prefix() if compose_file else []

  if compose_file:
    up_cmd = [*compose_prefix, "-f", str(compose_file), "up", "-d"]
    down_cmd = [*compose_prefix, "-f", str(compose_file), "down", "--remove-orphans"]
    try:
      if run(up_cmd) != 0:
        return 1
      if tests_present:
        return run(
          [
            sys.executable,
            "-m",
            "pytest",
            "-m",
            "integration",
            "tests/integration",
            "--cov=src",
            "--cov-branch",
            "--cov-append",
            "--cov-report=",
          ]
        )
      print("Container setup exists but no integration tests are defined; skipping pytest execution.")
      return 0
    finally:
      run(down_cmd)

  return run(
    [
      sys.executable,
      "-m",
      "pytest",
      "-m",
      "integration",
      "tests/integration",
      "--cov=src",
      "--cov-branch",
      "--cov-append",
      "--cov-report=",
    ]
  )


if __name__ == "__main__":
  raise SystemExit(main())
