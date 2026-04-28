#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTEGRATION_DIR = PROJECT_ROOT / "tests" / "integration"
COMPOSE_CANDIDATES = (
  PROJECT_ROOT / "compose.yaml",
  PROJECT_ROOT / "compose.yml",
  INTEGRATION_DIR / "compose.yaml",
  INTEGRATION_DIR / "compose.yml",
)


def run(cmd: list[str], *, env: Mapping[str, str] | None = None) -> int:
  completed = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env)
  return completed.returncode


def load_env_file(path: Path) -> dict[str, str]:
  if not path.exists():
    return {}
  values: dict[str, str] = {}
  for line in path.read_text(encoding="utf-8").splitlines():
    parsed = parse_env_line(line)
    if parsed is not None:
      key, value = parsed
      values[key] = value
  return values


def parse_env_line(line: str) -> tuple[str, str] | None:
  stripped = line.strip()
  if not stripped or stripped.startswith("#"):
    return None
  if stripped.startswith("export "):
    stripped = stripped.removeprefix("export ").lstrip()
  key, separator, raw_value = stripped.partition("=")
  if not separator:
    return None
  key = key.strip()
  if not key:
    return None
  return key, parse_env_value(raw_value)


def parse_env_value(raw_value: str) -> str:
  value = raw_value.strip()
  if not value:
    return ""
  if value[0] in ("'", '"'):
    return parse_quoted_env_value(value)
  return strip_unquoted_comment(value).strip()


def parse_quoted_env_value(value: str) -> str:
  quote = value[0]
  chars: list[str] = []
  escaped = False
  for char in value[1:]:
    if escaped:
      chars.append(char)
      escaped = False
    elif quote == '"' and char == "\\":
      escaped = True
    elif char == quote:
      return "".join(chars)
    else:
      chars.append(char)
  return "".join(chars)


def strip_unquoted_comment(value: str) -> str:
  for index, char in enumerate(value):
    if char == "#" and (index == 0 or value[index - 1].isspace()):
      return value[:index]
  return value


def build_integration_env(base_env: Mapping[str, str] | None = None) -> dict[str, str]:
  env = dict(os.environ if base_env is None else base_env)
  dotenv_api_key = load_env_file(PROJECT_ROOT / ".env").get("OPENAI_API_KEY")
  if dotenv_api_key:
    env["OPENAI_API_KEY"] = dotenv_api_key
  return env


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
  integration_env = build_integration_env()

  if not compose_file and not tests_present:
    print("No integration suite is configured; skipping integration tests.")
    return 0

  compose_prefix = podman_compose_prefix() if compose_file else []

  if compose_file:
    up_cmd = [*compose_prefix, "-f", str(compose_file), "up", "-d"]
    down_cmd = [*compose_prefix, "-f", str(compose_file), "down", "--remove-orphans"]
    try:
      if run(up_cmd, env=integration_env) != 0:
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
          ],
          env=integration_env,
        )
      print("Container setup exists but no integration tests are defined; skipping pytest execution.")
      return 0
    finally:
      run(down_cmd, env=integration_env)

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
    ],
    env=integration_env,
  )


if __name__ == "__main__":
  raise SystemExit(main())
