#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOCKET_BIN = PROJECT_ROOT / "node_modules" / ".bin" / "socket"
SOCKET_TOKEN_ENV_NAMES = ("SOCKET_CLI_API_TOKEN", "SOCKET_API_KEY", "SOCKET_API_TOKEN")
DEFAULT_SOCKET_ORG = "donadio-solutions"
SOURCE_MANIFESTS = (
  PROJECT_ROOT / "pyproject.toml",
  PROJECT_ROOT / "uv.lock",
  PROJECT_ROOT / "package.json",
  PROJECT_ROOT / "package-lock.json",
)


def run(cmd: Sequence[str | Path], *, env: Mapping[str, str] | None = None) -> int:
  completed = subprocess.run([str(item) for item in cmd], cwd=PROJECT_ROOT, env=env)
  return completed.returncode


def capture(cmd: Sequence[str], *, env: Mapping[str, str] | None = None) -> str:
  completed = subprocess.run(
    list(cmd),
    cwd=PROJECT_ROOT,
    env=env,
    check=False,
    capture_output=True,
    text=True,
  )
  if completed.returncode != 0:
    return ""
  return completed.stdout.strip()


def resolve_socket_bin() -> str:
  if SOCKET_BIN.exists():
    return str(SOCKET_BIN)
  socket_bin = shutil.which("socket")
  if socket_bin is None:
    raise RuntimeError("Socket CLI was not found. Run npm ci or install socket.")
  return socket_bin


def resolve_socket_token(env: Mapping[str, str]) -> str | None:
  for name in SOCKET_TOKEN_ENV_NAMES:
    value = env.get(name, "").strip()
    if value:
      return value
  return None


def build_socket_env(base_env: Mapping[str, str] | None = None) -> dict[str, str]:
  env = dict(os.environ if base_env is None else base_env)
  token = resolve_socket_token(env)
  if token is not None:
    env["SOCKET_CLI_API_TOKEN"] = token
    env["SOCKET_API_KEY"] = token
    env["SOCKET_API_TOKEN"] = token
  return env


def resolve_repo(env: Mapping[str, str]) -> str:
  github_repository = env.get("GITHUB_REPOSITORY", "").strip()
  if github_repository:
    return github_repository.rsplit("/", maxsplit=1)[-1]

  origin_url = capture(["git", "remote", "get-url", "origin"], env=env)
  if origin_url.endswith(".git"):
    origin_url = origin_url[:-4]
  if origin_url.startswith("git@github.com:"):
    return origin_url.removeprefix("git@github.com:").rsplit("/", maxsplit=1)[-1]
  if "github.com/" in origin_url:
    return origin_url.rsplit("/", maxsplit=1)[-1]
  return PROJECT_ROOT.name


def resolve_socket_org(env: Mapping[str, str]) -> str:
  return env.get("SOCKET_ORG", "").strip() or env.get("SOCKET_DEFAULT_ORG", "").strip() or DEFAULT_SOCKET_ORG


def resolve_branch(env: Mapping[str, str]) -> str:
  for name in ("GITHUB_HEAD_REF", "GITHUB_REF_NAME"):
    value = env.get(name, "").strip()
    if value:
      return value
  return capture(["git", "branch", "--show-current"], env=env) or "unknown"


def resolve_commit_hash(env: Mapping[str, str]) -> str:
  return env.get("GITHUB_SHA", "").strip() or capture(["git", "rev-parse", "HEAD"], env=env)


def resolve_default_branch(env: Mapping[str, str]) -> str | None:
  event_path = env.get("GITHUB_EVENT_PATH", "").strip()
  if event_path:
    try:
      event = json.loads(Path(event_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
      event = {}
    repository = event.get("repository")
    if isinstance(repository, dict) and isinstance(repository.get("default_branch"), str):
      return repository["default_branch"]
  return env.get("GITHUB_DEFAULT_BRANCH", "").strip() or None


def resolve_pull_request_number(env: Mapping[str, str]) -> str | None:
  event_path = env.get("GITHUB_EVENT_PATH", "").strip()
  if not event_path:
    return None
  try:
    event = json.loads(Path(event_path).read_text(encoding="utf-8"))
  except (OSError, json.JSONDecodeError):
    return None
  pull_request = event.get("pull_request")
  if isinstance(pull_request, dict) and isinstance(pull_request.get("number"), int):
    return str(pull_request["number"])
  if isinstance(event.get("number"), int):
    return str(event["number"])
  return None


def generate_manifest(socket_bin: str, output_path: Path, ecosystem: str, env: Mapping[str, str]) -> int:
  return run(
    [
      socket_bin,
      "manifest",
      "cdxgen",
      "-t",
      ecosystem,
      "-o",
      output_path,
      "--no-recurse",
      ".",
    ],
    env=env,
  )


def build_scan_command(
  *,
  socket_bin: str,
  env: Mapping[str, str],
  targets: Sequence[Path],
) -> list[str | Path]:
  command: list[str | Path] = [
    socket_bin,
    "scan",
    "create",
    "--read-only",
    "--tmp",
    "--no-interactive",
    "--no-banner",
    "--no-spinner",
    "--repo",
    resolve_repo(env),
    "--org",
    resolve_socket_org(env),
    "--branch",
    resolve_branch(env),
  ]

  commit_hash = resolve_commit_hash(env)
  if commit_hash:
    command.extend(["--commit-hash", commit_hash])

  pull_request_number = resolve_pull_request_number(env)
  if pull_request_number is not None:
    command.extend(["--pull-request", pull_request_number])

  default_branch = resolve_default_branch(env)
  if default_branch is not None and default_branch == resolve_branch(env):
    command.append("--default-branch")

  command.extend(targets)
  return command


def existing_source_manifests() -> list[Path]:
  return [path for path in SOURCE_MANIFESTS if path.exists()]


def main() -> int:
  env = build_socket_env()
  if resolve_socket_token(env) is None:
    print(
      "SOCKET_API_KEY is required to run the Socket scan.",
      file=sys.stderr,
    )
    return 2

  try:
    socket_bin = resolve_socket_bin()
  except RuntimeError as exc:
    print(str(exc), file=sys.stderr)
    return 2

  with tempfile.TemporaryDirectory(prefix="openai-tests-socket-") as tmpdir:
    tmp_path = Path(tmpdir)
    generated_manifests = [
      tmp_path / "python.cdx.json",
      tmp_path / "javascript.cdx.json",
    ]
    for ecosystem, output_path in zip(("python", "js"), generated_manifests, strict=True):
      manifest_status = generate_manifest(socket_bin, output_path, ecosystem, env)
      if manifest_status != 0:
        return manifest_status

    targets = [*existing_source_manifests(), *generated_manifests]
    return run(build_scan_command(socket_bin=socket_bin, env=env, targets=targets), env=env)


if __name__ == "__main__":
  raise SystemExit(main())
