from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import run_socket_scan


def test_build_socket_env_normalizes_supported_token_names() -> None:
  env = run_socket_scan.build_socket_env({"SOCKET_API_TOKEN": "socket-token"})

  assert env["SOCKET_CLI_API_TOKEN"] == "socket-token"
  assert env["SOCKET_API_KEY"] == "socket-token"
  assert env["SOCKET_API_TOKEN"] == "socket-token"


def test_resolve_repo_prefers_github_repository() -> None:
  assert run_socket_scan.resolve_repo({"GITHUB_REPOSITORY": "owner/repo"}) == "repo"


def test_resolve_repo_parses_github_origin(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setattr(run_socket_scan, "capture", lambda *_args, **_kwargs: "git@github.com:owner/repo.git")

  assert run_socket_scan.resolve_repo({}) == "repo"


def test_parse_github_repo_from_origin_handles_https_and_rejects_non_github_hosts() -> None:
  assert run_socket_scan.parse_github_repo_from_origin("https://github.com/owner/repo") == "repo"
  assert run_socket_scan.parse_github_repo_from_origin("ssh://git@github.com/owner/repo") == "repo"
  assert run_socket_scan.parse_github_repo_from_origin("https://example.test/github.com/owner/repo") is None
  assert run_socket_scan.parse_github_repo_from_origin("https://github.com/owner") is None


def test_resolve_repo_falls_back_to_project_name(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setattr(run_socket_scan, "capture", lambda *_args, **_kwargs: "")

  assert run_socket_scan.resolve_repo({}) == run_socket_scan.PROJECT_ROOT.name


def test_resolve_socket_org_prefers_environment_over_repo_default() -> None:
  assert run_socket_scan.resolve_socket_org({"SOCKET_ORG": "custom-org"}) == "custom-org"
  assert run_socket_scan.resolve_socket_org({"SOCKET_DEFAULT_ORG": "default-org"}) == "default-org"
  assert run_socket_scan.resolve_socket_org({}) == "donadio-solutions"


def test_resolve_branch_prefers_pull_request_head_ref(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setattr(run_socket_scan, "capture", lambda *_args, **_kwargs: "local-branch")

  assert (
    run_socket_scan.resolve_branch(
      {
        "GITHUB_HEAD_REF": "feature-branch",
        "GITHUB_REF_NAME": "3/merge",
      }
    )
    == "feature-branch"
  )
  assert run_socket_scan.resolve_branch({"GITHUB_REF_NAME": "main"}) == "main"
  assert run_socket_scan.resolve_branch({}) == "local-branch"


def test_resolve_commit_hash_prefers_github_sha(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setattr(run_socket_scan, "capture", lambda *_args, **_kwargs: "local-sha")

  assert run_socket_scan.resolve_commit_hash({"GITHUB_SHA": "github-sha"}) == "github-sha"
  assert run_socket_scan.resolve_commit_hash({}) == "local-sha"


def test_resolves_event_metadata(tmp_path: Path) -> None:
  event_path = tmp_path / "event.json"
  event_path.write_text(
    json.dumps(
      {
        "number": 12,
        "pull_request": {"number": 34},
        "repository": {"default_branch": "main"},
      }
    ),
    encoding="utf-8",
  )
  env = {"GITHUB_EVENT_PATH": str(event_path)}

  assert run_socket_scan.resolve_default_branch(env) == "main"
  assert run_socket_scan.resolve_pull_request_number(env) == "34"


def test_event_metadata_handles_missing_or_invalid_files(tmp_path: Path) -> None:
  invalid_path = tmp_path / "invalid.json"
  invalid_path.write_text("{bad", encoding="utf-8")

  assert run_socket_scan.resolve_default_branch({"GITHUB_EVENT_PATH": str(tmp_path / "missing.json")}) is None
  assert run_socket_scan.resolve_pull_request_number({"GITHUB_EVENT_PATH": str(invalid_path)}) is None


def test_build_scan_command_includes_github_metadata(tmp_path: Path) -> None:
  event_path = tmp_path / "event.json"
  event_path.write_text(
    json.dumps(
      {
        "number": 5,
        "repository": {"default_branch": "main"},
      }
    ),
    encoding="utf-8",
  )

  command = run_socket_scan.build_scan_command(
    socket_bin="socket",
    env={
      "GITHUB_REPOSITORY": "owner/repo",
      "GITHUB_REF_NAME": "main",
      "GITHUB_SHA": "abc123",
      "GITHUB_EVENT_PATH": str(event_path),
    },
    targets=[Path("pyproject.toml"), Path("uv.lock")],
  )

  assert command == [
    "socket",
    "scan",
    "create",
    "--read-only",
    "--tmp",
    "--no-interactive",
    "--no-banner",
    "--no-spinner",
    "--repo",
    "repo",
    "--org",
    "donadio-solutions",
    "--branch",
    "main",
    "--commit-hash",
    "abc123",
    "--pull-request",
    "5",
    "--default-branch",
    Path("pyproject.toml"),
    Path("uv.lock"),
  ]


def test_existing_source_manifests_returns_present_files(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
  present = tmp_path / "pyproject.toml"
  missing = tmp_path / "uv.lock"
  present.write_text("[project]\n", encoding="utf-8")
  monkeypatch.setattr(run_socket_scan, "SOURCE_MANIFESTS", (present, missing))

  assert run_socket_scan.existing_source_manifests() == [present]


def test_main_requires_socket_token(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
  monkeypatch.setattr(run_socket_scan, "build_socket_env", lambda: {})

  assert run_socket_scan.main() == 2
  assert "SOCKET_API_KEY is required" in capsys.readouterr().err
