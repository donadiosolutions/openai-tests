from __future__ import annotations

from pathlib import Path

import pytest

from scripts import run_integration_tests


def test_parse_env_line_handles_comments_exports_and_quotes() -> None:
  assert run_integration_tests.parse_env_line("") is None
  assert run_integration_tests.parse_env_line("# comment") is None
  assert run_integration_tests.parse_env_line("not-an-assignment") is None
  assert run_integration_tests.parse_env_line("OPENAI_API_KEY=env-value # comment") == (
    "OPENAI_API_KEY",
    "env-value",
  )
  assert run_integration_tests.parse_env_line("export OPENAI_API_KEY='quoted # value'") == (
    "OPENAI_API_KEY",
    "quoted # value",
  )
  assert run_integration_tests.parse_env_line('OPENAI_API_KEY="quoted \\"value\\""') == (
    "OPENAI_API_KEY",
    'quoted "value"',
  )


def test_build_integration_env_prefers_dotenv_openai_api_key(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  (tmp_path / ".env").write_text(
    """
OPENAI_API_KEY=dotenv-key
OTHER=value
""".strip(),
    encoding="utf-8",
  )
  monkeypatch.setattr(run_integration_tests, "PROJECT_ROOT", tmp_path)

  env = run_integration_tests.build_integration_env({"OPENAI_API_KEY": "outer-key", "PATH": "/bin"})

  assert env["OPENAI_API_KEY"] == "dotenv-key"
  assert env["PATH"] == "/bin"


def test_build_integration_env_uses_environment_key_without_dotenv(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  monkeypatch.setattr(run_integration_tests, "PROJECT_ROOT", tmp_path)

  env = run_integration_tests.build_integration_env({"OPENAI_API_KEY": "outer-key"})

  assert env["OPENAI_API_KEY"] == "outer-key"
