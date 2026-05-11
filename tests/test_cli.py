from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pytest

from openai_tests.cli import build_parser, main


def test_parser_defaults_to_no_command() -> None:
  parser = build_parser()
  parsed = parser.parse_args([])
  assert parsed.command is None


def test_main_prints_help_without_a_subcommand(capsys: pytest.CaptureFixture[str]) -> None:
  assert main([]) == 0
  captured = capsys.readouterr()
  assert "usage:" in captured.out
  assert "modules" in captured.out
  assert "asr-simple" in captured.out
  assert "asr-wer" in captured.out
  assert "list-models" in captured.out
  assert "text-simple" in captured.out


def test_main_lists_registered_modules(capsys: pytest.CaptureFixture[str]) -> None:
  assert main(["modules"]) == 0
  captured = capsys.readouterr()
  assert "asr-simple" in captured.out
  assert "asr-wer" in captured.out
  assert "list-models" in captured.out
  assert "text-simple" in captured.out


def test_parser_accepts_text_simple_command() -> None:
  parser = build_parser()
  parsed = parser.parse_args(["text-simple"])
  assert parsed.command == "text-simple"


def test_parser_accepts_asr_simple_command() -> None:
  parser = build_parser()
  parsed = parser.parse_args(["asr-simple"])
  assert parsed.command == "asr-simple"


def test_parser_accepts_asr_wer_command(tmp_path: Path) -> None:
  parser = build_parser()
  parsed = parser.parse_args(["asr-wer", "ground", str(tmp_path)])
  assert parsed.command == "asr-wer"


def test_parser_accepts_list_models_command() -> None:
  parser = build_parser()
  parsed = parser.parse_args(["list-models"])
  assert parsed.command == "list-models"


def test_main_module_raises_system_exit(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setattr(sys, "argv", ["openai_tests"])
  with pytest.raises(SystemExit) as exc_info:
    runpy.run_module("openai_tests.__main__", run_name="__main__")
  assert exc_info.value.code == 0
