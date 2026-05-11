from __future__ import annotations

import argparse
import json
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from openai_tests.test_modules import asr_simple, asr_wer


def build_args(*raw_args: str, **overrides: object) -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  asr_wer.configure_parser(parser)
  args = parser.parse_args([*raw_args])
  for key, value in overrides.items():
    setattr(args, key, value)
  return args


def write_audio(path: Path) -> Path:
  path.write_bytes(b"RIFF")
  return path


def test_parser_accepts_batch_endpoint_prompt_service_tier_and_endpoint_options(tmp_path: Path) -> None:
  args = build_args(
    "eval",
    str(tmp_path),
    "--endpoint",
    "completions",
    "--batch",
    "3",
    "--prompt",
    "Use product names.",
    "--service-tier",
    "priority",
    "--completions-temperature",
    "0.2",
    "--transcriptions-language",
    "en",
    "--prep",
    "--overlap",
    "3.5",
  )

  assert args.mode == "eval"
  assert args.audio_dir == str(tmp_path)
  assert args.endpoint == "completions"
  assert args.batch == 3
  assert args.prompt == "Use product names."
  assert args.service_tier == "priority"
  assert args.completions_temperature == 0.2
  assert args.transcriptions_language == "en"
  assert args.prep is True
  assert args.overlap == 3.5


def test_configuration_errors_cover_batch_audio_discovery_and_prompt_conflicts(tmp_path: Path) -> None:
  with pytest.raises(ValueError, match="batch must be at least 1"):
    asr_wer.validate_args(build_args("ground", str(tmp_path), "--batch", "0"))

  with pytest.raises(ValueError, match="overlap can only be used"):
    asr_wer.validate_args(build_args("ground", str(tmp_path), "--overlap", "3"))

  missing_dir = tmp_path / "missing"
  with pytest.raises(ValueError, match="Audio directory does not exist"):
    asr_wer.discover_audio_files(missing_dir)

  empty_dir = tmp_path / "empty"
  empty_dir.mkdir()
  with pytest.raises(ValueError, match="No supported audio files"):
    asr_wer.discover_audio_files(empty_dir)

  with pytest.raises(ValueError, match="Audio path is not a directory"):
    asr_wer.discover_audio_files(write_audio(tmp_path / "not-a-dir.wav"))

  duplicate_dir = tmp_path / "dupes"
  duplicate_dir.mkdir()
  write_audio(duplicate_dir / "clip.wav")
  write_audio(duplicate_dir / "clip.mp3")
  with pytest.raises(ValueError, match="Duplicate audio file stem"):
    asr_wer.discover_audio_files(duplicate_dir)

  case_duplicate_dir = tmp_path / "case-dupes"
  case_duplicate_dir.mkdir()
  write_audio(case_duplicate_dir / "clip.wav")
  write_audio(case_duplicate_dir / "Clip.mp3")
  with pytest.raises(ValueError, match="Duplicate audio file stem"):
    asr_wer.discover_audio_files(case_duplicate_dir)

  collision_dir = tmp_path / "collisions"
  collision_dir.mkdir()
  write_audio(collision_dir / "clip.wav")
  write_audio(collision_dir / "clip_normalized.wav")
  with pytest.raises(ValueError, match="Output artifact collision"):
    asr_wer.discover_audio_files(collision_dir)

  case_collision_dir = tmp_path / "case-collisions"
  case_collision_dir.mkdir()
  write_audio(case_collision_dir / "Clip.wav")
  write_audio(case_collision_dir / "clip_normalized.wav")
  with pytest.raises(ValueError, match="Output artifact collision"):
    asr_wer.discover_audio_files(case_collision_dir)

  reserved_dir = tmp_path / "reserved"
  reserved_dir.mkdir()
  write_audio(reserved_dir / "report.wav")
  with pytest.raises(ValueError, match="reserved output artifact"):
    asr_wer.discover_audio_files(reserved_dir)

  with pytest.raises(ValueError, match="prompt cannot be provided with transcriptions-prompt"):
    asr_wer.validate_args(build_args("ground", str(tmp_path), "--prompt", "A", "--transcriptions-prompt", "B"))

  with pytest.raises(ValueError, match="prompt cannot be provided with transcriptions-prompt"):
    asr_wer.validate_args(build_args("ground", str(tmp_path), "--prompt", "", "--transcriptions-prompt", ""))

  with pytest.raises(ValueError, match="completions prompt flags cannot be used"):
    asr_wer.validate_args(build_args("ground", str(tmp_path), "--system-prompt", "ignored"))

  with pytest.raises(ValueError, match="prompt cannot be provided with completions prompt overrides"):
    asr_wer.validate_args(
      build_args("ground", str(tmp_path), "--endpoint", "completions", "--prompt", "A", "--user-prompt", "B")
    )

  with pytest.raises(ValueError, match="prompt cannot be provided with completions prompt overrides"):
    asr_wer.validate_args(
      build_args("ground", str(tmp_path), "--endpoint", "completions", "--prompt", "", "--user-prompt", "")
    )

  with pytest.raises(ValueError, match="completions-messages-json cannot be used"):
    asr_wer.validate_args(
      build_args("ground", str(tmp_path), "--endpoint", "completions", "--completions-messages-json", "[]")
    )

  with pytest.raises(ValueError, match="service-tier cannot be provided with completions-service-tier"):
    asr_wer.validate_args(
      build_args(
        "ground",
        str(tmp_path),
        "--endpoint",
        "completions",
        "--service-tier",
        "flex",
        "--completions-service-tier",
        "scale",
      )
    )

  with pytest.raises(ValueError, match="plain text"):
    asr_wer.validate_args(
      build_args(
        "ground",
        str(tmp_path),
        "--endpoint",
        "completions",
        "--completions-response-format-json",
        '{"type":"json_object"}',
      )
    )

  with pytest.raises(ValueError, match="plain text"):
    asr_wer.validate_completions_response_format_for_wer("json")
  asr_wer.validate_completions_response_format_for_wer({})
  asr_wer.validate_completions_response_format_for_wer({"type": "text"})

  with pytest.raises(ValueError, match="transcript-only"):
    asr_wer.validate_args(build_args("ground", str(tmp_path), "--transcriptions-response-format", "srt"))

  with pytest.raises(ValueError, match="transcript-only"):
    asr_wer.validate_args(build_args("ground", str(tmp_path), "--transcriptions-response-format", "diarized_json"))


def test_discover_audio_files_reports_unreadable_directories(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  audio_dir = tmp_path / "audio"
  audio_dir.mkdir()
  original_iterdir = Path.iterdir

  def fake_iterdir(path: Path):
    if path == audio_dir:
      raise PermissionError("denied")
    return original_iterdir(path)

  monkeypatch.setattr(Path, "iterdir", fake_iterdir)
  with pytest.raises(ValueError, match="Unable to list audio directory"):
    asr_wer.discover_audio_files(audio_dir)


def test_empty_completions_prompts_are_preserved(tmp_path: Path) -> None:
  request_args = asr_wer.build_completions_request_args(
    build_args(
      "ground",
      str(tmp_path),
      "--endpoint",
      "completions",
      "--prompt",
      "",
      "--system-prompt",
      "",
      "--user-prompt",
      "",
    )
  )

  assert request_args.system_prompt == ""
  assert request_args.developer_prompt == ""
  assert request_args.user_prompt == ""

  explicit_developer_args = asr_wer.build_completions_request_args(
    build_args("ground", str(tmp_path), "--endpoint", "completions", "--developer-prompt", "")
  )
  assert explicit_developer_args.developer_prompt == ""


def test_run_returns_configuration_error_for_invalid_endpoint_json(
  capsys: pytest.CaptureFixture[str],
  tmp_path: Path,
) -> None:
  audio_dir = tmp_path / "audio"
  audio_dir.mkdir()
  write_audio(audio_dir / "clip.wav")

  assert (
    asr_wer.run(
      build_args("ground", str(audio_dir), "--endpoint", "completions", "--completions-response-format-json", "{")
    )
    == 2
  )
  captured = capsys.readouterr()
  assert "Configuration error: Invalid JSON" in captured.err


def test_run_returns_configuration_error_for_unreadable_json_option_file(
  capsys: pytest.CaptureFixture[str],
  tmp_path: Path,
) -> None:
  audio_dir = tmp_path / "audio"
  audio_dir.mkdir()
  write_audio(audio_dir / "clip.wav")

  assert (
    asr_wer.run(
      build_args(
        "ground",
        str(audio_dir),
        "--endpoint",
        "completions",
        "--completions-response-format-json",
        f"@{tmp_path / 'missing.json'}",
      )
    )
    == 2
  )
  captured = capsys.readouterr()
  assert "Configuration error: Unable to read JSON option" in captured.err


def test_resolve_endpoint_model_uses_transcriptions_default_without_shared_model(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.delenv("OPENAI_MODEL", raising=False)
  monkeypatch.delenv("OPENAI_TESTS_MODEL", raising=False)
  args = build_args("ground", "audio")

  assert asr_wer.resolve_endpoint_model(args) == asr_simple.DEFAULT_TRANSCRIPTIONS_MODEL

  args = build_args("ground", "audio", "--model", "shared-model")
  assert asr_wer.resolve_endpoint_model(args) == "shared-model"

  monkeypatch.setenv("OPENAI_MODEL", "env-shared-model")
  args = build_args("ground", "audio")
  assert asr_wer.resolve_endpoint_model(args) == "env-shared-model"


def test_create_output_dir_avoids_eval_collisions(tmp_path: Path) -> None:
  args = build_args("eval", str(tmp_path))
  first = tmp_path / "model_1234"
  first.mkdir()

  assert asr_wer.create_output_dir(args, first) == tmp_path / "model_1234-1"
  assert (tmp_path / "model_1234-1").is_dir()


def test_eval_output_os_error_returns_configuration_error(
  capsys: pytest.CaptureFixture[str],
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  audio_dir = tmp_path / "audio"
  audio_dir.mkdir()
  write_audio(audio_dir / "clip.wav")
  (audio_dir / "ground").mkdir()
  (audio_dir / "ground" / "clip_normalized.txt").write_text("alpha", encoding="utf-8")

  def fake_mkdir(
    path: Path,
    mode: int = 0o777,
    parents: bool = False,
    exist_ok: bool = False,
  ) -> None:
    if path.name.startswith(asr_simple.DEFAULT_TRANSCRIPTIONS_MODEL):
      raise PermissionError("denied")
    original_mkdir(path, mode=mode, parents=parents, exist_ok=exist_ok)

  original_mkdir = Path.mkdir
  monkeypatch.setattr(Path, "mkdir", fake_mkdir)
  assert asr_wer.run(build_args("eval", str(audio_dir))) == 2
  captured = capsys.readouterr()
  assert "Unable to create eval output directory" in captured.err


def test_ground_output_file_collision_returns_configuration_error(
  capsys: pytest.CaptureFixture[str],
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  audio_dir = tmp_path / "audio"
  audio_dir.mkdir()
  write_audio(audio_dir / "clip.wav")
  (audio_dir / "ground").write_text("not a directory", encoding="utf-8")

  def fail_send(**_: object) -> asr_simple.HttpExchange:
    raise AssertionError("endpoint should not be called with an invalid output path")

  monkeypatch.setattr(asr_simple, "send_multipart_request", fail_send)
  assert asr_wer.run(build_args("ground", str(audio_dir))) == 2
  captured = capsys.readouterr()
  assert "Ground output path is not a directory" in captured.err


def test_ground_output_os_error_returns_configuration_error(
  capsys: pytest.CaptureFixture[str],
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  audio_dir = tmp_path / "audio"
  audio_dir.mkdir()
  write_audio(audio_dir / "clip.wav")
  original_mkdir = Path.mkdir

  def fake_mkdir(
    path: Path,
    mode: int = 0o777,
    parents: bool = False,
    exist_ok: bool = False,
  ) -> None:
    if path == audio_dir / "ground":
      raise PermissionError("denied")
    original_mkdir(path, mode=mode, parents=parents, exist_ok=exist_ok)

  monkeypatch.setattr(Path, "mkdir", fake_mkdir)
  assert asr_wer.run(build_args("ground", str(audio_dir))) == 2
  captured = capsys.readouterr()
  assert "Unable to create ground output directory" in captured.err


def test_ground_transcriptions_writes_exact_and_normalized_outputs(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  audio_dir = tmp_path / "audio"
  audio_dir.mkdir()
  write_audio(audio_dir / "clip.wav")
  sent_fields: list[dict[str, object]] = []

  def fake_send_multipart_request(**kwargs: object) -> asr_simple.HttpExchange:
    fields = kwargs["fields"]
    assert isinstance(fields, Mapping)
    sent_fields.append({str(key): value for key, value in fields.items()})
    return asr_simple.HttpExchange(
      method="POST",
      url=str(kwargs["url"]),
      request_headers={},
      request_body=kwargs["fields"],
      response_status=200,
      response_headers={"Content-Type": "application/json"},
      response_body_text='{"text":"Um, colour won\\u2019t be twenty one."}',
      response_json={"text": "Um, colour won\u2019t be twenty one."},
    )

  monkeypatch.setattr(asr_simple, "send_multipart_request", fake_send_multipart_request)
  monkeypatch.setattr(asr_wer, "get_audio_duration_seconds", lambda path: 10.0)
  monkeypatch.setattr(asr_wer.time, "perf_counter", iter([100.0, 102.0, 105.0, 110.0]).__next__)

  result = asr_wer.run(
    build_args(
      "ground",
      str(audio_dir),
      "--api-key",
      "key",
      "--base-url",
      "https://example.com",
      "--transcriptions-model",
      "gpt-asr",
      "--prompt",
      "Names matter.",
      "--service-tier",
      "flex",
    )
  )

  assert result == 0
  assert (audio_dir / "ground" / "clip.txt").read_text(encoding="utf-8") == "Um, colour won\u2019t be twenty one."
  assert (audio_dir / "ground" / "clip_normalized.txt").read_text(encoding="utf-8") == "color will not be 21"
  assert (audio_dir / "ground" / "report.txt").is_file()
  assert sent_fields[0]["prompt"] == "Names matter."
  assert sent_fields[0]["service_tier"] == "flex"


def test_report_records_endpoint_specific_prompt_and_service_tier(tmp_path: Path) -> None:
  result = asr_wer.FileResult(
    audio=asr_wer.AudioInput(path=Path("clip.wav"), stem="clip", format="wav"),
    status="transcribed",
    transcript="Alpha",
    normalized_transcript="alpha",
    output_path=tmp_path / "clip.txt",
    normalized_output_path=tmp_path / "clip_normalized.txt",
    elapsed_seconds=1.0,
    duration_seconds=1.0,
    rtfx=1.0,
    exact_word_count=1,
    normalized_word_count=1,
  )

  transcriptions_dir = tmp_path / "transcriptions"
  transcriptions_dir.mkdir()
  asr_wer.write_report(
    args=build_args("ground", str(tmp_path), "--transcriptions-prompt", "Names matter."),
    model="asr-model",
    output_dir=transcriptions_dir,
    results=[result],
    wall_elapsed_seconds=1.0,
  )
  transcriptions_report = (transcriptions_dir / "report.txt").read_text(encoding="utf-8")
  assert "prompt_present: True" in transcriptions_report
  assert "temperature: provider_default" in transcriptions_report
  assert "prepared_source: false" in transcriptions_report

  completions_dir = tmp_path / "completions"
  completions_dir.mkdir()
  asr_wer.write_report(
    args=build_args(
      "ground",
      str(tmp_path),
      "--endpoint",
      "completions",
      "--completions-service-tier",
      "scale",
      "--developer-prompt",
      "Use custom terms.",
    ),
    model="chat-model",
    output_dir=completions_dir,
    results=[result],
    wall_elapsed_seconds=1.0,
  )
  completions_report = (completions_dir / "report.txt").read_text(encoding="utf-8")
  assert "service_tier: scale" in completions_report
  assert "prompt_present: True" in completions_report
  assert "temperature: provider_default" in completions_report


def test_default_transcriptions_payload_uses_asr_model(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  audio_dir = tmp_path / "audio"
  audio_dir.mkdir()
  write_audio(audio_dir / "clip.wav")
  sent_fields: list[dict[str, object]] = []

  def fake_send_multipart_request(**kwargs: object) -> asr_simple.HttpExchange:
    fields = kwargs["fields"]
    assert isinstance(fields, Mapping)
    sent_fields.append({str(key): value for key, value in fields.items()})
    return asr_simple.HttpExchange(
      method="POST",
      url=str(kwargs["url"]),
      request_headers={},
      request_body=kwargs["fields"],
      response_status=200,
      response_headers={"Content-Type": "application/json"},
      response_body_text='{"text":"Alpha"}',
      response_json={"text": "Alpha"},
    )

  monkeypatch.setattr(asr_simple, "send_multipart_request", fake_send_multipart_request)
  monkeypatch.setattr(asr_wer, "get_audio_duration_seconds", lambda path: 1.0)

  assert asr_wer.run(build_args("ground", str(audio_dir))) == 0
  assert sent_fields[0]["model"] == asr_simple.DEFAULT_TRANSCRIPTIONS_MODEL


def test_transcript_write_failure_does_not_leave_partial_artifacts(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  audio_dir = tmp_path / "audio"
  audio_dir.mkdir()
  audio = write_audio(audio_dir / "clip.wav")
  output_dir = audio_dir / "ground"
  output_dir.mkdir()
  exact_path = output_dir / "clip.txt"
  original_write_text = Path.write_text

  def partial_write_then_fail(
    path: Path,
    data: str,
    encoding: str | None = None,
    errors: str | None = None,
    newline: str | None = None,
  ) -> int:
    if path in {exact_path, exact_path.with_suffix(".txt.tmp")}:
      original_write_text(path, data[:2], encoding=encoding, errors=errors, newline=newline)
      raise OSError("disk full")
    return original_write_text(path, data, encoding=encoding, errors=errors, newline=newline)

  monkeypatch.setattr(Path, "write_text", partial_write_then_fail)
  monkeypatch.setattr(asr_wer, "get_audio_duration_seconds", lambda path: 1.0)
  monkeypatch.setattr(asr_wer, "transcribe_with_selected_endpoint", lambda **_: "Alpha")

  result = asr_wer.transcribe_audio_file(
    args=build_args("ground", str(audio_dir)),
    audio_file=asr_wer.AudioInput(path=audio, stem="clip", format="wav"),
    output_dir=output_dir,
    base_url="https://example.com",
    api_key=None,
  )

  assert result.status == "failed"
  assert not exact_path.exists()
  assert not exact_path.with_suffix(".txt.tmp").exists()
  assert not (output_dir / "clip_normalized.txt").exists()


def test_verbose_transcriptions_prints_http_exchange(
  capsys: pytest.CaptureFixture[str],
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  audio_dir = tmp_path / "audio"
  audio_dir.mkdir()
  write_audio(audio_dir / "clip.wav")

  def fake_send_multipart_request(**kwargs: object) -> asr_simple.HttpExchange:
    return asr_simple.HttpExchange(
      method="POST",
      url=str(kwargs["url"]),
      request_headers={"Authorization": "Bearer secret"},
      request_body={"model": "gpt-asr"},
      response_status=200,
      response_headers={"Content-Type": "application/json"},
      response_body_text='{"text":"Alpha"}',
      response_json={"text": "Alpha"},
    )

  monkeypatch.setattr(asr_simple, "send_multipart_request", fake_send_multipart_request)
  monkeypatch.setattr(asr_wer, "get_audio_duration_seconds", lambda path: 1.0)

  assert asr_wer.run(build_args("ground", str(audio_dir), "--verbose")) == 0
  captured = capsys.readouterr()
  assert "Request:" in captured.out
  assert "POST https://api.openai.com/v1/audio/transcriptions" in captured.out
  assert '"Authorization": "Bearer ***REDACTED***"' in captured.out
  assert "Response:" in captured.out
  assert "HTTP 200" in captured.out


def test_verbose_completions_prints_http_exchange(
  capsys: pytest.CaptureFixture[str],
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  audio_path = write_audio(tmp_path / "clip.wav")

  def fake_send_json_request(**kwargs: object) -> asr_simple.HttpExchange:
    return asr_simple.HttpExchange(
      method="POST",
      url=str(kwargs["url"]),
      request_headers={"Authorization": "Bearer secret"},
      request_body=kwargs["payload"],
      response_status=200,
      response_headers={"Content-Type": "application/json"},
      response_body_text=json.dumps({"choices": [{"message": {"content": "Alpha"}}]}),
      response_json={"choices": [{"message": {"content": "Alpha"}}]},
    )

  monkeypatch.setattr(asr_simple, "send_json_request", fake_send_json_request)
  assert (
    asr_wer.transcribe_with_completions(
      args=build_args("ground", str(tmp_path), "--endpoint", "completions", "--verbose"),
      audio_file=asr_wer.AudioInput(path=audio_path, stem="clip", format="wav"),
      base_url="https://example.com",
      api_key="key",
    )
    == "Alpha"
  )
  captured = capsys.readouterr()
  assert "POST https://example.com/v1/chat/completions" in captured.out
  assert "HTTP 200" in captured.out


def test_verbose_exchange_uses_single_locked_print(monkeypatch: pytest.MonkeyPatch) -> None:
  printed: list[str] = []

  def fake_print(value: str = "") -> None:
    printed.append(value)

  monkeypatch.setattr("builtins.print", fake_print)
  asr_wer.print_verbose_exchange(
    asr_simple.HttpExchange(
      method="POST",
      url="https://example.com/v1/audio/transcriptions",
      request_headers={"Authorization": "Bearer secret"},
      request_body={"model": "gpt-asr"},
      response_status=200,
      response_headers={},
      response_body_text='{"text":"Alpha"}',
      response_json={"text": "Alpha"},
    )
  )

  assert len(printed) == 1
  assert "Request:\nPOST https://example.com/v1/audio/transcriptions" in printed[0]
  assert "\nResponse:\nHTTP 200" in printed[0]


def test_ground_skips_existing_exact_transcript_and_backfills_normalized(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  audio_dir = tmp_path / "audio"
  audio_dir.mkdir()
  write_audio(audio_dir / "clip.wav")
  ground_dir = audio_dir / "ground"
  ground_dir.mkdir()
  (ground_dir / "clip.txt").write_text("I can\u2019t analyse colours.", encoding="utf-8")

  def fail_send(**_: object) -> asr_simple.HttpExchange:
    raise AssertionError("endpoint should not be called for skipped ground transcript")

  monkeypatch.setattr(asr_simple, "send_multipart_request", fail_send)
  monkeypatch.setattr(asr_wer, "get_audio_duration_seconds", lambda path: 4.0)

  assert asr_wer.run(build_args("ground", str(audio_dir))) == 0
  assert (ground_dir / "clip_normalized.txt").read_text(encoding="utf-8") == "i can not analyze colors"
  report = (ground_dir / "report.txt").read_text(encoding="utf-8")
  assert "skipped" in report


def test_ground_skip_uses_existing_normalized_transcript(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  audio_dir = tmp_path / "audio"
  audio_dir.mkdir()
  audio = write_audio(audio_dir / "clip.wav")
  ground_dir = audio_dir / "ground"
  ground_dir.mkdir()
  (ground_dir / "clip.txt").write_text("Raw transcript", encoding="utf-8")
  (ground_dir / "clip_normalized.txt").write_text("custom normalized", encoding="utf-8")
  monkeypatch.setattr(asr_wer, "get_audio_duration_seconds", lambda path: 2.0)

  result = asr_wer.maybe_skip_ground_file(
    build_args("ground", str(audio_dir)),
    asr_wer.AudioInput(path=audio, stem="clip", format="wav"),
    ground_dir,
  )

  assert result is not None
  assert result.normalized_transcript == "custom normalized"


def test_ground_skip_backfill_uses_atomic_write(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  audio_dir = tmp_path / "audio"
  audio_dir.mkdir()
  audio = write_audio(audio_dir / "clip.wav")
  ground_dir = audio_dir / "ground"
  ground_dir.mkdir()
  (ground_dir / "clip.txt").write_text("Raw transcript", encoding="utf-8")
  normalized_path = ground_dir / "clip_normalized.txt"
  calls: list[Path] = []

  def fake_atomic_write_text(path: Path, text: str) -> None:
    calls.append(path)
    path.write_text(text, encoding="utf-8")

  monkeypatch.setattr(asr_wer, "atomic_write_text", fake_atomic_write_text)
  monkeypatch.setattr(asr_wer, "get_audio_duration_seconds", lambda path: 2.0)

  result = asr_wer.maybe_skip_ground_file(
    build_args("ground", str(audio_dir)),
    asr_wer.AudioInput(path=audio, stem="clip", format="wav"),
    ground_dir,
  )

  assert result is not None
  assert calls == [normalized_path]
  assert normalized_path.read_text(encoding="utf-8") == "raw transcript"


def test_ground_skip_records_duration_failure_without_aborting(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  audio_dir = tmp_path / "audio"
  audio_dir.mkdir()
  audio = write_audio(audio_dir / "clip.wav")
  ground_dir = audio_dir / "ground"
  ground_dir.mkdir()
  (ground_dir / "clip.txt").write_text("Raw transcript", encoding="utf-8")
  monkeypatch.setattr(
    asr_wer, "get_audio_duration_seconds", lambda path: (_ for _ in ()).throw(ValueError("bad audio"))
  )

  result = asr_wer.maybe_skip_ground_file(
    build_args("ground", str(audio_dir)),
    asr_wer.AudioInput(path=audio, stem="clip", format="wav"),
    ground_dir,
  )

  assert result is not None
  assert result.status == "failed"
  assert result.error_message == "bad audio"


def test_ground_skip_records_corrupt_cached_transcript_without_aborting(
  tmp_path: Path,
) -> None:
  audio_dir = tmp_path / "audio"
  audio_dir.mkdir()
  audio = write_audio(audio_dir / "clip.wav")
  ground_dir = audio_dir / "ground"
  ground_dir.mkdir()
  (ground_dir / "clip.txt").write_bytes(b"\xff")

  result = asr_wer.maybe_skip_ground_file(
    build_args("ground", str(audio_dir)),
    asr_wer.AudioInput(path=audio, stem="clip", format="wav"),
    ground_dir,
  )

  assert result is not None
  assert result.status == "failed"
  assert result.error_message is not None
  assert "decode" in result.error_message


def test_eval_requires_ground_before_sending_requests(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  audio_dir = tmp_path / "audio"
  audio_dir.mkdir()
  write_audio(audio_dir / "clip.wav")

  def fail_send(**_: object) -> asr_simple.HttpExchange:
    raise AssertionError("endpoint should not be called without ground")

  monkeypatch.setattr(asr_simple, "send_multipart_request", fail_send)
  assert asr_wer.run(build_args("eval", str(audio_dir))) == 2


def test_eval_requires_matching_ground_normalized_transcript(tmp_path: Path) -> None:
  audio_dir = tmp_path / "audio"
  audio_dir.mkdir()
  write_audio(audio_dir / "clip.wav")
  (audio_dir / "ground").mkdir()

  assert asr_wer.run(build_args("eval", str(audio_dir))) == 2


def test_eval_rejects_unreadable_ground_before_sending_requests(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  audio_dir = tmp_path / "audio"
  audio_dir.mkdir()
  write_audio(audio_dir / "clip.wav")
  ground_dir = audio_dir / "ground"
  ground_dir.mkdir()
  (ground_dir / "clip_normalized.txt").write_bytes(b"\xff")

  def fail_send(**_: object) -> asr_simple.HttpExchange:
    raise AssertionError("endpoint should not be called when ground cannot be read")

  monkeypatch.setattr(asr_simple, "send_multipart_request", fail_send)
  assert asr_wer.run(build_args("eval", str(audio_dir))) == 2


def test_eval_reports_ground_os_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  audio_dir = tmp_path / "audio"
  audio_dir.mkdir()
  audio_path = write_audio(audio_dir / "clip.wav")
  ground_dir = audio_dir / "ground"
  ground_dir.mkdir()
  ground_path = ground_dir / "clip_normalized.txt"
  ground_path.write_text("alpha", encoding="utf-8")
  original_read_text = Path.read_text

  def fake_read_text(path: Path, *args: str | None, **kwargs: str | None) -> str:
    if path == ground_path:
      raise OSError("permission denied")
    return original_read_text(path, *args, **kwargs)

  monkeypatch.setattr(Path, "read_text", fake_read_text)
  with pytest.raises(ValueError, match="permission denied"):
    asr_wer.validate_eval_ground(audio_dir, [asr_wer.AudioInput(path=audio_path, stem="clip", format="wav")])


def test_eval_allows_empty_ground_normalized_transcripts(tmp_path: Path) -> None:
  audio_dir = tmp_path / "audio"
  audio_dir.mkdir()
  audio_path = write_audio(audio_dir / "clip.wav")
  ground_dir = audio_dir / "ground"
  ground_dir.mkdir()
  (ground_dir / "clip_normalized.txt").write_text(" \n\t", encoding="utf-8")

  asr_wer.validate_eval_ground(audio_dir, [asr_wer.AudioInput(path=audio_path, stem="clip", format="wav")])


def test_eval_completions_scores_wer_and_sends_service_tier_and_prompt(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  audio_dir = tmp_path / "audio"
  audio_dir.mkdir()
  write_audio(audio_dir / "clip.wav")
  ground_dir = audio_dir / "ground"
  ground_dir.mkdir()
  (ground_dir / "clip_normalized.txt").write_text("alpha bravo charlie", encoding="utf-8")
  sent_payloads: list[dict[str, object]] = []

  def fake_send_json_request(**kwargs: object) -> asr_simple.HttpExchange:
    raw_payload = kwargs["payload"]
    assert isinstance(raw_payload, Mapping)
    payload = dict(raw_payload)
    sent_payloads.append(payload)
    return asr_simple.HttpExchange(
      method="POST",
      url=str(kwargs["url"]),
      request_headers={},
      request_body=payload,
      response_status=200,
      response_headers={"Content-Type": "application/json"},
      response_body_text=json.dumps({"choices": [{"message": {"content": "Alpha bravo"}}]}),
      response_json={"choices": [{"message": {"content": "Alpha bravo"}}]},
    )

  monkeypatch.setattr(asr_simple, "send_json_request", fake_send_json_request)
  monkeypatch.setattr(asr_wer, "get_audio_duration_seconds", lambda path: 8.0)
  monkeypatch.setattr(asr_wer.time, "time", lambda: 1234)
  monkeypatch.setattr(asr_wer.time, "perf_counter", iter([10.0, 12.0, 15.0, 20.0]).__next__)

  assert (
    asr_wer.run(
      build_args(
        "eval",
        str(audio_dir),
        "--endpoint",
        "completions",
        "--completions-model",
        "gpt-audio/custom",
        "--service-tier",
        "priority",
        "--prompt",
        "Return only transcript text.",
      )
    )
    == 0
  )

  payload = sent_payloads[0]
  assert payload["service_tier"] == "priority"
  assert payload["model"] == "gpt-audio/custom"
  messages = payload["messages"]
  assert isinstance(messages, list)
  first_message = messages[0]
  assert "Return only transcript text." in json.dumps(first_message)
  output_dir = audio_dir / "gpt-audio_custom_1234"
  assert (output_dir / "clip_normalized.txt").read_text(encoding="utf-8") == "alpha bravo"
  report = (output_dir / "report.txt").read_text(encoding="utf-8")
  assert "WER" in report
  assert "33.33%" in report
  assert "aggregate_wer_percent: 33.33%" in report


def test_batch_limits_concurrent_transcriptions(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  audio_dir = tmp_path / "audio"
  audio_dir.mkdir()
  for index in range(5):
    write_audio(audio_dir / f"clip-{index}.wav")

  active = 0
  max_active = 0
  lock = threading.Lock()

  def fake_send_multipart_request(**kwargs: object) -> asr_simple.HttpExchange:
    nonlocal active, max_active
    with lock:
      active += 1
      max_active = max(max_active, active)
    time.sleep(0.02)
    with lock:
      active -= 1
    return asr_simple.HttpExchange(
      method="POST",
      url=str(kwargs["url"]),
      request_headers={},
      request_body=kwargs["fields"],
      response_status=200,
      response_headers={"Content-Type": "application/json"},
      response_body_text='{"text":"Alpha"}',
      response_json={"text": "Alpha"},
    )

  monkeypatch.setattr(asr_simple, "send_multipart_request", fake_send_multipart_request)
  monkeypatch.setattr(asr_wer, "get_audio_duration_seconds", lambda path: 1.0)

  assert asr_wer.run(build_args("ground", str(audio_dir), "--batch", "2")) == 0
  assert max_active <= 2


def write_prep_manifest(audio_dir: Path, *, overlap: float = 3.0) -> None:
  """Write the standard prepared-mode fixture manifest under an audio directory.

  The helper creates `audio_dir/prep`, writes the two sample chunk audio files
  used by prepared-mode tests, and emits `manifest.json` with source/chunk timing
  metadata. The overlap parameter controls the manifest `overlap_seconds` value.
  """

  prep_dir = audio_dir / "prep"
  prep_dir.mkdir()
  for name in ("call_0000_000000_030000.wav", "call_0001_027000_050000.wav"):
    write_audio(prep_dir / name)
  manifest = {
    "tool": "openai-tests asr-prep",
    "segment_duration_seconds": 30.0,
    "overlap_seconds": overlap,
    "sources": [{"source_file": "call.wav", "duration_seconds": 50.0, "chunk_count": 2}],
    "chunks": [
      {
        "source_file": "call.wav",
        "source_stem": "call",
        "chunk_file": "call_0000_000000_030000.wav",
        "chunk_index": 0,
        "start_seconds": 0.0,
        "end_seconds": 30.0,
        "duration_seconds": 30.0,
      },
      {
        "source_file": "call.wav",
        "source_stem": "call",
        "chunk_file": "call_0001_027000_050000.wav",
        "chunk_index": 1,
        "start_seconds": 27.0,
        "end_seconds": 50.0,
        "duration_seconds": 23.0,
      },
    ],
  }
  (prep_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def create_prepared_manifest_workspace(tmp_path: Path) -> tuple[Path, Path, Path]:
  """Create an audio/prep workspace for prepared manifest validation tests."""

  audio_dir = tmp_path / "audio"
  prep_dir = audio_dir / "prep"
  prep_dir.mkdir(parents=True)
  return audio_dir, prep_dir, prep_dir / "manifest.json"


def base_prepared_manifest() -> dict[str, Any]:
  """Return a minimal valid one-source, one-chunk prepared manifest mapping."""

  return {
    "overlap_seconds": 3.0,
    "segment_duration_seconds": 30.0,
    "sources": [{"source_file": "call.wav", "duration_seconds": 1.0, "chunk_count": 1}],
    "chunks": [
      {
        "source_file": "call.wav",
        "source_stem": "call",
        "chunk_file": "call_0000_000000_001000.wav",
        "chunk_index": 0,
        "start_seconds": 0.0,
        "end_seconds": 1.0,
        "duration_seconds": 1.0,
      }
    ],
  }


def write_manifest_json(manifest_path: Path, manifest: dict[str, Any]) -> None:
  """Serialize one prepared manifest test case to manifest.json."""

  manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def write_manifest_chunk_fixtures(prep_dir: Path) -> None:
  """Write prepared chunk fixture files used by manifest validation cases."""

  for name in (
    "call_0000_000000_001000.wav",
    "call_0001_001000_002000.wav",
    "call_0000_001000_002000.wav",
    "call_0002_001000_002000.wav",
    "call_0000_000000_020000.wav",
    "call_0000_000000_030000.wav",
    "call_0000_000000_031000.wav",
    "call_0000_000500_001000.wav",
    "call_0001_017000_040000.wav",
    "call_0001_027000_020000.wav",
    "call_0001_027000_030000.wav",
    "call_0001_028000_058000.wav",
    "call_0001_030000_050000.wav",
    "CALL_0000_000000_001000.wav",
    "call_0000_000000_001000.mp3",
    "other_0000_000000_001000.wav",
  ):
    write_audio(prep_dir / name)


def test_prepared_mode_requires_manifest_and_validates_overlap(tmp_path: Path) -> None:
  """Prepared mode requires a manifest and validates requested overlap."""

  audio_dir = tmp_path / "audio"
  audio_dir.mkdir()
  write_audio(audio_dir / "call.wav")

  with pytest.raises(ValueError, match=r"prep/manifest\.json"):
    asr_wer.resolve_prepared_audio_files(audio_dir, requested_overlap=None)

  prep_target = tmp_path / "prep-target"
  prep_target.mkdir()
  (audio_dir / "prep").symlink_to(prep_target, target_is_directory=True)
  with pytest.raises(ValueError, match="prep directory must not be a symlink"):
    asr_wer.resolve_prepared_audio_files(audio_dir, requested_overlap=None)
  (audio_dir / "prep").unlink()

  write_prep_manifest(audio_dir, overlap=3.0)
  with pytest.raises(ValueError, match="does not match manifest"):
    asr_wer.resolve_prepared_audio_files(audio_dir, requested_overlap=2.0)


def test_prepared_manifest_read_errors(tmp_path: Path) -> None:
  """Prepared manifest loading rejects symlinks, malformed JSON, and arrays."""

  audio_dir, _prep_dir, manifest_path = create_prepared_manifest_workspace(tmp_path)
  outside_manifest = tmp_path / "manifest.json"
  outside_manifest.write_text("{}", encoding="utf-8")
  manifest_path.symlink_to(outside_manifest)
  with pytest.raises(ValueError, match="manifest must not be a symlink"):
    asr_wer.resolve_prepared_audio_files(audio_dir, requested_overlap=None)
  manifest_path.unlink()

  manifest_path.write_text("{", encoding="utf-8")
  with pytest.raises(ValueError, match="Unable to read prepared manifest"):
    asr_wer.resolve_prepared_audio_files(audio_dir, requested_overlap=None)

  manifest_path.write_text("[]", encoding="utf-8")
  with pytest.raises(ValueError, match="must be a JSON object"):
    asr_wer.resolve_prepared_audio_files(audio_dir, requested_overlap=None)


def test_prepared_manifest_structure_errors(tmp_path: Path) -> None:
  """Prepared manifest parsing rejects missing arrays and malformed rows."""

  audio_dir, _prep_dir, manifest_path = create_prepared_manifest_workspace(tmp_path)
  write_manifest_json(manifest_path, {"overlap_seconds": 3.0, "segment_duration_seconds": 30.0})
  with pytest.raises(ValueError, match="requires sources and chunks arrays"):
    asr_wer.resolve_prepared_audio_files(audio_dir, requested_overlap=None)

  write_manifest_json(
    manifest_path, {"overlap_seconds": 3.0, "segment_duration_seconds": 30.0, "sources": [{}], "chunks": []}
  )
  with pytest.raises(ValueError, match="source rows must include source_file"):
    asr_wer.resolve_prepared_audio_files(audio_dir, requested_overlap=None)

  write_manifest_json(
    manifest_path,
    {
      "overlap_seconds": 3.0,
      "segment_duration_seconds": 30.0,
      "sources": [{"source_file": "call.wav", "duration_seconds": 1.0, "chunk_count": 1}],
      "chunks": ["bad"],
    },
  )
  with pytest.raises(ValueError, match="chunk rows must be objects"):
    asr_wer.resolve_prepared_audio_files(audio_dir, requested_overlap=None)

  missing_chunk = base_prepared_manifest()
  missing_chunk["chunks"] = [{**missing_chunk["chunks"][0], "chunk_file": "missing.wav"}]
  write_manifest_json(manifest_path, missing_chunk)
  with pytest.raises(ValueError, match="Prepared chunk file does not exist"):
    asr_wer.resolve_prepared_audio_files(audio_dir, requested_overlap=None)


def test_prepared_manifest_path_and_type_errors(tmp_path: Path) -> None:
  """Prepared manifest validation rejects unsafe paths and field types."""

  audio_dir, prep_dir, manifest_path = create_prepared_manifest_workspace(tmp_path)
  unsafe_base = base_prepared_manifest()
  write_audio(prep_dir / "call_0000_000000_001000.wav")
  symlink_target = write_audio(audio_dir / "outside.wav")
  (prep_dir / "call_0000_000000_001000.wav").unlink()
  (prep_dir / "call_0000_000000_001000.wav").symlink_to(symlink_target)
  write_manifest_json(manifest_path, unsafe_base)
  with pytest.raises(ValueError, match="must not be a symlink"):
    asr_wer.resolve_prepared_audio_files(audio_dir, requested_overlap=None)
  (prep_dir / "call_0000_000000_001000.wav").unlink()
  write_audio(prep_dir / "call_0000_000000_001000.wav")

  duplicate_source = {
    **unsafe_base,
    "sources": [
      {"source_file": "call.wav", "duration_seconds": 1.0, "chunk_count": 1},
      {"source_file": "call.wav", "duration_seconds": 2.0, "chunk_count": 1},
    ],
  }
  write_manifest_json(manifest_path, duplicate_source)
  with pytest.raises(ValueError, match=r"duplicate source_file call\.wav"):
    asr_wer.resolve_prepared_audio_files(audio_dir, requested_overlap=None)

  unsafe_source = {**unsafe_base, "sources": [{"source_file": "../call.wav", "duration_seconds": 1.0}]}
  write_manifest_json(manifest_path, unsafe_source)
  with pytest.raises(ValueError, match="plain filename source_file"):
    asr_wer.resolve_prepared_audio_files(audio_dir, requested_overlap=None)

  unsafe_chunk_source = {**unsafe_base, "chunks": [{**unsafe_base["chunks"][0], "source_file": "/call.wav"}]}
  write_manifest_json(manifest_path, unsafe_chunk_source)
  with pytest.raises(ValueError, match="plain filename source_file"):
    asr_wer.resolve_prepared_audio_files(audio_dir, requested_overlap=None)

  unsafe_chunk_file = {**unsafe_base, "chunks": [{**unsafe_base["chunks"][0], "chunk_file": "../secret.wav"}]}
  write_manifest_json(manifest_path, unsafe_chunk_file)
  with pytest.raises(ValueError, match="plain filename chunk_file"):
    asr_wer.resolve_prepared_audio_files(audio_dir, requested_overlap=None)

  unsafe_stem = {**unsafe_base, "chunks": [{**unsafe_base["chunks"][0], "source_stem": "../call"}]}
  write_manifest_json(manifest_path, unsafe_stem)
  with pytest.raises(ValueError, match="plain filename stem source_stem"):
    asr_wer.resolve_prepared_audio_files(audio_dir, requested_overlap=None)

  mismatched_stem = {**unsafe_base, "chunks": [{**unsafe_base["chunks"][0], "source_stem": "other"}]}
  write_manifest_json(manifest_path, mismatched_stem)
  with pytest.raises(ValueError, match="does not match source_file"):
    asr_wer.resolve_prepared_audio_files(audio_dir, requested_overlap=None)

  fractional_index = {**unsafe_base, "chunks": [{**unsafe_base["chunks"][0], "chunk_index": 1.9}]}
  write_manifest_json(manifest_path, fractional_index)
  with pytest.raises(ValueError, match="integer chunk_index"):
    asr_wer.resolve_prepared_audio_files(audio_dir, requested_overlap=None)


def test_prepared_manifest_chunk_identity_errors(tmp_path: Path) -> None:
  """Prepared manifest validation rejects duplicate or inconsistent chunks."""

  audio_dir, prep_dir, manifest_path = create_prepared_manifest_workspace(tmp_path)
  unsafe_base = base_prepared_manifest()
  write_manifest_chunk_fixtures(prep_dir)
  duplicate_index = {
    **unsafe_base,
    "chunks": [
      unsafe_base["chunks"][0],
      {
        **unsafe_base["chunks"][0],
        "chunk_file": "call_0000_001000_002000.wav",
        "start_seconds": 1.0,
        "end_seconds": 2.0,
      },
    ],
  }
  write_manifest_json(manifest_path, duplicate_index)
  with pytest.raises(ValueError, match=r"duplicate chunk_index 0 for call\.wav"):
    asr_wer.resolve_prepared_audio_files(audio_dir, requested_overlap=None)

  duplicate_chunk_file = {
    **unsafe_base,
    "sources": [{"source_file": "call.wav", "duration_seconds": 2.0, "chunk_count": 2}],
    "chunks": [unsafe_base["chunks"][0], {**unsafe_base["chunks"][0], "chunk_index": 1}],
  }
  write_manifest_json(manifest_path, duplicate_chunk_file)
  with pytest.raises(ValueError, match="duplicate chunk_file"):
    asr_wer.resolve_prepared_audio_files(audio_dir, requested_overlap=None)

  case_duplicate_chunk_file = {
    **unsafe_base,
    "sources": [{"source_file": "call.wav", "duration_seconds": 2.0, "chunk_count": 2}],
    "chunks": [
      unsafe_base["chunks"][0],
      {**unsafe_base["chunks"][0], "chunk_file": "CALL_0000_000000_001000.wav", "chunk_index": 1},
    ],
  }
  write_manifest_json(manifest_path, case_duplicate_chunk_file)
  with pytest.raises(ValueError, match="duplicate chunk_file"):
    asr_wer.resolve_prepared_audio_files(audio_dir, requested_overlap=None)

  duplicate_chunk_stem = {
    **unsafe_base,
    "sources": [{"source_file": "call.wav", "duration_seconds": 2.0, "chunk_count": 2}],
    "chunks": [
      unsafe_base["chunks"][0],
      {**unsafe_base["chunks"][0], "chunk_file": "call_0000_000000_001000.mp3", "chunk_index": 1},
    ],
  }
  write_manifest_json(manifest_path, duplicate_chunk_stem)
  with pytest.raises(ValueError, match="duplicate chunk stem"):
    asr_wer.resolve_prepared_audio_files(audio_dir, requested_overlap=None)

  mismatched_chunk_name = {
    **unsafe_base,
    "chunks": [{**unsafe_base["chunks"][0], "chunk_file": "other_0000_000000_001000.wav"}],
  }
  write_manifest_json(manifest_path, mismatched_chunk_name)
  with pytest.raises(ValueError, match=r"chunk_file .* does not match expected"):
    asr_wer.resolve_prepared_audio_files(audio_dir, requested_overlap=None)

  non_contiguous_index = {
    **unsafe_base,
    "sources": [{"source_file": "call.wav", "duration_seconds": 2.0, "chunk_count": 2}],
    "chunks": [
      unsafe_base["chunks"][0],
      {
        **unsafe_base["chunks"][0],
        "chunk_file": "call_0002_001000_002000.wav",
        "chunk_index": 2,
        "start_seconds": 1.0,
        "end_seconds": 2.0,
      },
    ],
  }
  write_manifest_json(manifest_path, non_contiguous_index)
  with pytest.raises(ValueError, match=r"chunk_index values for call\.wav must be contiguous"):
    asr_wer.resolve_prepared_audio_files(audio_dir, requested_overlap=None)


def test_prepared_manifest_range_and_duration_errors(tmp_path: Path) -> None:
  """Prepared manifest validation rejects invalid durations and chunk ranges."""

  audio_dir, prep_dir, manifest_path = create_prepared_manifest_workspace(tmp_path)
  unsafe_base = base_prepared_manifest()
  write_manifest_chunk_fixtures(prep_dir)
  non_positive_source_duration = {
    **unsafe_base,
    "sources": [{"source_file": "call.wav", "duration_seconds": 0.0, "chunk_count": 1}],
  }
  write_manifest_json(manifest_path, non_positive_source_duration)
  with pytest.raises(ValueError, match=r"source duration_seconds for call\.wav must be greater than 0"):
    asr_wer.resolve_prepared_audio_files(audio_dir, requested_overlap=None)

  non_positive_chunk_duration = {**unsafe_base, "chunks": [{**unsafe_base["chunks"][0], "duration_seconds": 0.0}]}
  write_manifest_json(manifest_path, non_positive_chunk_duration)
  with pytest.raises(
    ValueError, match=r"chunk duration_seconds for call_0000_000000_001000\.wav must be greater than 0"
  ):
    asr_wer.resolve_prepared_audio_files(audio_dir, requested_overlap=None)

  impossible_ranges = {
    **unsafe_base,
    "sources": [{"source_file": "call.wav", "duration_seconds": 50.0, "chunk_count": 2}],
    "chunks": [
      {
        **unsafe_base["chunks"][0],
        "chunk_file": "call_0000_000000_030000.wav",
        "end_seconds": 30.0,
        "duration_seconds": 30.0,
      },
      {
        **unsafe_base["chunks"][0],
        "chunk_file": "call_0001_027000_020000.wav",
        "chunk_index": 1,
        "start_seconds": 27.0,
        "end_seconds": 20.0,
        "duration_seconds": 1.0,
      },
    ],
  }
  write_manifest_json(manifest_path, impossible_ranges)
  with pytest.raises(ValueError, match=r"chunk duration mismatch for call\.wav"):
    asr_wer.resolve_prepared_audio_files(audio_dir, requested_overlap=None)

  short_non_final_chunk = {
    **unsafe_base,
    "sources": [{"source_file": "call.wav", "duration_seconds": 40.0, "chunk_count": 2}],
    "chunks": [
      {
        **unsafe_base["chunks"][0],
        "chunk_file": "call_0000_000000_020000.wav",
        "end_seconds": 20.0,
        "duration_seconds": 20.0,
      },
      {
        **unsafe_base["chunks"][0],
        "chunk_file": "call_0001_017000_040000.wav",
        "chunk_index": 1,
        "start_seconds": 17.0,
        "end_seconds": 40.0,
        "duration_seconds": 23.0,
      },
    ],
  }
  write_manifest_json(manifest_path, short_non_final_chunk)
  with pytest.raises(ValueError, match=r"non-final chunks for call\.wav must match segment_duration_seconds"):
    asr_wer.resolve_prepared_audio_files(audio_dir, requested_overlap=None)

  oversized_chunk = {
    **unsafe_base,
    "sources": [{"source_file": "call.wav", "duration_seconds": 58.0, "chunk_count": 2}],
    "chunks": [
      {
        **unsafe_base["chunks"][0],
        "chunk_file": "call_0000_000000_031000.wav",
        "end_seconds": 31.0,
        "duration_seconds": 31.0,
      },
      {
        **unsafe_base["chunks"][0],
        "chunk_file": "call_0001_028000_058000.wav",
        "chunk_index": 1,
        "start_seconds": 28.0,
        "end_seconds": 58.0,
        "duration_seconds": 30.0,
      },
    ],
  }
  write_manifest_json(manifest_path, oversized_chunk)
  with pytest.raises(ValueError, match=r"chunk duration exceeds segment_duration_seconds for call\.wav"):
    asr_wer.resolve_prepared_audio_files(audio_dir, requested_overlap=None)

  duplicate_tail = {
    **unsafe_base,
    "sources": [{"source_file": "call.wav", "duration_seconds": 30.0, "chunk_count": 2}],
    "chunks": [
      {
        **unsafe_base["chunks"][0],
        "chunk_file": "call_0000_000000_030000.wav",
        "end_seconds": 30.0,
        "duration_seconds": 30.0,
      },
      {
        **unsafe_base["chunks"][0],
        "chunk_file": "call_0001_027000_030000.wav",
        "chunk_index": 1,
        "start_seconds": 27.0,
        "end_seconds": 30.0,
        "duration_seconds": 3.0,
      },
    ],
  }
  write_manifest_json(manifest_path, duplicate_tail)
  with pytest.raises(ValueError, match=r"non-final chunks for call\.wav must end before source duration"):
    asr_wer.resolve_prepared_audio_files(audio_dir, requested_overlap=None)

  unsupported_chunk_file = {**unsafe_base, "chunks": [{**unsafe_base["chunks"][0], "chunk_file": "manifest.json"}]}
  write_manifest_json(manifest_path, unsupported_chunk_file)
  with pytest.raises(ValueError, match="unsupported prepared chunk extension"):
    asr_wer.resolve_prepared_audio_files(audio_dir, requested_overlap=None)

  truncated_ranges = {
    **unsafe_base,
    "sources": [{"source_file": "call.wav", "duration_seconds": 50.0, "chunk_count": 1}],
  }
  write_manifest_json(manifest_path, truncated_ranges)
  with pytest.raises(ValueError, match=r"chunk ranges for call\.wav must end at source duration"):
    asr_wer.resolve_prepared_audio_files(audio_dir, requested_overlap=None)

  shifted_start_range = {
    **unsafe_base,
    "chunks": [
      {
        **unsafe_base["chunks"][0],
        "chunk_file": "call_0000_000500_001000.wav",
        "start_seconds": 0.5,
        "end_seconds": 1.0,
        "duration_seconds": 0.5,
      }
    ],
  }
  write_manifest_json(manifest_path, shifted_start_range)
  with pytest.raises(ValueError, match=r"chunk ranges for call\.wav must start at 0"):
    asr_wer.resolve_prepared_audio_files(audio_dir, requested_overlap=None)

  gap_ranges = {
    **unsafe_base,
    "sources": [{"source_file": "call.wav", "duration_seconds": 50.0, "chunk_count": 2}],
    "chunks": [
      {
        **unsafe_base["chunks"][0],
        "chunk_file": "call_0000_000000_030000.wav",
        "end_seconds": 30.0,
        "duration_seconds": 30.0,
      },
      {
        **unsafe_base["chunks"][0],
        "chunk_file": "call_0001_030000_050000.wav",
        "chunk_index": 1,
        "start_seconds": 30.0,
        "end_seconds": 50.0,
        "duration_seconds": 20.0,
      },
    ],
  }
  write_manifest_json(manifest_path, gap_ranges)
  with pytest.raises(ValueError, match=r"chunk range gap for call\.wav"):
    asr_wer.resolve_prepared_audio_files(audio_dir, requested_overlap=None)


def test_prepared_manifest_source_chunk_metadata_errors(tmp_path: Path) -> None:
  """Prepared manifest validation rejects inconsistent source/chunk metadata."""

  audio_dir, prep_dir, manifest_path = create_prepared_manifest_workspace(tmp_path)
  unsafe_base = base_prepared_manifest()
  write_manifest_chunk_fixtures(prep_dir)
  missing_chunk_source = {
    **unsafe_base,
    "sources": [
      {"source_file": "call.wav", "duration_seconds": 1.0, "chunk_count": 1},
      {"source_file": "other.wav", "duration_seconds": 1.0, "chunk_count": 1},
    ],
  }
  write_manifest_json(manifest_path, missing_chunk_source)
  with pytest.raises(ValueError, match=r"sources without chunks: other\.wav"):
    asr_wer.resolve_prepared_audio_files(audio_dir, requested_overlap=None)

  undeclared_chunk_source = {**unsafe_base, "sources": []}
  write_manifest_json(manifest_path, undeclared_chunk_source)
  with pytest.raises(ValueError, match=r"chunks for undeclared sources: call\.wav"):
    asr_wer.resolve_prepared_audio_files(audio_dir, requested_overlap=None)

  wrong_chunk_count = {
    **unsafe_base,
    "sources": [{"source_file": "call.wav", "duration_seconds": 1.0, "chunk_count": 2}],
  }
  write_manifest_json(manifest_path, wrong_chunk_count)
  with pytest.raises(ValueError, match=r"chunk_count for call\.wav is 2"):
    asr_wer.resolve_prepared_audio_files(audio_dir, requested_overlap=None)

  write_manifest_json(
    manifest_path, {"overlap_seconds": True, "segment_duration_seconds": 30.0, "sources": [], "chunks": []}
  )
  with pytest.raises(ValueError, match="numeric overlap_seconds"):
    asr_wer.resolve_prepared_audio_files(audio_dir, requested_overlap=None)

  write_manifest_json(
    manifest_path, {"overlap_seconds": 0.0, "segment_duration_seconds": 0.0, "sources": [], "chunks": []}
  )
  with pytest.raises(ValueError, match="segment_duration_seconds must be greater than 0"):
    asr_wer.resolve_prepared_audio_files(audio_dir, requested_overlap=None)

  write_manifest_json(
    manifest_path, {"overlap_seconds": 30.0, "segment_duration_seconds": 30.0, "sources": [], "chunks": []}
  )
  with pytest.raises(ValueError, match="overlap_seconds must be at least 0"):
    asr_wer.resolve_prepared_audio_files(audio_dir, requested_overlap=None)

  write_manifest_json(
    manifest_path, {"overlap_seconds": 3.0, "segment_duration_seconds": 30.0, "sources": [], "chunks": []}
  )
  with pytest.raises(ValueError, match="does not contain any chunks"):
    asr_wer.resolve_prepared_audio_files(audio_dir, requested_overlap=None)


def test_prepared_manifest_helper_validators() -> None:
  """Prepared manifest scalar validators reject unsafe or invalid values."""

  with pytest.raises(ValueError, match="string source_file"):
    asr_wer.require_manifest_string({}, "source_file")
  with pytest.raises(ValueError, match="plain filename source_file"):
    asr_wer.require_manifest_filename({"source_file": "nested/call.wav"}, "source_file")
  with pytest.raises(ValueError, match="plain filename source_file"):
    asr_wer.require_manifest_filename({"source_file": "C:call.wav"}, "source_file")
  with pytest.raises(ValueError, match="plain filename source_file"):
    asr_wer.require_manifest_filename({"source_file": "call\r\nbad.wav"}, "source_file")
  with pytest.raises(ValueError, match="plain filename stem source_stem"):
    asr_wer.require_manifest_stem({"source_stem": "/call"}, "source_stem")
  with pytest.raises(ValueError, match="plain filename stem source_stem"):
    asr_wer.require_manifest_stem({"source_stem": "C:call"}, "source_stem")
  with pytest.raises(ValueError, match="plain filename stem source_stem"):
    asr_wer.require_manifest_stem({"source_stem": "call\tbad"}, "source_stem")
  with pytest.raises(ValueError, match="numeric duration_seconds"):
    asr_wer.require_manifest_number({"duration_seconds": "1"}, "duration_seconds")
  with pytest.raises(ValueError, match="finite numeric duration_seconds"):
    asr_wer.require_manifest_number({"duration_seconds": float("inf")}, "duration_seconds")
  with pytest.raises(ValueError, match="integer chunk_index"):
    asr_wer.require_manifest_integer({"chunk_index": True}, "chunk_index")


def test_prepared_ground_reads_chunks_and_writes_combined_root_artifacts(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  """Prepared ground reads chunk audio and writes combined root artifacts."""

  audio_dir = tmp_path / "audio"
  audio_dir.mkdir()
  write_audio(audio_dir / "call.wav")
  write_prep_manifest(audio_dir)
  seen_files: list[str] = []
  transcripts = iter(["Alpha Bravo repeated", "repeated Charlie"])

  def fake_transcribe(**kwargs: object) -> str:
    audio_file = kwargs["audio_file"]
    assert isinstance(audio_file, asr_wer.AudioInput)
    seen_files.append(audio_file.path.name)
    return next(transcripts)

  monkeypatch.setattr(asr_wer, "transcribe_with_selected_endpoint", fake_transcribe)

  assert asr_wer.run(build_args("ground", str(audio_dir), "--prep")) == 0

  assert seen_files == ["call_0000_000000_030000.wav", "call_0001_027000_050000.wav"]
  ground_dir = audio_dir / "ground"
  assert (ground_dir / "call.txt").read_text(encoding="utf-8") == "Alpha Bravo repeated\nrepeated Charlie"
  assert (ground_dir / "call_normalized.txt").read_text(encoding="utf-8") == "alpha bravo repeated charlie"
  assert (ground_dir / "chunks" / "call_0000_000000_030000.txt").read_text(encoding="utf-8") == "Alpha Bravo repeated"
  report = (ground_dir / "report.txt").read_text(encoding="utf-8")
  assert "temperature: 0.0" in report
  assert "prepared_source: true" in report
  assert "prep_folder:" in report
  assert "\tchunk_count\t" in report
  assert "call.wav\ttranscribed" in report


def test_prepared_eval_requires_combined_ground_and_honors_batch(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  """Prepared eval requires combined ground files and honors batch concurrency."""

  audio_dir = tmp_path / "audio"
  audio_dir.mkdir()
  write_audio(audio_dir / "call.wav")
  write_prep_manifest(audio_dir)

  assert asr_wer.run(build_args("eval", str(audio_dir), "--prep")) == 2

  ground_dir = audio_dir / "ground"
  ground_dir.mkdir()
  (ground_dir / "call.txt").write_text("Alpha bravo charlie", encoding="utf-8")
  (ground_dir / "call_normalized.txt").write_text("alpha bravo charlie", encoding="utf-8")
  active = 0
  max_active = 0
  lock = threading.Lock()

  def fake_transcribe(**kwargs: object) -> str:
    nonlocal active, max_active
    with lock:
      active += 1
      max_active = max(max_active, active)
    time.sleep(0.02)
    with lock:
      active -= 1
    audio_file = kwargs["audio_file"]
    assert isinstance(audio_file, asr_wer.AudioInput)
    return "Alpha bravo" if audio_file.path.name.endswith("030000.wav") else "charlie"

  monkeypatch.setattr(asr_wer, "transcribe_with_selected_endpoint", fake_transcribe)
  monkeypatch.setattr(asr_wer.time, "time", lambda: 1234)

  assert asr_wer.run(build_args("eval", str(audio_dir), "--prep", "--batch", "2")) == 0
  assert max_active == 2
  output_dir = audio_dir / f"{asr_simple.DEFAULT_TRANSCRIPTIONS_MODEL}_1234"
  assert (output_dir / "call_normalized.txt").read_text(encoding="utf-8") == "alpha bravo charlie"
  report = (output_dir / "report.txt").read_text(encoding="utf-8")
  assert "prepared_source: true" in report
  assert "0.00%" in report


def test_prepared_temperature_defaults_only_selected_endpoint(tmp_path: Path) -> None:
  """Prepared mode defaults only the selected endpoint temperature."""

  assert (
    asr_wer.build_transcriptions_request_args(build_args("ground", str(tmp_path), "--prep")).transcriptions_temperature
    == 0.0
  )
  assert (
    asr_wer.build_transcriptions_request_args(
      build_args("ground", str(tmp_path), "--prep", "--transcriptions-temperature", "0.4")
    ).transcriptions_temperature
    == 0.4
  )
  assert (
    asr_wer.build_completions_request_args(
      build_args("ground", str(tmp_path), "--prep", "--endpoint", "completions")
    ).completions_temperature
    == 0.0
  )
  assert (
    asr_wer.build_completions_request_args(
      build_args("ground", str(tmp_path), "--prep", "--endpoint", "completions", "--completions-temperature", "0.2")
    ).completions_temperature
    == 0.2
  )


def test_prepared_failed_chunk_fails_parent_original(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  """A failed prepared chunk marks its parent original source as failed."""

  audio_dir = tmp_path / "audio"
  audio_dir.mkdir()
  write_audio(audio_dir / "call.wav")
  write_prep_manifest(audio_dir)

  def fake_transcribe(**kwargs: object) -> str:
    audio_file = kwargs["audio_file"]
    assert isinstance(audio_file, asr_wer.AudioInput)
    if audio_file.path.name.endswith("050000.wav"):
      raise ValueError("provider failed")
    return "Alpha"

  monkeypatch.setattr(asr_wer, "transcribe_with_selected_endpoint", fake_transcribe)

  assert asr_wer.run(build_args("ground", str(audio_dir), "--prep")) == 1
  report = (audio_dir / "ground" / "report.txt").read_text(encoding="utf-8")
  assert "call.wav\tfailed" in report
  assert "provider failed" in report


def test_prepared_skip_and_failure_branches(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
  """Prepared result helpers cover skip, write-failure, stitching, and mismatch branches."""

  audio_dir = tmp_path / "audio"
  audio_dir.mkdir()
  write_audio(audio_dir / "call.wav")
  write_prep_manifest(audio_dir)
  sources = asr_wer.resolve_prepared_audio_files(audio_dir, requested_overlap=None)
  output_dir = audio_dir / "ground"
  output_dir.mkdir()
  (output_dir / "call.txt").write_text("Alpha", encoding="utf-8")
  (output_dir / "call_normalized.txt").write_text("alpha", encoding="utf-8")

  skipped = asr_wer.maybe_skip_prepared_ground_file(
    build_args("ground", str(audio_dir), "--prep"), sources[0], output_dir
  )
  assert skipped is not None
  assert skipped.status == "skipped"
  assert skipped.chunk_count == 2

  (output_dir / "call.txt").write_bytes(b"\xff")
  failed_skip = asr_wer.maybe_skip_prepared_ground_file(
    build_args("ground", str(audio_dir), "--prep"), sources[0], output_dir
  )
  assert failed_skip is not None
  assert failed_skip.status == "failed"
  assert failed_skip.duration_seconds == 50.0

  missing_chunk_result = asr_wer.build_prepared_source_result(
    args=build_args("ground", str(audio_dir), "--prep"),
    source=sources[0],
    output_dir=output_dir,
    chunk_transcripts={0: "Alpha"},
    chunk_errors=[],
    elapsed_seconds=1.0,
  )
  assert missing_chunk_result.status == "failed"
  assert "missing chunk transcripts" in str(missing_chunk_result.error_message)

  def fail_write(path: Path, text: str) -> None:
    raise OSError("disk full")

  original_atomic_write_text = asr_wer.atomic_write_text
  monkeypatch.setattr(asr_wer, "atomic_write_text", fail_write)
  write_failed = asr_wer.build_prepared_source_result(
    args=build_args("ground", str(audio_dir), "--prep"),
    source=sources[0],
    output_dir=output_dir,
    chunk_transcripts={0: "Alpha", 1: "Bravo"},
    chunk_errors=[],
    elapsed_seconds=1.0,
  )
  assert write_failed.status == "failed"
  assert write_failed.error_message == "disk full"
  monkeypatch.setattr(asr_wer, "atomic_write_text", original_atomic_write_text)
  source_level_normalized = asr_wer.build_prepared_source_result(
    args=build_args("ground", str(audio_dir), "--prep"),
    source=sources[0],
    output_dir=output_dir,
    chunk_transcripts={0: "twenty", 1: "one pilots"},
    chunk_errors=[],
    elapsed_seconds=1.0,
  )
  assert source_level_normalized.normalized_transcript == "21 pilots"
  normalized_overlap = asr_wer.build_prepared_source_result(
    args=build_args("ground", str(audio_dir), "--prep"),
    source=sources[0],
    output_dir=output_dir,
    chunk_transcripts={0: "Alpha repeated,", 1: "REPEATED Bravo"},
    chunk_errors=[],
    elapsed_seconds=1.0,
  )
  assert normalized_overlap.normalized_transcript == "alpha repeated bravo"
  assert asr_wer.stitch_normalized_transcripts(["alpha repeated", "repeated bravo"], overlap_seconds=0.0) == (
    "alpha repeated repeated bravo"
  )
  assert asr_wer.stitch_normalized_transcripts(["alpha repeated", "repeated bravo"], overlap_seconds=0.001) == (
    "alpha repeated repeated bravo"
  )
  assert asr_wer.stitch_normalized_transcripts(["alpha repeated", "repeated bravo"], overlap_seconds=1.0) == (
    "alpha repeated bravo"
  )
  assert asr_wer.stitch_exact_transcripts(["Alpha repeated", "repeated Bravo"], overlap_seconds=1.0) == (
    "Alpha repeated Bravo"
  )

  duplicate_chunk_source = asr_wer.PreparedSource(
    audio=sources[0].audio,
    chunks=(sources[0].chunks[0], sources[0].chunks[0]),
    duration_seconds=sources[0].duration_seconds,
    overlap_seconds=sources[0].overlap_seconds,
    segment_duration_seconds=sources[0].segment_duration_seconds,
  )
  count_mismatch = asr_wer.build_prepared_source_result(
    args=build_args("ground", str(audio_dir), "--prep"),
    source=duplicate_chunk_source,
    output_dir=output_dir,
    chunk_transcripts={0: "Alpha"},
    chunk_errors=[],
    elapsed_seconds=1.0,
  )
  assert count_mismatch.status == "failed"
  assert count_mismatch.error_message == "chunk transcript count mismatch"


def test_process_prepared_sources_uses_existing_combined_ground(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  """Prepared ground skips request submission when combined outputs already exist."""

  audio_dir = tmp_path / "audio"
  audio_dir.mkdir()
  write_audio(audio_dir / "call.wav")
  write_prep_manifest(audio_dir)
  source = asr_wer.resolve_prepared_audio_files(audio_dir, requested_overlap=None)[0]
  output_dir = audio_dir / "ground"
  output_dir.mkdir()
  (output_dir / "call.txt").write_text("Alpha", encoding="utf-8")
  (output_dir / "call_normalized.txt").write_text("alpha", encoding="utf-8")

  def fail_transcribe(**_: object) -> str:
    raise AssertionError("prepared ground skip should not send chunk requests")

  monkeypatch.setattr(asr_wer, "transcribe_with_selected_endpoint", fail_transcribe)
  results = asr_wer.process_prepared_sources(
    args=build_args("ground", str(audio_dir), "--prep"),
    prepared_sources=[source],
    output_dir=output_dir,
    base_url="https://example.com",
    api_key=None,
  )

  assert [result.status for result in results] == ["skipped"]


def test_process_prepared_sources_reports_chunks_output_collision(tmp_path: Path) -> None:
  """Prepared processing reports a controlled failure for a file named chunks."""

  audio_dir = tmp_path / "audio"
  audio_dir.mkdir()
  write_audio(audio_dir / "call.wav")
  write_prep_manifest(audio_dir)
  source = asr_wer.resolve_prepared_audio_files(audio_dir, requested_overlap=None)[0]
  output_dir = audio_dir / "ground"
  output_dir.mkdir()
  (output_dir / "chunks").write_text("not a directory", encoding="utf-8")

  results = asr_wer.process_prepared_sources(
    args=build_args("ground", str(audio_dir), "--prep"),
    prepared_sources=[source],
    output_dir=output_dir,
    base_url="https://example.com",
    api_key=None,
  )

  assert [result.status for result in results] == ["failed"]
  assert results[0].error_message is not None
  assert "Prepared chunks output path is not a directory" in results[0].error_message


def test_prepare_prepared_chunks_output_dir_rejects_symlink(tmp_path: Path) -> None:
  """Prepared chunk audit directory creation rejects symlinked chunks paths."""

  output_dir = tmp_path / "ground"
  target_dir = tmp_path / "audit-target"
  target_dir.mkdir()
  output_dir.mkdir()
  (output_dir / "chunks").symlink_to(target_dir, target_is_directory=True)

  error = asr_wer.prepare_prepared_chunks_output_dir(output_dir)

  assert error is not None
  assert "Prepared chunks output path must not be a symlink" in error


def test_prepare_prepared_chunks_output_dir_reports_os_errors(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  """Prepared chunk audit directory creation reports filesystem errors."""

  output_dir = tmp_path / "ground"
  original_mkdir = Path.mkdir

  def fail_mkdir(
    path: Path,
    mode: int = 0o777,
    parents: bool = False,
    exist_ok: bool = False,
  ) -> None:
    if path == output_dir / "chunks":
      raise PermissionError("denied")
    original_mkdir(path, mode=mode, parents=parents, exist_ok=exist_ok)

  monkeypatch.setattr(Path, "mkdir", fail_mkdir)
  error = asr_wer.prepare_prepared_chunks_output_dir(output_dir)

  assert error is not None
  assert "Unable to create prepared chunks output directory" in error


def test_process_prepared_sources_uses_per_source_finish_times(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  """Prepared processing uses per-source chunk completion times for elapsed metrics."""

  audio_dir = tmp_path / "audio"
  prep_dir = audio_dir / "prep"
  prep_dir.mkdir(parents=True)
  for name in (
    "call-a_0000_000000_030000.wav",
    "call-a_0001_027000_050000.wav",
    "call-b_0000_000000_030000.wav",
    "call-b_0001_027000_050000.wav",
  ):
    write_audio(prep_dir / name)
  manifest = {
    "segment_duration_seconds": 30.0,
    "overlap_seconds": 3.0,
    "sources": [
      {"source_file": "call-a.wav", "duration_seconds": 50.0, "chunk_count": 2},
      {"source_file": "call-b.wav", "duration_seconds": 50.0, "chunk_count": 2},
    ],
    "chunks": [
      {
        "source_file": "call-a.wav",
        "source_stem": "call-a",
        "chunk_file": "call-a_0000_000000_030000.wav",
        "chunk_index": 0,
        "start_seconds": 0.0,
        "end_seconds": 30.0,
        "duration_seconds": 30.0,
      },
      {
        "source_file": "call-a.wav",
        "source_stem": "call-a",
        "chunk_file": "call-a_0001_027000_050000.wav",
        "chunk_index": 1,
        "start_seconds": 27.0,
        "end_seconds": 50.0,
        "duration_seconds": 23.0,
      },
      {
        "source_file": "call-b.wav",
        "source_stem": "call-b",
        "chunk_file": "call-b_0000_000000_030000.wav",
        "chunk_index": 0,
        "start_seconds": 0.0,
        "end_seconds": 30.0,
        "duration_seconds": 30.0,
      },
      {
        "source_file": "call-b.wav",
        "source_stem": "call-b",
        "chunk_file": "call-b_0001_027000_050000.wav",
        "chunk_index": 1,
        "start_seconds": 27.0,
        "end_seconds": 50.0,
        "duration_seconds": 23.0,
      },
    ],
  }
  (prep_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
  sources = asr_wer.resolve_prepared_audio_files(audio_dir, requested_overlap=None)
  output_dir = audio_dir / "ground"
  output_dir.mkdir()

  monkeypatch.setattr(asr_wer, "transcribe_with_selected_endpoint", lambda **_: "Alpha")
  timestamps = iter([10.0, 11.0, 20.0, 21.0, 100.0, 101.0, 102.0, 103.0])
  monkeypatch.setattr(asr_wer.time, "perf_counter", lambda: next(timestamps))

  results = asr_wer.process_prepared_sources(
    args=build_args("ground", str(audio_dir), "--prep", "--batch", "1"),
    prepared_sources=sources,
    output_dir=output_dir,
    base_url="https://example.com",
    api_key=None,
  )

  assert [result.elapsed_seconds for result in results] == [10.0, 80.0]


def test_transcribe_prepared_chunk_starts_timer_inside_worker(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  """Prepared chunk workers start source timers when transcription begins."""

  source = asr_wer.AudioInput(path=tmp_path / "call.wav", stem="call", format="wav")
  chunk = asr_wer.PreparedChunk(
    audio=asr_wer.AudioInput(path=tmp_path / "prep" / "call_0000_000000_030000.wav", stem="chunk", format="wav"),
    source=source,
    index=0,
    start_seconds=0.0,
    end_seconds=30.0,
    duration_seconds=30.0,
  )
  started_by_source: dict[str, float] = {}
  timestamps = iter([123.0, 124.0])
  monkeypatch.setattr(asr_wer.time, "perf_counter", lambda: next(timestamps))
  monkeypatch.setattr(asr_wer, "transcribe_with_selected_endpoint", lambda **_: "Alpha")

  assert asr_wer.transcribe_prepared_chunk(
    args=build_args("ground", str(tmp_path), "--prep"),
    chunk=chunk,
    base_url="https://example.com",
    api_key=None,
    started_by_source=started_by_source,
    timing_lock=threading.Lock(),
  ) == ("Alpha", 124.0)

  assert started_by_source == {"call.wav": 123.0}


def test_prepared_chunk_audit_write_failure_marks_source_failed(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  """A failed per-chunk audit transcript write marks the parent source failed."""

  audio_dir = tmp_path / "audio"
  audio_dir.mkdir()
  write_audio(audio_dir / "call.wav")
  write_prep_manifest(audio_dir)
  source = asr_wer.resolve_prepared_audio_files(audio_dir, requested_overlap=None)[0]
  output_dir = audio_dir / "ground"
  (output_dir / "chunks").mkdir(parents=True)
  original_atomic_write_text = asr_wer.atomic_write_text

  def fail_chunk_audit_write(path: Path, text: str) -> None:
    """Fail only per-chunk audit transcript writes."""

    if path.parent.name == "chunks":
      raise OSError("audit write failed")
    original_atomic_write_text(path, text)

  monkeypatch.setattr(asr_wer, "transcribe_with_selected_endpoint", lambda **_: "Alpha")
  monkeypatch.setattr(asr_wer, "atomic_write_text", fail_chunk_audit_write)

  result = asr_wer.process_prepared_sources(
    args=build_args("ground", str(audio_dir), "--prep", "--batch", "1"),
    prepared_sources=[source],
    output_dir=output_dir,
    base_url="https://example.com",
    api_key=None,
  )[0]

  assert result.status == "failed"
  assert "audit write failed" in str(result.error_message)


def test_prepared_chunk_error_without_missing_chunks(tmp_path: Path) -> None:
  """Prepared source result preserves explicit chunk errors without missing chunks."""

  audio_dir = tmp_path / "audio"
  audio_dir.mkdir()
  write_audio(audio_dir / "call.wav")
  write_prep_manifest(audio_dir)
  source = asr_wer.resolve_prepared_audio_files(audio_dir, requested_overlap=None)[0]
  output_dir = audio_dir / "ground"
  output_dir.mkdir()

  result = asr_wer.build_prepared_source_result(
    args=build_args("ground", str(audio_dir), "--prep"),
    source=source,
    output_dir=output_dir,
    chunk_transcripts={0: "Alpha", 1: "Bravo"},
    chunk_errors=["provider failed"],
    elapsed_seconds=1.0,
  )

  assert result.status == "failed"
  assert result.error_message == "provider failed"


def test_prepared_eval_ground_read_failure_and_report_temperature(tmp_path: Path) -> None:
  """Prepared eval reports ground read failures and temperature metadata."""

  audio_dir = tmp_path / "audio"
  audio_dir.mkdir()
  write_audio(audio_dir / "call.wav")
  write_prep_manifest(audio_dir)
  source = asr_wer.resolve_prepared_audio_files(audio_dir, requested_overlap=None)[0]
  output_dir = audio_dir / "eval"
  output_dir.mkdir()
  ground_dir = audio_dir / "ground"
  ground_dir.mkdir()
  (ground_dir / "call_normalized.txt").write_bytes(b"\xff")

  result = asr_wer.build_prepared_source_result(
    args=build_args("eval", str(audio_dir), "--prep"),
    source=source,
    output_dir=output_dir,
    chunk_transcripts={0: "Alpha", 1: "Bravo"},
    chunk_errors=[],
    elapsed_seconds=1.0,
  )
  assert result.status == "failed"
  assert result.error_message is not None
  assert "decode" in result.error_message

  assert (
    asr_wer.resolve_report_temperature(
      build_args("ground", str(audio_dir), "--endpoint", "completions", "--completions-temperature", "0.7")
    )
    == "0.7"
  )
  assert (
    asr_wer.resolve_report_temperature(build_args("ground", str(audio_dir), "--transcriptions-temperature", "0.6"))
    == "0.6"
  )


def test_transcribe_audio_file_records_failures_and_eval_scores(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  audio_dir = tmp_path / "audio"
  audio_dir.mkdir()
  audio_path = write_audio(audio_dir / "clip.wav")
  ground_dir = audio_dir / "ground"
  ground_dir.mkdir()
  (ground_dir / "clip_normalized.txt").write_text("alpha", encoding="utf-8")

  def fail_transcribe(**_: object) -> str:
    raise ValueError("provider failed")

  monkeypatch.setattr(asr_wer, "transcribe_with_selected_endpoint", fail_transcribe)
  monkeypatch.setattr(asr_wer, "get_audio_duration_seconds", lambda path: 3.0)

  result = asr_wer.transcribe_audio_file(
    args=build_args("eval", str(audio_dir)),
    audio_file=asr_wer.AudioInput(path=audio_path, stem="clip", format="wav"),
    output_dir=tmp_path / "out",
    base_url="https://example.com",
    api_key=None,
  )

  assert result.status == "failed"
  assert result.error_message == "provider failed"
  assert result.duration_seconds == 3.0
  assert result.wer == 1.0
  assert "error=provider failed" in asr_wer.format_file_result(result, mode="eval")


def test_transcribe_audio_file_records_duration_failures_before_requests(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  audio_dir = tmp_path / "audio"
  audio_dir.mkdir()
  audio_path = write_audio(audio_dir / "clip.wav")

  def fail_transcribe(**_: object) -> str:
    raise AssertionError("request should not be sent after duration failure")

  monkeypatch.setattr(asr_wer, "transcribe_with_selected_endpoint", fail_transcribe)
  monkeypatch.setattr(
    asr_wer, "get_audio_duration_seconds", lambda path: (_ for _ in ()).throw(ValueError("bad audio"))
  )

  result = asr_wer.transcribe_audio_file(
    args=build_args("ground", str(audio_dir)),
    audio_file=asr_wer.AudioInput(path=audio_path, stem="clip", format="wav"),
    output_dir=tmp_path / "out",
    base_url="https://example.com",
    api_key=None,
  )

  assert result.status == "failed"
  assert result.error_message == "bad audio"


def test_transcribe_audio_file_records_eval_ground_read_failures(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  audio_dir = tmp_path / "audio"
  audio_dir.mkdir()
  audio_path = write_audio(audio_dir / "clip.wav")
  ground_dir = audio_dir / "ground"
  ground_dir.mkdir()
  (ground_dir / "clip_normalized.txt").write_bytes(b"\xff")
  monkeypatch.setattr(asr_wer, "get_audio_duration_seconds", lambda path: 2.0)
  monkeypatch.setattr(asr_wer, "transcribe_with_selected_endpoint", lambda **kwargs: "Alpha")

  result = asr_wer.transcribe_audio_file(
    args=build_args("eval", str(audio_dir)),
    audio_file=asr_wer.AudioInput(path=audio_path, stem="clip", format="wav"),
    output_dir=tmp_path / "out",
    base_url="https://example.com",
    api_key=None,
  )

  assert result.status == "failed"
  assert result.error_message is not None
  assert "decode" in result.error_message
  assert result.duration_seconds == 2.0


def test_endpoint_helpers_raise_on_error_responses(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  audio_dir = tmp_path / "audio"
  audio_dir.mkdir()
  audio_path = write_audio(audio_dir / "clip.wav")
  audio = asr_wer.AudioInput(path=audio_path, stem="clip", format="wav")

  monkeypatch.setattr(
    asr_simple,
    "send_json_request",
    lambda **kwargs: asr_simple.HttpExchange(
      method="POST",
      url=str(kwargs["url"]),
      request_headers={},
      request_body=kwargs["payload"],
      response_status=500,
      response_headers={},
      response_body_text="{}",
      response_json={},
    ),
  )
  with pytest.raises(ValueError, match="HTTP 500"):
    asr_wer.transcribe_with_completions(
      args=build_args("ground", str(audio_dir), "--endpoint", "completions"),
      audio_file=audio,
      base_url="https://example.com",
      api_key=None,
    )

  monkeypatch.setattr(
    asr_simple,
    "send_multipart_request",
    lambda **kwargs: asr_simple.HttpExchange(
      method="POST",
      url=str(kwargs["url"]),
      request_headers={},
      request_body=kwargs["fields"],
      response_status=400,
      response_headers={},
      response_body_text="{}",
      response_json={},
    ),
  )
  with pytest.raises(ValueError, match="HTTP 400"):
    asr_wer.transcribe_with_transcriptions(
      args=build_args("ground", str(audio_dir)),
      audio_file=audio,
      base_url="https://example.com",
      api_key=None,
    )


def test_plain_wer_and_normalizer_regressions() -> None:
  assert asr_wer.compute_plain_word_error_rate("alpha bravo charlie", "alpha bravo") == (1, 3, pytest.approx(1 / 3))
  assert asr_wer.compute_plain_word_error_rate("", "") == (0, 0, 0.0)
  assert asr_wer.compute_plain_word_error_rate("", "hallucinated speech") == (2, 0, 1.0)
  assert asr_wer.compute_aggregate_wer(errors=2, reference_words=0) == 1.0
  assert asr_wer.sanitize_output_field("a\tb\nc\rd\\e") == r"a\tb\nc\rd\\e"
  skipped = asr_wer.FileResult(
    audio=asr_wer.AudioInput(path=Path("skipped.wav"), stem="skipped", format="wav"),
    status="skipped",
    transcript="Alpha",
    normalized_transcript="alpha",
    output_path=Path("skipped.txt"),
    normalized_output_path=Path("skipped_normalized.txt"),
    elapsed_seconds=None,
    duration_seconds=100.0,
    rtfx=None,
    exact_word_count=1,
    normalized_word_count=1,
  )
  transcribed = asr_wer.FileResult(
    audio=asr_wer.AudioInput(path=Path("transcribed.wav"), stem="transcribed", format="wav"),
    status="transcribed",
    transcript="Bravo",
    normalized_transcript="bravo",
    output_path=Path("transcribed.txt"),
    normalized_output_path=Path("transcribed_normalized.txt"),
    elapsed_seconds=1.0,
    duration_seconds=10.0,
    rtfx=10.0,
    exact_word_count=1,
    normalized_word_count=1,
  )
  assert asr_wer.compute_aggregate_rtfx([skipped, transcribed], wall_elapsed_seconds=2.0) == 5.0
  unsafe = asr_wer.FileResult(
    audio=asr_wer.AudioInput(path=Path("bad\tname.wav"), stem="bad\tname", format="wav"),
    status="failed",
    transcript="",
    normalized_transcript="",
    output_path=Path("out") / "bad\tname.txt",
    normalized_output_path=Path("out") / "bad\tname_normalized.txt",
    elapsed_seconds=1.0,
    duration_seconds=1.0,
    rtfx=1.0,
    exact_word_count=0,
    normalized_word_count=0,
    error_message="line one\nline two",
  )
  assert "\n" not in asr_wer.format_file_result(unsafe, mode="ground")
  assert "line one\\nline two" in asr_wer.format_file_result(unsafe, mode="ground")
  row = asr_wer.render_report_row(unsafe, eval_mode=False)
  assert row.count("\t") == len(asr_wer.render_report_header(eval_mode=False).split("\t")) - 1
  assert "bad\\tname" in row
  assert asr_wer.normalize_transcript("Um, I can\u2019t analyse the colour in caf\u00e9 number twenty-one.") == (
    "i can not analyze the color in cafe number 21"
  )
  assert asr_wer.normalize_transcript("[noise] (aside) y\u2018all paid 1,200 dollars and one favour.") == (
    "you all paid 1200 dollars and one favor"
  )
  assert asr_wer.normalize_transcript("zero two thirty five") == "0 2 35"
  assert asr_wer.normalize_transcript("zero one two") == "0 1 2"
  assert asr_wer.normalize_transcript("0 one 2") == "0 1 2"


def test_duration_helper_uses_mutagen_file_and_handles_missing_length(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  audio_path = tmp_path / "clip.mp3"
  write_audio(audio_path)
  monkeypatch.setattr(asr_wer.mutagen, "File", lambda path: SimpleNamespace(info=SimpleNamespace(length=12.5)))

  assert asr_wer.get_audio_duration_seconds(audio_path) == 12.5

  monkeypatch.setattr(asr_wer.mutagen, "File", lambda path: SimpleNamespace(info=SimpleNamespace()))
  assert asr_wer.get_audio_duration_seconds(audio_path) == 0.0

  monkeypatch.setattr(asr_wer.mutagen, "File", lambda path: SimpleNamespace(info=SimpleNamespace(length=True)))
  assert asr_wer.get_audio_duration_seconds(audio_path) == 0.0
