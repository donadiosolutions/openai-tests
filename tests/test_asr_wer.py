from __future__ import annotations

import argparse
import json
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

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
  )

  assert args.mode == "eval"
  assert args.audio_dir == str(tmp_path)
  assert args.endpoint == "completions"
  assert args.batch == 3
  assert args.prompt == "Use product names."
  assert args.service_tier == "priority"
  assert args.completions_temperature == 0.2
  assert args.transcriptions_language == "en"


def test_configuration_errors_cover_batch_audio_discovery_and_prompt_conflicts(tmp_path: Path) -> None:
  with pytest.raises(ValueError, match="batch must be at least 1"):
    asr_wer.validate_args(build_args("ground", str(tmp_path), "--batch", "0"))

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

  collision_dir = tmp_path / "collisions"
  collision_dir.mkdir()
  write_audio(collision_dir / "clip.wav")
  write_audio(collision_dir / "clip_normalized.wav")
  with pytest.raises(ValueError, match="Output artifact collision"):
    asr_wer.discover_audio_files(collision_dir)

  reserved_dir = tmp_path / "reserved"
  reserved_dir.mkdir()
  write_audio(reserved_dir / "report.wav")
  with pytest.raises(ValueError, match="reserved output artifact"):
    asr_wer.discover_audio_files(reserved_dir)

  with pytest.raises(ValueError, match="prompt cannot be provided with transcriptions-prompt"):
    asr_wer.validate_args(build_args("ground", str(tmp_path), "--prompt", "A", "--transcriptions-prompt", "B"))

  with pytest.raises(ValueError, match="completions prompt flags cannot be used"):
    asr_wer.validate_args(build_args("ground", str(tmp_path), "--system-prompt", "ignored"))

  with pytest.raises(ValueError, match="prompt cannot be provided with completions prompt overrides"):
    asr_wer.validate_args(
      build_args("ground", str(tmp_path), "--endpoint", "completions", "--prompt", "A", "--user-prompt", "B")
    )

  with pytest.raises(ValueError, match="completions-messages-json cannot be used"):
    asr_wer.validate_args(
      build_args("ground", str(tmp_path), "--endpoint", "completions", "--completions-messages-json", "[]")
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


def test_resolve_endpoint_model_uses_transcriptions_default_without_shared_model() -> None:
  args = build_args("ground", "audio")

  assert asr_wer.resolve_endpoint_model(args) == asr_simple.DEFAULT_TRANSCRIPTIONS_MODEL

  args = build_args("ground", "audio", "--model", "shared-model")
  assert asr_wer.resolve_endpoint_model(args) == "shared-model"


def test_create_output_dir_avoids_eval_collisions(tmp_path: Path) -> None:
  args = build_args("eval", str(tmp_path))
  first = tmp_path / "model_1234"
  first.mkdir()

  assert asr_wer.create_output_dir(args, first) == tmp_path / "model_1234-1"
  assert (tmp_path / "model_1234-1").is_dir()


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

  def fake_read_text(
    path: Path,
    encoding: str | None = None,
    errors: str | None = None,
    newline: str | None = None,
  ) -> str:
    if path == ground_path:
      raise OSError("permission denied")
    return original_read_text(path, encoding=encoding, errors=errors, newline=newline)

  monkeypatch.setattr(Path, "read_text", fake_read_text)
  with pytest.raises(ValueError, match="permission denied"):
    asr_wer.validate_eval_ground(audio_dir, [asr_wer.AudioInput(path=audio_path, stem="clip", format="wav")])


def test_eval_rejects_empty_ground_normalized_transcripts(tmp_path: Path) -> None:
  audio_dir = tmp_path / "audio"
  audio_dir.mkdir()
  audio_path = write_audio(audio_dir / "clip.wav")
  ground_dir = audio_dir / "ground"
  ground_dir.mkdir()
  (ground_dir / "clip_normalized.txt").write_text(" \n\t", encoding="utf-8")

  with pytest.raises(ValueError, match="Empty normalized ground transcript"):
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
