from __future__ import annotations

import argparse
import io
import json
import subprocess
from email.message import Message
from pathlib import Path
from types import SimpleNamespace
from urllib import error, request

import pytest

from openai_tests.test_modules import asr_simple


class FakeResponse:
  def __init__(self, body: bytes, status: int = 200, headers: dict[str, str] | None = None) -> None:
    self._body = body
    self._status = status
    self.headers = headers or {"Content-Type": "application/json"}

  def __enter__(self) -> FakeResponse:
    return self

  def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
    return None

  def read(self) -> bytes:
    return self._body

  def getcode(self) -> int:
    return self._status


def build_args(**overrides: object) -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  asr_simple.configure_parser(parser)
  args = parser.parse_args([])
  for key, value in overrides.items():
    setattr(args, key, value)
  return args


def test_build_completions_request_config_uses_audio_message_and_overrides() -> None:
  args = build_args(
    model="gpt-common",
    completions_model="gpt-audio",
    completions_metadata_json='{"suite":"asr-simple"}',
    completions_modalities_json='["text"]',
    completions_response_format_json='{"type":"text"}',
    completions_temperature=0.2,
    completions_top_p=0.9,
    completions_tool_choice="none",
    completions_tools_json="[]",
    completions_stream=True,
  )
  payload = asr_simple.build_completions_request_config(args, b"WAVE", "wav")
  assert payload["model"] == "gpt-audio"
  assert payload["messages"] == [
    {
      "role": "system",
      "content": f"{asr_simple.DEFAULT_SYSTEM_PROMPT}\n\n{asr_simple.DEFAULT_DEVELOPER_PROMPT}",
    },
    {
      "role": "user",
      "content": [
        {"type": "text", "text": asr_simple.DEFAULT_USER_PROMPT},
        {"type": "input_audio", "input_audio": {"data": "V0FWRQ==", "format": "wav"}},
      ],
    },
  ]
  assert payload["metadata"] == {"suite": "asr-simple"}
  assert payload["modalities"] == ["text"]
  assert payload["response_format"] == {"type": "text"}
  assert payload["temperature"] == 0.2
  assert payload["top_p"] == 0.9
  assert payload["tool_choice"] == "none"
  assert payload["tools"] == []
  assert payload["stream"] is True


def test_build_completions_request_config_supports_json_inputs() -> None:
  args = build_args(
    completions_messages_json='[{"role":"user","content":"Use this instead."}]',
    completions_audio_json='{"voice":"alloy","format":"wav"}',
    completions_function_call_json='{"name":"transcribe"}',
    completions_functions_json='[{"name":"transcribe"}]',
    completions_logit_bias_json='{"42":1}',
    completions_prediction_json='{"type":"content","content":"Alpha"}',
    completions_stop_json='["stop"]',
    completions_stream_options_json='{"include_usage":true}',
    completions_tool_choice_json='{"type":"function","function":{"name":"transcribe"}}',
    completions_web_search_options_json='{"search_context_size":"low"}',
    completions_reasoning_effort="low",
    completions_prompt_cache_retention="in-memory",
  )
  payload = asr_simple.build_completions_request_config(args, b"WAVE", "wav")
  assert payload["messages"] == [{"role": "user", "content": "Use this instead."}]
  assert payload["audio"] == {"voice": "alloy", "format": "wav"}
  assert payload["function_call"] == {"name": "transcribe"}
  assert payload["functions"] == [{"name": "transcribe"}]
  assert payload["logit_bias"] == {"42": 1}
  assert payload["prediction"] == {"type": "content", "content": "Alpha"}
  assert payload["stop"] == ["stop"]
  assert payload["stream_options"] == {"include_usage": True}
  assert payload["tool_choice"] == {"type": "function", "function": {"name": "transcribe"}}
  assert payload["web_search_options"] == {"search_context_size": "low"}
  assert payload["reasoning_effort"] == "low"
  assert payload["prompt_cache_retention"] == "in-memory"


def test_build_request_configs_reject_invalid_combinations() -> None:
  args = build_args(completions_tool_choice="none", completions_tool_choice_json='{"type":"function"}')
  with pytest.raises(ValueError, match="tool_choice cannot be provided"):
    asr_simple.build_completions_request_config(args, b"WAVE", "wav")

  args = build_args(
    transcriptions_chunking_strategy="auto",
    transcriptions_chunking_strategy_json='{"type":"server_vad"}',
  )
  with pytest.raises(ValueError, match="chunking_strategy cannot be provided"):
    asr_simple.build_transcriptions_request_config(args)


def test_build_transcriptions_request_config_uses_defaults_and_overrides() -> None:
  args = build_args(
    model="gpt-common",
    transcriptions_model="gpt-asr",
    transcriptions_language="en",
    transcriptions_prompt="Spell the NATO alphabet words correctly.",
    transcriptions_response_format="verbose_json",
    transcriptions_temperature=0.1,
    transcriptions_timestamp_granularities=["word"],
    transcriptions_timestamp_granularities_json='["segment"]',
    transcriptions_include=["logprobs"],
    transcriptions_include_json='["extra"]',
    transcriptions_stream=True,
    transcriptions_chunking_strategy="auto",
    transcriptions_known_speaker_names=["agent"],
    transcriptions_known_speaker_names_json='["customer"]',
    transcriptions_known_speaker_references=["data:audio/wav;base64,AAA="],
    transcriptions_known_speaker_references_json='["data:audio/wav;base64,BBB="]',
  )
  payload = asr_simple.build_transcriptions_request_config(args)
  assert payload == {
    "chunking_strategy": "auto",
    "include": ["logprobs", "extra"],
    "known_speaker_names": ["agent", "customer"],
    "known_speaker_references": ["data:audio/wav;base64,AAA=", "data:audio/wav;base64,BBB="],
    "language": "en",
    "model": "gpt-asr",
    "prompt": "Spell the NATO alphabet words correctly.",
    "response_format": "verbose_json",
    "stream": True,
    "temperature": 0.1,
    "timestamp_granularities": ["word", "segment"],
  }


def test_build_transcriptions_request_config_uses_shared_model_when_specific_model_is_omitted() -> None:
  args = build_args(model="shared-asr-model", transcriptions_model=None)
  payload = asr_simple.build_transcriptions_request_config(args)
  assert payload["model"] == "shared-asr-model"


def test_resolve_transcriptions_model_uses_legacy_default_when_no_fallback_is_provided() -> None:
  assert asr_simple.resolve_transcriptions_model(None) == asr_simple.DEFAULT_TRANSCRIPTIONS_MODEL


def test_prepare_audio_cases_can_use_existing_file(tmp_path: Path) -> None:
  audio_path = tmp_path / "speech.wav"
  audio_path.write_bytes(b"RIFF")
  args = build_args(audio_file=str(audio_path), audio_format="wav", expected_transcript="Alpha Bravo")
  cases = asr_simple.prepare_audio_cases(args, tmp_path)
  assert len(cases) == 1
  assert cases[0].label == "speech.wav"
  assert cases[0].expected_transcript == "Alpha Bravo"
  assert cases[0].fixture.path == audio_path
  assert cases[0].fixture.format == "wav"
  assert cases[0].fixture.bytes == b"RIFF"


def test_prepare_audio_cases_use_repo_samples_by_default(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  monkeypatch.setattr(asr_simple, "REPO_ROOT", tmp_path)
  (tmp_path / "asr_default_nato.mp3").write_bytes(b"NATO")
  (tmp_path / "asr_default_pangram.mp3").write_bytes(b"PANGRAM")

  cases = asr_simple.prepare_audio_cases(build_args(), tmp_path)

  assert len(cases) == 2
  assert [case.label for case in cases] == ["nato alphabet", "quick brown fox"]
  assert [case.fixture.path.name for case in cases] == ["asr_default_nato.mp3", "asr_default_pangram.mp3"]
  assert [case.fixture.format for case in cases] == ["mp3", "mp3"]
  assert cases[0].expected_transcript == (
    "Alpha Bravo Charlie Delta Echo Foxtrot Golf Hotel India Juliet Kilo Lima Mike November Oscar Papa Quebec "
    "Romeo Sierra Tango Uniform Victor Whiskey X Ray Yankee Zulu"
  )
  assert cases[1].expected_transcript == "The quick brown fox jumps over the lazy dog"


def test_load_bundled_audio_fixture_falls_back_to_packaged_assets(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  asset_root = tmp_path / "assets"
  asset_root.mkdir()
  (asset_root / "asr_default_nato.mp3").write_bytes(b"NATO")

  monkeypatch.setattr(asr_simple, "REPO_ROOT", tmp_path / "missing-root")
  monkeypatch.setattr(asr_simple.resources, "files", lambda package: asset_root)

  fixture = asr_simple.load_bundled_audio_fixture(asr_simple.DEFAULT_BUNDLED_AUDIO_SAMPLES[0], tmp_path / "out")
  assert fixture.path == tmp_path / "out" / "asr_default_nato.mp3"
  assert fixture.bytes == b"NATO"
  assert fixture.format == "mp3"


def test_load_bundled_audio_fixture_reports_missing_packaged_asset(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  asset_root = tmp_path / "assets"
  asset_root.mkdir()

  monkeypatch.setattr(asr_simple, "REPO_ROOT", tmp_path / "missing-root")
  monkeypatch.setattr(asr_simple.resources, "files", lambda package: asset_root)
  with pytest.raises(ValueError, match="Bundled audio sample not found"):
    asr_simple.load_bundled_audio_fixture(asr_simple.DEFAULT_BUNDLED_AUDIO_SAMPLES[0], tmp_path / "out")

  def raise_module_not_found(package: str) -> Path:
    raise ModuleNotFoundError(package)

  monkeypatch.setattr(asr_simple.resources, "files", raise_module_not_found)
  with pytest.raises(ValueError, match="Bundled audio sample not found"):
    asr_simple.load_bundled_audio_fixture(asr_simple.DEFAULT_BUNDLED_AUDIO_SAMPLES[0], tmp_path / "out")


def test_prepare_audio_cases_synthesizes_espeak_audio(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  def fake_run(
    command: list[str],
    *,
    check: bool,
    capture_output: bool,
    text: bool,
  ) -> subprocess.CompletedProcess[str]:
    assert command == [
      "espeak-ng",
      "-v",
      "en-us",
      "-s",
      "150",
      "-w",
      str(tmp_path / "asr-simple.wav"),
      "Alpha Bravo",
    ]
    assert check is True
    assert capture_output is True
    assert text is True
    (tmp_path / "asr-simple.wav").write_bytes(b"RIFF")
    return subprocess.CompletedProcess(command, 0, "", "")

  monkeypatch.setattr(asr_simple.subprocess, "run", fake_run)
  cases = asr_simple.prepare_audio_cases(build_args(expected_transcript="Alpha Bravo"), tmp_path)
  assert len(cases) == 1
  assert cases[0].label == "synthesized"
  assert cases[0].expected_transcript == "Alpha Bravo"
  assert cases[0].fixture.path == tmp_path / "asr-simple.wav"
  assert cases[0].fixture.bytes == b"RIFF"


def test_prepare_audio_cases_report_missing_audio_and_espeak_failures(tmp_path: Path) -> None:
  with pytest.raises(ValueError, match="Audio file does not exist"):
    asr_simple.prepare_audio_cases(
      build_args(audio_file=str(tmp_path / "missing.wav"), expected_transcript="Alpha Bravo"),
      tmp_path,
    )

  directory_path = tmp_path / "directory.wav"
  directory_path.mkdir()
  with pytest.raises(ValueError, match="Audio path is not a file"):
    asr_simple.prepare_audio_cases(
      build_args(audio_file=str(directory_path), expected_transcript="Alpha Bravo"),
      tmp_path,
    )

  unsupported_path = tmp_path / "speech.flac"
  unsupported_path.write_bytes(b"RIFF")
  with pytest.raises(ValueError, match="Unsupported audio format"):
    asr_simple.prepare_audio_cases(
      build_args(audio_file=str(unsupported_path), audio_format="flac", expected_transcript="Alpha Bravo"),
      tmp_path,
    )

  with pytest.raises(ValueError, match="expected-transcript is required with --audio-file"):
    asr_simple.prepare_audio_cases(
      build_args(audio_file=str(tmp_path / "missing.wav"), expected_transcript=None), tmp_path
    )

  def missing_run(*_: object, **__: object) -> subprocess.CompletedProcess[str]:
    raise FileNotFoundError

  with pytest.MonkeyPatch.context() as monkeypatch:
    monkeypatch.setattr(asr_simple.subprocess, "run", missing_run)
    with pytest.raises(ValueError, match="espeak-ng was not found"):
      asr_simple.prepare_audio_cases(build_args(expected_transcript="Alpha Bravo"), tmp_path)

  def fail_run(*_: object, **__: object) -> subprocess.CompletedProcess[str]:
    raise subprocess.CalledProcessError(1, ["espeak-ng"], stderr="voice missing")

  args = build_args(expected_transcript="Alpha Bravo")
  with pytest.MonkeyPatch.context() as monkeypatch:
    monkeypatch.setattr(asr_simple.subprocess, "run", fail_run)
    with pytest.raises(ValueError, match="espeak-ng failed"):
      asr_simple.prepare_audio_cases(args, tmp_path)


def test_send_multipart_request_sends_repeated_fields_and_file(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  audio_path = tmp_path / "speech.wav"
  audio_path.write_bytes(b"RIFF")
  captured_headers: dict[str, str] = {}
  captured_body = bytearray()

  def fake_urlopen(http_request: request.Request, timeout: float) -> FakeResponse:
    assert timeout == 5.0
    captured_headers.update(dict(http_request.header_items()))
    assert isinstance(http_request.data, bytes)
    captured_body.extend(http_request.data)
    return FakeResponse(b'{"text":"Alpha Bravo"}', status=200)

  monkeypatch.setattr(asr_simple.request, "urlopen", fake_urlopen)
  exchange = asr_simple.send_multipart_request(
    url="https://example.com/v1/audio/transcriptions",
    api_key="secret",
    fields={"model": "gpt-asr", "include": ["logprobs"], "stream": False},
    file_path=audio_path,
    file_format="wav",
    timeout=5.0,
  )
  body = bytes(captured_body)
  assert exchange.response_status == 200
  assert exchange.response_json == {"text": "Alpha Bravo"}
  assert exchange.request_body["file"] == {"filename": "speech.wav", "content_type": "audio/wav", "size": 4}
  assert captured_headers["Authorization"] == "Bearer secret"
  assert b'name="include[]"' in body
  assert b'name="stream"\r\n\r\nfalse' in body
  assert b"RIFF" in body


def test_send_multipart_request_handles_http_and_url_errors(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  audio_path = tmp_path / "speech.wav"
  audio_path.write_bytes(b"RIFF")

  def fake_http_error(_: object, timeout: float) -> FakeResponse:
    headers = Message()
    headers["Content-Type"] = "application/json"
    raise error.HTTPError(
      url="https://example.com/v1/audio/transcriptions",
      code=400,
      msg="Bad Request",
      hdrs=headers,
      fp=io.BytesIO(b'{"error":"bad request"}'),
    )

  monkeypatch.setattr(asr_simple.request, "urlopen", fake_http_error)
  exchange = asr_simple.send_multipart_request(
    url="https://example.com/v1/audio/transcriptions",
    api_key=None,
    fields={"model": "gpt-asr"},
    file_path=audio_path,
    file_format="wav",
    timeout=5.0,
  )
  assert exchange.response_status == 400
  assert exchange.response_json == {"error": "bad request"}
  assert exchange.error_message is not None

  def fake_url_error(_: object, timeout: float) -> FakeResponse:
    raise error.URLError("no route")

  monkeypatch.setattr(asr_simple.request, "urlopen", fake_url_error)
  exchange = asr_simple.send_multipart_request(
    url="https://example.com/v1/audio/transcriptions",
    api_key=None,
    fields={"model": "gpt-asr"},
    file_path=audio_path,
    file_format="wav",
    timeout=5.0,
  )
  assert exchange.response_status is None
  assert exchange.error_message == "no route"


def test_extract_completions_response_text_handles_json_and_stream() -> None:
  assert (
    asr_simple.extract_completions_response_text(
      {"choices": [{"message": {"content": "Alpha Bravo"}}]},
      "",
      stream=False,
    )
    == "Alpha Bravo"
  )
  stream_body = (
    'data: {"choices":[{"delta":{"content":"Alpha "}}]}\n\n'
    'data: {"choices":[{"delta":{"content":"Bravo"}}]}\n\n'
    "data: [DONE]\n\n"
  )
  assert asr_simple.extract_completions_response_text(None, stream_body, stream=True) == "Alpha Bravo"


def test_extract_transcription_response_text_handles_json_text_and_stream() -> None:
  assert asr_simple.extract_transcription_response_text({"text": "Alpha Bravo"}, "", stream=False) == "Alpha Bravo"
  assert (
    asr_simple.extract_transcription_response_text(
      {"segments": [{"text": " Alpha"}, {"text": " Bravo"}]},
      "",
      stream=False,
    )
    == "Alpha\nBravo"
  )
  assert asr_simple.extract_transcription_response_text(None, "Alpha Bravo", stream=False) == "Alpha Bravo"
  stream_body = (
    'data: {"type":"transcript.text.delta","delta":"Alpha "}\n\n'
    'data: {"type":"transcript.text.done","text":"Alpha Bravo"}\n\n'
  )
  assert asr_simple.extract_transcription_response_text(None, stream_body, stream=True) == "Alpha Bravo"


def test_build_accuracy_error_message_counts_normalized_expected_words() -> None:
  assert asr_simple.build_accuracy_error_message("Alpha, bravo delta", "Alpha Bravo Charlie Delta", 3) is None
  assert asr_simple.build_accuracy_error_message("Alpha bravo", "Alpha Bravo Charlie Delta", 3) == (
    "Transcript matched 2 of 4 expected words; required at least 3. Missing: charlie, delta."
  )
  assert asr_simple.resolve_minimum_expected_words(2, "Alpha Bravo Charlie") == 2
  with pytest.raises(ValueError, match="min-expected-words"):
    asr_simple.resolve_minimum_expected_words(0, "Alpha Bravo")


def test_build_accuracy_error_message_accepts_common_asr_word_splits() -> None:
  assert asr_simple.build_accuracy_error_message("Alpha fox trot xray", "Alpha Foxtrot X Ray", 3) is None


def test_build_accuracy_error_message_accepts_low_wer_even_if_one_word_is_missed() -> None:
  expected = (
    "Alpha Bravo Charlie Delta Echo Foxtrot Golf Hotel India Juliet Kilo Lima Mike November Oscar Papa Quebec Romeo "
    "Sierra Tango Uniform Victor Whiskey X Ray Yankee Zulu"
  )
  response = (
    "alpha bravo charlie delta echo foxtrot golf hotel india juliet keelor lima mike november oscar papa quebec romeo "
    "sierra tango uniform victor whiskey x-ray yankee zulu"
  )
  assert asr_simple.build_accuracy_error_message(response, expected, 26) is None


def test_build_accuracy_error_message_accepts_common_nato_spelling_variants() -> None:
  expected = (
    "Alpha Bravo Charlie Delta Echo Foxtrot Golf Hotel India Juliet Kilo Lima Mike November Oscar Papa Quebec Romeo "
    "Sierra Tango Uniform Victor Whiskey X Ray Yankee Zulu"
  )
  response = (
    "alfa bravo charly delta echo foxtrot golf hotel india juliette kelo lima mike november oskar papa quebec "
    "romeu sierra tango uniforme viktor whisky x-ray yanke zooloo"
  )
  assert asr_simple.build_accuracy_error_message(response, expected, 26) is None


def test_normalize_words_maps_representative_nato_spelling_variants_to_canonical_forms() -> None:
  assert asr_simple.normalize_words(
    "alfa bravo charley delter eko fokstrot gulf otel indya juliette keelo lema mic novemba "
    "oskar pappa kebec romeu siera tengo uniforme viktor whisky x-ray yankie zooloo"
  ) == [
    "alpha",
    "bravo",
    "charlie",
    "delta",
    "echo",
    "foxtrot",
    "golf",
    "hotel",
    "india",
    "juliet",
    "kilo",
    "lima",
    "mike",
    "november",
    "oscar",
    "papa",
    "quebec",
    "romeo",
    "sierra",
    "tango",
    "uniform",
    "victor",
    "whiskey",
    "xray",
    "yankee",
    "zulu",
  ]


def test_compute_word_error_rate_reports_edit_distance() -> None:
  errors, total_words, wer = asr_simple.compute_word_error_rate("Alpha Bravo Charlie", "Alpha Bravo")
  assert errors == 1
  assert total_words == 3
  assert wer == pytest.approx(1 / 3)


def test_compute_word_error_rate_handles_empty_reference() -> None:
  assert asr_simple.compute_word_error_rate("", "Alpha Bravo") == (0, 0, 0.0)


def test_normalize_known_model_transcript_strips_qwen_asr_wrapper() -> None:
  assert (
    asr_simple.normalize_known_model_transcript(
      "language English<asr_text>Alpha Bravo Charlie",
      requested_model="Qwen/Qwen3-ASR-1.7B",
    )
    == "Alpha Bravo Charlie"
  )


def test_normalize_known_model_transcript_is_model_specific() -> None:
  raw = "language English<asr_text>Alpha Bravo Charlie"
  assert asr_simple.normalize_known_model_transcript(raw, requested_model="ibm-granite/granite-speech-4.1-2b") == raw
  assert asr_simple.normalize_known_model_transcript(raw, requested_model=None) == raw
  assert asr_simple.normalize_known_model_transcript("", requested_model="Qwen/Qwen3-ASR-1.7B") == ""


def test_resolve_audio_format_uses_extension_and_requires_supported_suffix() -> None:
  assert asr_simple.resolve_audio_format(None, audio_path=Path("speech.mp3")) == "mp3"
  with pytest.raises(ValueError, match="audio-format is required"):
    asr_simple.resolve_audio_format(None, audio_path=Path("speech"))


def test_validate_response_format_reports_endpoint_specific_errors() -> None:
  exchange = asr_simple.HttpExchange(
    method="POST",
    url="https://example.com",
    request_headers={},
    request_body={},
    response_status=200,
    response_headers={"Content-Type": "application/json"},
    response_body_text="not-json",
    response_json=None,
  )
  assert asr_simple.validate_completions_response_format(exchange, stream=False) == "Expected a JSON object response."
  assert (
    asr_simple.validate_transcriptions_response_format(exchange, response_format="json", stream=False)
    == "Expected a JSON object response."
  )
  assert asr_simple.validate_transcriptions_response_format(exchange, response_format="text", stream=False) is None

  no_response = asr_simple.HttpExchange(
    method="POST",
    url="https://example.com",
    request_headers={},
    request_body={},
    response_status=None,
    response_headers={},
    response_body_text="",
    response_json=None,
  )
  assert asr_simple.validate_completions_response_format(no_response, stream=False) is None
  assert asr_simple.validate_transcriptions_response_format(no_response, response_format="json", stream=False) is None

  assert (
    asr_simple.validate_completions_response_format(exchange, stream=True) == "Expected a text/event-stream response."
  )
  assert (
    asr_simple.validate_transcriptions_response_format(exchange, response_format="json", stream=True)
    == "Expected a text/event-stream response."
  )
  stream_exchange = asr_simple.HttpExchange(
    method="POST",
    url="https://example.com",
    request_headers={},
    request_body={},
    response_status=200,
    response_headers={"Content-Type": "text/event-stream"},
    response_body_text='data: {"type":"transcript.text.done","text":"Alpha"}\n\n',
    response_json=None,
  )
  assert asr_simple.validate_completions_response_format(stream_exchange, stream=True) is None
  assert (
    asr_simple.validate_transcriptions_response_format(stream_exchange, response_format="json", stream=True) is None
  )

  assert (
    asr_simple.determine_asr_error_message(
      exchange=asr_simple.HttpExchange(
        method="POST",
        url="https://example.com",
        request_headers={},
        request_body={},
        response_status=500,
        response_headers={},
        response_body_text="{}",
        response_json={},
      ),
      response_text="Alpha",
      expected_transcript="Alpha",
      minimum_expected_words=1,
      format_error_message="format error",
    )
    == "HTTP 500"
  )
  assert (
    asr_simple.determine_asr_error_message(
      exchange=exchange,
      response_text="Alpha",
      expected_transcript="Alpha",
      minimum_expected_words=1,
      format_error_message="format error",
    )
    == "format error"
  )
  assert (
    asr_simple.determine_asr_error_message(
      exchange=asr_simple.HttpExchange(
        method="POST",
        url="https://example.com",
        request_headers={},
        request_body={},
        response_status=200,
        response_headers={"Content-Type": "application/json"},
        response_body_text=(
          '{"choices":[{"message":{"content":"alpha bravo charlie delta echo foxtrot golf hotel india juliet '
          "kelo lima mike november oscar papa quebec romeo sierra tango uniform victor whiskey x-ray yankee "
          'zulu"}}]}'
        ),
        response_json={},
      ),
      response_text=(
        "alpha bravo charlie delta echo foxtrot golf hotel india juliet kelo lima mike november oscar papa quebec "
        "romeo sierra tango uniform victor whiskey x-ray yankee zulu"
      ),
      expected_transcript=(
        "Alpha Bravo Charlie Delta Echo Foxtrot Golf Hotel India Juliet Kilo Lima Mike November Oscar Papa Quebec "
        "Romeo Sierra Tango Uniform Victor Whiskey X Ray Yankee Zulu"
      ),
      minimum_expected_words=26,
      format_error_message=None,
    )
    is None
  )


def test_warning_builders_report_mismatches_and_unavailable_tools() -> None:
  completions_warnings = asr_simple.build_completions_warnings(
    request_body={"model": "gpt-audio", "tool_choice": None, "tools": None},
    response_json={
      "model": "gpt-other",
      "choices": [
        {
          "message": {
            "tool_calls": [{"type": "function", "function": {"name": "transcribe"}}],
          }
        }
      ],
    },
  )
  assert completions_warnings == [
    'WARNING: argument model was sent as "gpt-audio" and returned as "gpt-other".',
    'WARNING: a tool call was returned for tool "transcribe", but no tools were available in the request.',
  ]

  transcriptions_warnings = asr_simple.build_transcriptions_warnings(
    request_body={"language": "en", "temperature": 0.0},
    response_json={"language": "english", "temperature": 0.0},
  )
  assert transcriptions_warnings == [
    'WARNING: argument language was sent as "en" and returned as "english".',
  ]


def test_warning_builders_ignore_non_dict_responses_and_available_tools() -> None:
  assert (
    asr_simple.build_completions_warnings(
      request_body={
        "model": "gpt-audio",
        "tools": [{"type": "function", "function": {"name": "transcribe"}}],
      },
      response_json={
        "model": "gpt-audio-2026-01-01",
        "choices": [
          {
            "message": {
              "tool_calls": [{"type": "function", "function": {"name": "transcribe"}}],
            }
          }
        ],
      },
    )
    == []
  )
  assert asr_simple.build_completions_warnings(request_body={}, response_json=["not-a-dict"]) == []
  assert asr_simple.build_transcriptions_warnings(request_body={}, response_json=["not-a-dict"]) == []


def test_warning_and_extraction_helpers_cover_malformed_shapes() -> None:
  assert asr_simple.build_completions_warnings(
    request_body={"tools": [{"type": "function", "function": {"name": "lookup"}}]},
    response_json={
      "choices": [
        {
          "message": {
            "tool_calls": [{"type": "function", "function": {"name": "transcribe"}}],
          }
        }
      ],
    },
  ) == ['WARNING: a tool call was returned for tool "transcribe", but that tool was not available in the request.']
  assert asr_simple.extract_completions_tool_call_names({"choices": "not-a-list"}) == []
  assert asr_simple.extract_completions_tool_call_names(
    {
      "choices": [
        "not-a-dict",
        {"message": "not-a-dict"},
        {"message": {"tool_calls": "not-a-list"}},
        {"message": {"tool_calls": ["not-a-dict", {"function": {}}, {"function": {"name": "transcribe"}}]}},
      ]
    }
  ) == ["transcribe"]
  assert asr_simple.format_multipart_scalar({"type": "server_vad"}) == '{"type": "server_vad"}'
  assert asr_simple.content_type_for_audio_format("unknown") == "application/octet-stream"

  malformed_stream = (
    "event: ignored\n\n"
    "data: [DONE]\n\n"
    "data: invalid-json\n\n"
    "data: []\n\n"
    'data: {"choices":"not-a-list"}\n\n'
    'data: {"choices":["not-a-dict",{"delta":"not-a-dict"},{"delta":{"content":""}},'
    '{"delta":{"content":[{"text":"Alpha"}]}}]}\n\n'
  )
  assert asr_simple.extract_completions_response_text(None, malformed_stream, stream=True) == "Alpha"
  assert (
    asr_simple.extract_transcription_response_text(
      None,
      'data: {"type":"transcript.text.delta","delta":"Alpha "}\n\n'
      'data: {"type":"transcript.text.delta","delta":"Bravo"}\n\n',
      stream=True,
    )
    == "Alpha Bravo"
  )
  assert (
    asr_simple.extract_completions_response_text(
      None,
      'data: {"choices":[{"delta":{"content":"Alpha "}}]}\r\n\r\n'
      'data: {"choices":[{"delta":{"content":"Bravo"}}]}\r\n\r\n',
      stream=True,
    )
    == "Alpha Bravo"
  )
  assert asr_simple.extract_transcription_response_text({"segments": ["skip", {"text": ""}]}, "", stream=False) == ""
  assert asr_simple.extract_transcription_response_text({"other": "value"}, "", stream=False) == ""


def test_endpoint_warning_checks_ignore_omitted_optional_fields(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  transcript = "Alpha Bravo Charlie Delta Echo Foxtrot Golf Hotel India Juliet"
  audio_path = tmp_path / "speech.wav"
  audio_path.write_bytes(b"RIFF")
  fixture = asr_simple.AudioFixture(path=audio_path, format="wav", bytes=b"RIFF")

  def fake_send_json_request(**kwargs: object) -> asr_simple.HttpExchange:
    payload = kwargs["payload"]
    assert isinstance(payload, dict)
    assert "n" not in payload
    return asr_simple.HttpExchange(
      method="POST",
      url=str(kwargs["url"]),
      request_headers={},
      request_body=payload,
      response_status=200,
      response_headers={},
      response_body_text=f'{{"n":1,"choices":[{{"message":{{"content":"{transcript}"}}}}]}}',
      response_json={"n": 1, "choices": [{"message": {"content": transcript}}]},
    )

  def fake_send_multipart_request(**kwargs: object) -> asr_simple.HttpExchange:
    fields = kwargs["fields"]
    assert isinstance(fields, dict)
    assert "language" not in fields
    return asr_simple.HttpExchange(
      method="POST",
      url=str(kwargs["url"]),
      request_headers={},
      request_body=fields,
      response_status=200,
      response_headers={},
      response_body_text=f'{{"language":"en","text":"{transcript}"}}',
      response_json={"language": "en", "text": transcript},
    )

  monkeypatch.setattr(asr_simple, "send_json_request", fake_send_json_request)
  monkeypatch.setattr(asr_simple, "send_multipart_request", fake_send_multipart_request)

  completions_result = asr_simple.run_completions_test(
    base_url="https://example.com",
    api_key=None,
    normalized_payload={"model": "gpt-audio", "messages": [], "n": None},
    expected_transcript=transcript,
    minimum_expected_words=10,
    timeout=5.0,
  )
  transcriptions_result = asr_simple.run_transcriptions_test(
    base_url="https://example.com",
    api_key=None,
    normalized_payload={"model": "gpt-asr", "language": None},
    audio_fixture=fixture,
    expected_transcript=transcript,
    minimum_expected_words=10,
    timeout=5.0,
  )

  assert completions_result.warnings == ()
  assert completions_result.partial_success is False
  assert transcriptions_result.warnings == ()
  assert transcriptions_result.partial_success is False


def test_run_executes_all_default_samples_and_renders_verbose_output(
  monkeypatch: pytest.MonkeyPatch,
  capsys: pytest.CaptureFixture[str],
  tmp_path: Path,
) -> None:
  audio_path = tmp_path / "speech.wav"
  audio_path.write_bytes(b"RIFF")
  sent_json_payloads: list[tuple[str, dict[str, object]]] = []
  sent_multipart_payloads: list[tuple[str, dict[str, object]]] = []

  def fake_prepare_audio_cases(args: argparse.Namespace, output_dir: Path) -> list[SimpleNamespace]:
    assert output_dir.exists()
    fixture = asr_simple.AudioFixture(path=audio_path, format="mp3", bytes=b"RIFF")
    return [
      SimpleNamespace(
        label="nato alphabet",
        fixture=fixture,
        expected_transcript="Alpha Bravo Charlie",
        minimum_expected_words=3,
      ),
      SimpleNamespace(
        label="quick brown fox",
        fixture=fixture,
        expected_transcript="The quick brown fox jumps over the lazy dog",
        minimum_expected_words=9,
      ),
    ]

  def fake_send_json_request(
    *, url: str, api_key: str | None, payload: dict[str, object], timeout: float
  ) -> asr_simple.HttpExchange:
    assert api_key == "cli-key"
    assert timeout == 12.0
    sent_json_payloads.append((url, payload))
    transcript = (
      "Alpha Bravo Charlie" if len(sent_json_payloads) == 1 else "The quick brown fox jumps over the lazy dog"
    )
    return asr_simple.HttpExchange(
      method="POST",
      url=url,
      request_headers={"Authorization": "Bearer cli-key"},
      request_body=payload,
      response_status=200,
      response_headers={"Content-Type": "application/json"},
      response_body_text=json.dumps({"choices": [{"message": {"content": transcript}}]}),
      response_json={"choices": [{"message": {"content": transcript}}]},
    )

  def fake_send_multipart_request(
    *,
    url: str,
    api_key: str | None,
    fields: dict[str, object],
    file_path: Path,
    file_format: str,
    timeout: float,
  ) -> asr_simple.HttpExchange:
    assert api_key == "cli-key"
    assert file_path == audio_path
    assert file_format == "mp3"
    assert timeout == 12.0
    sent_multipart_payloads.append((url, fields))
    transcript = (
      "Alpha Bravo Charlie" if len(sent_multipart_payloads) == 1 else "The quick brown fox jumps over the lazy dog"
    )
    return asr_simple.HttpExchange(
      method="POST",
      url=url,
      request_headers={"Authorization": "Bearer cli-key"},
      request_body=fields,
      response_status=200,
      response_headers={"Content-Type": "application/json"},
      response_body_text=json.dumps({"text": transcript}),
      response_json={"text": transcript},
    )

  monkeypatch.setattr(asr_simple, "prepare_audio_cases", fake_prepare_audio_cases)
  monkeypatch.setattr(asr_simple, "send_json_request", fake_send_json_request)
  monkeypatch.setattr(asr_simple, "send_multipart_request", fake_send_multipart_request)
  args = build_args(
    base_url="https://example.com/v1",
    api_key="cli-key",
    model="gpt-shared",
    transcriptions_model="gpt-asr",
    timeout=12.0,
    verbose=True,
  )
  assert asr_simple.run(args) == 0
  captured = capsys.readouterr()
  assert "/v1/chat/completions (nato alphabet):" in captured.out
  assert "/v1/audio/transcriptions (quick brown fox):" in captured.out
  assert "Expected transcript: Alpha Bravo Charlie" in captured.out
  assert "Expected transcript: The quick brown fox jumps over the lazy dog" in captured.out
  assert "WER:" in captured.out
  assert "***REDACTED***" in captured.out
  assert "Overall:" in captured.out
  assert len(sent_json_payloads) == 2
  assert len(sent_multipart_payloads) == 2
  assert sent_json_payloads[0][0] == "https://example.com/v1/chat/completions"
  assert sent_multipart_payloads[0][0] == "https://example.com/v1/audio/transcriptions"
  assert sent_json_payloads[0][1]["model"] == "gpt-shared"
  assert sent_multipart_payloads[0][1]["model"] == "gpt-asr"


def test_run_returns_failure_for_missing_words(
  monkeypatch: pytest.MonkeyPatch,
  capsys: pytest.CaptureFixture[str],
  tmp_path: Path,
) -> None:
  audio_path = tmp_path / "speech.wav"
  audio_path.write_bytes(b"RIFF")

  monkeypatch.setattr(
    asr_simple,
    "prepare_audio_cases",
    lambda args, output_dir: [
      SimpleNamespace(
        label="nato alphabet",
        fixture=asr_simple.AudioFixture(path=audio_path, format="mp3", bytes=b"RIFF"),
        expected_transcript="Alpha Bravo Charlie",
        minimum_expected_words=3,
      )
    ],
  )
  monkeypatch.setattr(
    asr_simple,
    "send_json_request",
    lambda **kwargs: asr_simple.HttpExchange(
      method="POST",
      url=kwargs["url"],
      request_headers={},
      request_body=kwargs["payload"],
      response_status=200,
      response_headers={},
      response_body_text='{"choices":[{"message":{"content":"Alpha Bravo"}}]}',
      response_json={"choices": [{"message": {"content": "Alpha Bravo"}}]},
    ),
  )
  monkeypatch.setattr(
    asr_simple,
    "send_multipart_request",
    lambda **kwargs: asr_simple.HttpExchange(
      method="POST",
      url=kwargs["url"],
      request_headers={},
      request_body=kwargs["fields"],
      response_status=200,
      response_headers={},
      response_body_text='{"text":"Alpha Bravo Charlie Delta Echo Foxtrot Golf Hotel India Juliet"}',
      response_json={"text": "Alpha Bravo Charlie Delta Echo Foxtrot Golf Hotel India Juliet"},
    ),
  )
  assert asr_simple.run(build_args()) == 1
  captured = capsys.readouterr()
  assert "Transcript matched 2 of 3 expected words" in captured.out


def test_run_returns_partial_success_for_warnings(
  monkeypatch: pytest.MonkeyPatch,
  capsys: pytest.CaptureFixture[str],
  tmp_path: Path,
) -> None:
  audio_path = tmp_path / "speech.wav"
  audio_path.write_bytes(b"RIFF")
  transcript = "Alpha Bravo Charlie"

  monkeypatch.setattr(
    asr_simple,
    "prepare_audio_cases",
    lambda args, output_dir: [
      SimpleNamespace(
        label="nato alphabet",
        fixture=asr_simple.AudioFixture(path=audio_path, format="mp3", bytes=b"RIFF"),
        expected_transcript=transcript,
        minimum_expected_words=3,
      )
    ],
  )
  monkeypatch.setattr(
    asr_simple,
    "send_json_request",
    lambda **kwargs: asr_simple.HttpExchange(
      method="POST",
      url=kwargs["url"],
      request_headers={},
      request_body=kwargs["payload"],
      response_status=200,
      response_headers={},
      response_body_text=f'{{"model":"changed","choices":[{{"message":{{"content":"{transcript}"}}}}]}}',
      response_json={"model": "changed", "choices": [{"message": {"content": transcript}}]},
    ),
  )
  monkeypatch.setattr(
    asr_simple,
    "send_multipart_request",
    lambda **kwargs: asr_simple.HttpExchange(
      method="POST",
      url=kwargs["url"],
      request_headers={},
      request_body=kwargs["fields"],
      response_status=200,
      response_headers={},
      response_body_text=f'{{"text":"{transcript}"}}',
      response_json={"text": transcript},
    ),
  )
  assert asr_simple.run(build_args()) == 1
  captured = capsys.readouterr()
  assert "PARTIAL SUCCESS" in captured.out
  assert "WARNING: argument model was sent" in captured.out


def test_print_endpoint_result_skips_wer_for_empty_reference(capsys: pytest.CaptureFixture[str]) -> None:
  result = asr_simple.EndpointExecutionResult(
    name="/v1/chat/completions",
    question="",
    response_text="Alpha Bravo",
    success=True,
    exchange=asr_simple.HttpExchange(
      method="POST",
      url="https://example.com",
      request_headers={},
      request_body={},
      response_status=200,
      response_headers={},
      response_body_text='{"choices":[{"message":{"content":"Alpha Bravo"}}]}',
      response_json={"choices": [{"message": {"content": "Alpha Bravo"}}]},
    ),
  )
  asr_simple.print_endpoint_result(result, verbose=False)
  captured = capsys.readouterr()
  assert "WER:" not in captured.out


def test_run_returns_configuration_error_for_invalid_json(capsys: pytest.CaptureFixture[str]) -> None:
  args = build_args(completions_metadata_json="{invalid")
  assert asr_simple.run(args) == 2
  captured = capsys.readouterr()
  assert "Configuration error:" in captured.err
