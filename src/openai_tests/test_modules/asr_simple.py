from __future__ import annotations

import argparse
import base64
import json
import os
import re
import subprocess
import sys
import tempfile
import uuid
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any
from urllib import error, request

from ..core import EndpointTestModule
from ._shared import (
  EndpointExecutionResult,
  HttpExchange,
  build_api_url,
  build_http_exchange,
  choose_string_or_json,
  colorize_status,
  determine_endpoint_status,
  determine_error_message,
  determine_overall_status,
  extract_chat_response_text,
  find_argument_mismatch_warnings,
  format_json_like,
  format_json_scalar,
  normalize_text_content,
  parse_json_dict,
  parse_json_list,
  prune_none,
  redact_headers,
  send_json_request,
  tool_is_available,
  try_parse_json,
)

DEFAULT_BASE_URL = "https://api.openai.com"
DEFAULT_COMPLETIONS_MODEL = "gpt-4o-audio-preview"
DEFAULT_TRANSCRIPTIONS_MODEL = "gpt-4o-transcribe"
DEFAULT_SYSTEM_PROMPT = "You are a precise speech recognition assistant."
DEFAULT_DEVELOPER_PROMPT = "Transcribe only the spoken English words from the audio."
DEFAULT_USER_PROMPT = "Transcribe this audio exactly."
DEFAULT_EXPECTED_TRANSCRIPT = "Alpha Bravo Charlie Delta Echo Foxtrot Golf Hotel India Juliet"
DEFAULT_NATO_TRANSCRIPT = (
  "Alpha Bravo Charlie Delta Echo Foxtrot Golf Hotel India Juliet Kilo Lima Mike November Oscar Papa Quebec Romeo "
  "Sierra Tango Uniform Victor Whiskey X Ray Yankee Zulu"
)
DEFAULT_PANGRAM_TRANSCRIPT = "The quick brown fox jumps over the lazy dog"
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_ESPEAK_VOICE = "en-us"
DEFAULT_ESPEAK_SPEED = 150
DEFAULT_SYNTHESIZED_AUDIO_FORMAT = "wav"
DEFAULT_MAX_WORD_ERROR_RATE = 0.15
REPO_ROOT = Path(__file__).resolve().parents[3]
WORD_VARIANT_GROUPS = {
  "alpha": ("alfa", "alfah", "alfer"),
  "bravo": ("brava", "brahvo", "bravoe"),
  "charlie": ("charli", "charly", "charley", "sharly"),
  "delta": ("delter", "dellta", "deltah"),
  "echo": ("eco", "ecko", "eko"),
  "foxtrot": ("fokstrot", "foxtrott", "foxtrot"),
  "golf": ("goff", "golph", "gulf"),
  "hotel": ("hoteil", "hotelh", "otel"),
  "india": ("indiah", "indiya", "indya"),
  "juliet": ("juliete", "juliett", "juliette"),
  "kilo": ("keelo", "kelo", "keylo", "killo"),
  "lima": ("leema", "lema", "lyma"),
  "mike": ("maik", "mic", "myke"),
  "november": ("novemba", "novemberr", "novemver"),
  "oscar": ("oscah", "oskar", "osker"),
  "papa": ("papah", "pappa", "poppa"),
  "quebec": ("kebec", "kwebec", "quebek", "quebeck"),
  "romeo": ("romeu", "romeyo", "romio"),
  "sierra": ("siarra", "siera", "syerra"),
  "tango": ("tanga", "tangoe", "tengo"),
  "uniform": ("uniforme", "unyform", "youniform"),
  "victor": ("viktor", "victorh", "viktor"),
  "whiskey": ("whiskey", "whisky", "wiskey"),
  "xray": ("exray", "xrei", "xrey"),
  "yankee": ("yanke", "yankie", "yanky"),
  "zulu": ("zooloo", "zoulou", "zuloo"),
}
WORD_VARIANT_ALIASES = {alias: canonical for canonical, aliases in WORD_VARIANT_GROUPS.items() for alias in aliases}

JSON_TRANSCRIPTION_FORMATS = frozenset(("json", "verbose_json", "diarized_json"))
TEXT_TRANSCRIPTION_FORMATS = frozenset(("text", "srt", "vtt"))
TRANSCRIPTION_CONTENT_TYPES = {
  "mp3": "audio/mpeg",
  "mp4": "audio/mp4",
  "mpeg": "audio/mpeg",
  "mpga": "audio/mpeg",
  "m4a": "audio/mp4",
  "wav": "audio/wav",
  "webm": "audio/webm",
}


@dataclass(frozen=True, slots=True)
class AudioFixture:
  path: Path
  format: str
  bytes: bytes


@dataclass(frozen=True, slots=True)
class BundledAudioSample:
  label: str
  filename: str
  expected_transcript: str
  format: str


@dataclass(frozen=True, slots=True)
class AudioCase:
  label: str
  fixture: AudioFixture
  expected_transcript: str
  minimum_expected_words: int


DEFAULT_BUNDLED_AUDIO_SAMPLES = (
  BundledAudioSample(
    label="nato alphabet",
    filename="asr_default_nato.mp3",
    expected_transcript=DEFAULT_NATO_TRANSCRIPT,
    format="mp3",
  ),
  BundledAudioSample(
    label="quick brown fox",
    filename="asr_default_pangram.mp3",
    expected_transcript=DEFAULT_PANGRAM_TRANSCRIPT,
    format="mp3",
  ),
)


def configure_parser(parser: argparse.ArgumentParser) -> None:
  parser.add_argument(
    "--base-url",
    help="Base URL for the OpenAI-compatible API. Defaults to $OPENAI_BASE_URL or OpenAI's public API.",
  )
  parser.add_argument(
    "--model",
    help=(
      "Model to use for chat completions. Defaults to $OPENAI_MODEL, $OPENAI_TESTS_MODEL, "
      f"or {DEFAULT_COMPLETIONS_MODEL}."
    ),
  )
  parser.add_argument("--completions-model", help="Override the model used only for /v1/chat/completions.")
  parser.add_argument(
    "--transcriptions-model",
    help=(
      "Override the model used only for /v1/audio/transcriptions. Defaults to "
      "$OPENAI_TRANSCRIPTIONS_MODEL, $OPENAI_TESTS_TRANSCRIPTIONS_MODEL, or the resolved shared --model value."
    ),
  )
  parser.add_argument(
    "--api-key",
    help="API key to use. Defaults to $OPENAI_API_KEY or $OPENAI_TESTS_API_KEY when available.",
  )
  parser.add_argument(
    "--timeout",
    type=float,
    default=DEFAULT_TIMEOUT_SECONDS,
    help="HTTP timeout in seconds. Defaults to 30.",
  )
  parser.add_argument(
    "-v",
    "--verbose",
    action="store_true",
    help="Show the full HTTP requests and responses for both endpoints.",
  )

  audio = parser.add_argument_group("Audio fixture")
  audio.add_argument("--audio-file", help="Use an existing audio file instead of synthesizing one with espeak-ng.")
  audio.add_argument(
    "--audio-format",
    help="Audio format for --audio-file. Defaults to the file extension when omitted.",
  )
  audio.add_argument(
    "--expected-transcript",
    help=(
      "Expected transcript for --audio-file. When provided without --audio-file, espeak-ng synthesizes this text "
      "into a temporary WAV fixture. When omitted, the bundled default MP3 samples are used."
    ),
  )
  audio.add_argument(
    "--min-expected-words",
    type=int,
    help="Minimum number of expected words that must appear in each returned transcript. Defaults to all words.",
  )
  audio.add_argument("--espeak-voice", default=DEFAULT_ESPEAK_VOICE)
  audio.add_argument("--espeak-speed", type=int, default=DEFAULT_ESPEAK_SPEED)

  prompts = parser.add_argument_group("Prompt text")
  prompts.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
  prompts.add_argument("--developer-prompt", default=DEFAULT_DEVELOPER_PROMPT)
  prompts.add_argument("--user-prompt", default=DEFAULT_USER_PROMPT)

  completions = parser.add_argument_group("Chat completions API parameters")
  completions.add_argument("--completions-audio-json")
  completions.add_argument("--completions-frequency-penalty", type=float)
  completions.add_argument("--completions-function-call")
  completions.add_argument("--completions-function-call-json")
  completions.add_argument("--completions-functions-json")
  completions.add_argument("--completions-logit-bias-json")
  completions.add_argument("--completions-logprobs", action=argparse.BooleanOptionalAction, default=None)
  completions.add_argument("--completions-max-completion-tokens", type=int)
  completions.add_argument("--completions-max-tokens", type=int)
  completions.add_argument("--completions-messages-json")
  completions.add_argument("--completions-metadata-json")
  completions.add_argument("--completions-modalities-json")
  completions.add_argument("--completions-n", type=int)
  completions.add_argument("--completions-parallel-tool-calls", action=argparse.BooleanOptionalAction, default=None)
  completions.add_argument("--completions-prediction-json")
  completions.add_argument("--completions-presence-penalty", type=float)
  completions.add_argument("--completions-prompt-cache-key")
  completions.add_argument(
    "--completions-prompt-cache-retention",
    choices=("in-memory", "24h"),
  )
  completions.add_argument(
    "--completions-reasoning-effort",
    choices=("minimal", "low", "medium", "high", "xhigh"),
  )
  completions.add_argument("--completions-response-format-json")
  completions.add_argument("--completions-safety-identifier")
  completions.add_argument("--completions-seed", type=int)
  completions.add_argument(
    "--completions-service-tier",
    choices=("auto", "default", "flex", "scale", "priority"),
  )
  completions.add_argument("--completions-stop")
  completions.add_argument("--completions-stop-json")
  completions.add_argument("--completions-store", action=argparse.BooleanOptionalAction, default=None)
  completions.add_argument("--completions-stream", action=argparse.BooleanOptionalAction, default=None)
  completions.add_argument("--completions-stream-options-json")
  completions.add_argument("--completions-temperature", type=float)
  completions.add_argument("--completions-tool-choice")
  completions.add_argument("--completions-tool-choice-json")
  completions.add_argument("--completions-tools-json")
  completions.add_argument("--completions-top-logprobs", type=int)
  completions.add_argument("--completions-top-p", type=float)
  completions.add_argument("--completions-user")
  completions.add_argument("--completions-web-search-options-json")

  transcriptions = parser.add_argument_group("Transcriptions API parameters")
  transcriptions.add_argument("--transcriptions-chunking-strategy")
  transcriptions.add_argument("--transcriptions-chunking-strategy-json")
  transcriptions.add_argument("--transcriptions-include", action="append")
  transcriptions.add_argument("--transcriptions-include-json")
  transcriptions.add_argument("--transcriptions-known-speaker-names", action="append")
  transcriptions.add_argument("--transcriptions-known-speaker-names-json")
  transcriptions.add_argument("--transcriptions-known-speaker-references", action="append")
  transcriptions.add_argument("--transcriptions-known-speaker-references-json")
  transcriptions.add_argument("--transcriptions-language")
  transcriptions.add_argument("--transcriptions-prompt")
  transcriptions.add_argument(
    "--transcriptions-response-format",
    choices=("json", "text", "srt", "verbose_json", "vtt", "diarized_json"),
  )
  transcriptions.add_argument("--transcriptions-stream", action=argparse.BooleanOptionalAction, default=None)
  transcriptions.add_argument("--transcriptions-temperature", type=float)
  transcriptions.add_argument(
    "--transcriptions-timestamp-granularities",
    action="append",
    choices=("word", "segment"),
  )
  transcriptions.add_argument("--transcriptions-timestamp-granularities-json")


def run(args: argparse.Namespace) -> int:
  try:
    base_url = resolve_base_url(args.base_url)
    api_key = resolve_api_key(args.api_key)
    with tempfile.TemporaryDirectory(prefix="openai-tests-asr-") as temporary_directory:
      audio_cases = prepare_audio_cases(args, Path(temporary_directory))
      transcriptions_payload = build_transcriptions_request_config(args)
      results: list[EndpointExecutionResult] = []
      for audio_case in audio_cases:
        completions_result = run_completions_test(
          base_url=base_url,
          api_key=api_key,
          normalized_payload=build_completions_request_config(
            args, audio_case.fixture.bytes, audio_case.fixture.format
          ),
          expected_transcript=audio_case.expected_transcript,
          minimum_expected_words=audio_case.minimum_expected_words,
          timeout=args.timeout,
          case_label=audio_case.label,
        )
        transcriptions_result = run_transcriptions_test(
          base_url=base_url,
          api_key=api_key,
          normalized_payload=transcriptions_payload,
          audio_fixture=audio_case.fixture,
          expected_transcript=audio_case.expected_transcript,
          minimum_expected_words=audio_case.minimum_expected_words,
          timeout=args.timeout,
          case_label=audio_case.label,
        )
        results.extend((completions_result, transcriptions_result))
  except ValueError as exc:
    print(f"Configuration error: {exc}", file=sys.stderr)
    return 2

  for index, result in enumerate(results):
    if index:
      print()
    print_endpoint_result(result, verbose=args.verbose)
  print()

  overall_status = determine_overall_status(*results)
  print(f"Overall: {colorize_status(overall_status)}")
  return 0 if overall_status == "passed" else 1


def resolve_base_url(cli_value: str | None) -> str:
  return cli_value or os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_TESTS_BASE_URL") or DEFAULT_BASE_URL


def resolve_model(cli_value: str | None) -> str:
  return cli_value or os.getenv("OPENAI_MODEL") or os.getenv("OPENAI_TESTS_MODEL") or DEFAULT_COMPLETIONS_MODEL


def resolve_transcriptions_model(cli_value: str | None) -> str:
  return resolve_transcriptions_model_with_fallback(cli_value, fallback_model=None)


def resolve_transcriptions_model_with_fallback(cli_value: str | None, *, fallback_model: str | None) -> str:
  return (
    cli_value
    or os.getenv("OPENAI_TRANSCRIPTIONS_MODEL")
    or os.getenv("OPENAI_TESTS_TRANSCRIPTIONS_MODEL")
    or fallback_model
    or DEFAULT_TRANSCRIPTIONS_MODEL
  )


def resolve_api_key(cli_value: str | None) -> str | None:
  return cli_value or os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_TESTS_API_KEY")


def resolve_minimum_expected_words(cli_value: int | None, expected_transcript: str) -> int:
  expected_word_count = len(normalize_words(expected_transcript))
  if cli_value is None:
    return expected_word_count
  if cli_value < 1:
    raise ValueError("min-expected-words must be at least 1")
  return cli_value


def prepare_audio_cases(args: argparse.Namespace, output_dir: Path) -> list[AudioCase]:
  if args.audio_file is not None:
    expected_transcript = require_expected_transcript(args.expected_transcript, context="--audio-file")
    fixture = load_audio_fixture(Path(args.audio_file), args.audio_format)
    return [
      AudioCase(
        label=fixture.path.name,
        fixture=fixture,
        expected_transcript=expected_transcript,
        minimum_expected_words=resolve_minimum_expected_words(args.min_expected_words, expected_transcript),
      )
    ]

  if args.expected_transcript is not None:
    expected_transcript = require_expected_transcript(args.expected_transcript, context="synthesized audio")
    fixture = synthesize_audio_fixture(
      expected_transcript=expected_transcript,
      output_dir=output_dir,
      voice=args.espeak_voice,
      speed=args.espeak_speed,
    )
    return [
      AudioCase(
        label="synthesized",
        fixture=fixture,
        expected_transcript=expected_transcript,
        minimum_expected_words=resolve_minimum_expected_words(args.min_expected_words, expected_transcript),
      )
    ]

  return [
    AudioCase(
      label=sample.label,
      fixture=load_bundled_audio_fixture(sample, output_dir),
      expected_transcript=sample.expected_transcript,
      minimum_expected_words=resolve_minimum_expected_words(args.min_expected_words, sample.expected_transcript),
    )
    for sample in DEFAULT_BUNDLED_AUDIO_SAMPLES
  ]


def require_expected_transcript(raw_value: str | None, *, context: str) -> str:
  if raw_value is None or not raw_value.strip():
    raise ValueError(f"expected-transcript is required with {context}")
  return raw_value.strip()


def load_audio_fixture(audio_path: Path, raw_audio_format: str | None) -> AudioFixture:
  if not audio_path.exists():
    raise ValueError(f"Audio file does not exist: {audio_path}")
  if not audio_path.is_file():
    raise ValueError(f"Audio path is not a file: {audio_path}")
  audio_format = resolve_audio_format(raw_audio_format, audio_path=audio_path)
  return AudioFixture(path=audio_path, format=audio_format, bytes=audio_path.read_bytes())


def load_bundled_audio_fixture(sample: BundledAudioSample, output_dir: Path) -> AudioFixture:
  source_path = REPO_ROOT / sample.filename
  if source_path.is_file():
    return AudioFixture(path=source_path, format=sample.format, bytes=source_path.read_bytes())

  try:
    asset_file = resources.files("openai_tests.assets") / sample.filename
  except ModuleNotFoundError as exc:
    raise ValueError(f"Bundled audio sample not found: {sample.filename}") from exc
  if not asset_file.is_file():
    raise ValueError(f"Bundled audio sample not found: {sample.filename}")

  output_dir.mkdir(parents=True, exist_ok=True)
  audio_path = output_dir / sample.filename
  audio_bytes = asset_file.read_bytes()
  audio_path.write_bytes(audio_bytes)
  return AudioFixture(path=audio_path, format=sample.format, bytes=audio_bytes)


def synthesize_audio_fixture(
  *,
  expected_transcript: str,
  output_dir: Path,
  voice: str,
  speed: int,
) -> AudioFixture:
  output_dir.mkdir(parents=True, exist_ok=True)
  audio_path = output_dir / f"asr-simple.{DEFAULT_SYNTHESIZED_AUDIO_FORMAT}"
  command = [
    "espeak-ng",
    "-v",
    voice,
    "-s",
    str(speed),
    "-w",
    str(audio_path),
    expected_transcript,
  ]
  try:
    subprocess.run(
      command,
      check=True,
      capture_output=True,
      text=True,
    )
  except FileNotFoundError as exc:
    raise ValueError("espeak-ng was not found; install it or pass --audio-file") from exc
  except subprocess.CalledProcessError as exc:
    stderr = exc.stderr.strip() if isinstance(exc.stderr, str) else str(exc.stderr)
    raise ValueError(f"espeak-ng failed: {stderr}") from exc
  return AudioFixture(path=audio_path, format=DEFAULT_SYNTHESIZED_AUDIO_FORMAT, bytes=audio_path.read_bytes())


def resolve_audio_format(raw_format: str | None, *, audio_path: Path | None = None) -> str:
  if raw_format is not None:
    return normalize_audio_format(raw_format)
  if audio_path is not None and audio_path.suffix:
    return normalize_audio_format(audio_path.suffix)
  raise ValueError("audio-format is required when the audio file extension does not identify a supported format")


def normalize_audio_format(raw_format: str) -> str:
  audio_format = raw_format.lower().lstrip(".")
  if audio_format not in TRANSCRIPTION_CONTENT_TYPES:
    supported = ", ".join(sorted(TRANSCRIPTION_CONTENT_TYPES))
    raise ValueError(f"Unsupported audio format {raw_format!r}; expected one of: {supported}")
  return audio_format


def build_completions_request_config(
  args: argparse.Namespace,
  audio_bytes: bytes,
  audio_format: str,
) -> dict[str, Any]:
  function_call = choose_string_or_json(
    string_value=args.completions_function_call,
    json_value=args.completions_function_call_json,
    field_name="function_call",
  )
  stop = choose_string_or_json(
    string_value=args.completions_stop,
    json_value=args.completions_stop_json,
    field_name="stop",
  )
  tool_choice = choose_string_or_json(
    string_value=args.completions_tool_choice,
    json_value=args.completions_tool_choice_json,
    field_name="tool_choice",
  )
  messages = parse_json_list(args.completions_messages_json, "completions-messages-json")
  if messages is None:
    messages = build_default_completions_messages(args, audio_bytes, audio_format)

  return {
    "audio": parse_json_dict(args.completions_audio_json, "completions-audio-json"),
    "frequency_penalty": args.completions_frequency_penalty,
    "function_call": function_call,
    "functions": parse_json_list(args.completions_functions_json, "completions-functions-json"),
    "logit_bias": parse_json_dict(args.completions_logit_bias_json, "completions-logit-bias-json"),
    "logprobs": args.completions_logprobs,
    "max_completion_tokens": args.completions_max_completion_tokens,
    "max_tokens": args.completions_max_tokens,
    "messages": messages,
    "metadata": parse_json_dict(args.completions_metadata_json, "completions-metadata-json"),
    "modalities": parse_json_list(args.completions_modalities_json, "completions-modalities-json"),
    "model": resolve_model(args.completions_model or args.model),
    "n": args.completions_n,
    "parallel_tool_calls": args.completions_parallel_tool_calls,
    "prediction": parse_json_dict(args.completions_prediction_json, "completions-prediction-json"),
    "presence_penalty": args.completions_presence_penalty,
    "prompt_cache_key": args.completions_prompt_cache_key,
    "prompt_cache_retention": args.completions_prompt_cache_retention,
    "reasoning_effort": args.completions_reasoning_effort,
    "response_format": parse_json_dict(args.completions_response_format_json, "completions-response-format-json"),
    "safety_identifier": args.completions_safety_identifier,
    "seed": args.completions_seed,
    "service_tier": args.completions_service_tier,
    "stop": stop,
    "store": args.completions_store,
    "stream": args.completions_stream,
    "stream_options": parse_json_dict(args.completions_stream_options_json, "completions-stream-options-json"),
    "temperature": args.completions_temperature,
    "tool_choice": tool_choice,
    "tools": parse_json_list(args.completions_tools_json, "completions-tools-json"),
    "top_logprobs": args.completions_top_logprobs,
    "top_p": args.completions_top_p,
    "user": args.completions_user,
    "web_search_options": parse_json_dict(
      args.completions_web_search_options_json,
      "completions-web-search-options-json",
    ),
  }


def build_default_completions_messages(
  args: argparse.Namespace,
  audio_bytes: bytes,
  audio_format: str,
) -> list[dict[str, Any]]:
  return [
    {
      "role": "system",
      "content": build_chat_system_prompt(args.system_prompt, args.developer_prompt),
    },
    {
      "role": "user",
      "content": [
        {"type": "text", "text": args.user_prompt},
        {
          "type": "input_audio",
          "input_audio": {
            "data": base64.b64encode(audio_bytes).decode("ascii"),
            "format": audio_format,
          },
        },
      ],
    },
  ]


def build_chat_system_prompt(system_prompt: str, developer_prompt: str) -> str:
  prompts = [prompt.strip() for prompt in (system_prompt, developer_prompt) if prompt.strip()]
  return "\n\n".join(prompts)


def build_transcriptions_request_config(args: argparse.Namespace) -> dict[str, Any]:
  timestamp_granularities = list(args.transcriptions_timestamp_granularities or [])
  timestamp_granularities_json = parse_json_list(
    args.transcriptions_timestamp_granularities_json,
    "transcriptions-timestamp-granularities-json",
  )
  if timestamp_granularities_json is not None:
    timestamp_granularities.extend(str(item) for item in timestamp_granularities_json)

  include = list(args.transcriptions_include or [])
  include_json = parse_json_list(args.transcriptions_include_json, "transcriptions-include-json")
  if include_json is not None:
    include.extend(str(item) for item in include_json)

  known_speaker_names = list(args.transcriptions_known_speaker_names or [])
  known_speaker_names_json = parse_json_list(
    args.transcriptions_known_speaker_names_json,
    "transcriptions-known-speaker-names-json",
  )
  if known_speaker_names_json is not None:
    known_speaker_names.extend(str(item) for item in known_speaker_names_json)

  known_speaker_references = list(args.transcriptions_known_speaker_references or [])
  known_speaker_references_json = parse_json_list(
    args.transcriptions_known_speaker_references_json,
    "transcriptions-known-speaker-references-json",
  )
  if known_speaker_references_json is not None:
    known_speaker_references.extend(str(item) for item in known_speaker_references_json)

  chunking_strategy = choose_string_or_json(
    string_value=args.transcriptions_chunking_strategy,
    json_value=args.transcriptions_chunking_strategy_json,
    field_name="chunking_strategy",
  )

  return {
    "chunking_strategy": chunking_strategy,
    "include": include or None,
    "known_speaker_names": known_speaker_names or None,
    "known_speaker_references": known_speaker_references or None,
    "language": args.transcriptions_language,
    "model": resolve_transcriptions_model_with_fallback(
      args.transcriptions_model,
      fallback_model=resolve_model(args.model),
    ),
    "prompt": args.transcriptions_prompt,
    "response_format": args.transcriptions_response_format,
    "stream": args.transcriptions_stream,
    "temperature": args.transcriptions_temperature,
    "timestamp_granularities": timestamp_granularities or None,
  }


def run_completions_test(
  *,
  base_url: str,
  api_key: str | None,
  normalized_payload: dict[str, Any],
  expected_transcript: str,
  minimum_expected_words: int,
  timeout: float,
  case_label: str | None = None,
) -> EndpointExecutionResult:
  pruned_payload = prune_none(normalized_payload)
  exchange = send_json_request(
    url=build_api_url(base_url, "/v1/chat/completions"),
    api_key=api_key,
    payload=pruned_payload,
    timeout=timeout,
  )
  stream = pruned_payload.get("stream") is True
  response_text = extract_completions_response_text(
    exchange.response_json,
    exchange.response_body_text,
    stream=stream,
  )
  response_text = normalize_known_model_transcript(
    response_text,
    requested_model=pruned_payload.get("model") if isinstance(pruned_payload.get("model"), str) else None,
  )
  error_message = determine_asr_error_message(
    exchange=exchange,
    response_text=response_text,
    expected_transcript=expected_transcript,
    minimum_expected_words=minimum_expected_words,
    format_error_message=validate_completions_response_format(
      exchange,
      stream=stream,
    ),
  )
  warnings = build_completions_warnings(request_body=pruned_payload, response_json=exchange.response_json)
  return EndpointExecutionResult(
    name=format_endpoint_name("/v1/chat/completions", case_label),
    question=expected_transcript,
    response_text=response_text,
    success=error_message is None,
    exchange=HttpExchange(
      method=exchange.method,
      url=exchange.url,
      request_headers=exchange.request_headers,
      request_body=pruned_payload,
      response_status=exchange.response_status,
      response_headers=exchange.response_headers,
      response_body_text=exchange.response_body_text,
      response_json=exchange.response_json,
      error_message=exchange.error_message,
    ),
    error_message=error_message,
    partial_success=error_message is None and bool(warnings),
    warnings=tuple(warnings),
  )


def run_transcriptions_test(
  *,
  base_url: str,
  api_key: str | None,
  normalized_payload: dict[str, Any],
  audio_fixture: AudioFixture,
  expected_transcript: str,
  minimum_expected_words: int,
  timeout: float,
  case_label: str | None = None,
) -> EndpointExecutionResult:
  pruned_payload = prune_none(normalized_payload)
  exchange = send_multipart_request(
    url=build_api_url(base_url, "/v1/audio/transcriptions"),
    api_key=api_key,
    fields=pruned_payload,
    file_path=audio_fixture.path,
    file_format=audio_fixture.format,
    timeout=timeout,
  )
  stream = pruned_payload.get("stream") is True
  response_format = pruned_payload.get("response_format")
  response_text = extract_transcription_response_text(
    exchange.response_json, exchange.response_body_text, stream=stream
  )
  error_message = determine_asr_error_message(
    exchange=exchange,
    response_text=response_text,
    expected_transcript=expected_transcript,
    minimum_expected_words=minimum_expected_words,
    format_error_message=validate_transcriptions_response_format(
      exchange,
      response_format=response_format if isinstance(response_format, str) else None,
      stream=stream,
    ),
  )
  warnings = build_transcriptions_warnings(request_body=pruned_payload, response_json=exchange.response_json)
  request_body = dict(pruned_payload)
  request_body["file"] = {
    "filename": audio_fixture.path.name,
    "content_type": content_type_for_audio_format(audio_fixture.format),
    "size": len(audio_fixture.bytes),
  }
  return EndpointExecutionResult(
    name=format_endpoint_name("/v1/audio/transcriptions", case_label),
    question=expected_transcript,
    response_text=response_text,
    success=error_message is None,
    exchange=HttpExchange(
      method=exchange.method,
      url=exchange.url,
      request_headers=exchange.request_headers,
      request_body=request_body,
      response_status=exchange.response_status,
      response_headers=exchange.response_headers,
      response_body_text=exchange.response_body_text,
      response_json=exchange.response_json,
      error_message=exchange.error_message,
    ),
    error_message=error_message,
    partial_success=error_message is None and bool(warnings),
    warnings=tuple(warnings),
  )


def determine_asr_error_message(
  *,
  exchange: HttpExchange,
  response_text: str,
  expected_transcript: str,
  minimum_expected_words: int,
  format_error_message: str | None,
) -> str | None:
  error_message = determine_error_message(exchange, response_text)
  if error_message is not None:
    return error_message
  if format_error_message is not None:
    return format_error_message
  return build_accuracy_error_message(response_text, expected_transcript, minimum_expected_words)


def validate_completions_response_format(exchange: HttpExchange, *, stream: bool) -> str | None:
  if exchange.response_status is None or exchange.error_message is not None:
    return None
  if stream:
    if not any(parse_sse_events(exchange.response_body_text)):
      return "Expected a text/event-stream response."
    return None
  if not isinstance(exchange.response_json, dict):
    return "Expected a JSON object response."
  return None


def validate_transcriptions_response_format(
  exchange: HttpExchange,
  *,
  response_format: str | None,
  stream: bool,
) -> str | None:
  if exchange.response_status is None or exchange.error_message is not None:
    return None
  if stream:
    if not any(parse_sse_events(exchange.response_body_text)):
      return "Expected a text/event-stream response."
    return None

  normalized_response_format = response_format or "json"
  if normalized_response_format in JSON_TRANSCRIPTION_FORMATS and not isinstance(exchange.response_json, dict):
    return "Expected a JSON object response."
  if normalized_response_format in TEXT_TRANSCRIPTION_FORMATS and exchange.response_body_text.strip():
    return None
  return None


def build_accuracy_error_message(
  response_text: str, expected_transcript: str, minimum_expected_words: int
) -> str | None:
  response_words = set(normalize_words(response_text))
  expected_words = normalize_words(expected_transcript)
  matched_words = [word for word in expected_words if word in response_words]
  if len(matched_words) >= minimum_expected_words:
    return None
  _, total_words, wer = compute_word_error_rate(expected_transcript, response_text)
  if total_words and wer < DEFAULT_MAX_WORD_ERROR_RATE:
    return None
  missing_words = [word for word in expected_words if word not in response_words]
  return (
    f"Transcript matched {len(matched_words)} of {len(expected_words)} expected words; "
    f"required at least {minimum_expected_words}. Missing: {', '.join(missing_words)}."
  )


def normalize_words(value: str) -> list[str]:
  raw_words = re.findall(r"[a-z0-9]+", value.lower())
  normalized: list[str] = []
  index = 0
  while index < len(raw_words):
    if raw_words[index : index + 2] == ["fox", "trot"]:
      normalized.append("foxtrot")
      index += 2
      continue
    if raw_words[index : index + 2] == ["x", "ray"]:
      normalized.append("xray")
      index += 2
      continue
    normalized.append(WORD_VARIANT_ALIASES.get(raw_words[index], raw_words[index]))
    index += 1
  return normalized


def compute_word_error_rate(expected_transcript: str, response_text: str) -> tuple[int, int, float]:
  expected_words = normalize_words(expected_transcript)
  response_words = normalize_words(response_text)
  if not expected_words:
    return (0, 0, 0.0)

  previous_row = list(range(len(response_words) + 1))
  for expected_index, expected_word in enumerate(expected_words, start=1):
    current_row = [expected_index]
    for response_index, response_word in enumerate(response_words, start=1):
      substitution_cost = 0 if expected_word == response_word else 1
      current_row.append(
        min(
          previous_row[response_index] + 1,
          current_row[response_index - 1] + 1,
          previous_row[response_index - 1] + substitution_cost,
        )
      )
    previous_row = current_row

  errors = previous_row[-1]
  return (errors, len(expected_words), errors / len(expected_words))


def normalize_known_model_transcript(response_text: str, *, requested_model: str | None) -> str:
  if not response_text:
    return response_text
  if not is_qwen_asr_model(requested_model):
    return response_text
  return re.sub(r"^\s*language\b[^<]*<asr_text>\s*", "", response_text, flags=re.IGNORECASE).strip()


def is_qwen_asr_model(requested_model: str | None) -> bool:
  if not requested_model:
    return False
  normalized_model = requested_model.casefold()
  return "qwen" in normalized_model and "asr" in normalized_model


def format_endpoint_name(endpoint_path: str, case_label: str | None) -> str:
  return endpoint_path if not case_label else f"{endpoint_path} ({case_label})"


def build_completions_warnings(*, request_body: dict[str, Any], response_json: Any | None) -> list[str]:
  if not isinstance(response_json, dict):
    return []
  warnings = find_argument_mismatch_warnings(
    request_body,
    response_json,
    (
      "audio",
      "frequency_penalty",
      "function_call",
      "functions",
      "logprobs",
      "max_completion_tokens",
      "max_tokens",
      "metadata",
      "modalities",
      "model",
      "n",
      "parallel_tool_calls",
      "prediction",
      "presence_penalty",
      "prompt_cache_key",
      "prompt_cache_retention",
      "reasoning_effort",
      "response_format",
      "safety_identifier",
      "seed",
      "service_tier",
      "stop",
      "store",
      "stream",
      "stream_options",
      "temperature",
      "tool_choice",
      "tools",
      "top_logprobs",
      "top_p",
      "user",
      "web_search_options",
    ),
  )
  for tool_name in extract_completions_tool_call_names(response_json):
    request_tools = request_body.get("tools")
    if tool_is_available(tool_name, request_tools):
      continue
    if request_tools:
      warnings.append(
        f"WARNING: a tool call was returned for tool {format_json_scalar(tool_name)}, "
        "but that tool was not available in the request."
      )
    else:
      warnings.append(
        f"WARNING: a tool call was returned for tool {format_json_scalar(tool_name)}, "
        "but no tools were available in the request."
      )
  return warnings


def extract_completions_tool_call_names(response_json: dict[str, Any]) -> list[str]:
  choices = response_json.get("choices")
  if not isinstance(choices, list):
    return []
  tool_names: list[str] = []
  for choice in choices:
    if not isinstance(choice, dict):
      continue
    message = choice.get("message")
    if not isinstance(message, dict):
      continue
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list):
      continue
    for tool_call in tool_calls:
      if not isinstance(tool_call, dict):
        continue
      function_block = tool_call.get("function")
      if isinstance(function_block, dict) and isinstance(function_block.get("name"), str):
        tool_names.append(function_block["name"])
  return tool_names


def build_transcriptions_warnings(*, request_body: dict[str, Any], response_json: Any | None) -> list[str]:
  if not isinstance(response_json, dict):
    return []
  return find_argument_mismatch_warnings(
    request_body,
    response_json,
    (
      "chunking_strategy",
      "include",
      "known_speaker_names",
      "known_speaker_references",
      "language",
      "model",
      "prompt",
      "response_format",
      "stream",
      "temperature",
      "timestamp_granularities",
    ),
  )


def send_multipart_request(
  *,
  url: str,
  api_key: str | None,
  fields: dict[str, Any],
  file_path: Path,
  file_format: str,
  timeout: float,
) -> HttpExchange:
  request_headers = {"Accept": "application/json"}
  if api_key:
    request_headers["Authorization"] = f"Bearer {api_key}"

  boundary = f"openai-tests-{uuid.uuid4().hex}"
  encoded_payload = build_multipart_body(
    boundary=boundary,
    fields=fields,
    file_path=file_path,
    file_format=file_format,
  )
  request_headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"
  http_request = request.Request(
    url=url,
    data=encoded_payload,
    headers=request_headers,
    method="POST",
  )
  request_body = dict(fields)
  request_body["file"] = {
    "filename": file_path.name,
    "content_type": content_type_for_audio_format(file_format),
    "size": file_path.stat().st_size,
  }

  try:
    with request.urlopen(http_request, timeout=timeout) as response:
      body = response.read()
      return build_http_exchange(
        method="POST",
        url=url,
        request_headers=request_headers,
        request_body=request_body,
        response_status=response.getcode(),
        response_headers=dict(response.headers.items()),
        response_body=body,
      )
  except error.HTTPError as exc:
    body = exc.read()
    return build_http_exchange(
      method="POST",
      url=url,
      request_headers=request_headers,
      request_body=request_body,
      response_status=exc.code,
      response_headers=dict(exc.headers.items()),
      response_body=body,
      error_message=str(exc),
    )
  except error.URLError as exc:
    return HttpExchange(
      method="POST",
      url=url,
      request_headers=request_headers,
      request_body=request_body,
      response_status=None,
      response_headers={},
      response_body_text="",
      response_json=None,
      error_message=str(exc.reason),
    )


def build_multipart_body(
  *,
  boundary: str,
  fields: dict[str, Any],
  file_path: Path,
  file_format: str,
) -> bytes:
  chunks: list[bytes] = []
  for field_name, field_value in fields.items():
    chunks.extend(build_multipart_field_chunks(boundary, field_name, field_value))
  chunks.extend(
    [
      f"--{boundary}\r\n".encode(),
      (
        f'Content-Disposition: form-data; name="file"; filename="{file_path.name}"\r\n'
        f"Content-Type: {content_type_for_audio_format(file_format)}\r\n\r\n"
      ).encode(),
      file_path.read_bytes(),
      b"\r\n",
      f"--{boundary}--\r\n".encode(),
    ]
  )
  return b"".join(chunks)


def build_multipart_field_chunks(boundary: str, field_name: str, field_value: Any) -> list[bytes]:
  if isinstance(field_value, list):
    return [chunk for item in field_value for chunk in build_multipart_field_chunks(boundary, f"{field_name}[]", item)]
  encoded_value = format_multipart_scalar(field_value)
  return [
    f"--{boundary}\r\n".encode(),
    f'Content-Disposition: form-data; name="{field_name}"\r\n\r\n'.encode(),
    encoded_value.encode("utf-8"),
    b"\r\n",
  ]


def format_multipart_scalar(value: Any) -> str:
  if isinstance(value, bool):
    return "true" if value else "false"
  if isinstance(value, (dict, list)):
    return json.dumps(value, sort_keys=True)
  return str(value)


def content_type_for_audio_format(audio_format: str) -> str:
  return TRANSCRIPTION_CONTENT_TYPES.get(audio_format, "application/octet-stream")


def extract_completions_response_text(response_json: Any | None, response_body_text: str, *, stream: bool) -> str:
  if not stream:
    return extract_chat_response_text(response_json)
  parts: list[str] = []
  for event in parse_sse_events(response_body_text):
    choices = event.get("choices")
    if not isinstance(choices, list):
      continue
    for choice in choices:
      if not isinstance(choice, dict):
        continue
      delta = choice.get("delta")
      if isinstance(delta, dict):
        content = delta.get("content")
        text = content if isinstance(content, str) else normalize_text_content(content)
        if text:
          parts.append(text)
  return "".join(parts).strip()


def extract_transcription_response_text(response_json: Any | None, response_body_text: str, *, stream: bool) -> str:
  if stream:
    done_text = ""
    deltas: list[str] = []
    for event in parse_sse_events(response_body_text):
      if isinstance(event.get("text"), str):
        done_text = str(event["text"]).strip()
      if isinstance(event.get("delta"), str):
        deltas.append(str(event["delta"]))
    return done_text or "".join(deltas).strip()

  if isinstance(response_json, dict):
    text = response_json.get("text")
    if isinstance(text, str):
      return text.strip()
    segments = response_json.get("segments")
    if isinstance(segments, list):
      segment_texts = []
      for segment in segments:
        if isinstance(segment, dict) and isinstance(segment.get("text"), str):
          segment_texts.append(segment["text"].strip())
      return "\n".join(segment_text for segment_text in segment_texts if segment_text)
    return ""
  return response_body_text.strip()


def parse_sse_events(response_body_text: str) -> list[dict[str, Any]]:
  events: list[dict[str, Any]] = []
  normalized_body = response_body_text.replace("\r\n", "\n").replace("\r", "\n")
  for raw_event in normalized_body.split("\n\n"):
    data_parts: list[str] = []
    for line in raw_event.splitlines():
      if not line.startswith("data:"):
        continue
      data = line[5:].strip()
      if data == "[DONE]":
        continue
      data_parts.append(data)
    if not data_parts:
      continue
    parsed_event = try_parse_json("\n".join(data_parts))
    if isinstance(parsed_event, dict):
      events.append(parsed_event)
  return events


def print_endpoint_result(result: EndpointExecutionResult, *, verbose: bool) -> None:
  print(f"{result.name}: {colorize_status(determine_endpoint_status(result))}")
  print(f"Expected transcript: {result.question}")
  print(f"Transcript: {result.response_text or '(none)'}")
  wer_errors, wer_total_words, wer = compute_word_error_rate(result.question, result.response_text)
  if wer_total_words:
    print(f"WER: {wer:.2%} ({wer_errors}/{wer_total_words})")
  if result.error_message is not None:
    print(f"Error: {result.error_message}")
  for warning in result.warnings:
    print(warning)
  if verbose:
    print()
    print("Request:")
    print(f"{result.exchange.method} {result.exchange.url}")
    print(json.dumps(redact_headers(result.exchange.request_headers), indent=2, sort_keys=True))
    print(format_json_like(result.exchange.request_body))
    print()
    print("Response:")
    print(f"HTTP {result.exchange.response_status if result.exchange.response_status is not None else 'N/A'}")
    print(json.dumps(result.exchange.response_headers, indent=2, sort_keys=True))
    print(result.exchange.response_body_text or "(empty)")


ASR_SIMPLE_MODULE = EndpointTestModule(
  name="asr-simple",
  summary="Transcribe bundled or custom speech through both chat completions and audio transcriptions.",
  configure_parser=configure_parser,
  handler=run,
)
