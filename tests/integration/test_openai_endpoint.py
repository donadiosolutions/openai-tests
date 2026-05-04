from __future__ import annotations

import argparse
import base64
import os
from collections.abc import Callable
from pathlib import Path

import pytest

from openai_tests.test_modules import asr_simple, list_models, text_simple
from openai_tests.test_modules._shared import EndpointExecutionResult, determine_overall_status

OPENAI_BASE_URL = "https://api.openai.com"
FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
ASR_AUDIO_FIXTURE = FIXTURE_DIR / "asr_simple.mp3.base64"

pytestmark = pytest.mark.integration


def require_openai_api_key() -> str:
  api_key = os.getenv("OPENAI_API_KEY")
  if not api_key:
    if os.getenv("OPENAI_TESTS_REQUIRE_OPENAI_API_KEY"):
      pytest.fail("OPENAI_API_KEY is required for OpenAI endpoint integration tests")
    pytest.skip("OPENAI_API_KEY is required for OpenAI endpoint integration tests")
  return api_key


def parse_module_args(
  configure_parser: Callable[[argparse.ArgumentParser], None], argv: list[str]
) -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  configure_parser(parser)
  return parser.parse_args(argv)


def assert_endpoints_succeeded(*results: EndpointExecutionResult) -> None:
  failures = [result for result in results if not result.success]
  assert failures == [], "\n\n".join(
    f"{result.name} failed: {result.error_message}\n{result.exchange.response_body_text}" for result in failures
  )
  assert determine_overall_status(*results) in {"passed", "partial"}


def test_text_simple_runs_against_openai_endpoint() -> None:
  args = parse_module_args(
    text_simple.configure_parser,
    [
      "--base-url",
      OPENAI_BASE_URL,
      "--api-key",
      require_openai_api_key(),
      "--model",
      os.getenv("OPENAI_TESTS_INTEGRATION_TEXT_MODEL", text_simple.DEFAULT_MODEL),
      "--responses-model",
      os.getenv("OPENAI_TESTS_INTEGRATION_RESPONSES_MODEL", text_simple.DEFAULT_MODEL),
      "--timeout",
      "90",
      "--responses-max-output-tokens",
      "16",
    ],
  )

  chat_result = text_simple.run_chat_completion_test(
    base_url=text_simple.resolve_base_url(args.base_url),
    api_key=text_simple.resolve_api_key(args.api_key),
    payload=text_simple.build_chat_request_payload(args, text_simple.resolve_model(args.model)),
    question=args.user_prompt,
    timeout=args.timeout,
  )
  responses_result = text_simple.run_responses_test(
    base_url=text_simple.resolve_base_url(args.base_url),
    api_key=text_simple.resolve_api_key(args.api_key),
    normalized_payload=text_simple.build_responses_request_config(args),
    question=args.user_prompt,
    timeout=args.timeout,
  )

  assert_endpoints_succeeded(chat_result, responses_result)
  assert "paris" in chat_result.response_text.lower()
  assert "paris" in responses_result.response_text.lower()


def test_asr_simple_runs_against_openai_endpoint(tmp_path: Path) -> None:
  audio_path = tmp_path / "asr-simple.mp3"
  audio_path.write_bytes(base64.b64decode(ASR_AUDIO_FIXTURE.read_text(encoding="ascii")))

  args = parse_module_args(
    asr_simple.configure_parser,
    [
      "--base-url",
      OPENAI_BASE_URL,
      "--api-key",
      require_openai_api_key(),
      "--model",
      os.getenv("OPENAI_TESTS_INTEGRATION_ASR_COMPLETIONS_MODEL", asr_simple.DEFAULT_COMPLETIONS_MODEL),
      "--transcriptions-model",
      os.getenv("OPENAI_TESTS_INTEGRATION_ASR_TRANSCRIPTIONS_MODEL", asr_simple.DEFAULT_TRANSCRIPTIONS_MODEL),
      "--audio-file",
      str(audio_path),
      "--audio-format",
      "mp3",
      "--expected-transcript",
      asr_simple.DEFAULT_EXPECTED_TRANSCRIPT,
      "--min-expected-words",
      "8",
      "--transcriptions-language",
      "en",
      "--transcriptions-prompt",
      asr_simple.DEFAULT_EXPECTED_TRANSCRIPT,
      "--transcriptions-response-format",
      "json",
      "--timeout",
      "120",
    ],
  )

  audio_fixture = asr_simple.prepare_audio_fixture(args, tmp_path)
  minimum_expected_words = asr_simple.resolve_minimum_expected_words(
    args.min_expected_words,
    args.expected_transcript,
  )
  completions_result = asr_simple.run_completions_test(
    base_url=asr_simple.resolve_base_url(args.base_url),
    api_key=asr_simple.resolve_api_key(args.api_key),
    normalized_payload=asr_simple.build_completions_request_config(args, audio_fixture.bytes, audio_fixture.format),
    expected_transcript=args.expected_transcript,
    minimum_expected_words=minimum_expected_words,
    timeout=args.timeout,
  )
  transcriptions_result = asr_simple.run_transcriptions_test(
    base_url=asr_simple.resolve_base_url(args.base_url),
    api_key=asr_simple.resolve_api_key(args.api_key),
    normalized_payload=asr_simple.build_transcriptions_request_config(args),
    audio_fixture=audio_fixture,
    expected_transcript=args.expected_transcript,
    minimum_expected_words=minimum_expected_words,
    timeout=args.timeout,
  )

  assert_endpoints_succeeded(completions_result, transcriptions_result)


def test_list_models_runs_against_openai_endpoint() -> None:
  args = parse_module_args(
    list_models.configure_parser,
    [
      "--base-url",
      OPENAI_BASE_URL,
      "--api-key",
      require_openai_api_key(),
      "--timeout",
      "90",
    ],
  )

  result = list_models.run_models_test(
    base_url=list_models.resolve_base_url(args.base_url),
    api_key=list_models.resolve_api_key(args.api_key),
    timeout=args.timeout,
  )

  assert_endpoints_succeeded(result)
  assert result.response_text.strip()
