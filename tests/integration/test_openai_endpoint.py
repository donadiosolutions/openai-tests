from __future__ import annotations

import argparse
import os
from collections.abc import Callable
from pathlib import Path

import pytest

from openai_tests.test_modules import asr_simple, list_models, text_simple
from openai_tests.test_modules._shared import EndpointExecutionResult, determine_overall_status

OPENAI_BASE_URL = "https://api.openai.com"

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
      "--min-expected-words",
      "8",
      "--transcriptions-language",
      "en",
      "--transcriptions-response-format",
      "json",
      "--timeout",
      "120",
    ],
  )

  results: list[EndpointExecutionResult] = []
  for audio_case in asr_simple.prepare_audio_cases(args, tmp_path):
    results.append(
      asr_simple.run_completions_test(
        base_url=asr_simple.resolve_base_url(args.base_url),
        api_key=asr_simple.resolve_api_key(args.api_key),
        normalized_payload=asr_simple.build_completions_request_config(
          args,
          audio_case.fixture.bytes,
          audio_case.fixture.format,
        ),
        expected_transcript=audio_case.expected_transcript,
        minimum_expected_words=audio_case.minimum_expected_words,
        timeout=args.timeout,
        case_label=audio_case.label,
      )
    )
    results.append(
      asr_simple.run_transcriptions_test(
        base_url=asr_simple.resolve_base_url(args.base_url),
        api_key=asr_simple.resolve_api_key(args.api_key),
        normalized_payload=asr_simple.build_transcriptions_request_config(args),
        audio_fixture=audio_case.fixture,
        expected_transcript=audio_case.expected_transcript,
        minimum_expected_words=audio_case.minimum_expected_words,
        timeout=args.timeout,
        case_label=audio_case.label,
      )
    )

  assert len(results) == 4
  assert_endpoints_succeeded(*results)


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
