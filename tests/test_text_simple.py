from __future__ import annotations

import argparse
import io
from email.message import Message
from pathlib import Path
from urllib import error

import pytest

from openai_tests.test_modules import text_simple


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
  text_simple.configure_parser(parser)
  args = parser.parse_args([])
  for key, value in overrides.items():
    setattr(args, key, value)
  return args


def test_build_chat_request_payload_combines_system_and_developer_prompts() -> None:
  args = build_args(
    system_prompt="System prompt.",
    developer_prompt="Developer prompt.",
    user_prompt="Question?",
  )
  payload = text_simple.build_chat_request_payload(args, "gpt-test")
  assert payload == {
    "model": "gpt-test",
    "messages": [
      {
        "role": "system",
        "content": "System prompt.\n\nDeveloper prompt.",
      },
      {
        "role": "user",
        "content": "Question?",
      },
    ],
  }


def test_build_responses_request_config_uses_defaults_and_specific_overrides(tmp_path: Path) -> None:
  metadata_path = tmp_path / "metadata.json"
  metadata_path.write_text('{"suite":"text-simple"}', encoding="utf-8")
  args = build_args(
    model="gpt-common",
    responses_model="gpt-responses",
    responses_background=True,
    responses_include=["message.output_text.logprobs"],
    responses_include_json='["reasoning.encrypted_content"]',
    responses_metadata_json=f"@{metadata_path}",
    responses_reasoning_effort="high",
    responses_reasoning_summary="concise",
    responses_text_verbosity="low",
    responses_temperature=0.2,
    responses_top_p=0.9,
    responses_top_logprobs=3,
    responses_truncation="auto",
  )
  payload = text_simple.build_responses_request_config(args)
  assert payload["model"] == "gpt-responses"
  assert payload["background"] is True
  assert payload["include"] == [
    "message.output_text.logprobs",
    "reasoning.encrypted_content",
  ]
  assert payload["input"] == [
    {"role": "system", "content": text_simple.DEFAULT_SYSTEM_PROMPT},
    {"role": "developer", "content": text_simple.DEFAULT_DEVELOPER_PROMPT},
    {"role": "user", "content": text_simple.DEFAULT_USER_PROMPT},
  ]
  assert payload["metadata"] == {"suite": "text-simple"}
  assert payload["reasoning"] == {
    "effort": "high",
    "generate_summary": None,
    "summary": "concise",
  }
  assert payload["text"] == {"format": None, "verbosity": "low"}
  assert payload["temperature"] == 0.2
  assert payload["top_p"] == 0.9
  assert payload["top_logprobs"] == 3
  assert payload["truncation"] == "auto"


def test_build_responses_request_config_supports_json_inputs() -> None:
  args = build_args(
    model="gpt-common",
    responses_conversation_json='{"id":"conv_123"}',
    responses_input_json='[{"role":"user","content":"Question?"}]',
    responses_instructions_json='{"role":"developer","content":"Extra instructions."}',
    responses_prompt_id="pmpt_123",
    responses_prompt_version="9",
    responses_prompt_variables_json='{"name":"world"}',
    responses_stream=True,
    responses_include_obfuscation=False,
    responses_stream_options_json='{"include_obfuscation": true}',
    responses_text_json='{"verbosity":"medium"}',
    responses_text_format_json='{"type":"text"}',
    responses_tool_choice_json='{"type":"function","name":"lookup"}',
    responses_tools_json='[{"type":"function","name":"lookup","parameters":{"type":"object"}}]',
    responses_user="user-123",
  )
  payload = text_simple.build_responses_request_config(args)
  assert payload["conversation"] == {"id": "conv_123"}
  assert payload["input"] == [{"role": "user", "content": "Question?"}]
  assert payload["instructions"] == {"role": "developer", "content": "Extra instructions."}
  assert payload["prompt"] == {
    "id": "pmpt_123",
    "variables": {"name": "world"},
    "version": "9",
  }
  assert payload["stream"] is True
  assert payload["stream_options"] == {"include_obfuscation": False}
  assert payload["text"] == {"format": {"type": "text"}, "verbosity": "medium"}
  assert payload["tool_choice"] == {"type": "function", "name": "lookup"}
  assert payload["tools"] == [{"type": "function", "name": "lookup", "parameters": {"type": "object"}}]
  assert payload["user"] == "user-123"


def test_build_text_config_preserves_existing_format_and_sets_missing_verbosity() -> None:
  args = build_args(responses_text_json='{"format":{"type":"json_object"}}')
  assert text_simple.build_text_config(args) == {
    "format": {"type": "json_object"},
    "verbosity": None,
  }


def test_build_responses_request_config_rejects_invalid_combinations() -> None:
  args = build_args(
    responses_conversation="conv_123",
    responses_conversation_json='{"id":"conv_456"}',
  )
  with pytest.raises(ValueError, match="conversation cannot be provided"):
    text_simple.build_responses_request_config(args)

  args = build_args(responses_prompt_version="1")
  with pytest.raises(ValueError, match="responses-prompt-id is required"):
    text_simple.build_responses_request_config(args)


def test_parse_json_value_supports_file_paths(tmp_path: Path) -> None:
  payload_path = tmp_path / "payload.json"
  payload_path.write_text('{"name":"value"}', encoding="utf-8")
  assert text_simple.parse_json_value(f"@{payload_path}", "payload") == {"name": "value"}


def test_parse_json_helpers_reject_invalid_types() -> None:
  with pytest.raises(ValueError, match="object"):
    text_simple.parse_json_dict('["not-an-object"]', "metadata")
  with pytest.raises(ValueError, match="array"):
    text_simple.parse_json_list('{"not":"a-list"}', "include")
  with pytest.raises(ValueError, match="Invalid JSON"):
    text_simple.parse_json_value("{invalid", "payload")


def test_build_api_url_handles_existing_v1_suffix() -> None:
  assert text_simple.build_api_url("https://example.com", "/v1/responses") == "https://example.com/v1/responses"
  assert text_simple.build_api_url("https://example.com/v1", "/v1/responses") == "https://example.com/v1/responses"


def test_extract_chat_response_text_handles_multiple_payload_shapes() -> None:
  direct = {
    "choices": [
      {
        "message": {
          "content": "Paris",
        }
      }
    ]
  }
  structured = {
    "choices": [
      {
        "message": {
          "content": [{"text": "Paris"}],
        }
      }
    ]
  }
  assert text_simple.extract_chat_response_text(direct) == "Paris"
  assert text_simple.extract_chat_response_text(structured) == "Paris"
  assert text_simple.extract_chat_response_text(None) == ""
  assert text_simple.extract_chat_response_text({"choices": []}) == ""
  assert text_simple.extract_chat_response_text({"choices": ["not-a-dict"]}) == ""
  assert text_simple.extract_chat_response_text({"choices": [{"text": "Paris"}]}) == "Paris"


def test_extract_responses_output_text_handles_output_text_and_messages() -> None:
  with_output_text = {"output_text": "Paris"}
  with_output_items = {
    "output": [
      {
        "type": "message",
        "content": [{"text": "Paris"}],
      }
    ]
  }
  with_mixed_output_items = {
    "output": [
      "skip-me",
      {"type": "other"},
      {"type": "message", "content": [{}]},
      {"text": "Paris"},
      {"type": "message", "content": [{"text": "France"}]},
    ]
  }
  assert text_simple.extract_responses_output_text(with_output_text) == "Paris"
  assert text_simple.extract_responses_output_text(with_output_items) == "Paris"
  assert text_simple.extract_responses_output_text(with_mixed_output_items) == "Paris\nFrance"
  assert text_simple.extract_responses_output_text(None) == ""


def test_normalize_text_content_handles_multiple_shapes() -> None:
  assert text_simple.normalize_text_content({"text": "", "content": "Paris"}) == "Paris"
  assert text_simple.normalize_text_content({"unknown": "value"}) == ""
  assert text_simple.normalize_text_content([{"text": ""}, {"refusal": "No"}]) == "No"
  assert text_simple.normalize_text_content(123) == ""


def test_determine_error_message_handles_all_failure_modes() -> None:
  error_exchange = text_simple.HttpExchange(
    method="POST",
    url="https://example.com",
    request_headers={},
    request_body={},
    response_status=500,
    response_headers={},
    response_body_text="{}",
    response_json={},
    error_message="network error",
  )
  assert text_simple.determine_error_message(error_exchange, "Paris") == "network error"

  no_response_exchange = text_simple.HttpExchange(
    method="POST",
    url="https://example.com",
    request_headers={},
    request_body={},
    response_status=None,
    response_headers={},
    response_body_text="",
    response_json=None,
  )
  assert text_simple.determine_error_message(no_response_exchange, "Paris") == "No HTTP response was received."

  status_exchange = text_simple.HttpExchange(
    method="POST",
    url="https://example.com",
    request_headers={},
    request_body={},
    response_status=429,
    response_headers={},
    response_body_text="{}",
    response_json={},
  )
  assert text_simple.determine_error_message(status_exchange, "Paris") == "HTTP 429"


def test_build_responses_warnings_reports_argument_mismatches_and_unavailable_tool_call() -> None:
  warnings = text_simple.build_responses_warnings(
    request_body={
      "tool_choice": None,
      "tools": None,
      "parallel_tool_calls": None,
    },
    response_json={
      "tool_choice": "auto",
      "tools": [],
      "parallel_tool_calls": True,
    },
    response_text='{"name":"capital","parameters":{"country":"France"}}',
  )
  assert warnings == [
    'WARNING: argument tool_choice was sent as null and returned as "auto".',
    "WARNING: argument tools was sent as null and returned as [].",
    "WARNING: argument parallel_tool_calls was sent as null and returned as true.",
    'WARNING: a tool call was returned for tool "capital", but no tools were available in the request.',
  ]


def test_build_responses_warnings_reports_invalid_json() -> None:
  warnings = text_simple.build_responses_warnings(
    request_body={
      "tool_choice": None,
      "tools": None,
      "parallel_tool_calls": None,
    },
    response_json={
      "tool_choice": None,
      "tools": None,
      "parallel_tool_calls": None,
    },
    response_text='{"name":"capital"',
  )
  assert warnings == ["WARNING: returned JSON was not valid."]


def test_build_responses_warnings_ignores_non_dict_response_json_and_non_tool_json() -> None:
  warnings = text_simple.build_responses_warnings(
    request_body={
      "tool_choice": None,
      "tools": None,
      "parallel_tool_calls": None,
    },
    response_json=["not-a-dict"],
    response_text='{"answer":"Paris"}',
  )
  assert warnings == []


def test_build_responses_warnings_does_not_warn_for_available_tool_call() -> None:
  warnings = text_simple.build_responses_warnings(
    request_body={
      "tool_choice": "auto",
      "tools": [{"type": "function", "name": "capital"}],
      "parallel_tool_calls": True,
    },
    response_json={
      "tool_choice": "auto",
      "tools": [{"type": "function", "name": "capital"}],
      "parallel_tool_calls": True,
    },
    response_text='{"name":"capital","parameters":{"country":"France"}}',
  )
  assert warnings == []


def test_build_responses_warnings_reports_unavailable_tool_when_tools_were_provided() -> None:
  warnings = text_simple.build_responses_warnings(
    request_body={
      "tool_choice": "auto",
      "tools": [{"type": "function", "name": "lookup"}],
      "parallel_tool_calls": True,
    },
    response_json={
      "tool_choice": "auto",
      "tools": [{"type": "function", "name": "lookup"}],
      "parallel_tool_calls": True,
    },
    response_text='{"name":"capital","parameters":{"country":"France"}}',
  )
  assert warnings == [
    'WARNING: a tool call was returned for tool "capital", but that tool was not available in the request.'
  ]


def test_tool_is_available_supports_nested_function_blocks() -> None:
  assert text_simple.tool_is_available(
    "capital",
    [
      "skip-me",
      {"type": "function", "function": {"name": "capital"}},
    ],
  )


def test_send_json_request_returns_http_error_exchange(monkeypatch: pytest.MonkeyPatch) -> None:
  def fake_urlopen(_: object, timeout: float) -> FakeResponse:
    assert timeout == 5.0
    headers = Message()
    headers["Content-Type"] = "application/json"
    raise error.HTTPError(
      url="https://example.com/v1/responses",
      code=400,
      msg="Bad Request",
      hdrs=headers,
      fp=io.BytesIO(b'{"error":"bad request"}'),
    )

  monkeypatch.setattr(text_simple.request, "urlopen", fake_urlopen)
  exchange = text_simple.send_json_request(
    url="https://example.com/v1/responses",
    api_key="secret",
    payload={"model": "gpt-test"},
    timeout=5.0,
  )
  assert exchange.response_status == 400
  assert exchange.response_json == {"error": "bad request"}
  assert exchange.error_message is not None


def test_send_json_request_handles_success_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
  def fake_urlopen(_: object, timeout: float) -> FakeResponse:
    assert timeout == 5.0
    return FakeResponse(b'{"ok":true}', status=201, headers={"Content-Type": "application/json"})

  monkeypatch.setattr(text_simple.request, "urlopen", fake_urlopen)
  exchange = text_simple.send_json_request(
    url="https://example.com/v1/responses",
    api_key=None,
    payload={"model": "gpt-test"},
    timeout=5.0,
  )
  assert exchange.response_status == 201
  assert exchange.response_json == {"ok": True}
  assert "Authorization" not in exchange.request_headers


def test_send_json_request_returns_url_error_exchange(monkeypatch: pytest.MonkeyPatch) -> None:
  def fake_urlopen(_: object, timeout: float) -> FakeResponse:
    raise error.URLError("no route")

  monkeypatch.setattr(text_simple.request, "urlopen", fake_urlopen)
  exchange = text_simple.send_json_request(
    url="https://example.com/v1/responses",
    api_key=None,
    payload={"model": "gpt-test"},
    timeout=5.0,
  )
  assert exchange.response_status is None
  assert exchange.error_message == "no route"


def test_try_parse_json_handles_empty_and_invalid_payloads() -> None:
  assert text_simple.try_parse_json("") is None
  assert text_simple.try_parse_json("not-json") is None


def test_run_executes_both_endpoints_and_renders_verbose_output(
  monkeypatch: pytest.MonkeyPatch,
  capsys: pytest.CaptureFixture[str],
) -> None:
  sent_payloads: list[tuple[str, dict[str, object]]] = []

  def fake_send_json_request(
    *, url: str, api_key: str | None, payload: dict[str, object], timeout: float
  ) -> text_simple.HttpExchange:
    assert api_key == "cli-key"
    assert timeout == 12.0
    sent_payloads.append((url, payload))
    if url.endswith("/v1/chat/completions"):
      return text_simple.HttpExchange(
        method="POST",
        url=url,
        request_headers={"Authorization": "Bearer cli-key"},
        request_body=payload,
        response_status=200,
        response_headers={"Content-Type": "application/json"},
        response_body_text='{"choices":[{"message":{"content":"Paris"}}]}',
        response_json={"choices": [{"message": {"content": "Paris"}}]},
      )
    return text_simple.HttpExchange(
      method="POST",
      url=url,
      request_headers={"Authorization": "Bearer cli-key"},
      request_body=payload,
      response_status=200,
      response_headers={"Content-Type": "application/json"},
      response_body_text='{"output_text":"Paris"}',
      response_json={"output_text": "Paris"},
    )

  monkeypatch.setattr(text_simple, "send_json_request", fake_send_json_request)
  args = build_args(
    base_url="https://example.com/v1",
    api_key="cli-key",
    model="gpt-shared",
    timeout=12.0,
    verbose=True,
  )
  assert text_simple.run(args) == 0
  captured = capsys.readouterr()
  assert "/v1/chat/completions:" in captured.out
  assert "/v1/responses:" in captured.out
  assert "Question: What is the capital of France?" in captured.out
  assert "Response: Paris" in captured.out
  assert "***REDACTED***" in captured.out
  assert "Overall:" in captured.out
  assert "PARTIAL SUCCESS" not in captured.out
  assert sent_payloads[0][0] == "https://example.com/v1/chat/completions"
  assert sent_payloads[1][0] == "https://example.com/v1/responses"
  assert sent_payloads[1][1]["model"] == "gpt-shared"


def test_run_returns_failure_when_one_endpoint_does_not_return_text(
  monkeypatch: pytest.MonkeyPatch,
  capsys: pytest.CaptureFixture[str],
) -> None:
  def fake_send_json_request(
    *, url: str, api_key: str | None, payload: dict[str, object], timeout: float
  ) -> text_simple.HttpExchange:
    if url.endswith("/v1/chat/completions"):
      return text_simple.HttpExchange(
        method="POST",
        url=url,
        request_headers={},
        request_body=payload,
        response_status=200,
        response_headers={},
        response_body_text='{"choices":[{"message":{"content":"Paris"}}]}',
        response_json={"choices": [{"message": {"content": "Paris"}}]},
      )
    return text_simple.HttpExchange(
      method="POST",
      url=url,
      request_headers={},
      request_body=payload,
      response_status=200,
      response_headers={},
      response_body_text="{}",
      response_json={},
    )

  monkeypatch.setattr(text_simple, "send_json_request", fake_send_json_request)
  args = build_args()
  assert text_simple.run(args) == 1
  captured = capsys.readouterr()
  assert "Error: No response text was returned." in captured.out


def test_run_returns_partial_success_when_responses_endpoint_returns_tool_call_warnings(
  monkeypatch: pytest.MonkeyPatch,
  capsys: pytest.CaptureFixture[str],
) -> None:
  def fake_send_json_request(
    *, url: str, api_key: str | None, payload: dict[str, object], timeout: float
  ) -> text_simple.HttpExchange:
    if url.endswith("/v1/chat/completions"):
      return text_simple.HttpExchange(
        method="POST",
        url=url,
        request_headers={},
        request_body=payload,
        response_status=200,
        response_headers={},
        response_body_text='{"choices":[{"message":{"content":"Paris"}}]}',
        response_json={"choices": [{"message": {"content": "Paris"}}]},
      )
    return text_simple.HttpExchange(
      method="POST",
      url=url,
      request_headers={},
      request_body=payload,
      response_status=200,
      response_headers={},
      response_body_text=(
        '{"output":[{"type":"message","content":[{"text":"{\\"name\\": \\"capital\\", '
        '\\"parameters\\": {\\"country\\": \\"France\\"}}"}]}],'
        '"tool_choice":"auto","tools":[],"parallel_tool_calls":true}'
      ),
      response_json={
        "output": [
          {
            "type": "message",
            "content": [
              {
                "text": '{"name": "capital", "parameters": {"country": "France"}}',
              }
            ],
          }
        ],
        "tool_choice": "auto",
        "tools": [],
        "parallel_tool_calls": True,
      },
    )

  monkeypatch.setattr(text_simple, "send_json_request", fake_send_json_request)
  args = build_args()
  assert text_simple.run(args) == 1
  captured = capsys.readouterr()
  assert "PARTIAL SUCCESS" in captured.out
  assert "WARNING: argument tool_choice was sent" not in captured.out
  assert "WARNING: argument tools was sent" not in captured.out
  assert "WARNING: argument parallel_tool_calls was sent" not in captured.out
  assert (
    'WARNING: a tool call was returned for tool "capital", but no tools were available in the request.' in captured.out
  )


def test_run_responses_test_ignores_omitted_optional_fields_in_warning_checks(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  def fake_send_json_request(
    *, url: str, api_key: str | None, payload: dict[str, object], timeout: float
  ) -> text_simple.HttpExchange:
    assert "tool_choice" not in payload
    assert "tools" not in payload
    assert "parallel_tool_calls" not in payload
    return text_simple.HttpExchange(
      method="POST",
      url=url,
      request_headers={},
      request_body=payload,
      response_status=200,
      response_headers={},
      response_body_text='{"output_text":"Paris","tool_choice":"auto","tools":[],"parallel_tool_calls":true}',
      response_json={
        "output_text": "Paris",
        "tool_choice": "auto",
        "tools": [],
        "parallel_tool_calls": True,
      },
    )

  monkeypatch.setattr(text_simple, "send_json_request", fake_send_json_request)
  result = text_simple.run_responses_test(
    base_url="https://example.com",
    api_key=None,
    normalized_payload={
      "model": "gpt-test",
      "input": "Question?",
      "tool_choice": None,
      "tools": None,
      "parallel_tool_calls": None,
    },
    question="Question?",
    timeout=5.0,
  )

  assert result.warnings == ()
  assert result.partial_success is False


def test_run_returns_configuration_error_for_invalid_json(capsys: pytest.CaptureFixture[str]) -> None:
  args = build_args(responses_metadata_json="{invalid")
  assert text_simple.run(args) == 2
  captured = capsys.readouterr()
  assert "Configuration error:" in captured.err


def test_redact_headers_and_colorize_status() -> None:
  assert text_simple.redact_headers({"Content-Type": "application/json"}) == {"Content-Type": "application/json"}
  assert "PASSED" in text_simple.colorize_status("passed")
  assert "PARTIAL SUCCESS" in text_simple.colorize_status("partial")
  assert "FAILED" in text_simple.colorize_status("failed")


def test_determine_status_helpers() -> None:
  exchange = text_simple.HttpExchange(
    method="POST",
    url="https://example.com",
    request_headers={},
    request_body={},
    response_status=200,
    response_headers={},
    response_body_text="{}",
    response_json={},
  )
  passed = text_simple.EndpointExecutionResult(
    name="chat",
    question="Question?",
    response_text="Paris",
    success=True,
    exchange=exchange,
  )
  partial = text_simple.EndpointExecutionResult(
    name="responses",
    question="Question?",
    response_text="Paris",
    success=True,
    partial_success=True,
    warnings=("warning",),
    exchange=exchange,
  )
  failed = text_simple.EndpointExecutionResult(
    name="responses",
    question="Question?",
    response_text="",
    success=False,
    error_message="HTTP 500",
    exchange=exchange,
  )
  assert text_simple.determine_endpoint_status(passed) == "passed"
  assert text_simple.determine_endpoint_status(partial) == "partial"
  assert text_simple.determine_endpoint_status(failed) == "failed"
  assert text_simple.determine_overall_status(passed, partial) == "partial"
  assert text_simple.determine_overall_status(passed, failed) == "failed"
