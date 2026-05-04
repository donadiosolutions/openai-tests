from __future__ import annotations

import argparse
import io
from email.message import Message
from urllib import error, request

import pytest

from openai_tests.test_modules import list_models
from openai_tests.test_modules._shared import HttpExchange


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
  list_models.configure_parser(parser)
  args = parser.parse_args([])
  for key, value in overrides.items():
    setattr(args, key, value)
  return args


def valid_models_response() -> dict[str, object]:
  return {
    "object": "list",
    "data": [
      {
        "id": "gpt-4.1-mini",
        "object": "model",
        "created": 1715367049,
        "owned_by": "openai",
      },
      {
        "id": "gpt-4o-transcribe",
        "object": "model",
        "created": 1741399000,
        "owned_by": "openai",
      },
    ],
  }


def test_parser_defaults_to_common_options() -> None:
  args = build_args()
  assert args.base_url is None
  assert args.api_key is None
  assert args.timeout == list_models.DEFAULT_TIMEOUT_SECONDS
  assert args.verbose is False


def test_resolve_base_url_and_api_key_use_cli_then_environment(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setenv("OPENAI_BASE_URL", "https://env.example.com")
  monkeypatch.setenv("OPENAI_TESTS_BASE_URL", "https://tests.example.com")
  monkeypatch.setenv("OPENAI_API_KEY", "env-key")
  monkeypatch.setenv("OPENAI_TESTS_API_KEY", "tests-key")

  assert list_models.resolve_base_url("https://cli.example.com") == "https://cli.example.com"
  assert list_models.resolve_base_url(None) == "https://env.example.com"
  assert list_models.resolve_api_key("cli-key") == "cli-key"
  assert list_models.resolve_api_key(None) == "env-key"


def test_extract_model_ids_lists_ids_and_skips_invalid_items() -> None:
  response_json = valid_models_response()
  response_json["data"] = [
    {"id": "gpt-4.1-mini", "object": "model", "created": 1, "owned_by": "openai"},
    {"object": "model", "created": 1, "owned_by": "openai"},
    "not-a-model",
    {"id": "gpt-4o-transcribe", "object": "model", "created": 1, "owned_by": "openai"},
  ]
  assert list_models.extract_model_ids(response_json) == ["gpt-4.1-mini", "gpt-4o-transcribe"]
  assert list_models.extract_model_ids(None) == []
  assert list_models.extract_model_ids({"data": "not-a-list"}) == []


def test_validate_models_response_schema_accepts_valid_and_empty_lists() -> None:
  assert list_models.validate_models_response_schema(valid_models_response()) is None
  assert list_models.validate_models_response_schema({"object": "list", "data": []}) is None


@pytest.mark.parametrize(
  ("payload", "message"),
  [
    ([], "Expected response JSON to be an object."),
    ({"object": "model", "data": []}, 'Expected response.object to be "list".'),
    ({"object": "list"}, "Expected response.data to be an array."),
    ({"object": "list", "data": ["bad"]}, "Expected response.data[0] to be an object."),
    (
      {"object": "list", "data": [{"object": "model", "created": 1, "owned_by": "openai"}]},
      "Expected response.data[0].id to be a non-empty string.",
    ),
    (
      {"object": "list", "data": [{"id": "gpt", "object": "not-model", "created": 1, "owned_by": "openai"}]},
      'Expected response.data[0].object to be "model".',
    ),
    (
      {"object": "list", "data": [{"id": "gpt", "object": "model", "created": "1", "owned_by": "openai"}]},
      "Expected response.data[0].created to be an integer.",
    ),
    (
      {"object": "list", "data": [{"id": "gpt", "object": "model", "created": 1, "owned_by": None}]},
      "Expected response.data[0].owned_by to be a non-empty string.",
    ),
  ],
)
def test_validate_models_response_schema_reports_schema_errors(payload: object, message: str) -> None:
  assert list_models.validate_models_response_schema(payload) == message


def test_send_get_request_returns_success_exchange(monkeypatch: pytest.MonkeyPatch) -> None:
  captured_headers: dict[str, str] = {}

  def fake_urlopen(http_request: request.Request, timeout: float) -> FakeResponse:
    assert timeout == 5.0
    captured_headers.update(dict(http_request.header_items()))
    return FakeResponse(b'{"object":"list","data":[]}', status=200)

  monkeypatch.setattr(list_models.request, "urlopen", fake_urlopen)
  exchange = list_models.send_get_request(
    url="https://example.com/v1/models",
    api_key="secret",
    timeout=5.0,
  )
  assert exchange.method == "GET"
  assert exchange.request_body is None
  assert exchange.response_status == 200
  assert exchange.response_json == {"object": "list", "data": []}
  assert captured_headers["Authorization"] == "Bearer secret"


def test_send_get_request_handles_success_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
  def fake_urlopen(_: object, timeout: float) -> FakeResponse:
    assert timeout == 5.0
    return FakeResponse(b'{"object":"list","data":[]}', status=200)

  monkeypatch.setattr(list_models.request, "urlopen", fake_urlopen)
  exchange = list_models.send_get_request(
    url="https://example.com/v1/models",
    api_key=None,
    timeout=5.0,
  )
  assert "Authorization" not in exchange.request_headers


def test_send_get_request_returns_http_error_exchange(monkeypatch: pytest.MonkeyPatch) -> None:
  def fake_urlopen(_: object, timeout: float) -> FakeResponse:
    assert timeout == 5.0
    headers = Message()
    headers["Content-Type"] = "application/json"
    raise error.HTTPError(
      url="https://example.com/v1/models",
      code=401,
      msg="Unauthorized",
      hdrs=headers,
      fp=io.BytesIO(b'{"error":"unauthorized"}'),
    )

  monkeypatch.setattr(list_models.request, "urlopen", fake_urlopen)
  exchange = list_models.send_get_request(
    url="https://example.com/v1/models",
    api_key="secret",
    timeout=5.0,
  )
  assert exchange.response_status == 401
  assert exchange.response_json == {"error": "unauthorized"}
  assert exchange.error_message is not None


def test_send_get_request_returns_url_error_exchange(monkeypatch: pytest.MonkeyPatch) -> None:
  def fake_urlopen(_: object, timeout: float) -> FakeResponse:
    raise error.URLError("no route")

  monkeypatch.setattr(list_models.request, "urlopen", fake_urlopen)
  exchange = list_models.send_get_request(
    url="https://example.com/v1/models",
    api_key=None,
    timeout=5.0,
  )
  assert exchange.response_status is None
  assert exchange.error_message == "no route"


def test_run_models_test_passes_when_response_conforms(monkeypatch: pytest.MonkeyPatch) -> None:
  def fake_send_get_request(*, url: str, api_key: str | None, timeout: float) -> HttpExchange:
    assert url == "https://example.com/v1/models"
    assert api_key == "secret"
    assert timeout == 5.0
    return HttpExchange(
      method="GET",
      url=url,
      request_headers={"Authorization": "Bearer secret"},
      request_body=None,
      response_status=200,
      response_headers={"Content-Type": "application/json"},
      response_body_text="{}",
      response_json=valid_models_response(),
    )

  monkeypatch.setattr(list_models, "send_get_request", fake_send_get_request)
  result = list_models.run_models_test(
    base_url="https://example.com",
    api_key="secret",
    timeout=5.0,
  )
  assert result.success is True
  assert result.response_text == "gpt-4.1-mini\ngpt-4o-transcribe"
  assert result.error_message is None


def test_run_models_test_fails_when_response_schema_is_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
  def fake_send_get_request(*, url: str, api_key: str | None, timeout: float) -> HttpExchange:
    return HttpExchange(
      method="GET",
      url=url,
      request_headers={},
      request_body=None,
      response_status=200,
      response_headers={},
      response_body_text='{"object":"list"}',
      response_json={"object": "list"},
    )

  monkeypatch.setattr(list_models, "send_get_request", fake_send_get_request)
  result = list_models.run_models_test(
    base_url="https://example.com",
    api_key=None,
    timeout=5.0,
  )
  assert result.success is False
  assert result.error_message == "Expected response.data to be an array."


def test_run_models_test_preserves_transport_errors(monkeypatch: pytest.MonkeyPatch) -> None:
  def fake_send_get_request(*, url: str, api_key: str | None, timeout: float) -> HttpExchange:
    return HttpExchange(
      method="GET",
      url=url,
      request_headers={},
      request_body=None,
      response_status=503,
      response_headers={},
      response_body_text='{"error":"unavailable"}',
      response_json={"error": "unavailable"},
    )

  monkeypatch.setattr(list_models, "send_get_request", fake_send_get_request)
  result = list_models.run_models_test(
    base_url="https://example.com",
    api_key=None,
    timeout=5.0,
  )
  assert result.success is False
  assert result.error_message == "HTTP 503"


def test_run_executes_models_endpoint_and_renders_verbose_output(
  monkeypatch: pytest.MonkeyPatch,
  capsys: pytest.CaptureFixture[str],
) -> None:
  def fake_send_get_request(*, url: str, api_key: str | None, timeout: float) -> HttpExchange:
    assert url == "https://example.com/v1/models"
    assert api_key == "cli-key"
    assert timeout == 12.0
    return HttpExchange(
      method="GET",
      url=url,
      request_headers={"Authorization": "Bearer cli-key", "Accept": "application/json"},
      request_body=None,
      response_status=200,
      response_headers={"Content-Type": "application/json"},
      response_body_text='{"object":"list","data":[]}',
      response_json=valid_models_response(),
    )

  monkeypatch.setattr(list_models, "send_get_request", fake_send_get_request)
  args = build_args(
    base_url="https://example.com/v1",
    api_key="cli-key",
    timeout=12.0,
    verbose=True,
  )
  assert list_models.run(args) == 0
  captured = capsys.readouterr()
  assert "/v1/models:" in captured.out
  assert "Models:" in captured.out
  assert "- gpt-4.1-mini" in captured.out
  assert "- gpt-4o-transcribe" in captured.out
  assert "GET https://example.com/v1/models" in captured.out
  assert "***REDACTED***" in captured.out
  assert "Overall:" in captured.out


def test_run_returns_failure_when_schema_is_invalid(
  monkeypatch: pytest.MonkeyPatch,
  capsys: pytest.CaptureFixture[str],
) -> None:
  def fake_send_get_request(*, url: str, api_key: str | None, timeout: float) -> HttpExchange:
    return HttpExchange(
      method="GET",
      url=url,
      request_headers={},
      request_body=None,
      response_status=200,
      response_headers={},
      response_body_text='{"object":"list"}',
      response_json={"object": "list"},
    )

  monkeypatch.setattr(list_models, "send_get_request", fake_send_get_request)
  assert list_models.run(build_args()) == 1
  captured = capsys.readouterr()
  assert "FAILED" in captured.out
  assert "Error: Expected response.data to be an array." in captured.out
