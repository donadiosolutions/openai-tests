from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import error, request

GREEN = "\033[32m"
ORANGE = "\033[38;5;214m"
RED = "\033[31m"
RESET = "\033[0m"


@dataclass(frozen=True, slots=True)
class HttpExchange:
  method: str
  url: str
  request_headers: dict[str, str]
  request_body: Any
  response_status: int | None
  response_headers: dict[str, str]
  response_body_text: str
  response_json: Any | None
  error_message: str | None = None


@dataclass(frozen=True, slots=True)
class EndpointExecutionResult:
  name: str
  question: str
  response_text: str
  success: bool
  exchange: HttpExchange
  error_message: str | None = None
  partial_success: bool = False
  warnings: tuple[str, ...] = ()


def choose_string_or_json(
  *,
  string_value: str | None,
  json_value: str | None,
  field_name: str,
) -> Any | None:
  if string_value is not None and json_value is not None:
    raise ValueError(f"{field_name} cannot be provided as both a string and JSON")
  if json_value is not None:
    return parse_json_value(json_value, field_name)
  return string_value


def parse_json_value(raw_value: str | None, field_name: str) -> Any | None:
  if raw_value is None:
    return None
  payload = Path(raw_value[1:]).read_text(encoding="utf-8") if raw_value.startswith("@") else raw_value
  try:
    return json.loads(payload)
  except json.JSONDecodeError as exc:
    raise ValueError(f"Invalid JSON for {field_name}: {exc.msg}") from exc


def parse_json_dict(raw_value: str | None, field_name: str) -> dict[str, Any] | None:
  value = parse_json_value(raw_value, field_name)
  if value is None:
    return None
  if not isinstance(value, dict):
    raise ValueError(f"{field_name} must decode to a JSON object")
  return value


def parse_json_list(raw_value: str | None, field_name: str) -> list[Any] | None:
  value = parse_json_value(raw_value, field_name)
  if value is None:
    return None
  if not isinstance(value, list):
    raise ValueError(f"{field_name} must decode to a JSON array")
  return value


def prune_none(value: Any) -> Any:
  if isinstance(value, dict):
    return {key: prune_none(item) for key, item in value.items() if item is not None}
  if isinstance(value, list):
    return [prune_none(item) for item in value]
  return value


def send_json_request(
  *,
  url: str,
  api_key: str | None,
  payload: dict[str, Any],
  timeout: float,
) -> HttpExchange:
  request_headers = {
    "Accept": "application/json",
    "Content-Type": "application/json",
  }
  if api_key:
    request_headers["Authorization"] = f"Bearer {api_key}"

  encoded_payload = json.dumps(payload).encode("utf-8")
  http_request = request.Request(
    url=url,
    data=encoded_payload,
    headers=request_headers,
    method="POST",
  )

  try:
    with request.urlopen(http_request, timeout=timeout) as response:
      body = response.read()
      return build_http_exchange(
        method="POST",
        url=url,
        request_headers=request_headers,
        request_body=payload,
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
      request_body=payload,
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
      request_body=payload,
      response_status=None,
      response_headers={},
      response_body_text="",
      response_json=None,
      error_message=str(exc.reason),
    )


def build_http_exchange(
  *,
  method: str,
  url: str,
  request_headers: dict[str, str],
  request_body: Any,
  response_status: int,
  response_headers: dict[str, str],
  response_body: bytes,
  error_message: str | None = None,
) -> HttpExchange:
  response_body_text = response_body.decode("utf-8", errors="replace")
  response_json = try_parse_json(response_body_text)
  return HttpExchange(
    method=method,
    url=url,
    request_headers=request_headers,
    request_body=request_body,
    response_status=response_status,
    response_headers=response_headers,
    response_body_text=response_body_text,
    response_json=response_json,
    error_message=error_message,
  )


def try_parse_json(raw_text: str) -> Any | None:
  if not raw_text.strip():
    return None
  try:
    return json.loads(raw_text)
  except json.JSONDecodeError:
    return None


def build_api_url(base_url: str, path: str) -> str:
  normalized_base = base_url.rstrip("/")
  if normalized_base.endswith("/v1") and path.startswith("/v1/"):
    normalized_base = normalized_base[:-3]
  return f"{normalized_base}{path}"


def extract_chat_response_text(response_json: Any | None) -> str:
  if not isinstance(response_json, dict):
    return ""
  choices = response_json.get("choices")
  if not isinstance(choices, list) or not choices:
    return ""
  first_choice = choices[0]
  if not isinstance(first_choice, dict):
    return ""
  message = first_choice.get("message")
  if isinstance(message, dict):
    return normalize_text_content(message.get("content"))
  return normalize_text_content(first_choice.get("text"))


def normalize_text_content(content: Any) -> str:
  if isinstance(content, str):
    return content.strip()
  if isinstance(content, dict):
    for key in ("text", "content", "refusal"):
      value = content.get(key)
      if isinstance(value, str) and value.strip():
        return value.strip()
    return ""
  if isinstance(content, list):
    parts: list[str] = []
    for item in content:
      text = normalize_text_content(item)
      if text:
        parts.append(text)
    return "\n".join(parts)
  return ""


def determine_error_message(exchange: HttpExchange, response_text: str) -> str | None:
  if exchange.error_message:
    return exchange.error_message
  if exchange.response_status is None:
    return "No HTTP response was received."
  if exchange.response_status < 200 or exchange.response_status >= 300:
    return f"HTTP {exchange.response_status}"
  if not response_text.strip():
    return "No response text was returned."
  return None


def find_argument_mismatch_warnings(
  request_body: dict[str, Any],
  response_json: dict[str, Any],
  field_names: tuple[str, ...] = ("model", "tool_choice", "tools", "parallel_tool_calls"),
) -> list[str]:
  warnings: list[str] = []
  for field_name in field_names:
    if field_name not in request_body or field_name not in response_json:
      continue
    sent_value = request_body.get(field_name)
    returned_value = response_json.get(field_name)
    if not argument_values_match(field_name, sent_value, returned_value):
      warnings.append(
        f"WARNING: argument {field_name} was sent as {format_json_scalar(sent_value)} "
        f"and returned as {format_json_scalar(returned_value)}."
      )
  return warnings


def argument_values_match(field_name: str, sent_value: Any, returned_value: Any) -> bool:
  if sent_value == returned_value:
    return True
  if field_name == "model":
    return returned_model_name_matches(sent_value, returned_value)
  return False


def returned_model_name_matches(sent_value: Any, returned_value: Any) -> bool:
  if not isinstance(sent_value, str) or not isinstance(returned_value, str):
    return False
  if len(returned_value) <= len(sent_value) or not returned_value.startswith(sent_value):
    return False
  return returned_value[len(sent_value)] in "-._"


def looks_like_json_payload(value: str) -> bool:
  return value.startswith("{") or value.startswith("[")


def is_tool_call_payload(value: Any) -> bool:
  return isinstance(value, dict) and isinstance(value.get("name"), str) and "parameters" in value


def tool_is_available(tool_name: str, tools: Any) -> bool:
  if not isinstance(tools, list):
    return False
  for tool in tools:
    if not isinstance(tool, dict):
      continue
    if tool.get("name") == tool_name:
      return True
    function_block = tool.get("function")
    if isinstance(function_block, dict) and function_block.get("name") == tool_name:
      return True
  return False


def redact_headers(headers: dict[str, str]) -> dict[str, str]:
  redacted = dict(headers)
  authorization = redacted.get("Authorization")
  if authorization:
    redacted["Authorization"] = "Bearer ***REDACTED***"
  return redacted


def format_json_like(value: Any) -> str:
  return json.dumps(value, indent=2, sort_keys=True)


def format_json_scalar(value: Any) -> str:
  return json.dumps(value, sort_keys=True)


def determine_endpoint_status(result: EndpointExecutionResult) -> str:
  if not result.success:
    return "failed"
  if result.partial_success:
    return "partial"
  return "passed"


def determine_overall_status(*results: EndpointExecutionResult) -> str:
  if any(not result.success for result in results):
    return "failed"
  if any(result.partial_success for result in results):
    return "partial"
  return "passed"


def colorize_status(status: str) -> str:
  if status == "passed":
    label = "PASSED"
    color = GREEN
  elif status == "partial":
    label = "PARTIAL SUCCESS"
    color = ORANGE
  else:
    label = "FAILED"
    color = RED
  return f"{color}{label}{RESET}"
