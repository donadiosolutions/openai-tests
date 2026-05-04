from __future__ import annotations

import argparse
import json
import os
from typing import Any, cast

from ..core import EndpointTestModule
from ._shared import (
  EndpointExecutionResult,
  build_api_url,
  colorize_status,
  determine_endpoint_status,
  determine_error_message,
  format_json_like,
  redact_headers,
  send_get_request,
)
from ._shared import (
  request as request,
)

DEFAULT_BASE_URL = "https://api.openai.com"
DEFAULT_TIMEOUT_SECONDS = 30.0


def configure_parser(parser: argparse.ArgumentParser) -> None:
  parser.add_argument(
    "--base-url",
    help="Base URL for the OpenAI-compatible API. Defaults to $OPENAI_BASE_URL or OpenAI's public API.",
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
    help="Show the full HTTP request and response.",
  )


def run(args: argparse.Namespace) -> int:
  result = run_models_test(
    base_url=resolve_base_url(args.base_url),
    api_key=resolve_api_key(args.api_key),
    timeout=args.timeout,
  )

  print_endpoint_result(result, verbose=args.verbose)
  print()

  overall_status = determine_endpoint_status(result)
  print(f"Overall: {colorize_status(overall_status)}")
  return 0 if overall_status == "passed" else 1


def resolve_base_url(cli_value: str | None) -> str:
  return cli_value or os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_TESTS_BASE_URL") or DEFAULT_BASE_URL


def resolve_api_key(cli_value: str | None) -> str | None:
  return cli_value or os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_TESTS_API_KEY")


def run_models_test(
  *,
  base_url: str,
  api_key: str | None,
  timeout: float,
) -> EndpointExecutionResult:
  exchange = send_get_request(
    url=build_api_url(base_url, "/v1/models"),
    api_key=api_key,
    timeout=timeout,
  )
  model_ids = extract_model_ids(exchange.response_json)
  response_text = "\n".join(model_ids)
  transport_error = determine_error_message(exchange, "ok")
  schema_error = None if transport_error is not None else validate_models_response_schema(exchange.response_json)
  error_message = transport_error or schema_error
  return EndpointExecutionResult(
    name="/v1/models",
    question="List available models.",
    response_text=response_text,
    success=error_message is None,
    exchange=exchange,
    error_message=error_message,
  )


def extract_model_ids(response_json: Any | None) -> list[str]:
  if not isinstance(response_json, dict):
    return []
  data = response_json.get("data")
  if not isinstance(data, list):
    return []
  model_ids: list[str] = []
  for item in data:
    if isinstance(item, dict) and isinstance(item.get("id"), str):
      model_ids.append(item["id"])
  return model_ids


def validate_models_response_schema(response_json: Any | None) -> str | None:
  if not isinstance(response_json, dict):
    return "Expected response JSON to be an object."
  if response_json.get("object") != "list":
    return 'Expected response.object to be "list".'

  data = response_json.get("data")
  if not isinstance(data, list):
    return "Expected response.data to be an array."

  for index, item in enumerate(data):
    path = f"response.data[{index}]"
    if not isinstance(item, dict):
      return f"Expected {path} to be an object."
    item_dict = cast("dict[str, Any]", item)
    model_id = item_dict.get("id")
    if not isinstance(model_id, str) or not model_id.strip():
      return f"Expected {path}.id to be a non-empty string."
    if item_dict.get("object") != "model":
      return f'Expected {path}.object to be "model".'
    if not isinstance(item_dict.get("created"), int):
      return f"Expected {path}.created to be an integer."
    owned_by = item_dict.get("owned_by")
    if not isinstance(owned_by, str) or not owned_by.strip():
      return f"Expected {path}.owned_by to be a non-empty string."

  return None


def print_endpoint_result(result: EndpointExecutionResult, *, verbose: bool) -> None:
  print(f"{result.name}: {colorize_status(determine_endpoint_status(result))}")
  print("Models:")
  for model_id in result.response_text.splitlines():
    print(f"- {model_id}")
  if not result.response_text:
    print("(none)")
  if result.error_message is not None:
    print(f"Error: {result.error_message}")
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


LIST_MODELS_MODULE = EndpointTestModule(
  name="list-models",
  summary="List available models through the models endpoint.",
  configure_parser=configure_parser,
  handler=run,
)
