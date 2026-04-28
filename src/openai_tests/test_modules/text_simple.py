from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from ..core import EndpointTestModule
from ._shared import (
  EndpointExecutionResult,
  HttpExchange,
  build_api_url,
  choose_string_or_json,
  colorize_status,
  determine_endpoint_status,
  determine_error_message,
  determine_overall_status,
  extract_chat_response_text,
  find_argument_mismatch_warnings,
  format_json_like,
  format_json_scalar,
  is_tool_call_payload,
  looks_like_json_payload,
  normalize_text_content,
  parse_json_dict,
  parse_json_list,
  parse_json_value,
  prune_none,
  redact_headers,
  send_json_request,
  tool_is_available,
  try_parse_json,
)
from ._shared import (
  request as request,
)

DEFAULT_BASE_URL = "https://api.openai.com"
DEFAULT_MODEL = "gpt-4.1-mini"
DEFAULT_SYSTEM_PROMPT = "You are a concise assistant."
DEFAULT_DEVELOPER_PROMPT = "Answer the user's question in one short sentence."
DEFAULT_USER_PROMPT = "What is the capital of France?"
DEFAULT_TIMEOUT_SECONDS = 30.0


def configure_parser(parser: argparse.ArgumentParser) -> None:
  parser.add_argument(
    "--base-url",
    help="Base URL for the OpenAI-compatible API. Defaults to $OPENAI_BASE_URL or OpenAI's public API.",
  )
  parser.add_argument(
    "--model",
    help="Model to use for both endpoints. Defaults to $OPENAI_MODEL or gpt-4.1-mini.",
  )
  parser.add_argument(
    "--responses-model",
    help="Override the model used only for the /v1/responses request.",
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

  prompts = parser.add_argument_group("Prompt text")
  prompts.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
  prompts.add_argument("--developer-prompt", default=DEFAULT_DEVELOPER_PROMPT)
  prompts.add_argument("--user-prompt", default=DEFAULT_USER_PROMPT)

  responses = parser.add_argument_group("Responses API parameters")
  responses.add_argument("--responses-background", action=argparse.BooleanOptionalAction, default=None)
  responses.add_argument("--responses-context-management-json")
  responses.add_argument("--responses-conversation")
  responses.add_argument("--responses-conversation-json")
  responses.add_argument("--responses-include", action="append")
  responses.add_argument("--responses-include-json")
  responses.add_argument("--responses-input-json")
  responses.add_argument("--responses-instructions")
  responses.add_argument("--responses-instructions-json")
  responses.add_argument("--responses-max-output-tokens", type=int)
  responses.add_argument("--responses-max-tool-calls", type=int)
  responses.add_argument("--responses-metadata-json")
  responses.add_argument("--responses-parallel-tool-calls", action=argparse.BooleanOptionalAction, default=None)
  responses.add_argument("--responses-previous-response-id")
  responses.add_argument("--responses-prompt-id")
  responses.add_argument("--responses-prompt-version")
  responses.add_argument("--responses-prompt-variables-json")
  responses.add_argument("--responses-prompt-cache-key")
  responses.add_argument(
    "--responses-prompt-cache-retention",
    choices=("in-memory", "24h"),
  )
  responses.add_argument("--responses-reasoning-json")
  responses.add_argument(
    "--responses-reasoning-effort",
    choices=("none", "minimal", "low", "medium", "high", "xhigh"),
  )
  responses.add_argument(
    "--responses-reasoning-generate-summary",
    choices=("auto", "concise", "detailed"),
  )
  responses.add_argument(
    "--responses-reasoning-summary",
    choices=("auto", "concise", "detailed"),
  )
  responses.add_argument("--responses-safety-identifier")
  responses.add_argument(
    "--responses-service-tier",
    choices=("auto", "default", "flex", "scale", "priority"),
  )
  responses.add_argument("--responses-store", action=argparse.BooleanOptionalAction, default=None)
  responses.add_argument("--responses-stream", action=argparse.BooleanOptionalAction, default=None)
  responses.add_argument("--responses-stream-options-json")
  responses.add_argument("--responses-include-obfuscation", action=argparse.BooleanOptionalAction, default=None)
  responses.add_argument("--responses-temperature", type=float)
  responses.add_argument("--responses-text-json")
  responses.add_argument("--responses-text-format-json")
  responses.add_argument(
    "--responses-text-verbosity",
    choices=("low", "medium", "high"),
  )
  responses.add_argument("--responses-tool-choice")
  responses.add_argument("--responses-tool-choice-json")
  responses.add_argument("--responses-tools-json")
  responses.add_argument("--responses-top-logprobs", type=int)
  responses.add_argument("--responses-top-p", type=float)
  responses.add_argument(
    "--responses-truncation",
    choices=("auto", "disabled"),
  )
  responses.add_argument("--responses-user")


def run(args: argparse.Namespace) -> int:
  try:
    base_url = resolve_base_url(args.base_url)
    api_key = resolve_api_key(args.api_key)
    chat_payload = build_chat_request_payload(args, resolve_model(args.model))
    responses_payload = build_responses_request_config(args)
  except ValueError as exc:
    print(f"Configuration error: {exc}", file=sys.stderr)
    return 2

  chat_result = run_chat_completion_test(
    base_url=base_url,
    api_key=api_key,
    payload=chat_payload,
    question=args.user_prompt,
    timeout=args.timeout,
  )
  responses_result = run_responses_test(
    base_url=base_url,
    api_key=api_key,
    normalized_payload=responses_payload,
    question=args.user_prompt,
    timeout=args.timeout,
  )

  print_endpoint_result(chat_result, verbose=args.verbose)
  print()
  print_endpoint_result(responses_result, verbose=args.verbose)
  print()

  overall_status = determine_overall_status(chat_result, responses_result)
  print(f"Overall: {colorize_status(overall_status)}")
  return 0 if overall_status == "passed" else 1


def resolve_base_url(cli_value: str | None) -> str:
  return cli_value or os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_TESTS_BASE_URL") or DEFAULT_BASE_URL


def resolve_model(cli_value: str | None) -> str:
  return cli_value or os.getenv("OPENAI_MODEL") or os.getenv("OPENAI_TESTS_MODEL") or DEFAULT_MODEL


def resolve_api_key(cli_value: str | None) -> str | None:
  return cli_value or os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_TESTS_API_KEY")


def build_chat_request_payload(args: argparse.Namespace, model: str) -> dict[str, Any]:
  return {
    "model": model,
    "messages": [
      {
        "role": "system",
        "content": build_chat_system_prompt(args.system_prompt, args.developer_prompt),
      },
      {
        "role": "user",
        "content": args.user_prompt,
      },
    ],
  }


def build_chat_system_prompt(system_prompt: str, developer_prompt: str) -> str:
  prompts = [prompt.strip() for prompt in (system_prompt, developer_prompt) if prompt.strip()]
  return "\n\n".join(prompts)


def build_responses_request_config(args: argparse.Namespace) -> dict[str, Any]:
  include_values = list(args.responses_include or [])
  include_json = parse_json_list(args.responses_include_json, "responses-include-json")
  if include_json is not None:
    include_values.extend(str(item) for item in include_json)

  conversation = choose_string_or_json(
    string_value=args.responses_conversation,
    json_value=args.responses_conversation_json,
    field_name="conversation",
  )
  instructions = choose_string_or_json(
    string_value=args.responses_instructions,
    json_value=args.responses_instructions_json,
    field_name="instructions",
  )
  tool_choice = choose_string_or_json(
    string_value=args.responses_tool_choice,
    json_value=args.responses_tool_choice_json,
    field_name="tool_choice",
  )

  prompt_config = build_prompt_config(args)
  reasoning_config = build_reasoning_config(args)
  stream_options_config = build_stream_options_config(args)
  text_config = build_text_config(args)
  tools_config = parse_json_list(args.responses_tools_json, "responses-tools-json")
  context_management = parse_json_list(
    args.responses_context_management_json,
    "responses-context-management-json",
  )
  metadata = parse_json_dict(args.responses_metadata_json, "responses-metadata-json")
  input_value = parse_json_value(args.responses_input_json, "responses-input-json")
  if input_value is None:
    input_value = build_default_responses_input(args.system_prompt, args.developer_prompt, args.user_prompt)

  return {
    "background": args.responses_background,
    "context_management": context_management,
    "conversation": conversation,
    "include": include_values or None,
    "input": input_value,
    "instructions": instructions,
    "max_output_tokens": args.responses_max_output_tokens,
    "max_tool_calls": args.responses_max_tool_calls,
    "metadata": metadata,
    "model": resolve_model(args.responses_model or args.model),
    "parallel_tool_calls": args.responses_parallel_tool_calls,
    "previous_response_id": args.responses_previous_response_id,
    "prompt": prompt_config,
    "prompt_cache_key": args.responses_prompt_cache_key,
    "prompt_cache_retention": args.responses_prompt_cache_retention,
    "reasoning": reasoning_config,
    "safety_identifier": args.responses_safety_identifier,
    "service_tier": args.responses_service_tier,
    "store": args.responses_store,
    "stream": args.responses_stream,
    "stream_options": stream_options_config,
    "temperature": args.responses_temperature,
    "text": text_config,
    "tool_choice": tool_choice,
    "tools": tools_config,
    "top_logprobs": args.responses_top_logprobs,
    "top_p": args.responses_top_p,
    "truncation": args.responses_truncation,
    "user": args.responses_user,
  }


def build_default_responses_input(
  system_prompt: str,
  developer_prompt: str,
  user_prompt: str,
) -> list[dict[str, str]]:
  return [
    {"role": "system", "content": system_prompt},
    {"role": "developer", "content": developer_prompt},
    {"role": "user", "content": user_prompt},
  ]


def build_prompt_config(args: argparse.Namespace) -> dict[str, Any] | None:
  variables = parse_json_dict(args.responses_prompt_variables_json, "responses-prompt-variables-json")
  if args.responses_prompt_id is None and args.responses_prompt_version is None and variables is None:
    return None
  if args.responses_prompt_id is None:
    raise ValueError("responses-prompt-id is required when prompt version or variables are provided")
  return {
    "id": args.responses_prompt_id,
    "variables": variables,
    "version": args.responses_prompt_version,
  }


def build_reasoning_config(args: argparse.Namespace) -> dict[str, Any] | None:
  reasoning = parse_json_dict(args.responses_reasoning_json, "responses-reasoning-json") or None
  if (
    reasoning is None
    and args.responses_reasoning_effort is None
    and args.responses_reasoning_generate_summary is None
    and args.responses_reasoning_summary is None
  ):
    return None
  config = reasoning or {}
  config["effort"] = args.responses_reasoning_effort or config.get("effort")
  config["generate_summary"] = args.responses_reasoning_generate_summary or config.get("generate_summary")
  config["summary"] = args.responses_reasoning_summary or config.get("summary")
  return {
    "effort": config.get("effort"),
    "generate_summary": config.get("generate_summary"),
    "summary": config.get("summary"),
  }


def build_stream_options_config(args: argparse.Namespace) -> dict[str, Any] | None:
  stream_options = parse_json_dict(args.responses_stream_options_json, "responses-stream-options-json") or None
  if stream_options is None and args.responses_include_obfuscation is None:
    return None
  config = stream_options or {}
  config["include_obfuscation"] = (
    args.responses_include_obfuscation
    if args.responses_include_obfuscation is not None
    else config.get("include_obfuscation")
  )
  return {"include_obfuscation": config.get("include_obfuscation")}


def build_text_config(args: argparse.Namespace) -> dict[str, Any] | None:
  text_config = parse_json_dict(args.responses_text_json, "responses-text-json") or None
  format_config = parse_json_value(args.responses_text_format_json, "responses-text-format-json")
  if text_config is None and format_config is None and args.responses_text_verbosity is None:
    return None
  config = text_config or {}
  if format_config is not None:
    config["format"] = format_config
  elif "format" not in config:
    config["format"] = None
  if args.responses_text_verbosity is not None:
    config["verbosity"] = args.responses_text_verbosity
  elif "verbosity" not in config:
    config["verbosity"] = None
  return {
    "format": config.get("format"),
    "verbosity": config.get("verbosity"),
  }


def run_chat_completion_test(
  *,
  base_url: str,
  api_key: str | None,
  payload: dict[str, Any],
  question: str,
  timeout: float,
) -> EndpointExecutionResult:
  exchange = send_json_request(
    url=build_api_url(base_url, "/v1/chat/completions"),
    api_key=api_key,
    payload=payload,
    timeout=timeout,
  )
  response_text = extract_chat_response_text(exchange.response_json)
  error_message = determine_error_message(exchange, response_text)
  return EndpointExecutionResult(
    name="/v1/chat/completions",
    question=question,
    response_text=response_text,
    success=error_message is None,
    exchange=exchange,
    error_message=error_message,
  )


def run_responses_test(
  *,
  base_url: str,
  api_key: str | None,
  normalized_payload: dict[str, Any],
  question: str,
  timeout: float,
) -> EndpointExecutionResult:
  pruned_payload = prune_none(normalized_payload)
  exchange = send_json_request(
    url=build_api_url(base_url, "/v1/responses"),
    api_key=api_key,
    payload=pruned_payload,
    timeout=timeout,
  )
  response_text = extract_responses_output_text(exchange.response_json)
  error_message = determine_error_message(exchange, response_text)
  warnings = build_responses_warnings(
    request_body=pruned_payload,
    response_json=exchange.response_json,
    response_text=response_text,
  )
  return EndpointExecutionResult(
    name="/v1/responses",
    question=question,
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


def build_responses_warnings(
  *,
  request_body: dict[str, Any],
  response_json: Any | None,
  response_text: str,
) -> list[str]:
  warnings: list[str] = []
  if isinstance(response_json, dict):
    warnings.extend(find_argument_mismatch_warnings(request_body, response_json))

  stripped_response_text = response_text.strip()
  if looks_like_json_payload(stripped_response_text):
    parsed_response_text = try_parse_json(stripped_response_text)
    if parsed_response_text is None:
      warnings.append("WARNING: returned JSON was not valid.")
    elif is_tool_call_payload(parsed_response_text):
      tool_name = str(parsed_response_text.get("name"))
      request_tools = request_body.get("tools")
      if tool_is_available(tool_name, request_tools):
        return warnings
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


def extract_responses_output_text(response_json: Any | None) -> str:
  if not isinstance(response_json, dict):
    return ""
  output_text = response_json.get("output_text")
  if isinstance(output_text, str):
    return output_text.strip()

  output_items = response_json.get("output")
  if isinstance(output_items, list):
    parts: list[str] = []
    for item in output_items:
      if not isinstance(item, dict):
        continue
      if item.get("type") == "message" or item.get("role") == "assistant":
        text = normalize_text_content(item.get("content"))
        if text:
          parts.append(text)
      elif isinstance(item.get("text"), str):
        parts.append(item["text"].strip())
    return "\n".join(part for part in parts if part)

  return ""


def print_endpoint_result(result: EndpointExecutionResult, *, verbose: bool) -> None:
  print(f"{result.name}: {colorize_status(determine_endpoint_status(result))}")
  print(f"Question: {result.question}")
  print(f"Response: {result.response_text or '(none)'}")
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


TEXT_SIMPLE_MODULE = EndpointTestModule(
  name="text-simple",
  summary="Ask a simple question through both chat completions and responses.",
  configure_parser=configure_parser,
  handler=run,
)
