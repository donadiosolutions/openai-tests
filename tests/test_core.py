from __future__ import annotations

import argparse

from openai_tests import EndpointTestModule, render_module_listing


def configure_parser(_: argparse.ArgumentParser) -> None:
  return None


def handle_module(_: argparse.Namespace) -> int:
  return 0


def test_endpoint_test_module_render_formats_name_and_summary() -> None:
  module = EndpointTestModule(
    name="models",
    summary="List available models from a target endpoint.",
    configure_parser=configure_parser,
    handler=handle_module,
  )
  assert module.render() == "models: List available models from a target endpoint."


def test_render_module_listing_returns_placeholder_when_empty() -> None:
  assert render_module_listing(()) == (
    "No test modules are registered yet.\n"
    "Add them under src/openai_tests/test_modules/ and wire them into openai_tests.registry."
  )


def test_render_module_listing_renders_each_registered_module() -> None:
  modules = (
    EndpointTestModule(
      name="chat",
      summary="Exercise a chat-style endpoint.",
      configure_parser=configure_parser,
      handler=handle_module,
    ),
    EndpointTestModule(
      name="models",
      summary="List available models.",
      configure_parser=configure_parser,
      handler=handle_module,
    ),
  )
  assert render_module_listing(modules) == ("chat: Exercise a chat-style endpoint.\nmodels: List available models.")
