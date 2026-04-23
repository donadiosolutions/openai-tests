from __future__ import annotations

from openai_tests import EndpointTestModule, render_module_listing


def test_endpoint_test_module_render_formats_name_and_summary() -> None:
  module = EndpointTestModule(name="models", summary="List available models from a target endpoint.")
  assert module.render() == "models: List available models from a target endpoint."


def test_render_module_listing_returns_placeholder_when_empty() -> None:
  assert render_module_listing(()) == (
    "No test modules are registered yet.\n"
    "Add them under src/openai_tests/test_modules/ and wire them into openai_tests.registry."
  )


def test_render_module_listing_renders_each_registered_module() -> None:
  modules = (
    EndpointTestModule(name="chat", summary="Exercise a chat-style endpoint."),
    EndpointTestModule(name="models", summary="List available models."),
  )
  assert render_module_listing(modules) == ("chat: Exercise a chat-style endpoint.\nmodels: List available models.")
