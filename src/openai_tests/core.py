from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass

ModuleHandler = Callable[[argparse.Namespace], int]
ParserConfigurator = Callable[[argparse.ArgumentParser], None]


@dataclass(frozen=True, slots=True)
class EndpointTestModule:
  name: str
  summary: str
  configure_parser: ParserConfigurator
  handler: ModuleHandler

  def render(self) -> str:
    return f"{self.name}: {self.summary}"


def render_module_listing(modules: tuple[EndpointTestModule, ...]) -> str:
  if not modules:
    return (
      "No test modules are registered yet.\n"
      "Add them under src/openai_tests/test_modules/ and wire them into openai_tests.registry."
    )
  return "\n".join(module.render() for module in modules)
