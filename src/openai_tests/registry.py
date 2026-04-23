from __future__ import annotations

from .core import EndpointTestModule

_TEST_MODULES: tuple[EndpointTestModule, ...] = ()


def list_test_modules() -> tuple[EndpointTestModule, ...]:
  return tuple(sorted(_TEST_MODULES, key=lambda module: module.name))
