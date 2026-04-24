from __future__ import annotations

from .core import EndpointTestModule
from .test_modules.text_simple import TEXT_SIMPLE_MODULE

_TEST_MODULES: tuple[EndpointTestModule, ...] = (TEXT_SIMPLE_MODULE,)


def list_test_modules() -> tuple[EndpointTestModule, ...]:
  return tuple(sorted(_TEST_MODULES, key=lambda module: module.name))
