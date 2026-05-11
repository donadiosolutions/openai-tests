from __future__ import annotations

from .core import EndpointTestModule
from .test_modules.asr_simple import ASR_SIMPLE_MODULE
from .test_modules.asr_wer import ASR_WER_MODULE
from .test_modules.list_models import LIST_MODELS_MODULE
from .test_modules.text_simple import TEXT_SIMPLE_MODULE

_TEST_MODULES: tuple[EndpointTestModule, ...] = (
  ASR_SIMPLE_MODULE,
  ASR_WER_MODULE,
  LIST_MODELS_MODULE,
  TEXT_SIMPLE_MODULE,
)


def list_test_modules() -> tuple[EndpointTestModule, ...]:
  return tuple(sorted(_TEST_MODULES, key=lambda module: module.name))
