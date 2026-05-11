from __future__ import annotations

from openai_tests import list_test_modules


def test_list_test_modules_returns_registered_modules() -> None:
  modules = list_test_modules()
  assert [module.name for module in modules] == ["asr-prep", "asr-simple", "asr-wer", "list-models", "text-simple"]
