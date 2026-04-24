from __future__ import annotations

from openai_tests import list_test_modules


def test_list_test_modules_returns_text_simple_module() -> None:
  modules = list_test_modules()
  assert len(modules) == 1
  assert modules[0].name == "text-simple"
