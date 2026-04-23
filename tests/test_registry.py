from __future__ import annotations

from openai_tests import list_test_modules


def test_list_test_modules_defaults_to_empty_tuple() -> None:
  assert list_test_modules() == ()
