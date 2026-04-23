from __future__ import annotations

import io
import logging
import re
import sys

import pytest

from openai_tests.logging import Rfc3339Formatter, configure_logging, logger


def test_formatter_uses_expected_prefix_without_color() -> None:
  record = logging.LogRecord(
    name="test",
    level=logging.INFO,
    pathname=__file__,
    lineno=1,
    msg="hello",
    args=(),
    exc_info=None,
  )
  formatter = Rfc3339Formatter(use_color=False)
  rendered = formatter.format(record)
  assert rendered.endswith("hello")
  assert re.match(r"^\[INFO\] .+\+00:00: hello$", rendered)


def test_formatter_adds_color_when_enabled() -> None:
  record = logging.LogRecord(
    name="test",
    level=logging.WARNING,
    pathname=__file__,
    lineno=1,
    msg="careful",
    args=(),
    exc_info=None,
  )
  formatter = Rfc3339Formatter(use_color=True)
  rendered = formatter.format(record)
  assert "\x1b[" in rendered
  assert rendered.endswith("careful")


def test_formatter_appends_exception_output() -> None:
  try:
    raise ValueError("broken")
  except ValueError:
    record = logging.LogRecord(
      name="test",
      level=logging.ERROR,
      pathname=__file__,
      lineno=1,
      msg="failure",
      args=(),
      exc_info=sys.exc_info(),
    )
  formatter = Rfc3339Formatter(use_color=False)
  rendered = formatter.format(record)
  assert "failure" in rendered
  assert "ValueError: broken" in rendered


def test_configure_logging_replaces_handlers() -> None:
  stream = io.StringIO()
  configured = configure_logging(force_color=False, stream=stream)
  assert configured is logger
  assert len(logger.handlers) == 1

  configured.info("sample")
  output = stream.getvalue().strip()
  assert output.endswith("sample")


def test_configure_logging_defaults_to_stderr(capsys: pytest.CaptureFixture[str]) -> None:
  configured = configure_logging(force_color=False)
  configured.error("default-stream")
  captured = capsys.readouterr()
  assert "default-stream" in captured.err


class TtyStream(io.StringIO):
  def isatty(self) -> bool:
    return True


def test_configure_logging_uses_stream_tty_when_force_color_is_none() -> None:
  stream = TtyStream()
  configured = configure_logging(stream=stream)
  configured.warning("tty")
  assert "\x1b[" in stream.getvalue()
