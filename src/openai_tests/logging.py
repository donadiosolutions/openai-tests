from __future__ import annotations

import logging
import sys
from datetime import UTC, datetime
from typing import TextIO

_COLOR_MAP = {
  "INFO": "\x1b[32m",
  "WARNING": "\x1b[33m",
  "ERROR": "\x1b[31m",
}

logger = logging.getLogger("openai_tests")


class Rfc3339Formatter(logging.Formatter):
  def __init__(self, use_color: bool) -> None:
    super().__init__()
    self.use_color = use_color

  def format(self, record: logging.LogRecord) -> str:
    timestamp = datetime.fromtimestamp(record.created, tz=UTC).isoformat(timespec="seconds")
    level = record.levelname.upper()
    prefix = f"[{level}] {timestamp}: "
    if self.use_color and level in _COLOR_MAP:
      prefix = f"{_COLOR_MAP[level]}{prefix}\x1b[0m"
    message = record.getMessage()
    if record.exc_info:
      message = f"{message}\n{self.formatException(record.exc_info)}"
    return f"{prefix}{message}"


def configure_logging(
  *,
  force_color: bool | None = None,
  level: int = logging.INFO,
  stream: TextIO | None = None,
) -> logging.Logger:
  target_stream = stream or sys.stderr
  use_color = force_color if force_color is not None else bool(getattr(target_stream, "isatty", lambda: False)())
  handler = logging.StreamHandler(target_stream)
  handler.setFormatter(Rfc3339Formatter(use_color=use_color))

  logger.handlers.clear()
  logger.addHandler(handler)
  logger.setLevel(level)
  logger.propagate = False
  return logger
