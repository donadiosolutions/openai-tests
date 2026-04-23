from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from typing import cast

from .core import render_module_listing
from .logging import configure_logging
from .registry import list_test_modules

CommandHandler = Callable[[argparse.Namespace], int]


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    prog="openai-tests",
    description="CLI scaffold for OpenAI-compatible endpoint tests.",
  )
  subparsers = parser.add_subparsers(dest="command")
  modules_parser = subparsers.add_parser(
    "modules",
    help="List the registered endpoint test modules.",
  )
  modules_parser.set_defaults(handler=handle_modules)
  return parser


def handle_modules(_: argparse.Namespace) -> int:
  print(render_module_listing(list_test_modules()))
  return 0


def main(argv: Sequence[str] | None = None) -> int:
  configure_logging()
  parser = build_parser()
  args = parser.parse_args(argv)
  handler = cast(CommandHandler | None, getattr(args, "handler", None))
  if handler is None:
    parser.print_help()
    return 0
  return handler(args)
