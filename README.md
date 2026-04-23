# openai-tests

CLI scaffold for OpenAI-compatible endpoint tests.

## Current status

This repository currently contains the CLI shell, module registry, and project automation baseline only. No endpoint test modules are implemented yet.

Future endpoint test modules should live under `src/openai_tests/test_modules/` and be registered in `src/openai_tests/registry.py`.

## Prerequisites

- `pyenv`
- `uv`
- Python `3.13.3`

## Bootstrap

```bash
uv sync --all-groups
uv run poe check
uv run poe safety
```

## Common tasks

```bash
uv run poe fmt
uv run poe lint
uv run poe typecheck
uv run poe test
uv run poe coverage
uv run poe safety
```

`uv run poe safety` uses the Safety CLI and requires `SAFETY_API_KEY` to be set in the environment. GitHub Actions uses the organization `SAFETY_API_KEY` secret for the same command.

## CLI

Show the available command surface:

```bash
uv run openai-tests --help
```

List the registered endpoint test modules:

```bash
uv run openai-tests modules
```
