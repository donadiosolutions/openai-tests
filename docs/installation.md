# Installation and Configuration

## Prerequisites

The repository expects these tools:

- `pyenv`
- `uv`
- Python `3.13.3`, managed through `.python-version`

The `asr-simple` module also needs `espeak-ng` when it synthesizes its default audio fixture. If `espeak-ng` is not installed, pass an
existing audio file with `--audio-file`.

## Bootstrap

Install the project and development dependencies:

```bash
uv sync --all-groups
```

Confirm that the CLI is available:

```bash
uv run openai-tests --help
uv run openai-tests modules
```

## Authentication

All endpoint modules use bearer-token authentication when an API key is available. The lookup order is:

1. `--api-key`
2. `OPENAI_API_KEY`
3. `OPENAI_TESTS_API_KEY`

If no key is available, requests are sent without an `Authorization` header. This is useful for local OpenAI-compatible services that
do not require authentication.

The integration runner has one extra convenience for live OpenAI tests: when `.env` exists at the repository root and contains
`OPENAI_API_KEY`, that value is inserted into the integration-test environment before falling back to the inherited environment. This
lets local runs use `.env`, while CI can provide the same variable through GitHub Actions secrets.

## Base URL

The target base URL lookup order is:

1. `--base-url`
2. `OPENAI_BASE_URL`
3. `OPENAI_TESTS_BASE_URL`
4. `https://api.openai.com`

The CLI handles base URLs with or without a trailing `/v1`. For example, both `https://example.test` and `https://example.test/v1`
produce endpoint URLs under `/v1/...`.

## Model Defaults

`text-simple` uses `--model`, `OPENAI_MODEL`, `OPENAI_TESTS_MODEL`, or `gpt-4.1-mini`.

`asr-simple` uses `--model`, `OPENAI_MODEL`, `OPENAI_TESTS_MODEL`, or `gpt-4o-audio-preview` for chat completions. It uses
`--transcriptions-model`, `OPENAI_TRANSCRIPTIONS_MODEL`, `OPENAI_TESTS_TRANSCRIPTIONS_MODEL`, or `gpt-4o-transcribe` for audio
transcriptions.

## Safety Scan

The repository includes a Safety CLI task:

```bash
uv run poe safety
```

This task requires `SAFETY_API_KEY` in the environment. CI uses the organization secret with the same name.

## Local Verification

The standard local verification commands are:

```bash
uv run poe fmt
uv run poe check
uv run poe safety
```

`uv run poe check` runs formatting checks, Ruff linting, type checking, actionlint, unit and integration tests, coverage validation, and
pre-commit hooks.
