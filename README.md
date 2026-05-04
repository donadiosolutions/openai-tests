# openai-tests

[![codecov](https://codecov.io/gh/donadiosolutions/openai-tests/graph/badge.svg)](https://codecov.io/gh/donadiosolutions/openai-tests)

`openai-tests` is a small CLI for probing OpenAI-compatible endpoints with focused compatibility checks. Each test module sends a
simple, inspectable request through two related API surfaces, prints the observed responses, and flags failures or suspicious response
shape changes.

## Quickstart

Install the project dependencies:

```bash
uv sync --all-groups
```

List the available test modules:

```bash
uv run openai-tests modules
```

Run the text-generation check against chat completions and responses:

```bash
uv run openai-tests text-simple \
  --base-url https://api.openai.com \
  --model gpt-4.1-mini
```

Run the speech-recognition check against chat completions and audio transcriptions:

```bash
uv run openai-tests asr-simple \
  --base-url https://api.openai.com \
  --model gpt-4o-audio-preview \
  --transcriptions-model gpt-4o-transcribe
```

List available models and validate the models-list response schema:

```bash
uv run openai-tests list-models \
  --base-url https://api.openai.com
```

These commands read the API key from `OPENAI_API_KEY` or `OPENAI_TESTS_API_KEY` unless `--api-key` is provided.

Run the live integration suite against OpenAI:

```bash
uv run poe test-integration
```

For integration tests, `OPENAI_API_KEY` is loaded from `.env` first when that file is present, then from the process environment.

Run the Socket dependency-security gate:

```bash
uv run poe socket
```

The Socket task reads `SOCKET_API_KEY`, `SOCKET_API_TOKEN`, or
`SOCKET_CLI_API_TOKEN`.

## Documentation

- [Documentation index](docs/index.md)
- [Installation and configuration](docs/installation.md)
- [CLI usage](docs/cli.md)
- [Live OpenAI integration tests](docs/integration.md)
- [text-simple module](docs/text-simple.md)
- [asr-simple module](docs/asr-simple.md)
- [list-models module](docs/list-models.md)
- [Development and verification](docs/development.md)
