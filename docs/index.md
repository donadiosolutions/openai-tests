# Documentation Index

`openai-tests` is organized around endpoint test modules. Each module owns a CLI subcommand, builds requests for one or more API
surfaces, runs those requests against a target base URL, and reports whether the target behaved like an OpenAI-compatible endpoint.

## Topics

- [Installation and configuration](installation.md): required tools, dependency bootstrap, API key configuration, and environment
  variable precedence.
- [CLI usage](cli.md): top-level commands, common arguments, JSON argument conventions, output format, status labels, and exit codes.
- [Live OpenAI integration tests](integration.md): live endpoint coverage, API-key loading, model overrides, and CI layout.
- [text-simple module](text-simple.md): the text-generation compatibility check for `/v1/chat/completions` and `/v1/responses`.
- [asr-simple module](asr-simple.md): the speech-recognition compatibility check for `/v1/chat/completions` and
  `/v1/audio/transcriptions`.
- [list-models module](list-models.md): the models-listing compatibility check for `GET /v1/models`.
- [Development and verification](development.md): module architecture, shared utilities, adding modules, required checks, and
  repository quality gates.

## Implemented Modules

`text-simple` asks a deterministic text question through both chat completions and responses. It verifies that both endpoints return
text and warns when the responses endpoint echoes selected parameters differently or returns a tool call when no suitable tool was sent.

`asr-simple` uses two bundled MP3 speech samples by default, or a caller-provided
audio file or synthesized text when requested, then transcribes each sample
through both chat completions and audio transcriptions, verifies that enough
expected words are present, reports WER, and warns when returned metadata
suggests parameters changed in transit.

`list-models` lists every available model through `GET /v1/models` and verifies the response shape against the official
models-list schema.

## Source Layout

- `src/openai_tests/cli.py`: builds the top-level CLI and registers module subcommands.
- `src/openai_tests/registry.py`: contains the module registry used by the CLI.
- `src/openai_tests/core.py`: defines the `EndpointTestModule` interface.
- `src/openai_tests/test_modules/text_simple.py`: implements `text-simple`.
- `src/openai_tests/test_modules/asr_simple.py`: implements `asr-simple`.
- `src/openai_tests/test_modules/list_models.py`: implements `list-models`.
- `src/openai_tests/test_modules/_shared.py`: shared HTTP, JSON, output formatting, status, warning, and parsing helpers.
- `tests/`: isolated unit tests for the CLI, module registry, shared behavior, and endpoint modules.
- `tests/integration/`: live OpenAI integration tests that exercise the implemented modules against `https://api.openai.com`.
- `scripts/run_integration_tests.py`: integration-test runner with optional container setup and `.env` API-key loading.

## Design Principles

The project favors direct HTTP requests and standard-library code so endpoint behavior is easy to inspect. Optional API parameters are
omitted unless the user provides them, while required fields receive conservative defaults. Every module prints a concise per-endpoint
summary by default and can print complete redacted HTTP exchanges with `--verbose`.
