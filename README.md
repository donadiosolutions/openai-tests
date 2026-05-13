# openai-tests

**Stop hand-building one-off cURL probes for every OpenAI-compatible endpoint.**

[![tests](https://img.shields.io/github/actions/workflow/status/donadiosolutions/openai-tests/ci.yml?branch=main&label=tests&logo=github)](https://github.com/donadiosolutions/openai-tests/actions/workflows/ci.yml)
[![code quality](https://img.shields.io/github/actions/workflow/status/donadiosolutions/openai-tests/github-code-scanning/codeql?branch=main&label=code%20quality&logo=github)](https://github.com/donadiosolutions/openai-tests/actions/workflows/github-code-scanning/codeql)
[![socket](https://img.shields.io/github/check-runs/donadiosolutions/openai-tests/main?nameFilter=Socket%20Security%3A%20Project%20Report&label=socket)](https://github.com/donadiosolutions/openai-tests/commits/main)
[![codecov](https://codecov.io/gh/donadiosolutions/openai-tests/graph/badge.svg)](https://codecov.io/gh/donadiosolutions/openai-tests)

[Quickstart](#quickstart) | [See It Work](#see-it-work) | [Checks](#checks) | [Recipes](#recipes) | [Documentation](#documentation)

`openai-tests` is a small CLI for proving whether an API really behaves like an OpenAI endpoint. It sends known-good requests, checks
the response shape and content, compares related API surfaces, and can print the exact redacted HTTP exchange when something looks
wrong.

If you only need one raw request, `curl` is still perfect. If you are validating a gateway, proxy, hosted model, local server, or
OpenAI-compatible deployment more than once, this gives you repeatable smoke tests instead of a folder full of hand-edited JSON bodies.

## Quickstart

```bash
pipx run openai-tests text-simple --help
```

Run the fastest useful check against OpenAI:

```bash
export OPENAI_API_KEY="sk-..."
pipx run openai-tests text-simple --model gpt-4.1-mini
```

Or point the same check at any compatible endpoint:

```bash
export OPENAI_TESTS_API_KEY="your-token"
pipx run openai-tests text-simple \
  --base-url https://your-openai-compatible-service.example \
  --model your-model
```

Base URLs may include `/v1` or omit it. Both `https://example.test` and `https://example.test/v1` work.

If you plan to use the tool repeatedly, install it once instead:

```bash
pipx install openai-tests
openai-tests text-simple --model gpt-4.1-mini
```

If you want to hack on the project itself, clone the repository and use the development environment:

```bash
git clone https://github.com/donadiosolutions/openai-tests.git
cd openai-tests
uv sync --all-groups
```

> [!IMPORTANT]
> **Trust surface:** endpoint tests read API keys from CLI flags or environment variables, send HTTP requests only to the configured
> `--base-url`, and redact `Authorization` in verbose output. `asr-simple` uses checked-in MP3 fixtures by default and runs `espeak-ng`
> only when you supply custom text through `--expected-transcript` without `--audio-file`; synthesized WAV files are written to a
> temporary directory and removed after the run. `uv sync` installs project dependencies into
> the local environment; `uv run poe socket` also runs `npm ci` from the checked-in lockfile for the pinned Socket CLI. To remove a local
> checkout, delete the repository directory and any generated `.venv` or `node_modules` directories.

## See It Work

```bash
$ pipx run openai-tests text-simple --model gpt-4.1-mini
/v1/chat/completions: PASSED
Question: What is the capital of France?
Response: Paris is the capital of France.

/v1/responses: PASSED
Question: What is the capital of France?
Response: Paris is the capital of France.

Overall: PASSED
```

That run did more than check for HTTP 200. It asked the same simple question through both `/v1/chat/completions` and `/v1/responses`,
extracted text from each response, verified the text was usable, and would have warned if the responses endpoint echoed important
parameters differently.

When you need to inspect the actual payloads, add `--verbose`:

```bash
pipx run openai-tests text-simple \
  --base-url https://your-openai-compatible-service.example \
  --model your-model \
  --verbose
```

Verbose mode prints the request URL, headers, JSON body, response status, response headers, and raw response body. Bearer tokens are
redacted.

## Checks

| Module | What it exercises | What it catches |
| --- | --- | --- |
| `list-models` | `GET /v1/models` | malformed model-list responses, missing required fields, non-JSON responses, HTTP failures |
| `text-simple` | `/v1/chat/completions` and `/v1/responses` | empty text, incompatible response shapes, parameter mismatches, unexpected tool-call-like output |
| `asr-simple` | `/v1/chat/completions` with audio input and `/v1/audio/transcriptions` | missing transcripts, wrong transcript content, streaming/non-streaming shape issues, metadata mismatches |
| `asr-prep` | local audio files | non-deterministic long-audio inputs by segmenting direct-child audio into fixed chunks |
| `asr-wer` | `/v1/audio/transcriptions` or `/v1/chat/completions` | batch ASR regressions, WER drift, throughput changes, prepared long-audio stitching issues |

Each module is intentionally small. The point is not to benchmark model quality. The point is to answer: "Can this endpoint accept the
same request shape my OpenAI client will send, and can I trust the response shape I get back?"

## Recipes

### List available models

```bash
pipx run openai-tests list-models \
  --base-url https://api.openai.com
```

Output is a schema check plus the returned model IDs:

```text
/v1/models: PASSED
Models:
- gpt-4.1-mini
- gpt-4.1
- gpt-4o-transcribe

Overall: PASSED
```

### Compare chat completions and responses

```bash
pipx run openai-tests text-simple \
  --base-url https://api.openai.com \
  --model gpt-4.1-mini
```

Use separate models when a provider routes the two APIs differently:

```bash
pipx run openai-tests text-simple \
  --model gpt-4.1-mini \
  --responses-model gpt-4.1
```

### Test speech recognition

```bash
pipx run openai-tests asr-simple \
  --base-url https://api.openai.com \
  --model gpt-4o-audio-preview
```

If the transcriptions endpoint needs a different model than chat completions,
pass `--transcriptions-model` explicitly.

For OpenAI-compatible providers that support penalty controls, pass
`--completions-frequency-penalty`, `--completions-repetition-penalty`,
`--transcriptions-frequency-penalty`, or
`--transcriptions-repetition-penalty`. Frequency penalty applies to token
frequency in generated text only; repetition penalty applies to tokens appearing
in the prompt and generated text.

By default, `asr-simple` sends two checked-in MP3 fixtures:

```text
1. Alpha through Zulu in NATO spelling words
2. The quick brown fox jumps over the lazy dog
```

To test your own fixture:

```bash
pipx run openai-tests asr-simple \
  --audio-file ./speech.wav \
  --audio-format wav \
  --expected-transcript \
  "Alpha Bravo Charlie Delta Echo Foxtrot Golf Hotel India Juliet"
```

To synthesize custom spoken text on demand with `espeak-ng`, omit `--audio-file` and provide only the transcript text:

```bash
pipx run openai-tests asr-simple \
  --expected-transcript "Please transcribe this sentence exactly."
```

### Run prepared ASR WER benchmarks

Use `asr-prep` when long audio should be segmented the same way every run:

```bash
uvx openai-tests asr-prep ./calls --overlap 3.0
```

This writes `./calls/prep/manifest.json`, `./calls/prep/report.txt`, and
30-second WAV chunks for supported direct-child audio files. The overlap
defaults to `3.0` seconds.

Create prepared ground truth, then evaluate a model against the stitched
per-source transcripts:

```bash
uvx openai-tests asr-wer ground ./calls --prep \
  --endpoint transcriptions \
  --transcriptions-model gpt-4o-transcribe

uvx openai-tests asr-wer eval ./calls --prep \
  --endpoint transcriptions \
  --transcriptions-model gpt-4o-transcribe
```

Prepared runs read chunks from `./calls/prep` and write combined transcripts to
root-level output folders such as `./calls/ground` or
`./calls/gpt-4o-transcribe_<epoch>`. Per-chunk audit transcripts are kept under
each output folder's `chunks/` directory.

### Pass provider-specific knobs

Optional API parameters stay unset until you pass them. JSON values can be inline or loaded from a file with `@path`.

```bash
pipx run openai-tests text-simple \
  --responses-metadata-json '{"suite":"compatibility-smoke"}' \
  --responses-temperature 0
```

Boolean parameters use paired flags, so you can distinguish "unset" from explicit true or false:

```bash
pipx run openai-tests text-simple --responses-store
pipx run openai-tests text-simple --no-responses-store
```

## Status Labels

| Status | Meaning |
| --- | --- |
| `PASSED` | The endpoint returned a usable response and no warnings were produced. |
| `PARTIAL SUCCESS` | The endpoint returned usable content, but a warning suggests compatibility drift. |
| `FAILED` | The request failed, the response shape was invalid, or the content check did not pass. |

The CLI exits with `0` only when all checked endpoints pass. It exits with `1` for failures or partial successes, and `2` for local
configuration errors such as invalid JSON arguments.

For ASR checks, each endpoint result also prints a simple word error rate counter as `WER: <percent> (<errors>/<reference words>)`.
The default acceptance rule allows the transcript to pass when either the expected-word threshold is met or the WER stays below `15%`.
Common NATO-style spelling variants such as `viktor`, `whisky`, `charly`, `romeu`, `uniforme`, `yanke`, and `zooloo` are normalized
before scoring.

## Configuration

Common options:

| Option | Environment fallback | Default |
| --- | --- | --- |
| `--api-key` | `OPENAI_API_KEY`, then `OPENAI_TESTS_API_KEY` | no authorization header |
| `--base-url` | `OPENAI_BASE_URL`, then `OPENAI_TESTS_BASE_URL` | `https://api.openai.com` |
| `--model` | `OPENAI_MODEL`, then `OPENAI_TESTS_MODEL` | module-specific |
| `--timeout` | none | `30` seconds |
| `--verbose` | none | off |

The live integration runner also loads `OPENAI_API_KEY` from a repository-root `.env` file before falling back to the inherited
environment.

## How It Works

The CLI keeps request construction explicit and inspectable. Modules use direct HTTP requests from the Python standard library rather
than an SDK, so the payloads stay close to the API surface being tested.

- Required endpoint fields receive conservative defaults.
- Optional endpoint fields remain `None` until the user passes them.
- `None` values are pruned before JSON or multipart requests are sent.
- String-or-object API parameters expose both plain string flags and `-json` flags.
- Full HTTP exchanges are captured for verbose output.
- Secrets are redacted before printing.

The module registry lives in `src/openai_tests/registry.py`. New endpoint checks belong under `src/openai_tests/test_modules/` and are
documented under `docs/`.

## Development and CI

Run the standard local checks before merging changes:

```bash
uv run poe fmt
uv run poe check
uv run poe socket
```

`uv run poe check` runs formatting checks, Ruff linting, type checking, actionlint, unit tests, live OpenAI integration tests, coverage
validation, and pre-commit hooks. The repository requires 100% line and branch coverage.

`uv run poe socket` installs the pinned Socket CLI from `package-lock.json`, generates CycloneDX manifests, and runs an authenticated
read-only Socket scan preflight. It requires `SOCKET_API_KEY`, `SOCKET_API_TOKEN`, or `SOCKET_CLI_API_TOKEN`.

GitHub Actions runs `unit` and `integration` in parallel, then a `validate` job succeeds only when both passed. Socket's GitHub App
publishes separate required dependency-security checks.

## Documentation

- [Documentation index](docs/index.md)
- [Installation and configuration](docs/installation.md)
- [CLI usage](docs/cli.md)
- [Live OpenAI integration tests](docs/integration.md)
- [text-simple module](docs/text-simple.md)
- [asr-simple module](docs/asr-simple.md)
- [asr-prep module](docs/asr-prep.md)
- [asr-wer module](docs/asr-wer.md)
- [list-models module](docs/list-models.md)
- [Development and verification](docs/development.md)

## FAQ

**Can I use it against a local service with no auth?**

Yes. If no API key is provided through `--api-key`, `OPENAI_API_KEY`, or `OPENAI_TESTS_API_KEY`, no `Authorization` header is sent.

**Is this a replacement for a full API conformance suite?**

No. It is a focused smoke-test tool. It is meant to catch obvious request/response incompatibilities quickly and repeatedly.

**Why not use the OpenAI SDK?**

The tests deliberately use direct HTTP requests so the request body, endpoint URL, response status, and raw response are easy to inspect.

## License

MIT. See [LICENSE](LICENSE).
