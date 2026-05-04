# Live OpenAI Integration Tests

The integration suite runs the implemented modules against the actual OpenAI API at `https://api.openai.com`. It is intended to catch
drift between the local request builders and the live endpoint behavior.

## Authentication

Run the suite through Poe:

```bash
uv run poe test-integration
```

The runner resolves `OPENAI_API_KEY` in this order:

1. repository-root `.env`, when it contains `OPENAI_API_KEY`
2. inherited process environment

If no key is available, the integration tests are skipped by default for local
runs. CI provides the key through `secrets.OPENAI_API_KEY` and sets
`OPENAI_TESTS_REQUIRE_OPENAI_API_KEY=1`, so the integration job fails instead
of silently skipping when the secret is missing.

## Covered Modules

The suite currently runs:

- `text-simple` against `/v1/chat/completions` and `/v1/responses`
- `asr-simple` against `/v1/chat/completions` and `/v1/audio/transcriptions`
- `list-models` against `/v1/models`

The ASR test uses a small checked-in MP3 fixture containing the default expected transcript. This keeps the live integration suite
independent from `espeak-ng` availability on the CI runner while preserving the module's normal synthesis behavior for CLI users.

## Model Overrides

The integration tests use the module defaults unless these environment variables are set:

- `OPENAI_TESTS_INTEGRATION_TEXT_MODEL`
- `OPENAI_TESTS_INTEGRATION_RESPONSES_MODEL`
- `OPENAI_TESTS_INTEGRATION_ASR_COMPLETIONS_MODEL`
- `OPENAI_TESTS_INTEGRATION_ASR_TRANSCRIPTIONS_MODEL`

These variables are integration-test specific. They do not change the CLI module defaults.

## CI Layout

GitHub Actions runs three jobs:

- `unit`: formatting, linting, typing, actionlint, unit tests, coverage validation, pre-commit, and Safety.
- `integration`: live OpenAI integration tests through `uv run poe check-integration`.
- `validate`: depends on `unit` and `integration` and succeeds only when both jobs succeeded.

The `unit` and `integration` jobs run in parallel. The `validate` job is the aggregate status check.
