# Development and Verification

## Module Interface

Endpoint test modules are registered with `EndpointTestModule` from `src/openai_tests/core.py`. A module provides:

- `name`: CLI subcommand name.
- `summary`: short description used in module listings and help text.
- `configure_parser`: function that adds module-specific CLI arguments.
- `handler`: function that receives parsed arguments and returns an exit code.

The CLI loads modules from `src/openai_tests/registry.py`, sorts them by name for display, and creates one subcommand per module.

## Shared Utilities

Common endpoint-test behavior lives in `src/openai_tests/test_modules/_shared.py`. It provides:

- `HttpExchange`: captured HTTP request and response data.
- `EndpointExecutionResult`: endpoint-level result data used by printers.
- JSON parsing helpers with `@file` support.
- optional-value pruning for request payloads.
- GET request sending with HTTP and URL error capture.
- JSON request sending with HTTP and URL error capture.
- base URL normalization.
- chat-completions text extraction.
- text-content normalization for string, object, and list shapes.
- error-message, status, and overall-status helpers.
- argument-mismatch and tool-availability warning helpers.
- redaction and JSON formatting helpers.

Endpoint modules should use these utilities before adding module-local equivalents.

## Adding a Module

1. Add the implementation under `src/openai_tests/test_modules/`.
2. Add unit tests under `tests/`.
3. Register the module in `src/openai_tests/registry.py`.
4. Add documentation under `docs/`.
5. Update `README.md` with a quickstart if the module is user-facing.

For behavior changes, write or update tests first and verify the new or changed test fails for the expected reason before implementing
the behavior.

## Request Design

Required endpoint parameters should receive sensible defaults. Optional endpoint parameters should remain unset until the user provides
them and should be omitted from the final request.

For JSON body requests, use `prune_none` before sending. For multipart form data, omit fields whose values are `None`.

When an API parameter supports either a string shorthand or a structured JSON object, expose both a string flag and a `-json` flag, and
reject calls that provide both.

## Tests

Unit tests are isolated and mock external systems. They cover:

- request payload construction
- JSON parsing and invalid JSON paths
- base URL normalization
- response text extraction
- warning construction
- HTTP error and URL error handling
- CLI execution paths
- status rendering and redaction
- edge cases needed to preserve 100% line and branch coverage

Integration tests are reserved for `tests/integration/` and are marked with `@pytest.mark.integration`. The current suite calls the
actual OpenAI API at `https://api.openai.com` and requires `OPENAI_API_KEY`. The integration runner loads that key from repository-root
`.env` first when available, then falls back to the inherited environment.

Run only unit checks:

```bash
uv run poe test-unit
```

Run only live integration checks:

```bash
uv run poe test-integration
```

## Verification Tasks

Run formatting and safe fixes:

```bash
uv run poe fmt
```

Run the full local check suite:

```bash
uv run poe check
```

Run the authenticated Socket manifest generation and scan:

```bash
uv run poe socket
```

`uv run poe check` includes:

- `fmt-check`
- `lint`
- `typecheck`
- `actionlint`
- Python unit tests with coverage
- live OpenAI integration tests
- coverage threshold validation
- pre-commit hooks

`uv run poe socket` installs the pinned Socket CLI from `package-lock.json`,
generates CycloneDX manifests through `socket manifest cdxgen`, and runs a
read-only Socket scan preflight to verify the manifests are accepted. It requires
`SOCKET_API_KEY`, `SOCKET_API_TOKEN`, or `SOCKET_CLI_API_TOKEN`. The Socket org
defaults to `donadio-solutions` and can be overridden with `SOCKET_ORG` or
`SOCKET_DEFAULT_ORG`.

The repository requires 100% line coverage and 100% branch coverage.

CI uses the same Poe entry points for the Python checks. The `unit` job
generates `coverage.xml` through `uv run poe check-unit` and uploads that report
to Codecov. The Codecov GitHub Action is pinned by commit SHA, the Codecov CLI
version is pinned explicitly, and uploads authenticate through GitHub OIDC.

The GitHub Actions workflow has separate `unit` and `integration` jobs that run
in parallel, and the `integration` job receives `OPENAI_API_KEY` from
`secrets.OPENAI_API_KEY`. A final `validate` job depends on both jobs and
succeeds only when both completed successfully.

Socket's GitHub App supplies the dependency-security checks as separate PR
checks:

- `Socket Security: Pull Request Alerts`
- `Socket Security: Project Report`

Those Socket App checks are required in the repository ruleset. They are not
duplicated as GitHub Actions jobs.

## actionlint

GitHub Actions workflows are linted through the `actionlint` Poe task. The pinned Python package is `actionlint-py`, and the workflow
runner label `blacksmith-2vcpu-ubuntu-2404` is declared in `.github/actionlint.yaml` so actionlint recognizes the custom runner.

## Dependency Security

Dependencies are pinned in `pyproject.toml` and `package.json`, and locked in
`uv.lock` and `package-lock.json`. Before adding a new package, run
`socket package score <ecosystem> <name>@<version> --json` and evaluate the
score, alerts, and transitive dependency findings. If Socket reports high-risk
findings, stop and explain the risk before changing dependencies.

## Documentation

Keep `README.md` brief. Put detailed usage, module behavior, verification, and development notes in topic files under `docs/`.

When module behavior changes, update the matching module document and the documentation index.
