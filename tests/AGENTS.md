# Test Suite Guidance

These instructions apply to all work under `tests/` and to any task that changes behavior in `src/openai_tests/`.

## Invariants

- Tests must preserve 100% line coverage and 100% branch coverage.
- Never weaken verification to obtain a passing result.
- Never disable, delete, skip, xfail, dilute, or replace a meaningful test with a weaker one unless explicitly instructed by a human.
- Never change fixtures, snapshots, golden files, expected outputs, CI checks, or test data merely to make failures disappear unless explicitly instructed by a human.
- Always mock external systems in unit tests.
- For complex interaction testing, prefer container-backed integration tests using rootless `podman`, `buildah`, and `skopeo`, plus `Containerfile` and `compose.yaml`.

## Workflow

- Prefer observable outcomes over implementation-detail assertions.
- Keep unit tests isolated and fast.
- Keep integration tests under `tests/integration/` and mark them with `@pytest.mark.integration`.
- Run `uv run poe test` and `uv run poe coverage` after test changes.
