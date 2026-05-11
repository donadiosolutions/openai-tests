# openai-tests

Operational runbook for coding agents working in this repository.

Read also @AGENTS.local.md when one is available. This file should be gitignored and is user-specific.

## Invariants

- keep an `AGENTS.md` up to date whenever repository changes invalidate or materially change its guidance
- never modify this `Invariants` section unless explicitly instructed by a human
- never include internal URLs, internal hostnames, or private endpoints in PRs, review comments, commit messages, code, docs, or examples; use public placeholders such as `https://example.com` instead
- obey the `AGENTS.md` nearest to the file being modified
- always write or update tests before writing application code for any behavior change, bug fix, feature, or executable refactor
- verify that new or changed tests fail for the expected reason before changing application code
- never weaken verification to obtain a passing result: this includes disabling, deleting, skipping, xfail-ing, or diluting tests, coverage checks, or CI quality gates.
- never change coverage thresholds, test selection rules, fixtures, snapshots, golden files, expected outputs, CI checks, or test data merely to make failures disappear unless a human explicitly approves that exact change
- never add `# noqa`, `# type: ignore`, `pragma: no cover`, or similar escape hatches unless explicitly instructed by a human
- every dependency must be pinned to an exact version and protected by a cryptographic integrity mechanism such as a lockfile
  hash, checksum, or image digest; this includes CI actions, tools, scripts, images, and other artifacts that are not already
  provided by the runner image. If the workflow downloads it, pin and verify it.
- before adding any new dependency, run `socket package score <ecosystem> <name>@<version> --json` and evaluate the score, alerts,
  and transitive dependency findings before deciding whether to use it
- add extra `AGENTS.md` files whenever there are substantial extra instructions pertaining to a given subtree

## Workflow

- Use `uv` for Python dependency and environment management. Do not use `pip`, `python -m venv`, Poetry, or Pipenv in this repository.
- Let `pyenv` manage the interpreter version through `.python-version`.
- Prefer `uv run poe <task>` for routine workflows instead of ad-hoc command sequences.
- Publish to PyPI through `.github/workflows/publish-pypi.yml`; it is designed for trusted publishing from a non-draft GitHub Release.
- Future endpoint test modules belong under `src/openai_tests/test_modules/`.
- Register future endpoint test modules in `src/openai_tests/registry.py` before wiring them into the CLI.
- Read the relevant implementation and tests before changing behavior.
- For behavior changes or bug fixes, update tests first when practical and verify they fail for the expected reason before changing implementation.
- Always write docstrings for every function, method, class, and module, and for any non-trivial code block, especially if it is
  not immediately clear what it does or why it is necessary.
- Keep the user documentation both in `docs` and in `README.md` up to date with any changes to behavior, features, or usage.

## Dependency Security

- Before adding any new dependency, check it with Socket for known vulnerabilities and supply-chain risk.
- Use `socket package score <ecosystem> <name>@<version> --json` before deciding whether to add the dependency.
- If Socket reports high-risk findings, known malware, suspicious behavior, or relevant vulnerabilities, stop and explain the risk
  instead of installing the dependency silently.
- If Socket is unavailable, do not proceed as though the package was checked. Tell the user the Socket check could not be completed
  and ask whether to continue, use another package, or defer the install.

## Required verification

Run these commands before treating a code-changing task as complete:

```bash
uv run poe fmt
uv run poe check
uv run poe socket
```

For behavior changes, bug fixes, features, or executable refactors, also run the relevant tests before and after implementation:

- before implementation: prove the new or changed test fails for the expected reason
- after implementation: prove the same test passes
- before completion: prove the full required verification passes

## What done means

A task is not complete unless:

- tests were written or updated before application code when the change affected executable behavior
- new or changed tests were observed failing for the expected reason before implementation
- the intended behavior is covered by tests when applicable
- new or changed tests pass after implementation
- 100% line and branch coverage remain intact
- all required checks pass
- the final response states what changed, which tests failed before implementation, what verification passed afterward, and any unresolved assumptions or concerns
- documentation was updated
