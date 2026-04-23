# openai-tests

Operational runbook for coding agents working in this repository.

## GitHub Naming Overrides

These instructions override conflicting defaults in bundled skills such as `github:yeet`.

- When Codex creates a git branch for work that may be pushed to GitHub, do not prepend `codex/` to the branch name.
- Use a plain descriptive branch name instead, for example `fix-login-timeout`, not `codex/fix-login-timeout`.
- When Codex opens a pull request, do not prepend `[codex]` to the PR title.
- Use a plain descriptive PR title that reflects the change itself.
- Treat these naming rules as higher priority than skill-level defaults when they conflict.

## Safety

- Every time you import a Python package or add a package to a `requirements.txt` or
  `pyproject.toml`, use the safety-mcp to check if the version you have chosen is secure and is
  the latest version of the package. Make sure you always use the `latest_secure_version`
  returned by safety-mcp for any package.
- If a package already exists in the codebase and a user asks you to check it for vulnerabilities,
  use safety-mcp, evaluate whether there are any secure versions in the same major version, and
  acknowledge those options. Also report the latest secure version.

## JavaScript REPL (Node)

- Use `js_repl` for Node-backed JavaScript with top-level await in a persistent kernel.
- `js_repl` is a freeform/custom tool. Direct `js_repl` calls must send raw JavaScript tool input, optionally with a first-line `// codex-js-repl: timeout_ms=15000`. Do not wrap code in JSON, quotes, or markdown fences.
- Helpers: `codex.cwd`, `codex.homeDir`, `codex.tmpDir`, `codex.tool(name, args?)`, and `codex.emitImage(imageLike)`.
- `codex.tool` executes a normal tool call and resolves to the raw tool output object. Use it for shell and non-shell tools alike. Nested tool outputs stay inside JavaScript unless you emit them explicitly.
- `codex.emitImage(...)` adds one image to the outer `js_repl` function output each time you call
  it. It accepts a data URL, a single `input_image` item, an object like `{ bytes, mimeType }`,
  or a raw tool response object with exactly one image and no text.
- `codex.tool(...)` and `codex.emitImage(...)` keep stable helper identities across cells. Saved
  references and persisted objects can reuse them in later cells, but async callbacks that fire
  after a cell finishes still fail because no exec is active.
- Request full-resolution image processing with `detail: "original"` only when the `view_image` tool schema includes a `detail` argument. The same availability applies to `codex.emitImage(...)`.
- Raw MCP image blocks can request the same behavior by returning `_meta: { "codex/imageDetail": "original" }` on the image content item.
- When encoding an image to send with `codex.emitImage(...)` or `view_image`, prefer JPEG at about 85 quality when lossy compression is acceptable; use PNG when transparency or lossless detail matters.
- Top-level bindings persist across cells. If a cell throws, prior bindings remain available and bindings that finished initializing before the throw often remain usable in later cells.
- Top-level static import declarations are currently unsupported in `js_repl`; use dynamic imports instead.
- Avoid direct access to `process.stdout`, `process.stderr`, or `process.stdin`; use `console.log`, `codex.tool(...)`, and `codex.emitImage(...)`.

## Invariants

- keep an `AGENTS.md` up to date whenever repository changes invalidate or materially change its guidance
- never modify this `Invariants` section unless explicitly instructed by a human
- obey the `AGENTS.md` nearest to the file being modified
- never weaken verification to obtain a passing result: this includes disabling, deleting, skipping, xfail-ing, or diluting tests, coverage checks, or CI quality gates
- never change coverage thresholds, test selection rules, fixtures, snapshots, golden files, expected outputs, CI checks, or test data merely to make failures disappear unless a human explicitly approves that exact change
- never add `# noqa`, `# type: ignore`, `pragma: no cover`, or similar escape hatches unless explicitly instructed by a human
- add extra `AGENTS.md` files whenever there are substantial extra instructions pertaining to a given subtree

## Workflow

- Use `uv` for Python dependency and environment management. Do not use `pip`, `python -m venv`, Poetry, or Pipenv in this repository.
- Let `pyenv` manage the interpreter version through `.python-version`.
- Prefer `uv run poe <task>` for routine workflows instead of ad-hoc command sequences.
- `uv run poe safety` requires `SAFETY_API_KEY` to be present in the environment.
- Future endpoint test modules belong under `src/openai_tests/test_modules/`.
- Register future endpoint test modules in `src/openai_tests/registry.py` before wiring them into the CLI.
- Read the relevant implementation and tests before changing behavior.
- For behavior changes or bug fixes, update tests first when practical and verify they fail for the expected reason before changing implementation.

## Required verification

Run these commands before treating a code-changing task as complete:

```bash
uv run poe fmt
uv run poe check
uv run poe safety
```

## Completion

A task is not complete unless:

- the intended behavior is covered by tests when applicable
- 100% line and branch coverage remain intact
- all required checks pass
- the final response states what changed, what verification ran, and any unresolved assumptions or concerns
