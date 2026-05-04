# CLI Usage

## Top-Level Commands

Show global help:

```bash
uv run openai-tests --help
```

List registered modules:

```bash
uv run openai-tests modules
```

Run a module:

```bash
uv run openai-tests text-simple
uv run openai-tests asr-simple
uv run openai-tests list-models
```

Each module exposes its own help:

```bash
uv run openai-tests text-simple --help
uv run openai-tests asr-simple --help
uv run openai-tests list-models --help
```

## Common Arguments

All implemented modules support:

- `--base-url`: target OpenAI-compatible API base URL.
- `--api-key`: explicit bearer token.
- `--timeout`: HTTP timeout in seconds. The default is `30`.
- `--verbose`: print full redacted HTTP requests and responses.

`text-simple` and `asr-simple` also expose prompt text controls:

- `--system-prompt`
- `--developer-prompt`
- `--user-prompt`

The module combines system and developer prompts for chat completions and sends the user prompt as the user message. For responses or
audio-specific calls, the same text is adapted to the endpoint shape.

## JSON Arguments

Arguments ending in `-json` are parsed as JSON values. They can be passed inline:

```bash
uv run openai-tests text-simple \
  --responses-metadata-json '{"suite":"smoke"}'
```

They can also be loaded from a file by prefixing the path with `@`:

```bash
uv run openai-tests text-simple \
  --responses-metadata-json @metadata.json
```

When a parameter has both a string and JSON form, provide only one. For example, `--responses-tool-choice auto` and
`--responses-tool-choice-json '{"type":"function"}'` are mutually exclusive.

## Optional Parameters

Optional endpoint parameters are represented internally as `None` until the user provides a value. Before sending requests, `None`
values are pruned from JSON payloads or omitted from multipart form fields. This keeps requests close to the API defaults while still
making every exposed knob available from the CLI.

Boolean flags use argparse's paired form:

```bash
--responses-stream
--no-responses-stream
--completions-store
--no-completions-store
```

If neither form is provided, the value remains unset and is omitted.

## Output

Text and ASR endpoint checks print:

- endpoint name
- status label
- question or expected transcript
- response text or transcript
- error message, if any
- warnings, if any

`list-models` prints the endpoint name, status label, returned model IDs, and
error message when the response does not conform to the schema.

Status labels are:

- `PASSED`: the endpoint returned usable text and no warnings were produced.
- `PARTIAL SUCCESS`: the endpoint returned usable text but warnings were produced.
- `FAILED`: an HTTP, response-format, empty-text, or accuracy check failed.

With `--verbose`, the CLI also prints request headers, request body, response status, response headers, and raw response body. The
`Authorization` header is redacted as `Bearer ***REDACTED***`.

## Exit Codes

- `0`: all checked endpoints passed.
- `1`: at least one endpoint failed or at least one endpoint produced a partial-success warning.
- `2`: configuration failed before requests were sent, such as invalid JSON or mutually exclusive arguments.
