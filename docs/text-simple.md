# text-simple Module

`text-simple` asks one short question through two text-generation APIs:

- `POST /v1/chat/completions`
- `POST /v1/responses`

The default question is `What is the capital of France?`. The module expects each endpoint to return non-empty text.

## Quickstart

```bash
uv run openai-tests text-simple \
  --base-url https://api.openai.com \
  --model gpt-4.1-mini
```

Use a separate model only for responses:

```bash
uv run openai-tests text-simple \
  --model gpt-4.1-mini \
  --responses-model gpt-4.1
```

Inspect the full redacted HTTP exchanges:

```bash
uv run openai-tests text-simple --verbose
```

## Request Construction

The chat-completions request always includes:

- `model`
- `messages`

The default messages are:

- a system message containing the system prompt and developer prompt separated by a blank line
- a user message containing the user prompt

The responses request includes:

- `model`
- `input`

When `--responses-input-json` is omitted, the default responses input is:

```json
[
  {"role": "system", "content": "You are a concise assistant."},
  {"role": "developer", "content": "Answer the user's question in one short sentence."},
  {"role": "user", "content": "What is the capital of France?"}
]
```

Optional responses parameters are included only when set.

## Common Parameters

- `--base-url`: target API base URL.
- `--model`: model used by both endpoints unless overridden.
- `--responses-model`: model used only by `/v1/responses`.
- `--api-key`: explicit bearer token.
- `--timeout`: HTTP timeout in seconds.
- `--verbose`: print full redacted HTTP exchanges.

## Prompt Parameters

- `--system-prompt`: defaults to `You are a concise assistant.`
- `--developer-prompt`: defaults to `Answer the user's question in one short sentence.`
- `--user-prompt`: defaults to `What is the capital of France?`

## Responses API Parameters

The module exposes these responses parameters:

- `--responses-background`, `--no-responses-background`
- `--responses-context-management-json`
- `--responses-conversation`, `--responses-conversation-json`
- `--responses-include`, `--responses-include-json`
- `--responses-input-json`
- `--responses-instructions`, `--responses-instructions-json`
- `--responses-max-output-tokens`
- `--responses-max-tool-calls`
- `--responses-metadata-json`
- `--responses-parallel-tool-calls`, `--no-responses-parallel-tool-calls`
- `--responses-previous-response-id`
- `--responses-prompt-id`
- `--responses-prompt-version`
- `--responses-prompt-variables-json`
- `--responses-prompt-cache-key`
- `--responses-prompt-cache-retention`
- `--responses-reasoning-json`
- `--responses-reasoning-effort`
- `--responses-reasoning-generate-summary`
- `--responses-reasoning-summary`
- `--responses-safety-identifier`
- `--responses-service-tier`
- `--responses-store`, `--no-responses-store`
- `--responses-stream`, `--no-responses-stream`
- `--responses-stream-options-json`
- `--responses-include-obfuscation`, `--no-responses-include-obfuscation`
- `--responses-temperature`
- `--responses-text-json`
- `--responses-text-format-json`
- `--responses-text-verbosity`
- `--responses-tool-choice`, `--responses-tool-choice-json`
- `--responses-tools-json`
- `--responses-top-logprobs`
- `--responses-top-p`
- `--responses-truncation`
- `--responses-user`

JSON arguments must decode to the expected type. For example, `--responses-metadata-json` must decode to an object, while
`--responses-tools-json` must decode to an array.

## Response Extraction

For chat completions, text is extracted from:

- `choices[0].message.content`
- structured message content items with `text`, `content`, or `refusal`
- `choices[0].text` for legacy-compatible shapes

For responses, text is extracted from:

- top-level `output_text`
- message items under `output`
- text values in compatible output item shapes

## Success and Failure Checks

An endpoint fails when:

- no HTTP response is received
- the HTTP status is outside the 2xx range
- no response text can be extracted

The module reports partial success when the endpoint returns text but warnings are produced.

## Warnings

The responses endpoint is inspected for selected argument mismatches:

- `tool_choice`
- `tools`
- `parallel_tool_calls`

If the extracted response text looks like JSON, the module also checks whether it resembles a tool-call payload. A returned tool call
produces a warning when no matching tool was sent in the request.

## Examples

Add metadata and inspect verbose output:

```bash
uv run openai-tests text-simple \
  --responses-metadata-json '{"suite":"text-simple"}' \
  --verbose
```

Ask a different question:

```bash
uv run openai-tests text-simple \
  --user-prompt "Name one primary color." \
  --developer-prompt "Answer with one word."
```

Use a prompt object:

```bash
uv run openai-tests text-simple \
  --responses-prompt-id pmpt_123 \
  --responses-prompt-version 1 \
  --responses-prompt-variables-json '{"name":"world"}'
```
