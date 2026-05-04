# list-models Module

`list-models` checks the OpenAI-compatible models listing endpoint:

- `GET /v1/models`

The module sends one request, prints every returned model ID, and passes only when
the JSON response conforms to the official list-models response shape.

## Quickstart

```bash
uv run openai-tests list-models \
  --base-url https://api.openai.com
```

The command reads the API key from `OPENAI_API_KEY` or
`OPENAI_TESTS_API_KEY` unless `--api-key` is provided.

## Arguments

`GET /v1/models` has no endpoint-specific request parameters in the official
OpenAI spec. The module therefore exposes only the shared execution controls:

- `--base-url`: base URL for the OpenAI-compatible API. Defaults to
  `OPENAI_BASE_URL`, then `OPENAI_TESTS_BASE_URL`, then
  `https://api.openai.com`.
- `--api-key`: bearer token for the request. Defaults to `OPENAI_API_KEY`, then
  `OPENAI_TESTS_API_KEY`.
- `--timeout`: HTTP timeout in seconds. Defaults to `30`.
- `--verbose`: print the redacted HTTP request and full response body.

Because the endpoint has no optional query or body fields, no nullable
endpoint-specific parameters are sent.

## Pass Criteria

The module reports `PASSED` when the HTTP request succeeds and the response is a
JSON object with:

- `object` equal to `list`
- `data` as an array
- every `data` item as a model object with a non-empty string `id`, `object`
  equal to `model`, integer `created`, and non-empty string `owned_by`

An empty `data` array is schema-valid and passes, although the rendered model
list will show `(none)`.

## Failure Examples

The module reports `FAILED` when:

- the endpoint returns a non-2xx HTTP status
- the response is not JSON
- `object` is not `list`
- `data` is missing or not an array
- any listed model does not match the model-object schema
