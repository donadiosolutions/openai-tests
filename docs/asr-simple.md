# asr-simple Module

`asr-simple` checks basic speech-recognition compatibility through two APIs:

- `POST /v1/chat/completions`
- `POST /v1/audio/transcriptions`

By default, it synthesizes a WAV file with `espeak-ng` saying:

```text
Alpha Bravo Charlie Delta Echo Foxtrot Golf Hotel India Juliet
```

It sends that audio to both endpoints and verifies that the returned transcript contains the expected words.

## Quickstart

```bash
uv run openai-tests asr-simple \
  --base-url https://api.openai.com \
  --model gpt-4o-audio-preview \
  --transcriptions-model gpt-4o-transcribe
```

Use an existing audio file instead of `espeak-ng`:

```bash
uv run openai-tests asr-simple \
  --audio-file ./speech.wav \
  --audio-format wav \
  --expected-transcript \
  "Alpha Bravo Charlie Delta Echo Foxtrot Golf Hotel India Juliet"
```

Inspect the full redacted HTTP exchanges:

```bash
uv run openai-tests asr-simple --verbose
```

## Audio Fixture

When `--audio-file` is omitted, the module creates a temporary file named
`asr-simple.wav` by running:

```bash
espeak-ng -v en-us -s 150 -w asr-simple.wav \
  "Alpha Bravo Charlie Delta Echo Foxtrot Golf Hotel India Juliet"
```

The temporary directory is removed after the run. Use `--audio-file` when
`espeak-ng` is unavailable or when testing a specific fixture.

Supported audio format names are:

- `mp3`
- `mp4`
- `mpeg`
- `mpga`
- `m4a`
- `wav`
- `webm`

The format is used for chat completions `input_audio.format` and for the
multipart file content type sent to transcriptions.

## Common Parameters

- `--base-url`: target API base URL.
- `--model`: chat-completions model.
- `--completions-model`: explicit chat-completions model override.
- `--transcriptions-model`: audio-transcriptions model.
- `--api-key`: explicit bearer token.
- `--timeout`: HTTP timeout in seconds.
- `--verbose`: print full redacted HTTP exchanges.

## Audio Parameters

- `--audio-file`: existing audio fixture path.
- `--audio-format`: format name for the fixture. The default is `wav`.
- `--expected-transcript`: transcript used for accuracy checks and default
  synthesis text.
- `--min-expected-words`: minimum expected words required in each returned
  transcript. The default is all words.
- `--espeak-voice`: voice passed to `espeak-ng`. The default is `en-us`.
- `--espeak-speed`: speed passed to `espeak-ng`. The default is `150`.

## Prompt Parameters

- `--system-prompt`: defaults to `You are a precise speech recognition assistant.`
- `--developer-prompt`: defaults to `Transcribe only the spoken English words from the audio.`
- `--user-prompt`: defaults to `Transcribe this audio exactly.`

For chat completions, the module sends a system message containing the system
and developer prompts, then a user message containing a text instruction plus
an `input_audio` content item.

## Chat Completions Parameters

The module exposes these chat-completions parameters:

- `--completions-audio-json`
- `--completions-frequency-penalty`
- `--completions-function-call`, `--completions-function-call-json`
- `--completions-functions-json`
- `--completions-logit-bias-json`
- `--completions-logprobs`, `--no-completions-logprobs`
- `--completions-max-completion-tokens`
- `--completions-max-tokens`
- `--completions-messages-json`
- `--completions-metadata-json`
- `--completions-modalities-json`
- `--completions-n`
- `--completions-parallel-tool-calls`, `--no-completions-parallel-tool-calls`
- `--completions-prediction-json`
- `--completions-presence-penalty`
- `--completions-prompt-cache-key`
- `--completions-prompt-cache-retention`
- `--completions-reasoning-effort`
- `--completions-response-format-json`
- `--completions-safety-identifier`
- `--completions-seed`
- `--completions-service-tier`
- `--completions-stop`, `--completions-stop-json`
- `--completions-store`, `--no-completions-store`
- `--completions-stream`, `--no-completions-stream`
- `--completions-stream-options-json`
- `--completions-temperature`
- `--completions-tool-choice`, `--completions-tool-choice-json`
- `--completions-tools-json`
- `--completions-top-logprobs`
- `--completions-top-p`
- `--completions-user`
- `--completions-web-search-options-json`

If `--completions-messages-json` is provided, it fully replaces the default messages. In that case, the caller is responsible for
including audio in the request payload if desired.

## Transcriptions Parameters

The module exposes these transcriptions parameters:

- `--transcriptions-chunking-strategy`, `--transcriptions-chunking-strategy-json`
- `--transcriptions-include`, `--transcriptions-include-json`
- `--transcriptions-known-speaker-names`, `--transcriptions-known-speaker-names-json`
- `--transcriptions-known-speaker-references`, `--transcriptions-known-speaker-references-json`
- `--transcriptions-language`
- `--transcriptions-prompt`
- `--transcriptions-response-format`
- `--transcriptions-stream`, `--no-transcriptions-stream`
- `--transcriptions-temperature`
- `--transcriptions-timestamp-granularities`
- `--transcriptions-timestamp-granularities-json`

List-style arguments such as `--transcriptions-include` and `--transcriptions-timestamp-granularities` can be repeated. Their JSON
counterparts must decode to arrays and are appended to repeated CLI values.

## Response Extraction

For non-streaming chat completions, text is extracted from the same chat-completions shapes used by `text-simple`. For streaming chat
completions, the module parses SSE `data:` events and concatenates `choices[].delta.content`.

For non-streaming transcriptions, text is extracted from:

- top-level `text`
- `segments[].text`
- raw response body text for plain text response formats

For streaming transcriptions, the module parses SSE `data:` events and prefers a final `text` value when present. Otherwise, it
concatenates `delta` values.

## Success and Failure Checks

An endpoint fails when:

- no HTTP response is received
- the HTTP status is outside the 2xx range
- no transcript text can be extracted
- the response format is incompatible with the requested mode
- the transcript does not contain enough expected words

Accuracy is case-insensitive and punctuation-insensitive. For the default transcript, every one of the ten words must appear unless
`--min-expected-words` lowers the requirement.

## Warnings

The chat-completions response is inspected for many top-level request parameters, including model, sampling parameters, tool settings,
stream settings, metadata, and service-tier settings. If the response reports a different value for a field that was sent, the module
prints a warning.

The module also warns if chat completions returns a tool call and no matching tool was available in the request.

The transcriptions response is inspected for selected transcriptions parameters, including model, language, response format, stream,
temperature, timestamp granularities, chunking strategy, and diarization speaker hints.

## Examples

Require only eight of the ten default words:

```bash
uv run openai-tests asr-simple --min-expected-words 8
```

Ask the transcriptions endpoint for JSON log probabilities:

```bash
uv run openai-tests asr-simple \
  --transcriptions-response-format json \
  --transcriptions-include logprobs
```

Ask for word timestamps with `whisper-1`:

```bash
uv run openai-tests asr-simple \
  --transcriptions-model whisper-1 \
  --transcriptions-response-format verbose_json \
  --transcriptions-timestamp-granularities word
```

Pass custom chat-completions metadata:

```bash
uv run openai-tests asr-simple \
  --completions-metadata-json '{"suite":"asr-simple"}'
```
