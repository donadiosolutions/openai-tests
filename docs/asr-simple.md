# asr-simple Module

`asr-simple` checks basic speech-recognition compatibility through two APIs:

- `POST /v1/chat/completions`
- `POST /v1/audio/transcriptions`

By default, it sends two checked-in MP3 fixtures through both APIs:

```text
1. Alpha through Zulu in NATO spelling words
2. The quick brown fox jumps over the lazy dog
```

It sends both files to both endpoints, verifies that enough expected words are
present in each returned transcript, and prints a simple word error rate (WER)
counter for each endpoint result.

## Quickstart

```bash
uv run openai-tests asr-simple \
  --base-url https://api.openai.com \
  --model gpt-4o-audio-preview
```

If the transcriptions endpoint uses a different model than chat completions,
pass `--transcriptions-model` explicitly.

Use an existing audio file:

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

When neither `--audio-file` nor `--expected-transcript` is provided, the module
loads these bundled MP3 files from the repository:

- `asr_default_nato.mp3`
- `asr_default_pangram.mp3`

The chat-completions request base64-encodes each file on the fly, and the
transcriptions request sends the same binary file as multipart form data.
Nothing is transcoded at runtime.

When `--expected-transcript` is provided without `--audio-file`, the module
falls back to `espeak-ng` and creates a temporary WAV file named
`asr-simple.wav` by running:

```bash
espeak-ng -v en-us -s 150 -w asr-simple.wav \
  "Your custom transcript text"
```

The temporary directory is removed after the run.

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
- `--transcriptions-model`: audio-transcriptions model. When omitted, it
  inherits the resolved `--model` value unless a transcriptions-specific
  environment variable is set.
- `--api-key`: explicit bearer token.
- `--timeout`: HTTP timeout in seconds.
- `--verbose`: print full redacted HTTP exchanges.

## Audio Parameters

- `--audio-file`: existing audio fixture path.
- `--audio-format`: format name for `--audio-file`. If omitted, the CLI uses
  the file extension.
- `--expected-transcript`: transcript used for accuracy checks. When provided
  without `--audio-file`, it is also the text that `espeak-ng` synthesizes.
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
- `--completions-repetition-penalty`
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

Penalty controls differ by scope. `--completions-frequency-penalty` penalizes
tokens based on their frequency in generated text only. The provider-compatible
`--completions-repetition-penalty` penalizes tokens based on whether they
appear in the prompt and generated text.

## Transcriptions Parameters

The module exposes these transcriptions parameters:

- `--transcriptions-chunking-strategy`, `--transcriptions-chunking-strategy-json`
- `--transcriptions-include`, `--transcriptions-include-json`
- `--transcriptions-known-speaker-names`, `--transcriptions-known-speaker-names-json`
- `--transcriptions-known-speaker-references`, `--transcriptions-known-speaker-references-json`
- `--transcriptions-language`
- `--transcriptions-prompt`
- `--transcriptions-frequency-penalty`
- `--transcriptions-repetition-penalty`
- `--transcriptions-response-format`
- `--transcriptions-stream`, `--no-transcriptions-stream`
- `--transcriptions-temperature`
- `--transcriptions-timestamp-granularities`
- `--transcriptions-timestamp-granularities-json`

List-style arguments such as `--transcriptions-include` and `--transcriptions-timestamp-granularities` can be repeated. Their JSON
counterparts must decode to arrays and are appended to repeated CLI values.

Penalty controls for transcriptions are provider-compatible passthroughs for
OpenAI-compatible servers such as vLLM. `--transcriptions-frequency-penalty`
penalizes tokens based on their frequency in generated text only.
`--transcriptions-repetition-penalty` penalizes tokens based on whether they
appear in the prompt and generated text.

## Response Extraction

For non-streaming chat completions, text is extracted from the same chat-completions shapes used by `text-simple`. For streaming chat
completions, the module parses SSE `data:` events and concatenates `choices[].delta.content`.

For Qwen ASR model families, the module also strips known wrapper prefixes such
as `language English<asr_text>` from chat-completions output before matching
and WER calculation.

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

Accuracy is case-insensitive and punctuation-insensitive. By default, each
bundled sample requires all of its expected words unless `--min-expected-words`
lowers the requirement.

The CLI also prints a WER counter for each endpoint result as
`WER: <percent> (<errors>/<reference words>)`.

The default acceptance rule allows the endpoint to pass when either:

- the expected-word threshold is met
- the WER is below `15%`

Common NATO-style spelling variants are normalized before both matching and WER
calculation. For example, variants such as `viktor`, `whisky`, `charly`,
`romeu`, `uniforme`, `yanke`, and `zooloo` are treated as their canonical
forms.

## Warnings

The chat-completions response is inspected for many top-level request parameters, including model, sampling parameters, tool settings,
stream settings, metadata, and service-tier settings. If the response reports a different value for a field that was sent, the module
prints a warning.

For `model`, a returned value may append a provider-specific suffix to the
requested model alias without producing a warning. For example,
`gpt-4o-audio-preview-2025-06-03` is accepted when `gpt-4o-audio-preview`
was sent.

The module also warns if chat completions returns a tool call and no matching tool was available in the request.

The transcriptions response is inspected for selected transcriptions parameters, including model, language, response format, stream,
temperature, frequency penalty, repetition penalty, timestamp granularities, chunking strategy, and diarization speaker hints.

## Examples

Require only eight expected words per sample:

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

Synthesize custom text with `espeak-ng` instead of using the bundled samples:

```bash
uv run openai-tests asr-simple \
  --expected-transcript "Please transcribe this sentence exactly."
```
