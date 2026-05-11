# asr-wer Module

`asr-wer` batch-transcribes an audio directory and reports word error rate
(WER), real-time factor (RTFx), duration, and request timing. It has two modes:

- `ground`: create reference transcripts under `AUDIO_DIR/ground`.
- `eval`: transcribe into a model-specific output directory and score against
  `AUDIO_DIR/ground`.

The module can call either `/v1/audio/transcriptions` or
`/v1/chat/completions` with audio input.

## Basic Usage

Create ground truth:

```bash
uv run openai-tests asr-wer ground ./audio \
  --endpoint transcriptions \
  --transcriptions-model gpt-4o-transcribe
```

Evaluate a model:

```bash
uv run openai-tests asr-wer eval ./audio \
  --endpoint transcriptions \
  --transcriptions-model gpt-4o-transcribe
```

Use chat completions instead:

```bash
uv run openai-tests asr-wer eval ./audio \
  --endpoint completions \
  --completions-model gpt-4o-audio-preview
```

## Audio Discovery

Without `--prep`, the command transcribes supported direct-child audio files in
`AUDIO_DIR`. It writes exact transcripts as `<stem>.txt` and normalized
transcripts as `<stem>_normalized.txt`.

`eval` requires matching normalized ground files under `AUDIO_DIR/ground`.

## Prepared Runs

Use `asr-prep` first when long audio should be split deterministically before
calling the provider:

```bash
uv run openai-tests asr-prep ./audio --overlap 3.0
uv run openai-tests asr-wer ground ./audio --prep
uv run openai-tests asr-wer eval ./audio --prep
```

With `--prep`, `asr-wer` reads chunk files and source metadata from
`AUDIO_DIR/prep/manifest.json`, transcribes chunks as requests, and writes one
combined transcript per original source file. WER rows use the original source
filename, not the chunk filename.

Prepared output still goes to root-level folders:

- `AUDIO_DIR/ground` for ground transcripts
- `AUDIO_DIR/<model>_<epoch>` for eval transcripts

Each prepared output folder also keeps per-chunk exact transcripts under
`chunks/` for auditability.

## Stitching and Overlap

Prepared runs combine chunk transcripts in chunk-index order. Exact transcripts
are concatenated. Normalized transcripts are stitched with exact token
de-duplication across overlaps: the command removes the longest exact match
between the previous chunk suffix and current chunk prefix, bounded by the
configured overlap. It does not fuzzy-delete non-matching text.

Use `--overlap` with `--prep` when you want to assert the expected manifest
overlap:

```bash
uv run openai-tests asr-wer eval ./audio --prep --overlap 3.0
```

The value is validated against `prep/manifest.json`.

## Temperature and Concurrency

For prepared runs, the selected endpoint temperature defaults to `0.0` when the
user did not provide an endpoint-specific temperature:

- `--transcriptions-temperature` for `/v1/audio/transcriptions`
- `--completions-temperature` for `/v1/chat/completions`

User-provided temperatures always win. Non-prepared runs leave temperature
unset unless the user provides it, so providers keep their default behavior.

`--batch` controls maximum concurrent in-flight requests and must be at least
`1`. Prepared mode does not override it. Use `--batch 1` only when you want the
strictest repeatability over throughput.

## Reports

Reports include:

- selected endpoint
- selected model
- temperature, or `provider_default` when unset
- whether prepared audio was used
- prep folder, overlap, and segment duration for prepared runs
- one row per original source file
- chunk count for prepared rows
- WER, duration, elapsed seconds, and RTFx

If any chunk for a prepared source fails, that source is marked failed and is
not scored as a successful WER row.
