# asr-prep Module

`asr-prep` is a local preparation command for reproducible ASR WER runs. It
segments supported direct-child audio files into deterministic 30-second WAV
chunks under `AUDIO_DIR/prep`.

It does not call an OpenAI-compatible endpoint. It uses `ffmpeg` from `PATH` to
write chunks and Mutagen to read source durations.

## Basic Usage

```bash
uv run openai-tests asr-prep ./audio
```

Use a custom overlap:

```bash
uv run openai-tests asr-prep ./audio --overlap 1.5
```

`--overlap` defaults to `3.0` seconds. It must be at least `0` and less than the
fixed `30.0` second segment duration.

## Inputs

`asr-prep` only scans files that are direct children of `AUDIO_DIR`. It does not
recurse into subdirectories. Supported extensions match the ASR transcription
content types used by `asr-simple` and `asr-wer`.

Source filenames and stems must be plain filenames. The command rejects stems
that would collide with later `asr-wer` output artifacts, such as `report.wav`,
`call.wav` plus `call_normalized.wav`, or duplicate stems like `call.wav` and
`call.mp3`.

## Outputs

The command creates `AUDIO_DIR/prep` and writes:

- `manifest.json`: machine-readable metadata used by `asr-wer --prep`.
- `report.txt`: human-readable source and chunk summary.
- `*.wav`: deterministic chunks named from source stem, chunk index, start
  milliseconds, and end milliseconds.

Example chunk name:

```text
call_0001_027000_057000.wav
```

This is chunk index `1` from `call.*`, spanning `27.000` through `57.000`
seconds.

`asr-prep` refuses to run when `AUDIO_DIR/prep` already exists and is non-empty.
Work is staged in a unique hidden temporary directory and moved into place only
after all chunks, the manifest, and the report are written.

## Manifest Contract

`manifest.json` records:

- tool name
- fixed segment duration
- overlap seconds
- source filenames, durations, and chunk counts
- chunk filename, source filename, chunk index, start seconds, end seconds, and
  duration seconds

Prepared `asr-wer` runs require this manifest so chunk transcripts can be
stitched back into one transcript per original audio file.

## Failure Modes

The command exits `2` for local configuration failures, including:

- missing or non-directory `AUDIO_DIR`
- no supported direct-child audio files
- invalid overlap
- missing `ffmpeg`
- unreadable or unknown audio duration
- non-empty `AUDIO_DIR/prep`
- source names that would produce ambiguous prepared outputs
