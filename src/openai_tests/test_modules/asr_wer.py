"""Batch ASR transcription and WER reporting module."""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import mutagen
from tqdm import tqdm

from ..core import EndpointTestModule
from . import asr_simple
from .whisper_english_normalizer import EnglishTextNormalizer

SERVICE_TIER_CHOICES = ("auto", "default", "flex", "priority")
TRANSCRIPT_NORMALIZER = EnglishTextNormalizer()


@dataclass(frozen=True, slots=True)
class AudioInput:
  """Supported direct-child audio file selected for a batch run."""

  path: Path
  stem: str
  format: str


@dataclass(frozen=True, slots=True)
class FileResult:
  """Per-file transcript, timing, scoring, and output metadata."""

  audio: AudioInput
  status: str
  transcript: str
  normalized_transcript: str
  output_path: Path
  normalized_output_path: Path
  elapsed_seconds: float | None
  duration_seconds: float
  rtfx: float | None
  exact_word_count: int
  normalized_word_count: int
  reference_word_count: int | None = None
  wer_errors: int | None = None
  wer_reference_words: int | None = None
  wer: float | None = None
  error_message: str | None = None


def configure_parser(parser: argparse.ArgumentParser) -> None:
  """Register the asr-wer command-line arguments."""

  parser.add_argument("mode", choices=("ground", "eval"), help="Create ground transcripts or evaluate against them.")
  parser.add_argument("audio_dir", help="Directory containing supported audio files.")
  parser.add_argument(
    "--endpoint",
    choices=("transcriptions", "completions"),
    default="transcriptions",
    help="Endpoint to use for transcription. Defaults to /v1/audio/transcriptions.",
  )
  parser.add_argument("--batch", type=int, default=1, help="Maximum concurrent in-flight requests. Defaults to 1.")
  parser.add_argument(
    "--service-tier",
    choices=SERVICE_TIER_CHOICES,
    help=(
      "Optional provider processing tier. Sent as JSON for completions and multipart pass-through for transcriptions."
    ),
  )
  parser.add_argument(
    "--prompt",
    help="Domain context prompt. Maps to transcriptions prompt or the default completions instruction.",
  )
  asr_simple.add_connection_arguments(parser)
  asr_simple.add_prompt_arguments(parser, system_default=None, developer_default=None, user_default=None)
  asr_simple.add_completions_arguments(parser)
  asr_simple.add_transcriptions_arguments(parser)


def run(args: argparse.Namespace) -> int:
  """Run batch ground generation or evaluation and return a CLI exit code."""

  try:
    validate_args(args)
    audio_dir = Path(args.audio_dir)
    audio_files = discover_audio_files(audio_dir)
    output_dir = resolve_output_dir(args, audio_dir)
    if args.mode == "eval":
      validate_eval_ground(audio_dir, audio_files)
    base_url = asr_simple.resolve_base_url(args.base_url)
    api_key = asr_simple.resolve_api_key(args.api_key)
    model = resolve_endpoint_model(args)
    output_dir = create_output_dir(args, output_dir)
  except ValueError as exc:
    print(f"Configuration error: {exc}", file=sys.stderr)
    return 2

  started_at = time.perf_counter()
  results = process_audio_files(
    args=args,
    audio_files=audio_files,
    output_dir=output_dir,
    base_url=base_url,
    api_key=api_key,
  )
  wall_elapsed_seconds = max(time.perf_counter() - started_at, 0.0)
  write_report(
    args=args,
    model=model,
    output_dir=output_dir,
    results=results,
    wall_elapsed_seconds=wall_elapsed_seconds,
  )
  print_final_stats(results, mode=args.mode, wall_elapsed_seconds=wall_elapsed_seconds)
  return 0 if all(result.status in {"transcribed", "skipped"} for result in results) else 1


def validate_args(args: argparse.Namespace) -> None:
  """Fail fast for argument combinations that cannot produce valid WER rows."""

  if args.batch < 1:
    raise ValueError("batch must be at least 1")
  if args.prompt and args.endpoint == "transcriptions" and args.transcriptions_prompt:
    raise ValueError("prompt cannot be provided with transcriptions-prompt")
  if args.endpoint == "transcriptions" and has_explicit_completions_prompt_override(args):
    raise ValueError("completions prompt flags cannot be used with transcriptions; use prompt or transcriptions-prompt")
  if args.endpoint == "transcriptions" and args.transcriptions_response_format in {"srt", "vtt"}:
    raise ValueError(
      "transcriptions-response-format must be transcript-only for asr-wer; srt and vtt include timestamps"
    )
  if args.prompt and args.endpoint == "completions" and has_explicit_completions_prompt_override(args):
    raise ValueError("prompt cannot be provided with completions prompt overrides")
  if args.endpoint == "completions" and args.completions_messages_json is not None:
    raise ValueError("completions-messages-json cannot be used with batch audio")
  validate_endpoint_request_config(args)


def validate_endpoint_request_config(args: argparse.Namespace) -> None:
  """Parse selected endpoint options once before any network requests are sent."""

  try:
    if args.endpoint == "completions":
      request_args = build_completions_request_args(args)
      config = asr_simple.build_completions_request_config(request_args, b"", "wav")
      validate_completions_response_format_for_wer(config.get("response_format"))
      return
    request_args = build_transcriptions_request_args(args)
    asr_simple.build_transcriptions_request_config(request_args)
  except OSError as exc:
    raise ValueError(f"Unable to read JSON option: {exc}") from exc


def validate_completions_response_format_for_wer(response_format: object) -> None:
  """Reject structured chat response formats that wrap the transcript in JSON."""

  if response_format is None:
    return
  if not isinstance(response_format, dict):
    raise ValueError("completions-response-format-json must request plain text for asr-wer")
  typed_response_format = cast("dict[str, Any]", response_format)
  response_type = typed_response_format.get("type")
  if response_type is not None and response_type != "text":
    raise ValueError("completions-response-format-json must request plain text for asr-wer")


def has_explicit_completions_prompt_override(args: argparse.Namespace) -> bool:
  """Return whether completions options replace the default audio prompt."""

  return any(
    getattr(args, name) is not None
    for name in ("system_prompt", "developer_prompt", "user_prompt", "completions_messages_json")
  )


def discover_audio_files(audio_dir: Path) -> list[AudioInput]:
  """Return sorted supported audio files and reject ambiguous input sets."""

  if not audio_dir.exists():
    raise ValueError(f"Audio directory does not exist: {audio_dir}")
  if not audio_dir.is_dir():
    raise ValueError(f"Audio path is not a directory: {audio_dir}")
  audio_files = [
    AudioInput(path=path, stem=path.stem, format=path.suffix.lower().lstrip("."))
    for path in sorted(audio_dir.iterdir(), key=lambda child: child.name)
    if path.is_file() and path.suffix.lower().lstrip(".") in asr_simple.TRANSCRIPTION_CONTENT_TYPES
  ]
  if not audio_files:
    supported = ", ".join(sorted(asr_simple.TRANSCRIPTION_CONTENT_TYPES))
    raise ValueError(f"No supported audio files found in {audio_dir}; expected extensions: {supported}")
  stems: dict[str, Path] = {}
  for audio_file in audio_files:
    if audio_file.stem in stems:
      raise ValueError(
        f"Duplicate audio file stem {audio_file.stem!r}: {stems[audio_file.stem].name} and {audio_file.path.name}"
      )
    stems[audio_file.stem] = audio_file.path
  validate_output_artifact_names(audio_files)
  return audio_files


def validate_output_artifact_names(audio_files: list[AudioInput]) -> None:
  """Reject audio stems that would overwrite generated transcripts or report.txt."""

  artifact_owner: dict[str, str] = {}
  for audio_file in audio_files:
    for artifact_name, kind in (
      (f"{audio_file.stem}.txt", "exact transcript"),
      (f"{audio_file.stem}_normalized.txt", "normalized transcript"),
    ):
      if artifact_name == "report.txt":
        raise ValueError(f"Audio stem {audio_file.stem!r} uses reserved output artifact {artifact_name!r}")
      previous = artifact_owner.get(artifact_name)
      if previous is not None:
        raise ValueError(
          f"Output artifact collision for {artifact_name!r}: {previous} and {audio_file.path.name} {kind}"
        )
      artifact_owner[artifact_name] = f"{audio_file.path.name} {kind}"


def resolve_output_dir(args: argparse.Namespace, audio_dir: Path) -> Path:
  """Build the requested output directory for the selected run mode."""

  if args.mode == "ground":
    return audio_dir / "ground"
  ground_dir = audio_dir / "ground"
  if not ground_dir.is_dir():
    raise ValueError(f"Ground directory does not exist: {ground_dir}")
  model = resolve_endpoint_model(args).replace("/", "_")
  return audio_dir / f"{model}_{int(time.time())}"


def create_output_dir(args: argparse.Namespace, requested_output_dir: Path) -> Path:
  """Create the output directory, suffixing eval directories on collision."""

  if args.mode == "ground":
    requested_output_dir.mkdir(parents=True, exist_ok=True)
    return requested_output_dir

  suffix = 0
  while True:
    output_dir = requested_output_dir if suffix == 0 else Path(f"{requested_output_dir}-{suffix}")
    try:
      output_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
      suffix += 1
      continue
    return output_dir


def validate_eval_ground(audio_dir: Path, audio_files: list[AudioInput]) -> None:
  """Ensure eval mode has a normalized ground transcript for every file."""

  ground_dir = audio_dir / "ground"
  missing = []
  unreadable = []
  empty = []
  for audio_file in audio_files:
    ground_path = ground_dir / f"{audio_file.stem}_normalized.txt"
    if not ground_path.is_file():
      missing.append(audio_file.stem)
      continue
    try:
      ground_text = ground_path.read_text(encoding="utf-8")
    except OSError as exc:
      unreadable.append(f"{audio_file.stem}: {exc}")
      continue
    except UnicodeDecodeError as exc:
      unreadable.append(f"{audio_file.stem}: {exc}")
      continue
    if not ground_text.strip():
      empty.append(audio_file.stem)
  if missing:
    raise ValueError(f"Missing normalized ground transcript for: {', '.join(missing)}")
  if unreadable:
    raise ValueError(f"Unreadable normalized ground transcript: {'; '.join(unreadable)}")
  if empty:
    raise ValueError(f"Empty normalized ground transcript for: {', '.join(empty)}")


def resolve_endpoint_model(args: argparse.Namespace) -> str:
  """Resolve the model name for reports and output directory naming."""

  if args.endpoint == "completions":
    return asr_simple.resolve_model(args.completions_model or args.model)
  fallback_model = asr_simple.resolve_model(args.model) if args.model is not None else None
  return asr_simple.resolve_transcriptions_model_with_fallback(
    args.transcriptions_model,
    fallback_model=fallback_model,
  )


def process_audio_files(
  *,
  args: argparse.Namespace,
  audio_files: list[AudioInput],
  output_dir: Path,
  base_url: str,
  api_key: str | None,
) -> list[FileResult]:
  """Process files with bounded concurrency while preserving input order."""

  results_by_name: dict[str, FileResult] = {}
  futures: dict[Future[FileResult], AudioInput] = {}
  with tqdm(total=len(audio_files), unit="file") as progress:
    with ThreadPoolExecutor(max_workers=args.batch) as executor:
      for audio_file in audio_files:
        skipped_result = maybe_skip_ground_file(args, audio_file, output_dir)
        if skipped_result is not None:
          results_by_name[audio_file.path.name] = skipped_result
          progress.update(1)
          tqdm.write(format_file_result(skipped_result, mode=args.mode))
          continue
        futures[
          executor.submit(
            transcribe_audio_file,
            args=args,
            audio_file=audio_file,
            output_dir=output_dir,
            base_url=base_url,
            api_key=api_key,
          )
        ] = audio_file

      for future in as_completed(futures):
        result = future.result()
        results_by_name[result.audio.path.name] = result
        progress.update(1)
        tqdm.write(format_file_result(result, mode=args.mode))
  return [results_by_name[audio_file.path.name] for audio_file in audio_files]


def maybe_skip_ground_file(args: argparse.Namespace, audio_file: AudioInput, output_dir: Path) -> FileResult | None:
  """Return an existing ground transcript result when reruns can skip a file."""

  exact_path = output_dir / f"{audio_file.stem}.txt"
  normalized_path = output_dir / f"{audio_file.stem}_normalized.txt"
  if args.mode != "ground" or not exact_path.is_file():
    return None
  try:
    transcript = exact_path.read_text(encoding="utf-8")
    normalized = (
      normalized_path.read_text(encoding="utf-8") if normalized_path.is_file() else normalize_transcript(transcript)
    )
    if not normalized_path.is_file():
      normalized_path.write_text(normalized, encoding="utf-8")
    duration = get_audio_duration_seconds(audio_file.path)
  except Exception as exc:
    return build_failed_file_result(
      audio_file=audio_file,
      output_path=exact_path,
      normalized_output_path=normalized_path,
      elapsed_seconds=None,
      error_message=str(exc),
    )
  return FileResult(
    audio=audio_file,
    status="skipped",
    transcript=transcript,
    normalized_transcript=normalized,
    output_path=exact_path,
    normalized_output_path=normalized_path,
    elapsed_seconds=None,
    duration_seconds=duration,
    rtfx=None,
    exact_word_count=count_words(transcript),
    normalized_word_count=count_words(normalized),
  )


def transcribe_audio_file(
  *,
  args: argparse.Namespace,
  audio_file: AudioInput,
  output_dir: Path,
  base_url: str,
  api_key: str | None,
) -> FileResult:
  """Transcribe one file, write outputs, and attach eval scores when needed."""

  exact_path = output_dir / f"{audio_file.stem}.txt"
  normalized_path = output_dir / f"{audio_file.stem}_normalized.txt"
  started_at = time.perf_counter()
  duration = 0.0
  try:
    duration = get_audio_duration_seconds(audio_file.path)
    transcript = transcribe_with_selected_endpoint(
      args=args,
      audio_file=audio_file,
      base_url=base_url,
      api_key=api_key,
    )
    elapsed = max(time.perf_counter() - started_at, 0.0)
    normalized = normalize_transcript(transcript)
    exact_path.write_text(transcript, encoding="utf-8")
    normalized_path.write_text(normalized, encoding="utf-8")
    status = "transcribed"
    error_message = None
  except Exception as exc:
    elapsed = max(time.perf_counter() - started_at, 0.0)
    transcript = ""
    normalized = ""
    status = "failed"
    error_message = str(exc)

  result = FileResult(
    audio=audio_file,
    status=status,
    transcript=transcript,
    normalized_transcript=normalized,
    output_path=exact_path,
    normalized_output_path=normalized_path,
    elapsed_seconds=elapsed,
    duration_seconds=duration,
    rtfx=duration / elapsed if elapsed > 0 else None,
    exact_word_count=count_words(transcript),
    normalized_word_count=count_words(normalized),
    error_message=error_message,
  )
  if args.mode == "eval":
    try:
      return add_eval_scores(result, audio_file=audio_file)
    except Exception as exc:
      return FileResult(
        audio=result.audio,
        status="failed",
        transcript=result.transcript,
        normalized_transcript=result.normalized_transcript,
        output_path=result.output_path,
        normalized_output_path=result.normalized_output_path,
        elapsed_seconds=result.elapsed_seconds,
        duration_seconds=result.duration_seconds,
        rtfx=result.rtfx,
        exact_word_count=result.exact_word_count,
        normalized_word_count=result.normalized_word_count,
        error_message=str(exc),
      )
  return result


def build_failed_file_result(
  *,
  audio_file: AudioInput,
  output_path: Path,
  normalized_output_path: Path,
  elapsed_seconds: float | None,
  error_message: str,
) -> FileResult:
  """Construct a standard failed result for pre-request per-file errors."""

  return FileResult(
    audio=audio_file,
    status="failed",
    transcript="",
    normalized_transcript="",
    output_path=output_path,
    normalized_output_path=normalized_output_path,
    elapsed_seconds=elapsed_seconds,
    duration_seconds=0.0,
    rtfx=None,
    exact_word_count=0,
    normalized_word_count=0,
    error_message=error_message,
  )


def transcribe_with_selected_endpoint(
  *,
  args: argparse.Namespace,
  audio_file: AudioInput,
  base_url: str,
  api_key: str | None,
) -> str:
  """Dispatch one audio file to the selected transcription endpoint."""

  if args.endpoint == "completions":
    return transcribe_with_completions(args=args, audio_file=audio_file, base_url=base_url, api_key=api_key)
  return transcribe_with_transcriptions(args=args, audio_file=audio_file, base_url=base_url, api_key=api_key)


def transcribe_with_completions(
  *,
  args: argparse.Namespace,
  audio_file: AudioInput,
  base_url: str,
  api_key: str | None,
) -> str:
  """Send one audio file through the chat completions transcription path."""

  request_args = build_completions_request_args(args)
  payload = asr_simple.prune_none(
    asr_simple.build_completions_request_config(request_args, audio_file.path.read_bytes(), audio_file.format)
  )
  exchange = asr_simple.send_json_request(
    url=asr_simple.build_api_url(base_url, "/v1/chat/completions"),
    api_key=api_key,
    payload=payload,
    timeout=args.timeout,
  )
  stream = payload.get("stream") is True
  transcript = asr_simple.extract_completions_response_text(
    exchange.response_json, exchange.response_body_text, stream=stream
  )
  transcript = asr_simple.normalize_known_model_transcript(
    transcript,
    requested_model=payload.get("model") if isinstance(payload.get("model"), str) else None,
  )
  error_message = asr_simple.determine_error_message(
    exchange, transcript
  ) or asr_simple.validate_completions_response_format(exchange, stream=stream)
  if error_message is not None:
    raise ValueError(error_message)
  return transcript


def transcribe_with_transcriptions(
  *,
  args: argparse.Namespace,
  audio_file: AudioInput,
  base_url: str,
  api_key: str | None,
) -> str:
  """Send one audio file through the multipart transcriptions endpoint."""

  request_args = build_transcriptions_request_args(args)
  payload = asr_simple.prune_none(asr_simple.build_transcriptions_request_config(request_args))
  if args.service_tier is not None:
    payload["service_tier"] = args.service_tier
  exchange = asr_simple.send_multipart_request(
    url=asr_simple.build_api_url(base_url, "/v1/audio/transcriptions"),
    api_key=api_key,
    fields=payload,
    file_path=audio_file.path,
    file_format=audio_file.format,
    timeout=args.timeout,
  )
  stream = payload.get("stream") is True
  response_format = payload.get("response_format")
  transcript = asr_simple.extract_transcription_response_text(
    exchange.response_json, exchange.response_body_text, stream=stream
  )
  error_message = asr_simple.determine_error_message(
    exchange, transcript
  ) or asr_simple.validate_transcriptions_response_format(
    exchange,
    response_format=response_format if isinstance(response_format, str) else None,
    stream=stream,
  )
  if error_message is not None:
    raise ValueError(error_message)
  return transcript


def build_completions_request_args(args: argparse.Namespace) -> argparse.Namespace:
  """Derive per-file completions arguments from batch-level arguments."""

  request_args = argparse.Namespace(**vars(args))
  request_args.system_prompt = request_args.system_prompt or asr_simple.DEFAULT_SYSTEM_PROMPT
  request_args.developer_prompt = (
    request_args.prompt or request_args.developer_prompt or asr_simple.DEFAULT_DEVELOPER_PROMPT
  )
  request_args.user_prompt = request_args.user_prompt or asr_simple.DEFAULT_USER_PROMPT
  if request_args.service_tier is not None:
    request_args.completions_service_tier = request_args.service_tier
  return request_args


def build_transcriptions_request_args(args: argparse.Namespace) -> argparse.Namespace:
  """Derive transcriptions arguments with the ASR-specific model fallback."""

  request_args = argparse.Namespace(**vars(args))
  request_args.transcriptions_model = resolve_endpoint_model(args)
  request_args.model = None
  if request_args.prompt is not None:
    request_args.transcriptions_prompt = request_args.prompt
  return request_args


def add_eval_scores(result: FileResult, *, audio_file: AudioInput) -> FileResult:
  """Attach WER metrics by comparing a hypothesis to normalized ground text."""

  reference = audio_file.path.parent / "ground" / f"{audio_file.stem}_normalized.txt"
  reference_text = reference.read_text(encoding="utf-8")
  errors, reference_words, wer = compute_plain_word_error_rate(reference_text, result.normalized_transcript)
  return FileResult(
    audio=result.audio,
    status=result.status,
    transcript=result.transcript,
    normalized_transcript=result.normalized_transcript,
    output_path=result.output_path,
    normalized_output_path=result.normalized_output_path,
    elapsed_seconds=result.elapsed_seconds,
    duration_seconds=result.duration_seconds,
    rtfx=result.rtfx,
    exact_word_count=result.exact_word_count,
    normalized_word_count=result.normalized_word_count,
    reference_word_count=count_words(reference_text),
    wer_errors=errors,
    wer_reference_words=reference_words,
    wer=wer,
    error_message=result.error_message,
  )


def normalize_transcript(transcript: str) -> str:
  """Normalize transcript text before WER comparison."""

  return TRANSCRIPT_NORMALIZER(transcript)


def count_words(transcript: str) -> int:
  """Count whitespace-delimited words after exact or normalized transcription."""

  return len(transcript.split())


def compute_plain_word_error_rate(reference_text: str, hypothesis_text: str) -> tuple[int, int, float]:
  """Compute edit-distance WER over already-normalized whitespace tokens."""

  reference_words = reference_text.split()
  hypothesis_words = hypothesis_text.split()
  if not reference_words:
    errors = len(hypothesis_words)
    return (errors, 0, 1.0 if errors else 0.0)
  previous_row = list(range(len(hypothesis_words) + 1))
  for reference_index, reference_word in enumerate(reference_words, start=1):
    current_row = [reference_index]
    for hypothesis_index, hypothesis_word in enumerate(hypothesis_words, start=1):
      substitution_cost = 0 if reference_word == hypothesis_word else 1
      current_row.append(
        min(
          previous_row[hypothesis_index] + 1,
          current_row[hypothesis_index - 1] + 1,
          previous_row[hypothesis_index - 1] + substitution_cost,
        )
      )
    previous_row = current_row
  errors = previous_row[-1]
  return (errors, len(reference_words), errors / len(reference_words))


def compute_aggregate_wer(*, errors: int, reference_words: int) -> float:
  """Compute corpus WER, treating hallucinations on empty references as 100%."""

  if reference_words:
    return errors / reference_words
  return 1.0 if errors else 0.0


def compute_aggregate_rtfx(results: list[FileResult], *, wall_elapsed_seconds: float) -> float:
  """Compute batch RTFx over files that actually attempted processing."""

  processed_duration = sum(result.duration_seconds for result in results if result.elapsed_seconds is not None)
  return processed_duration / wall_elapsed_seconds if wall_elapsed_seconds > 0 else 0.0


def get_audio_duration_seconds(audio_path: Path) -> float:
  """Read audio duration from mutagen metadata, returning zero if unavailable."""

  audio = mutagen.File(audio_path)
  length = getattr(getattr(audio, "info", None), "length", None)
  if isinstance(length, int | float):
    return float(length)
  return 0.0


def format_file_result(result: FileResult, *, mode: str) -> str:
  """Render one terminal progress row for a completed file."""

  elapsed = "n/a" if result.elapsed_seconds is None else f"{result.elapsed_seconds:.2f}s"
  rtfx = "n/a" if result.rtfx is None else f"{result.rtfx:.2f}x"
  common = (
    f"{sanitize_output_field(result.audio.path.name)}: {result.status}, elapsed={elapsed}, "
    f"duration={result.duration_seconds:.2f}s, "
    f"RTFx={rtfx}, exact_words={result.exact_word_count}, normalized_words={result.normalized_word_count}"
  )
  if mode == "eval" and result.wer is not None:
    common += (
      f", WER={result.wer:.2%} ({result.wer_errors}/{result.wer_reference_words}), "
      f"reference_words={result.reference_word_count}"
    )
  if result.error_message:
    common += f", error={sanitize_output_field(result.error_message)}"
  return f"{common}, output={sanitize_output_field(str(result.output_path))}"


def sanitize_output_field(value: str) -> str:
  """Escape control characters before rendering terminal or TSV output."""

  return value.replace("\\", "\\\\").replace("\t", "\\t").replace("\r", "\\r").replace("\n", "\\n")


def print_final_stats(results: list[FileResult], *, mode: str, wall_elapsed_seconds: float) -> None:
  """Print aggregate terminal statistics for the batch run."""

  transcribed = sum(1 for result in results if result.status == "transcribed")
  skipped = sum(1 for result in results if result.status == "skipped")
  failed = sum(1 for result in results if result.status == "failed")
  aggregate_rtfx = compute_aggregate_rtfx(results, wall_elapsed_seconds=wall_elapsed_seconds)
  line = (
    f"Total files={len(results)}, transcribed={transcribed}, skipped={skipped}, failed={failed}, "
    f"aggregate RTFx={aggregate_rtfx:.2f}x"
  )
  if mode == "eval":
    errors = sum(result.wer_errors or 0 for result in results)
    reference_words = sum(result.wer_reference_words or 0 for result in results)
    wer = compute_aggregate_wer(errors=errors, reference_words=reference_words)
    line += f", aggregate WER={wer:.2%} ({errors}/{reference_words})"
  print(line)


def write_report(
  *,
  args: argparse.Namespace,
  model: str,
  output_dir: Path,
  results: list[FileResult],
  wall_elapsed_seconds: float,
) -> None:
  """Write report.txt with run metadata, per-file rows, and aggregates."""

  transcribed = sum(1 for result in results if result.status == "transcribed")
  skipped = sum(1 for result in results if result.status == "skipped")
  failed = sum(1 for result in results if result.status == "failed")
  aggregate_rtfx = compute_aggregate_rtfx(results, wall_elapsed_seconds=wall_elapsed_seconds)
  lines = [
    "asr-wer report",
    f"mode: {args.mode}",
    f"endpoint: {args.endpoint}",
    f"model: {model}",
    f"batch_size: {args.batch}",
    f"service_tier: {args.service_tier or 'none'}",
    f"prompt_present: {bool(args.prompt)}",
    f"output_folder: {output_dir}",
    f"total_files: {len(results)}",
    f"transcribed: {transcribed}",
    f"skipped: {skipped}",
    f"failed: {failed}",
    f"wall_elapsed_seconds: {wall_elapsed_seconds:.2f}",
    f"aggregate_rtfx: {aggregate_rtfx:.2f}x",
  ]
  if args.mode == "eval":
    errors = sum(result.wer_errors or 0 for result in results)
    reference_words = sum(result.wer_reference_words or 0 for result in results)
    wer = compute_aggregate_wer(errors=errors, reference_words=reference_words)
    lines.append(f"aggregate_wer_percent: {wer:.2%}")
    lines.append(f"aggregate_wer_errors: {errors}")
    lines.append(f"aggregate_wer_reference_words: {reference_words}")

  lines.append("")
  lines.append(render_report_header(eval_mode=args.mode == "eval"))
  for result in results:
    lines.append(render_report_row(result, eval_mode=args.mode == "eval"))
  output_dir.joinpath("report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_report_header(*, eval_mode: bool) -> str:
  """Render the tab-separated report header."""

  columns = [
    "file",
    "status",
    "elapsed_seconds",
    "duration_seconds",
    "rtfx",
    "exact_words",
    "normalized_words",
  ]
  if eval_mode:
    columns.extend(("reference_words", "WER", "wer_errors", "wer_reference_words"))
  columns.extend(("output_path", "error"))
  return "\t".join(columns)


def render_report_row(result: FileResult, *, eval_mode: bool) -> str:
  """Render one tab-separated report row."""

  columns = [
    sanitize_output_field(result.audio.path.name),
    result.status,
    "" if result.elapsed_seconds is None else f"{result.elapsed_seconds:.2f}",
    f"{result.duration_seconds:.2f}",
    "" if result.rtfx is None else f"{result.rtfx:.2f}",
    str(result.exact_word_count),
    str(result.normalized_word_count),
  ]
  if eval_mode:
    columns.extend(
      (
        "" if result.reference_word_count is None else str(result.reference_word_count),
        "" if result.wer is None else f"{result.wer:.2%}",
        "" if result.wer_errors is None else str(result.wer_errors),
        "" if result.wer_reference_words is None else str(result.wer_reference_words),
      )
    )
  columns.extend((sanitize_output_field(str(result.output_path)), sanitize_output_field(result.error_message or "")))
  return "\t".join(columns)


ASR_WER_MODULE = EndpointTestModule(
  name="asr-wer",
  summary="Batch transcribe audio folders and report WER and RTFx.",
  configure_parser=configure_parser,
  handler=run,
)
