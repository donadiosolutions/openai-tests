"""Batch ASR transcription and WER reporting module."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
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
VERBOSE_OUTPUT_LOCK = threading.Lock()


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
  chunk_count: int | None = None


@dataclass(frozen=True, slots=True)
class PreparedChunk:
  """One chunk row from prep/manifest.json."""

  audio: AudioInput
  source: AudioInput
  index: int
  start_seconds: float
  end_seconds: float
  duration_seconds: float


@dataclass(frozen=True, slots=True)
class PreparedSource:
  """A source audio file and all manifest chunks that rebuild it."""

  audio: AudioInput
  chunks: tuple[PreparedChunk, ...]
  duration_seconds: float
  overlap_seconds: float
  segment_duration_seconds: float


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
    "--prep", action="store_true", help="Use AUDIO_DIR/prep chunks and stitch results per source file."
  )
  parser.add_argument(
    "--overlap", type=float, help="Expected prep overlap seconds; validated against prep/manifest.json."
  )
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
    prepared_sources = resolve_prepared_audio_files(audio_dir, requested_overlap=args.overlap) if args.prep else None
    if prepared_sources is not None:
      args._prep_overlap_seconds = prepared_sources[0].overlap_seconds
      args._prep_segment_duration_seconds = prepared_sources[0].segment_duration_seconds
    audio_files = (
      [source.audio for source in prepared_sources] if prepared_sources is not None else discover_audio_files(audio_dir)
    )
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
  if prepared_sources is None:
    results = process_audio_files(
      args=args,
      audio_files=audio_files,
      output_dir=output_dir,
      base_url=base_url,
      api_key=api_key,
    )
  else:
    results = process_prepared_sources(
      args=args,
      prepared_sources=prepared_sources,
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
  if args.overlap is not None and not args.prep:
    raise ValueError("overlap can only be used with --prep")
  if args.prompt is not None and args.endpoint == "transcriptions" and args.transcriptions_prompt is not None:
    raise ValueError("prompt cannot be provided with transcriptions-prompt")
  if args.endpoint == "transcriptions" and has_explicit_completions_prompt_override(args):
    raise ValueError("completions prompt flags cannot be used with transcriptions; use prompt or transcriptions-prompt")
  if args.endpoint == "transcriptions" and args.transcriptions_response_format in {"diarized_json", "srt", "vtt"}:
    raise ValueError(
      "transcriptions-response-format must be transcript-only for asr-wer; "
      "diarized_json, srt, and vtt include labels or timestamps"
    )
  if args.prompt is not None and args.endpoint == "completions" and has_explicit_completions_prompt_override(args):
    raise ValueError("prompt cannot be provided with completions prompt overrides")
  if args.endpoint == "completions" and args.completions_messages_json is not None:
    raise ValueError("completions-messages-json cannot be used with batch audio")
  if args.endpoint == "completions" and args.service_tier is not None and args.completions_service_tier is not None:
    raise ValueError("service-tier cannot be provided with completions-service-tier")
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
  try:
    children = sorted(audio_dir.iterdir(), key=lambda child: child.name)
  except OSError as exc:
    raise ValueError(f"Unable to list audio directory {audio_dir}: {exc}") from exc
  audio_files = [
    AudioInput(path=path, stem=path.stem, format=path.suffix.lower().lstrip("."))
    for path in children
    if path.is_file() and path.suffix.lower().lstrip(".") in asr_simple.TRANSCRIPTION_CONTENT_TYPES
  ]
  if not audio_files:
    supported = ", ".join(sorted(asr_simple.TRANSCRIPTION_CONTENT_TYPES))
    raise ValueError(f"No supported audio files found in {audio_dir}; expected extensions: {supported}")
  stems: dict[str, Path] = {}
  for audio_file in audio_files:
    stem_key = audio_file.stem.casefold()
    if stem_key in stems:
      raise ValueError(
        f"Duplicate audio file stem {audio_file.stem!r}: {stems[stem_key].name} and {audio_file.path.name}"
      )
    stems[stem_key] = audio_file.path
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
      artifact_key = artifact_name.casefold()
      if artifact_key == "report.txt":
        raise ValueError(f"Audio stem {audio_file.stem!r} uses reserved output artifact {artifact_name!r}")
      previous = artifact_owner.get(artifact_key)
      if previous is not None:
        raise ValueError(
          f"Output artifact collision for {artifact_name!r}: {previous} and {audio_file.path.name} {kind}"
        )
      artifact_owner[artifact_key] = f"{audio_file.path.name} {kind}"


def resolve_prepared_audio_files(audio_dir: Path, *, requested_overlap: float | None) -> list[PreparedSource]:
  """Load prep/manifest.json and return original source files grouped with chunks."""

  manifest_path = audio_dir / "prep" / "manifest.json"
  if not manifest_path.is_file():
    raise ValueError(f"Prepared runs require {manifest_path}")
  try:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
  except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
    raise ValueError(f"Unable to read prepared manifest {manifest_path}: {exc}") from exc
  if not isinstance(manifest, dict):
    raise ValueError(f"Prepared manifest must be a JSON object: {manifest_path}")
  overlap = require_manifest_number(manifest, "overlap_seconds")
  segment_duration = require_manifest_number(manifest, "segment_duration_seconds")
  if requested_overlap is not None and not math.isclose(requested_overlap, overlap, rel_tol=0.0, abs_tol=0.001):
    raise ValueError(f"Requested overlap {requested_overlap} does not match manifest overlap {overlap}")
  chunks_raw = manifest.get("chunks")
  sources_raw = manifest.get("sources")
  if not isinstance(chunks_raw, list) or not isinstance(sources_raw, list):
    raise ValueError("Prepared manifest requires sources and chunks arrays")

  source_durations: dict[str, float] = {}
  for source_raw in sources_raw:
    if not isinstance(source_raw, dict) or not isinstance(source_raw.get("source_file"), str):
      raise ValueError("Prepared manifest source rows must include source_file")
    source_durations[source_raw["source_file"]] = require_manifest_number(source_raw, "duration_seconds")

  grouped: dict[str, list[PreparedChunk]] = {}
  for chunk_raw in chunks_raw:
    if not isinstance(chunk_raw, dict):
      raise ValueError("Prepared manifest chunk rows must be objects")
    source_file = require_manifest_string(chunk_raw, "source_file")
    source_stem = require_manifest_string(chunk_raw, "source_stem")
    chunk_file = require_manifest_string(chunk_raw, "chunk_file")
    chunk_path = audio_dir / "prep" / chunk_file
    if not chunk_path.is_file():
      raise ValueError(f"Prepared chunk file does not exist: {chunk_path}")
    source_path = audio_dir / source_file
    source_audio = AudioInput(
      path=source_path,
      stem=source_stem,
      format=source_path.suffix.lower().lstrip("."),
    )
    chunk_audio = AudioInput(path=chunk_path, stem=chunk_path.stem, format=chunk_path.suffix.lower().lstrip("."))
    grouped.setdefault(source_file, []).append(
      PreparedChunk(
        audio=chunk_audio,
        source=source_audio,
        index=int(require_manifest_number(chunk_raw, "chunk_index")),
        start_seconds=require_manifest_number(chunk_raw, "start_seconds"),
        end_seconds=require_manifest_number(chunk_raw, "end_seconds"),
        duration_seconds=require_manifest_number(chunk_raw, "duration_seconds"),
      )
    )

  prepared_sources = [
    PreparedSource(
      audio=chunks[0].source,
      chunks=tuple(sorted(chunks, key=lambda chunk: chunk.index)),
      duration_seconds=source_durations.get(source_file, sum(chunk.duration_seconds for chunk in chunks)),
      overlap_seconds=overlap,
      segment_duration_seconds=segment_duration,
    )
    for source_file, chunks in sorted(grouped.items())
  ]
  if not prepared_sources:
    raise ValueError("Prepared manifest does not contain any chunks")
  validate_output_artifact_names([source.audio for source in prepared_sources])
  return prepared_sources


def require_manifest_string(row: dict[str, Any], key: str) -> str:
  value = row.get(key)
  if not isinstance(value, str) or not value:
    raise ValueError(f"Prepared manifest row requires string {key}")
  return value


def require_manifest_number(row: dict[str, Any], key: str) -> float:
  value = row.get(key)
  if not isinstance(value, (int, float)) or isinstance(value, bool):
    raise ValueError(f"Prepared manifest row requires numeric {key}")
  return float(value)


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
    if requested_output_dir.exists() and not requested_output_dir.is_dir():
      raise ValueError(f"Ground output path is not a directory: {requested_output_dir}")
    try:
      requested_output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
      raise ValueError(f"Unable to create ground output directory {requested_output_dir}: {exc}") from exc
    return requested_output_dir

  suffix = 0
  while True:
    output_dir = requested_output_dir if suffix == 0 else Path(f"{requested_output_dir}-{suffix}")
    try:
      output_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
      suffix += 1
      continue
    except OSError as exc:
      raise ValueError(f"Unable to create eval output directory {output_dir}: {exc}") from exc
    return output_dir


def validate_eval_ground(audio_dir: Path, audio_files: list[AudioInput]) -> None:
  """Ensure eval mode has a normalized ground transcript for every file."""

  ground_dir = audio_dir / "ground"
  missing = []
  unreadable = []
  for audio_file in audio_files:
    ground_path = ground_dir / f"{audio_file.stem}_normalized.txt"
    if not ground_path.is_file():
      missing.append(audio_file.stem)
      continue
    try:
      ground_path.read_text(encoding="utf-8")
    except OSError as exc:
      unreadable.append(f"{audio_file.stem}: {exc}")
      continue
    except UnicodeDecodeError as exc:
      unreadable.append(f"{audio_file.stem}: {exc}")
      continue
  if missing:
    raise ValueError(f"Missing normalized ground transcript for: {', '.join(missing)}")
  if unreadable:
    raise ValueError(f"Unreadable normalized ground transcript: {'; '.join(unreadable)}")


def resolve_endpoint_model(args: argparse.Namespace) -> str:
  """Resolve the model name for reports and output directory naming."""

  if args.endpoint == "completions":
    return asr_simple.resolve_model(args.completions_model or args.model)
  fallback_model = args.model or os.getenv("OPENAI_MODEL") or os.getenv("OPENAI_TESTS_MODEL")
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


def process_prepared_sources(
  *,
  args: argparse.Namespace,
  prepared_sources: list[PreparedSource],
  output_dir: Path,
  base_url: str,
  api_key: str | None,
) -> list[FileResult]:
  """Transcribe prepared chunks with shared concurrency and stitch by source."""

  output_dir.joinpath("chunks").mkdir(parents=True, exist_ok=True)
  results_by_source: dict[str, FileResult] = {}
  all_chunks = [chunk for source in prepared_sources for chunk in source.chunks]
  chunk_transcripts: dict[str, dict[int, str]] = {source.audio.path.name: {} for source in prepared_sources}
  chunk_errors: dict[str, list[str]] = {source.audio.path.name: [] for source in prepared_sources}
  started_by_source: dict[str, float] = {}

  with tqdm(total=len(all_chunks), unit="chunk") as progress:
    with ThreadPoolExecutor(max_workers=args.batch) as executor:
      futures: dict[Future[str], PreparedChunk] = {}
      for source in prepared_sources:
        skipped_result = maybe_skip_prepared_ground_file(args, source, output_dir)
        if skipped_result is not None:
          results_by_source[source.audio.path.name] = skipped_result
          progress.update(len(source.chunks))
          tqdm.write(format_file_result(skipped_result, mode=args.mode))
          continue
        started_by_source[source.audio.path.name] = time.perf_counter()
        for chunk in source.chunks:
          futures[
            executor.submit(
              transcribe_with_selected_endpoint,
              args=args,
              audio_file=chunk.audio,
              base_url=base_url,
              api_key=api_key,
            )
          ] = chunk

      for future in as_completed(futures):
        chunk = futures[future]
        try:
          transcript = future.result()
          chunk_transcripts[chunk.source.path.name][chunk.index] = transcript
          chunk_output = output_dir / "chunks" / f"{chunk.audio.stem}.txt"
          atomic_write_text(chunk_output, transcript)
        except Exception as exc:
          chunk_errors[chunk.source.path.name].append(str(exc))
        progress.update(1)

  for source in prepared_sources:
    if source.audio.path.name in results_by_source:
      continue
    started_at = started_by_source.get(source.audio.path.name, time.perf_counter())
    elapsed = max(time.perf_counter() - started_at, 0.0)
    results_by_source[source.audio.path.name] = build_prepared_source_result(
      args=args,
      source=source,
      output_dir=output_dir,
      chunk_transcripts=chunk_transcripts[source.audio.path.name],
      chunk_errors=chunk_errors[source.audio.path.name],
      elapsed_seconds=elapsed,
    )
    tqdm.write(format_file_result(results_by_source[source.audio.path.name], mode=args.mode))

  return [results_by_source[source.audio.path.name] for source in prepared_sources]


def maybe_skip_prepared_ground_file(
  args: argparse.Namespace,
  source: PreparedSource,
  output_dir: Path,
) -> FileResult | None:
  """Skip prepared ground only when combined exact and normalized artifacts exist."""

  exact_path = output_dir / f"{source.audio.stem}.txt"
  normalized_path = output_dir / f"{source.audio.stem}_normalized.txt"
  if args.mode != "ground" or not exact_path.is_file() or not normalized_path.is_file():
    return None
  try:
    transcript = exact_path.read_text(encoding="utf-8")
    normalized = normalized_path.read_text(encoding="utf-8")
  except Exception as exc:
    return build_failed_file_result(
      audio_file=source.audio,
      output_path=exact_path,
      normalized_output_path=normalized_path,
      elapsed_seconds=None,
      error_message=str(exc),
      chunk_count=len(source.chunks),
    )
  return FileResult(
    audio=source.audio,
    status="skipped",
    transcript=transcript,
    normalized_transcript=normalized,
    output_path=exact_path,
    normalized_output_path=normalized_path,
    elapsed_seconds=None,
    duration_seconds=source.duration_seconds,
    rtfx=None,
    exact_word_count=count_words(transcript),
    normalized_word_count=count_words(normalized),
    chunk_count=len(source.chunks),
  )


def build_prepared_source_result(
  *,
  args: argparse.Namespace,
  source: PreparedSource,
  output_dir: Path,
  chunk_transcripts: dict[int, str],
  chunk_errors: list[str],
  elapsed_seconds: float,
) -> FileResult:
  exact_path = output_dir / f"{source.audio.stem}.txt"
  normalized_path = output_dir / f"{source.audio.stem}_normalized.txt"
  if chunk_errors or len(chunk_transcripts) != len(source.chunks):
    missing = sorted({chunk.index for chunk in source.chunks} - set(chunk_transcripts))
    errors = [*chunk_errors]
    if missing:
      errors.append(f"missing chunk transcripts: {', '.join(str(index) for index in missing)}")
    result = build_failed_file_result(
      audio_file=source.audio,
      output_path=exact_path,
      normalized_output_path=normalized_path,
      elapsed_seconds=elapsed_seconds,
      error_message="; ".join(errors),
      duration_seconds=source.duration_seconds,
      chunk_count=len(source.chunks),
    )
  else:
    ordered_transcripts = [chunk_transcripts[chunk.index] for chunk in source.chunks]
    transcript = "\n".join(ordered_transcripts)
    normalized = stitch_normalized_transcripts(
      [normalize_transcript(transcript) for transcript in ordered_transcripts],
      overlap_seconds=source.overlap_seconds,
    )
    try:
      atomic_write_text(exact_path, transcript)
      atomic_write_text(normalized_path, normalized)
      result = FileResult(
        audio=source.audio,
        status="transcribed",
        transcript=transcript,
        normalized_transcript=normalized,
        output_path=exact_path,
        normalized_output_path=normalized_path,
        elapsed_seconds=elapsed_seconds,
        duration_seconds=source.duration_seconds,
        rtfx=source.duration_seconds / elapsed_seconds if elapsed_seconds > 0 else None,
        exact_word_count=count_words(transcript),
        normalized_word_count=count_words(normalized),
        chunk_count=len(source.chunks),
      )
    except Exception as exc:
      result = build_failed_file_result(
        audio_file=source.audio,
        output_path=exact_path,
        normalized_output_path=normalized_path,
        elapsed_seconds=elapsed_seconds,
        error_message=str(exc),
        duration_seconds=source.duration_seconds,
        chunk_count=len(source.chunks),
      )
  if args.mode == "eval":
    try:
      return add_eval_scores(result, audio_file=source.audio)
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
        chunk_count=result.chunk_count,
      )
  return result


def stitch_normalized_transcripts(normalized_chunks: list[str], *, overlap_seconds: float) -> str:
  stitched: list[str] = []
  max_overlap_tokens = math.ceil(6 * overlap_seconds) + 6
  for normalized in normalized_chunks:
    words = normalized.split()
    if not stitched:
      stitched.extend(words)
      continue
    limit = min(max_overlap_tokens, len(stitched), len(words))
    overlap = 0
    for size in range(limit, 0, -1):
      if stitched[-size:] == words[:size]:
        overlap = size
        break
    stitched.extend(words[overlap:])
  return " ".join(stitched)


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
      atomic_write_text(normalized_path, normalized)
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
    atomic_write_text(exact_path, transcript)
    atomic_write_text(normalized_path, normalized)
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
        chunk_count=result.chunk_count,
      )
  return result


def atomic_write_text(path: Path, text: str) -> None:
  """Write text via a temporary sibling before replacing the final path."""

  temporary_path = path.with_suffix(f"{path.suffix}.tmp")
  try:
    temporary_path.write_text(text, encoding="utf-8")
    temporary_path.replace(path)
  except Exception:
    temporary_path.unlink(missing_ok=True)
    raise


def build_failed_file_result(
  *,
  audio_file: AudioInput,
  output_path: Path,
  normalized_output_path: Path,
  elapsed_seconds: float | None,
  error_message: str,
  duration_seconds: float = 0.0,
  chunk_count: int | None = None,
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
    duration_seconds=duration_seconds,
    rtfx=None,
    exact_word_count=0,
    normalized_word_count=0,
    error_message=error_message,
    chunk_count=chunk_count,
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
  if args.verbose:
    print_verbose_exchange(exchange)
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
  if args.verbose:
    print_verbose_exchange(exchange)
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
  if request_args.system_prompt is None:
    request_args.system_prompt = asr_simple.DEFAULT_SYSTEM_PROMPT
  if request_args.prompt is not None:
    request_args.developer_prompt = request_args.prompt
  elif request_args.developer_prompt is None:
    request_args.developer_prompt = asr_simple.DEFAULT_DEVELOPER_PROMPT
  if request_args.user_prompt is None:
    request_args.user_prompt = asr_simple.DEFAULT_USER_PROMPT
  if request_args.service_tier is not None:
    request_args.completions_service_tier = request_args.service_tier
  if getattr(request_args, "prep", False) and request_args.completions_temperature is None:
    request_args.completions_temperature = 0.0
  return request_args


def build_transcriptions_request_args(args: argparse.Namespace) -> argparse.Namespace:
  """Derive transcriptions arguments with the ASR-specific model fallback."""

  request_args = argparse.Namespace(**vars(args))
  request_args.transcriptions_model = resolve_endpoint_model(args)
  request_args.model = None
  if request_args.prompt is not None:
    request_args.transcriptions_prompt = request_args.prompt
  if getattr(request_args, "prep", False) and request_args.transcriptions_temperature is None:
    request_args.transcriptions_temperature = 0.0
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
    chunk_count=result.chunk_count,
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
  """Read audio duration from mutagen metadata, returning zero when length is absent."""

  audio = mutagen.File(audio_path)
  length = getattr(getattr(audio, "info", None), "length", None)
  if isinstance(length, (int, float)) and not isinstance(length, bool):
    return float(length)
  return 0.0


def print_verbose_exchange(exchange: asr_simple.HttpExchange) -> None:
  """Print request and response details for --verbose batch runs."""

  message = "\n".join(
    (
      "",
      "Request:",
      f"{exchange.method} {exchange.url}",
      json.dumps(asr_simple.redact_headers(exchange.request_headers), indent=2, sort_keys=True),
      asr_simple.format_json_like(exchange.request_body),
      "",
      "Response:",
      f"HTTP {exchange.response_status if exchange.response_status is not None else 'N/A'}",
      json.dumps(exchange.response_headers, indent=2, sort_keys=True),
      exchange.response_body_text or "(empty)",
    )
  )
  with VERBOSE_OUTPUT_LOCK:
    print(message)


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
    f"service_tier: {resolve_report_service_tier(args)}",
    f"temperature: {resolve_report_temperature(args)}",
    f"prompt_present: {has_report_prompt(args)}",
    f"prepared_source: {str(bool(getattr(args, 'prep', False))).lower()}",
    f"prep_folder: {Path(args.audio_dir) / 'prep' if getattr(args, 'prep', False) else 'none'}",
    f"prep_overlap_seconds: {resolve_report_prep_overlap(args, results)}",
    f"prep_segment_duration_seconds: {resolve_report_prep_segment_duration(args, results)}",
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


def resolve_report_service_tier(args: argparse.Namespace) -> str:
  """Return the effective service tier represented by report metadata."""

  if args.service_tier is not None:
    return str(args.service_tier)
  if args.endpoint == "completions" and args.completions_service_tier is not None:
    return str(args.completions_service_tier)
  return "none"


def resolve_report_temperature(args: argparse.Namespace) -> str:
  """Return the effective selected-endpoint temperature for report metadata."""

  if args.endpoint == "completions":
    if args.completions_temperature is not None:
      return str(args.completions_temperature)
    return "0.0" if getattr(args, "prep", False) else "provider_default"
  if args.transcriptions_temperature is not None:
    return str(args.transcriptions_temperature)
  return "0.0" if getattr(args, "prep", False) else "provider_default"


def resolve_report_prep_overlap(args: argparse.Namespace, results: list[FileResult]) -> str:
  del results
  if not getattr(args, "prep", False):
    return "none"
  return str(getattr(args, "_prep_overlap_seconds", args.overlap if args.overlap is not None else "unknown"))


def resolve_report_prep_segment_duration(args: argparse.Namespace, results: list[FileResult]) -> str:
  del results
  if not getattr(args, "prep", False):
    return "none"
  return str(getattr(args, "_prep_segment_duration_seconds", 30.0))


def has_report_prompt(args: argparse.Namespace) -> bool:
  """Return whether the run used an explicit prompt option."""

  if args.prompt is not None:
    return True
  if args.endpoint == "transcriptions":
    return args.transcriptions_prompt is not None
  return any(getattr(args, name) is not None for name in ("system_prompt", "developer_prompt", "user_prompt"))


def render_report_header(*, eval_mode: bool) -> str:
  """Render the tab-separated report header."""

  columns = [
    "file",
    "status",
    "chunk_count",
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
    "" if result.chunk_count is None else str(result.chunk_count),
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
