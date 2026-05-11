"""Batch ASR transcription and WER reporting module."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, as_completed, wait
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
OVERLAP_TOKEN_RATE_PER_SECOND = 6
OVERLAP_TOKEN_BUFFER = 6


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
  """
  Execute a batch ground-generation or evaluation run driven by parsed CLI arguments.
  
  Performs argument validation and input/output resolution, processes either regular audio files or prepared (pre-chunked) sources, writes the run report, prints final statistics, and exits with a status code that reflects overall success.
  
  Returns:
      int: 0 if all files were transcribed or skipped; 1 if any file failed during processing; 2 on configuration error detected before processing.
  """

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
  """
  Validate CLI argument combinations and raise a ValueError for any incompatible or invalid settings.
  
  This function enforces constraints needed to produce valid WER rows and valid endpoint requests, including:
  - `batch` must be >= 1.
  - `--overlap` may only be used when `--prep` is enabled.
  - Mutually exclusive prompt flags for `transcriptions` vs `completions` endpoints.
  - `transcriptions` must request a transcript-only response format (not `diarized_json`, `srt`, or `vtt`).
  - Disallowed combinations for completions-specific flags when using batch audio.
  
  Raises:
      ValueError: If any argument combination is invalid (see list above).
  """

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
  """
  Load and validate a prep manifest from audio_dir/prep/manifest.json and return reconstructed source audios with their ordered chunks.
  
  Parameters:
    audio_dir (Path): Directory containing a "prep" subfolder with manifest and chunk files.
    requested_overlap (float | None): If provided, must match the manifest's overlap_seconds within 0.001.
  
  Returns:
    list[PreparedSource]: Ordered list of PreparedSource objects, each containing the original source AudioInput and an ordered tuple of PreparedChunk entries with timing and duration metadata.
  
  Raises:
    ValueError: If the manifest file is missing, unreadable, malformed, or any manifest validation fails (including filename/stem checks, missing chunk files, nonpositive durations, duplicate or noncontiguous chunk indices, or timing inconsistencies). 
  """

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
  if segment_duration <= 0:
    raise ValueError("Prepared manifest segment_duration_seconds must be greater than 0")
  if overlap < 0 or overlap >= segment_duration:
    raise ValueError("Prepared manifest overlap_seconds must be at least 0 and less than segment_duration_seconds")
  if requested_overlap is not None and not math.isclose(requested_overlap, overlap, rel_tol=0.0, abs_tol=0.001):
    raise ValueError(f"Requested overlap {requested_overlap} does not match manifest overlap {overlap}")
  chunks_raw = manifest.get("chunks")
  sources_raw = manifest.get("sources")
  if not isinstance(chunks_raw, list) or not isinstance(sources_raw, list):
    raise ValueError("Prepared manifest requires sources and chunks arrays")

  source_durations: dict[str, float] = {}
  source_chunk_counts: dict[str, int] = {}
  seen_source_files: set[str] = set()
  for source_raw in sources_raw:
    if not isinstance(source_raw, dict) or not isinstance(source_raw.get("source_file"), str):
      raise ValueError("Prepared manifest source rows must include source_file")
    source_file = require_manifest_filename(source_raw, "source_file")
    source_key = source_file.casefold()
    if source_key in seen_source_files:
      raise ValueError(f"Prepared manifest duplicate source_file {source_file}")
    seen_source_files.add(source_key)
    source_duration = require_manifest_number(source_raw, "duration_seconds")
    if source_duration <= 0:
      raise ValueError(f"Prepared manifest source duration_seconds for {source_file} must be greater than 0")
    source_durations[source_file] = source_duration
    source_chunk_counts[source_file] = require_manifest_integer(source_raw, "chunk_count")

  grouped: dict[str, list[PreparedChunk]] = {}
  seen_chunk_indices: dict[str, set[int]] = {}
  seen_chunk_files: set[str] = set()
  for chunk_raw in chunks_raw:
    if not isinstance(chunk_raw, dict):
      raise ValueError("Prepared manifest chunk rows must be objects")
    source_file = require_manifest_filename(chunk_raw, "source_file")
    source_stem = require_manifest_stem(chunk_raw, "source_stem")
    if source_stem != Path(source_file).stem:
      raise ValueError(f"Prepared manifest source_stem {source_stem!r} does not match source_file {source_file!r}")
    chunk_file = require_manifest_filename(chunk_raw, "chunk_file")
    chunk_key = chunk_file.casefold()
    if chunk_key in seen_chunk_files:
      raise ValueError(f"Prepared manifest duplicate chunk_file {chunk_file}")
    seen_chunk_files.add(chunk_key)
    chunk_path = audio_dir / "prep" / chunk_file
    if not chunk_path.is_file():
      raise ValueError(f"Prepared chunk file does not exist: {chunk_path}")
    source_path = audio_dir / source_file
    chunk_format = chunk_path.suffix.lower().lstrip(".")
    if chunk_format not in asr_simple.TRANSCRIPTION_CONTENT_TYPES:
      raise ValueError(f"Prepared manifest unsupported prepared chunk extension for {chunk_file}")
    source_audio = AudioInput(
      path=source_path,
      stem=source_stem,
      format=source_path.suffix.lower().lstrip("."),
    )
    chunk_audio = AudioInput(path=chunk_path, stem=chunk_path.stem, format=chunk_format)
    chunk_index = require_manifest_integer(chunk_raw, "chunk_index")
    source_indices = seen_chunk_indices.setdefault(source_file, set())
    if chunk_index in source_indices:
      raise ValueError(f"Prepared manifest duplicate chunk_index {chunk_index} for {source_file}")
    source_indices.add(chunk_index)
    chunk_duration = require_manifest_number(chunk_raw, "duration_seconds")
    if chunk_duration <= 0:
      raise ValueError(f"Prepared manifest chunk duration_seconds for {chunk_file} must be greater than 0")
    grouped.setdefault(source_file, []).append(
      PreparedChunk(
        audio=chunk_audio,
        source=source_audio,
        index=chunk_index,
        start_seconds=require_manifest_number(chunk_raw, "start_seconds"),
        end_seconds=require_manifest_number(chunk_raw, "end_seconds"),
        duration_seconds=chunk_duration,
      )
    )

  declared_sources = set(source_durations)
  chunk_sources = set(grouped)
  if declared_sources != chunk_sources:
    missing_chunks = sorted(declared_sources - chunk_sources)
    undeclared_chunks = sorted(chunk_sources - declared_sources)
    details = []
    if missing_chunks:
      details.append(f"sources without chunks: {', '.join(missing_chunks)}")
    if undeclared_chunks:
      details.append(f"chunks for undeclared sources: {', '.join(undeclared_chunks)}")
    raise ValueError(f"Prepared manifest source/chunk mismatch: {'; '.join(details)}")
  for source_file, expected_chunk_count in source_chunk_counts.items():
    actual_chunk_count = len(grouped[source_file])
    if expected_chunk_count != actual_chunk_count:
      raise ValueError(
        f"Prepared manifest chunk_count for {source_file} is {expected_chunk_count}, "
        f"but {actual_chunk_count} chunk rows were found"
      )
    expected_indices = set(range(expected_chunk_count))
    actual_indices = {chunk.index for chunk in grouped[source_file]}
    if actual_indices != expected_indices:
      raise ValueError(
        f"Prepared manifest chunk_index values for {source_file} must be contiguous from "
        f"0 to {expected_chunk_count - 1}"
      )
    validate_prepared_chunk_ranges(
      source_file=source_file,
      source_duration=source_durations[source_file],
      overlap=overlap,
      chunks=grouped[source_file],
    )

  prepared_sources = [
    PreparedSource(
      audio=chunks[0].source,
      chunks=tuple(sorted(chunks, key=lambda chunk: chunk.index)),
      duration_seconds=source_durations[source_file],
      overlap_seconds=overlap,
      segment_duration_seconds=segment_duration,
    )
    for source_file, chunks in sorted(grouped.items())
  ]
  if not prepared_sources:
    raise ValueError("Prepared manifest does not contain any chunks")
  validate_output_artifact_names([source.audio for source in prepared_sources])
  return prepared_sources


def validate_prepared_chunk_ranges(
  *,
  source_file: str,
  source_duration: float,
  overlap: float,
  chunks: list[PreparedChunk],
) -> None:
  """
  Ensure prepared chunks reconstruct the source timeline with the declared duration and overlap.
  
  Validates that:
  - the first chunk starts at 0.0 (within 0.001s),
  - the last chunk ends at `source_duration` (within 0.001s),
  - each subsequent chunk's start matches the previous chunk's end minus `overlap` (within 0.001s).
  
  Parameters:
      source_file (str): Source filename (used in error messages).
      source_duration (float): Declared duration of the source in seconds.
      overlap (float): Declared overlap in seconds between adjacent chunks.
      chunks (list[PreparedChunk]): Chunk objects that should cover the source when stitched; each must have `index`, `start_seconds`, and `end_seconds`.
  
  Raises:
      ValueError: If the chunks do not start at 0, do not end at `source_duration`, or contain a gap/misaligned boundary relative to `overlap`.
  """

  sorted_chunks = sorted(chunks, key=lambda chunk: chunk.index)
  if not math.isclose(sorted_chunks[0].start_seconds, 0.0, rel_tol=0.0, abs_tol=0.001):
    raise ValueError(f"Prepared manifest chunk ranges for {source_file} must start at 0")
  if not math.isclose(sorted_chunks[-1].end_seconds, source_duration, rel_tol=0.0, abs_tol=0.001):
    raise ValueError(f"Prepared manifest chunk ranges for {source_file} must end at source duration")
  expected_next_start = sorted_chunks[0].end_seconds - overlap
  for chunk in sorted_chunks[1:]:
    if not math.isclose(chunk.start_seconds, expected_next_start, rel_tol=0.0, abs_tol=0.001):
      raise ValueError(f"Prepared manifest chunk range gap for {source_file}")
    expected_next_start = chunk.end_seconds - overlap


def require_manifest_string(row: dict[str, Any], key: str) -> str:
  """
  Validate and return a non-empty string field from a prepared manifest row.
  
  Parameters:
      row (dict[str, Any]): A single manifest row (JSON object) to validate.
      key (str): The required field name to extract from the row.
  
  Returns:
      The value of `row[key]` as a non-empty string.
  
  Raises:
      ValueError: If the key is missing, not a string, or an empty string.
  """
  value = row.get(key)
  if not isinstance(value, str) or not value:
    raise ValueError(f"Prepared manifest row requires string {key}")
  return value


def require_manifest_filename(row: dict[str, Any], key: str) -> str:
  """
  Validate and return a manifest field as a plain filename.
  
  Parameters:
      row (dict[str, Any]): A single manifest row mapping field names to values.
      key (str): The key in `row` whose value should be validated as a plain filename.
  
  Returns:
      str: The validated filename string.
  
  Raises:
      ValueError: If the value is missing, not a string, or is not a plain filename
          (absolute paths, path anchors, directory traversals like `..`, any of
          '/', '\\', ':' characters, or the special values '.' or '..').
  """
  value = require_manifest_string(row, key)
  if (
    Path(value).is_absolute()
    or Path(value).anchor
    or ":" in value
    or "/" in value
    or "\\" in value
    or value
    in {
      ".",
      "..",
    }
  ):
    raise ValueError(f"Prepared manifest row requires plain filename {key}")
  return value


def require_manifest_stem(row: dict[str, Any], key: str) -> str:
  """
  Validate and return a plain filename stem extracted from a manifest row.
  
  Parameters:
      row (dict[str, Any]): A manifest row dictionary.
      key (str): The key in `row` whose value should be a plain filename stem.
  
  Returns:
      str: The validated filename stem.
  
  Raises:
      ValueError: If the value is not a plain filename stem (absolute paths, path anchors,
                  path separators, colon characters, or special dot values are rejected).
  """
  value = require_manifest_string(row, key)
  if (
    Path(value).is_absolute()
    or Path(value).anchor
    or ":" in value
    or "/" in value
    or "\\" in value
    or value
    in {
      ".",
      "..",
    }
  ):
    raise ValueError(f"Prepared manifest row requires plain filename stem {key}")
  return value


def require_manifest_number(row: dict[str, Any], key: str) -> float:
  """
  Validate and return a numeric field from a manifest row.
  
  Parameters:
      row (dict[str, Any]): A manifest row dictionary to validate.
      key (str): The key whose value must be a finite number.
  
  Returns:
      float: The value converted to a Python float.
  
  Raises:
      ValueError: If the key is missing or its value is not a finite numeric (int/float, excluding booleans).
  """
  value = row.get(key)
  if not isinstance(value, (int, float)) or isinstance(value, bool):
    raise ValueError(f"Prepared manifest row requires numeric {key}")
  if not math.isfinite(value):
    raise ValueError(f"Prepared manifest row requires finite numeric {key}")
  return float(value)


def require_manifest_integer(row: dict[str, Any], key: str) -> int:
  """
  Validate and return an integer field from a manifest row.
  
  Parameters:
      row (dict[str, Any]): Manifest row dictionary to read the field from.
      key (str): Name of the required integer field.
  
  Returns:
      int: The integer value stored under `key`.
  
  Raises:
      ValueError: If the field is missing, not an integer, or is a boolean.
  """
  value = row.get(key)
  if not isinstance(value, int) or isinstance(value, bool):
    raise ValueError(f"Prepared manifest row requires integer {key}")
  return value


def resolve_output_dir(args: argparse.Namespace, audio_dir: Path) -> Path:
  """
  Resolve the filesystem path to the output directory for the current run mode.
  
  Parameters:
      args (argparse.Namespace): Parsed CLI arguments; `args.mode` selects either "ground" or "eval".
      audio_dir (Path): Directory containing input audio (and the `ground` subdirectory when evaluating).
  
  Returns:
      Path: The directory to use for output artifacts. For `mode == "ground"` this is `audio_dir/ground`; otherwise it is a new eval directory named with the resolved model and a timestamp.
  
  Raises:
      ValueError: If `args.mode` is not "ground" and the `audio_dir/ground` directory does not exist.
  """

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
  """
  Process a list of audio inputs with bounded concurrency and return per-file results in the original input order.
  
  Processes each item in `audio_files` using up to `args.batch` worker threads, skipping inputs that are already completed when operating in ground-generation mode; prints progress lines as each file is skipped or processed.
  
  Parameters:
      args (argparse.Namespace): Runtime options (must include `mode` and `batch`) that control skipping behavior and concurrency.
      audio_files (list[AudioInput]): Ordered list of audio inputs to process.
      output_dir (Path): Directory where per-file artifacts will be written.
      base_url (str): Base URL for the endpoint used to transcribe audio.
      api_key (str | None): Optional API key to authenticate requests.
  
  Returns:
      list[FileResult]: Per-file results corresponding one-for-one with `audio_files`, preserved in the same order.
  """

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
  """
  Transcribe pre-chunked audio sources in parallel, write per-chunk transcript files, and stitch per-source exact and normalized transcripts.
  
  For each PreparedSource this function:
  - Writes per-chunk transcripts to output_dir/chunks/{chunk_stem}.txt.
  - Produces stitched artifacts {stem}.txt and {stem}_normalized.txt (or a failed FileResult on errors).
  - Preserves skip behavior for sources that already have both stitched artifacts.
  - Records elapsed processing time and sets `chunk_count` on each returned FileResult.
  
  Returns:
      list[FileResult]: FileResult for each PreparedSource in the same order as `prepared_sources`.
  """

  chunks_dir_error = prepare_prepared_chunks_output_dir(output_dir)
  if chunks_dir_error is not None:
    return [
      build_failed_file_result(
        audio_file=source.audio,
        output_path=output_dir / f"{source.audio.stem}.txt",
        normalized_output_path=output_dir / f"{source.audio.stem}_normalized.txt",
        elapsed_seconds=None,
        error_message=chunks_dir_error,
        duration_seconds=source.duration_seconds,
        chunk_count=len(source.chunks),
      )
      for source in prepared_sources
    ]
  results_by_source: dict[str, FileResult] = {}
  total_chunks = sum(len(source.chunks) for source in prepared_sources)
  chunk_transcripts: dict[str, dict[int, str]] = {source.audio.path.name: {} for source in prepared_sources}
  chunk_errors: dict[str, list[str]] = {source.audio.path.name: [] for source in prepared_sources}
  started_by_source: dict[str, float] = {}
  finished_by_source: dict[str, float] = {}
  remaining_by_source: dict[str, int] = {source.audio.path.name: len(source.chunks) for source in prepared_sources}
  timing_lock = threading.Lock()

  with tqdm(total=total_chunks, unit="chunk") as progress:
    with ThreadPoolExecutor(max_workers=args.batch) as executor:
      futures: dict[Future[str], PreparedChunk] = {}
      chunks_to_submit: list[PreparedChunk] = []
      for source in prepared_sources:
        skipped_result = maybe_skip_prepared_ground_file(args, source, output_dir)
        if skipped_result is not None:
          results_by_source[source.audio.path.name] = skipped_result
          progress.update(len(source.chunks))
          tqdm.write(format_file_result(skipped_result, mode=args.mode))
          continue
        chunks_to_submit.extend(source.chunks)

      next_chunk_index = 0

      def submit_next_chunk() -> None:
        """
        Submit the next prepared chunk to the executor for transcription if any remain.
        
        Schedules the next chunk from `chunks_to_submit` on the thread pool, increments
        the local `next_chunk_index`, and records the returned future in `futures`
        mapped to its corresponding chunk.
        """
        nonlocal next_chunk_index
        if next_chunk_index >= len(chunks_to_submit):
          return
        chunk = chunks_to_submit[next_chunk_index]
        next_chunk_index += 1
        futures[
          executor.submit(
            transcribe_prepared_chunk,
            args=args,
            chunk=chunk,
            base_url=base_url,
            api_key=api_key,
            started_by_source=started_by_source,
            timing_lock=timing_lock,
          )
        ] = chunk

      for _ in range(min(args.batch, len(chunks_to_submit))):
        submit_next_chunk()

      while futures:
        done, _ = wait(futures, return_when=FIRST_COMPLETED)
        for future in done:
          chunk = futures.pop(future)
          try:
            transcript = future.result()
            chunk_transcripts[chunk.source.path.name][chunk.index] = transcript
            chunk_output = output_dir / "chunks" / f"{chunk.audio.stem}.txt"
            atomic_write_text(chunk_output, transcript)
          except Exception as exc:
            chunk_errors[chunk.source.path.name].append(str(exc))
          remaining_by_source[chunk.source.path.name] -= 1
          if remaining_by_source[chunk.source.path.name] == 0:
            finished_by_source[chunk.source.path.name] = time.perf_counter()
          progress.update(1)
          submit_next_chunk()

  for source in prepared_sources:
    if source.audio.path.name in results_by_source:
      continue
    started_at = started_by_source[source.audio.path.name]
    finished_at = finished_by_source[source.audio.path.name]
    elapsed = max(finished_at - started_at, 0.0)
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


def transcribe_prepared_chunk(
  *,
  args: argparse.Namespace,
  chunk: PreparedChunk,
  base_url: str,
  api_key: str | None,
  started_by_source: dict[str, float],
  timing_lock: threading.Lock,
) -> str:
  """
  Transcribes a single prepared chunk and records the source's start time when processing begins.
  
  If the source has not yet been marked, sets started_by_source[chunk.source.path.name] to the current perf_counter under timing_lock.
  
  Parameters:
      chunk (PreparedChunk): The prepared chunk to transcribe.
      started_by_source (dict[str, float]): Mapping updated with the source start time if not already present.
      timing_lock (threading.Lock): Lock used to synchronize access to started_by_source.
  
  Returns:
      str: The transcript text for the chunk.
  """

  with timing_lock:
    if chunk.source.path.name not in started_by_source:
      started_by_source[chunk.source.path.name] = time.perf_counter()
  return transcribe_with_selected_endpoint(args=args, audio_file=chunk.audio, base_url=base_url, api_key=api_key)


def prepare_prepared_chunks_output_dir(output_dir: Path) -> str | None:
  """
  Ensure the output directory's "chunks" subdirectory exists, creating it if necessary.
  
  Returns:
      None if the "chunks" directory was created or already exists; otherwise a string with a human-readable error message describing the failure.
  """

  chunks_dir = output_dir / "chunks"
  if chunks_dir.exists() and not chunks_dir.is_dir():
    return f"Prepared chunks output path is not a directory: {chunks_dir}"
  try:
    chunks_dir.mkdir(parents=True, exist_ok=True)
  except OSError as exc:
    return f"Unable to create prepared chunks output directory {chunks_dir}: {exc}"
  return None


def maybe_skip_prepared_ground_file(
  args: argparse.Namespace,
  source: PreparedSource,
  output_dir: Path,
) -> FileResult | None:
  """
  Return a skipped FileResult when both exact and normalized stitched artifacts already exist for a prepared source.
  
  Parameters:
      args (argparse.Namespace): Parsed CLI/runtime arguments; skipping only applies when `args.mode == "ground"`.
      source (PreparedSource): Prepared source metadata whose stitched artifacts would be checked.
      output_dir (Path): Directory containing per-source output artifacts.
  
  Returns:
      FileResult: A result with `status == "skipped"` when both `{stem}.txt` and `{stem}_normalized.txt` exist and were read successfully.
      FileResult: A failed result if both artifacts exist but cannot be read.
      None: If skipping is not applicable (not in ground mode or one or both artifacts are missing).
  """

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
      duration_seconds=source.duration_seconds,
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
  """
  Build the final FileResult for a prepared source by stitching chunk transcripts, writing per-source artifacts, and optionally adding evaluation scores.
  
  If any chunk transcripts are missing or there are recorded chunk errors, returns a failed FileResult describing the problems. Otherwise stitches ordered chunk transcripts into a single exact transcript and a normalized transcript (deduplicating overlap), atomically writes both output files under output_dir, and returns a successful FileResult with timing and word-count metadata. If args.mode == "eval", attempts to attach WER scores; scoring exceptions are converted into a failed FileResult that preserves chunk_count and other fields.
  
  Parameters:
      args (argparse.Namespace): Parsed CLI arguments; used only to determine eval mode.
      source (PreparedSource): The prepared source being reconstructed and written.
      output_dir (Path): Directory where per-source artifacts ({stem}.txt and {stem}_normalized.txt) will be written.
      chunk_transcripts (dict[int, str]): Mapping from chunk index to that chunk's transcript.
      chunk_errors (list[str]): Collected error messages from chunk transcription workers.
      elapsed_seconds (float): Wall elapsed seconds spent transcribing this source (may be 0).
  
  Returns:
      FileResult: A FileResult representing the per-source outcome. On success, status is "transcribed" and output paths/word counts/rtfx are populated; on failure, status is "failed" and error_message explains the reason. The returned FileResult's chunk_count is set to the number of chunks for the source.
  """
  exact_path = output_dir / f"{source.audio.stem}.txt"
  normalized_path = output_dir / f"{source.audio.stem}_normalized.txt"
  if chunk_errors or len(chunk_transcripts) != len(source.chunks):
    missing = sorted({chunk.index for chunk in source.chunks} - set(chunk_transcripts))
    errors = [*chunk_errors]
    if missing:
      errors.append(f"missing chunk transcripts: {', '.join(str(index) for index in missing)}")
    if not errors:
      errors.append("chunk transcript count mismatch")
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
  """
  Stitches normalized transcript chunks into a single normalized transcript by removing duplicated tokens across chunk boundaries.
  
  Parameters:
      normalized_chunks (list[str]): Ordered normalized transcript text for each chunk (tokens separated by whitespace).
      overlap_seconds (float): Overlap duration in seconds that was used when creating chunks; used to bound how many tokens may be duplicated at chunk edges.
  
  Returns:
      str: A single normalized transcript formed by joining chunks with duplicated overlap tokens removed, separated by single spaces.
  """
  stitched: list[str] = []
  max_overlap_tokens = (
    0 if overlap_seconds == 0 else math.ceil(OVERLAP_TOKEN_RATE_PER_SECOND * overlap_seconds) + OVERLAP_TOKEN_BUFFER
  )
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
  """
  Checks whether an existing ground transcript can be reused and, if so, returns a corresponding skipped FileResult.
  
  If args.mode == "ground" and an exact transcript file for the audio exists in output_dir, this function reads the exact transcript, ensures a normalized transcript exists (writing one atomically if missing), reads the audio duration, and returns a FileResult with status "skipped" and populated transcript/word-count/timing fields. If the conditions to skip are not met, returns None. If reading/writing files or obtaining duration fails, returns a failed FileResult describing the error.
   
  Returns:
      FileResult | None: A `FileResult` with status `"skipped"` when an existing ground transcript is reused; `None` when no skip applies. On I/O or duration errors, returns a `FileResult` with status `"failed"` describing the error.
  """

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
  """
  Transcribe a single audio input, write exact and normalized transcript files, and optionally compute WER scores.
  
  This function:
  - Obtains the audio duration, requests a transcript from the selected endpoint, normalizes the transcript, and atomically writes two files into `output_dir`: `{stem}.txt` and `{stem}_normalized.txt`.
  - Builds and returns a FileResult capturing timing, word counts, output paths, and any error text.
  - If `args.mode == "eval"`, attempts to attach WER fields by calling `add_eval_scores`; on scoring failure it returns a `failed` FileResult containing the scoring error.
  
  Parameters:
      args (argparse.Namespace): Parsed CLI arguments (controls mode, endpoint selection, and other request configuration).
      audio_file (AudioInput): Source audio metadata (path, stem, format).
      output_dir (Path): Directory where transcript artifacts will be written.
      base_url (str): Base URL for the API endpoint used for transcription.
      api_key (str | None): Optional API key for the transcription endpoint.
  
  Returns:
      FileResult: Result record for the audio file including status (`transcribed` or `failed`), paths to written artifacts, timing (`elapsed_seconds`, `duration_seconds`, `rtfx`), word counts, and any evaluation fields when in eval mode.
  """

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
  """
  Builds a FileResult representing a failed transcription for a given audio file.
  
  Parameters:
    audio_file (AudioInput): Source audio metadata for the failed file.
    output_path (Path): Intended path for the exact transcript file.
    normalized_output_path (Path): Intended path for the normalized transcript file.
    elapsed_seconds (float | None): Wall-clock seconds spent before failure, or None if unknown.
    error_message (str): Human-readable error description to store on the result.
    duration_seconds (float): Known duration of the source audio in seconds (default 0.0).
    chunk_count (int | None): Number of prep chunks for this source when running in prep mode, or None.
  
  Returns:
    FileResult: A FileResult with status set to `"failed"`, empty transcript fields, word counts set to 0, and provided timing, paths, and error metadata.
  """

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
  """
  Produce a per-file completions request arguments namespace with defaults and prep-specific overrides.
  
  Parameters:
      args (argparse.Namespace): Batch-level CLI arguments used to derive per-file completions request values.
  
  Returns:
      argparse.Namespace: A copy of `args` with defaults applied and mappings for completions requests:
          - ensures `system_prompt`, `developer_prompt`, and `user_prompt` have defaults when unset
          - maps `prompt` to `developer_prompt` when provided
          - maps `service_tier` to `completions_service_tier` when set
          - forces `completions_temperature` to `0.0` when `prep` is enabled and temperature is unset
  """

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
  """
  Prepare a copy of CLI arguments adjusted for transcription requests.
  
  Parameters:
      args (argparse.Namespace): Original parsed CLI arguments.
  
  Returns:
      argparse.Namespace: A shallow copy of `args` with `transcriptions_model` set from endpoint resolution, `model` cleared, `transcriptions_prompt` populated from `prompt` when present, and `transcriptions_temperature` forced to 0.0 when `prep` is enabled and the transcription temperature was not explicitly set.
  """

  request_args = argparse.Namespace(**vars(args))
  request_args.transcriptions_model = resolve_endpoint_model(args)
  request_args.model = None
  if request_args.prompt is not None:
    request_args.transcriptions_prompt = request_args.prompt
  if getattr(request_args, "prep", False) and request_args.transcriptions_temperature is None:
    request_args.transcriptions_temperature = 0.0
  return request_args


def add_eval_scores(result: FileResult, *, audio_file: AudioInput) -> FileResult:
  """
  Add word-error-rate (WER) fields to a FileResult by comparing its normalized transcript to the ground normalized transcript on disk.
  
  Reads the reference from the file at `audio_file.path.parent/ground/{audio_file.stem}_normalized.txt`, computes plain-word WER against `result.normalized_transcript`, and returns a new FileResult preserving all original fields but with `reference_word_count`, `wer_errors`, `wer_reference_words`, and `wer` populated.
  
  Parameters:
      result (FileResult): The per-file result whose normalized transcript will be scored.
      audio_file (AudioInput): Source audio descriptor used to locate the ground reference file.
  
  Returns:
      FileResult: A copy of `result` with WER-related fields filled (`reference_word_count`, `wer_errors`, `wer_reference_words`, `wer`) and other fields preserved.
  """

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
  """
  Determine the service tier to record in the run report.
  
  Prefers `args.service_tier` if set; if unset and `args.endpoint == "completions"`, uses `args.completions_service_tier` if set; otherwise yields `"none"`.
  
  Returns:
      The chosen service tier as a string, or `"none"` if no tier is specified.
  """

  if args.service_tier is not None:
    return str(args.service_tier)
  if args.endpoint == "completions" and args.completions_service_tier is not None:
    return str(args.completions_service_tier)
  return "none"


def resolve_report_temperature(args: argparse.Namespace) -> str:
  """
  Resolve the effective temperature value to record in the run report.
  
  Considers the selected endpoint and explicit temperature overrides; when no explicit temperature is set, returns "0.0" if prep mode is enabled, otherwise "provider_default".
  
  Returns:
      str: The effective temperature as a string — an explicit numeric temperature if provided, otherwise "0.0" or "provider_default".
  """

  if args.endpoint == "completions":
    if args.completions_temperature is not None:
      return str(args.completions_temperature)
    return "0.0" if getattr(args, "prep", False) else "provider_default"
  if args.transcriptions_temperature is not None:
    return str(args.transcriptions_temperature)
  return "0.0" if getattr(args, "prep", False) else "provider_default"


def resolve_report_prep_overlap(args: argparse.Namespace, results: list[FileResult]) -> str:
  """
  Determine the printable prep overlap value used in report metadata.
  
  Parameters:
      args: CLI arguments; checks `args.prep`, `args._prep_overlap_seconds`, and `args.overlap`.
      results: Ignored (present for call-site compatibility).
  
  Returns:
      A string: `"none"` if prep is not enabled; otherwise the overlap seconds as a string, or `"unknown"` if no overlap is available.
  """
  del results
  if not getattr(args, "prep", False):
    return "none"
  return str(getattr(args, "_prep_overlap_seconds", args.overlap if args.overlap is not None else "unknown"))


def resolve_report_prep_segment_duration(args: argparse.Namespace, results: list[FileResult]) -> str:
  """
  Resolve the prep segment duration value to display in the run report.
  
  If prep mode is disabled on args, returns the literal string "none".
  If prep mode is enabled, returns the segment duration seconds from
  args._prep_segment_duration_seconds converted to a string (defaults to 30.0).
  
  Parameters:
      args (argparse.Namespace): Parsed CLI arguments; may contain `prep` (bool)
          and `_prep_segment_duration_seconds` (float).
      results (list[FileResult]): Ignored; present for call-site compatibility.
  
  Returns:
      str: `"none"` when prep is not enabled, otherwise the segment duration in seconds as a string.
  """
  del results
  if not getattr(args, "prep", False):
    return "none"
  return str(getattr(args, "_prep_segment_duration_seconds", 30.0))


def has_report_prompt(args: argparse.Namespace) -> bool:
  """
  Determine if any prompt-related option was provided for the run.
  
  Checks `args.prompt` always; if `args.endpoint == "transcriptions"` checks `args.transcriptions_prompt`; otherwise checks `args.system_prompt`, `args.developer_prompt`, and `args.user_prompt`.
  
  Returns:
      `true` if any prompt option was set, `false` otherwise.
  """

  if args.prompt is not None:
    return True
  if args.endpoint == "transcriptions":
    return args.transcriptions_prompt is not None
  return any(getattr(args, name) is not None for name in ("system_prompt", "developer_prompt", "user_prompt"))


def render_report_header(*, eval_mode: bool) -> str:
  """
  Render the TSV report header row with columns appropriate for evaluation mode.
  
  Parameters:
      eval_mode (bool): If True, include evaluation-specific columns (`reference_words`, `WER`, `wer_errors`, `wer_reference_words`).
  
  Returns:
      str: Tab-separated header row.
  """

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
  """
  Render a tab-separated values (TSV) row summarizing a FileResult.
  
  Parameters:
      result (FileResult): The per-file result record to render.
      eval_mode (bool): If True, append WER-related columns (reference word count, WER percentage,
          WER errors, WER reference words).
  
  Returns:
      str: A single TSV row string with these columns in order:
          filename, status, chunk_count (empty if None), elapsed_seconds (empty if None),
          duration_seconds, rtfx (empty if None), exact_word_count, normalized_word_count,
          (when eval_mode is True) reference_word_count, wer (formatted as a percentage, empty if None),
          wer_errors, wer_reference_words, output_path, error_message (empty if None).
  """

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
