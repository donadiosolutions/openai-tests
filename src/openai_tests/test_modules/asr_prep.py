"""Deterministic audio segmentation for prepared ASR WER runs."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mutagen

from ..core import EndpointTestModule
from . import asr_simple

SEGMENT_DURATION_SECONDS = 30.0
DEFAULT_OVERLAP_SECONDS = 3.0


@dataclass(frozen=True, slots=True)
class AudioInput:
  path: Path
  stem: str
  format: str


@dataclass(frozen=True, slots=True)
class Segment:
  source_path: Path
  source_file: str
  source_stem: str
  chunk_path: Path
  index: int
  start_seconds: float
  end_seconds: float
  duration_seconds: float


def configure_parser(parser: argparse.ArgumentParser) -> None:
  parser.add_argument("audio_dir", help="Directory containing supported direct-child audio files.")
  parser.add_argument(
    "--overlap",
    type=float,
    default=DEFAULT_OVERLAP_SECONDS,
    help="Seconds of overlap between 30-second chunks. Defaults to 3.0.",
  )


def run(args: argparse.Namespace) -> int:
  try:
    overlap = validate_overlap(args.overlap)
    audio_dir = Path(args.audio_dir)
    audio_files = discover_audio_files(audio_dir)
    prep_dir = prepare_output_dir(audio_dir)
    all_segments: list[Segment] = []
    source_rows: list[dict[str, Any]] = []
    for audio_file in audio_files:
      try:
        duration = get_audio_duration_seconds(audio_file.path)
      except Exception as exc:
        raise ValueError(f"Unable to read audio duration for {audio_file.path.name}: {exc}") from exc
      segments = plan_segments(audio_file, duration_seconds=duration, overlap_seconds=overlap, output_dir=prep_dir)
      for segment in segments:
        run_ffmpeg_segment(segment)
      all_segments.extend(segments)
      source_rows.append(
        {
          "source_file": audio_file.path.name,
          "duration_seconds": round(duration, 3),
          "chunk_count": len(segments),
        }
      )
    write_manifest(prep_dir, sources=source_rows, segments=all_segments, overlap_seconds=overlap)
    write_report(prep_dir, sources=source_rows, segments=all_segments, overlap_seconds=overlap)
  except ValueError as exc:
    print(f"Configuration error: {exc}", file=sys.stderr)
    return 2
  print(f"Wrote {len(all_segments)} chunks to {prep_dir}")
  return 0


def validate_overlap(overlap: float) -> float:
  if not math.isfinite(overlap):
    raise ValueError("overlap must be finite")
  if overlap < 0:
    raise ValueError("overlap must be at least 0 seconds")
  if overlap >= SEGMENT_DURATION_SECONDS:
    raise ValueError("overlap must be less than 30 seconds")
  return overlap


def discover_audio_files(audio_dir: Path) -> list[AudioInput]:
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
  validate_audio_inputs(audio_files)
  return audio_files


def validate_audio_inputs(audio_files: list[AudioInput]) -> None:
  """Reject input sets that would generate ambiguous prepared artifacts."""

  stems: dict[str, Path] = {}
  for audio_file in audio_files:
    validate_plain_filename(audio_file.path.name, "source filename")
    validate_plain_filename(audio_file.stem, "source filename stem")
    stem_key = audio_file.stem.casefold()
    if stem_key in stems:
      raise ValueError(
        f"Duplicate audio file stem {audio_file.stem!r}: {stems[stem_key].name} and {audio_file.path.name}"
      )
    stems[stem_key] = audio_file.path
  validate_output_artifact_names(audio_files)


def validate_plain_filename(value: str, kind: str) -> None:
  if (
    not value
    or Path(value).is_absolute()
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
    raise ValueError(f"Audio input {kind} must be a plain filename")


def validate_output_artifact_names(audio_files: list[AudioInput]) -> None:
  """Reject source stems that asr-wer cannot consume after preparation."""

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


def prepare_output_dir(audio_dir: Path) -> Path:
  prep_dir = audio_dir / "prep"
  if prep_dir.exists() and not prep_dir.is_dir():
    raise ValueError(f"Prep output path is not a directory: {prep_dir}")
  try:
    prep_dir.mkdir(parents=True, exist_ok=True)
  except OSError as exc:
    raise ValueError(f"Unable to create prep output directory {prep_dir}: {exc}") from exc
  try:
    if any(prep_dir.iterdir()):
      raise ValueError(f"Prep output directory already exists and is not empty: {prep_dir}")
  except OSError as exc:
    raise ValueError(f"Unable to list prep output directory {prep_dir}: {exc}") from exc
  return prep_dir


def plan_segments(
  audio_file: AudioInput,
  *,
  duration_seconds: float,
  overlap_seconds: float,
  output_dir: Path,
) -> list[Segment]:
  step = SEGMENT_DURATION_SECONDS - overlap_seconds
  starts: list[float] = [0.0]
  while starts[-1] + SEGMENT_DURATION_SECONDS < duration_seconds:
    starts.append(starts[-1] + step)
  segments: list[Segment] = []
  for index, start in enumerate(starts):
    end = min(start + SEGMENT_DURATION_SECONDS, duration_seconds)
    start_seconds = round(start, 3)
    end_seconds = round(end, 3)
    segment_duration_seconds = round(max(end_seconds - start_seconds, 0.0), 3)
    if segment_duration_seconds <= 0:
      continue
    start_ms = seconds_to_milliseconds(start_seconds)
    end_ms = seconds_to_milliseconds(end_seconds)
    chunk_path = output_dir / f"{audio_file.stem}_{index:04d}_{start_ms:06d}_{end_ms:06d}.wav"
    segments.append(
      Segment(
        source_path=audio_file.path,
        source_file=audio_file.path.name,
        source_stem=audio_file.stem,
        chunk_path=chunk_path,
        index=index,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        duration_seconds=segment_duration_seconds,
      )
    )
  return segments


def seconds_to_milliseconds(seconds: float) -> int:
  return round(seconds * 1000)


def run_ffmpeg_segment(segment: Segment) -> None:
  command = [
    "ffmpeg",
    "-hide_banner",
    "-nostdin",
    "-y",
    "-ss",
    f"{segment.start_seconds:.3f}",
    "-i",
    str(segment.source_path),
    "-t",
    f"{segment.duration_seconds:.3f}",
    "-vn",
    "-acodec",
    "pcm_s16le",
    str(segment.chunk_path),
  ]
  try:
    subprocess.run(command, check=True, capture_output=True, text=True)
  except FileNotFoundError as exc:
    raise ValueError("ffmpeg was not found; install it before running asr-prep") from exc
  except subprocess.CalledProcessError as exc:
    stderr = exc.stderr.strip() if isinstance(exc.stderr, str) else str(exc.stderr)
    raise ValueError(f"ffmpeg failed for {segment.source_file} chunk {segment.index}: {stderr}") from exc


def write_manifest(
  prep_dir: Path,
  *,
  sources: list[dict[str, Any]],
  segments: list[Segment],
  overlap_seconds: float,
) -> None:
  manifest = {
    "tool": "openai-tests asr-prep",
    "segment_duration_seconds": SEGMENT_DURATION_SECONDS,
    "overlap_seconds": overlap_seconds,
    "sources": sources,
    "chunks": [segment_to_manifest_row(segment) for segment in segments],
  }
  (prep_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def segment_to_manifest_row(segment: Segment) -> dict[str, Any]:
  row = asdict(segment)
  row.pop("source_path")
  row["chunk_file"] = segment.chunk_path.name
  row.pop("chunk_path")
  row["chunk_index"] = row.pop("index")
  return row


def write_report(
  prep_dir: Path,
  *,
  sources: list[dict[str, Any]],
  segments: list[Segment],
  overlap_seconds: float,
) -> None:
  lines = [
    "asr-prep report",
    f"segment_duration_seconds: {SEGMENT_DURATION_SECONDS}",
    f"overlap_seconds: {overlap_seconds}",
    f"source_files: {len(sources)}",
    f"chunks: {len(segments)}",
    "",
    "source_file\tchunk_file\tchunk_index\tstart_seconds\tend_seconds\tduration_seconds",
  ]
  for segment in segments:
    lines.append(
      "\t".join(
        (
          segment.source_file,
          segment.chunk_path.name,
          str(segment.index),
          f"{segment.start_seconds:.3f}",
          f"{segment.end_seconds:.3f}",
          f"{segment.duration_seconds:.3f}",
        )
      )
    )
  (prep_dir / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def get_audio_duration_seconds(audio_path: Path) -> float:
  audio = mutagen.File(audio_path)
  length = getattr(getattr(audio, "info", None), "length", None)
  if isinstance(length, (int, float)) and not isinstance(length, bool) and math.isfinite(length) and length > 0:
    return float(length)
  raise ValueError(f"Unable to determine audio duration for {audio_path}")


ASR_PREP_MODULE = EndpointTestModule(
  name="asr-prep",
  summary="Segment audio folders into deterministic chunks for prepared ASR WER runs.",
  configure_parser=configure_parser,
  handler=run,
)
