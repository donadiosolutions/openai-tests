"""Tests for deterministic ASR preparation and manifest generation."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from openai_tests.test_modules import asr_prep


def build_args(*raw_args: str) -> argparse.Namespace:
  """Build parsed asr-prep arguments from raw CLI tokens."""

  parser = argparse.ArgumentParser()
  asr_prep.configure_parser(parser)
  return parser.parse_args([*raw_args])


def write_audio(path: Path) -> Path:
  """Write a minimal RIFF-like audio fixture and return its path."""

  path.write_bytes(b"RIFF")
  return path


def test_parser_accepts_audio_dir_and_overlap(tmp_path: Path) -> None:
  """The parser accepts the audio directory and custom overlap."""

  args = build_args(str(tmp_path), "--overlap", "1.5")

  assert args.audio_dir == str(tmp_path)
  assert args.overlap == 1.5


def test_configuration_errors_cover_input_overlap_prep_and_ffmpeg(
  capsys: pytest.CaptureFixture[str],
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  """Configuration failures return clear validation errors."""

  with pytest.raises(ValueError, match="overlap must be at least 0"):
    asr_prep.validate_overlap(-0.1)
  with pytest.raises(ValueError, match="overlap must be less than 30"):
    asr_prep.validate_overlap(30.0)
  with pytest.raises(ValueError, match="overlap must be less than 30"):
    asr_prep.validate_overlap(29.9999)
  with pytest.raises(ValueError, match="overlap must be finite"):
    asr_prep.validate_overlap(float("nan"))
  assert asr_prep.validate_overlap(1.2345) == 1.234

  assert asr_prep.run(build_args(str(tmp_path / "missing"))) == 2
  assert "Audio directory does not exist" in capsys.readouterr().err

  not_dir = write_audio(tmp_path / "not-dir.wav")
  assert asr_prep.run(build_args(str(not_dir))) == 2
  assert "Audio path is not a directory" in capsys.readouterr().err

  empty = tmp_path / "empty"
  empty.mkdir()
  assert asr_prep.run(build_args(str(empty))) == 2
  assert "No supported audio files" in capsys.readouterr().err

  duplicate_dir = tmp_path / "dupes"
  duplicate_dir.mkdir()
  write_audio(duplicate_dir / "call.wav")
  write_audio(duplicate_dir / "call.mp3")
  assert asr_prep.run(build_args(str(duplicate_dir))) == 2
  assert "Duplicate audio file stem" in capsys.readouterr().err

  collision_dir = tmp_path / "collisions"
  collision_dir.mkdir()
  write_audio(collision_dir / "call.wav")
  write_audio(collision_dir / "call_normalized.wav")
  assert asr_prep.run(build_args(str(collision_dir))) == 2
  assert "Output artifact collision" in capsys.readouterr().err

  reserved_dir = tmp_path / "reserved"
  reserved_dir.mkdir()
  write_audio(reserved_dir / "report.wav")
  assert asr_prep.run(build_args(str(reserved_dir))) == 2
  assert "reserved output artifact" in capsys.readouterr().err

  audio_dir = tmp_path / "audio"
  audio_dir.mkdir()
  write_audio(audio_dir / "clip.wav")
  prep_dir = audio_dir / "prep"
  prep_dir.mkdir()
  (prep_dir / "old.wav").write_bytes(b"old")
  assert asr_prep.run(build_args(str(audio_dir))) == 2
  assert "already exists and is not empty" in capsys.readouterr().err

  (prep_dir / "old.wav").unlink()
  prep_dir.rmdir()

  symlink_target = tmp_path / "prep-target"
  symlink_target.mkdir()
  prep_dir.symlink_to(symlink_target, target_is_directory=True)
  assert asr_prep.run(build_args(str(audio_dir))) == 2
  assert "Prep output path must not be a symlink" in capsys.readouterr().err
  prep_dir.unlink()

  def missing_ffmpeg(*_: object, **__: object) -> subprocess.CompletedProcess[str]:
    """Simulate an environment without ffmpeg on PATH."""

    raise FileNotFoundError("ffmpeg")

  monkeypatch.setattr(asr_prep, "get_audio_duration_seconds", lambda path: 1.0)
  monkeypatch.setattr(asr_prep.subprocess, "run", missing_ffmpeg)
  assert asr_prep.run(build_args(str(audio_dir))) == 2
  assert "ffmpeg was not found" in capsys.readouterr().err


def test_failed_segmentation_cleans_staged_prep_output(
  capsys: pytest.CaptureFixture[str],
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  """Failed staged segmentation removes temporary output and leaves final prep absent."""

  audio_dir = tmp_path / "audio-cleanup"
  audio_dir.mkdir()
  write_audio(audio_dir / "clip.wav")

  def fail_ffmpeg(*_: object, **__: object) -> subprocess.CompletedProcess[str]:
    """Simulate a transient ffmpeg failure."""

    raise subprocess.CalledProcessError(1, ["ffmpeg"], stderr="transient")

  monkeypatch.setattr(asr_prep, "get_audio_duration_seconds", lambda path: 31.0)
  monkeypatch.setattr(asr_prep.subprocess, "run", fail_ffmpeg)

  assert asr_prep.run(build_args(str(audio_dir))) == 2
  assert "ffmpeg failed" in capsys.readouterr().err
  assert not (audio_dir / "prep").exists()
  assert not any(path.name.startswith(".prep.") and path.name.endswith(".tmp") for path in audio_dir.iterdir())


def test_run_uses_manifest_rounded_duration_for_segment_planning(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  """Planning uses the same rounded duration that is written to the manifest."""

  audio_dir = tmp_path / "audio-rounded-duration"
  audio_dir.mkdir()
  write_audio(audio_dir / "call.wav")
  commands: list[list[str]] = []

  monkeypatch.setattr(asr_prep, "get_audio_duration_seconds", lambda path: 30.0001)

  def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
    """Record chunk commands for a rounded-boundary source duration."""

    commands.append(command)
    Path(command[-1]).write_bytes(b"RIFF")
    return subprocess.CompletedProcess(command, 0, "", "")

  monkeypatch.setattr(asr_prep.subprocess, "run", fake_run)

  assert asr_prep.run(build_args(str(audio_dir))) == 0

  manifest = json.loads((audio_dir / "prep" / "manifest.json").read_text(encoding="utf-8"))
  assert manifest["sources"] == [{"chunk_count": 1, "duration_seconds": 30.0, "source_file": "call.wav"}]
  assert [chunk["chunk_file"] for chunk in manifest["chunks"]] == ["call_0000_000000_030000.wav"]
  assert len(commands) == 1


def test_staged_prep_output_does_not_delete_existing_temp_like_directory(
  capsys: pytest.CaptureFixture[str],
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  """Existing temp-like folders are not deleted by new prep runs."""

  audio_dir = tmp_path / "audio-existing-temp"
  audio_dir.mkdir()
  write_audio(audio_dir / "clip.wav")
  existing_temp = audio_dir / ".prep.existing.tmp"
  existing_temp.mkdir()
  (existing_temp / "keep.txt").write_text("keep", encoding="utf-8")
  commands: list[list[str]] = []

  monkeypatch.setattr(asr_prep, "get_audio_duration_seconds", lambda path: 1.0)

  def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
    """Record ffmpeg output paths while preserving existing temp-like folders."""

    commands.append(command)
    Path(command[-1]).write_bytes(b"RIFF")
    return subprocess.CompletedProcess(command, 0, "", "")

  monkeypatch.setattr(asr_prep.subprocess, "run", fake_run)

  assert asr_prep.run(build_args(str(audio_dir))) == 0
  assert "Wrote 1 chunks" in capsys.readouterr().out
  assert (existing_temp / "keep.txt").read_text(encoding="utf-8") == "keep"
  assert Path(commands[0][-1]).parent != existing_temp


def test_staged_prep_helpers_report_cleanup_and_finalize_failures(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  """Temporary prep helpers report cleanup and finalization filesystem failures."""

  prep_tmp_dir = tmp_path / ".prep.test.tmp"
  prep_tmp_dir.mkdir()

  def fail_rmtree(path: Path) -> None:
    """Simulate a failure while removing staged prep output."""

    raise PermissionError(f"denied: {path}")

  monkeypatch.setattr(asr_prep.shutil, "rmtree", fail_rmtree)
  with pytest.raises(ValueError, match="Unable to remove temporary prep output directory"):
    asr_prep.cleanup_temp_output_dir(prep_tmp_dir)

  monkeypatch.undo()
  asr_prep.cleanup_temp_output_dir(tmp_path / ".prep.missing.tmp")
  prep_dir = tmp_path / "prep"
  prep_dir.mkdir()
  asr_prep.finalize_output_dir(prep_tmp_dir, prep_dir)
  assert prep_dir.is_dir()
  assert not prep_tmp_dir.exists()

  prep_tmp_dir.mkdir()
  prep_dir.mkdir(exist_ok=True)

  def fail_rename(path: Path, target: Path) -> None:
    """Simulate a failure while moving staged prep output into place."""

    raise PermissionError(f"denied: {path} -> {target}")

  monkeypatch.setattr(Path, "rename", fail_rename)
  with pytest.raises(ValueError, match="Unable to move temporary prep output directory"):
    asr_prep.finalize_output_dir(prep_tmp_dir, tmp_path / "prep-failed")


def test_configuration_errors_cover_duration_listing_and_output_failures(
  capsys: pytest.CaptureFixture[str],
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  """Directory, duration, and prep metadata write failures exit cleanly."""

  audio_dir = tmp_path / "audio"
  audio_dir.mkdir()
  write_audio(audio_dir / "clip.wav")
  original_iterdir = Path.iterdir

  def fail_audio_iterdir(path: Path):
    """Simulate failure while listing the selected audio directory."""

    if path == audio_dir:
      raise PermissionError("denied")
    return original_iterdir(path)

  monkeypatch.setattr(Path, "iterdir", fail_audio_iterdir)
  assert asr_prep.run(build_args(str(audio_dir))) == 2
  assert "Unable to list audio directory" in capsys.readouterr().err
  monkeypatch.setattr(Path, "iterdir", original_iterdir)

  (audio_dir / "prep").write_text("not a directory", encoding="utf-8")
  assert asr_prep.run(build_args(str(audio_dir))) == 2
  assert "Prep output path is not a directory" in capsys.readouterr().err
  (audio_dir / "prep").unlink()

  monkeypatch.setattr(asr_prep.tempfile, "mkdtemp", lambda **kwargs: (_ for _ in ()).throw(PermissionError("denied")))
  assert asr_prep.run(build_args(str(audio_dir))) == 2
  assert "Unable to create temporary prep output directory" in capsys.readouterr().err
  monkeypatch.undo()

  (audio_dir / "prep").mkdir()

  def fail_prep_iterdir(path: Path):
    """Simulate failure while checking whether prep output is empty."""

    if path == audio_dir / "prep":
      raise PermissionError("denied")
    return original_iterdir(path)

  monkeypatch.setattr(Path, "iterdir", fail_prep_iterdir)
  assert asr_prep.run(build_args(str(audio_dir))) == 2
  assert "Unable to list prep output directory" in capsys.readouterr().err
  monkeypatch.setattr(Path, "iterdir", original_iterdir)

  monkeypatch.setattr(asr_prep, "get_audio_duration_seconds", lambda path: (_ for _ in ()).throw(ValueError("bad")))
  assert asr_prep.run(build_args(str(audio_dir))) == 2
  assert "Unable to read audio duration" in capsys.readouterr().err

  monkeypatch.undo()
  monkeypatch.setattr(asr_prep, "get_audio_duration_seconds", lambda path: 0.0004)
  assert asr_prep.run(build_args(str(audio_dir))) == 2
  assert "rounds to 0 seconds" in capsys.readouterr().err

  monkeypatch.undo()
  prep_tmp_dir = tmp_path / "prep-write-failures"
  prep_tmp_dir.mkdir()
  segment = asr_prep.Segment(
    source_path=tmp_path / "source.wav",
    source_file="source.wav",
    source_stem="source",
    chunk_path=prep_tmp_dir / "source_0000_000000_001000.wav",
    index=0,
    start_seconds=0.0,
    end_seconds=1.0,
    duration_seconds=1.0,
  )
  sources = [{"source_file": "source.wav", "duration_seconds": 1.0, "chunk_count": 1}]
  original_write_text = Path.write_text

  def fail_manifest_write(path: Path, *args: Any, **kwargs: Any) -> int:
    """Simulate an OSError while writing manifest.json."""

    if path.name == "manifest.json":
      raise PermissionError("denied")
    return original_write_text(path, *args, **kwargs)

  monkeypatch.setattr(Path, "write_text", fail_manifest_write)
  with pytest.raises(ValueError, match="Unable to write prep manifest"):
    asr_prep.write_manifest(prep_tmp_dir, sources=sources, segments=[segment], overlap_seconds=3.0)

  def fail_report_write(path: Path, *args: Any, **kwargs: Any) -> int:
    """Simulate an OSError while writing report.txt."""

    if path.name == "report.txt":
      raise PermissionError("denied")
    return original_write_text(path, *args, **kwargs)

  monkeypatch.setattr(Path, "write_text", fail_report_write)
  with pytest.raises(ValueError, match="Unable to write prep report"):
    asr_prep.write_report(prep_tmp_dir, sources=sources, segments=[segment], overlap_seconds=3.0)


def test_ffmpeg_failure_and_duration_fallbacks(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
  """ffmpeg and duration helpers surface controlled ValueError failures."""

  segment = asr_prep.Segment(
    source_path=tmp_path / "source.wav",
    source_file="source.wav",
    source_stem="source",
    chunk_path=tmp_path / "chunk.wav",
    index=0,
    start_seconds=0.0,
    end_seconds=1.0,
    duration_seconds=1.0,
  )

  def fail_run(*_: object, **__: object) -> subprocess.CompletedProcess[str]:
    """Simulate an ffmpeg process failure."""

    raise subprocess.CalledProcessError(1, ["ffmpeg"], stderr="nope")

  monkeypatch.setattr(asr_prep.subprocess, "run", fail_run)
  with pytest.raises(ValueError, match="ffmpeg failed"):
    asr_prep.run_ffmpeg_segment(segment)

  class Info:
    """Mutagen-like info object with a non-numeric length."""

    length = True

  class Audio:
    """Mutagen-like audio object with invalid info."""

    info = Info()

  monkeypatch.setattr(asr_prep.mutagen, "File", lambda path: Audio())
  with pytest.raises(ValueError, match="Unable to determine audio duration"):
    asr_prep.get_audio_duration_seconds(tmp_path / "clip.wav")

  class NumericInfo:
    """Mutagen-like info object with a valid length."""

    length = 1.5

  class NumericAudio:
    """Mutagen-like audio object with valid info."""

    info = NumericInfo()

  monkeypatch.setattr(asr_prep.mutagen, "File", lambda path: NumericAudio())
  assert asr_prep.get_audio_duration_seconds(tmp_path / "clip.wav") == 1.5

  class ZeroInfo:
    """Mutagen-like info object with a zero length."""

    length = 0.0

  class ZeroAudio:
    """Mutagen-like audio object with zero-duration info."""

    info = ZeroInfo()

  monkeypatch.setattr(asr_prep.mutagen, "File", lambda path: ZeroAudio())
  with pytest.raises(ValueError, match="Unable to determine audio duration"):
    asr_prep.get_audio_duration_seconds(tmp_path / "clip.wav")

  class InfiniteInfo:
    """Mutagen-like info object with infinite length."""

    length = float("inf")

  class InfiniteAudio:
    """Mutagen-like audio object with infinite-duration info."""

    info = InfiniteInfo()

  monkeypatch.setattr(asr_prep.mutagen, "File", lambda path: InfiniteAudio())
  with pytest.raises(ValueError, match="Unable to determine audio duration"):
    asr_prep.get_audio_duration_seconds(tmp_path / "clip.wav")


def test_validate_audio_inputs_rejects_names_that_prepared_wer_cannot_reload() -> None:
  """Prep rejects input names that would make prepared WER ambiguous."""

  with pytest.raises(ValueError, match="plain filename"):
    asr_prep.validate_audio_inputs([asr_prep.AudioInput(path=Path("bad\\name.wav"), stem="bad\\name", format="wav")])
  with pytest.raises(ValueError, match="plain filename"):
    asr_prep.validate_audio_inputs([asr_prep.AudioInput(path=Path("call\t1.wav"), stem="call\t1", format="wav")])

  with pytest.raises(ValueError, match="source filename stem"):
    asr_prep.validate_audio_inputs([asr_prep.AudioInput(path=Path("..wav"), stem=".", format="wav")])


def test_segment_planning_uses_stable_30_second_chunks_and_direct_children(tmp_path: Path) -> None:
  """Segment planning uses stable chunk names and direct-child discovery."""

  audio_dir = tmp_path / "audio"
  nested = audio_dir / "nested"
  nested.mkdir(parents=True)
  write_audio(audio_dir / "call.wav")
  write_audio(audio_dir / "notes.txt")
  write_audio(nested / "nested.wav")

  audio_files = asr_prep.discover_audio_files(audio_dir)
  assert [audio.path.name for audio in audio_files] == ["call.wav"]

  segments = asr_prep.plan_segments(
    audio_files[0], duration_seconds=65.2, overlap_seconds=3.0, output_dir=audio_dir / "prep"
  )
  assert [
    (segment.index, segment.start_seconds, segment.end_seconds, segment.chunk_path.name) for segment in segments
  ] == [
    (0, 0.0, 30.0, "call_0000_000000_030000.wav"),
    (1, 27.0, 57.0, "call_0001_027000_057000.wav"),
    (2, 54.0, 65.2, "call_0002_054000_065200.wav"),
  ]

  custom = asr_prep.plan_segments(
    audio_files[0], duration_seconds=61.0, overlap_seconds=5.0, output_dir=audio_dir / "prep"
  )
  assert [segment.start_seconds for segment in custom] == [0.0, 25.0, 50.0]

  sub_tolerance_tail = asr_prep.plan_segments(
    audio_files[0], duration_seconds=57.001, overlap_seconds=3.0, output_dir=audio_dir / "prep"
  )
  assert [(segment.start_seconds, segment.end_seconds) for segment in sub_tolerance_tail] == [(0.0, 30.0), (27.0, 57.0)]

  rounded_to_zero = asr_prep.plan_segments(
    audio_files[0], duration_seconds=30.0001, overlap_seconds=0.0, output_dir=audio_dir / "prep"
  )
  assert [(segment.start_seconds, segment.end_seconds, segment.duration_seconds) for segment in rounded_to_zero] == [
    (0.0, 30.0, 30.0)
  ]
  assert (
    asr_prep.plan_segments(audio_files[0], duration_seconds=0.0001, overlap_seconds=0.0, output_dir=audio_dir / "prep")
    == []
  )


def test_run_invokes_ffmpeg_and_writes_manifest_and_report(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  """A full prep run invokes ffmpeg and writes expected metadata artifacts."""

  audio_dir = tmp_path / "audio"
  audio_dir.mkdir()
  write_audio(audio_dir / "call.wav")
  commands: list[list[str]] = []

  monkeypatch.setattr(asr_prep, "get_audio_duration_seconds", lambda path: 31.0)

  def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
    """Record ffmpeg command shape and create fake chunk outputs."""

    commands.append(command)
    Path(command[-1]).write_bytes(b"RIFF")
    assert kwargs["check"] is True
    assert kwargs["capture_output"] is True
    assert kwargs["text"] is True
    return subprocess.CompletedProcess(command, 0, "", "")

  monkeypatch.setattr(asr_prep.subprocess, "run", fake_run)

  assert asr_prep.run(build_args(str(audio_dir), "--overlap", "1.25")) == 0

  assert commands == [
    [
      "ffmpeg",
      "-hide_banner",
      "-nostdin",
      "-y",
      "-ss",
      "0.000",
      "-i",
      str(audio_dir / "call.wav"),
      "-t",
      "30.000",
      "-vn",
      "-acodec",
      "pcm_s16le",
      commands[0][-1],
    ],
    [
      "ffmpeg",
      "-hide_banner",
      "-nostdin",
      "-y",
      "-ss",
      "28.750",
      "-i",
      str(audio_dir / "call.wav"),
      "-t",
      "2.250",
      "-vn",
      "-acodec",
      "pcm_s16le",
      commands[1][-1],
    ],
  ]
  assert Path(commands[0][-1]).parent.name.startswith(".prep.")
  assert Path(commands[0][-1]).parent.name.endswith(".tmp")
  assert Path(commands[1][-1]).parent == Path(commands[0][-1]).parent
  manifest = json.loads((audio_dir / "prep" / "manifest.json").read_text(encoding="utf-8"))
  assert manifest["segment_duration_seconds"] == 30.0
  assert manifest["overlap_seconds"] == 1.25
  assert manifest["sources"][0]["source_file"] == "call.wav"
  assert len(manifest["chunks"]) == 2
  assert manifest["chunks"][0]["chunk_file"] == "call_0000_000000_030000.wav"
  report = (audio_dir / "prep" / "report.txt").read_text(encoding="utf-8")
  assert "asr-prep report" in report
  assert "call.wav" in report
