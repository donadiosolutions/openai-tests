from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import pytest

from openai_tests.test_modules import asr_prep


def build_args(*raw_args: str) -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  asr_prep.configure_parser(parser)
  return parser.parse_args([*raw_args])


def write_audio(path: Path) -> Path:
  path.write_bytes(b"RIFF")
  return path


def test_parser_accepts_audio_dir_and_overlap(tmp_path: Path) -> None:
  args = build_args(str(tmp_path), "--overlap", "1.5")

  assert args.audio_dir == str(tmp_path)
  assert args.overlap == 1.5


def test_configuration_errors_cover_input_overlap_prep_and_ffmpeg(
  capsys: pytest.CaptureFixture[str],
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  with pytest.raises(ValueError, match="overlap must be at least 0"):
    asr_prep.validate_overlap(-0.1)
  with pytest.raises(ValueError, match="overlap must be less than 30"):
    asr_prep.validate_overlap(30.0)

  assert asr_prep.run(build_args(str(tmp_path / "missing"))) == 2
  assert "Audio directory does not exist" in capsys.readouterr().err

  not_dir = write_audio(tmp_path / "not-dir.wav")
  assert asr_prep.run(build_args(str(not_dir))) == 2
  assert "Audio path is not a directory" in capsys.readouterr().err

  empty = tmp_path / "empty"
  empty.mkdir()
  assert asr_prep.run(build_args(str(empty))) == 2
  assert "No supported audio files" in capsys.readouterr().err

  audio_dir = tmp_path / "audio"
  audio_dir.mkdir()
  write_audio(audio_dir / "clip.wav")
  prep_dir = audio_dir / "prep"
  prep_dir.mkdir()
  (prep_dir / "old.wav").write_bytes(b"old")
  assert asr_prep.run(build_args(str(audio_dir))) == 2
  assert "already exists and is not empty" in capsys.readouterr().err

  (prep_dir / "old.wav").unlink()

  def missing_ffmpeg(*_: object, **__: object) -> subprocess.CompletedProcess[str]:
    raise FileNotFoundError("ffmpeg")

  monkeypatch.setattr(asr_prep, "get_audio_duration_seconds", lambda path: 1.0)
  monkeypatch.setattr(asr_prep.subprocess, "run", missing_ffmpeg)
  assert asr_prep.run(build_args(str(audio_dir))) == 2
  assert "ffmpeg was not found" in capsys.readouterr().err


def test_configuration_errors_cover_duration_listing_and_output_failures(
  capsys: pytest.CaptureFixture[str],
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  audio_dir = tmp_path / "audio"
  audio_dir.mkdir()
  write_audio(audio_dir / "clip.wav")
  original_iterdir = Path.iterdir

  def fail_audio_iterdir(path: Path):
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

  original_mkdir = Path.mkdir

  def fail_mkdir(
    path: Path,
    mode: int = 0o777,
    parents: bool = False,
    exist_ok: bool = False,
  ) -> None:
    if path == audio_dir / "prep":
      raise PermissionError("denied")
    original_mkdir(path, mode=mode, parents=parents, exist_ok=exist_ok)

  monkeypatch.setattr(Path, "mkdir", fail_mkdir)
  assert asr_prep.run(build_args(str(audio_dir))) == 2
  assert "Unable to create prep output directory" in capsys.readouterr().err
  monkeypatch.setattr(Path, "mkdir", original_mkdir)

  (audio_dir / "prep").mkdir()

  def fail_prep_iterdir(path: Path):
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


def test_ffmpeg_failure_and_duration_fallbacks(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
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
    raise subprocess.CalledProcessError(1, ["ffmpeg"], stderr="nope")

  monkeypatch.setattr(asr_prep.subprocess, "run", fail_run)
  with pytest.raises(ValueError, match="ffmpeg failed"):
    asr_prep.run_ffmpeg_segment(segment)

  class Info:
    length = True

  class Audio:
    info = Info()

  monkeypatch.setattr(asr_prep.mutagen, "File", lambda path: Audio())
  assert asr_prep.get_audio_duration_seconds(tmp_path / "clip.wav") == 0.0

  class NumericInfo:
    length = 1.5

  class NumericAudio:
    info = NumericInfo()

  monkeypatch.setattr(asr_prep.mutagen, "File", lambda path: NumericAudio())
  assert asr_prep.get_audio_duration_seconds(tmp_path / "clip.wav") == 1.5


def test_segment_planning_uses_stable_30_second_chunks_and_direct_children(tmp_path: Path) -> None:
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


def test_run_invokes_ffmpeg_and_writes_manifest_and_report(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  audio_dir = tmp_path / "audio"
  audio_dir.mkdir()
  write_audio(audio_dir / "call.wav")
  commands: list[list[str]] = []

  monkeypatch.setattr(asr_prep, "get_audio_duration_seconds", lambda path: 31.0)

  def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
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
      str(audio_dir / "prep" / "call_0000_000000_030000.wav"),
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
      str(audio_dir / "prep" / "call_0001_028750_031000.wav"),
    ],
  ]
  manifest = json.loads((audio_dir / "prep" / "manifest.json").read_text(encoding="utf-8"))
  assert manifest["segment_duration_seconds"] == 30.0
  assert manifest["overlap_seconds"] == 1.25
  assert manifest["sources"][0]["source_file"] == "call.wav"
  assert len(manifest["chunks"]) == 2
  assert manifest["chunks"][0]["chunk_file"] == "call_0000_000000_030000.wav"
  report = (audio_dir / "prep" / "report.txt").read_text(encoding="utf-8")
  assert "asr-prep report" in report
  assert "call.wav" in report
