#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import warnings
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from base_transcript import (
    SPEAKER_BLOCK_GAP_SECONDS,
    UNKNOWN_SPEAKER_ID,
    base_transcript_json_path,
    build_and_write_base_transcript,
    speaker_blocks_text_path,
    write_speaker_blocks_text,
)
from podcast_editor import (
    PodcastEditorError,
    TICKS_PER_SECOND,
    build_object_maps,
    get_object_ref,
    get_object_uref,
    load_prproj,
)
from speaker_diarization import (
    DEFAULT_SPEAKER_TOKEN_ENV_VARS,
    SpeakerDiarizationConfig,
    add_speaker_diarization_args,
    resolve_speaker_auth_token,
    speaker_diarization_config_from_args,
)


def emit_status(message: str) -> None:
    print(f"[status] {message}", file=sys.stderr, flush=True)


DIARIZATION_STEP_WEIGHTS = (
    ("segmentation", 0.45),
    ("speaker_counting", 0.05),
    ("embeddings", 0.35),
    ("discrete_diarization", 0.15),
)


def format_elapsed(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "--:--:--"
    whole_seconds = int(seconds)
    hours, remainder = divmod(whole_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def preferred_torch_device() -> str:
    try:
        import torch
    except ImportError:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"

    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"

    return "cpu"


def load_whisper_model_with_fallback(whisper_module: Any, model_name: str, device_name: str) -> tuple[Any, str]:
    try:
        return whisper_module.load_model(model_name, device=device_name), device_name
    except Exception as exc:
        if device_name != "cpu":
            emit_status(f"Whisper could not use torch device '{device_name}' ({exc}); retrying on CPU")
            return whisper_module.load_model(model_name, device="cpu"), "cpu"
        raise


class PlainTextProgressHook:
    def __init__(self, label: str) -> None:
        self.label = label
        self.current_step_name = ""
        self.last_fraction = -1.0
        self.last_draw_time = 0.0
        self.start_time = time.monotonic()
        self.weight_by_step = dict(DIARIZATION_STEP_WEIGHTS)
        completed_weight = 0.0
        self.completed_weight_by_step: dict[str, float] = {}
        for step_name, weight in DIARIZATION_STEP_WEIGHTS:
            self.completed_weight_by_step[step_name] = completed_weight
            completed_weight += weight
        self.total_weight = completed_weight or 1.0

    def __enter__(self) -> PlainTextProgressHook:
        self._draw(0.0, "starting")
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        final_fraction = 1.0 if exc_type is None else max(0.0, self.last_fraction)
        final_label = "finished" if exc_type is None else f"{self.current_step_name} interrupted"
        self._draw(final_fraction, final_label)
        sys.stderr.write("\n")
        sys.stderr.flush()
        return None

    def __call__(
        self,
        step_name: str,
        step_artifact: Any,
        file: Any = None,
        total: int | None = None,
        completed: int | None = None,
    ) -> None:
        local_fraction = 1.0
        if total is not None and completed is not None and total > 0:
            local_fraction = min(1.0, max(0.0, completed / total))

        completed_weight = self.completed_weight_by_step.get(step_name, 0.0)
        step_weight = self.weight_by_step.get(step_name, 0.0)
        overall_fraction = min(1.0, max(0.0, (completed_weight + step_weight * local_fraction) / self.total_weight))
        pretty_step = step_name.replace("_", " ")
        self._draw(overall_fraction, pretty_step)

    def _draw(self, fraction: float, step_name: str) -> None:
        now = time.monotonic()
        if step_name == self.current_step_name and fraction < 1.0:
            if fraction <= self.last_fraction and (now - self.last_draw_time) < 0.5:
                return
            if (fraction - self.last_fraction) < 0.01 and (now - self.last_draw_time) < 0.5:
                return

        self.current_step_name = step_name
        self.last_fraction = fraction
        self.last_draw_time = now

        elapsed = now - self.start_time
        eta = None if fraction <= 0.0 else max(0.0, elapsed * (1.0 - fraction) / fraction)
        terminal_width = shutil.get_terminal_size((120, 20)).columns
        bar_width = max(12, min(32, terminal_width // 4))
        filled = min(bar_width, max(0, int(round(bar_width * fraction))))
        bar = "#" * filled + "-" * (bar_width - filled)
        percent = int(round(fraction * 100))
        line = (
            f"\r[{bar}] {percent:3d}% | elapsed {format_elapsed(elapsed)} | "
            f"eta {format_elapsed(eta)} | {self.label}: {step_name}"
        )
        visible_width = max(terminal_width - 1, 20)
        sys.stderr.write(line[:visible_width].ljust(visible_width))
        sys.stderr.flush()


@dataclass(frozen=True)
class AudioMixLayer:
    source_path: str
    source_start_ticks: int
    source_end_ticks: int


@dataclass(frozen=True)
class SequenceAudioLayer:
    timeline_start_ticks: int
    timeline_end_ticks: int
    source_start_ticks: int
    source_end_ticks: int
    source_path: str


@dataclass(frozen=True)
class SequenceAudioSegment:
    index: int
    timeline_start_ticks: int
    timeline_end_ticks: int
    source_start_ticks: int
    source_end_ticks: int
    source_path: str
    mix_layers: tuple[AudioMixLayer, ...] = ()

    @property
    def duration_ticks(self) -> int:
        return self.source_end_ticks - self.source_start_ticks


@dataclass(frozen=True)
class SpeakerTurn:
    speaker_id: str
    start: float
    end: float


@dataclass(frozen=True)
class SpeakerBlock:
    block_index: int
    speaker_id: str
    start: float
    end: float
    text: str
    segment_indexes: tuple[int, ...]


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a Premiere sequence's audio from its clip ranges and transcribe it with Whisper."
    )
    parser.add_argument("--project", required=True, type=Path, help="Path to the Premiere .prproj file.")
    parser.add_argument("--sequence", required=True, help="Sequence name to transcribe.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for the rendered audio and transcript artifacts. Defaults to the project directory.",
    )
    parser.add_argument(
        "--basename",
        help="Base filename for outputs. Defaults to the sequence name.",
    )
    parser.add_argument(
        "--whisper-model",
        default="base.en",
        help="Whisper model name, for example tiny.en, base.en, small.en, medium.en, turbo.",
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Language hint for Whisper. Defaults to en.",
    )
    parser.add_argument(
        "--skip-whisper",
        action="store_true",
        help="Only render audio and the segment manifest.",
    )
    add_speaker_diarization_args(parser)
    return parser.parse_args(argv)


def sanitize_filename(name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {" ", "-", "_", "."} else "_" for ch in name)
    return " ".join(safe.split()).strip() or "sequence"


def find_sequence(root: ET.Element, sequence_name: str) -> ET.Element:
    for sequence in [elem for elem in root if elem.tag == "Sequence"]:
        if sequence.findtext("Name") == sequence_name:
            return sequence
    raise PodcastEditorError(f"Sequence '{sequence_name}' was not found in the project.")


def extract_sequence_frame_ticks(root: ET.Element, sequence_name: str) -> int:
    id_map, _ = build_object_maps(root)
    sequence = find_sequence(root, sequence_name)
    group_refs = [get_object_ref(track_group, "Second") for track_group in sequence.findall("./TrackGroups/TrackGroup")]
    groups = [id_map[ref] for ref in group_refs]
    video_group = next((group for group in groups if group.tag == "VideoTrackGroup"), None)
    if video_group is None:
        raise PodcastEditorError("Sequence is missing a video track group.")

    track_group = video_group.find("./TrackGroup")
    if track_group is None:
        raise PodcastEditorError("Video track group is missing its TrackGroup node.")

    frame_rate_text = track_group.findtext("FrameRate")
    if not frame_rate_text:
        raise PodcastEditorError("Video track group is missing its frame rate.")
    return int(frame_rate_text)


def populated_audio_tracks(sequence: ET.Element, id_map: dict[int, ET.Element], uid_map: dict[str, ET.Element]) -> list[ET.Element]:
    group_refs = [get_object_ref(track_group, "Second") for track_group in sequence.findall("./TrackGroups/TrackGroup")]
    groups = [id_map[ref] for ref in group_refs]
    audio_group = next((group for group in groups if group.tag == "AudioTrackGroup"), None)
    if audio_group is None:
        raise PodcastEditorError("Sequence is missing an audio track group.")

    tracks: list[ET.Element] = []
    for track_ref in audio_group.findall("./TrackGroup/Tracks/Track"):
        track = uid_map[track_ref.attrib["ObjectURef"]]
        track_items = track.findall("./ClipTrack/ClipItems/TrackItems/TrackItem")
        if track_items:
            tracks.append(track)
    if not tracks:
        raise PodcastEditorError("Sequence does not contain any populated audio tracks.")
    return tracks


def media_path_for_clip_source(source: ET.Element, uid_map: dict[str, ET.Element]) -> str:
    media_uid = get_object_uref(source, "./MediaSource/Media")
    media = uid_map.get(media_uid)
    if media is None:
        raise PodcastEditorError(f"Missing media object for UID {media_uid}.")

    actual = media.findtext("ActualMediaFilePath")
    file_path = media.findtext("FilePath")
    path = actual or file_path
    if not path:
        raise PodcastEditorError("Media object is missing its file path.")
    return path


def extract_audio_layers(root: ET.Element, sequence_name: str) -> list[SequenceAudioLayer]:
    id_map, uid_map = build_object_maps(root)
    sequence = find_sequence(root, sequence_name)
    tracks = populated_audio_tracks(sequence, id_map, uid_map)

    layers: list[SequenceAudioLayer] = []
    for track in tracks:
        for track_item_ref in track.findall("./ClipTrack/ClipItems/TrackItems/TrackItem"):
            object_ref = track_item_ref.attrib.get("ObjectRef")
            if not object_ref or not object_ref.isdigit():
                continue

            track_item = id_map[int(object_ref)]
            track_item_body = track_item.find("./ClipTrackItem/TrackItem")
            if track_item_body is None:
                raise PodcastEditorError("Audio track item is missing its TrackItem node.")

            timeline_start_ticks = int(track_item_body.findtext("Start", "0"))
            timeline_end_ticks = int(track_item_body.findtext("End", "0"))
            if timeline_end_ticks <= timeline_start_ticks:
                continue

            subclip = id_map[get_object_ref(track_item, "./ClipTrackItem/SubClip")]
            clip = id_map[get_object_ref(subclip, "Clip")]
            clip_body = clip.find("Clip")
            if clip_body is None:
                raise PodcastEditorError("Audio clip is missing its Clip payload.")

            source_start_ticks = int(clip_body.findtext("InPoint", "0"))
            default_source_end = source_start_ticks + (timeline_end_ticks - timeline_start_ticks)
            source_end_ticks = int(clip_body.findtext("OutPoint", str(default_source_end)))
            if source_end_ticks <= source_start_ticks:
                source_end_ticks = default_source_end

            source = id_map[get_object_ref(clip, "./Clip/Source")]
            source_path = media_path_for_clip_source(source, uid_map)

            layers.append(
                SequenceAudioLayer(
                    timeline_start_ticks=timeline_start_ticks,
                    timeline_end_ticks=timeline_end_ticks,
                    source_start_ticks=source_start_ticks,
                    source_end_ticks=source_end_ticks,
                    source_path=source_path,
                )
            )

    if not layers:
        raise PodcastEditorError("No audio clip segments were found in the selected sequence.")
    return layers


def extract_audio_segments(root: ET.Element, sequence_name: str) -> list[SequenceAudioSegment]:
    layers = extract_audio_layers(root, sequence_name)
    boundaries = sorted(
        {
            boundary
            for layer in layers
            for boundary in (layer.timeline_start_ticks, layer.timeline_end_ticks)
        }
    )
    segments: list[SequenceAudioSegment] = []
    for boundary_index, interval_start in enumerate(boundaries[:-1]):
        interval_end = boundaries[boundary_index + 1]
        if interval_end <= interval_start:
            continue
        mix_layers: list[AudioMixLayer] = []
        for layer in layers:
            if layer.timeline_start_ticks >= interval_end or layer.timeline_end_ticks <= interval_start:
                continue
            offset = interval_start - layer.timeline_start_ticks
            source_start_ticks = layer.source_start_ticks + offset
            source_end_ticks = source_start_ticks + (interval_end - interval_start)
            mix_layers.append(
                AudioMixLayer(
                    source_path=layer.source_path,
                    source_start_ticks=source_start_ticks,
                    source_end_ticks=source_end_ticks,
                )
            )
        if not mix_layers:
            continue
        segments.append(
            SequenceAudioSegment(
                index=len(segments),
                timeline_start_ticks=interval_start,
                timeline_end_ticks=interval_end,
                source_start_ticks=interval_start,
                source_end_ticks=interval_end,
                source_path=mix_layers[0].source_path,
                mix_layers=tuple(mix_layers),
            )
        )

    if not segments:
        raise PodcastEditorError("No audio clip segments were found in the selected sequence.")
    return segments


def build_ffmpeg_command(segments: list[SequenceAudioSegment], output_audio_path: Path) -> list[str]:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise PodcastEditorError("ffmpeg is not installed or not on PATH.")

    input_paths: list[str] = []
    input_index_by_path: dict[str, int] = {}
    for segment in segments:
        mix_layers = segment.mix_layers or (
            AudioMixLayer(segment.source_path, segment.source_start_ticks, segment.source_end_ticks),
        )
        for layer in mix_layers:
            if layer.source_path not in input_index_by_path:
                input_index_by_path[layer.source_path] = len(input_paths)
                input_paths.append(layer.source_path)

    filter_parts: list[str] = []
    concat_inputs: list[str] = []
    for segment in segments:
        mix_layers = segment.mix_layers or (
            AudioMixLayer(segment.source_path, segment.source_start_ticks, segment.source_end_ticks),
        )
        input_labels: list[str] = []
        for layer_index, layer in enumerate(mix_layers):
            input_index = input_index_by_path[layer.source_path]
            start_seconds = layer.source_start_ticks / TICKS_PER_SECOND
            end_seconds = layer.source_end_ticks / TICKS_PER_SECOND
            label = f"s{segment.index}l{layer_index}"
            filter_parts.append(
                f"[{input_index}:a]atrim=start={start_seconds:.6f}:end={end_seconds:.6f},asetpts=PTS-STARTPTS[{label}]"
            )
            input_labels.append(f"[{label}]")

        output_label = f"a{segment.index}"
        if len(input_labels) == 1:
            filter_parts.append(
                f"{input_labels[0]}aformat=sample_rates=16000:channel_layouts=mono[{output_label}]"
            )
        else:
            filter_parts.append(
                f"{''.join(input_labels)}amix=inputs={len(input_labels)}:normalize=1,"
                f"aformat=sample_rates=16000:channel_layouts=mono[{output_label}]"
            )
        concat_inputs.append(f"[{output_label}]")

    filter_parts.append(f"{''.join(concat_inputs)}concat=n={len(segments)}:v=0:a=1[outa]")

    command = [ffmpeg, "-y"]
    for input_path in input_paths:
        command.extend(["-i", input_path])
    command.extend(
        [
            "-filter_complex",
            ";".join(filter_parts),
            "-map",
            "[outa]",
            "-ac",
            "1",
            "-ar",
            "16000",
            str(output_audio_path),
        ]
    )
    return command


def render_sequence_audio(segments: list[SequenceAudioSegment], output_audio_path: Path) -> None:
    emit_status(f"Rendering {len(segments)} audio segment(s) to {output_audio_path}")
    command = build_ffmpeg_command(segments, output_audio_path)
    subprocess.run(command, check=True)


def write_segment_manifest(segments: list[SequenceAudioSegment], output_path: Path) -> None:
    output_path.write_text(
        json.dumps([asdict(segment) for segment in segments], indent=2),
        encoding="utf-8",
    )


def overlap_duration(start: float, end: float, turn: SpeakerTurn) -> float:
    return max(0.0, min(end, turn.end) - max(start, turn.start))


def span_distance(start: float, end: float, turn: SpeakerTurn) -> float:
    if end < turn.start:
        return turn.start - end
    if start > turn.end:
        return start - turn.end
    return 0.0


def speaker_id_for_span(start: float, end: float, speaker_turns: list[SpeakerTurn]) -> str | None:
    best_turn: SpeakerTurn | None = None
    best_overlap = 0.0
    nearest_turn: SpeakerTurn | None = None
    nearest_distance = float("inf")

    for turn in speaker_turns:
        overlap = overlap_duration(start, end, turn)
        if overlap > best_overlap:
            best_overlap = overlap
            best_turn = turn

        distance = span_distance(start, end, turn)
        if distance < nearest_distance:
            nearest_distance = distance
            nearest_turn = turn

    if best_turn is not None and best_overlap > 0.0:
        return best_turn.speaker_id
    if nearest_turn is not None:
        return nearest_turn.speaker_id
    return None


def finalize_speaker_block(
    blocks: list[SpeakerBlock],
    speaker_id: str | None,
    start: float | None,
    end: float | None,
    tokens: list[str],
    segment_indexes: list[int],
) -> None:
    if start is None or end is None:
        return
    text = " ".join(part.strip() for part in tokens if part.strip()).strip()
    if not text:
        return
    blocks.append(
        SpeakerBlock(
            block_index=len(blocks),
            speaker_id=speaker_id or UNKNOWN_SPEAKER_ID,
            start=start,
            end=end,
            text=text,
            segment_indexes=tuple(segment_indexes),
        )
    )


def build_speaker_blocks_from_words(words: list[dict[str, Any]]) -> list[SpeakerBlock]:
    blocks: list[SpeakerBlock] = []
    current_speaker_id: str | None = None
    current_start: float | None = None
    current_end: float | None = None
    current_tokens: list[str] = []
    current_segment_indexes: list[int] = []

    for word in words:
        speaker_id = str(word.get("speaker_id") or UNKNOWN_SPEAKER_ID)
        start = float(word["start"])
        end = float(word["end"])
        gap = 0.0 if current_end is None else max(0.0, start - current_end)
        should_break = bool(current_tokens) and (
            speaker_id != current_speaker_id or gap >= SPEAKER_BLOCK_GAP_SECONDS
        )
        if should_break:
            finalize_speaker_block(
                blocks,
                current_speaker_id,
                current_start,
                current_end,
                current_tokens,
                current_segment_indexes,
            )
            current_speaker_id = None
            current_start = None
            current_end = None
            current_tokens = []
            current_segment_indexes = []

        if current_start is None:
            current_start = start
            current_speaker_id = speaker_id
        current_end = end
        current_tokens.append(str(word["text"]))
        segment_index = int(word["segment_index"])
        if not current_segment_indexes or current_segment_indexes[-1] != segment_index:
            current_segment_indexes.append(segment_index)

    finalize_speaker_block(
        blocks,
        current_speaker_id,
        current_start,
        current_end,
        current_tokens,
        current_segment_indexes,
    )
    return blocks


def build_speaker_blocks_from_segments(segments: list[dict[str, Any]]) -> list[SpeakerBlock]:
    blocks: list[SpeakerBlock] = []
    current_speaker_id: str | None = None
    current_start: float | None = None
    current_end: float | None = None
    current_text_parts: list[str] = []
    current_segment_indexes: list[int] = []

    for index, segment in enumerate(segments):
        text = " ".join(str(segment.get("text", "")).split()).strip()
        if not text:
            continue

        start = float(segment["start"])
        end = float(segment["end"])
        speaker_id = str(segment.get("speaker_id") or UNKNOWN_SPEAKER_ID)
        gap = 0.0 if current_end is None else max(0.0, start - current_end)
        should_break = bool(current_text_parts) and (
            speaker_id != current_speaker_id or gap >= SPEAKER_BLOCK_GAP_SECONDS
        )
        if should_break:
            finalize_speaker_block(
                blocks,
                current_speaker_id,
                current_start,
                current_end,
                current_text_parts,
                current_segment_indexes,
            )
            current_speaker_id = None
            current_start = None
            current_end = None
            current_text_parts = []
            current_segment_indexes = []

        if current_start is None:
            current_start = start
            current_speaker_id = speaker_id
        current_end = end
        current_text_parts.append(text)
        current_segment_indexes.append(index)

    finalize_speaker_block(
        blocks,
        current_speaker_id,
        current_start,
        current_end,
        current_text_parts,
        current_segment_indexes,
    )
    return blocks


def diarize_speakers(audio_path: Path, config: SpeakerDiarizationConfig) -> list[SpeakerTurn]:
    if not config.enabled:
        return []

    token = resolve_speaker_auth_token(config.auth_token)
    if not token and not Path(config.model).exists():
        env_names = ", ".join(DEFAULT_SPEAKER_TOKEN_ENV_VARS)
        raise PodcastEditorError(
            "Speaker diarization requires a Hugging Face or pyannote token unless "
            f"--speaker-diarization-model points to a local pipeline directory. Set one of: {env_names}."
        )

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r".*torchcodec is not installed correctly so built-in audio decoding will fail.*",
            )
            warnings.filterwarnings(
                "ignore",
                message=r".*torchaudio.load_with_torchcodec.*",
            )
            import torch
            from pyannote.audio import Pipeline
    except ImportError as exc:
        raise PodcastEditorError(
            "Speaker diarization requires `pyannote.audio` in the active Python environment."
        ) from exc
    emit_status(f"Preparing audio for speaker diarization from {audio_path.name}")
    audio_file = load_audio_file_for_diarization(audio_path)
    emit_status(f"Speaker diarization audio is ready for {audio_path.name}")

    emit_status(f"Loading speaker diarization pipeline '{config.model}'")
    try:
        if token:
            try:
                pipeline = Pipeline.from_pretrained(config.model, token=token)
            except TypeError:
                pipeline = Pipeline.from_pretrained(config.model, use_auth_token=token)
        else:
            pipeline = Pipeline.from_pretrained(config.model)
    except Exception as exc:
        raise PodcastEditorError(f"Failed to load speaker diarization pipeline '{config.model}': {exc}") from exc

    device_name = preferred_torch_device()
    if device_name != "cpu" and hasattr(pipeline, "to"):
        try:
            pipeline.to(torch.device(device_name))
            emit_status(f"Using torch device '{device_name}' for speaker diarization")
        except Exception as exc:
            emit_status(f"Could not move speaker diarization to {device_name} ({exc}); continuing on CPU")

    diarization_kwargs: dict[str, int] = {}
    if config.num_speakers is not None:
        diarization_kwargs["num_speakers"] = config.num_speakers
    if config.min_speakers is not None:
        diarization_kwargs["min_speakers"] = config.min_speakers
    if config.max_speakers is not None:
        diarization_kwargs["max_speakers"] = config.max_speakers

    emit_status(f"Detecting speakers in {audio_path.name}")
    emit_status("Speaker diarization progress will be reported step-by-step")
    with PlainTextProgressHook("speaker diarization") as hook:
        output = pipeline(audio_file, hook=hook, **diarization_kwargs)
    emit_status("Speaker diarization finished")
    diarization = getattr(output, "exclusive_speaker_diarization", None)
    if diarization is None:
        diarization = getattr(output, "speaker_diarization", output)

    raw_turns: list[tuple[float, float, str]] = []
    if hasattr(diarization, "itertracks"):
        for turn, _, speaker_label in diarization.itertracks(yield_label=True):
            start = float(turn.start)
            end = float(turn.end)
            if end <= start:
                continue
            raw_turns.append((start, end, str(speaker_label)))
    else:
        for turn, speaker_label in diarization:
            start = float(turn.start)
            end = float(turn.end)
            if end <= start:
                continue
            raw_turns.append((start, end, str(speaker_label)))

    if not raw_turns:
        raise PodcastEditorError("Speaker diarization completed but did not return any speaker turns.")

    raw_turns.sort(key=lambda item: (item[0], item[1], item[2]))
    speaker_id_by_label: dict[str, str] = {}
    normalized_turns: list[SpeakerTurn] = []
    for start, end, label in raw_turns:
        speaker_id = speaker_id_by_label.setdefault(label, f"speaker_{len(speaker_id_by_label):02d}")
        normalized_turns.append(SpeakerTurn(speaker_id=speaker_id, start=start, end=end))
    return normalized_turns


def load_audio_file_for_diarization(audio_path: Path) -> dict[str, Any]:
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r".*torchcodec is not installed correctly so built-in audio decoding will fail.*",
            )
            warnings.filterwarnings(
                "ignore",
                message=r".*torchaudio.load_with_torchcodec.*",
            )
            import torch
            import torchaudio
    except ImportError as exc:
        raise PodcastEditorError(
            "Speaker diarization requires `torchaudio` in the active Python environment."
        ) from exc

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r".*torchcodec is not installed correctly so built-in audio decoding will fail.*",
            )
            warnings.filterwarnings(
                "ignore",
                message=r".*torchaudio.load_with_torchcodec.*",
            )
            waveform, sample_rate = torchaudio.load(str(audio_path))
    except Exception as exc:
        raise PodcastEditorError(f"Failed to load audio for speaker diarization from {audio_path}.") from exc

    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    waveform = waveform.to(dtype=torch.float32).contiguous()
    return {
        "waveform": waveform,
        "sample_rate": int(sample_rate),
        "uri": audio_path.stem,
    }


def annotate_transcript_with_speakers(
    transcript: dict[str, Any],
    speaker_turns: list[SpeakerTurn],
) -> list[SpeakerBlock]:
    transcript["speaker_diarization"] = [asdict(turn) for turn in speaker_turns]
    transcript["speaker_count"] = len({turn.speaker_id for turn in speaker_turns})

    segments = transcript.get("segments", [])
    if not isinstance(segments, list):
        transcript["speaker_blocks"] = []
        return []

    words: list[dict[str, Any]] = []
    for segment_index, segment in enumerate(segments):
        if not isinstance(segment, dict):
            continue
        try:
            start = float(segment["start"])
            end = float(segment["end"])
        except (KeyError, TypeError, ValueError):
            continue
        segment_speaker_id = speaker_id_for_span(start, end, speaker_turns) or UNKNOWN_SPEAKER_ID
        segment["speaker_id"] = segment_speaker_id

        segment_words = segment.get("words")
        if not isinstance(segment_words, list):
            continue
        for word in segment_words:
            if not isinstance(word, dict):
                continue
            raw_text = str(word.get("word", ""))
            if not raw_text.strip():
                continue
            try:
                word_start = float(word["start"])
                word_end = float(word["end"])
            except (KeyError, TypeError, ValueError):
                continue
            if word_end <= word_start:
                continue
            word_speaker_id = speaker_id_for_span(word_start, word_end, speaker_turns) or segment_speaker_id
            word["speaker_id"] = word_speaker_id
            words.append(
                {
                    "text": raw_text,
                    "start": word_start,
                    "end": word_end,
                    "speaker_id": word_speaker_id,
                    "segment_index": segment_index,
                }
            )

    blocks = build_speaker_blocks_from_words(words) if words else build_speaker_blocks_from_segments(segments)
    transcript["speaker_blocks"] = [asdict(block) for block in blocks]
    return blocks


def transcribe_with_whisper(
    audio_path: Path,
    output_dir: Path,
    whisper_model: str,
    language: str,
    speaker_config: SpeakerDiarizationConfig | None = None,
) -> Path:
    try:
        import whisper
        from whisper.utils import get_writer
    except ImportError as exc:
        raise PodcastEditorError(
            "Whisper is not importable in this Python environment. Run this script via `conda run -n agent python ...`."
        ) from exc

    speaker_config = speaker_config or SpeakerDiarizationConfig()
    device_name = preferred_torch_device()
    emit_status(f"Preparing Whisper transcription for {audio_path.name}")
    emit_status(f"Loading Whisper model '{whisper_model}' on torch device '{device_name}'")
    model, actual_device_name = load_whisper_model_with_fallback(whisper, whisper_model, device_name)
    emit_status(f"Whisper model '{whisper_model}' loaded on '{actual_device_name}'")
    emit_status(f"Transcribing {audio_path.name}")
    result = model.transcribe(str(audio_path), language=language, verbose=False, word_timestamps=True)
    if speaker_config.enabled:
        try:
            speaker_turns = diarize_speakers(audio_path, speaker_config)
        except PodcastEditorError as exc:
            result["speaker_diarization_error"] = str(exc)
            result["speaker_detection_error"] = str(exc)
            emit_status(
                f"Speaker diarization unavailable ({exc}). Continuing with {UNKNOWN_SPEAKER_ID} speaker IDs "
                "in the base transcript."
            )
        else:
            annotate_transcript_with_speakers(result, speaker_turns)

    for fmt in ["txt", "json", "srt", "vtt", "tsv"]:
        writer = get_writer(fmt, str(output_dir))
        writer(result, str(audio_path))

    return output_dir / f"{audio_path.stem}.json"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    try:
        speaker_config = speaker_diarization_config_from_args(args)
    except ValueError as exc:
        raise PodcastEditorError(str(exc)) from exc
    emit_status(f"Opening Premiere project {args.project}")
    root = load_prproj(args.project)
    emit_status(f"Extracting audio segments from sequence '{args.sequence}'")
    segments = extract_audio_segments(root, args.sequence)
    frame_ticks = extract_sequence_frame_ticks(root, args.sequence)

    output_dir = args.output_dir or args.project.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    basename = sanitize_filename(args.basename or args.sequence)

    output_audio_path = output_dir / f"{basename}.wav"
    manifest_path = output_dir / f"{basename}.segments.json"

    render_sequence_audio(segments, output_audio_path)
    write_segment_manifest(segments, manifest_path)
    emit_status(f"Wrote segment manifest to {manifest_path}")

    transcript_json_path = None
    base_transcript_path = None
    if not args.skip_whisper:
        transcript_json_path = transcribe_with_whisper(
            output_audio_path,
            output_dir,
            args.whisper_model,
            args.language,
            speaker_config,
        )
        base_transcript_path = base_transcript_json_path(output_audio_path, output_dir)
        base_transcript = build_and_write_base_transcript(
            transcript_json_path,
            manifest_path,
            frame_ticks,
            base_transcript_path,
        )
        emit_status(f"Wrote base transcript to {base_transcript_path}")
        speaker_blocks_path = speaker_blocks_text_path(output_audio_path, output_dir)
        write_speaker_blocks_text(base_transcript, speaker_blocks_path)
        emit_status(f"Wrote speaker blocks to {speaker_blocks_path}")
        emit_status(f"Wrote transcript artifacts to {output_dir}")
    else:
        emit_status("Skipping Whisper transcription")

    print(f"sequence={args.sequence}")
    print(f"segments={len(segments)}")
    print(f"rendered_audio={output_audio_path}")
    print(f"segment_manifest={manifest_path}")
    if transcript_json_path is not None:
        print(f"transcript_json={transcript_json_path}")
        if base_transcript_path is not None:
            print(f"base_transcript_json={base_transcript_path}")
        speaker_blocks_path = speaker_blocks_text_path(output_audio_path, output_dir)
        if speaker_blocks_path.exists():
            print(f"speaker_blocks_txt={speaker_blocks_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
