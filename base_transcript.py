from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from podcast_editor import PodcastEditorError, TICKS_PER_SECOND, nominal_fps

UNKNOWN_SPEAKER_ID = "speaker_unknown"
SPEAKER_BLOCK_GAP_SECONDS = 1.0
SENTENCE_BREAK_GAP_SECONDS = 0.6
MIN_WORDS_FOR_GAP_BREAK = 3
MAX_WORDS_PER_TRANSCRIPT_LINE = 32
SENTENCE_END_PUNCTUATION = (".", "!", "?", "…")


@dataclass(frozen=True)
class BaseTranscriptWord:
    word_index: int
    segment_index: int
    speaker_id: str
    raw_text: str
    text: str
    start_seconds: float
    end_seconds: float
    start_ticks: int
    end_ticks: int
    start_timecode: str
    end_timecode: str


@dataclass(frozen=True)
class BaseTranscriptSegment:
    segment_index: int
    speaker_id: str
    raw_text: str
    text: str
    start_seconds: float
    end_seconds: float
    start_ticks: int
    end_ticks: int
    start_timecode: str
    end_timecode: str
    word_indexes: tuple[int, ...]


@dataclass(frozen=True)
class BaseTranscriptSpeakerBlock:
    block_index: int
    speaker_id: str
    text: str
    start_seconds: float
    end_seconds: float
    start_ticks: int
    end_ticks: int
    start_timecode: str
    end_timecode: str
    word_indexes: tuple[int, ...]
    segment_indexes: tuple[int, ...]


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def ticks_to_timecode(ticks: int, frame_ticks: int) -> str:
    fps = nominal_fps(frame_ticks)
    total_frames = max(0, int(round(ticks / frame_ticks)))
    frames = total_frames % fps
    total_seconds = total_frames // fps
    seconds = total_seconds % 60
    total_minutes = total_seconds // 60
    minutes = total_minutes % 60
    hours = total_minutes // 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}:{frames:02d}"


def seconds_to_ticks(seconds: float) -> int:
    return int(round(seconds * TICKS_PER_SECOND))


def rendered_audio_ticks_to_timeline_ticks(audio_ticks: int, manifest: list[dict[str, Any]]) -> int:
    cursor = 0
    for index, item in enumerate(manifest):
        duration = int(item["source_end_ticks"]) - int(item["source_start_ticks"])
        if duration <= 0:
            continue
        segment_end = cursor + duration
        if audio_ticks <= segment_end or index == len(manifest) - 1:
            local_offset = max(0, min(duration, audio_ticks - cursor))
            return int(item["timeline_start_ticks"]) + local_offset
        cursor = segment_end
    return 0


def rendered_audio_seconds_to_timeline_ticks(seconds: float, manifest: list[dict[str, Any]]) -> int:
    return rendered_audio_ticks_to_timeline_ticks(seconds_to_ticks(seconds), manifest)


def base_transcript_json_path(audio_path: Path, output_dir: Path) -> Path:
    return output_dir / f"{audio_path.stem}.base_transcript.json"


def speaker_blocks_text_path(audio_path: Path, output_dir: Path) -> Path:
    return output_dir / f"{audio_path.stem}.speaker_blocks.txt"


def editorial_blocks_text_path(audio_path: Path, output_dir: Path) -> Path:
    return output_dir / f"{audio_path.stem}.editorial_blocks.txt"


def build_base_speaker_blocks(
    words: list[BaseTranscriptWord],
    segments: list[BaseTranscriptSegment],
) -> list[BaseTranscriptSpeakerBlock]:
    if words:
        return build_base_speaker_blocks_from_words(words)
    return build_base_speaker_blocks_from_segments(segments)


def build_base_speaker_blocks_from_words(words: list[BaseTranscriptWord]) -> list[BaseTranscriptSpeakerBlock]:
    blocks: list[BaseTranscriptSpeakerBlock] = []
    current_words: list[BaseTranscriptWord] = []
    current_speaker_id: str | None = None

    for word in words:
        gap_seconds = 0.0 if not current_words else max(0.0, word.start_seconds - current_words[-1].end_seconds)
        should_break = bool(current_words) and (
            word.speaker_id != current_speaker_id or gap_seconds >= SPEAKER_BLOCK_GAP_SECONDS
        )
        if should_break:
            blocks.append(build_base_speaker_block_from_words(len(blocks), current_words))
            current_words = []
            current_speaker_id = None

        if not current_words:
            current_speaker_id = word.speaker_id
        current_words.append(word)

    if current_words:
        blocks.append(build_base_speaker_block_from_words(len(blocks), current_words))
    return blocks


def build_base_speaker_block_from_words(
    block_index: int,
    words: list[BaseTranscriptWord],
) -> BaseTranscriptSpeakerBlock:
    segment_indexes = list(dict.fromkeys(word.segment_index for word in words))
    return BaseTranscriptSpeakerBlock(
        block_index=block_index,
        speaker_id=words[0].speaker_id,
        text=" ".join(word.text for word in words if word.text).strip(),
        start_seconds=words[0].start_seconds,
        end_seconds=words[-1].end_seconds,
        start_ticks=words[0].start_ticks,
        end_ticks=words[-1].end_ticks,
        start_timecode=words[0].start_timecode,
        end_timecode=words[-1].end_timecode,
        word_indexes=tuple(word.word_index for word in words),
        segment_indexes=tuple(segment_indexes),
    )


def build_base_speaker_blocks_from_segments(
    segments: list[BaseTranscriptSegment],
) -> list[BaseTranscriptSpeakerBlock]:
    blocks: list[BaseTranscriptSpeakerBlock] = []
    current_segments: list[BaseTranscriptSegment] = []
    current_speaker_id: str | None = None

    for segment in segments:
        gap_seconds = 0.0 if not current_segments else max(0.0, segment.start_seconds - current_segments[-1].end_seconds)
        should_break = bool(current_segments) and (
            segment.speaker_id != current_speaker_id or gap_seconds >= SPEAKER_BLOCK_GAP_SECONDS
        )
        if should_break:
            blocks.append(build_base_speaker_block_from_segments(len(blocks), current_segments))
            current_segments = []
            current_speaker_id = None

        if not current_segments:
            current_speaker_id = segment.speaker_id
        current_segments.append(segment)

    if current_segments:
        blocks.append(build_base_speaker_block_from_segments(len(blocks), current_segments))
    return blocks


def build_base_speaker_block_from_segments(
    block_index: int,
    segments: list[BaseTranscriptSegment],
) -> BaseTranscriptSpeakerBlock:
    word_indexes: list[int] = []
    for segment in segments:
        word_indexes.extend(segment.word_indexes)
    return BaseTranscriptSpeakerBlock(
        block_index=block_index,
        speaker_id=segments[0].speaker_id,
        text=" ".join(segment.text for segment in segments if segment.text).strip(),
        start_seconds=segments[0].start_seconds,
        end_seconds=segments[-1].end_seconds,
        start_ticks=segments[0].start_ticks,
        end_ticks=segments[-1].end_ticks,
        start_timecode=segments[0].start_timecode,
        end_timecode=segments[-1].end_timecode,
        word_indexes=tuple(word_indexes),
        segment_indexes=tuple(segment.segment_index for segment in segments),
    )


def build_base_transcript(
    transcript: dict[str, Any],
    manifest: list[dict[str, Any]],
    frame_ticks: int,
) -> dict[str, Any]:
    raw_segments = transcript.get("segments", [])
    if not isinstance(raw_segments, list) or not raw_segments:
        raise PodcastEditorError("Whisper returned no transcript segments.")

    words: list[BaseTranscriptWord] = []
    segments: list[BaseTranscriptSegment] = []

    for segment_index, raw_segment in enumerate(raw_segments):
        if not isinstance(raw_segment, dict):
            continue
        try:
            start_seconds = float(raw_segment["start"])
            end_seconds = float(raw_segment["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if end_seconds <= start_seconds:
            continue

        segment_raw_text = str(raw_segment.get("text", ""))
        segment_text = normalize_text(segment_raw_text)
        segment_speaker_id = str(raw_segment.get("speaker_id") or UNKNOWN_SPEAKER_ID)
        segment_start_ticks = rendered_audio_seconds_to_timeline_ticks(start_seconds, manifest)
        segment_end_ticks = rendered_audio_seconds_to_timeline_ticks(end_seconds, manifest)

        word_indexes: list[int] = []
        segment_words = raw_segment.get("words")
        if isinstance(segment_words, list):
            for raw_word in segment_words:
                if not isinstance(raw_word, dict):
                    continue
                word_raw_text = str(raw_word.get("word", ""))
                word_text = normalize_text(word_raw_text)
                if not word_text:
                    continue
                try:
                    word_start_seconds = float(raw_word["start"])
                    word_end_seconds = float(raw_word["end"])
                except (KeyError, TypeError, ValueError):
                    continue
                if word_end_seconds <= word_start_seconds:
                    continue
                word_start_ticks = rendered_audio_seconds_to_timeline_ticks(word_start_seconds, manifest)
                word_end_ticks = rendered_audio_seconds_to_timeline_ticks(word_end_seconds, manifest)
                word_indexes.append(len(words))
                words.append(
                    BaseTranscriptWord(
                        word_index=len(words),
                        segment_index=segment_index,
                        speaker_id=str(raw_word.get("speaker_id") or segment_speaker_id or UNKNOWN_SPEAKER_ID),
                        raw_text=word_raw_text,
                        text=word_text,
                        start_seconds=word_start_seconds,
                        end_seconds=word_end_seconds,
                        start_ticks=word_start_ticks,
                        end_ticks=word_end_ticks,
                        start_timecode=ticks_to_timecode(word_start_ticks, frame_ticks),
                        end_timecode=ticks_to_timecode(word_end_ticks, frame_ticks),
                    )
                )

        segments.append(
            BaseTranscriptSegment(
                segment_index=segment_index,
                speaker_id=segment_speaker_id,
                raw_text=segment_raw_text,
                text=segment_text,
                start_seconds=start_seconds,
                end_seconds=end_seconds,
                start_ticks=segment_start_ticks,
                end_ticks=segment_end_ticks,
                start_timecode=ticks_to_timecode(segment_start_ticks, frame_ticks),
                end_timecode=ticks_to_timecode(segment_end_ticks, frame_ticks),
                word_indexes=tuple(word_indexes),
            )
        )

    if not segments:
        raise PodcastEditorError("Whisper returned no transcript segments.")

    speaker_blocks = build_base_speaker_blocks(words, segments)
    speaker_ids = sorted({entry.speaker_id for entry in [*words, *segments]} or {UNKNOWN_SPEAKER_ID})

    return {
        "schema_version": 1,
        "language": transcript.get("language"),
        "text": normalize_text(str(transcript.get("text", ""))),
        "speaker_ids": speaker_ids,
        "speaker_count": len(speaker_ids),
        "known_speaker_count": len([speaker_id for speaker_id in speaker_ids if speaker_id != UNKNOWN_SPEAKER_ID]),
        "word_count": len(words),
        "segment_count": len(segments),
        "words": [asdict(word) for word in words],
        "segments": [asdict(segment) for segment in segments],
        "speaker_blocks": [asdict(block) for block in speaker_blocks],
        "speaker_diarization": transcript.get("speaker_diarization", []),
        "speaker_diarization_error": transcript.get("speaker_diarization_error"),
        "speaker_detection_mode": transcript.get("speaker_detection_mode"),
        "speaker_detection_error": transcript.get("speaker_detection_error"),
    }


def build_base_transcript_from_paths(
    transcript_json_path: Path,
    manifest_path: Path,
    frame_ticks: int,
) -> dict[str, Any]:
    transcript = json.loads(transcript_json_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return build_base_transcript(transcript, manifest, frame_ticks)


def write_base_transcript(base_transcript: dict[str, Any], output_path: Path) -> None:
    output_path.write_text(json.dumps(base_transcript, indent=2), encoding="utf-8")


def build_and_write_base_transcript(
    transcript_json_path: Path,
    manifest_path: Path,
    frame_ticks: int,
    output_path: Path,
) -> dict[str, Any]:
    base_transcript = build_base_transcript_from_paths(transcript_json_path, manifest_path, frame_ticks)
    write_base_transcript(base_transcript, output_path)
    return base_transcript


def relabel_base_transcript_speakers(
    base_transcript: dict[str, Any],
    *,
    word_speaker_ids: list[str] | None = None,
    segment_speaker_ids: list[str] | None = None,
    detection_mode: str | None = None,
    detection_error: str | None = None,
) -> dict[str, Any]:
    raw_words = base_transcript.get("words", [])
    raw_segments = base_transcript.get("segments", [])

    words: list[BaseTranscriptWord] = []
    if isinstance(raw_words, list):
        for index, raw_word in enumerate(raw_words):
            if not isinstance(raw_word, dict):
                continue
            raw_word_indexes = int(raw_word.get("word_index", index))
            speaker_id = (
                word_speaker_ids[index]
                if word_speaker_ids is not None and index < len(word_speaker_ids)
                else str(raw_word.get("speaker_id") or UNKNOWN_SPEAKER_ID)
            )
            words.append(
                BaseTranscriptWord(
                    word_index=raw_word_indexes,
                    segment_index=int(raw_word.get("segment_index", 0)),
                    speaker_id=speaker_id,
                    raw_text=str(raw_word.get("raw_text", "")),
                    text=normalize_text(str(raw_word.get("text", ""))),
                    start_seconds=float(raw_word.get("start_seconds", 0.0)),
                    end_seconds=float(raw_word.get("end_seconds", 0.0)),
                    start_ticks=int(raw_word.get("start_ticks", 0)),
                    end_ticks=int(raw_word.get("end_ticks", 0)),
                    start_timecode=str(raw_word.get("start_timecode", "00:00:00:00")),
                    end_timecode=str(raw_word.get("end_timecode", "00:00:00:00")),
                )
            )

    segments: list[BaseTranscriptSegment] = []
    if isinstance(raw_segments, list):
        for index, raw_segment in enumerate(raw_segments):
            if not isinstance(raw_segment, dict):
                continue
            speaker_id = (
                segment_speaker_ids[index]
                if segment_speaker_ids is not None and index < len(segment_speaker_ids)
                else str(raw_segment.get("speaker_id") or UNKNOWN_SPEAKER_ID)
            )
            raw_word_indexes = raw_segment.get("word_indexes", ())
            word_indexes = tuple(int(item) for item in raw_word_indexes) if isinstance(raw_word_indexes, (list, tuple)) else ()
            segments.append(
                BaseTranscriptSegment(
                    segment_index=int(raw_segment.get("segment_index", index)),
                    speaker_id=speaker_id,
                    raw_text=str(raw_segment.get("raw_text", "")),
                    text=normalize_text(str(raw_segment.get("text", ""))),
                    start_seconds=float(raw_segment.get("start_seconds", 0.0)),
                    end_seconds=float(raw_segment.get("end_seconds", 0.0)),
                    start_ticks=int(raw_segment.get("start_ticks", 0)),
                    end_ticks=int(raw_segment.get("end_ticks", 0)),
                    start_timecode=str(raw_segment.get("start_timecode", "00:00:00:00")),
                    end_timecode=str(raw_segment.get("end_timecode", "00:00:00:00")),
                    word_indexes=word_indexes,
                )
            )

    if not segments:
        raise PodcastEditorError("Base transcript did not contain any segments to relabel.")

    speaker_blocks = build_base_speaker_blocks(words, segments)
    speaker_ids = sorted({entry.speaker_id for entry in [*words, *segments]} or {UNKNOWN_SPEAKER_ID})

    updated = dict(base_transcript)
    updated["speaker_ids"] = speaker_ids
    updated["speaker_count"] = len(speaker_ids)
    updated["known_speaker_count"] = len([speaker_id for speaker_id in speaker_ids if speaker_id != UNKNOWN_SPEAKER_ID])
    updated["words"] = [asdict(word) for word in words]
    updated["segments"] = [asdict(segment) for segment in segments]
    updated["speaker_blocks"] = [asdict(block) for block in speaker_blocks]
    if detection_mode is not None:
        updated["speaker_detection_mode"] = detection_mode
    if detection_error is not None:
        updated["speaker_detection_error"] = detection_error
    return updated


def ends_sentence(word_text: str) -> bool:
    trimmed = normalize_text(word_text).rstrip("\"')]}”’")
    return trimmed.endswith(SENTENCE_END_PUNCTUATION)


def build_sentence_timed_lines_from_base(base_transcript: dict[str, Any]) -> list[str]:
    words = base_transcript.get("words", [])
    if not isinstance(words, list) or not words:
        return []

    lines: list[str] = []
    sentence_tokens: list[str] = []
    sentence_start_timecode: str | None = None
    sentence_end_timecode: str | None = None

    for index, word in enumerate(words):
        if not isinstance(word, dict):
            continue
        raw_text = str(word.get("raw_text") or word.get("text") or "")
        if not normalize_text(raw_text):
            continue
        if sentence_start_timecode is None:
            sentence_start_timecode = str(word["start_timecode"])
        sentence_tokens.append(raw_text)
        sentence_end_timecode = str(word["end_timecode"])

        next_word = words[index + 1] if index + 1 < len(words) and isinstance(words[index + 1], dict) else None
        gap_seconds = 0.0
        if next_word is not None:
            gap_seconds = max(0.0, float(next_word["start_seconds"]) - float(word["end_seconds"]))

        should_break = (
            next_word is None
            or ends_sentence(raw_text)
            or (
                gap_seconds >= SENTENCE_BREAK_GAP_SECONDS
                and len(sentence_tokens) >= MIN_WORDS_FOR_GAP_BREAK
            )
            or len(sentence_tokens) >= MAX_WORDS_PER_TRANSCRIPT_LINE
        )
        if not should_break or sentence_start_timecode is None or sentence_end_timecode is None:
            continue

        text = normalize_text("".join(sentence_tokens))
        if text:
            lines.append(f"{sentence_start_timecode} - {sentence_end_timecode} | {text}")

        sentence_tokens = []
        sentence_start_timecode = None
        sentence_end_timecode = None

    return lines


def build_segment_timed_lines_from_base(base_transcript: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for segment in base_transcript.get("segments", []):
        if not isinstance(segment, dict):
            continue
        text = normalize_text(segment.get("text", ""))
        if not text:
            continue
        lines.append(f"{segment['start_timecode']} - {segment['end_timecode']} | {text}")
    return lines


def build_timecoded_transcript_from_base(base_transcript: dict[str, Any]) -> tuple[str, int]:
    lines = build_sentence_timed_lines_from_base(base_transcript)
    if not lines:
        lines = build_segment_timed_lines_from_base(base_transcript)
    if not lines:
        raise PodcastEditorError("Base transcript did not contain any parseable lines.")
    return "\n".join(lines) + "\n", len(lines)


def build_timecoded_transcript(
    transcript_json_path: Path,
    manifest_path: Path,
    frame_ticks: int,
) -> tuple[str, int]:
    base_transcript = build_base_transcript_from_paths(transcript_json_path, manifest_path, frame_ticks)
    return build_timecoded_transcript_from_base(base_transcript)


def format_speaker_block_line(block: dict[str, Any]) -> str:
    return f"{block['speaker_id']} {block['start_timecode']} - {block['end_timecode']} | {block['text']}"


def format_editorial_block_line(block: dict[str, Any], line_index: int) -> str:
    return (
        f"B{line_index + 1:04d} | {block['speaker_id']} | {block['start_timecode']} - {block['end_timecode']} | "
        f"{block['text']}"
    )


def write_speaker_blocks_text(base_transcript: dict[str, Any], output_path: Path) -> None:
    blocks = base_transcript.get("speaker_blocks", [])
    lines = [format_speaker_block_line(block) for block in blocks if isinstance(block, dict) and block.get("text")]
    output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def build_editorial_blocks_text_from_base(base_transcript: dict[str, Any]) -> tuple[str, int]:
    blocks = base_transcript.get("speaker_blocks", [])
    lines = [
        format_editorial_block_line(block, line_index)
        for line_index, block in enumerate(blocks)
        if isinstance(block, dict) and normalize_text(str(block.get("text", "")))
    ]
    return "\n".join(lines) + ("\n" if lines else ""), len(lines)


def write_editorial_blocks_text(base_transcript: dict[str, Any], output_path: Path) -> None:
    text, _ = build_editorial_blocks_text_from_base(base_transcript)
    output_path.write_text(text, encoding="utf-8")
