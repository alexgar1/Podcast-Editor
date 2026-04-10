#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from base_transcript import (
    MAX_WORDS_PER_TRANSCRIPT_LINE,
    MIN_WORDS_FOR_GAP_BREAK,
    SENTENCE_BREAK_GAP_SECONDS,
    base_transcript_json_path,
    build_base_transcript_from_paths,
    build_and_write_base_transcript,
    build_editorial_blocks_text_from_base,
    build_timecoded_transcript_from_base,
    editorial_blocks_text_path,
    normalize_text,
    rendered_audio_ticks_to_timeline_ticks,
    speaker_blocks_text_path,
    ticks_to_timecode,
    write_base_transcript,
    write_speaker_blocks_text,
)
from podcast_editor import (
    PodcastEditorError,
    TickRange,
    TICKS_PER_SECOND,
    build_template_context,
    create_concision_sequence,
    create_selects_sequence,
    default_output_path,
    extract_timecode_ranges,
    extract_timecode_ranges_or_empty,
    normalize_ranges,
    resolve_working_project,
    save_prproj,
    timecode_to_ticks,
)
from speaker_diarization import (
    SpeakerDiarizationConfig,
    add_speaker_diarization_args,
    speaker_diarization_config_from_args,
)
from transcribe_sequence import (
    extract_audio_segments,
    render_sequence_audio,
    sanitize_filename,
    transcribe_with_whisper,
    write_segment_manifest,
)

DEFAULT_CLAUDE_MODEL_CANDIDATES = (
    "claude-sonnet-4-20250514",
    "claude-3-7-sonnet-latest",
    "claude-3-5-sonnet-latest",
)
MAX_SELECTION_SENTENCES = 6
FIRST_PASS_CHUNK_SENTENCES = 180
FIRST_PASS_CHUNK_OVERLAP_SENTENCES = MAX_SELECTION_SENTENCES - 1
SHORTLIST_LIMIT = 40
FINAL_SELECTION_COUNT = 20
TIMECODED_TRANSCRIPT_LINE_PATTERN = re.compile(
    r"^(?P<start>(?:\d{2}:){3}\d{2})\s*[–—-]\s*(?P<end>(?:\d{2}:){3}\d{2})\s*\|\s*(?P<text>.+)$"
)
EDITORIAL_BLOCK_LINE_PATTERN = re.compile(
    r"^(?P<block_id>B\d{4})\s*\|\s*(?P<speaker_id>[^|]+?)\s*\|\s*"
    r"(?P<start>(?:\d{2}:){3}\d{2})\s*[–—-]\s*(?P<end>(?:\d{2}:){3}\d{2})\s*\|\s*(?P<text>.+)$"
)
JSON_BLOCK_PATTERN = re.compile(r"```(?:json)?\s*(.+?)\s*```", re.DOTALL)
RUBRIC_SCORE_KEYS = (
    "curiosity_gap",
    "emotional_peak",
    "bold_claim",
    "vivid_storytelling",
    "high_stakes_setup",
    "quotable_shareable",
)


@dataclass(frozen=True)
class TranscriptSentence:
    sentence_id: str
    index: int
    start_timecode: str
    end_timecode: str
    text: str


@dataclass(frozen=True)
class TranscriptBlock:
    block_id: str
    index: int
    speaker_id: str
    start_timecode: str
    end_timecode: str
    text: str


@dataclass(frozen=True)
class ClipCandidate:
    start_index: int
    end_index: int
    title: str
    total_score: int
    rubric_scores: dict[str, int]
    rationale: str
    candidate_id: str = ""


@dataclass(frozen=True)
class RemovalCandidate:
    start_index: int
    end_index: int
    kind: str
    rationale: str


def emit_status(message: str) -> None:
    print(f"[status] {message}", file=sys.stderr, flush=True)


def emit_stage_status(step: int, total: int, message: str) -> None:
    emit_status(f"[{step}/{total}] {message}")


def emit_progress_status(label: str, index: int, total: int, message: str) -> None:
    emit_status(f"[{label} {index}/{total}] {message}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Transcribe a source Premiere sequence with Whisper, send the transcript to Claude for highlight "
            "selection and/or broad concision review, then assemble the chosen time ranges into new Premiere sequences."
        )
    )
    parser.add_argument("--project", type=Path, help="Path to the Premiere .prproj file.")
    parser.add_argument(
        "--output-project",
        type=Path,
        help="Where to write the updated .prproj. Defaults to the original project path.",
    )
    parser.add_argument(
        "--sequence",
        help=(
            "Optional source sequence name. If omitted, the tool uses the only sequence in the project or the only "
            "sequence whose name does not look like a generated selects sequence."
        ),
    )
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        help="Directory for source transcript artifacts and Claude outputs. Defaults to <output-project-stem>.analysis/.",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        help="Optional prompt file. Defaults to claude_selects_prompt.md next to this script.",
    )
    parser.add_argument(
        "--create-selects-sequence",
        action="store_true",
        help="Run the trailer-selects workflow and create a selects sequence.",
    )
    parser.add_argument(
        "--create-concision-sequence",
        action="store_true",
        help=(
            "Run a transcript-driven broad concision pass that identifies removable ranges and creates a new "
            "cut-and-lift sibling sequence with timeline gaps at those ranges."
        ),
    )
    parser.add_argument(
        "--concision-prompt-file",
        type=Path,
        help="Optional prompt file for the broad concision pass. Defaults to claude_concision_prompt.md.",
    )
    parser.add_argument(
        "--whisper-model",
        default="base.en",
        help="Whisper model name, for example tiny.en, base.en, small.en, medium.en, turbo.",
    )
    parser.add_argument(
        "--whisper-language",
        default="en",
        help="Language hint for Whisper. Defaults to en.",
    )
    parser.add_argument(
        "--claude-model",
        help="Claude model override. If omitted, the tool tries a small fallback list of Sonnet models.",
    )
    parser.add_argument(
        "--claude-max-tokens",
        type=int,
        default=6000,
        help="Max output tokens for the Claude response.",
    )
    parser.add_argument(
        "--claude-temperature",
        type=float,
        default=0.2,
        help="Sampling temperature for the Claude response.",
    )
    parser.add_argument(
        "--claude-response-file",
        type=Path,
        help="Optional pre-generated Claude output to reuse instead of calling the API.",
    )
    parser.add_argument(
        "--concision-response-file",
        type=Path,
        help=(
            "Optional pre-generated Claude output for the broad concision pass. Accepts the JSON schema used by the "
            "workflow or a text file containing timecode ranges."
        ),
    )
    add_speaker_diarization_args(parser)
    return parser.parse_args(argv)


def load_dotenv_if_present(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or key in os.environ:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ[key] = value


def require_anthropic_api_key(repo_root: Path) -> str:
    load_dotenv_if_present(repo_root / ".env")
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise PodcastEditorError("ANTHROPIC_API_KEY is not set. Export it or place it in .env.")
    return api_key


def default_analysis_dir(output_project: Path) -> Path:
    return output_project.parent / f"{output_project.stem}.analysis"


def default_prompt_file() -> Path:
    return Path(__file__).resolve().with_name("claude_selects_prompt.md")


def default_concision_prompt_file() -> Path:
    return Path(__file__).resolve().with_name("claude_concision_prompt.md")


def default_first_pass_prompt_file() -> Path:
    return Path(__file__).resolve().with_name("claude_first_pass_prompt.md")


def default_final_pass_prompt_file() -> Path:
    return Path(__file__).resolve().with_name("claude_final_ranking_prompt.md")


def build_timecoded_transcript(
    transcript_json_path: Path,
    manifest_path: Path,
    frame_ticks: int,
) -> tuple[str, int]:
    base_transcript = build_base_transcript_from_paths(transcript_json_path, manifest_path, frame_ticks)
    return build_timecoded_transcript_from_base(base_transcript)


def render_prompt_template(template_text: str, values: dict[str, Any]) -> str:
    rendered = template_text
    for key, value in values.items():
        rendered = rendered.replace(f"{{{{{key}}}}}", str(value))

    remaining_tokens = sorted(set(re.findall(r"\{\{[A-Za-z0-9_]+\}\}", rendered)))
    if remaining_tokens:
        raise PodcastEditorError(
            "Prompt template contains unreplaced placeholders: " + ", ".join(remaining_tokens)
        )
    return rendered


def parse_timecoded_transcript(transcript_text: str) -> list[TranscriptSentence]:
    sentences: list[TranscriptSentence] = []
    for raw_line in transcript_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = TIMECODED_TRANSCRIPT_LINE_PATTERN.match(line)
        if not match:
            continue
        text = normalize_text(match.group("text"))
        if not text:
            continue
        index = len(sentences)
        sentences.append(
            TranscriptSentence(
                sentence_id=f"S{index + 1:04d}",
                index=index,
                start_timecode=match.group("start"),
                end_timecode=match.group("end"),
                text=text,
            )
        )

    if not sentences:
        raise PodcastEditorError("Timecoded transcript did not contain any parseable sentence lines.")
    return sentences


def parse_editorial_blocks_transcript(transcript_text: str) -> list[TranscriptBlock]:
    blocks: list[TranscriptBlock] = []
    for raw_line in transcript_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = EDITORIAL_BLOCK_LINE_PATTERN.match(line)
        if not match:
            continue
        text = normalize_text(match.group("text"))
        if not text:
            continue
        blocks.append(
            TranscriptBlock(
                block_id=match.group("block_id"),
                index=len(blocks),
                speaker_id=normalize_text(match.group("speaker_id")),
                start_timecode=match.group("start"),
                end_timecode=match.group("end"),
                text=text,
            )
        )

    if not blocks:
        raise PodcastEditorError("Editorial block transcript did not contain any parseable lines.")
    return blocks


def build_sentence_chunks(
    sentences: list[TranscriptSentence],
    max_chunk_sentences: int = FIRST_PASS_CHUNK_SENTENCES,
    overlap_sentences: int = FIRST_PASS_CHUNK_OVERLAP_SENTENCES,
) -> list[list[TranscriptSentence]]:
    if not sentences:
        return []
    if max_chunk_sentences <= overlap_sentences:
        raise PodcastEditorError("Chunk size must be greater than the overlap size.")

    chunks: list[list[TranscriptSentence]] = []
    start = 0
    step = max_chunk_sentences - overlap_sentences
    while start < len(sentences):
        end = min(len(sentences), start + max_chunk_sentences)
        chunks.append(sentences[start:end])
        if end >= len(sentences):
            break
        start += step
    return chunks


def format_sentence_chunk(sentences: list[TranscriptSentence]) -> str:
    return "\n".join(
        f"{sentence.sentence_id} | {sentence.start_timecode} - {sentence.end_timecode} | {sentence.text}"
        for sentence in sentences
    )


def format_block_chunk(blocks: list[TranscriptBlock]) -> str:
    return "\n".join(
        (
            f"{block.block_id} | {block.speaker_id} | {block.start_timecode} - {block.end_timecode} | "
            f"{block.text}"
        )
        for block in blocks
    )


def extract_json_payload(text: str) -> dict[str, Any]:
    stripped = text.strip()
    candidates = [stripped]

    fenced = JSON_BLOCK_PATTERN.search(stripped)
    if fenced is not None:
        candidates.append(fenced.group(1).strip())

    object_start = stripped.find("{")
    object_end = stripped.rfind("}")
    if object_start != -1 and object_end != -1 and object_start < object_end:
        candidates.append(stripped[object_start : object_end + 1].strip())

    for candidate in candidates:
        if not candidate:
            continue
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload

    raise PodcastEditorError("Claude response did not contain a valid JSON object.")


def normalize_score(value: Any, minimum: int = 0, maximum: int = 5) -> int:
    try:
        score = int(round(float(value)))
    except (TypeError, ValueError):
        score = minimum
    return max(minimum, min(maximum, score))


def parse_rubric_scores(raw_scores: Any) -> dict[str, int]:
    scores: dict[str, int] = {}
    if not isinstance(raw_scores, dict):
        raw_scores = {}
    for key in RUBRIC_SCORE_KEYS:
        scores[key] = normalize_score(raw_scores.get(key, 0))
    return scores


def build_first_pass_prompt(
    prompt_template: str,
    editorial_prompt: str,
    chunk_sentences: list[TranscriptSentence],
    chunk_index: int,
    chunk_total: int,
    candidate_limit: int,
) -> str:
    return render_prompt_template(
        prompt_template,
        {
            "editorial_prompt": editorial_prompt,
            "chunk_index": chunk_index,
            "chunk_total": chunk_total,
            "max_selection_sentences": MAX_SELECTION_SENTENCES,
            "candidate_limit": candidate_limit,
            "transcript_chunk": format_sentence_chunk(chunk_sentences),
        },
    )


def build_final_pass_prompt(
    prompt_template: str,
    editorial_prompt: str,
    candidates: list[ClipCandidate],
    sentences: list[TranscriptSentence],
    selection_count: int,
) -> str:
    candidate_blocks: list[str] = []
    for candidate in candidates:
        start_sentence = sentences[candidate.start_index]
        end_sentence = sentences[candidate.end_index]
        text = " ".join(sentence.text for sentence in sentences[candidate.start_index : candidate.end_index + 1])
        candidate_blocks.append(
            "\n".join(
                [
                    f"{candidate.candidate_id}",
                    f"Timecode: {start_sentence.start_timecode} - {end_sentence.end_timecode}",
                    f"Sentence IDs: {start_sentence.sentence_id} - {end_sentence.sentence_id}",
                    f'Text: "{text}"',
                ]
            )
        )

    return render_prompt_template(
        prompt_template,
        {
            "editorial_prompt": editorial_prompt,
            "selection_count": selection_count,
            "candidate_shortlist": "\n\n".join(candidate_blocks),
        },
    )


def build_concision_prompt(prompt_template: str, transcript_blocks: list[TranscriptBlock]) -> str:
    return render_prompt_template(
        prompt_template,
        {
            "transcript_blocks": format_block_chunk(transcript_blocks),
        },
    )


def parse_first_pass_candidates(
    payload: dict[str, Any],
    chunk_sentences: list[TranscriptSentence],
    sentence_by_id: dict[str, TranscriptSentence],
) -> list[ClipCandidate]:
    chunk_ids = {sentence.sentence_id for sentence in chunk_sentences}
    chunk_start = chunk_sentences[0].index
    chunk_end = chunk_sentences[-1].index
    raw_candidates = payload.get("candidates", [])
    if not isinstance(raw_candidates, list):
        raise PodcastEditorError("Claude first-pass JSON did not contain a candidates list.")

    candidates: list[ClipCandidate] = []
    for item in raw_candidates:
        if not isinstance(item, dict):
            continue
        start_id = str(item.get("start_id", "")).strip()
        end_id = str(item.get("end_id", "")).strip()
        if start_id not in chunk_ids or end_id not in chunk_ids:
            continue
        start_sentence = sentence_by_id.get(start_id)
        end_sentence = sentence_by_id.get(end_id)
        if start_sentence is None or end_sentence is None:
            continue
        start_index = min(start_sentence.index, end_sentence.index)
        end_index = max(start_sentence.index, end_sentence.index)
        if start_index < chunk_start or end_index > chunk_end:
            continue
        if end_index - start_index + 1 > MAX_SELECTION_SENTENCES:
            continue

        rubric_scores = parse_rubric_scores(item.get("scores"))
        total_score = normalize_score(item.get("total_score", sum(rubric_scores.values())), minimum=0, maximum=30)
        title = normalize_text(str(item.get("title", "")).strip()) or "Untitled selection"
        rationale = normalize_text(str(item.get("rationale", "")).strip())
        candidates.append(
            ClipCandidate(
                start_index=start_index,
                end_index=end_index,
                title=title,
                total_score=total_score,
                rubric_scores=rubric_scores,
                rationale=rationale,
            )
        )

    if not candidates:
        raise PodcastEditorError("Claude first-pass response did not contain any valid candidates.")
    return candidates


def normalize_removal_kind(value: Any) -> str:
    normalized = normalize_text(str(value or "")).lower()
    if normalized in {"redundant", "irrelevant", "both"}:
        return normalized
    return "both"


def parse_concision_candidates(
    payload: dict[str, Any],
    blocks: list[TranscriptBlock],
    block_by_id: dict[str, TranscriptBlock],
) -> list[RemovalCandidate]:
    raw_candidates = payload.get("removals", [])
    if not isinstance(raw_candidates, list):
        raise PodcastEditorError("Claude concision JSON did not contain a removals list.")

    known_ids = {block.block_id for block in blocks}
    candidates: list[RemovalCandidate] = []
    for item in raw_candidates:
        if not isinstance(item, dict):
            continue
        start_id = str(item.get("start_id", "")).strip()
        end_id = str(item.get("end_id", "")).strip()
        if start_id not in known_ids or end_id not in known_ids:
            continue
        start_block = block_by_id.get(start_id)
        end_block = block_by_id.get(end_id)
        if start_block is None or end_block is None:
            continue
        start_index = min(start_block.index, end_block.index)
        end_index = max(start_block.index, end_block.index)
        candidates.append(
            RemovalCandidate(
                start_index=start_index,
                end_index=end_index,
                kind=normalize_removal_kind(item.get("kind")),
                rationale=normalize_text(str(item.get("reason", "")).strip()),
            )
        )
    return candidates


def dedupe_candidates(candidates: list[ClipCandidate]) -> list[ClipCandidate]:
    by_range: dict[tuple[int, int], ClipCandidate] = {}
    for candidate in candidates:
        key = (candidate.start_index, candidate.end_index)
        existing = by_range.get(key)
        if existing is None or candidate.total_score > existing.total_score:
            by_range[key] = candidate
    return list(by_range.values())


def merge_candidate_pair(first: ClipCandidate, second: ClipCandidate) -> ClipCandidate:
    stronger = first if first.total_score >= second.total_score else second
    merged_scores = {
        key: max(first.rubric_scores.get(key, 0), second.rubric_scores.get(key, 0)) for key in RUBRIC_SCORE_KEYS
    }
    rationale_parts = [part for part in [first.rationale, second.rationale] if part]
    rationale = " ".join(dict.fromkeys(rationale_parts))
    return ClipCandidate(
        start_index=min(first.start_index, second.start_index),
        end_index=max(first.end_index, second.end_index),
        title=stronger.title,
        total_score=max(first.total_score, second.total_score),
        rubric_scores=merged_scores,
        rationale=rationale,
    )


def merge_adjacent_candidates(candidates: list[ClipCandidate]) -> list[ClipCandidate]:
    if not candidates:
        return []

    merged: list[ClipCandidate] = []
    for candidate in sorted(candidates, key=lambda item: (item.start_index, item.end_index, -item.total_score)):
        if not merged:
            merged.append(candidate)
            continue

        previous = merged[-1]
        union_start = min(previous.start_index, candidate.start_index)
        union_end = max(previous.end_index, candidate.end_index)
        union_length = union_end - union_start + 1
        if candidate.start_index <= previous.end_index + 1 and union_length <= MAX_SELECTION_SENTENCES:
            merged[-1] = merge_candidate_pair(previous, candidate)
            continue
        merged.append(candidate)

    return merged


def overlap_length(first: ClipCandidate, second: ClipCandidate) -> int:
    return max(0, min(first.end_index, second.end_index) - max(first.start_index, second.start_index) + 1)


def build_shortlist(candidates: list[ClipCandidate], limit: int) -> list[ClipCandidate]:
    shortlist: list[ClipCandidate] = []
    for candidate in sorted(candidates, key=lambda item: (-item.total_score, item.start_index, item.end_index)):
        if any(
            overlap_length(candidate, existing) >= min(
                candidate.end_index - candidate.start_index + 1,
                existing.end_index - existing.start_index + 1,
            )
            for existing in shortlist
        ):
            continue
        shortlist.append(candidate)
        if len(shortlist) >= limit:
            break
    return sorted(shortlist, key=lambda item: (item.start_index, item.end_index))


def assign_candidate_ids(candidates: list[ClipCandidate]) -> list[ClipCandidate]:
    return [replace(candidate, candidate_id=f"C{index + 1:03d}") for index, candidate in enumerate(candidates)]


def serialize_removal_candidate(candidate: RemovalCandidate, blocks: list[TranscriptBlock]) -> dict[str, Any]:
    start_block = blocks[candidate.start_index]
    end_block = blocks[candidate.end_index]
    return {
        "start_id": start_block.block_id,
        "end_id": end_block.block_id,
        "start_timecode": start_block.start_timecode,
        "end_timecode": end_block.end_timecode,
        "kind": candidate.kind,
        "reason": candidate.rationale,
        "text": " ".join(block.text for block in blocks[candidate.start_index : candidate.end_index + 1]),
    }


def serialize_candidate(candidate: ClipCandidate, sentences: list[TranscriptSentence]) -> dict[str, Any]:
    start_sentence = sentences[candidate.start_index]
    end_sentence = sentences[candidate.end_index]
    return {
        "candidate_id": candidate.candidate_id,
        "start_sentence_id": start_sentence.sentence_id,
        "end_sentence_id": end_sentence.sentence_id,
        "start_timecode": start_sentence.start_timecode,
        "end_timecode": end_sentence.end_timecode,
        "title": candidate.title,
        "total_score": candidate.total_score,
        "scores": candidate.rubric_scores,
        "rationale": candidate.rationale,
        "text": " ".join(sentence.text for sentence in sentences[candidate.start_index : candidate.end_index + 1]),
    }


def parse_final_ranked_candidates(
    payload: dict[str, Any],
    candidates_by_id: dict[str, ClipCandidate],
    selection_count: int,
) -> list[tuple[ClipCandidate, str, str]]:
    raw_selections = payload.get("selections", [])
    if not isinstance(raw_selections, list):
        raise PodcastEditorError("Claude final-pass JSON did not contain a selections list.")

    ranked: list[tuple[ClipCandidate, str, str]] = []
    seen_ids: set[str] = set()
    for item in raw_selections:
        if not isinstance(item, dict):
            continue
        candidate_id = str(item.get("candidate_id", "")).strip()
        if candidate_id in seen_ids:
            continue
        candidate = candidates_by_id.get(candidate_id)
        if candidate is None:
            continue
        seen_ids.add(candidate_id)
        title = normalize_text(str(item.get("title", "")).strip()) or candidate.title
        rationale = normalize_text(str(item.get("rationale", "")).strip())
        ranked.append((candidate, title, rationale))
        if len(ranked) >= selection_count:
            break

    if len(ranked) != selection_count:
        raise PodcastEditorError(
            f"Claude final-pass response returned {len(ranked)} valid selections; expected {selection_count}."
        )
    return ranked


def format_ranked_selections(
    ranked_candidates: list[tuple[ClipCandidate, str, str]],
    sentences: list[TranscriptSentence],
) -> str:
    blocks: list[str] = []
    for index, (candidate, title, _) in enumerate(ranked_candidates, start=1):
        start_sentence = sentences[candidate.start_index]
        end_sentence = sentences[candidate.end_index]
        text = " ".join(sentence.text for sentence in sentences[candidate.start_index : candidate.end_index + 1])
        blocks.append(
            "\n".join(
                [
                    f"#{index} — {title}",
                    f"Timecode: {start_sentence.start_timecode} – {end_sentence.end_timecode}",
                    f'Text: "{text}"',
                ]
            )
        )
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def format_removal_candidates(candidates: list[RemovalCandidate], blocks: list[TranscriptBlock]) -> str:
    if not candidates:
        return "No removable ranges identified.\n"

    rendered: list[str] = []
    for index, candidate in enumerate(candidates, start=1):
        start_block = blocks[candidate.start_index]
        end_block = blocks[candidate.end_index]
        text = " ".join(block.text for block in blocks[candidate.start_index : candidate.end_index + 1])
        label = candidate.kind.title()
        rendered.append(
            "\n".join(
                [
                    f"#{index} — {label}",
                    f"Timecode: {start_block.start_timecode} – {end_block.end_timecode}",
                    f"Blocks: {start_block.block_id} - {end_block.block_id}",
                    f'Reason: "{candidate.rationale or "Broad removable section."}"',
                    f'Text: "{text}"',
                ]
            )
        )
    return "\n\n".join(rendered) + "\n"


def nomination_limit_for_chunk_count(chunk_count: int) -> int:
    if chunk_count <= 0:
        return FINAL_SELECTION_COUNT
    desired_raw_candidates = max(FINAL_SELECTION_COUNT * 2, 30)
    return max(4, math.ceil(desired_raw_candidates / chunk_count))


def load_prompt(prompt_file: Path | None) -> str:
    path = prompt_file or default_prompt_file()
    if not path.exists():
        raise PodcastEditorError(f"Prompt file was not found: {path}")
    prompt = path.read_text(encoding="utf-8").strip()
    if not prompt:
        raise PodcastEditorError(f"Prompt file is empty: {path}")
    return prompt


def anthropic_headers(api_key: str) -> dict[str, str]:
    return {
        "content-type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }


def extract_text_blocks(response_payload: dict[str, Any]) -> str:
    chunks: list[str] = []
    for item in response_payload.get("content", []):
        if item.get("type") == "text" and item.get("text"):
            chunks.append(item["text"])
    text = "\n".join(chunks).strip()
    if not text:
        raise PodcastEditorError("Claude response did not contain any text blocks.")
    return text


def call_claude(
    api_key: str,
    content: str,
    model: str,
    max_tokens: int,
    temperature: float,
) -> str:
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [
            {
                "role": "user",
                "content": content,
            }
        ],
    }
    request = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(payload).encode("utf-8"),
        headers=anthropic_headers(api_key),
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise PodcastEditorError(f"Claude API request failed for model {model}: HTTP {exc.code} {body}") from exc
    except urllib.error.URLError as exc:
        raise PodcastEditorError(f"Claude API request failed for model {model}: {exc.reason}") from exc
    return extract_text_blocks(response_payload)


def call_claude_with_fallbacks(
    api_key: str,
    content: str,
    model_override: str | None,
    max_tokens: int,
    temperature: float,
) -> tuple[str, str]:
    candidates = [model_override] if model_override else list(DEFAULT_CLAUDE_MODEL_CANDIDATES)
    last_error: PodcastEditorError | None = None

    for model in candidates:
        if not model:
            continue
        try:
            return model, call_claude(api_key, content, model, max_tokens, temperature)
        except PodcastEditorError as exc:
            last_error = exc
            if "HTTP 404" in str(exc) or "HTTP 400" in str(exc):
                continue
            raise

    if last_error is not None:
        raise last_error
    raise PodcastEditorError("No Claude model candidates were available.")


def select_highlights_with_claude(
    api_key: str,
    editorial_prompt: str,
    first_pass_prompt_template: str,
    final_pass_prompt_template: str,
    transcript_text: str,
    model_override: str | None,
    max_tokens: int,
    temperature: float,
    analysis_dir: Path,
) -> dict[str, Any]:
    sentences = parse_timecoded_transcript(transcript_text)
    sentence_by_id = {sentence.sentence_id: sentence for sentence in sentences}
    chunks = build_sentence_chunks(sentences)
    nomination_limit = nomination_limit_for_chunk_count(len(chunks))
    emit_status(
        f"Prepared {len(sentences)} transcript sentences across {len(chunks)} Claude highlight chunks"
    )

    pass1_responses: list[dict[str, Any]] = []
    all_candidates: list[ClipCandidate] = []
    models_used: list[str] = []

    for chunk_index, chunk in enumerate(chunks, start=1):
        emit_status(f"Submitting highlight chunk {chunk_index}/{len(chunks)} to Claude")
        prompt = build_first_pass_prompt(
            first_pass_prompt_template,
            editorial_prompt,
            chunk,
            chunk_index,
            len(chunks),
            nomination_limit,
        )
        prompt_path = analysis_dir / f"claude_pass1_chunk_{chunk_index:02d}.prompt.md"
        prompt_path.write_text(prompt + "\n", encoding="utf-8")
        model, raw_text = call_claude_with_fallbacks(
            api_key,
            prompt,
            model_override,
            max_tokens,
            temperature,
        )
        models_used.append(model)
        payload = extract_json_payload(raw_text)
        chunk_candidates = parse_first_pass_candidates(payload, chunk, sentence_by_id)
        all_candidates.extend(chunk_candidates)
        emit_status(
            f"Claude returned {len(chunk_candidates)} highlight candidates for chunk {chunk_index}/{len(chunks)}"
        )

        response_path = analysis_dir / f"claude_pass1_chunk_{chunk_index:02d}.json"
        response_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        pass1_responses.append(
            {
                "chunk": chunk_index,
                "model": model,
                "prompt_path": str(prompt_path),
                "response_path": str(response_path),
                "candidate_count": len(chunk_candidates),
            }
        )

    merged_candidates = merge_adjacent_candidates(dedupe_candidates(all_candidates))
    shortlist = assign_candidate_ids(build_shortlist(merged_candidates, SHORTLIST_LIMIT))
    if not shortlist:
        raise PodcastEditorError("Claude first pass did not produce any shortlist candidates.")
    emit_status(f"Built highlight shortlist with {len(shortlist)} candidates")

    shortlist_path = analysis_dir / "claude_shortlist.json"
    shortlist_path.write_text(
        json.dumps([serialize_candidate(candidate, sentences) for candidate in shortlist], indent=2) + "\n",
        encoding="utf-8",
    )

    selection_count = min(FINAL_SELECTION_COUNT, len(shortlist))
    final_prompt = build_final_pass_prompt(
        final_pass_prompt_template,
        editorial_prompt,
        shortlist,
        sentences,
        selection_count,
    )
    final_prompt_path = analysis_dir / "claude_final_ranking.prompt.md"
    final_prompt_path.write_text(final_prompt + "\n", encoding="utf-8")
    emit_status(f"Submitting final highlight ranking request for {selection_count} picks")
    final_model, final_raw_text = call_claude_with_fallbacks(
        api_key,
        final_prompt,
        model_override,
        max_tokens,
        temperature,
    )
    models_used.append(final_model)
    final_payload = extract_json_payload(final_raw_text)
    final_payload_path = analysis_dir / "claude_final_ranking.json"
    final_payload_path.write_text(json.dumps(final_payload, indent=2) + "\n", encoding="utf-8")

    candidate_by_id = {candidate.candidate_id: candidate for candidate in shortlist}
    ranked_candidates = parse_final_ranked_candidates(final_payload, candidate_by_id, selection_count)
    final_text = format_ranked_selections(ranked_candidates, sentences)
    emit_status(f"Claude finalized {len(ranked_candidates)} highlight selections")
    selected_timecode_ranges = [
        (
            sentences[candidate.start_index].start_timecode,
            sentences[candidate.end_index].end_timecode,
        )
        for candidate, _, _ in ranked_candidates
    ]

    return {
        "models_used": models_used,
        "final_model": final_model,
        "final_text": final_text,
        "ranked_candidates": ranked_candidates,
        "selected_timecode_ranges": selected_timecode_ranges,
        "pass1_responses": pass1_responses,
        "shortlist_path": shortlist_path,
        "final_prompt_path": final_prompt_path,
        "final_payload_path": final_payload_path,
    }


def identify_concision_removals_with_claude(
    api_key: str,
    prompt_template: str,
    transcript_text: str,
    model_override: str | None,
    max_tokens: int,
    temperature: float,
    analysis_dir: Path,
) -> dict[str, Any]:
    blocks = parse_editorial_blocks_transcript(transcript_text)
    block_by_id = {block.block_id: block for block in blocks}
    emit_status(f"Prepared {len(blocks)} editorial blocks for a full-transcript concision review")

    prompt = build_concision_prompt(prompt_template, blocks)
    prompt_path = analysis_dir / "claude_concision.prompt.md"
    prompt_path.write_text(prompt + "\n", encoding="utf-8")
    emit_status("Submitting full editorial transcript to Claude for concision review")
    model, raw_text = call_claude_with_fallbacks(
        api_key,
        prompt,
        model_override,
        max_tokens,
        temperature,
    )
    payload = extract_json_payload(raw_text)
    candidates = sorted(
        parse_concision_candidates(payload, blocks, block_by_id),
        key=lambda item: (item.start_index, item.end_index),
    )
    emit_status(f"Claude identified {len(candidates)} removable ranges from the full transcript")

    payload_path = analysis_dir / "claude_concision.json"
    payload_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    candidate_path = analysis_dir / "claude_concision_candidates.json"
    candidate_path.write_text(
        json.dumps([serialize_removal_candidate(candidate, blocks) for candidate in candidates], indent=2) + "\n",
        encoding="utf-8",
    )
    final_text = format_removal_candidates(candidates, blocks)
    identified_ranges = [
        (
            blocks[candidate.start_index].start_timecode,
            blocks[candidate.end_index].end_timecode,
        )
        for candidate in candidates
    ]

    return {
        "models_used": [model],
        "final_model": model,
        "final_text": final_text,
        "removal_ranges": identified_ranges,
        "candidates": candidates,
        "prompt_path": prompt_path,
        "payload_path": payload_path,
        "response_path": candidate_path,
    }


def transcribe_source_sequence(
    root: Any,
    sequence_name: str,
    analysis_dir: Path,
    whisper_model: str,
    whisper_language: str,
    frame_ticks: int,
    speaker_config: SpeakerDiarizationConfig,
) -> dict[str, Any]:
    basename = sanitize_filename(sequence_name)
    audio_path = analysis_dir / f"{basename}.wav"
    manifest_path = analysis_dir / f"{basename}.segments.json"

    emit_status(f"Starting transcript build for '{sequence_name}'")
    emit_status(f"Extracting and mixing source audio for '{sequence_name}'")
    segments = extract_audio_segments(root, sequence_name)
    emit_status(f"Found {len(segments)} source audio segment(s) to mix for '{sequence_name}'")
    emit_status(f"Rendering source audio to {audio_path}")
    render_sequence_audio(segments, audio_path)
    write_segment_manifest(segments, manifest_path)
    emit_status(f"Running Whisper ({whisper_model}) on source audio")
    transcript_json_path = transcribe_with_whisper(
        audio_path,
        analysis_dir,
        whisper_model,
        whisper_language,
        speaker_config,
    )
    base_transcript_path = base_transcript_json_path(audio_path, analysis_dir)
    base_transcript = build_and_write_base_transcript(
        transcript_json_path,
        manifest_path,
        frame_ticks,
        base_transcript_path,
    )
    emit_status(f"Wrote base transcript to {base_transcript_path}")

    speaker_blocks_path = speaker_blocks_text_path(audio_path, analysis_dir)
    write_speaker_blocks_text(base_transcript, speaker_blocks_path)
    emit_status(f"Wrote speaker blocks to {speaker_blocks_path}")

    editorial_blocks, editorial_block_count = build_editorial_blocks_text_from_base(base_transcript)
    editorial_blocks_path = editorial_blocks_text_path(audio_path, analysis_dir)
    editorial_blocks_path.write_text(editorial_blocks, encoding="utf-8")
    emit_status(f"Wrote editorial blocks to {editorial_blocks_path}")

    timecoded_transcript, transcript_segment_count = build_timecoded_transcript_from_base(base_transcript)
    timecoded_transcript_path = analysis_dir / f"{basename}.timecoded.txt"
    timecoded_transcript_path.write_text(timecoded_transcript, encoding="utf-8")
    emit_status(f"Wrote timecoded transcript to {timecoded_transcript_path}")
    emit_status(
        f"Transcript ready with {transcript_segment_count} Whisper segment(s) and {editorial_block_count} editorial block(s)"
    )

    summary = {
        "audio_path": audio_path,
        "manifest_path": manifest_path,
        "transcript_json_path": transcript_json_path,
        "base_transcript": base_transcript,
        "base_transcript_path": base_transcript_path,
        "timecoded_transcript_path": timecoded_transcript_path,
        "transcript_segment_count": transcript_segment_count,
        "sequence_segment_count": len(segments),
        "timecoded_transcript": timecoded_transcript,
        "speaker_blocks_text_path": speaker_blocks_path,
        "editorial_blocks_text_path": editorial_blocks_path,
        "editorial_blocks_text": editorial_blocks,
        "editorial_block_count": editorial_block_count,
    }
    return summary


def run_concision_pass(
    root: Any,
    context: Any,
    transcript_summary: dict[str, Any],
    args: argparse.Namespace,
    analysis_dir: Path,
    require_api_key: Any,
) -> tuple[dict[str, Any] | None, Path | None, Path | None, str, int]:
    emit_status("Loading Claude concision prompt")
    concision_prompt_template = load_prompt(args.concision_prompt_file or default_concision_prompt_file())

    if args.concision_response_file is not None:
        emit_status(f"Using saved concision response from {args.concision_response_file}")
        concision_text = args.concision_response_file.read_text(encoding="utf-8")
        concision_model = "from-file"
        editorial_blocks = parse_editorial_blocks_transcript(str(transcript_summary["editorial_blocks_text"]))
        block_by_id = {block.block_id: block for block in editorial_blocks}
        try:
            emit_status("Parsing removable ranges from saved concision JSON")
            payload = extract_json_payload(concision_text)
            candidates = sorted(
                parse_concision_candidates(payload, editorial_blocks, block_by_id),
                key=lambda item: (item.start_index, item.end_index),
            )
            concision_text = format_removal_candidates(candidates, editorial_blocks)
            raw_concision_ranges = [
                TickRange(
                    timecode_to_ticks(editorial_blocks[candidate.start_index].start_timecode, context.frame_ticks),
                    timecode_to_ticks(editorial_blocks[candidate.end_index].end_timecode, context.frame_ticks),
                )
                for candidate in candidates
            ]
        except PodcastEditorError:
            emit_status("Saved concision response was not JSON; falling back to raw timecode parsing")
            raw_concision_ranges = [
                TickRange(
                    timecode_to_ticks(time_range.start, context.frame_ticks),
                    timecode_to_ticks(time_range.end, context.frame_ticks),
                )
                for time_range in extract_timecode_ranges_or_empty(concision_text)
            ]
        concision_prompt_path = None
    else:
        emit_status("Running broad concision review with Claude")
        concision_selection = identify_concision_removals_with_claude(
            require_api_key(),
            concision_prompt_template,
            str(transcript_summary["editorial_blocks_text"]),
            args.claude_model,
            args.claude_max_tokens,
            args.claude_temperature,
            analysis_dir,
        )
        concision_model = str(concision_selection["final_model"])
        concision_text = str(concision_selection["final_text"])
        concision_prompt_path = concision_selection.get("prompt_path")
        raw_concision_ranges = [
            TickRange(
                timecode_to_ticks(start_timecode, context.frame_ticks),
                timecode_to_ticks(end_timecode, context.frame_ticks),
            )
            for start_timecode, end_timecode in concision_selection["removal_ranges"]
        ]

    concision_response_path = analysis_dir / "claude_concision_response.txt"
    concision_response_path.write_text(concision_text, encoding="utf-8")
    emit_status(f"Wrote concision response to {concision_response_path}")
    normalized_concision_ranges = normalize_ranges(raw_concision_ranges, context.duration_ticks) if raw_concision_ranges else []
    concision_range_count = len(normalized_concision_ranges)
    emit_status(f"Normalized concision review to {concision_range_count} removable timeline ranges")

    if not normalized_concision_ranges:
        emit_status("No broad removable ranges were identified; skipping concision sequence creation")
        return None, concision_prompt_path, concision_response_path, concision_model, concision_range_count

    emit_status("Building concision sequence from identified removal ranges")
    concision_summary = create_concision_sequence(root, context, normalized_concision_ranges)
    return concision_summary, concision_prompt_path, concision_response_path, concision_model, concision_range_count


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    try:
        speaker_config = speaker_diarization_config_from_args(args)
    except ValueError as exc:
        raise PodcastEditorError(str(exc)) from exc

    if args.project is None:
        raise PodcastEditorError("Pass --project.")
    run_selects = args.create_selects_sequence
    run_concision = args.create_concision_sequence
    if not run_selects and not run_concision:
        raise PodcastEditorError(
            "Nothing to do. Pass --create-selects-sequence and/or --create-concision-sequence."
        )

    output_project = args.output_project or default_output_path(args.project)
    analysis_dir = args.analysis_dir or default_analysis_dir(output_project)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    emit_status(f"Starting sequence workflow from {args.project}")
    emit_status(f"Output project will be written to {output_project}")
    emit_status(f"Analysis artifacts will be written to {analysis_dir}")
    emit_status(f"Opening Premiere project {args.project}")
    source_project, root, context, used_autosave = resolve_working_project(args.project, output_project, args.sequence)
    if used_autosave:
        emit_status(f"Using autosave fallback {source_project}")
    else:
        emit_status(f"Using project file {source_project}")

    source_sequence_name = context.sequence.findtext("Name", "Sequence")
    if not source_sequence_name:
        raise PodcastEditorError("Source sequence is missing its name.")
    emit_status(f"Selected source sequence '{source_sequence_name}'")

    transcript_summary = transcribe_source_sequence(
        root,
        source_sequence_name,
        analysis_dir,
        args.whisper_model,
        args.whisper_language,
        context.frame_ticks,
        speaker_config,
    )

    api_key: str | None = None

    def require_api_key() -> str:
        nonlocal api_key
        if api_key is None:
            api_key = require_anthropic_api_key(Path.cwd())
        return api_key

    selects_summary: dict[str, Any] | None = None
    selects_prompt_path: Path | None = None
    selects_response_path: Path | None = None
    selects_claude_model = ""
    selects_range_count = 0

    concision_summary: dict[str, Any] | None = None
    concision_prompt_path: Path | None = None
    concision_response_path: Path | None = None
    concision_model = ""
    concision_range_count = 0

    project_modified = False

    if run_selects:
        emit_status("Loading Claude selection prompt")
        prompt = load_prompt(args.prompt_file)
        first_pass_prompt_template = load_prompt(default_first_pass_prompt_file())
        final_pass_prompt_template = load_prompt(default_final_pass_prompt_file())
        selects_prompt_path = analysis_dir / "claude_prompt.txt"
        selects_prompt_path.write_text(prompt + "\n", encoding="utf-8")
        first_pass_template_path = analysis_dir / "claude_first_pass_prompt_template.md"
        first_pass_template_path.write_text(first_pass_prompt_template + "\n", encoding="utf-8")
        final_pass_template_path = analysis_dir / "claude_final_ranking_prompt_template.md"
        final_pass_template_path.write_text(final_pass_prompt_template + "\n", encoding="utf-8")

        if args.claude_response_file is not None:
            emit_status(f"Using saved Claude response from {args.claude_response_file}")
            selects_text = args.claude_response_file.read_text(encoding="utf-8")
            selects_claude_model = "from-file"
            emit_status("Parsing selected time ranges from saved Claude response")
            raw_ranges = extract_timecode_ranges(selects_text)
            tick_ranges = normalize_ranges(
                [
                    TickRange(
                        timecode_to_ticks(time_range.start, context.frame_ticks),
                        timecode_to_ticks(time_range.end, context.frame_ticks),
                    )
                    for time_range in raw_ranges
                ],
                context.duration_ticks,
            )
        else:
            emit_status("Running multi-pass highlight selection with Claude")
            selection_summary = select_highlights_with_claude(
                require_api_key(),
                prompt,
                first_pass_prompt_template,
                final_pass_prompt_template,
                str(transcript_summary["timecoded_transcript"]),
                args.claude_model,
                args.claude_max_tokens,
                args.claude_temperature,
                analysis_dir,
            )
            selects_claude_model = str(selection_summary["final_model"])
            selects_text = str(selection_summary["final_text"])
            tick_ranges = normalize_ranges(
                [
                    TickRange(
                        timecode_to_ticks(start_timecode, context.frame_ticks),
                        timecode_to_ticks(end_timecode, context.frame_ticks),
                    )
                    for start_timecode, end_timecode in selection_summary["selected_timecode_ranges"]
                ],
                context.duration_ticks,
            )

        selects_response_path = analysis_dir / "claude_response.txt"
        selects_response_path.write_text(selects_text, encoding="utf-8")
        selects_range_count = len(tick_ranges)
        emit_status(f"Wrote highlight response to {selects_response_path}")
        emit_status(f"Normalized highlight selection to {selects_range_count} timeline ranges")

        emit_status("Building selects sequence from chosen ranges")
        selects_summary = create_selects_sequence(root, context, tick_ranges)
        project_modified = True

    if run_concision:
        (
            concision_summary,
            concision_prompt_path,
            concision_response_path,
            concision_model,
            concision_range_count,
        ) = run_concision_pass(root, context, transcript_summary, args, analysis_dir, require_api_key)
        if concision_summary is not None:
            project_modified = True

    if project_modified:
        emit_status(f"Saving updated project to {output_project}")
        save_prproj(root, output_project)
    else:
        emit_status("No Premiere sequence changes were required")
    emit_status("Finished")

    print(f"source_project={source_project}")
    print(f"used_autosave_fallback={'true' if used_autosave else 'false'}")
    print(f"source_sequence={source_sequence_name}")
    print(f"source_sequence_audio_segments={transcript_summary['sequence_segment_count']}")
    print(f"whisper_segments={transcript_summary['transcript_segment_count']}")
    print(f"source_audio={transcript_summary['audio_path']}")
    print(f"source_manifest={transcript_summary['manifest_path']}")
    print(f"source_transcript_json={transcript_summary['transcript_json_path']}")
    print(f"source_base_transcript={transcript_summary['base_transcript_path']}")
    print(f"source_transcript_timecoded={transcript_summary['timecoded_transcript_path']}")
    if "speaker_blocks_text_path" in transcript_summary:
        print(f"source_speaker_blocks={transcript_summary['speaker_blocks_text_path']}")
    if "editorial_blocks_text_path" in transcript_summary:
        print(f"source_editorial_blocks={transcript_summary['editorial_blocks_text_path']}")

    if selects_summary is not None:
        if selects_prompt_path is not None:
            print(f"claude_prompt={selects_prompt_path}")
        print(f"claude_model={selects_claude_model}")
        if selects_response_path is not None:
            print(f"claude_response={selects_response_path}")
        print(f"extracted_ranges={selects_range_count}")
        print(f"normalized_ranges={selects_summary['selected_ranges']}")
        print(f"new_sequence={selects_summary['new_sequence']}")
        print(f"segments_v1={selects_summary['video_segments']}")
        print(f"segments_a1={selects_summary['audio_segments']}")
        print(f"links={selects_summary['links']}")
        print(f"assembled_duration_ticks={selects_summary['assembled_duration_ticks']}")

    if run_concision:
        print(f"concision_sequence_created={'true' if concision_summary is not None else 'false'}")
        if concision_prompt_path is not None:
            print(f"concision_prompt={concision_prompt_path}")
        print(f"concision_model={concision_model}")
        if concision_response_path is not None:
            print(f"concision_response={concision_response_path}")
        print(f"concision_identified_ranges={concision_range_count}")
        if concision_summary is not None:
            print(f"concision_new_sequence={concision_summary['new_sequence']}")
            print(f"concision_segments_v1={concision_summary['video_segments']}")
            print(f"concision_segments_a1={concision_summary['audio_segments']}")
            print(f"concision_links={concision_summary['links']}")
            print(f"concision_assembled_duration_ticks={concision_summary['assembled_duration_ticks']}")
            print(f"concision_lifted_duration_ticks={concision_summary['lifted_duration_ticks']}")

    print(f"output_project={output_project}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
