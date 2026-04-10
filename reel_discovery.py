#!/usr/bin/env python3
"""Analyze a podcast video or audio file and identify sections with high viral reel potential.

This script:
  1. Extracts audio from a video file (or uses audio directly)
  2. Transcribes it with Whisper (word-level timestamps)
  3. Optionally runs speaker diarization
  4. Runs a multi-pass Claude analysis to discover, synthesize, and rank viral moments
  5. Outputs a ranked JSON report and a human-readable summary

Usage:
  conda run -n agent python reel_discovery.py --input video.mp4
  conda run -n agent python reel_discovery.py --input video.mp4 --output-dir ./reels_analysis
  conda run -n agent python reel_discovery.py --transcript existing_transcript.json --manifest segments.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Imports from the existing pcast-editor codebase
# ---------------------------------------------------------------------------

from base_transcript import (
    SENTENCE_BREAK_GAP_SECONDS,
    MAX_WORDS_PER_TRANSCRIPT_LINE,
    MIN_WORDS_FOR_GAP_BREAK,
    build_base_transcript,
    build_timecoded_transcript_from_base,
    normalize_text,
    ticks_to_timecode,
    write_base_transcript,
    write_speaker_blocks_text,
)
from podcast_editor import PodcastEditorError, TICKS_PER_SECOND
from speaker_diarization import (
    SpeakerDiarizationConfig,
    add_speaker_diarization_args,
    speaker_diarization_config_from_args,
)
from transcribe_sequence import (
    transcribe_with_whisper,
    diarize_speakers,
    annotate_transcript_with_speakers,
    build_speaker_blocks_from_words,
    build_speaker_blocks_from_segments,
    SpeakerTurn,
    preferred_torch_device,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_CLAUDE_MODEL_CANDIDATES = (
    "claude-sonnet-4-20250514",
    "claude-3-7-sonnet-latest",
    "claude-3-5-sonnet-latest",
)
# Discovery pass: how many sentences each chunk contains
DISCOVERY_CHUNK_SENTENCES = 180
# Overlap between adjacent chunks so we don't miss boundary moments
DISCOVERY_CHUNK_OVERLAP_SENTENCES = 5
# Max consecutive sentences in a single reel candidate
MAX_REEL_SENTENCES = 10
# How many candidates survive dedup / shortlisting before synthesis
SHORTLIST_LIMIT = 40
# How many sentences of surrounding context to include in the synthesis pass
SYNTHESIS_CONTEXT_WINDOW = 15

TIMECODED_TRANSCRIPT_LINE_PATTERN = re.compile(
    r"^(?P<start>(?:\d{2}:){3}\d{2})\s*[–—-]\s*(?P<end>(?:\d{2}:){3}\d{2})\s*\|\s*(?P<text>.+)$"
)
JSON_BLOCK_PATTERN = re.compile(r"```(?:json)?\s*(.+?)\s*```", re.DOTALL)

DISCOVERY_RUBRIC_KEYS = (
    "hook_potential",
    "self_contained_value",
    "emotional_peak",
    "contrarian_bold_claim",
    "shock_surprise",
    "quotable_shareable",
)
RANKING_RUBRIC_KEYS = (
    "scroll_stop_power",
    "retention",
    "shareability",
    "comment_magnetism",
    "rewatch_loop",
    "standalone_clarity",
)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TranscriptSentence:
    sentence_id: str
    index: int
    start_timecode: str
    end_timecode: str
    text: str


@dataclass(frozen=True)
class ReelCandidate:
    start_index: int
    end_index: int
    title: str
    total_score: int
    rubric_scores: dict[str, int]
    rationale: str
    candidate_id: str = ""


@dataclass(frozen=True)
class SynthesizedReel:
    candidate_id: str
    title: str
    text_overlay: str
    hook_start_id: str
    hook_end_id: str
    hook_text: str
    payoff_start_id: str
    payoff_end_id: str
    payoff_text: str
    hook_payoff_gap: str
    rationale: str


@dataclass(frozen=True)
class RankedReel:
    candidate_id: str
    virality_score: int
    ranking_scores: dict[str, int]
    rationale: str
    # Populated after joining with synthesis data
    title: str = ""
    text_overlay: str = ""
    hook_start_id: str = ""
    hook_end_id: str = ""
    hook_text: str = ""
    hook_start_timecode: str = ""
    hook_end_timecode: str = ""
    payoff_start_id: str = ""
    payoff_end_id: str = ""
    payoff_text: str = ""
    payoff_start_timecode: str = ""
    payoff_end_timecode: str = ""
    hook_payoff_gap: str = ""


# ---------------------------------------------------------------------------
# Status helpers
# ---------------------------------------------------------------------------


def emit_status(message: str) -> None:
    print(f"[status] {message}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze a podcast video/audio file and identify high-potential sections "
            "for short-form social media reels."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Path to a video or audio file. Audio is extracted via ffmpeg.",
    )
    parser.add_argument(
        "--transcript",
        type=Path,
        help=(
            "Path to a pre-existing Whisper JSON transcript. When provided, "
            "skips audio extraction and transcription."
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help=(
            "Path to a segment manifest JSON (from transcribe_sequence.py). "
            "Required when --transcript is used."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for analysis artifacts. Defaults to <input_stem>.reels/",
    )
    parser.add_argument(
        "--whisper-model",
        default="base.en",
        help="Whisper model name (e.g. tiny.en, base.en, small.en, medium.en, turbo).",
    )
    parser.add_argument(
        "--whisper-language",
        default="en",
        help="Language hint for Whisper. Defaults to en.",
    )
    parser.add_argument(
        "--claude-model",
        help="Claude model override. If omitted, tries a fallback list of Sonnet models.",
    )
    parser.add_argument(
        "--claude-max-tokens",
        type=int,
        default=8000,
        help="Max output tokens for Claude responses.",
    )
    parser.add_argument(
        "--claude-temperature",
        type=float,
        default=0.3,
        help="Sampling temperature for Claude responses.",
    )
    parser.add_argument(
        "--frame-rate",
        type=int,
        default=30,
        help="Video frame rate for timecode display. Defaults to 30.",
    )
    parser.add_argument(
        "--skip-synthesis",
        action="store_true",
        help="Stop after the discovery pass (skip hook/pay-off synthesis and ranking).",
    )
    parser.add_argument(
        "--discovery-response-file",
        type=Path,
        help="Pre-generated discovery pass JSON to reuse instead of calling Claude.",
    )
    add_speaker_diarization_args(parser)
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Environment / API key
# ---------------------------------------------------------------------------


def load_dotenv_if_present(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
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


def require_anthropic_api_key() -> str:
    load_dotenv_if_present(Path(__file__).resolve().parent / ".env")
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise PodcastEditorError("ANTHROPIC_API_KEY is not set. Export it or place it in .env.")
    return api_key


# ---------------------------------------------------------------------------
# Audio extraction
# ---------------------------------------------------------------------------


def extract_audio_from_video(input_path: Path, output_audio_path: Path) -> None:
    """Extract mono 16kHz WAV audio from a video (or audio) file via ffmpeg."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise PodcastEditorError("ffmpeg is not installed or not on PATH.")
    emit_status(f"Extracting audio from {input_path.name} → {output_audio_path.name}")
    command = [
        ffmpeg, "-y",
        "-i", str(input_path),
        "-vn",          # drop video
        "-ac", "1",     # mono
        "-ar", "16000", # 16kHz for Whisper
        "-f", "wav",
        str(output_audio_path),
    ]
    subprocess.run(command, check=True, capture_output=True)
    emit_status(f"Audio extracted: {output_audio_path}")


def build_identity_manifest(audio_path: Path) -> list[dict[str, Any]]:
    """Build a trivial 1:1 segment manifest for a standalone audio file.

    This maps rendered-audio ticks directly to timeline ticks so the base
    transcript builder can produce correct timecodes even though there is no
    Premiere sequence involved.
    """
    import wave

    try:
        with wave.open(str(audio_path), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            duration_seconds = frames / rate
    except Exception:
        # Fallback: probe with ffprobe
        ffprobe = shutil.which("ffprobe")
        if ffprobe is None:
            raise PodcastEditorError("Cannot determine audio duration — ffprobe not found.")
        result = subprocess.run(
            [ffprobe, "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", str(audio_path)],
            capture_output=True, text=True, check=True,
        )
        duration_seconds = float(result.stdout.strip())

    duration_ticks = int(round(duration_seconds * TICKS_PER_SECOND))
    return [
        {
            "index": 0,
            "timeline_start_ticks": 0,
            "timeline_end_ticks": duration_ticks,
            "source_start_ticks": 0,
            "source_end_ticks": duration_ticks,
            "source_path": str(audio_path),
        }
    ]


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------


def transcribe_audio(
    audio_path: Path,
    output_dir: Path,
    whisper_model: str,
    whisper_language: str,
    speaker_config: SpeakerDiarizationConfig,
    frame_ticks: int,
) -> dict[str, Any]:
    """Transcribe audio with Whisper and build the base transcript."""
    emit_status(f"Transcribing {audio_path.name} with Whisper ({whisper_model})")
    transcript_json_path = transcribe_with_whisper(
        audio_path,
        output_dir,
        whisper_model,
        whisper_language,
        speaker_config,
    )
    manifest = build_identity_manifest(audio_path)
    manifest_path = output_dir / f"{audio_path.stem}.segments.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    base_transcript = build_base_transcript(
        json.loads(transcript_json_path.read_text(encoding="utf-8")),
        manifest,
        frame_ticks,
    )
    base_transcript_path = output_dir / f"{audio_path.stem}.base_transcript.json"
    write_base_transcript(base_transcript, base_transcript_path)
    emit_status(f"Base transcript: {base_transcript_path}")

    speaker_blocks_path = output_dir / f"{audio_path.stem}.speaker_blocks.txt"
    write_speaker_blocks_text(base_transcript, speaker_blocks_path)

    timecoded_text, line_count = build_timecoded_transcript_from_base(base_transcript)
    timecoded_path = output_dir / f"{audio_path.stem}.timecoded.txt"
    timecoded_path.write_text(timecoded_text, encoding="utf-8")
    emit_status(f"Timecoded transcript: {timecoded_path} ({line_count} lines)")

    return {
        "transcript_json_path": transcript_json_path,
        "base_transcript": base_transcript,
        "base_transcript_path": base_transcript_path,
        "timecoded_transcript": timecoded_text,
        "timecoded_transcript_path": timecoded_path,
        "manifest_path": manifest_path,
        "manifest": manifest,
        "line_count": line_count,
    }


def load_existing_transcript(
    transcript_path: Path,
    manifest_path: Path,
    frame_ticks: int,
    output_dir: Path,
) -> dict[str, Any]:
    """Load a pre-existing Whisper transcript and build the base transcript."""
    emit_status(f"Loading existing transcript from {transcript_path}")
    transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    base_transcript = build_base_transcript(transcript, manifest, frame_ticks)
    base_transcript_path = output_dir / f"{transcript_path.stem}.base_transcript.json"
    write_base_transcript(base_transcript, base_transcript_path)

    timecoded_text, line_count = build_timecoded_transcript_from_base(base_transcript)
    timecoded_path = output_dir / f"{transcript_path.stem}.timecoded.txt"
    timecoded_path.write_text(timecoded_text, encoding="utf-8")
    emit_status(f"Timecoded transcript: {timecoded_path} ({line_count} lines)")

    return {
        "transcript_json_path": transcript_path,
        "base_transcript": base_transcript,
        "base_transcript_path": base_transcript_path,
        "timecoded_transcript": timecoded_text,
        "timecoded_transcript_path": timecoded_path,
        "manifest_path": manifest_path,
        "manifest": manifest,
        "line_count": line_count,
    }


# ---------------------------------------------------------------------------
# Transcript sentence parsing and chunking
# ---------------------------------------------------------------------------


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


def build_sentence_chunks(
    sentences: list[TranscriptSentence],
    max_chunk: int = DISCOVERY_CHUNK_SENTENCES,
    overlap: int = DISCOVERY_CHUNK_OVERLAP_SENTENCES,
) -> list[list[TranscriptSentence]]:
    if not sentences:
        return []
    if max_chunk <= overlap:
        raise PodcastEditorError("Chunk size must be greater than the overlap size.")
    chunks: list[list[TranscriptSentence]] = []
    start = 0
    step = max_chunk - overlap
    while start < len(sentences):
        end = min(len(sentences), start + max_chunk)
        chunks.append(sentences[start:end])
        if end >= len(sentences):
            break
        start += step
    return chunks


def format_sentence_chunk(sentences: list[TranscriptSentence]) -> str:
    return "\n".join(
        f"{s.sentence_id} | {s.start_timecode} - {s.end_timecode} | {s.text}"
        for s in sentences
    )


# ---------------------------------------------------------------------------
# Claude API calling (reused from auto_podcast_editor patterns)
# ---------------------------------------------------------------------------

import urllib.error
import urllib.request


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
        "messages": [{"role": "user", "content": content}],
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


def extract_json_payload(text: str) -> dict[str, Any]:
    stripped = text.strip()
    candidates = [stripped]
    fenced = JSON_BLOCK_PATTERN.search(stripped)
    if fenced is not None:
        candidates.append(fenced.group(1).strip())
    object_start = stripped.find("{")
    object_end = stripped.rfind("}")
    if object_start != -1 and object_end != -1 and object_start < object_end:
        candidates.append(stripped[object_start:object_end + 1].strip())
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


# ---------------------------------------------------------------------------
# Prompt loading and rendering
# ---------------------------------------------------------------------------


def load_prompt(prompt_path: Path) -> str:
    if not prompt_path.exists():
        raise PodcastEditorError(f"Prompt file not found: {prompt_path}")
    text = prompt_path.read_text(encoding="utf-8").strip()
    if not text:
        raise PodcastEditorError(f"Prompt file is empty: {prompt_path}")
    return text


def render_prompt_template(template_text: str, values: dict[str, Any]) -> str:
    rendered = template_text
    for key, value in values.items():
        rendered = rendered.replace(f"{{{{{key}}}}}", str(value))
    remaining = sorted(set(re.findall(r"\{\{[A-Za-z0-9_]+\}\}", rendered)))
    if remaining:
        raise PodcastEditorError(
            "Prompt template contains unreplaced placeholders: " + ", ".join(remaining)
        )
    return rendered


# ---------------------------------------------------------------------------
# Score helpers
# ---------------------------------------------------------------------------


def normalize_score(value: Any, minimum: int = 0, maximum: int = 5) -> int:
    try:
        score = int(round(float(value)))
    except (TypeError, ValueError):
        score = minimum
    return max(minimum, min(maximum, score))


def parse_rubric_scores(raw_scores: Any, keys: tuple[str, ...], maximum: int = 5) -> dict[str, int]:
    scores: dict[str, int] = {}
    if not isinstance(raw_scores, dict):
        raw_scores = {}
    for key in keys:
        scores[key] = normalize_score(raw_scores.get(key, 0), maximum=maximum)
    return scores


# ---------------------------------------------------------------------------
# Pass 1: Discovery
# ---------------------------------------------------------------------------


def nomination_limit_for_chunk_count(chunk_count: int) -> int:
    if chunk_count <= 0:
        return 20
    desired = max(40, 30)
    return max(4, math.ceil(desired / chunk_count))


def parse_discovery_candidates(
    payload: dict[str, Any],
    chunk_sentences: list[TranscriptSentence],
    sentence_by_id: dict[str, TranscriptSentence],
) -> list[ReelCandidate]:
    chunk_ids = {s.sentence_id for s in chunk_sentences}
    chunk_start = chunk_sentences[0].index
    chunk_end = chunk_sentences[-1].index
    raw_candidates = payload.get("candidates", [])
    if not isinstance(raw_candidates, list):
        raise PodcastEditorError("Discovery JSON did not contain a candidates list.")

    candidates: list[ReelCandidate] = []
    for item in raw_candidates:
        if not isinstance(item, dict):
            continue
        start_id = str(item.get("start_id", "")).strip()
        end_id = str(item.get("end_id", "")).strip()
        if start_id not in chunk_ids or end_id not in chunk_ids:
            continue
        start_s = sentence_by_id.get(start_id)
        end_s = sentence_by_id.get(end_id)
        if start_s is None or end_s is None:
            continue
        start_index = min(start_s.index, end_s.index)
        end_index = max(start_s.index, end_s.index)
        if start_index < chunk_start or end_index > chunk_end:
            continue
        if end_index - start_index + 1 > MAX_REEL_SENTENCES:
            continue
        rubric = parse_rubric_scores(item.get("scores"), DISCOVERY_RUBRIC_KEYS)
        total = normalize_score(item.get("total_score", sum(rubric.values())), minimum=0, maximum=30)
        title = normalize_text(str(item.get("title", "")).strip()) or "Untitled reel"
        rationale = normalize_text(str(item.get("rationale", "")).strip())
        candidates.append(
            ReelCandidate(
                start_index=start_index,
                end_index=end_index,
                title=title,
                total_score=total,
                rubric_scores=rubric,
                rationale=rationale,
            )
        )
    if not candidates:
        emit_status("Warning: Discovery pass returned zero valid candidates for this chunk")
    return candidates


def dedupe_candidates(candidates: list[ReelCandidate]) -> list[ReelCandidate]:
    by_range: dict[tuple[int, int], ReelCandidate] = {}
    for c in candidates:
        key = (c.start_index, c.end_index)
        existing = by_range.get(key)
        if existing is None or c.total_score > existing.total_score:
            by_range[key] = c
    return list(by_range.values())


def overlap_length(a: ReelCandidate, b: ReelCandidate) -> int:
    return max(0, min(a.end_index, b.end_index) - max(a.start_index, b.start_index) + 1)


def build_shortlist(candidates: list[ReelCandidate], limit: int) -> list[ReelCandidate]:
    shortlist: list[ReelCandidate] = []
    for c in sorted(candidates, key=lambda x: (-x.total_score, x.start_index)):
        if any(
            overlap_length(c, ex) >= min(
                c.end_index - c.start_index + 1,
                ex.end_index - ex.start_index + 1,
            )
            for ex in shortlist
        ):
            continue
        shortlist.append(c)
        if len(shortlist) >= limit:
            break
    return sorted(shortlist, key=lambda x: (x.start_index, x.end_index))


def assign_candidate_ids(candidates: list[ReelCandidate]) -> list[ReelCandidate]:
    return [replace(c, candidate_id=f"C{i + 1:03d}") for i, c in enumerate(candidates)]


def serialize_candidate(c: ReelCandidate, sentences: list[TranscriptSentence]) -> dict[str, Any]:
    start_s = sentences[c.start_index]
    end_s = sentences[c.end_index]
    return {
        "candidate_id": c.candidate_id,
        "start_sentence_id": start_s.sentence_id,
        "end_sentence_id": end_s.sentence_id,
        "start_timecode": start_s.start_timecode,
        "end_timecode": end_s.end_timecode,
        "title": c.title,
        "total_score": c.total_score,
        "scores": c.rubric_scores,
        "rationale": c.rationale,
        "text": " ".join(s.text for s in sentences[c.start_index:c.end_index + 1]),
    }


def run_discovery_pass(
    api_key: str,
    transcript_text: str,
    model_override: str | None,
    max_tokens: int,
    temperature: float,
    analysis_dir: Path,
) -> tuple[list[ReelCandidate], list[TranscriptSentence]]:
    """Pass 1: Chunked discovery of viral reel moments."""
    prompt_template = load_prompt(
        Path(__file__).resolve().with_name("claude_reel_discovery_prompt.md")
    )
    sentences = parse_timecoded_transcript(transcript_text)
    sentence_by_id = {s.sentence_id: s for s in sentences}
    chunks = build_sentence_chunks(sentences)
    nom_limit = nomination_limit_for_chunk_count(len(chunks))
    emit_status(f"Discovery: {len(sentences)} sentences across {len(chunks)} chunks")

    all_candidates: list[ReelCandidate] = []

    for chunk_idx, chunk in enumerate(chunks, start=1):
        emit_status(f"Discovery chunk {chunk_idx}/{len(chunks)} → Claude")
        prompt = render_prompt_template(
            prompt_template,
            {
                "chunk_index": chunk_idx,
                "chunk_total": len(chunks),
                "max_selection_sentences": MAX_REEL_SENTENCES,
                "candidate_limit": nom_limit,
                "transcript_chunk": format_sentence_chunk(chunk),
            },
        )
        prompt_path = analysis_dir / f"reel_discovery_chunk_{chunk_idx:02d}.prompt.md"
        prompt_path.write_text(prompt + "\n", encoding="utf-8")

        model, raw_text = call_claude_with_fallbacks(
            api_key, prompt, model_override, max_tokens, temperature,
        )
        payload = extract_json_payload(raw_text)
        chunk_candidates = parse_discovery_candidates(payload, chunk, sentence_by_id)
        all_candidates.extend(chunk_candidates)
        emit_status(f"  → {len(chunk_candidates)} candidates from chunk {chunk_idx}")

        response_path = analysis_dir / f"reel_discovery_chunk_{chunk_idx:02d}.json"
        response_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    deduped = dedupe_candidates(all_candidates)
    shortlist = assign_candidate_ids(build_shortlist(deduped, SHORTLIST_LIMIT))
    emit_status(f"Discovery complete: {len(all_candidates)} raw → {len(deduped)} deduped → {len(shortlist)} shortlisted")

    shortlist_path = analysis_dir / "reel_shortlist.json"
    shortlist_path.write_text(
        json.dumps([serialize_candidate(c, sentences) for c in shortlist], indent=2) + "\n",
        encoding="utf-8",
    )

    return shortlist, sentences


# ---------------------------------------------------------------------------
# Pass 2: Hook / Pay-off Synthesis
# ---------------------------------------------------------------------------


def build_candidates_with_context(
    shortlist: list[ReelCandidate],
    sentences: list[TranscriptSentence],
    context_window: int = SYNTHESIS_CONTEXT_WINDOW,
) -> str:
    """Format each shortlisted candidate with surrounding context for synthesis."""
    blocks: list[str] = []
    for c in shortlist:
        ctx_start = max(0, c.start_index - context_window)
        ctx_end = min(len(sentences) - 1, c.end_index + context_window)

        lines: list[str] = [
            f"--- {c.candidate_id}: \"{c.title}\" ---",
            f"Original range: {sentences[c.start_index].sentence_id} – {sentences[c.end_index].sentence_id}",
            f"Timecode: {sentences[c.start_index].start_timecode} – {sentences[c.end_index].end_timecode}",
            "",
            "Surrounding context (use these IDs):",
        ]
        for s in sentences[ctx_start:ctx_end + 1]:
            marker = " <<<" if c.start_index <= s.index <= c.end_index else ""
            lines.append(
                f"{s.sentence_id} | {s.start_timecode} - {s.end_timecode} | {s.text}{marker}"
            )
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def parse_synthesis_results(
    payload: dict[str, Any],
    shortlist_ids: set[str],
) -> list[SynthesizedReel]:
    raw = payload.get("synthesized_reels", [])
    if not isinstance(raw, list):
        raise PodcastEditorError("Synthesis JSON did not contain a synthesized_reels list.")

    results: list[SynthesizedReel] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        cid = str(item.get("candidate_id", "")).strip()
        if cid not in shortlist_ids or cid in seen:
            continue
        seen.add(cid)

        hook = item.get("hook", {})
        payoff = item.get("payoff", {})
        if not isinstance(hook, dict) or not isinstance(payoff, dict):
            continue

        results.append(
            SynthesizedReel(
                candidate_id=cid,
                title=normalize_text(str(item.get("title", "")).strip()) or "Untitled",
                text_overlay=normalize_text(str(item.get("text_overlay", "")).strip()),
                hook_start_id=str(hook.get("start_id", "")).strip(),
                hook_end_id=str(hook.get("end_id", "")).strip(),
                hook_text=normalize_text(str(hook.get("text", "")).strip()),
                payoff_start_id=str(payoff.get("start_id", "")).strip(),
                payoff_end_id=str(payoff.get("end_id", "")).strip(),
                payoff_text=normalize_text(str(payoff.get("text", "")).strip()),
                hook_payoff_gap=normalize_text(str(item.get("hook_payoff_gap", "same_range")).strip()),
                rationale=normalize_text(str(item.get("rationale", "")).strip()),
            )
        )
    return results


def run_synthesis_pass(
    api_key: str,
    shortlist: list[ReelCandidate],
    sentences: list[TranscriptSentence],
    model_override: str | None,
    max_tokens: int,
    temperature: float,
    analysis_dir: Path,
) -> list[SynthesizedReel]:
    """Pass 2: Synthesize hook/pay-off pairings for each shortlisted candidate."""
    prompt_template = load_prompt(
        Path(__file__).resolve().with_name("claude_reel_synthesis_prompt.md")
    )
    context_text = build_candidates_with_context(shortlist, sentences)
    prompt = render_prompt_template(
        prompt_template,
        {"candidates_with_context": context_text},
    )
    prompt_path = analysis_dir / "reel_synthesis.prompt.md"
    prompt_path.write_text(prompt + "\n", encoding="utf-8")
    emit_status(f"Synthesis: submitting {len(shortlist)} candidates to Claude")

    model, raw_text = call_claude_with_fallbacks(
        api_key, prompt, model_override, max_tokens, temperature,
    )
    payload = extract_json_payload(raw_text)
    response_path = analysis_dir / "reel_synthesis.json"
    response_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    shortlist_ids = {c.candidate_id for c in shortlist}
    results = parse_synthesis_results(payload, shortlist_ids)
    emit_status(f"Synthesis complete: {len(results)} reels synthesized")
    return results


# ---------------------------------------------------------------------------
# Pass 3: Virality Ranking
# ---------------------------------------------------------------------------


def format_synthesized_for_ranking(
    synthesized: list[SynthesizedReel],
    sentences: list[TranscriptSentence],
    sentence_by_id: dict[str, TranscriptSentence],
) -> str:
    blocks: list[str] = []
    for r in synthesized:
        hook_start_s = sentence_by_id.get(r.hook_start_id)
        hook_end_s = sentence_by_id.get(r.hook_end_id)
        payoff_start_s = sentence_by_id.get(r.payoff_start_id)
        payoff_end_s = sentence_by_id.get(r.payoff_end_id)

        hook_tc = ""
        if hook_start_s and hook_end_s:
            hook_tc = f"{hook_start_s.start_timecode} – {hook_end_s.end_timecode}"
        payoff_tc = ""
        if payoff_start_s and payoff_end_s:
            payoff_tc = f"{payoff_start_s.start_timecode} – {payoff_end_s.end_timecode}"

        lines = [
            f"--- {r.candidate_id}: \"{r.title}\" ---",
            f"Text overlay: \"{r.text_overlay}\"",
            f"Hook ({hook_tc}): \"{r.hook_text}\"",
            f"Pay-off ({payoff_tc}): \"{r.payoff_text}\"",
            f"Hook/pay-off gap: {r.hook_payoff_gap}",
        ]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def parse_ranking_results(
    payload: dict[str, Any],
    valid_ids: set[str],
) -> list[RankedReel]:
    raw = payload.get("ranked_reels", [])
    if not isinstance(raw, list):
        raise PodcastEditorError("Ranking JSON did not contain a ranked_reels list.")

    results: list[RankedReel] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        cid = str(item.get("candidate_id", "")).strip()
        if cid not in valid_ids or cid in seen:
            continue
        seen.add(cid)
        ranking_scores = parse_rubric_scores(item.get("scores"), RANKING_RUBRIC_KEYS, maximum=10)
        virality = normalize_score(item.get("virality_score", 0), minimum=0, maximum=100)
        rationale = normalize_text(str(item.get("rationale", "")).strip())
        results.append(
            RankedReel(
                candidate_id=cid,
                virality_score=virality,
                ranking_scores=ranking_scores,
                rationale=rationale,
            )
        )
    return results


def run_ranking_pass(
    api_key: str,
    synthesized: list[SynthesizedReel],
    sentences: list[TranscriptSentence],
    model_override: str | None,
    max_tokens: int,
    temperature: float,
    analysis_dir: Path,
) -> list[RankedReel]:
    """Pass 3: Rank all synthesized reels by virality potential."""
    prompt_template = load_prompt(
        Path(__file__).resolve().with_name("claude_reel_ranking_prompt.md")
    )
    sentence_by_id = {s.sentence_id: s for s in sentences}
    formatted = format_synthesized_for_ranking(synthesized, sentences, sentence_by_id)
    prompt = render_prompt_template(
        prompt_template,
        {
            "candidate_count": len(synthesized),
            "synthesized_candidates": formatted,
        },
    )
    prompt_path = analysis_dir / "reel_ranking.prompt.md"
    prompt_path.write_text(prompt + "\n", encoding="utf-8")
    emit_status(f"Ranking: submitting {len(synthesized)} synthesized reels to Claude")

    model, raw_text = call_claude_with_fallbacks(
        api_key, prompt, model_override, max_tokens, temperature,
    )
    payload = extract_json_payload(raw_text)
    response_path = analysis_dir / "reel_ranking.json"
    response_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    valid_ids = {r.candidate_id for r in synthesized}
    results = parse_ranking_results(payload, valid_ids)
    # Sort by virality_score descending
    results.sort(key=lambda x: -x.virality_score)
    emit_status(f"Ranking complete: {len(results)} reels ranked")
    return results


# ---------------------------------------------------------------------------
# Final output: join all passes into a clean report
# ---------------------------------------------------------------------------


def build_final_output(
    ranked: list[RankedReel],
    synthesized: list[SynthesizedReel],
    shortlist: list[ReelCandidate],
    sentences: list[TranscriptSentence],
) -> list[dict[str, Any]]:
    """Join ranking, synthesis, and discovery data into the final output."""
    synth_by_id = {r.candidate_id: r for r in synthesized}
    candidate_by_id = {c.candidate_id: c for c in shortlist}
    sentence_by_id = {s.sentence_id: s for s in sentences}

    output: list[dict[str, Any]] = []
    for rank_idx, r in enumerate(ranked, start=1):
        synth = synth_by_id.get(r.candidate_id)
        candidate = candidate_by_id.get(r.candidate_id)
        if synth is None or candidate is None:
            continue

        # Resolve timecodes for hook and payoff
        hook_start_s = sentence_by_id.get(synth.hook_start_id)
        hook_end_s = sentence_by_id.get(synth.hook_end_id)
        payoff_start_s = sentence_by_id.get(synth.payoff_start_id)
        payoff_end_s = sentence_by_id.get(synth.payoff_end_id)

        entry = {
            "rank": rank_idx,
            "candidate_id": r.candidate_id,
            "virality_score": r.virality_score,
            "title": synth.title,
            "text_overlay": synth.text_overlay,
            "hook": {
                "start_id": synth.hook_start_id,
                "end_id": synth.hook_end_id,
                "start_timecode": hook_start_s.start_timecode if hook_start_s else "",
                "end_timecode": hook_end_s.end_timecode if hook_end_s else "",
                "text": synth.hook_text,
            },
            "payoff": {
                "start_id": synth.payoff_start_id,
                "end_id": synth.payoff_end_id,
                "start_timecode": payoff_start_s.start_timecode if payoff_start_s else "",
                "end_timecode": payoff_end_s.end_timecode if payoff_end_s else "",
                "text": synth.payoff_text,
            },
            "hook_payoff_gap": synth.hook_payoff_gap,
            "discovery_scores": candidate.rubric_scores,
            "discovery_total": candidate.total_score,
            "ranking_scores": r.ranking_scores,
            "ranking_rationale": r.rationale,
            "synthesis_rationale": synth.rationale,
            "full_candidate_text": " ".join(
                s.text for s in sentences[candidate.start_index:candidate.end_index + 1]
            ),
            "candidate_timecode": {
                "start": sentences[candidate.start_index].start_timecode,
                "end": sentences[candidate.end_index].end_timecode,
            },
        }
        output.append(entry)
    return output


def format_human_readable_report(final_output: list[dict[str, Any]]) -> str:
    """Generate a human-readable summary report."""
    lines: list[str] = [
        "=" * 72,
        "VIRAL REEL DISCOVERY REPORT",
        "=" * 72,
        "",
        f"Total reels identified: {len(final_output)}",
        "",
    ]

    for entry in final_output:
        lines.extend([
            "-" * 72,
            f"#{entry['rank']}  |  Virality: {entry['virality_score']}/100  |  {entry['candidate_id']}",
            f"Title: {entry['title']}",
            f"Text Overlay: \"{entry['text_overlay']}\"",
            "",
            f"  HOOK  [{entry['hook']['start_timecode']} – {entry['hook']['end_timecode']}]",
            f"  \"{entry['hook']['text']}\"",
            "",
            f"  PAY-OFF  [{entry['payoff']['start_timecode']} – {entry['payoff']['end_timecode']}]",
            f"  \"{entry['payoff']['text']}\"",
            "",
            f"  Hook/Pay-off gap: {entry['hook_payoff_gap']}",
            f"  Full candidate: {entry['candidate_timecode']['start']} – {entry['candidate_timecode']['end']}",
            "",
            f"  Ranking rationale: {entry['ranking_rationale']}",
            "",
        ])

    lines.append("=" * 72)
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])

    try:
        speaker_config = speaker_diarization_config_from_args(args)
    except ValueError as exc:
        raise PodcastEditorError(str(exc)) from exc

    # Validate inputs
    if args.input is None and args.transcript is None:
        raise PodcastEditorError("Provide --input (video/audio file) or --transcript (pre-existing Whisper JSON).")
    if args.transcript is not None and args.manifest is None:
        raise PodcastEditorError("--manifest is required when using --transcript.")

    # Frame ticks for timecode display
    frame_ticks = int(round(TICKS_PER_SECOND / args.frame_rate))

    # Output directory
    if args.output_dir:
        output_dir = args.output_dir
    elif args.input:
        output_dir = args.input.parent / f"{args.input.stem}.reels"
    else:
        output_dir = args.transcript.parent / f"{args.transcript.stem}.reels"
    output_dir.mkdir(parents=True, exist_ok=True)
    emit_status(f"Output directory: {output_dir}")

    # --- Stage 1: Transcription ---
    if args.transcript is not None:
        transcript_summary = load_existing_transcript(
            args.transcript, args.manifest, frame_ticks, output_dir,
        )
    else:
        audio_path = output_dir / f"{args.input.stem}.wav"
        extract_audio_from_video(args.input, audio_path)
        transcript_summary = transcribe_audio(
            audio_path, output_dir, args.whisper_model, args.whisper_language,
            speaker_config, frame_ticks,
        )

    timecoded_transcript = str(transcript_summary["timecoded_transcript"])

    # --- Stage 2: Discovery pass ---
    api_key = require_anthropic_api_key()

    if args.discovery_response_file is not None:
        emit_status(f"Loading pre-existing discovery results from {args.discovery_response_file}")
        # Load the shortlist JSON directly
        raw_shortlist = json.loads(args.discovery_response_file.read_text(encoding="utf-8"))
        sentences = parse_timecoded_transcript(timecoded_transcript)
        sentence_by_id = {s.sentence_id: s for s in sentences}
        shortlist: list[ReelCandidate] = []
        for item in raw_shortlist:
            start_s = sentence_by_id.get(str(item.get("start_sentence_id", "")))
            end_s = sentence_by_id.get(str(item.get("end_sentence_id", "")))
            if start_s is None or end_s is None:
                continue
            rubric = parse_rubric_scores(item.get("scores"), DISCOVERY_RUBRIC_KEYS)
            shortlist.append(
                ReelCandidate(
                    start_index=start_s.index,
                    end_index=end_s.index,
                    title=str(item.get("title", "")),
                    total_score=int(item.get("total_score", 0)),
                    rubric_scores=rubric,
                    rationale=str(item.get("rationale", "")),
                    candidate_id=str(item.get("candidate_id", "")),
                )
            )
    else:
        shortlist, sentences = run_discovery_pass(
            api_key, timecoded_transcript,
            args.claude_model, args.claude_max_tokens, args.claude_temperature,
            output_dir,
        )

    if not shortlist:
        emit_status("No reel candidates were discovered. Exiting.")
        return 1

    emit_status(f"Shortlist contains {len(shortlist)} candidates")

    if args.skip_synthesis:
        emit_status("Skipping synthesis and ranking (--skip-synthesis)")
        # Write just the shortlist as the final output
        final_path = output_dir / "reel_candidates.json"
        final_path.write_text(
            json.dumps([serialize_candidate(c, sentences) for c in shortlist], indent=2) + "\n",
            encoding="utf-8",
        )
        emit_status(f"Wrote {len(shortlist)} candidates to {final_path}")
        return 0

    # --- Stage 3: Synthesis pass ---
    synthesized = run_synthesis_pass(
        api_key, shortlist, sentences,
        args.claude_model, args.claude_max_tokens, args.claude_temperature,
        output_dir,
    )
    if not synthesized:
        emit_status("Synthesis pass returned no results. Check the synthesis JSON for errors.")
        return 1

    # --- Stage 4: Ranking pass ---
    ranked = run_ranking_pass(
        api_key, synthesized, sentences,
        args.claude_model, args.claude_max_tokens, args.claude_temperature,
        output_dir,
    )
    if not ranked:
        emit_status("Ranking pass returned no results. Check the ranking JSON for errors.")
        return 1

    # --- Stage 5: Final output ---
    final_output = build_final_output(ranked, synthesized, shortlist, sentences)

    # Write JSON
    final_json_path = output_dir / "reel_candidates.json"
    final_json_path.write_text(
        json.dumps(final_output, indent=2) + "\n",
        encoding="utf-8",
    )
    emit_status(f"Wrote {len(final_output)} ranked reels to {final_json_path}")

    # Write human-readable report
    report = format_human_readable_report(final_output)
    report_path = output_dir / "reel_report.txt"
    report_path.write_text(report, encoding="utf-8")
    emit_status(f"Wrote human-readable report to {report_path}")

    # Print summary to stdout
    print(f"output_dir={output_dir}")
    print(f"total_reels={len(final_output)}")
    print(f"reel_candidates_json={final_json_path}")
    print(f"reel_report={report_path}")

    if final_output:
        top = final_output[0]
        print(f"top_reel_title={top['title']}")
        print(f"top_reel_virality={top['virality_score']}")
        print(f"top_reel_hook={top['hook']['start_timecode']} – {top['hook']['end_timecode']}")

    emit_status("Finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
