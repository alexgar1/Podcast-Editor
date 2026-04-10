import json
import os
import tempfile
import unittest
from pathlib import Path

from auto_podcast_editor import (
    ClipCandidate,
    RemovalCandidate,
    TranscriptBlock,
    TranscriptSentence,
    assign_candidate_ids,
    build_concision_prompt,
    build_sentence_chunks,
    build_final_pass_prompt,
    build_first_pass_prompt,
    build_timecoded_transcript,
    default_concision_prompt_file,
    default_final_pass_prompt_file,
    default_first_pass_prompt_file,
    default_analysis_dir,
    default_prompt_file,
    extract_json_payload,
    format_removal_candidates,
    format_ranked_selections,
    load_dotenv_if_present,
    load_prompt,
    merge_adjacent_candidates,
    parse_concision_candidates,
    parse_editorial_blocks_transcript,
    parse_args,
    parse_timecoded_transcript,
    render_prompt_template,
    rendered_audio_ticks_to_timeline_ticks,
    ticks_to_timecode,
)
from podcast_editor import PodcastEditorError, TICKS_PER_SECOND, TickRange


class AutoPodcastEditorTests(unittest.TestCase):
    def test_ticks_to_timecode_at_sixty_fps(self) -> None:
        frame_ticks = TICKS_PER_SECOND // 60
        self.assertEqual(ticks_to_timecode(0, frame_ticks), "00:00:00:00")
        self.assertEqual(ticks_to_timecode(frame_ticks * 61, frame_ticks), "00:00:01:01")

    def test_rendered_audio_ticks_map_across_multiple_segments(self) -> None:
        manifest = [
            {
                "timeline_start_ticks": 0,
                "timeline_end_ticks": 10 * TICKS_PER_SECOND,
                "source_start_ticks": 100 * TICKS_PER_SECOND,
                "source_end_ticks": 110 * TICKS_PER_SECOND,
            },
            {
                "timeline_start_ticks": 20 * TICKS_PER_SECOND,
                "timeline_end_ticks": 30 * TICKS_PER_SECOND,
                "source_start_ticks": 200 * TICKS_PER_SECOND,
                "source_end_ticks": 210 * TICKS_PER_SECOND,
            },
        ]
        self.assertEqual(rendered_audio_ticks_to_timeline_ticks(5 * TICKS_PER_SECOND, manifest), 5 * TICKS_PER_SECOND)
        self.assertEqual(
            rendered_audio_ticks_to_timeline_ticks(15 * TICKS_PER_SECOND, manifest),
            25 * TICKS_PER_SECOND,
        )

    def test_load_dotenv_if_present_does_not_override_existing_values(self) -> None:
        previous = os.environ.get("TEST_AUTO_EDITOR_ENV")
        os.environ["TEST_AUTO_EDITOR_ENV"] = "existing"
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                dotenv_path = Path(tmp_dir) / ".env"
                dotenv_path.write_text("TEST_AUTO_EDITOR_ENV=from_file\nNEW_AUTO_EDITOR_ENV='new'\n", encoding="utf-8")
                load_dotenv_if_present(dotenv_path)
                self.assertEqual(os.environ["TEST_AUTO_EDITOR_ENV"], "existing")
                self.assertEqual(os.environ["NEW_AUTO_EDITOR_ENV"], "new")
                del os.environ["NEW_AUTO_EDITOR_ENV"]
        finally:
            if previous is None:
                del os.environ["TEST_AUTO_EDITOR_ENV"]
            else:
                os.environ["TEST_AUTO_EDITOR_ENV"] = previous

    def test_default_prompt_file_exists_and_loads(self) -> None:
        prompt_path = default_prompt_file()
        self.assertTrue(prompt_path.exists())
        prompt = prompt_path.read_text(encoding="utf-8").strip()
        self.assertEqual(load_prompt(None), prompt)

    def test_stage_prompt_template_files_exist_and_load(self) -> None:
        concision_path = default_concision_prompt_file()
        first_pass_path = default_first_pass_prompt_file()
        final_pass_path = default_final_pass_prompt_file()

        self.assertTrue(concision_path.exists())
        self.assertTrue(first_pass_path.exists())
        self.assertTrue(final_pass_path.exists())
        self.assertIn("broad concision pass", concision_path.read_text(encoding="utf-8"))
        self.assertIn("{{transcript_blocks}}", concision_path.read_text(encoding="utf-8"))
        self.assertIn("{{editorial_prompt}}", first_pass_path.read_text(encoding="utf-8"))
        self.assertIn("{{candidate_shortlist}}", final_pass_path.read_text(encoding="utf-8"))

    def test_default_analysis_dir_uses_output_project_stem(self) -> None:
        output_project = Path("/tmp/example.prproj")
        self.assertEqual(default_analysis_dir(output_project), Path("/tmp/example.analysis"))

    def test_parse_args_requires_explicit_selects_flag(self) -> None:
        args = parse_args(["--project", "/tmp/example.prproj"])

        self.assertFalse(args.create_selects_sequence)
        self.assertFalse(args.create_concision_sequence)

    def test_parse_args_accepts_explicit_selects_flag(self) -> None:
        args = parse_args(["--project", "/tmp/example.prproj", "--create-selects-sequence"])

        self.assertTrue(args.create_selects_sequence)
        self.assertFalse(args.create_concision_sequence)

    def test_build_timecoded_transcript_uses_word_timestamps_for_sentence_starts(self) -> None:
        transcript = {
            "segments": [
                {
                    "start": 0.0,
                    "end": 4.0,
                    "text": " Hello world. Next bit.",
                    "words": [
                        {"word": " Hello", "start": 1.0, "end": 1.2},
                        {"word": " world.", "start": 1.2, "end": 1.6},
                        {"word": " Next", "start": 3.0, "end": 3.2},
                        {"word": " bit.", "start": 3.2, "end": 3.5},
                    ],
                }
            ]
        }
        manifest = [
            {
                "timeline_start_ticks": 0,
                "timeline_end_ticks": 10 * TICKS_PER_SECOND,
                "source_start_ticks": 0,
                "source_end_ticks": 10 * TICKS_PER_SECOND,
            }
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            transcript_path = Path(tmp_dir) / "transcript.json"
            manifest_path = Path(tmp_dir) / "manifest.json"
            transcript_path.write_text(json.dumps(transcript), encoding="utf-8")
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            text, count = build_timecoded_transcript(
                transcript_path,
                manifest_path,
                TICKS_PER_SECOND // 60,
            )

        self.assertEqual(count, 2)
        self.assertEqual(
            text,
            "00:00:01:00 - 00:00:01:36 | Hello world.\n"
            "00:00:03:00 - 00:00:03:30 | Next bit.\n",
        )

    def test_build_timecoded_transcript_falls_back_to_segments_without_words(self) -> None:
        transcript = {
            "segments": [
                {
                    "start": 0.5,
                    "end": 2.0,
                    "text": " Hello there",
                }
            ]
        }
        manifest = [
            {
                "timeline_start_ticks": 0,
                "timeline_end_ticks": 10 * TICKS_PER_SECOND,
                "source_start_ticks": 0,
                "source_end_ticks": 10 * TICKS_PER_SECOND,
            }
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            transcript_path = Path(tmp_dir) / "transcript.json"
            manifest_path = Path(tmp_dir) / "manifest.json"
            transcript_path.write_text(json.dumps(transcript), encoding="utf-8")
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            text, count = build_timecoded_transcript(
                transcript_path,
                manifest_path,
                TICKS_PER_SECOND // 60,
            )

        self.assertEqual(count, 1)
        self.assertEqual(text, "00:00:00:30 - 00:00:02:00 | Hello there\n")

    def test_parse_timecoded_transcript_assigns_sentence_ids(self) -> None:
        transcript_text = (
            "00:00:00:00 - 00:00:01:00 | First sentence.\n"
            "00:00:01:01 - 00:00:02:00 | Second sentence.\n"
        )

        sentences = parse_timecoded_transcript(transcript_text)

        self.assertEqual(
            sentences,
            [
                TranscriptSentence("S0001", 0, "00:00:00:00", "00:00:01:00", "First sentence."),
                TranscriptSentence("S0002", 1, "00:00:01:01", "00:00:02:00", "Second sentence."),
            ],
        )

    def test_build_sentence_chunks_overlaps_by_max_selection_window(self) -> None:
        sentences = [
            TranscriptSentence(f"S{index + 1:04d}", index, "00:00:00:00", "00:00:00:01", f"Sentence {index + 1}.")
            for index in range(12)
        ]

        chunks = build_sentence_chunks(sentences, max_chunk_sentences=6, overlap_sentences=5)

        self.assertEqual(len(chunks), 7)
        self.assertEqual(chunks[0][0].sentence_id, "S0001")
        self.assertEqual(chunks[0][-1].sentence_id, "S0006")
        self.assertEqual(chunks[1][0].sentence_id, "S0002")
        self.assertEqual(chunks[1][-1].sentence_id, "S0007")

    def test_parse_editorial_blocks_transcript_assigns_block_ids(self) -> None:
        transcript_text = (
            "B0001 | speaker_01 | 00:00:00:00 - 00:00:05:00 | Opening setup.\n"
            "B0002 | speaker_02 | 00:00:05:01 - 00:00:10:00 | Repeated explanation.\n"
        )

        blocks = parse_editorial_blocks_transcript(transcript_text)

        self.assertEqual(
            blocks,
            [
                TranscriptBlock("B0001", 0, "speaker_01", "00:00:00:00", "00:00:05:00", "Opening setup."),
                TranscriptBlock("B0002", 1, "speaker_02", "00:00:05:01", "00:00:10:00", "Repeated explanation."),
            ],
        )

    def test_merge_adjacent_candidates_keeps_longer_context_window(self) -> None:
        merged = merge_adjacent_candidates(
            [
                ClipCandidate(0, 1, "Setup", 18, {}, "Setup matters."),
                ClipCandidate(2, 3, "Payoff", 21, {}, "Payoff lands here."),
            ]
        )

        self.assertEqual(
            merged,
            [
                ClipCandidate(
                    0,
                    3,
                    "Payoff",
                    21,
                    {
                        "curiosity_gap": 0,
                        "emotional_peak": 0,
                        "bold_claim": 0,
                        "vivid_storytelling": 0,
                        "high_stakes_setup": 0,
                        "quotable_shareable": 0,
                    },
                    "Setup matters. Payoff lands here.",
                ),
            ],
        )

    def test_assign_candidate_ids_and_format_ranked_selections(self) -> None:
        sentences = [
            TranscriptSentence("S0001", 0, "00:00:00:00", "00:00:01:00", "First sentence."),
            TranscriptSentence("S0002", 1, "00:00:01:01", "00:00:02:00", "Second sentence."),
        ]
        candidates = assign_candidate_ids([ClipCandidate(0, 1, "Strong opener", 24, {}, "Great hook.")])

        text = format_ranked_selections([(candidates[0], "Strong opener", "Great hook.")], sentences)

        self.assertEqual(
            text,
            '#1 — Strong opener\n'
            'Timecode: 00:00:00:00 – 00:00:02:00\n'
            'Text: "First sentence. Second sentence."\n',
        )

    def test_parse_concision_candidates_accepts_empty_removals(self) -> None:
        blocks = [TranscriptBlock("B0001", 0, "speaker_01", "00:00:00:00", "00:00:05:00", "Focused section.")]

        candidates = parse_concision_candidates({"removals": []}, blocks, {"B0001": blocks[0]})

        self.assertEqual(candidates, [])

    def test_parse_concision_candidates_maps_block_ranges(self) -> None:
        blocks = [
            TranscriptBlock("B0001", 0, "speaker_01", "00:00:00:00", "00:00:05:00", "Opening setup."),
            TranscriptBlock("B0002", 1, "speaker_02", "00:00:05:01", "00:00:10:00", "Repeated explanation."),
        ]

        candidates = parse_concision_candidates(
            {
                "removals": [
                    {
                        "start_id": "B0001",
                        "end_id": "B0002",
                        "kind": "redundant",
                        "reason": "Repeats the same point.",
                    }
                ]
            },
            blocks,
            {block.block_id: block for block in blocks},
        )

        self.assertEqual(candidates, [RemovalCandidate(0, 1, "redundant", "Repeats the same point.")])

    def test_format_removal_candidates_renders_timecodes(self) -> None:
        blocks = [
            TranscriptBlock("B0001", 0, "speaker_01", "00:00:00:00", "00:00:05:00", "Opening setup."),
            TranscriptBlock("B0002", 1, "speaker_02", "00:00:05:01", "00:00:10:00", "Repeated explanation."),
        ]

        text = format_removal_candidates([RemovalCandidate(0, 1, "redundant", "Repeats the same point.")], blocks)

        self.assertIn("#1 — Redundant", text)
        self.assertIn("Timecode: 00:00:00:00 – 00:00:10:00", text)
        self.assertIn("Blocks: B0001 - B0002", text)

    def test_extract_json_payload_accepts_fenced_json(self) -> None:
        payload = extract_json_payload('```json\n{"candidates":[{"start_id":"S0001","end_id":"S0001"}]}\n```')

        self.assertEqual(payload["candidates"][0]["start_id"], "S0001")

    def test_render_prompt_template_replaces_all_placeholders(self) -> None:
        rendered = render_prompt_template("Hello {{name}} from {{place}}.", {"name": "Alex", "place": "Codex"})

        self.assertEqual(rendered, "Hello Alex from Codex.")

    def test_build_first_pass_prompt_from_template(self) -> None:
        template = (
            "{{editorial_prompt}}\n"
            "Chunk {{chunk_index}}/{{chunk_total}}\n"
            "Max {{max_selection_sentences}}\n"
            "Limit {{candidate_limit}}\n"
            "{{transcript_chunk}}"
        )
        sentences = [TranscriptSentence("S0001", 0, "00:00:00:00", "00:00:01:00", "First sentence.")]

        rendered = build_first_pass_prompt(template, "Editorial rules.", sentences, 1, 3, 7)

        self.assertIn("Editorial rules.", rendered)
        self.assertIn("Chunk 1/3", rendered)
        self.assertIn("Limit 7", rendered)
        self.assertIn("S0001 | 00:00:00:00 - 00:00:01:00 | First sentence.", rendered)

    def test_build_concision_prompt_from_template(self) -> None:
        template = "{{transcript_blocks}}"
        blocks = [TranscriptBlock("B0001", 0, "speaker_01", "00:00:00:00", "00:00:05:00", "Opening setup.")]

        rendered = build_concision_prompt(template, blocks)

        self.assertIn("B0001 | speaker_01 | 00:00:00:00 - 00:00:05:00 | Opening setup.", rendered)

    def test_build_final_pass_prompt_from_template(self) -> None:
        template = "{{editorial_prompt}}\nSelect {{selection_count}}\n{{candidate_shortlist}}"
        sentences = [
            TranscriptSentence("S0001", 0, "00:00:00:00", "00:00:01:00", "First sentence."),
            TranscriptSentence("S0002", 1, "00:00:01:01", "00:00:02:00", "Second sentence."),
        ]
        candidates = assign_candidate_ids([ClipCandidate(0, 1, "Strong opener", 24, {}, "Great hook.")])

        rendered = build_final_pass_prompt(template, "Editorial rules.", candidates, sentences, 1)

        self.assertIn("Editorial rules.", rendered)
        self.assertIn("Select 1", rendered)
        self.assertIn("C001", rendered)
        self.assertIn('Text: "First sentence. Second sentence."', rendered)

if __name__ == "__main__":
    unittest.main()
