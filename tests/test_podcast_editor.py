import unittest
import xml.etree.ElementTree as ET

from pathlib import Path

from podcast_editor import (
    PodcastEditorError,
    Segment,
    TickRange,
    TICKS_PER_SECOND,
    build_transcription_command,
    build_segments,
    build_template_context,
    create_concision_sequence,
    create_selects_sequence,
    default_output_path,
    default_transcript_dir,
    extract_timecode_ranges,
    extract_timecode_ranges_or_empty,
    merge_touching_ranges,
    normalize_timecode,
    normalize_ranges,
    parse_key_value_lines,
)
from speaker_diarization import SpeakerDiarizationConfig


class PodcastEditorTests(unittest.TestCase):
    def fixture_root(self, name: str = "reference_2026.xml") -> ET.Element:
        fixture_path = Path(__file__).resolve().parents[1] / "fixtures" / "premiere_xml" / name
        return ET.parse(fixture_path).getroot()

    def test_extract_timecode_ranges(self) -> None:
        sample_text = """
        junk text
        Timecode: 00:05:38:38 – 00:05:52:57
        more junk
        Timecode: 00:10:42:57 – 00:11:04:10
        """
        ranges = extract_timecode_ranges(sample_text)
        self.assertEqual(
            [(time_range.start, time_range.end) for time_range in ranges],
            [("00:05:38:38", "00:05:52:57"), ("00:10:42:57", "00:11:04:10")],
        )

    def test_extract_timecode_ranges_normalizes_three_part_timecodes(self) -> None:
        sample_text = "Timecode: 05:38:38 – 05:52:57"
        ranges = extract_timecode_ranges(sample_text)
        self.assertEqual(
            [(time_range.start, time_range.end) for time_range in ranges],
            [("00:05:38:38", "00:05:52:57")],
        )

    def test_extract_timecode_ranges_or_empty_allows_no_matches(self) -> None:
        self.assertEqual(extract_timecode_ranges_or_empty("no timecodes here"), [])

    def test_normalize_timecode_rejects_unsupported_format(self) -> None:
        with self.assertRaises(PodcastEditorError):
            normalize_timecode("5:38")

    def test_normalize_ranges_clips_and_preserves_overlaps(self) -> None:
        ranges = normalize_ranges(
            [
                TickRange(100, 200),
                TickRange(150, 250),
                TickRange(400, 500),
                TickRange(-50, 20),
                TickRange(100, 200),
            ],
            duration_ticks=450,
        )
        self.assertEqual(ranges, [TickRange(0, 20), TickRange(100, 200), TickRange(150, 250), TickRange(400, 450)])

    def test_merge_touching_ranges_unions_overlaps(self) -> None:
        merged = merge_touching_ranges([TickRange(100, 200), TickRange(150, 250), TickRange(300, 350)])

        self.assertEqual(merged, [TickRange(100, 250), TickRange(300, 350)])

    def test_build_segments(self) -> None:
        segments = build_segments([TickRange(10, 20), TickRange(30, 40)], duration_ticks=50)
        self.assertEqual(
            segments,
            [
                Segment(0, 10, False),
                Segment(10, 20, True),
                Segment(20, 30, False),
                Segment(30, 40, True),
                Segment(40, 50, False),
            ],
        )

    def test_default_output_path_reuses_source_project(self) -> None:
        project_path = Path("/tmp/example.prproj")
        self.assertEqual(
            default_output_path(project_path),
            project_path,
        )

    def test_default_transcript_dir_uses_output_project_stem(self) -> None:
        output_project = Path("/tmp/example.prproj")
        self.assertEqual(
            default_transcript_dir(output_project),
            Path("/tmp/example.transcript"),
        )

    def test_parse_key_value_lines(self) -> None:
        output = "sequence=Episode - Selects\nsegments=20\nnoise line\ntranscript_json=/tmp/out.json\n"
        self.assertEqual(
            parse_key_value_lines(output),
            {
                "sequence": "Episode - Selects",
                "segments": "20",
                "transcript_json": "/tmp/out.json",
            },
        )

    def test_build_transcription_command_includes_speaker_diarization_flags(self) -> None:
        command = build_transcription_command(
            project_path=Path("/tmp/example.prproj"),
            sequence_name="Episode - Selects",
            output_dir=Path("/tmp/example.transcript"),
            whisper_model="base.en",
            whisper_language="en",
            whisper_conda_env=None,
            whisper_python=Path("/usr/bin/python3"),
            speaker_config=SpeakerDiarizationConfig(
                enabled=True,
                model="/tmp/pyannote-community-1",
                auth_token="token-123",
                max_speakers=3,
            ),
        )

        self.assertIn("--detect-speakers", command)
        self.assertIn("--speaker-diarization-model", command)
        self.assertIn("/tmp/pyannote-community-1", command)
        self.assertIn("--speaker-auth-token", command)
        self.assertIn("token-123", command)
        self.assertIn("--max-speakers", command)
        self.assertIn("3", command)

    def test_build_template_context_handles_multilayer_sequences(self) -> None:
        context = build_template_context(self.fixture_root(), "omi launch video")
        self.assertEqual(len(context.video_tracks), 3)
        self.assertEqual(len(context.audio_tracks), 4)
        self.assertEqual(len(context.source_video_items), 6)
        self.assertEqual(len(context.source_audio_items), 11)
        self.assertEqual(len(context.link_groups), 5)

    def test_build_template_context_prefers_lone_non_selects_sequence(self) -> None:
        context = build_template_context(self.fixture_root("selects_output.xml"), None)
        self.assertEqual(context.sequence.findtext("Name"), "Kaedim Podcast Episode 9")

    def test_build_template_context_ignores_generated_concision_sequence(self) -> None:
        root = self.fixture_root("selects_output.xml")
        source_sequence = next(
            sequence
            for sequence in root
            if sequence.tag == "Sequence" and sequence.findtext("Name") == "Kaedim Podcast Episode 9"
        )
        concision_sequence = ET.fromstring(ET.tostring(source_sequence, encoding="unicode"))
        concision_sequence.find("Name").text = "Kaedim Podcast Episode 9 - Concision"  # type: ignore[union-attr]
        root.append(concision_sequence)

        context = build_template_context(root, None)

        self.assertEqual(context.sequence.findtext("Name"), "Kaedim Podcast Episode 9")

    def test_build_template_context_requires_sequence_when_multiple_sources_exist(self) -> None:
        root = self.fixture_root("selects_output.xml")
        source_sequence = next(
            sequence
            for sequence in root
            if sequence.tag == "Sequence" and sequence.findtext("Name") == "Kaedim Podcast Episode 9"
        )
        duplicate_source = ET.fromstring(ET.tostring(source_sequence, encoding="unicode"))
        duplicate_source.find("Name").text = "Kaedim Podcast Episode 9 Alternate"  # type: ignore[union-attr]
        root.append(duplicate_source)

        with self.assertRaisesRegex(
            PodcastEditorError,
            "Project contains multiple sequences; choose one with --sequence",
        ):
            build_template_context(root, None)

    def test_create_selects_sequence_copies_full_stack_for_selected_range(self) -> None:
        root = self.fixture_root()
        context = build_template_context(root, "omi launch video")
        selected_end = context.source_video_items[1].timeline_end

        result = create_selects_sequence(root, context, [TickRange(0, selected_end)])
        new_context = build_template_context(root, str(result["new_sequence"]))

        self.assertEqual(result["video_segments"], 3)
        self.assertEqual(result["audio_segments"], 4)
        self.assertEqual(result["links"], 2)
        self.assertEqual(
            [(item.track_index, item.timeline_start, item.timeline_end) for item in new_context.source_video_items],
            [
                (0, 0, 2559651494400),
                (0, 3161423865600, 3695390899200),
                (1, 0, selected_end),
            ],
        )
        self.assertEqual(
            [(item.track_index, item.timeline_start, item.timeline_end) for item in new_context.source_audio_items],
            [
                (0, 0, 2559651494400),
                (0, 3161423865600, 3695390899200),
                (1, 0, 2559651494400),
                (1, 3161423865600, 3695390899200),
            ],
        )
        self.assertEqual([len(link.track_item_ids) for link in new_context.link_groups], [3, 3])

    def test_create_concision_sequence_lifts_removed_ranges_to_review_tracks(self) -> None:
        root = self.fixture_root()
        context = build_template_context(root, "omi launch video")
        removal = TickRange(TICKS_PER_SECOND, 2 * TICKS_PER_SECOND)

        result = create_concision_sequence(root, context, [removal])
        new_context = build_template_context(root, str(result["new_sequence"]))
        original_track_one_end = next(
            item.timeline_end for item in context.source_video_items if item.track_index == 1
        )
        video_segments = [
            (item.track_index, item.timeline_start, item.timeline_end)
            for item in new_context.source_video_items
        ]
        audio_segments = [
            (item.track_index, item.timeline_start, item.timeline_end)
            for item in new_context.source_audio_items
        ]
        video_track_zero = [
            (item.timeline_start, item.timeline_end)
            for item in new_context.source_video_items
            if item.track_index == 0
        ]
        lifted_video_segments = [
            (item.track_index, item.timeline_start, item.timeline_end)
            for item in new_context.source_video_items
            if item.track_index >= 2
        ]

        self.assertEqual(len(new_context.video_tracks), 4)
        self.assertEqual(len(new_context.audio_tracks), 4)
        self.assertIn((0, removal.start), video_track_zero)
        self.assertIn((removal.end, context.source_video_items[0].timeline_end), video_track_zero)
        self.assertIn((2, removal.start, removal.end), lifted_video_segments)
        self.assertIn((3, removal.start, removal.end), lifted_video_segments)
        self.assertIn((2, removal.start, removal.end), audio_segments)
        self.assertIn((3, removal.start, removal.end), audio_segments)
        self.assertIn((1, 0, removal.start), video_segments)
        self.assertIn((1, removal.end, original_track_one_end), video_segments)
        self.assertEqual(result["assembled_duration_ticks"], context.duration_ticks)
        self.assertEqual(result["lifted_duration_ticks"], removal.end - removal.start)

if __name__ == "__main__":
    unittest.main()
