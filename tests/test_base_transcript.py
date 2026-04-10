import unittest

from base_transcript import (
    UNKNOWN_SPEAKER_ID,
    build_base_transcript,
    build_editorial_blocks_text_from_base,
    build_timecoded_transcript_from_base,
)
from podcast_editor import TICKS_PER_SECOND


class BaseTranscriptTests(unittest.TestCase):
    def test_build_base_transcript_maps_word_timings_and_speaker_ids(self) -> None:
        transcript = {
            "text": "Hello world",
            "segments": [
                {
                    "start": 0.5,
                    "end": 2.0,
                    "text": " Hello world",
                    "speaker_id": "speaker_01",
                    "words": [
                        {"word": " Hello", "start": 0.5, "end": 1.0, "speaker_id": "speaker_01"},
                        {"word": " world", "start": 1.0, "end": 1.5, "speaker_id": "speaker_01"},
                    ],
                }
            ],
        }
        manifest = [
            {
                "timeline_start_ticks": 10 * TICKS_PER_SECOND,
                "timeline_end_ticks": 20 * TICKS_PER_SECOND,
                "source_start_ticks": 0,
                "source_end_ticks": 10 * TICKS_PER_SECOND,
            }
        ]

        base = build_base_transcript(transcript, manifest, TICKS_PER_SECOND // 60)

        self.assertEqual(base["word_count"], 2)
        self.assertEqual(base["speaker_ids"], ["speaker_01"])
        self.assertEqual(base["words"][0]["speaker_id"], "speaker_01")
        self.assertEqual(base["words"][0]["start_timecode"], "00:00:10:30")
        self.assertEqual(base["words"][1]["end_timecode"], "00:00:11:30")
        self.assertEqual(base["speaker_blocks"][0]["text"], "Hello world")

    def test_build_base_transcript_defaults_missing_speaker_ids(self) -> None:
        transcript = {
            "segments": [
                {
                    "start": 0.0,
                    "end": 1.0,
                    "text": " Hello there.",
                    "words": [
                        {"word": " Hello", "start": 0.0, "end": 0.4},
                        {"word": " there.", "start": 0.4, "end": 0.9},
                    ],
                }
            ],
        }
        manifest = [
            {
                "timeline_start_ticks": 0,
                "timeline_end_ticks": 10 * TICKS_PER_SECOND,
                "source_start_ticks": 0,
                "source_end_ticks": 10 * TICKS_PER_SECOND,
            }
        ]

        base = build_base_transcript(transcript, manifest, TICKS_PER_SECOND // 60)
        text, count = build_timecoded_transcript_from_base(base)

        self.assertEqual(base["words"][0]["speaker_id"], UNKNOWN_SPEAKER_ID)
        self.assertEqual(base["segments"][0]["speaker_id"], UNKNOWN_SPEAKER_ID)
        self.assertEqual(base["known_speaker_count"], 0)
        self.assertEqual(count, 1)
        self.assertEqual(text, "00:00:00:00 - 00:00:00:54 | Hello there.\n")

    def test_build_base_transcript_preserves_diarization_error(self) -> None:
        transcript = {
            "speaker_diarization_error": "403 gated model",
            "segments": [
                {
                    "start": 0.0,
                    "end": 1.0,
                    "text": " Hello there.",
                }
            ],
        }
        manifest = [
            {
                "timeline_start_ticks": 0,
                "timeline_end_ticks": 10 * TICKS_PER_SECOND,
                "source_start_ticks": 0,
                "source_end_ticks": 10 * TICKS_PER_SECOND,
            }
        ]

        base = build_base_transcript(transcript, manifest, TICKS_PER_SECOND // 60)

        self.assertEqual(base["speaker_diarization_error"], "403 gated model")

    def test_build_editorial_blocks_text_from_base_numbers_speaker_blocks(self) -> None:
        base_transcript = {
            "speaker_blocks": [
                {
                    "speaker_id": "speaker_01",
                    "start_timecode": "00:00:00:00",
                    "end_timecode": "00:00:05:00",
                    "text": "Opening thought.",
                },
                {
                    "speaker_id": "speaker_02",
                    "start_timecode": "00:00:05:01",
                    "end_timecode": "00:00:10:00",
                    "text": "Follow-up response.",
                },
            ]
        }

        text, count = build_editorial_blocks_text_from_base(base_transcript)

        self.assertEqual(count, 2)
        self.assertEqual(
            text,
            "B0001 | speaker_01 | 00:00:00:00 - 00:00:05:00 | Opening thought.\n"
            "B0002 | speaker_02 | 00:00:05:01 - 00:00:10:00 | Follow-up response.\n",
        )


if __name__ == "__main__":
    unittest.main()
