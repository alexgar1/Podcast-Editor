import sys
import unittest
import xml.etree.ElementTree as ET

from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

from speaker_diarization import SpeakerDiarizationConfig
from transcribe_sequence import (
    AudioMixLayer,
    SequenceAudioSegment,
    SpeakerTurn,
    annotate_transcript_with_speakers,
    build_ffmpeg_command,
    diarize_speakers,
    extract_audio_segments,
)


class TranscribeSequenceTests(unittest.TestCase):
    def fixture_root(self, name: str = "reference_2026.xml") -> ET.Element:
        fixture_path = Path(__file__).resolve().parents[1] / "fixtures" / "premiere_xml" / name
        return ET.parse(fixture_path).getroot()

    def test_extract_audio_segments_mixes_overlapping_tracks(self) -> None:
        segments = extract_audio_segments(self.fixture_root(), "omi launch video")

        self.assertEqual(len(segments), 6)
        self.assertEqual((segments[0].timeline_start_ticks, segments[0].timeline_end_ticks), (0, 2559651494400))
        self.assertEqual(len(segments[0].mix_layers), 2)
        self.assertEqual(len(segments[4].mix_layers), 2)
        self.assertEqual(len(segments[5].mix_layers), 1)

    def test_build_ffmpeg_command_uses_amix_for_multilayer_segments(self) -> None:
        segments = [
            SequenceAudioSegment(
                index=0,
                timeline_start_ticks=0,
                timeline_end_ticks=10,
                source_start_ticks=0,
                source_end_ticks=10,
                source_path="/tmp/a.wav",
                mix_layers=(
                    AudioMixLayer("/tmp/a.wav", 0, 10),
                    AudioMixLayer("/tmp/b.wav", 5, 15),
                ),
            ),
            SequenceAudioSegment(
                index=1,
                timeline_start_ticks=20,
                timeline_end_ticks=30,
                source_start_ticks=20,
                source_end_ticks=30,
                source_path="/tmp/c.wav",
                mix_layers=(AudioMixLayer("/tmp/c.wav", 20, 30),),
            ),
        ]

        command = build_ffmpeg_command(segments, Path("/tmp/out.wav"))
        command_text = " ".join(command)

        self.assertIn("amix=inputs=2:normalize=1", command_text)
        self.assertIn("concat=n=2:v=0:a=1[outa]", command_text)
        self.assertEqual(command.count("-i"), 3)

    def test_annotate_transcript_with_speakers_groups_contiguous_words(self) -> None:
        transcript = {
            "segments": [
                {
                    "start": 0.0,
                    "end": 2.0,
                    "text": " Hello there again",
                    "words": [
                        {"word": " Hello", "start": 0.0, "end": 0.4},
                        {"word": " there", "start": 0.4, "end": 0.8},
                        {"word": " again", "start": 1.6, "end": 1.9},
                    ],
                },
                {
                    "start": 2.0,
                    "end": 3.0,
                    "text": " General Kenobi",
                    "words": [
                        {"word": " General", "start": 2.0, "end": 2.4},
                        {"word": " Kenobi", "start": 2.4, "end": 2.8},
                    ],
                },
            ]
        }

        blocks = annotate_transcript_with_speakers(
            transcript,
            [
                SpeakerTurn("speaker_00", 0.0, 1.0),
                SpeakerTurn("speaker_01", 1.0, 3.0),
            ],
        )

        self.assertEqual(transcript["segments"][0]["speaker_id"], "speaker_00")
        self.assertEqual(transcript["segments"][0]["words"][2]["speaker_id"], "speaker_01")
        self.assertEqual(
            transcript["speaker_blocks"],
            [
                {
                    "block_index": 0,
                    "speaker_id": "speaker_00",
                    "start": 0.0,
                    "end": 0.8,
                    "text": "Hello there",
                    "segment_indexes": (0,),
                },
                {
                    "block_index": 1,
                    "speaker_id": "speaker_01",
                    "start": 1.6,
                    "end": 2.8,
                    "text": "again General Kenobi",
                    "segment_indexes": (0, 1),
                },
            ],
        )
        self.assertEqual(len(blocks), 2)

    def test_annotate_transcript_with_speakers_falls_back_to_segments_without_words(self) -> None:
        transcript = {
            "segments": [
                {"start": 0.0, "end": 1.0, "text": " Hello there"},
                {"start": 1.1, "end": 2.0, "text": " again"},
                {"start": 3.6, "end": 4.0, "text": " bye"},
            ]
        }

        blocks = annotate_transcript_with_speakers(
            transcript,
            [
                SpeakerTurn("speaker_00", 0.0, 2.1),
                SpeakerTurn("speaker_01", 3.5, 4.2),
            ],
        )

        self.assertEqual(len(blocks), 2)
        self.assertEqual(blocks[0].speaker_id, "speaker_00")
        self.assertEqual(blocks[0].text, "Hello there again")
        self.assertEqual(blocks[1].speaker_id, "speaker_01")
        self.assertEqual(blocks[1].text, "bye")

    def test_diarize_speakers_preloads_audio_before_running_pipeline(self) -> None:
        captured: dict[str, object] = {}

        class FakePipelineInstance:
            def __call__(self, file: object, **kwargs: object) -> object:
                captured["file"] = file
                captured["kwargs"] = kwargs

                class FakeDiarization:
                    def itertracks(self, yield_label: bool = False):
                        yield SimpleNamespace(start=0.0, end=1.0), None, "SPEAKER_A"

                return FakeDiarization()

        fake_audio_module = ModuleType("pyannote.audio")
        fake_audio_module.Pipeline = SimpleNamespace(
            from_pretrained=mock.Mock(return_value=FakePipelineInstance())
        )
        fake_pyannote_module = ModuleType("pyannote")
        fake_pyannote_module.audio = fake_audio_module

        audio_file = {"waveform": object(), "sample_rate": 16000, "uri": "test"}
        with mock.patch.dict(
            sys.modules,
            {"pyannote": fake_pyannote_module, "pyannote.audio": fake_audio_module},
        ):
            with mock.patch("transcribe_sequence.load_audio_file_for_diarization", return_value=audio_file) as load_audio:
                turns = diarize_speakers(
                    Path("/tmp/test.wav"),
                    SpeakerDiarizationConfig(enabled=True, model="."),
                )

        load_audio.assert_called_once_with(Path("/tmp/test.wav"))
        self.assertEqual(captured["file"], audio_file)
        self.assertIn("hook", captured["kwargs"])
        self.assertEqual(turns, [SpeakerTurn("speaker_00", 0.0, 1.0)])


if __name__ == "__main__":
    unittest.main()
