from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

DEFAULT_SPEAKER_DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"
DEFAULT_SPEAKER_TOKEN_ENV_VARS = ("PYANNOTE_AUTH_TOKEN", "HUGGINGFACE_TOKEN", "HF_TOKEN")


@dataclass(frozen=True)
class SpeakerDiarizationConfig:
    enabled: bool = True
    model: str = DEFAULT_SPEAKER_DIARIZATION_MODEL
    auth_token: str | None = None
    num_speakers: int | None = None
    min_speakers: int | None = None
    max_speakers: int | None = None


def add_speaker_diarization_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("speaker diarization")
    group.add_argument(
        "--detect-speakers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable or disable speaker detection. When enabled, transcription tries to assign a speaker ID "
            "to each word and continuous speaker block."
        ),
    )
    group.add_argument(
        "--speaker-diarization-model",
        default=DEFAULT_SPEAKER_DIARIZATION_MODEL,
        help="Speaker diarization pipeline to load, or a local path to a downloaded pyannote pipeline.",
    )
    group.add_argument(
        "--speaker-auth-token",
        help=(
            "Optional Hugging Face or pyannote token. Falls back to PYANNOTE_AUTH_TOKEN, "
            "HUGGINGFACE_TOKEN, or HF_TOKEN."
        ),
    )
    group.add_argument(
        "--num-speakers",
        type=int,
        help="Known speaker count for diarization. Cannot be combined with --min-speakers/--max-speakers.",
    )
    group.add_argument(
        "--min-speakers",
        type=int,
        help="Lower bound for the number of speakers during diarization.",
    )
    group.add_argument(
        "--max-speakers",
        type=int,
        help="Upper bound for the number of speakers during diarization.",
    )


def speaker_diarization_config_from_args(args: argparse.Namespace) -> SpeakerDiarizationConfig:
    config = SpeakerDiarizationConfig(
        enabled=bool(getattr(args, "detect_speakers", True)),
        model=str(getattr(args, "speaker_diarization_model", DEFAULT_SPEAKER_DIARIZATION_MODEL)),
        auth_token=getattr(args, "speaker_auth_token", None),
        num_speakers=getattr(args, "num_speakers", None),
        min_speakers=getattr(args, "min_speakers", None),
        max_speakers=getattr(args, "max_speakers", None),
    )
    validate_speaker_diarization_config(config)
    return config


def validate_speaker_diarization_config(config: SpeakerDiarizationConfig) -> None:
    if config.num_speakers is not None and (config.min_speakers is not None or config.max_speakers is not None):
        raise ValueError("--num-speakers cannot be combined with --min-speakers or --max-speakers.")
    if config.min_speakers is not None and config.max_speakers is not None and config.min_speakers > config.max_speakers:
        raise ValueError("--min-speakers cannot be greater than --max-speakers.")


def resolve_speaker_auth_token(explicit_token: str | None) -> str | None:
    if explicit_token:
        return explicit_token
    for env_name in DEFAULT_SPEAKER_TOKEN_ENV_VARS:
        token = os.environ.get(env_name)
        if token:
            return token
    return None


def append_speaker_diarization_command_args(command: list[str], config: SpeakerDiarizationConfig) -> None:
    command.append("--detect-speakers" if config.enabled else "--no-detect-speakers")
    command.extend(["--speaker-diarization-model", config.model])
    if config.auth_token:
        command.extend(["--speaker-auth-token", config.auth_token])
    if config.num_speakers is not None:
        command.extend(["--num-speakers", str(config.num_speakers)])
    if config.min_speakers is not None:
        command.extend(["--min-speakers", str(config.min_speakers)])
    if config.max_speakers is not None:
        command.extend(["--max-speakers", str(config.max_speakers)])
