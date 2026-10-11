"""Entry point: python -m wyoming_whistle"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import numpy as np
from wyoming.info import AsrModel, AsrProgram, Attribution, Info
from wyoming.server import AsyncServer

from . import MODEL_REVISION, __version__
from .audio import ClipSaver
from .handler import Settings, WhistleEventHandler
from .hass_api import HassKeywords
from .model import LANGUAGES, SAMPLE_RATE, WhistleModel

_LOGGER = logging.getLogger(__name__)

ATTRIBUTION = Attribution(name="Cactus Compute", url="https://huggingface.co/Cactus-Compute/whistle")


def build_info() -> Info:
    return Info(
        asr=[
            AsrProgram(
                name="whistle",
                description="Cactus Whistle speech-to-text (local, offline)",
                attribution=ATTRIBUTION,
                installed=True,
                version=__version__,
                models=[
                    AsrModel(
                        name="whistle",
                        description="Whistle 16.9 MB",
                        attribution=ATTRIBUTION,
                        installed=True,
                        languages=list(LANGUAGES),
                        version=MODEL_REVISION[:7],
                    )
                ],
                supports_transcript_streaming=False,
            )
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="wyoming_whistle")
    parser.add_argument("--uri", default="tcp://0.0.0.0:10300", help="unix:// or tcp://")
    parser.add_argument("--weights", default=os.environ.get("WHISTLE_WEIGHTS", "/models/whistle.cact"))
    parser.add_argument("--engine-lib", default=os.environ.get("WHISTLE_ENGINE_LIB", "/models/libneedle3.so"))
    parser.add_argument(
        "--language",
        choices=LANGUAGES,
        help="Language used when the client sends none or an unsupported one (default: detect)",
    )
    parser.add_argument(
        "--keyword",
        action="append",
        default=[],
        help="Word or phrase to bias recognition towards (repeatable)",
    )
    parser.add_argument(
        "--no-trim",
        action="store_true",
        help="Transcribe the whole clip instead of trimming leading/trailing silence",
    )
    parser.add_argument(
        "--min-speech-seconds",
        type=float,
        default=0.3,
        help="Clips with less speech than this return an empty transcript (default: 0.3)",
    )
    parser.add_argument(
        "--vad-floor-db",
        type=float,
        default=-50.0,
        help="Frames quieter than this (dBFS) never count as speech (default: -50)",
    )
    parser.add_argument(
        "--save-audio-dir",
        help="Opt-in: keep the last utterances here as WAV + JSON for debugging. Contains voices.",
    )
    parser.add_argument("--save-audio-max", type=int, default=100, help="Clips kept by --save-audio-dir")
    parser.add_argument(
        "--hass-url",
        default=os.environ.get("WHISTLE_HASS_URL"),
        help="Home Assistant websocket URL, e.g. ws://homeassistant.local:8123/api/websocket. "
        "Entity, area and floor names become keywords. Token from WHISTLE_HASS_TOKEN.",
    )
    parser.add_argument("--hass-refresh-seconds", type=float, default=300.0)
    parser.add_argument("--max-keywords", type=int, default=200)
    parser.add_argument("--debug", action="store_true", help="Log DEBUG messages, including transcripts")
    parser.add_argument("--log-format", default=logging.BASIC_FORMAT)
    parser.add_argument("--version", action="version", version=__version__)
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO, format=args.log_format)

    _LOGGER.info("Loading Whistle (model %s)", MODEL_REVISION[:7])
    model = WhistleModel(weights=args.weights, engine_lib=args.engine_lib)

    # Self-test: fails at startup, not on the first voice command, if the engine
    # or weights are unusable.
    model.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32), language=args.language or "en")
    _LOGGER.info("Model ready")

    clip_saver = None
    if args.save_audio_dir:
        clip_saver = ClipSaver(args.save_audio_dir, args.save_audio_max)
        _LOGGER.warning("Saving the last %s clips to %s (contains voices)", args.save_audio_max, args.save_audio_dir)

    hass = None
    if args.hass_url:
        token = os.environ.get("WHISTLE_HASS_TOKEN")
        if not token:
            raise SystemExit("--hass-url needs a long-lived access token in WHISTLE_HASS_TOKEN")
        hass = HassKeywords(args.hass_url, token, args.hass_refresh_seconds, args.max_keywords)

    settings = Settings(
        default_language=args.language,
        keywords=args.keyword,
        trim=not args.no_trim,
        min_speech_seconds=args.min_speech_seconds,
        floor_db=args.vad_floor_db,
        clip_saver=clip_saver,
        hass=hass,
        max_keywords=args.max_keywords,
    )
    if settings.keywords:
        _LOGGER.info("Keywords: %s", ", ".join(settings.keywords))

    if hass is not None:
        # Background refresh; transcription never waits for Home Assistant.
        hass_task = asyncio.create_task(hass.run())  # noqa: F841  pylint: disable=unused-variable

    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="whistle")
    server = AsyncServer.from_uri(args.uri)
    _LOGGER.info("Listening on %s", args.uri)
    await server.run(partial(WhistleEventHandler, build_info(), model, asyncio.Lock(), executor, settings))


def run() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        pass
