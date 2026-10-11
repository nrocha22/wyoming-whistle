"""Wyoming event handler: buffer audio, transcribe on AudioStop."""

from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import Executor
from dataclasses import dataclass, field
from typing import List, Optional

from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioChunkConverter, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import Describe, Info
from wyoming.server import AsyncEventHandler

from .audio import ClipSaver, find_speech
from .model import CHANNELS, SAMPLE_RATE, SAMPLE_WIDTH, WhistleModel, normalize_language, pcm16_to_float

_LOGGER = logging.getLogger(__name__)


@dataclass
class Settings:
    default_language: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    trim: bool = True
    min_speech_seconds: float = 0.3
    floor_db: float = -50.0
    clip_saver: Optional[ClipSaver] = None


class WhistleEventHandler(AsyncEventHandler):
    """Handles one client connection."""

    def __init__(
        self,
        wyoming_info: Info,
        model: WhistleModel,
        model_lock: asyncio.Lock,
        executor: Executor,
        settings: Settings,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._info_event = wyoming_info.event()
        self._model = model
        self._model_lock = model_lock
        self._executor = executor
        self._settings = settings
        self._default_language = settings.default_language
        self._keywords = settings.keywords
        self._converter = AudioChunkConverter(rate=SAMPLE_RATE, width=SAMPLE_WIDTH, channels=CHANNELS)
        self._reset()

    def _reset(self) -> None:
        self._audio = bytearray()
        self._language = self._default_language
        self._request_terms: List[str] = []

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(self._info_event)
            return True

        if Transcribe.is_type(event.type):
            transcribe = Transcribe.from_event(event)
            self._language = normalize_language(transcribe.language) or self._default_language
            # Names and terms the client wants recognized (e.g. device names).
            self._request_terms = list(transcribe.transcript_names or []) + list(transcribe.transcript_terms or [])
            return True

        if AudioStart.is_type(event.type):
            self._audio = bytearray()
            return True

        if AudioChunk.is_type(event.type):
            chunk = self._converter.convert(AudioChunk.from_event(event))
            self._audio.extend(chunk.audio)
            return True

        if AudioStop.is_type(event.type):
            await self._transcribe()
            self._reset()
            # Same as the reference servers: one transcript per connection.
            return False

        return True

    async def _transcribe(self) -> None:
        settings = self._settings
        pcm = bytes(self._audio)
        samples = pcm16_to_float(pcm)
        keywords = self._keywords + self._request_terms
        started = time.perf_counter()

        bounds = find_speech(samples, SAMPLE_RATE, floor_db=settings.floor_db)
        if bounds is None or bounds.speech_seconds < settings.min_speech_seconds:
            # No speech: answer empty instead of letting the model invent text.
            result = {"text": "", "language": self._language}
            kept = 0.0
        else:
            if settings.trim:
                samples = samples[bounds.start : bounds.end]
            kept = len(samples) / SAMPLE_RATE
            async with self._model_lock:
                result = await asyncio.get_running_loop().run_in_executor(
                    self._executor, self._model.transcribe, samples, self._language, keywords
                )

        text = (result.get("text") or "").strip()
        language = result.get("language") or self._language
        elapsed_ms = (time.perf_counter() - started) * 1000
        _LOGGER.info(
            "transcribed audio_s=%.2f kept_s=%.2f ms=%.0f language=%s empty=%s",
            len(pcm) / (SAMPLE_RATE * SAMPLE_WIDTH),
            kept,
            elapsed_ms,
            language,
            not text,
        )
        _LOGGER.debug("text=%r", text)
        await self.write_event(Transcript(text=text, language=language).event())

        if settings.clip_saver is not None:
            meta = {
                "text": text,
                "language": language,
                "requested_language": self._language,
                "keywords": keywords,
                "audio_s": round(len(pcm) / (SAMPLE_RATE * SAMPLE_WIDTH), 3),
                "kept_s": round(kept, 3),
                "speech_s": round(bounds.speech_seconds, 3) if bounds else 0.0,
                "ms": round(elapsed_ms),
            }
            try:
                settings.clip_saver.save(pcm, SAMPLE_RATE, meta)
            except OSError as err:
                _LOGGER.warning("Could not save clip: %s", err)
