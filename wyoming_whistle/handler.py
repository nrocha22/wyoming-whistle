"""Wyoming event handler: buffer audio, transcribe on AudioStop."""

from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import Executor
from typing import List, Optional

from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioChunkConverter, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import Describe, Info
from wyoming.server import AsyncEventHandler

from .model import CHANNELS, SAMPLE_RATE, SAMPLE_WIDTH, WhistleModel, normalize_language, pcm16_to_float

_LOGGER = logging.getLogger(__name__)


class WhistleEventHandler(AsyncEventHandler):
    """Handles one client connection."""

    def __init__(
        self,
        wyoming_info: Info,
        model: WhistleModel,
        model_lock: asyncio.Lock,
        executor: Executor,
        default_language: Optional[str],
        keywords: List[str],
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._info_event = wyoming_info.event()
        self._model = model
        self._model_lock = model_lock
        self._executor = executor
        self._default_language = default_language
        self._keywords = keywords
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
        samples = pcm16_to_float(bytes(self._audio))
        keywords = self._keywords + self._request_terms
        started = time.perf_counter()

        async with self._model_lock:
            result = await asyncio.get_running_loop().run_in_executor(
                self._executor, self._model.transcribe, samples, self._language, keywords
            )

        text = (result.get("text") or "").strip()
        _LOGGER.info(
            "transcribed audio_s=%.2f ms=%.0f language=%s empty=%s",
            len(samples) / SAMPLE_RATE,
            (time.perf_counter() - started) * 1000,
            result.get("language") or self._language,
            not text,
        )
        _LOGGER.debug("text=%r", text)
        await self.write_event(Transcript(text=text, language=result.get("language") or self._language).event())
