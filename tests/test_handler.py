"""Protocol round-trip against a fake model: no engine needed."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import numpy as np
import pytest
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.client import AsyncTcpClient
from wyoming.info import Describe, Info
from wyoming.server import AsyncTcpServer

from wyoming_whistle.__main__ import build_info
from wyoming_whistle.handler import WhistleEventHandler


class FakeModel:
    def __init__(self):
        self.calls = []

    def transcribe(self, samples, language=None, keywords=None):
        self.calls.append((samples, language, list(keywords or [])))
        return {"text": " turn on the living room light ", "language": language or "en"}


@pytest.fixture
async def server():
    model = FakeModel()
    srv = AsyncTcpServer("127.0.0.1", 0)
    factory = partial(
        WhistleEventHandler, build_info(), model, asyncio.Lock(), ThreadPoolExecutor(1), None, ["kitchen"]
    )
    task = asyncio.create_task(srv.run(factory))
    for _ in range(50):
        if srv._server is not None:  # pylint: disable=protected-access
            break
        await asyncio.sleep(0.02)
    port = srv._server.sockets[0].getsockname()[1]  # pylint: disable=protected-access
    yield port, model
    task.cancel()


async def read_until(client, predicate):
    while True:
        event = await asyncio.wait_for(client.read_event(), timeout=5)
        assert event is not None
        if predicate(event.type):
            return event


async def test_describe(server):
    port, _ = server
    async with AsyncTcpClient("127.0.0.1", port) as client:
        await client.write_event(Describe().event())
        info = Info.from_event(await read_until(client, Info.is_type))
    assert info.asr[0].name == "whistle"
    assert "es" in info.asr[0].models[0].languages


async def test_transcribe_round_trip(server):
    port, model = server
    pcm = (np.ones(8000, dtype="<i2") * 16384).tobytes()  # 0.5 s at 16 kHz
    async with AsyncTcpClient("127.0.0.1", port) as client:
        await client.write_event(Transcribe(language="es-ES", transcript_names=["living room"]).event())
        await client.write_event(AudioStart(rate=16000, width=2, channels=1).event())
        await client.write_event(AudioChunk(rate=16000, width=2, channels=1, audio=pcm).event())
        await client.write_event(AudioStop().event())
        transcript = Transcript.from_event(await read_until(client, Transcript.is_type))

    assert transcript.text == "turn on the living room light"
    samples, language, keywords = model.calls[0]
    assert language == "es"
    assert keywords == ["kitchen", "living room"]
    assert samples.dtype == np.float32 and len(samples) == 8000
    assert np.allclose(samples, 0.5)


async def test_resamples_other_formats(server):
    port, model = server
    pcm = np.zeros(4000, dtype="<i2").tobytes()  # 0.5 s at 8 kHz
    async with AsyncTcpClient("127.0.0.1", port) as client:
        await client.write_event(AudioStart(rate=8000, width=2, channels=1).event())
        await client.write_event(AudioChunk(rate=8000, width=2, channels=1, audio=pcm).event())
        await client.write_event(AudioStop().event())
        await read_until(client, Transcript.is_type)

    samples, _, _ = model.calls[0]
    assert abs(len(samples) - 8000) <= 2  # upsampled to 16 kHz
