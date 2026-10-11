"""Send WAV files to a Wyoming speech-to-text server and print the transcripts.

    python scripts/transcribe_file.py --uri tcp://127.0.0.1:10300 --language es a.wav b.wav

Works with any Wyoming ASR server, which makes it handy for side-by-side checks.
Accepts streaming WAVs whose header carries no length (as Piper writes them).
"""

from __future__ import annotations

import argparse
import asyncio
import time
import wave

from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.client import AsyncClient

CHUNK_SECONDS = 0.1


def read_wav(path: str):
    with wave.open(path, "rb") as source:
        rate, width, channels = source.getframerate(), source.getsampwidth(), source.getnchannels()
        frames = source.getnframes()
        audio = source.readframes(frames) if frames else b""
    if not audio:  # streaming header: take everything after the data chunk
        with open(path, "rb") as handle:
            raw = handle.read()
        audio = raw[raw.index(b"data") + 8 :]
        audio = audio[: len(audio) - len(audio) % (width * channels)]
    return rate, width, channels, audio


async def transcribe(uri: str, path: str, language: str | None) -> tuple[str, float]:
    rate, width, channels, audio = read_wav(path)
    step = int(rate * CHUNK_SECONDS) * width * channels
    async with AsyncClient.from_uri(uri) as client:
        await client.write_event(Transcribe(language=language).event())
        await client.write_event(AudioStart(rate=rate, width=width, channels=channels).event())
        for start in range(0, len(audio), step):
            chunk = AudioChunk(rate=rate, width=width, channels=channels, audio=audio[start : start + step])
            await client.write_event(chunk.event())
        sent = time.perf_counter()
        await client.write_event(AudioStop().event())
        while True:
            event = await client.read_event()
            if event is None:
                raise RuntimeError("server closed the connection without a transcript")
            if Transcript.is_type(event.type):
                return Transcript.from_event(event).text, time.perf_counter() - sent


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uri", default="tcp://127.0.0.1:10300")
    parser.add_argument("--language")
    parser.add_argument("wav", nargs="+")
    args = parser.parse_args()
    for path in args.wav:
        text, seconds = await transcribe(args.uri, path, args.language)
        print(f"{seconds * 1000:6.0f} ms  {path}: {text}")


if __name__ == "__main__":
    asyncio.run(main())
