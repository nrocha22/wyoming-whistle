"""Audio helpers: silence detection, trimming, and opt-in clip saving.

Why trimming matters for Whistle (measured with real phone audio): clients often
send 5-7 s of audio for a 2 s command. Transcription time grew from ~0.25 s to
~1.5 s on that trailing silence, and the model sometimes invented words for it
("... al 50%"). Near-empty clips (a tap on the mic button) came back as
"Gracias.", the usual hallucination of this kind of model on silence.
"""

from __future__ import annotations

import json
import os
import time
import wave
from dataclasses import dataclass
from typing import Optional

import numpy as np

FRAME_SECONDS = 0.03


@dataclass
class SpeechBounds:
    start: int  # first sample to keep
    end: int  # one past the last sample to keep
    speech_seconds: float  # total length of the frames classified as speech


def frame_db(samples: np.ndarray, rate: int) -> np.ndarray:
    """RMS level of each 30 ms frame, in dBFS."""
    size = max(1, int(rate * FRAME_SECONDS))
    count = len(samples) // size
    if count == 0:
        return np.zeros(0, dtype=np.float32)
    frames = samples[: count * size].reshape(count, size)
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    return (20 * np.log10(np.maximum(rms, 1e-10))).astype(np.float32)


def find_speech(
    samples: np.ndarray,
    rate: int,
    floor_db: float = -50.0,
    below_peak_db: float = 30.0,
    above_noise_db: float = 9.0,
    pad_seconds: float = 0.25,
) -> Optional[SpeechBounds]:
    """Locate speech by energy. Returns None when there is none.

    A frame counts as speech when it is louder than ``floor_db``, within
    ``below_peak_db`` of the loudest frame, and ``above_noise_db`` over the
    clip's noise floor (its 10th percentile level).

    The noise-floor test is what makes this work on phones: with automatic gain,
    measured phone clips sit at a steady -35 dBFS between words while speech
    peaks at -12 to -25, so the peak-relative test alone kept every frame.
    Clean audio (silence near -inf) is unaffected: there the peak test decides.
    """
    levels = frame_db(samples, rate)
    if len(levels) == 0:
        return None
    peak = float(levels.max())
    noise = float(np.percentile(levels, 10))
    threshold = max(floor_db, peak - below_peak_db, min(peak - 6.0, noise + above_noise_db))
    speech = np.flatnonzero(levels > threshold)
    if len(speech) == 0:
        return None
    size = int(rate * FRAME_SECONDS)
    pad = int(rate * pad_seconds)
    start = max(0, int(speech[0]) * size - pad)
    end = min(len(samples), (int(speech[-1]) + 1) * size + pad)
    return SpeechBounds(start=start, end=end, speech_seconds=len(speech) * FRAME_SECONDS)


class ClipSaver:
    """Opt-in: keep the last ``max_clips`` utterances as WAV + JSON, for local debugging.

    Off by default. The files contain people's voices: keep them on the machine,
    out of backups you share, and delete them when done.
    """

    def __init__(self, directory: str, max_clips: int = 100) -> None:
        self.directory = directory
        self.max_clips = max_clips
        os.makedirs(directory, mode=0o700, exist_ok=True)

    def save(self, pcm16: bytes, rate: int, meta: dict) -> str:
        stamp = time.strftime("%Y%m%d-%H%M%S") + f"-{int(time.time() * 1000) % 1000:03d}"
        base = os.path.join(self.directory, stamp)
        old_umask = os.umask(0o077)
        try:
            with wave.open(base + ".wav", "wb") as out:
                out.setnchannels(1)
                out.setsampwidth(2)
                out.setframerate(rate)
                out.writeframes(pcm16)
            with open(base + ".json", "w", encoding="utf-8") as out:
                json.dump(meta, out, ensure_ascii=False, indent=1)
        finally:
            os.umask(old_umask)
        self._rotate()
        return base + ".wav"

    def _rotate(self) -> None:
        clips = sorted(f for f in os.listdir(self.directory) if f.endswith(".wav"))
        for name in clips[: max(0, len(clips) - self.max_clips)]:
            for ext in (".wav", ".json"):
                try:
                    os.remove(os.path.join(self.directory, name[:-4] + ext))
                except FileNotFoundError:
                    pass
