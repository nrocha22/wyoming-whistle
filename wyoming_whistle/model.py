"""Thin adapter over the Whistle class from cactus-needle.

Two choices here keep the server private and offline, and both are deliberate:

* The ``Whistle`` class is used directly. The module-level ``needle.transcribe``
  helper sends a telemetry event on every call; the class does not.
* Weights and the native engine are loaded from explicit local paths. If either
  is resolved by cactus-needle itself, it calls the Hugging Face Hub (to download,
  or to count a download even when the file is already cached).
"""

from __future__ import annotations

import logging
import os
from typing import Iterable, Optional

import numpy as np

_LOGGER = logging.getLogger(__name__)

LANGUAGES = ("en", "de", "fr", "es", "it", "nl", "pl")
SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2
CHANNELS = 1

# The model card describes a 30 s window. Longer audio is still passed through,
# but it is worth a warning: voice commands are a few seconds long.
LONG_AUDIO_SECONDS = 30


def normalize_language(language: Optional[str]) -> Optional[str]:
    """Map a Home Assistant language tag (es, es-ES, es_MX) to a Whistle code.

    Returns None for unsupported languages, which makes Whistle detect it.
    """
    if not language:
        return None
    base = language.replace("_", "-").split("-", 1)[0].lower()
    return base if base in LANGUAGES else None


def pcm16_to_float(pcm: bytes) -> np.ndarray:
    """16-bit little-endian PCM to float32 samples in [-1, 1].

    Whistle reads raw ``bytes`` as float32, not as PCM, so passing Wyoming audio
    straight through would transcribe noise without raising an error.
    """
    usable = len(pcm) - (len(pcm) % SAMPLE_WIDTH)
    return np.frombuffer(pcm[:usable], dtype="<i2").astype(np.float32) / 32768.0


class WhistleModel:
    """One loaded Whistle model. Not thread-safe: callers must serialize."""

    def __init__(self, weights: str, engine_lib: str) -> None:
        for path, what in ((weights, "weights"), (engine_lib, "engine library")):
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Whistle {what} not found: {path}")

        # Read by cactus-needle when the engine is loaded. Set before import so no
        # code path can fall back to downloading a library.
        os.environ["NEEDLE3_LIB_PATH"] = engine_lib
        os.environ["NEEDLE_TELEMETRY"] = "0"

        from needle.agent.whistle import Whistle  # pylint: disable=import-outside-toplevel

        self.weights = weights
        self._model = Whistle(weights=weights)

    def transcribe(
        self,
        samples: np.ndarray,
        language: Optional[str] = None,
        keywords: Optional[Iterable[str]] = None,
    ) -> dict:
        """Returns Whistle's result dict: text, language, ttft_ms, decode_tps."""
        seconds = len(samples) / SAMPLE_RATE
        if seconds > LONG_AUDIO_SECONDS:
            _LOGGER.warning("Audio is %.1f s, longer than the %s s model window", seconds, LONG_AUDIO_SECONDS)

        terms = [k for k in (keywords or []) if k and k.strip()]
        return self._model.transcribe(samples, language=language, keywords=terms or None)
