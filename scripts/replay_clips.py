"""Re-run saved clips (from --save-audio-dir) through Whistle with different settings.

Compares what the server answered then (the clip's .json) with what the current
code answers now, so a change can be judged against real voices without asking
anyone to repeat themselves. Runs inside the image:

    docker exec wyoming-whistle python /app/scripts/replay_clips.py /clips --language es
    docker exec wyoming-whistle python /app/scripts/replay_clips.py /clips --no-trim --no-keywords
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time
import wave

import numpy as np

from wyoming_whistle.audio import find_speech
from wyoming_whistle.model import SAMPLE_RATE, WhistleModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("clips_dir")
    parser.add_argument("--language")
    parser.add_argument("--no-trim", action="store_true")
    parser.add_argument("--no-keywords", action="store_true", help="ignore the keywords stored with each clip")
    parser.add_argument("--keyword", action="append", default=[], help="override keywords (repeatable)")
    parser.add_argument("--above-noise-db", type=float, default=9.0)
    parser.add_argument("--min-speech-seconds", type=float, default=0.3)
    parser.add_argument("--weights", default=os.environ.get("WHISTLE_WEIGHTS", "/models/whistle.cact"))
    parser.add_argument("--engine-lib", default=os.environ.get("WHISTLE_ENGINE_LIB", "/models/libneedle3.so"))
    args = parser.parse_args()

    model = WhistleModel(args.weights, args.engine_lib)
    total_ms = 0.0
    for path in sorted(glob.glob(os.path.join(args.clips_dir, "*.wav"))):
        meta_path = path[:-4] + ".json"
        meta = json.load(open(meta_path, encoding="utf-8")) if os.path.exists(meta_path) else {}
        with wave.open(path, "rb") as clip:
            samples = np.frombuffer(clip.readframes(clip.getnframes()), dtype="<i2").astype(np.float32) / 32768

        keywords = args.keyword or ([] if args.no_keywords else meta.get("keywords", []))
        language = args.language or meta.get("requested_language")
        started = time.perf_counter()
        bounds = find_speech(samples, SAMPLE_RATE, above_noise_db=args.above_noise_db)
        if bounds is None or bounds.speech_seconds < args.min_speech_seconds:
            text, kept = "", 0.0
        else:
            if not args.no_trim:
                samples = samples[bounds.start : bounds.end]
            kept = len(samples) / SAMPLE_RATE
            text = (model.transcribe(samples, language=language, keywords=keywords).get("text") or "").strip()
        ms = (time.perf_counter() - started) * 1000
        total_ms += ms

        before = meta.get("text")
        mark = "=" if before == text else "~"
        print(f"{os.path.basename(path)[:-4]}  kept {kept:4.1f}s  {ms:5.0f} ms  {mark}  {text!r}")
        if before is not None and before != text:
            print(f"{'':19}antes: {before!r}")
    print(f"total {total_ms / 1000:.1f} s")


if __name__ == "__main__":
    main()
