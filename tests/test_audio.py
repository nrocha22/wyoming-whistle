import json
import os

import numpy as np

from wyoming_whistle.audio import ClipSaver, find_speech

RATE = 16000


def tone(seconds, level=0.3):
    t = np.arange(int(seconds * RATE)) / RATE
    return (np.sin(2 * np.pi * 220 * t) * level).astype(np.float32)


def test_silence_has_no_speech():
    assert find_speech(np.zeros(RATE, dtype=np.float32), RATE) is None


def test_faint_noise_is_not_speech():
    noise = (np.random.default_rng(0).standard_normal(RATE) * 1e-4).astype(np.float32)  # about -80 dBFS
    assert find_speech(noise, RATE) is None


def test_bounds_include_padding():
    samples = np.concatenate([np.zeros(RATE, np.float32), tone(1.0), np.zeros(2 * RATE, np.float32)])
    bounds = find_speech(samples, RATE, pad_seconds=0.25)
    assert 0.7 * RATE <= bounds.start <= 0.8 * RATE
    assert 2.2 * RATE <= bounds.end <= 2.3 * RATE
    assert 0.9 <= bounds.speech_seconds <= 1.05


def test_clip_saver_rotates(tmp_path):
    saver = ClipSaver(str(tmp_path / "clips"), max_clips=2)
    pcm = (tone(0.1) * 32767).astype("<i2").tobytes()
    for i in range(3):
        saver.save(pcm, RATE, {"text": f"clip {i}"})
        os.utime(tmp_path / "clips", None)
    wavs = sorted(f for f in os.listdir(tmp_path / "clips") if f.endswith(".wav"))
    assert len(wavs) <= 2
    meta = json.loads((tmp_path / "clips" / (wavs[-1][:-4] + ".json")).read_text())
    assert meta["text"] == "clip 2"
    assert oct(os.stat(tmp_path / "clips").st_mode & 0o777) == "0o700"
