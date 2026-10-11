import numpy as np

from wyoming_whistle.model import normalize_language, pcm16_to_float


def test_normalize_language():
    assert normalize_language("es") == "es"
    assert normalize_language("es-ES") == "es"
    assert normalize_language("es_MX") == "es"
    assert normalize_language("PT-br") is None  # unsupported -> detect
    assert normalize_language(None) is None
    assert normalize_language("") is None


def test_pcm16_to_float_range_and_dtype():
    pcm = np.array([0, 16384, -32768, 32767], dtype="<i2").tobytes()
    samples = pcm16_to_float(pcm)
    assert samples.dtype == np.float32
    assert np.allclose(samples, [0.0, 0.5, -1.0, 32767 / 32768])


def test_pcm16_to_float_drops_odd_trailing_byte():
    assert len(pcm16_to_float(b"\x00\x00\x01")) == 1
