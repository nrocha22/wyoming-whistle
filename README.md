# wyoming-whistle

A [Wyoming protocol](https://github.com/OHF-Voice/wyoming) speech-to-text server for
[Home Assistant](https://www.home-assistant.io/voice_control/), backed by
[Cactus Whistle](https://huggingface.co/Cactus-Compute/whistle), a 16.9 MB on-device
speech recognition model (English, German, French, Spanish, Italian, Dutch, Polish).

> **Status: early (0.1.0).** Not affiliated with Home Assistant, the Open Home
> Foundation or Cactus Compute.

## Why

- **Fast on modest CPUs.** On an Intel i5-6500 (no GPU), a 2-3 s Spanish command
  transcribes in about 0.3 s, versus about 3 s for faster-whisper `small` at similar
  accuracy on the same clips. (Small synthetic test set; a public benchmark is planned.)
- **Open vocabulary.** Unlike phrase-based recognizers, it transcribes free speech.
- **Private and offline by default.** Weights and the native engine are fetched at
  image build time at pinned revisions and verified against sha256 sums. At runtime the
  server makes no network calls: telemetry is off, and the Hugging Face Hub is never
  contacted (see `wyoming_whistle/model.py` for why the `Whistle` class is used directly).

## Run (Docker)

```sh
docker build -t wyoming-whistle .
docker run -d --name wyoming-whistle -p 10300:10300 wyoming-whistle --language en
```

Then in Home Assistant: **Settings → Devices & services → Add integration → Wyoming
Protocol**, host = the Docker host, port = 10300, and pick `whistle` as the
speech-to-text engine of an Assist pipeline.

Options: `--language` (fallback when the client sends none), `--keyword` (repeatable,
biases recognition toward names), `--debug` (logs transcripts).

## Related work

[Schnurzel700/whistle-wyoming](https://github.com/Schnurzel700/whistle-wyoming) is an
earlier Home Assistant add-on for Whistle. This project focuses on a standalone Docker
image for HA Container users, multi-architecture builds, pinned and verified model
downloads, and offline-by-default operation.

## License

Apache-2.0. Whistle weights and the Needle engine are published by Cactus Compute under
Apache-2.0.
