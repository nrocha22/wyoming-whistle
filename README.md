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
docker run -d --name wyoming-whistle -p 10300:10300 \
  ghcr.io/nrocha22/wyoming-whistle:main --language en
```

Images are built for `linux/amd64` and `linux/arm64`. Or build it yourself:
`docker build -t wyoming-whistle .` (the build downloads the pinned model and engine and
checks their sha256).

> arm64 is verified in CI (under emulation: the engine loads and the server answers
> with no network), but speed on a Raspberry Pi has not been measured yet. Reports welcome.

Then in Home Assistant: **Settings → Devices & services → Add integration → Wyoming
Protocol**, host = the Docker host, port = 10300, and pick `whistle` as the
speech-to-text engine of an Assist pipeline.

Options: `--language` (fallback when the client sends none), `--keyword` (repeatable,
biases recognition toward names), `--debug` (logs transcripts).

### Keywords from Home Assistant

```sh
docker run -d --name wyoming-whistle -p 10300:10300 \
  -e WHISTLE_HASS_TOKEN=<long-lived access token> \
  wyoming-whistle --language es --hass-url ws://homeassistant.local:8123/api/websocket \
  --keyword apaga --keyword enciende --keyword abre --keyword cierra
```

With `--hass-url`, the names and aliases of entities exposed to Assist, plus area and floor
names, are used as keywords and refreshed every 5 minutes. Transcription never waits for
Home Assistant; if it is unreachable the last list is kept. This is the only network
connection the server makes, and only when configured. The token is read from the
environment so it does not show up in process listings.

**Add your command verbs with `--keyword`.** Measured on 15 real phone recordings in
Spanish: with no keywords, "apaga" came back as "A paga" (and Assist no longer understood
the command); Home Assistant names alone did not fix it, because they contain no verbs;
names plus a handful of verbs did. Combined with a few custom sentences, Assist understood
15 of 15 commands, versus 5 of 9 for a phrase-based recognizer on the same speaker.

### Debugging with real voices

`--save-audio-dir DIR` keeps the last utterances (`--save-audio-max`, default 100) as WAV +
JSON, mode 0600. `scripts/replay_clips.py` re-runs them with different settings, so changes
can be judged on real speech without asking anyone to repeat themselves. These files are
recordings of people: keep them local, out of shared backups, and delete them when done.

## Related work

[Schnurzel700/whistle-wyoming](https://github.com/Schnurzel700/whistle-wyoming) is an
earlier Home Assistant add-on for Whistle. This project focuses on a standalone Docker
image for HA Container users, multi-architecture builds, pinned and verified model
downloads, and offline-by-default operation.

## License

Apache-2.0. Whistle weights and the Needle engine are published by Cactus Compute under
Apache-2.0.
