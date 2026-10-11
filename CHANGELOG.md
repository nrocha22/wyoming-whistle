# Changelog

## 0.2.0 - 2026-10-11

- Keywords from Home Assistant: names and aliases of entities exposed to Assist, plus
  area and floor names (`--hass-url`, token in `WHISTLE_HASS_TOKEN`), refreshed in the
  background.
- Speech detection measured against each clip's noise floor, so silence trimming works
  on phone microphones with automatic gain.
- Empty transcript for clips without speech, instead of a hallucinated word.
- Opt-in `--save-audio-dir` and `scripts/replay_clips.py` to evaluate changes on real voices.

## 0.1.0 - 2026-10-11

- Wyoming speech-to-text server backed by Cactus Whistle.
- Offline by default: model and engine fetched at build time at pinned revisions and
  verified by sha256; no telemetry, no Hugging Face Hub calls at runtime.
- Docker image for amd64 and arm64, health check, `scripts/transcribe_file.py`.
