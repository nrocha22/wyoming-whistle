# Stage 1: fetch weights + engine at pinned revisions and verify their sha256.
FROM python:3.12-slim-bookworm AS fetch
ARG TARGETARCH
RUN pip install --no-cache-dir huggingface_hub==0.36.0
COPY scripts/fetch_models.py models/CHECKSUMS.sha256 /build/
RUN python /build/fetch_models.py --arch "${TARGETARCH}" --out /models --checksums /build/CHECKSUMS.sha256

# Stage 2: the server. Offline by default: no telemetry, no Hub access.
FROM python:3.12-slim-bookworm
ENV PYTHONUNBUFFERED=1 \
    NEEDLE_TELEMETRY=0 \
    DO_NOT_TRACK=1 \
    HF_HUB_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    WHISTLE_WEIGHTS=/models/whistle.cact \
    WHISTLE_ENGINE_LIB=/models/libneedle3.so

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY wyoming_whistle ./wyoming_whistle
COPY scripts ./scripts
RUN pip install --no-cache-dir . && useradd --system --uid 10300 whistle
COPY --from=fetch /models /models

USER whistle
EXPOSE 10300
HEALTHCHECK --interval=60s --timeout=10s --start-period=60s --retries=3 \
    CMD ["python", "-m", "wyoming_whistle.health_check"]
ENTRYPOINT ["python", "-m", "wyoming_whistle"]
