# callswap: serves a trained run as an OpenAI-compatible endpoint with LLM fallback.
# The image holds the base install only (TF-IDF models). No API key is baked in: configure the teacher at run time.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CALLSWAP_RUN_DIR=/runs/demo \
    CALLSWAP_HOST=0.0.0.0 \
    CALLSWAP_PORT=8000

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY callswap ./callswap
RUN pip install . \
    && useradd --create-home --uid 10001 callswap \
    && mkdir -p /runs /data \
    && chown callswap /runs /data

USER callswap
WORKDIR /runs
VOLUME /runs
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=3)"

# `docker run IMAGE` serves; `docker run IMAGE run /data/logs.csv --out /runs/mine` trains.
ENTRYPOINT ["callswap"]
CMD ["serve", "--demo-if-missing"]
