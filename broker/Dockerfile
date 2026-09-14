FROM python:3.12-slim

WORKDIR /srv
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install from the fully-pinned lockfile for reproducible builds. requirements.txt
# is kept as the human-readable top-level intent; requirements.lock is the frozen
# set actually installed (regenerate with `pip freeze` after a deliberate bump).
#
# The Live engine (ENGINE=live, gpt-live-1) needs pipecat from main, pinned by
# commit in requirements-live.txt; the Realtime engine keeps the lockfile.
# One Dockerfile serves both:
#   docker build broker/                                        # realtime
#   docker build --build-arg REQS=requirements-live.txt broker/ # live
ARG REQS=requirements.lock
COPY requirements.txt requirements.lock requirements-live.txt ./
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && pip install --no-cache-dir --timeout 180 --retries 6 -r ${REQS} \
    && apt-get purge -y git && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

COPY realtime_broker/ /srv/realtime_broker/

ENV PYTHONUNBUFFERED=1
CMD ["python", "-m", "realtime_broker"]
