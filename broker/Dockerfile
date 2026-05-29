FROM python:3.12-slim

WORKDIR /srv
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install from the fully-pinned lockfile for reproducible builds. requirements.txt
# is kept as the human-readable top-level intent; requirements.lock is the frozen
# set actually installed (regenerate with `pip freeze` after a deliberate bump).
COPY requirements.txt requirements.lock ./
RUN pip install --no-cache-dir --timeout 180 --retries 6 -r requirements.lock

COPY realtime_broker/ /srv/realtime_broker/

ENV PYTHONUNBUFFERED=1
CMD ["python", "-m", "realtime_broker"]
