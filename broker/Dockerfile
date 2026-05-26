FROM python:3.12-slim

WORKDIR /srv
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --timeout 180 --retries 6 -r requirements.txt

COPY realtime_broker/ /srv/realtime_broker/

ENV PYTHONUNBUFFERED=1
CMD ["python", "-m", "realtime_broker"]
