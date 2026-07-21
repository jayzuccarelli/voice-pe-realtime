"""M2: play the reference clip through the puck's speaker via HA media_player.

Generates a known TTS clip (once, cached next to the captures), serves it
over HTTP from this machine, and asks Home Assistant to play it on the
puck's media_player entity. This drives the speaker through the normal HA
media path, the broker never sends a binary frame, so the firmware mic
gate stays open and m2_capture.py records the XMOS post-AEC residual of
exactly this clip.

    export OPENAI_API_KEY=...   # only needed the first time (TTS)
    export HA_TOKEN=...
    uv run tools/m2_play_ref.py --entity media_player.home_assistant_voice_xxxxxx_media_player

Keep the clip short (default text is ~8s): the firmware kills an idle
voice session after 10s, and the mic must still be streaming while the
clip plays.
"""

from __future__ import annotations

import argparse
import functools
import http.server
import json
import os
import socket
import struct
import threading
import time
import urllib.request
from pathlib import Path

RATE = 24000
REF_TEXT = (
    "This is the echo calibration reference. The quick brown fox jumps over "
    "the lazy dog while seven wizards brew fresh coffee at midnight. "
    "Calibration clip ending now."
)


def _wav(pcm: bytes) -> bytes:
    n = len(pcm)
    return (b"RIFF" + struct.pack("<I", 36 + n) + b"WAVEfmt "
            + struct.pack("<IHHIIHH", 16, 1, 1, RATE, RATE * 2, 2, 16)
            + b"data" + struct.pack("<I", n) + pcm)


def synth_ref(path: Path, voice: str) -> None:
    key = os.environ["OPENAI_API_KEY"]
    req = urllib.request.Request(
        "https://api.openai.com/v1/audio/speech",
        data=json.dumps({
            "model": "gpt-4o-mini-tts", "voice": voice,
            "input": REF_TEXT, "response_format": "pcm",
        }).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        pcm = r.read()
    path.write_bytes(_wav(pcm))
    print(f"[m2] synthesized {path} ({len(pcm) / 2 / RATE:.1f}s)")


def lan_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))  # no traffic sent; just picks the outbound iface
    ip = s.getsockname()[0]
    s.close()
    return ip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", required=True, help="puck media_player entity_id")
    ap.add_argument("--ha-url", default="http://127.0.0.1:8123")
    ap.add_argument("--voice", default="alloy")
    ap.add_argument("--http-port", type=int, default=8099)
    ap.add_argument("--out-dir", default="m2_captures")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(exist_ok=True)
    ref = out / "ref.wav"
    if not ref.exists():
        synth_ref(ref, args.voice)
    duration = (ref.stat().st_size - 44) / 2 / RATE

    handler = functools.partial(
        http.server.SimpleHTTPRequestHandler, directory=str(out)
    )
    httpd = http.server.ThreadingHTTPServer(("0.0.0.0", args.http_port), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    media_url = f"http://{lan_ip()}:{args.http_port}/ref.wav"
    print(f"[m2] serving {media_url}")

    req = urllib.request.Request(
        f"{args.ha_url}/api/services/media_player/play_media",
        data=json.dumps({
            "entity_id": args.entity,
            "media_content_id": media_url,
            "media_content_type": "music",
        }).encode(),
        headers={
            "Authorization": f"Bearer {os.environ['HA_TOKEN']}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        r.read()
    print(f"[m2] playing on {args.entity} ({duration:.1f}s clip)...")
    time.sleep(duration + 3)
    httpd.shutdown()
    print("[m2] done")


if __name__ == "__main__":
    main()
