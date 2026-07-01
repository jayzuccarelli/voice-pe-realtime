"""Entry point: `python -m realtime_broker`."""

from __future__ import annotations

import asyncio
import logging

from dotenv import load_dotenv

from .config import Config
from .server import run


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("websockets").setLevel(logging.WARNING)
    # Load a local .env if present (no-op in Docker, which uses --env-file).
    load_dotenv()
    try:
        asyncio.run(run(Config.from_env()))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
