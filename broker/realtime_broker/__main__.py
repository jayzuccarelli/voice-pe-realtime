"""Entry point: `python -m realtime_broker`."""

from __future__ import annotations

import asyncio
import logging

from .config import Config
from .server import run


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("websockets").setLevel(logging.WARNING)
    try:
        asyncio.run(run(Config.from_env()))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
