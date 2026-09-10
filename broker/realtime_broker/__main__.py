"""Entry point: `python -m realtime_broker`."""

from __future__ import annotations

import asyncio
import logging

from dotenv import load_dotenv

from .config import Config


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("websockets").setLevel(logging.WARNING)
    # Load a local .env if present (no-op in Docker, which uses --env-file).
    load_dotenv()
    config = Config.from_env()
    if config.engine == "live":
        from .live_server import run_live as serve
    elif config.engine == "realtime":
        from .server import run as serve
    else:
        raise RuntimeError(f"ENGINE must be 'realtime' or 'live', got {config.engine!r}")
    logging.getLogger(__name__).info("Starting %s engine", config.engine)
    try:
        asyncio.run(serve(config))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
