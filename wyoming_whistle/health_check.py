"""Docker HEALTHCHECK: Describe round-trip. Exit 0 if the server answers as whistle."""

from __future__ import annotations

import argparse
import asyncio
import sys

from wyoming.client import AsyncClient
from wyoming.info import Describe, Info


async def check(uri: str, timeout: float) -> bool:
    async def _round_trip() -> bool:
        async with AsyncClient.from_uri(uri) as client:
            await client.write_event(Describe().event())
            while True:
                event = await client.read_event()
                if event is None:
                    return False
                if Info.is_type(event.type):
                    info = Info.from_event(event)
                    return any(program.name == "whistle" for program in info.asr)

    try:
        return await asyncio.wait_for(_round_trip(), timeout=timeout)
    except (OSError, asyncio.TimeoutError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser(prog="wyoming_whistle.health_check")
    parser.add_argument("--uri", default="tcp://127.0.0.1:10300")
    parser.add_argument("--timeout", type=float, default=5.0)
    args = parser.parse_args()
    sys.exit(0 if asyncio.run(check(args.uri, args.timeout)) else 1)


if __name__ == "__main__":
    main()
