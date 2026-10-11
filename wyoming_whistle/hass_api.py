"""Keyword biasing from Home Assistant: the names people actually say.

Collects, over the Home Assistant websocket API:
  * names and aliases of entities exposed to Assist (conversation),
  * names and aliases of areas and floors.

These are passed to Whistle as keywords, so a name like "Marta" stops coming back as
"Martha". Transcription never waits for Home Assistant: the list refreshes in
the background and the last good list is kept when Home Assistant is down.

This is the only network connection the server makes, and only when configured.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
from typing import Iterable, List, Optional

import aiohttp

_LOGGER = logging.getLogger(__name__)


def dedupe(words: Iterable[Optional[str]], limit: int) -> List[str]:
    """Case-insensitive de-duplication, first spelling wins, capped at ``limit``."""
    seen, out = set(), []
    for word in words:
        word = (word or "").strip()
        key = word.casefold()
        if word and key not in seen:
            seen.add(key)
            out.append(word)
            if len(out) >= limit:
                break
    return out


async def fetch_keywords(url: str, token: str, timeout: float = 10.0, limit: int = 200) -> List[str]:
    """One round trip to Home Assistant. Raises on any failure."""
    ids = itertools.count(1)
    client_timeout = aiohttp.ClientTimeout(total=timeout)
    async with aiohttp.ClientSession(timeout=client_timeout) as session:
        async with session.ws_connect(url) as ws:
            await ws.receive_json()  # auth_required
            await ws.send_json({"type": "auth", "access_token": token})
            auth = await ws.receive_json()
            if auth.get("type") != "auth_ok":
                raise PermissionError("Home Assistant rejected the token")

            async def call(command: str, **params):
                msg_id = next(ids)
                await ws.send_json({"id": msg_id, "type": command, **params})
                while True:
                    reply = await ws.receive_json()
                    if reply.get("id") == msg_id:
                        if not reply.get("success"):
                            raise RuntimeError(f"{command} failed: {reply.get('error')}")
                        return reply["result"]

            exposed = (await call("homeassistant/expose_entity/list"))["exposed_entities"]
            wanted = [eid for eid, assistants in exposed.items() if assistants.get("conversation")]
            # get_entries returns full entries; config/entity_registry/list is the
            # display form and leaves out aliases, the words people actually say.
            entries = await call("config/entity_registry/get_entries", entity_ids=wanted) if wanted else {}
            registry = {eid: entry for eid, entry in entries.items() if entry}
            states = {s["entity_id"]: s for s in await call("get_states")}
            areas = await call("config/area_registry/list")
            floors = await call("config/floor_registry/list")

    words: List[Optional[str]] = []
    for area in areas:
        words.append(area.get("name"))
        words.extend(area.get("aliases") or [])
    for floor in floors:
        words.append(floor.get("name"))
        words.extend(floor.get("aliases") or [])
    for entity_id, assistants in exposed.items():
        if not assistants.get("conversation"):
            continue
        entry = registry.get(entity_id, {})
        state = states.get(entity_id, {})
        words.append(entry.get("name") or state.get("attributes", {}).get("friendly_name"))
        words.extend(entry.get("aliases") or [])  # may contain None (= the entity's own name)
    return dedupe(words, limit)


class HassKeywords:
    """Keeps a fresh keyword list from Home Assistant in the background."""

    def __init__(self, url: str, token: str, refresh_seconds: float = 300.0, limit: int = 200) -> None:
        self.url = url
        self._token = token
        self.refresh_seconds = refresh_seconds
        self.limit = limit
        self.keywords: List[str] = []
        self._failing = False

    async def refresh(self) -> None:
        try:
            self.keywords = await fetch_keywords(self.url, self._token, limit=self.limit)
        except Exception as err:  # pylint: disable=broad-except
            if not self._failing:  # log once per outage, not every refresh
                _LOGGER.warning("Could not read names from Home Assistant (%s); keeping %s keywords", err, len(self.keywords))
            self._failing = True
            return
        if self._failing:
            _LOGGER.info("Home Assistant reachable again")
        self._failing = False
        _LOGGER.info("Loaded %s keywords from Home Assistant", len(self.keywords))
        _LOGGER.debug("Home Assistant keywords: %s", self.keywords)

    async def run(self) -> None:
        while True:
            await self.refresh()
            await asyncio.sleep(self.refresh_seconds)


def main() -> None:
    """Print the keyword list: python -m wyoming_whistle.hass_api (reads WHISTLE_HASS_URL/TOKEN)."""
    import os  # pylint: disable=import-outside-toplevel

    words = asyncio.run(fetch_keywords(os.environ["WHISTLE_HASS_URL"], os.environ["WHISTLE_HASS_TOKEN"]))
    print("\n".join(words))


if __name__ == "__main__":
    main()
