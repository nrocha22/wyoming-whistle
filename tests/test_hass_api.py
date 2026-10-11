import asyncio

from aiohttp import web

from wyoming_whistle.hass_api import HassKeywords, dedupe, fetch_keywords


def test_dedupe_is_case_insensitive_and_capped():
    assert dedupe(["Sala", "sala", None, " ", "Comedor", "Cocina"], limit=2) == ["Sala", "Comedor"]


async def fake_hass(token="good"):
    """A minimal Home Assistant websocket: auth plus the five commands we call."""
    results = {
        "homeassistant/expose_entity/list": {
            "exposed_entities": {
                "light.living": {"conversation": True},
                "climate.ac": {"conversation": True},
                "light.hidden": {"conversation": False},
            }
        },
        "config/entity_registry/get_entries": {
            "light.living": {"entity_id": "light.living", "name": None, "aliases": [None, "living room light"]},
            "climate.ac": {"entity_id": "climate.ac", "name": "Bedroom AC", "aliases": []},
        },
        "get_states": [{"entity_id": "light.living", "attributes": {"friendly_name": "Living Light"}}],
        "config/area_registry/list": [{"name": "Living Room", "aliases": ["lounge"]}],
        "config/floor_registry/list": [{"name": "Ground Floor", "aliases": []}],
    }

    async def handler(request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.send_json({"type": "auth_required"})
        auth = await ws.receive_json()
        if auth.get("access_token") != token:
            await ws.send_json({"type": "auth_invalid"})
            await ws.close()
            return ws
        await ws.send_json({"type": "auth_ok"})
        async for msg in ws:
            data = msg.json()
            if data["type"] == "config/entity_registry/get_entries":
                assert "light.hidden" not in data["entity_ids"]  # only exposed entities are asked for
            await ws.send_json({"id": data["id"], "type": "result", "success": True, "result": results[data["type"]]})
        return ws

    app = web.Application()
    app.router.add_get("/api/websocket", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]  # pylint: disable=protected-access
    return runner, f"http://127.0.0.1:{port}/api/websocket"


async def test_fetch_keywords_only_exposed_entities():
    runner, url = await fake_hass()
    try:
        words = await fetch_keywords(url, "good")
    finally:
        await runner.cleanup()
    assert words == ["Living Room", "lounge", "Ground Floor", "Living Light", "living room light", "Bedroom AC"]
    assert "Secret lamp" not in words and "nope" not in words


async def test_refresh_keeps_last_list_when_hass_fails():
    runner, url = await fake_hass()
    keywords = HassKeywords(url, "good")
    await keywords.refresh()
    assert "Bedroom AC" in keywords.keywords
    await runner.cleanup()  # Home Assistant goes away
    await keywords.refresh()  # must not raise
    assert "Bedroom AC" in keywords.keywords


async def test_bad_token_keeps_empty_list():
    runner, url = await fake_hass(token="other")
    try:
        keywords = HassKeywords(url, "good")
        await asyncio.wait_for(keywords.refresh(), timeout=5)
    finally:
        await runner.cleanup()
    assert keywords.keywords == []
