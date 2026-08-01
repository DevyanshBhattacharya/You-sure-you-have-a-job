"""WebSocket feed for live dashboard updates.

Subscribes to the in-process event bus, so the agent pushes to the browser the
moment it finishes with an email — no polling anywhere in the chain.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.events import (
    APPLICATION_UPDATED,
    EMAIL_PROCESSED,
    NOTIFICATION_CREATED,
    SYNC_STATUS,
    bus,
)

log = logging.getLogger(__name__)
router = APIRouter(tags=["realtime"])

TOPICS = (NOTIFICATION_CREATED, APPLICATION_UPDATED, EMAIL_PROCESSED, SYNC_STATUS)

# Without periodic traffic, idle proxies drop the socket after ~60s.
HEARTBEAT_SECONDS = 25


@router.websocket("/ws/notifications")
async def notifications_socket(websocket: WebSocket) -> None:
    await websocket.accept()

    async with bus.subscribe(*TOPICS) as queue:
        await websocket.send_json({"topic": "connected", "data": {"topics": list(TOPICS)}})

        # Drain the client side so a disconnect is noticed promptly even while
        # the server has nothing to send.
        reader = asyncio.create_task(_drain(websocket))
        try:
            while True:
                try:
                    message = await asyncio.wait_for(queue.get(), timeout=HEARTBEAT_SECONDS)
                except TimeoutError:
                    await websocket.send_json({"topic": "ping", "data": {}})
                    continue

                if reader.done():
                    break
                await websocket.send_json(message)
        except WebSocketDisconnect:
            pass
        except Exception:  # noqa: BLE001
            log.exception("WebSocket send loop failed")
        finally:
            reader.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await reader
            with contextlib.suppress(Exception):
                await websocket.close()


async def _drain(websocket: WebSocket) -> None:
    """Consume inbound frames; returns when the client goes away."""
    try:
        while True:
            await websocket.receive_text()
    except (WebSocketDisconnect, RuntimeError):
        return
