"""In-process event plumbing.

Two distinct shapes, deliberately kept separate:

* **Work queue** — each item must be handled exactly once (the agent pipeline).
* **Event bus** — each item is broadcast to every subscriber (dashboard sockets).

Both are safe to feed from the background threads that run the watcher and the
backfill, via the `*_threadsafe` helpers.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

log = logging.getLogger(__name__)

# Topics
EMAIL_PROCESSED = "email.processed"
NOTIFICATION_CREATED = "notification.created"
APPLICATION_UPDATED = "application.updated"
SYNC_STATUS = "sync.status"

_loop: asyncio.AbstractEventLoop | None = None


def bind_loop(loop: asyncio.AbstractEventLoop) -> None:
    """Record the loop that background threads should schedule onto.

    Also rebuilds the work queue: `asyncio.Queue` binds itself to the loop that
    first uses it and refuses to be driven from another, so a second app
    instance in the same process (tests, an embedded runner) needs a fresh one.
    """
    global _loop
    _loop = loop
    work_queue.reset()


def get_loop() -> asyncio.AbstractEventLoop | None:
    return _loop


@dataclass(slots=True)
class EmailWork:
    """One unit of pipeline work.

    Exactly one of `gmail_id` (not yet fetched) or `email_id` (already stored)
    is required.
    """

    gmail_id: str | None = None
    email_id: int | None = None
    reclassify: bool = False


class WorkQueue:
    """Single-consumer queue for pipeline work.

    The inner queue is created lazily so importing this module never touches
    an event loop, and can be replaced via `reset()` when a new loop takes over.
    """

    def __init__(self, maxsize: int = 10_000) -> None:
        self._maxsize = maxsize
        self._queue: asyncio.Queue[EmailWork] | None = None

    def reset(self) -> None:
        self._queue = asyncio.Queue(maxsize=self._maxsize)

    def _q(self) -> asyncio.Queue[EmailWork]:
        if self._queue is None:
            self._queue = asyncio.Queue(maxsize=self._maxsize)
        return self._queue

    def submit_threadsafe(self, item: EmailWork) -> None:
        loop = get_loop()
        if loop is None or loop.is_closed():
            log.warning("No event loop bound; dropping work item %r", item)
            return
        loop.call_soon_threadsafe(self._offer, item)

    def _offer(self, item: EmailWork) -> None:
        try:
            self._q().put_nowait(item)
        except asyncio.QueueFull:
            # Better to drop one item than to wedge the watcher. The next
            # backfill picks it up again.
            log.error("Pipeline queue full; dropping %r", item)

    async def submit(self, item: EmailWork) -> None:
        await self._q().put(item)

    async def get(self) -> EmailWork:
        return await self._q().get()

    def task_done(self) -> None:
        self._q().task_done()

    def qsize(self) -> int:
        return self._q().qsize()


class EventBus:
    """Fan-out pub/sub. Slow subscribers are dropped, never blocked on."""

    def __init__(self, per_subscriber_buffer: int = 256) -> None:
        self._buffer = per_subscriber_buffer
        self._subscribers: dict[str, set[asyncio.Queue]] = {}

    @asynccontextmanager
    async def subscribe(self, *topics: str) -> AsyncIterator[asyncio.Queue]:
        queue: asyncio.Queue = asyncio.Queue(maxsize=self._buffer)
        for topic in topics:
            self._subscribers.setdefault(topic, set()).add(queue)
        try:
            yield queue
        finally:
            for topic in topics:
                subs = self._subscribers.get(topic)
                if subs:
                    subs.discard(queue)
                    if not subs:
                        self._subscribers.pop(topic, None)

    def publish(self, topic: str, payload: dict) -> None:
        message = {"topic": topic, "data": payload}
        for queue in list(self._subscribers.get(topic, ())):
            try:
                queue.put_nowait(message)
            except asyncio.QueueFull:
                log.warning("Subscriber buffer full on %s; dropping message", topic)

    def publish_threadsafe(self, topic: str, payload: dict) -> None:
        loop = get_loop()
        if loop is None or loop.is_closed():
            return
        loop.call_soon_threadsafe(self.publish, topic, payload)

    def subscriber_count(self) -> int:
        return len({q for subs in self._subscribers.values() for q in subs})


work_queue = WorkQueue()
bus = EventBus()
