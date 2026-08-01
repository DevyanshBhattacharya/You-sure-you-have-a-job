"""FastAPI application entrypoint."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import events
from app.agent import pipeline
from app.api import (
    routes_applications,
    routes_chat,
    routes_emails,
    routes_notifications,
    routes_sync,
    routes_ws,
)
from app.config import get_settings
from app.db import init_db
from app.gmail import auth as gmail_auth
from app.gmail import watcher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)
log = logging.getLogger("app")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    events.bind_loop(asyncio.get_running_loop())
    pipeline.start_workers()

    # Only start the watcher once Gmail is genuinely reachable. access_status()
    # never opens a consent window, which must never happen inside a server.
    access = gmail_auth.access_status()
    if access.usable:
        log.info("Gmail ready as %s", access.address)
        watcher.start()
    else:
        log.warning("Gmail watcher not started: %s", access.error)
        if access.hint:
            log.warning("  %s", access.hint)

    try:
        yield
    finally:
        watcher.stop()
        await pipeline.stop_workers()


settings = get_settings()

app = FastAPI(
    title="Job Mail Agent",
    version="0.1.0",
    description="Tracks job-related email, keeps a knowledge base, and answers questions about it.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(routes_sync.router)
app.include_router(routes_emails.router)
app.include_router(routes_applications.router)
app.include_router(routes_notifications.router)
app.include_router(routes_chat.router)
app.include_router(routes_ws.router)


@app.get("/", include_in_schema=False)
def root() -> dict:
    return {"service": "job-mail-agent", "docs": "/docs", "health": "/api/health"}
