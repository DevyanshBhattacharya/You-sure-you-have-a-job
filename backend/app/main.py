"""FastAPI application entrypoint."""

from __future__ import annotations

import asyncio
import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import backfill, events, security
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


def _configure_logging() -> None:
    """Set up logging, and make the console safe for real email content.

    A Windows console encodes as cp1252 by default, so logging a subject line
    containing an emoji — routine in recruiter mail — raises UnicodeEncodeError
    inside the handler and prints a logging traceback instead of the record.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):  # pragma: no cover - detached stream
                pass

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )


_configure_logging()
log = logging.getLogger("app")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    security.warn_if_unprotected()
    events.bind_loop(asyncio.get_running_loop())
    pipeline.start_workers()

    # Only start the watcher once Gmail is genuinely reachable. access_status()
    # never opens a consent window, which must never happen inside a server.
    # Pick up anything a previous run stored but never classified.
    backlog = pipeline.count_unprocessed()
    if backlog:
        if settings.process_backlog_on_start:
            pipeline.enqueue_unprocessed(settings.backlog_batch_limit)
        else:
            log.warning(
                "%d email(s) are stored but unclassified. PROCESS_BACKLOG_ON_START is off; "
                "run scripts/replay.py --only-unprocessed to handle them.",
                backlog,
            )

    access = gmail_auth.access_status()
    if access.usable:
        log.info("Gmail ready as %s", access.address)
        # Import before starting the watcher, not after. The watcher polls
        # immediately and skips its cycle while an import owns the history
        # cursor — but only if the import is already marked running by then.
        started = False
        if settings.resume_backfill_on_start:
            # A dropped connection used to mean clicking "Import mail" again.
            started = backfill.resume_if_interrupted(on_email=pipeline.submit_email_id)
        if not started and settings.auto_backfill_on_start:
            backfill.start_if_never_run(on_email=pipeline.submit_email_id)
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

# Swagger fetches /openapi.json from the browser without an Authorization
# header, so the docs cannot be put behind the token — they either stay open or
# they go. Tie them to the deployment posture instead: a configured token means
# this is reachable from somewhere, and an unauthenticated map of every endpoint
# is not something to publish alongside it.
_docs_enabled = not settings.app_auth_token

app = FastAPI(
    title="Job Mail Agent",
    version="0.1.0",
    description="Tracks job-related email, keeps a knowledge base, and answers questions about it.",
    lifespan=lifespan,
    docs_url="/docs" if _docs_enabled else None,
    redoc_url="/redoc" if _docs_enabled else None,
    openapi_url="/openapi.json" if _docs_enabled else None,
)

@app.middleware("http")
async def require_token(request: Request, call_next):
    """Guard every /api route with the shared secret, when one is configured.

    A middleware rather than a per-route dependency, so a router added later
    cannot quietly ship unauthenticated — the failure mode of the dependency
    approach is an endpoint nobody remembered to annotate.
    """
    # Preflight carries no credentials by design; blocking it breaks CORS
    # without protecting anything, since the real request is still checked.
    if request.method == "OPTIONS" or not security.requires_token(request.url.path):
        return await call_next(request)

    if not security.token_is_valid(security.token_from_request(request)):
        return JSONResponse(
            status_code=401,
            content={"detail": "Missing or invalid API token."},
            headers={"WWW-Authenticate": "Bearer"},
        )

    return await call_next(request)


# Registered AFTER the auth middleware on purpose. Starlette makes the
# last-added middleware the outermost, so this ordering puts CORS on the
# outside — which is the only way a 401 comes back carrying
# `Access-Control-Allow-Origin`. With the order reversed, a cross-origin
# dashboard sees an opaque network error instead of a 401 and never learns to
# ask for a token.
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
