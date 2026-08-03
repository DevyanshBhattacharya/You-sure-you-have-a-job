"""Access control for a single-user deployment.

This app holds the full text of one person's mailbox and will answer questions
about it. On a laptop, bound to loopback, that is fine. The moment it is reachable
from anywhere else, every endpoint is an unauthenticated read of that mailbox:
`GET /api/emails/1` returns a message body, and `POST /api/chat` will summarise
the lot on request.

So there is one shared secret, `APP_AUTH_TOKEN`. Set it and every `/api` and
`/ws` route requires it; leave it unset and the server runs open but says so
loudly at startup. It is deliberately not a user system — this is one person's
mailbox, and pretending otherwise would imply an isolation the data model does
not have (see the note in README about going multi-user).
"""

from __future__ import annotations

import hmac
import logging
from urllib.parse import urlsplit

from fastapi import Request

from app.config import get_settings

log = logging.getLogger(__name__)

# Reachable without a token. Deliberately tiny: `/` carries no data, and the
# OpenAPI routes are needed to render the docs page at all. `/api/health` is
# NOT here — it reports the linked Gmail address.
PUBLIC_PATHS = frozenset({"/", "/openapi.json", "/docs", "/docs/oauth2-redirect", "/redoc"})

PROTECTED_PREFIXES = ("/api", "/ws")


def is_enabled() -> bool:
    return bool(get_settings().app_auth_token)


def requires_token(path: str) -> bool:
    if path in PUBLIC_PATHS:
        return False
    return path.startswith(PROTECTED_PREFIXES)


def token_from_request(request: Request) -> str | None:
    """Read the token from a header, or the query string as a fallback.

    The query string exists for WebSockets: browsers give no way to set headers
    on a WebSocket handshake. It is accepted on HTTP too so that `curl` and the
    dashboard behave the same way.
    """
    header = request.headers.get("authorization", "")
    if header[:7].lower() == "bearer ":
        return header[7:].strip() or None
    direct = request.headers.get("x-api-token")
    if direct:
        return direct.strip() or None
    return request.query_params.get("token") or None


def token_is_valid(provided: str | None) -> bool:
    expected = get_settings().app_auth_token
    if not expected:
        return True
    if not provided:
        return False
    # Constant-time: a plain `==` leaks the shared secret one character at a
    # time to anyone who can measure the response.
    return hmac.compare_digest(provided, expected)


def _host_of(origin: str) -> str:
    parts = urlsplit(origin)
    return f"{parts.scheme}://{parts.netloc}" if parts.scheme else origin


def origin_is_allowed(origin: str | None) -> bool:
    """Check a WebSocket handshake's Origin against the CORS allow-list.

    WebSockets are exempt from the same-origin policy and CORS does not apply to
    them, so without this any page the user visits can open a socket to a
    localhost server and receive their notification feed — subjects, companies,
    interview times. The browser attaches `Origin` automatically and a page
    cannot forge it, which is what makes the check worth anything.

    A missing Origin means a non-browser client (curl, a test, a script). Those
    are allowed through here and stopped by the token instead.
    """
    if origin is None:
        return True
    allowed = {_host_of(o) for o in get_settings().cors_origin_list}
    return "*" in allowed or _host_of(origin) in allowed


def warn_if_unprotected() -> None:
    """Say plainly, at startup, what is and isn't guarded."""
    settings = get_settings()

    if "*" in settings.cors_origin_list:
        log.error(
            "CORS_ORIGINS is '*'. Combined with credentialed requests this lets any "
            "site read this API from a logged-in browser. Name the exact origins."
        )

    if is_enabled():
        log.info("API access requires APP_AUTH_TOKEN.")
        return

    log.warning(
        "APP_AUTH_TOKEN is not set: every /api and /ws route is open to anyone who "
        "can reach this port, including the full text of your mail. That is fine "
        "bound to 127.0.0.1 and unsafe anywhere else. Set APP_AUTH_TOKEN before "
        "exposing this host."
    )
