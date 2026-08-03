"""Gmail OAuth: installed-app flow with on-disk token persistence.

Read-only scope. The agent never needs to modify the mailbox.
"""

from __future__ import annotations

import logging
import os
import re
import threading
from dataclasses import dataclass

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from app.config import get_settings

log = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

_lock = threading.Lock()

# One service per thread, never one shared between them. See get_service().
_local = threading.local()
# Bumped by reset_service(); a thread rebuilds when its cached copy is stale.
_generation = 0


class GmailAuthError(RuntimeError):
    pass


def load_credentials(*, allow_interactive: bool = True) -> Credentials:
    """Load cached credentials, refreshing or running consent as needed."""
    settings = get_settings()
    token_path = settings.token_path
    creds: Credentials | None = None

    if token_path.exists():
        try:
            creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
        except Exception as exc:  # corrupt/rotated token file
            log.warning("Ignoring unreadable token file %s: %s", token_path, exc)
            creds = None

    if creds and creds.valid:
        return creds

    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            _persist(creds)
            return creds
        except Exception as exc:
            log.warning("Token refresh failed, falling back to consent: %s", exc)
            creds = None

    if not allow_interactive:
        raise GmailAuthError(
            "No valid Gmail credentials and interactive consent is disabled. "
            "Run `python -m app.gmail.auth` once to authorise."
        )

    cred_file = settings.credentials_path
    if not cred_file.exists():
        raise GmailAuthError(
            f"OAuth client secret not found at {cred_file}.\n"
            "Create a Desktop-app OAuth client in Google Cloud Console "
            "(APIs & Services > Credentials), download the JSON, and save it there."
        )

    flow = InstalledAppFlow.from_client_secrets_file(str(cred_file), SCOPES)
    creds = flow.run_local_server(port=settings.oauth_redirect_port, prompt="consent")
    _persist(creds)
    return creds


def _persist(creds: Credentials) -> None:
    """Write the token file atomically.

    Each thread holds its own credentials, so two can refresh at once. Writing
    in place would let one truncate the file while the other reads it, leaving
    a corrupt token and an unexplained re-consent. Write, then rename.
    """
    token_path = get_settings().token_path
    token_path.parent.mkdir(parents=True, exist_ok=True)

    tmp = token_path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        tmp.write_text(creds.to_json(), encoding="utf-8")
        os.replace(tmp, token_path)
    finally:
        tmp.unlink(missing_ok=True)
    log.info("Saved Gmail token to %s", token_path)


@dataclass
class AccessStatus:
    """Whether the agent can actually read mail, and why not if it can't.

    Holding valid credentials and being able to call the API are different
    things — the Gmail API has to be enabled on the Cloud project separately.
    Collapsing both into one boolean sends people back through a consent flow
    that cannot fix an API-not-enabled error.
    """

    authorised: bool
    usable: bool
    address: str | None = None
    error: str | None = None
    hint: str | None = None


_CONSOLE_URL = re.compile(r"https://console\.(?:developers|cloud)\.google\.com/[^\s\"']+")


def _extract_console_url(message: str) -> str | None:
    """Pull the 'enable it here' link out of a Google API error.

    Consumes greedily and trims trailing sentence punctuation afterwards —
    stopping the match at the first `.` would cut the URL in half, since every
    one of these contains `gmail.googleapis.com`.
    """
    match = _CONSOLE_URL.search(message)
    if not match:
        return None
    return match.group(0).rstrip(".,);\\")

AUTH_HINT = "Run `python -m app.gmail.auth` from the backend directory."


def access_status() -> AccessStatus:
    """Probe real access. Never triggers an interactive consent flow."""
    try:
        load_credentials(allow_interactive=False)
    except GmailAuthError as exc:
        return AccessStatus(False, False, error=str(exc), hint=AUTH_HINT)
    except Exception as exc:  # noqa: BLE001 - a corrupt token lands here too
        return AccessStatus(False, False, error=str(exc), hint=AUTH_HINT)

    # Imported lazily: client imports this module, so a top-level import here
    # would be circular.
    from googleapiclient.errors import HttpError

    from app.gmail import client as gmail_client

    try:
        profile = gmail_client.get_profile()
    except HttpError as exc:
        message = str(exc)
        status = getattr(exc.resp, "status", None)

        if status == 403 and "has not been used in project" in message:
            url = _extract_console_url(message)
            return AccessStatus(
                True,
                False,
                error="The Gmail API is not enabled on this Google Cloud project.",
                hint=(
                    f"Enable it at {url} and retry in a minute."
                    if url
                    else "Enable the Gmail API in the Google Cloud console, then retry."
                ),
            )

        if status in (401, 403):
            return AccessStatus(
                True,
                False,
                error=f"Gmail rejected the request ({status}). The token may be revoked or "
                "missing the read-only scope.",
                hint=AUTH_HINT,
            )

        return AccessStatus(True, False, error=f"Gmail API error: {message[:300]}")
    except Exception as exc:  # noqa: BLE001 - network failures, DNS, proxies
        return AccessStatus(True, False, error=f"Could not reach Gmail: {exc}")

    return AccessStatus(True, True, address=profile.get("emailAddress"))


def get_service(*, allow_interactive: bool = True):
    """Return this thread's Gmail API service client.

    Deliberately per-thread, not a shared singleton. `build()` creates a single
    `httplib2.Http` and every request from that service reuses it — and httplib2
    is not thread-safe. Two threads calling Gmail at once (the watcher polling
    while a backfill imports) interleave writes on the same TLS socket, then
    each reads back bytes meant for the other. OpenSSL sees a record header that
    isn't one and reports:

        [SSL: WRONG_VERSION_NUMBER] wrong version number

    which looks like a proxy or TLS problem and is neither. Retrying does not
    help, because both threads simply collide again.

    A thread keeps its own connection pool across calls, so an import still
    reuses one connection for thousands of messages.
    """
    cached = getattr(_local, "service", None)
    if cached is not None and getattr(_local, "generation", None) == _generation:
        return cached

    # Only credential loading is shared; the service itself must not be.
    with _lock:
        creds = load_credentials(allow_interactive=allow_interactive)
        generation = _generation

    service = build("gmail", "v1", credentials=creds, cache_discovery=False)
    _local.service = service
    _local.generation = generation
    return service


def reset_service() -> None:
    """Invalidate every thread's cached service."""
    global _generation
    with _lock:
        _generation += 1
    _local.service = None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    get_service()  # runs consent if needed

    status = access_status()
    if status.usable:
        print(f"Authorised as {status.address}")
    else:
        print("Authorisation stored, but Gmail is not reachable yet.")
        print(f"  {status.error}")
        if status.hint:
            print(f"  {status.hint}")
        raise SystemExit(1)
