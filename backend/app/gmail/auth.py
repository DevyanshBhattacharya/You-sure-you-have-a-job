"""Gmail OAuth: installed-app flow with on-disk token persistence.

Read-only scope. The agent never needs to modify the mailbox.
"""

from __future__ import annotations

import logging
import threading

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from app.config import get_settings

log = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

_lock = threading.Lock()
_service = None


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
    creds = flow.run_local_server(port=0, prompt="consent")
    _persist(creds)
    return creds


def _persist(creds: Credentials) -> None:
    token_path = get_settings().token_path
    token_path.parent.mkdir(parents=True, exist_ok=True)
    token_path.write_text(creds.to_json(), encoding="utf-8")
    log.info("Saved Gmail token to %s", token_path)


def get_service(*, allow_interactive: bool = True):
    """Return a cached Gmail API service client.

    The googleapiclient service object is not thread-safe for concurrent use of
    the same http object, but each request builds its own; guarding creation is
    enough for our access pattern (one watcher thread + threadpool endpoints).
    """
    global _service
    with _lock:
        if _service is None:
            creds = load_credentials(allow_interactive=allow_interactive)
            _service = build("gmail", "v1", credentials=creds, cache_discovery=False)
        return _service


def reset_service() -> None:
    global _service
    with _lock:
        _service = None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    svc = get_service()
    profile = svc.users().getProfile(userId="me").execute()
    print(f"Authorised as {profile['emailAddress']}")
    print(f"Messages in mailbox: {profile.get('messagesTotal')}")
    print(f"Current historyId: {profile.get('historyId')}")
