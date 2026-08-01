"""Distinguishing "no credentials" from "credentials fine, API unreachable".

Collapsing these into one boolean sends people back through a consent flow
that cannot fix an API-not-enabled error — which is exactly what happened.
"""

from __future__ import annotations

import httplib2
import pytest
from googleapiclient.errors import HttpError

from app.gmail import auth
from app.gmail.auth import AccessStatus, GmailAuthError, _extract_console_url

ENABLE_URL = (
    "https://console.developers.google.com/apis/api/gmail.googleapis.com"
    "/overview?project=709508293093"
)

API_DISABLED_BODY = (
    b'{"error": {"message": "Gmail API has not been used in project 709508293093 '
    b"before or it is disabled. Enable it by visiting "
    b'https://console.developers.google.com/apis/api/gmail.googleapis.com/overview'
    b'?project=709508293093 then retry."}}'
)


def http_error(status: int, body: bytes) -> HttpError:
    return HttpError(httplib2.Response({"status": status}), body)


class TestExtractConsoleUrl:
    def test_keeps_the_whole_url_including_dots(self):
        """A regex that stops at the first '.' truncates gmail.googleapis.com."""
        message = f"Enable it by visiting {ENABLE_URL} then retry."
        assert _extract_console_url(message) == ENABLE_URL

    def test_trims_trailing_sentence_punctuation(self):
        assert _extract_console_url(f"see {ENABLE_URL}.") == ENABLE_URL
        assert _extract_console_url(f"(see {ENABLE_URL})") == ENABLE_URL

    def test_returns_none_when_absent(self):
        assert _extract_console_url("no link in this message") is None


class TestAccessStatus:
    def test_missing_credentials_points_at_the_auth_command(self, monkeypatch):
        def boom(**_kwargs):
            raise GmailAuthError("No valid Gmail credentials")

        monkeypatch.setattr(auth, "load_credentials", boom)
        status = auth.access_status()

        assert status.authorised is False
        assert status.usable is False
        assert "app.gmail.auth" in status.hint

    def test_disabled_api_is_not_reported_as_unauthorised(self, monkeypatch):
        """The credentials are fine here. Re-authorising would fix nothing."""
        monkeypatch.setattr(auth, "load_credentials", lambda **_k: object())
        monkeypatch.setattr(
            "app.gmail.client.get_profile",
            lambda: (_ for _ in ()).throw(http_error(403, API_DISABLED_BODY)),
        )

        status = auth.access_status()

        assert status.authorised is True
        assert status.usable is False
        assert "not enabled" in status.error
        assert ENABLE_URL in status.hint
        # Crucially, it must NOT send the user back through consent.
        assert "app.gmail.auth" not in status.hint

    def test_revoked_token_does_point_at_reauthorisation(self, monkeypatch):
        monkeypatch.setattr(auth, "load_credentials", lambda **_k: object())
        monkeypatch.setattr(
            "app.gmail.client.get_profile",
            lambda: (_ for _ in ()).throw(
                http_error(401, b'{"error": {"message": "Invalid Credentials"}}')
            ),
        )

        status = auth.access_status()

        assert status.authorised is True
        assert status.usable is False
        assert "app.gmail.auth" in status.hint

    def test_network_failure_is_reported_as_such(self, monkeypatch):
        monkeypatch.setattr(auth, "load_credentials", lambda **_k: object())
        monkeypatch.setattr(
            "app.gmail.client.get_profile",
            lambda: (_ for _ in ()).throw(OSError("getaddrinfo failed")),
        )

        status = auth.access_status()

        assert status.authorised is True
        assert status.usable is False
        assert "Could not reach Gmail" in status.error

    def test_working_access_reports_the_address(self, monkeypatch):
        monkeypatch.setattr(auth, "load_credentials", lambda **_k: object())
        monkeypatch.setattr(
            "app.gmail.client.get_profile",
            lambda: {"emailAddress": "someone@example.com", "messagesTotal": 12},
        )

        status = auth.access_status()

        assert status == AccessStatus(True, True, address="someone@example.com")


class TestHealthAndBackfillReporting:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient

        from app.main import app

        with TestClient(app) as c:
            yield c

    def test_health_separates_authorised_from_usable(self, client, monkeypatch):
        monkeypatch.setattr(
            auth,
            "access_status",
            lambda: AccessStatus(
                True,
                False,
                error="The Gmail API is not enabled...",
                hint=f"Enable it at {ENABLE_URL}",
            ),
        )

        body = client.get("/api/health").json()

        assert body["gmail_authorised"] is True
        assert body["gmail_usable"] is False
        assert "not enabled" in body["gmail_error"]

    def test_backfill_refusal_explains_the_real_cause(self, client, monkeypatch):
        monkeypatch.setattr(
            auth,
            "access_status",
            lambda: AccessStatus(
                True,
                False,
                error="The Gmail API is not enabled on this Google Cloud project.",
                hint=f"Enable it at {ENABLE_URL} and retry in a minute.",
            ),
        )

        response = client.post("/api/sync/backfill", json={"days": 30})

        assert response.status_code == 409
        detail = response.json()["detail"]
        assert "not enabled" in detail
        assert ENABLE_URL in detail
