"""Access control.

This app serves the full text of one person's mailbox and will summarise it on
request. Every test here is about the difference between "runs on a laptop" and
"is reachable from the internet".
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app import security
from app.config import get_settings
from app.main import app
from tests.fixtures import make_email

TOKEN = "s3cret-token-value"


@pytest.fixture
def protected(monkeypatch):
    """Run the app as it would be deployed: with a token configured."""
    settings = get_settings()
    monkeypatch.setattr(settings, "app_auth_token", TOKEN)
    return settings


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


class TestTokenEnforcement:
    def test_open_by_default_for_local_use(self, client, db):
        """No token configured means no prompt on a laptop."""
        assert client.get("/api/health").status_code == 200

    def test_mail_is_not_readable_without_the_token(self, client, protected, db):
        email = make_email(db, subject="Interview with Acme", body="Very private")
        db.commit()

        response = client.get(f"/api/emails/{email.id}")
        assert response.status_code == 401
        assert "Very private" not in response.text

    @pytest.mark.parametrize(
        "path",
        ["/api/health", "/api/emails", "/api/applications/board", "/api/notifications"],
    )
    def test_every_data_route_is_guarded(self, client, protected, path):
        assert client.get(path).status_code == 401

    def test_health_is_not_public(self, client, protected):
        """It reports the linked Gmail address — that is not a liveness probe."""
        assert "/api/health" not in security.PUBLIC_PATHS

    def test_a_valid_bearer_token_is_accepted(self, client, protected, db):
        response = client.get("/api/health", headers={"Authorization": f"Bearer {TOKEN}"})
        assert response.status_code == 200

    def test_the_query_parameter_form_also_works(self, client, protected, db):
        assert client.get("/api/health", params={"token": TOKEN}).status_code == 200

    def test_a_wrong_token_is_rejected(self, client, protected):
        response = client.get("/api/health", headers={"Authorization": "Bearer wrong"})
        assert response.status_code == 401

    def test_a_prefix_of_the_token_is_rejected(self, client, protected):
        """Guards against a comparison that stops at the first mismatch."""
        response = client.get("/api/health", headers={"Authorization": f"Bearer {TOKEN[:-1]}"})
        assert response.status_code == 401

    def test_the_root_page_stays_public(self, client, protected):
        assert client.get("/").status_code == 200

    def test_writes_are_guarded_too(self, client, protected, db):
        email = make_email(db)
        db.commit()
        response = client.post(
            f"/api/emails/{email.id}/classification", json={"is_job_related": True}
        )
        assert response.status_code == 401


class TestWebSocketIsNotAnOpenDoor:
    """CORS does not apply to WebSockets.

    Without an Origin check, any page the user visits can open a socket to a
    localhost server and read their live notification feed — company names,
    subjects, interview times. The browser sets Origin itself and a page cannot
    forge it, which is the whole basis of the check.
    """

    def test_a_hostile_origin_is_refused(self, client, db):
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect(
                "/ws/notifications", headers={"Origin": "https://evil.example"}
            ) as ws:
                ws.receive_json()

    def test_the_dashboard_origin_is_allowed(self, client, db):
        origin = get_settings().cors_origin_list[0]
        with client.websocket_connect("/ws/notifications", headers={"Origin": origin}) as ws:
            assert ws.receive_json()["topic"] == "connected"

    def test_a_socket_needs_the_token_when_one_is_set(self, client, protected, db):
        origin = get_settings().cors_origin_list[0]
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect(
                "/ws/notifications", headers={"Origin": origin}
            ) as ws:
                ws.receive_json()

    def test_a_socket_with_the_token_connects(self, client, protected, db):
        origin = get_settings().cors_origin_list[0]
        with client.websocket_connect(
            f"/ws/notifications?token={TOKEN}", headers={"Origin": origin}
        ) as ws:
            assert ws.receive_json()["topic"] == "connected"

    def test_non_browser_clients_are_left_to_the_token(self):
        """curl and scripts send no Origin; the token is what stops them."""
        assert security.origin_is_allowed(None) is True


class TestRequestBounds:
    """An unbounded request is a free way to make the server hold megabytes and
    hand them to a model that charges per token."""

    def test_an_oversized_question_is_rejected(self, client, db):
        response = client.post("/api/chat", json={"message": "x" * 50_000})
        assert response.status_code == 422

    def test_an_empty_question_is_rejected(self, client, db):
        assert client.post("/api/chat", json={"message": ""}).status_code == 422

    def test_an_enormous_history_is_rejected(self, client, db):
        history = [{"role": "user", "content": "hi"} for _ in range(500)]
        response = client.post("/api/chat", json={"message": "hello", "history": history})
        assert response.status_code == 422


class TestKnowledgeBaseHygiene:
    """What the Q&A agent can retrieve has to track what is still true."""

    def test_a_corrected_email_stops_being_searchable(self, db, monkeypatch):
        """Indexing only ever ran on the way in. An email later corrected to
        'not job related' kept its chunks, so the agent went on quoting a
        LinkedIn digest as evidence about the job search."""
        from app.agent import classify as classify_mod
        from app.agent import pipeline
        from app.kb import indexer
        from app.models import KBChunk

        email = make_email(db, subject="Torinit is hiring", body="Apply now to this role.")
        indexer.index_email(db, email)
        db.flush()
        assert db.query(KBChunk).filter_by(email_id=email.id).count() > 0

        monkeypatch.setattr(
            classify_mod,
            "classify",
            lambda _s, _e: classify_mod.Verdict(
                is_job_related=False, confidence=0.95, source="prefilter"
            ),
        )
        pipeline.process_email(db, email, reclassify=True)

        assert db.query(KBChunk).filter_by(email_id=email.id).count() == 0

    def test_unsearchable_vectors_are_reported_not_hidden(self, db):
        """Switching embedding model leaves the old vectors a different width,
        so the store skips them — correctly, and completely silently. An answer
        drawn from half the mailbox must not look complete."""
        from app.kb import indexer
        from app.kb.store import to_blob
        from app.models import KBChunk

        email = make_email(db)
        for i in range(5):
            db.add(
                KBChunk(
                    email_id=email.id,
                    chunk_index=i,
                    text=f"chunk {i}",
                    embedding=to_blob([0.1] * 768),
                    dim=768,
                )
            )
        for i in range(2):
            db.add(
                KBChunk(
                    email_id=email.id,
                    chunk_index=100 + i,
                    text=f"old {i}",
                    embedding=to_blob([0.1] * 1536),
                    dim=1536,
                )
            )
        db.flush()

        assert indexer.stale_dimension_count(db) == 2

    def test_a_single_embedding_model_reports_nothing_stale(self, db):
        from app.kb import indexer
        from app.kb.store import to_blob
        from app.models import KBChunk

        email = make_email(db)
        for i in range(3):
            db.add(
                KBChunk(
                    email_id=email.id,
                    chunk_index=i,
                    text=f"chunk {i}",
                    embedding=to_blob([0.1] * 768),
                    dim=768,
                )
            )
        db.flush()
        assert indexer.stale_dimension_count(db) == 0


class TestCorsWrapsTheAuthFailure:
    """Ordering that is invisible until a browser tries it.

    Starlette makes the last-added middleware outermost. With the auth check
    registered after CORS, a 401 comes back with no `Access-Control-Allow-Origin`
    header, the browser discards it, and a cross-origin dashboard sees an opaque
    network error instead of a 401 — so it never knows to ask for a token.
    """

    def test_a_rejected_request_still_carries_cors_headers(self, client, protected):
        origin = get_settings().cors_origin_list[0]
        response = client.get("/api/health", headers={"Origin": origin})

        assert response.status_code == 401
        assert response.headers.get("access-control-allow-origin") == origin

    def test_preflight_is_not_blocked_by_auth(self, client, protected):
        """A preflight carries no credentials by design; rejecting it breaks
        CORS without protecting anything, since the real request is checked."""
        origin = get_settings().cors_origin_list[0]
        response = client.options(
            "/api/health",
            headers={
                "Origin": origin,
                "Access-Control-Request-Method": "GET",
            },
        )
        assert response.status_code == 200


class TestDocsFollowDeploymentPosture:
    def test_docs_are_available_for_local_use(self, client):
        assert client.get("/docs").status_code == 200

    def test_a_configured_token_means_this_is_deployed(self):
        """Swagger fetches /openapi.json without an Authorization header, so the
        docs cannot sit behind the token — they either stay open or they go."""
        from app import main

        assert main._docs_enabled is (not get_settings().app_auth_token)
