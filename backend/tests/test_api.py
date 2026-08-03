"""HTTP surface smoke tests.

Exercises the app through its real lifespan, so a broken startup (bad router
wiring, a mis-declared dependency) fails here rather than at `uvicorn` time.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.agent.classify import Extraction, Verdict
from app.agent.resolve import apply
from app.main import app
from app.models import Application
from tests.fixtures import make_email


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def job_verdict(**kwargs) -> Verdict:
    extraction = Extraction(is_job_related=True, confidence=0.9, **kwargs)
    return Verdict(
        is_job_related=True,
        confidence=0.9,
        source="llm",
        raw=extraction.model_dump(),
        extraction=extraction,
    )


class TestHealth:
    def test_reports_status_and_counters(self, client, db):
        body = client.get("/api/health").json()
        assert body["status"] == "ok"
        # No credentials in the test environment.
        assert body["gmail_authorised"] is False
        assert body["gmail_usable"] is False
        assert body["llm_calls"] == 0
        assert "emails_stored" in body

    def test_root_points_at_the_docs(self, client):
        assert client.get("/").json()["health"] == "/api/health"


class TestEmails:
    def test_lists_stored_emails(self, client, db):
        make_email(db, gmail_id="a", subject="First")
        make_email(db, gmail_id="b", subject="Second")
        db.commit()

        body = client.get("/api/emails").json()
        assert body["total"] == 2
        assert {i["subject"] for i in body["items"]} == {"First", "Second"}

    def test_filters_unclassified(self, client, db):
        classified = make_email(db, gmail_id="a")
        classified.is_job_related = True
        make_email(db, gmail_id="b")
        db.commit()

        body = client.get("/api/emails", params={"unclassified": True}).json()
        assert body["total"] == 1
        assert body["items"][0]["gmail_id"] == "b"

    def test_search_matches_subject(self, client, db):
        make_email(db, gmail_id="a", subject="Interview invitation")
        make_email(db, gmail_id="b", subject="Parcel delivered")
        db.commit()

        body = client.get("/api/emails", params={"q": "Interview"}).json()
        assert body["total"] == 1

    def test_detail_returns_body(self, client, db):
        email = make_email(db, body="The full message body.")
        db.commit()

        body = client.get(f"/api/emails/{email.id}").json()
        assert body["body_text"] == "The full message body."

    def test_missing_email_is_404(self, client):
        assert client.get("/api/emails/999999").status_code == 404

    def test_manual_override_is_recorded_as_manual(self, client, db):
        email = make_email(db)
        db.commit()

        body = client.post(
            f"/api/emails/{email.id}/classification", json={"is_job_related": True}
        ).json()

        assert body["is_job_related"] is True
        assert body["classification_source"] == "manual"


class TestApplications:
    def test_board_groups_by_status(self, client, db):
        email = make_email(db)
        apply(db, email, job_verdict(company="Acme", event_type="interview_scheduled"))
        db.commit()

        columns = {c["status"]: c for c in client.get("/api/applications/board").json()["columns"]}
        assert columns["interviewing"]["count"] == 1
        assert columns["applied"]["count"] == 0

    def test_stats_counts_the_pipeline(self, client, db):
        e1 = make_email(db, gmail_id="a", thread_id="t1")
        apply(db, e1, job_verdict(company="Acme", event_type="interview_scheduled"))
        e2 = make_email(db, gmail_id="b", thread_id="t2")
        apply(db, e2, job_verdict(company="Globex", event_type="rejection"))
        db.commit()

        body = client.get("/api/applications/stats").json()
        assert body["total"] == 2
        assert body["interviewing"] == 1
        assert body["rejected"] == 1

    def test_detail_includes_the_timeline(self, client, db):
        email = make_email(db)
        outcome = apply(db, email, job_verdict(company="Acme", event_type="applied"))
        db.commit()

        body = client.get(f"/api/applications/{outcome.application.id}").json()
        assert body["company"] == "Acme"
        assert len(body["timeline"]) == 1
        assert body["timeline"][0]["event_type"] == "applied"

    def test_status_override_accepts_a_valid_status(self, client, db):
        email = make_email(db)
        outcome = apply(db, email, job_verdict(company="Acme", event_type="applied"))
        db.commit()

        body = client.post(
            f"/api/applications/{outcome.application.id}/status", json={"status": "accepted"}
        ).json()
        assert body["status"] == "accepted"

    def test_status_override_rejects_an_unknown_status(self, client, db):
        email = make_email(db)
        outcome = apply(db, email, job_verdict(company="Acme", event_type="applied"))
        db.commit()

        response = client.post(
            f"/api/applications/{outcome.application.id}/status", json={"status": "banana"}
        )
        assert response.status_code == 422


class TestNotifications:
    def test_lists_and_marks_read(self, client, db):
        email = make_email(db)
        apply(db, email, job_verdict(company="Acme", event_type="offer"))
        db.commit()

        listing = client.get("/api/notifications").json()
        assert listing["unread"] == 1
        notification_id = listing["items"][0]["id"]

        client.post(f"/api/notifications/{notification_id}/read")
        assert client.get("/api/notifications").json()["unread"] == 0


class TestSync:
    def test_backfill_is_refused_without_gmail_credentials(self, client):
        response = client.post("/api/sync/backfill", json={"days": 30})
        assert response.status_code == 409
        # The message must name the fix, not just the symptom.
        assert "app.gmail.auth" in response.json()["detail"]

    def test_status_is_idle_before_any_run(self, client, db):
        assert client.get("/api/sync/status").json()["running"] is False


class TestChat:
    def test_reports_a_missing_api_key_instead_of_failing(self, client, db):
        body = client.post("/api/chat", json={"message": "how many applications?"}).json()
        assert "GEMINI_API_KEY" in body["answer"]


class TestWebSocket:
    def test_handshake_announces_subscribed_topics(self, client):
        with client.websocket_connect("/ws/notifications") as ws:
            hello = ws.receive_json()
            assert hello["topic"] == "connected"
            assert "notification.created" in hello["data"]["topics"]


class TestManualCorrectionActs:
    """The override buttons have to change the board, not just a label.

    Marking a bogus entry "not job related" used to leave the application it had
    already created sitting on the board, and marking a missed acknowledgement
    "job related" changed a flag and nothing else. Neither was a correction.
    """

    def test_rejecting_an_email_removes_its_application(self, client, db):
        email = make_email(db, subject="Torinit is hiring for a Remote role")
        apply(db, email, job_verdict(company="Torinit", event_type="applied"))
        db.commit()
        assert db.query(Application).count() == 1

        body = client.post(
            f"/api/emails/{email.id}/classification", json={"is_job_related": False}
        ).json()

        assert body["is_job_related"] is False
        assert db.query(Application).count() == 0

    def test_rejecting_keeps_an_application_that_has_other_evidence(self, db, client):
        """Only applications left with no events at all are removed."""
        first = make_email(db, gmail_id="a", thread_id="t", subject="Thanks for applying")
        outcome = apply(db, first, job_verdict(company="Acme", event_type="applied"))
        app_id = outcome.application.id

        second = make_email(db, gmail_id="b", thread_id="t", subject="Interview")
        apply(db, second, job_verdict(company="Acme", event_type="interview_scheduled"))
        db.commit()

        client.post(f"/api/emails/{second.id}/classification", json={"is_job_related": False})

        assert db.get(Application, app_id) is not None

    def test_a_confirmed_email_skips_the_prefilter_next_time(self, db):
        """A correction that lasts only until the next re-classification is not
        a correction — the prefilter would just re-reject it."""
        from app.agent import classify as classify_mod

        email = make_email(
            db, subject="I want to connect", sender="invitations@linkedin.com"
        )
        assert classify_mod.classify(db, email).source == "prefilter"

        email.is_job_related = True
        email.classification_source = "manual"
        db.flush()

        assert classify_mod.is_manually_confirmed(email) is True
        # No LLM is configured in tests, so this proves only that the prefilter
        # no longer short-circuits it — the verdict now comes from further down.
        assert classify_mod.classify(db, email).source != "prefilter"
